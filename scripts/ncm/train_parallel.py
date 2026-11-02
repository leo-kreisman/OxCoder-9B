"""Continue FP32-master NCM training with two qualified compact replicas.

No frozen-weight broadcasts, teacher fallback, or automatic data replacement.
Each worker performs a disjoint subset of the same four-example global batch.
Corruption uses recorded per-example seeds independent of worker assignment.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

import torch
import torch.distributed as dist

from .anvil import OptimizerBundle
from .backbone import digest
from .coding_eval import evaluate_model
from .compact_training_policy import logical_gradient_groups, prefix_policy
from .distillation_curriculum import filter_teacher_actions
from .gguf_training_backbone import load_candidate
from .master_muon import FP32MasterMuon
from .cpu_adamw import CPUAdamW, migrate_bundle as migrate_cpu_optimizer
from .parallel_training import assignments, backward_example, check_inventory, example_seed, sum_gradients
from .train import admission, cpu_tree, evaluate_candidate, exact_training_curriculum, headroom, save, write_json
from .train_geometry import ResponseSampler
from .train_independent import validate_expansion


def agree(value):
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    if any(x != value for x in gathered):
        raise ValueError('Worker identities, curriculum, RNG selection, or weights diverged')


def qualify(path, parent, data_sha):
    report = json.loads(Path(path).read_text())
    if not report.get('complete') or not report.get('passed') or not report.get('speed_qualified'):
        raise ValueError('Parallel audit did not qualify')
    if report['prepared_sha256'] != data_sha or report['devices'] != ['cuda:0', 'cuda:1']:
        raise ValueError('Parallel audit uses different data or devices')
    for name in ('source_sha256', 'codec_sha256', 'rank', 'head_kind', 'head_input_norm', 'concept_input_norm', 'layer_split'):
        if parent.get(name) != report['model_identity'].get(name):
            raise ValueError('Parallel audit model differs: ' + name)
    for name, expected in report['implementation_sha256'].items():
        if digest(Path(__file__).parent / name) != expected:
            raise ValueError('Audited implementation changed: ' + name)
    records = report['records']
    if len(records) != 2 or {tuple(r['requested_prompt_tokens']) for r in records} != {
            (64, 69, 128, 256), (64, 512, 1024, 2048)}:
        raise ValueError('Parallel audit coverage is incomplete')
    for row in records:
        groups = row['gradient_groups']
        logits = row['logit_comparison']
        if (not row['passed'] or row['speed_ratio'] <= 1 or not logits['finite'] or logits['relative_l2'] >= .02
                or set(groups) != {'adapters', 'classifiers', 'interfaces'}
                or any(not g['finite'] or g['relative_l2'] >= .05 or g['cosine'] <= .99
                       for g in [row['gradient_comparison'], *groups.values()])):
            raise ValueError('Parallel audit violates a numerical or speed gate')
    return {'audit_sha256': digest(path), 'audited_checkpoint_sha256': report['checkpoint_sha256'],
            'scope': 'Two-replica summed gradients; finite numerical control, not exact training trajectory equivalence'}


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def qualify_cpu_optimizer(path, parent):
    report = json.loads(Path(path).read_text())
    if not report.get('complete') or not report.get('passed') or len(report.get('records', [])) != 8:
        raise ValueError('CPU optimizer audit is incomplete or failed')
    for name, value in report['model_identity'].items():
        if parent.get(name) != value:
            raise ValueError('CPU optimizer audit model or head learning rate differs')
    if set(report['model_identity']) != {'source_sha256', 'codec_sha256', 'head_kind', 'head_input_norm', 'head_lr'}:
        raise ValueError('CPU optimizer audit identity is incomplete')
    for name in ['cpu_adamw.py', 'audit_cpu_adamw.py']:
        if report['implementation_sha256'].get(name) != digest(Path(__file__).parent / name):
            raise ValueError('CPU optimizer audit implementation changed')
    if (report['offloaded_optimizer_gpu_bytes'] != 0 or report['resident_optimizer_gpu_bytes'] <= 0
            or any(not row['passed'] or not 0 <= row['relative_update_l2'] < .001
                   or not 0 <= row['relative_parameter_l2'] < .000002 for row in report['records'])):
        raise ValueError('CPU optimizer numerical or residency gate failed')
    return {'audit_sha256': digest(path), 'audited_checkpoint_sha256': report['checkpoint_sha256'],
            'scope': 'Same FP32 AdamW rule on CPU; bounded numerical difference, not bit-exact CUDA updates'}


def run(a):
    rank = int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE']) != 2:
        raise ValueError('Exactly two workers are required')
    torch.cuda.set_device(rank); torch.set_num_threads(2)
    device, out = f'cuda:{rank}', Path(a.output)
    limits = admission([device], a.allocator_cap_gib)
    dist.init_process_group('nccl', timeout=timedelta(minutes=20))
    try:
        if rank == 0:
            out.mkdir(parents=True, exist_ok=a.resume)
        dist.barrier()
        write_json(out / f'admission-rank{rank}.json', limits)
        folder = Path(a.data)
        manifest = json.loads((folder / 'manifest.json').read_text())
        if digest(folder / 'prepared.pt') != manifest['prepared_sha256']:
            raise ValueError('Prepared data changed')
        model, codec, parent, parent_sha = load_candidate(a.source, a.codec, a.init, [device])
        if model.cached_prefix or model.head_kind != 'untied' or not model.head.input_norm:
            raise ValueError('Parallel continuation requires unchanged architecture')
        data = torch.load(folder / 'prepared.pt', map_location='cpu', weights_only=True)
        if a.data_expansion:
            validate_expansion(manifest, data, parent)
        elif parent['prepared_sha256'] != manifest['prepared_sha256']:
            raise ValueError('Data expansion must be explicitly selected')
        # Numerical controls remain tied to their actual original examples.
        # Expansion retains those rows and uses the same bounded execution
        # policy; do not relabel the old audit as a test on the new dataset.
        policy = qualify(a.parallel_audit, parent, parent['prepared_sha256'])
        prefix = prefix_policy(a.runtime_audit, parent, parent['prepared_sha256'])
        checkpoint = torch.load(a.init, map_location='cpu', weights_only=True)
        allowed = [['AdamW', 'AdamW', 'FP32MasterMuon']]
        if a.cpu_head_optimizer:
            allowed.append(['CPUAdamW', 'AdamW', 'FP32MasterMuon'])
        if digest(a.init) != parent_sha or checkpoint['optimizer']['classes'] not in allowed:
            raise ValueError('Initialization changed or does not contain FP32-master optimizer state')
        filtered, teacher_filter = filter_teacher_actions(data['prepared'], manifest)
        prepared, curriculum = exact_training_curriculum(codec, filtered, device)
        del filtered, data
        sampler = ResponseSampler(prepared)
        agree((parent_sha, hashlib.sha256(json.dumps([r['id'] for r, c in prepared]).encode()).hexdigest()))
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        check_inventory(named)
        clipping = logical_gradient_groups(named, parent['layer_split'])
        heads = [p for n, p in named if n.startswith(('head.', 'stop_head.'))]
        matrices = [p for n, p in named if (n.startswith('backbone.adapters.') and p.ndim == 2) or n == 'project.weight']
        used = {id(p) for p in heads + matrices}
        other = [p for _, p in named if id(p) not in used]
        optimizer = OptimizerBundle([
            (CPUAdamW if a.cpu_head_optimizer else torch.optim.AdamW)(heads, lr=parent['head_lr'], weight_decay=.01, foreach=False),
            torch.optim.AdamW(other, lr=parent['body_lr'], weight_decay=.01, foreach=False),
            FP32MasterMuon(matrices, lr=parent['body_lr'], weight_decay=.01, adjust_lr_fn='match_rms_adamw')])
        if parent['effective_batch_examples'] != 4:
            raise ValueError('Only the qualified four-example global batch is supported')
        implementation = {name: digest(Path(__file__).parent / name) for name in
            ('train_parallel.py', 'parallel_training.py', 'master_muon.py', 'backbone.py', 'codec.py',
             'gguf_training_backbone.py', 'gguf_training_weights.py', 'shared_training.py', 'train.py',
             'train_accumulated.py', 'train_geometry.py', 'distillation_curriculum.py', 'coding_eval.py',
             'generate_candidate.py', 'geometry.py', 'training_runtime.py', 'vendor/veomni_shared_prefix.py',
             'compact_training_policy.py', 'train_independent.py', 'audit_parallel_training.py',
             'cpu_adamw.py', 'audit_cpu_adamw.py')}
        identity = {**parent, 'training_recipe': ('compact-two-replica-cpu-head-optimizer-v24'
                                                if a.cpu_head_optimizer else 'compact-two-replica-fp32-master-v23'),
            'initialization': {'checkpoint_sha256': parent_sha,
                'scope': 'Retain weights, optimizer/master states, Python sampler RNG and update count; record new corruption RNG'},
            'parallel_execution': policy, 'execution_policy': prefix,
            'compact_execution': {**parent.get('compact_execution', {}),
                'physical_devices': ['cuda:0', 'cuda:1'], 'replicas': 2,
                'all_layers_per_replica': True},
            'optimizer': {**parent['optimizer'], 'explicit_legacy_migration': False,
                          'state_continuation': True},
            'corruption_rng': 'SHA256(seed, global step, global example index); device-local independent generator',
            'implementation_sha256': implementation, 'planned_steps': a.steps,
            'prepared_sha256': manifest['prepared_sha256'],
            'teacher_curriculum': teacher_filter, 'quality_ready': False}
        if a.cpu_head_optimizer:
            if not a.cpu_optimizer_audit:
                raise ValueError('CPU optimizer offloading requires a passing audit')
            identity['cpu_head_optimizer'] = qualify_cpu_optimizer(a.cpu_optimizer_audit, parent)
        if a.data_expansion:
            identity['dataset_expansion'] = {key: manifest[key] for key in
                ('base_prepared_sha256', 'addition_manifest_sha256', 'frozen_evaluation_sha256', 'counts')}
            identity['parallel_execution']['data_scope'] = 'Audited on retained base rows; new data retains the same 2048-token bound'
        if a.resume:
            checkpoint = torch.load(out / 'latest.pt', map_location='cpu', weights_only=True)
            if checkpoint['identity'] != identity or set(checkpoint['model']) != {n for n, p in named}:
                raise ValueError('Resume identity or parameter inventory changed')
            with torch.no_grad():
                for name, parameter in named:
                    parameter.copy_(checkpoint['model'][name].to(parameter))
        if a.cpu_head_optimizer and checkpoint['optimizer']['classes'][0] == 'AdamW':
            migrate_cpu_optimizer(optimizer, checkpoint['optimizer'])
        else:
            optimizer.load_state_dict(checkpoint['optimizer'])
        random.setstate(checkpoint['rng']['python'])
        torch.set_rng_state(checkpoint['rng']['torch'])
        # Do not initialize the other worker's GPU context through set_rng_state_all.
        if checkpoint['rng']['cuda']:
            state_index = rank if a.resume else len(checkpoint['rng']['cuda']) - 1
            torch.cuda.set_rng_state(checkpoint['rng']['cuda'][state_index])
        start, history = checkpoint['step'], checkpoint['history']
        del checkpoint
        if start >= a.steps:
            raise ValueError('No remaining updates')
        if rank == 0:
            write_json(out / 'identity.json', identity)
            write_json(out / 'curriculum.json', curriculum)
            write_json(out / 'sampling.json', sampler.report)
            for name in implementation:
                target = out / 'implementation' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not a.resume:
                    shutil.copyfile(Path(__file__).parent / name, target)
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('results/tokenizer', local_files_only=True, trust_remote_code=False)
        for step in range(start, a.steps):
            headroom([device])
            if shutil.disk_usage(out).free < 4 << 30:
                raise RuntimeError('Checkpoint disk reserve exhausted')
            model.train(); optimizer.zero_grad()
            batch = [sampler.sample(random) for _ in range(4)]
            if any(not 0 < len(c['ids']) <= 2048 for r, c in batch):
                raise ValueError('Example exceeds the qualified context range')
            agree([r['id'] for r, c in batch])
            allocation = assignments([len(c['ids']) for r, c in batch], 2)
            dist.barrier(); torch.cuda.synchronize(); began = time.monotonic()
            metrics = torch.zeros(1 + len(codec.books), device=device, dtype=torch.float64)
            for index in allocation[rank]:
                result = backward_example(model, codec, batch[index], 4,
                    parent['conditioning_corruption']['maximum'], example_seed(parent['seed'], step, index),
                    prefix['maximum_shared_prefix'])
                metrics += torch.tensor([result['loss'], *result['code_accuracy']], device=device)
            sum_gradients(named)
            for group in clipping:
                torch.nn.utils.clip_grad_norm_(group, 1., error_if_nonfinite=True, foreach=False)
            optimizer.step()
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            seconds = torch.tensor(time.monotonic() - began, device=device)
            dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
            record = {'step': step + 1, 'loss': float(metrics[0]), 'code_accuracy': metrics[1:].tolist(),
                'seconds': float(seconds), 'allocation': allocation,
                'examples': [{'id': r['id'], 'source': r['source'], 'prompt_tokens': len(c['ids']),
                              'distilled': bool(r.get('distilled'))} for r, c in batch]}
            write_json(out / f'memory-rank{rank}.json', {
                'step': step + 1, 'device': device, 'allocated_bytes': torch.cuda.memory_allocated(),
                'reserved_bytes': torch.cuda.memory_reserved(), 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'scope': 'This worker PyTorch allocator; excludes driver and other processes',
                'cpu_head_optimizer': a.cpu_head_optimizer})
            history.append(record)
            if rank == 0:
                print(json.dumps(record), flush=True)
            evaluate = (step + 1) % a.eval_every == 0 or step + 1 == a.steps
            if (step + 1) % a.save_every == 0 or evaluate:
                state = model.trainable_state()
                agree(state_hash(state))
                rngs = [None, None]
                dist.all_gather_object(rngs, {'python': random.getstate(), 'torch': torch.get_rng_state(),
                                             'cuda': torch.cuda.get_rng_state()})
                if rank == 0:
                    save(out / 'latest.pt', {'identity': identity, 'model': state,
                        'optimizer': cpu_tree(optimizer.state_dict()), 'step': step + 1, 'history': history,
                        'rng': {'python': rngs[0]['python'], 'torch': rngs[0]['torch'], 'cuda': [x['cuda'] for x in rngs]}})
                del state
                dist.barrier()
            if rank == 0:
                write_json(out / 'progress.json', {'step': step + 1, 'training_complete': step + 1 == a.steps,
                    'training_examples_consumed': (step + 1) * 4, 'history_tail': history[-8:], 'quality_ready': False})
                if evaluate:
                    model.eval()
                    write_json(out / f'chunks-{step+1}.json', evaluate_candidate(model, codec, prepared, device))
                    evaluate_model(model, codec, tokenizer, identity['prompt_tokens'], digest(out / 'latest.pt'),
                                   out / f'coding-{step+1}.json')
            dist.barrier()
    except BaseException:
        # A peer may be blocked inside a collective. Waiting for NCCL shutdown
        # here can hide the original error for the entire group timeout. Let
        # torchrun observe this worker's failure and terminate its peers.
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    else:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'codec', 'init', 'data', 'output', 'runtime-audit', 'parallel-audit'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--allocator-cap-gib', type=float, default=11.25)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--data-expansion', action='store_true')
    parser.add_argument('--cpu-head-optimizer', action='store_true')
    parser.add_argument('--cpu-optimizer-audit')
    args = parser.parse_args()
    if min(args.steps, args.save_every, args.eval_every) < 1:
        parser.error('Positive schedule bounds are required')
    run(args)
