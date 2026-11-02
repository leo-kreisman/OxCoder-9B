"""NCM adapter training with exact compact frozen weights and optional packing.

The learned model and equal-example objective are unchanged. Physical placement
uses one GPU while norm clipping preserves the reference model's logical groups.
Checkpoint adapters remain reloadable by the original resident BF16 backend.
"""
import argparse
import fcntl
import json
from pathlib import Path
import random
import shutil
import time

import torch

from .anvil import OptimizerBundle
from .backbone import digest
from .codec import encode
from .coding_eval import evaluate_model
from .compact_training_policy import logical_gradient_groups, packs, qualified, prefix_policy, placement_policy, storage_policy
from .gguf_training_backbone import load_candidate
from .distillation_curriculum import filter_teacher_actions
from .packed_training import forward_examples
from .shared_training import forward_levels
from .train import (admission, headroom, write_json, save, cpu_tree, rng_state, set_rng,
                    exact_training_curriculum, evaluate_candidate)
from .train_accumulated import example_loss
from .train_geometry import ResponseSampler
from .train_independent import corrupted_inputs, validate_expansion
from .training_runtime import execution
from .master_muon import FP32MasterMuon, migrate_bundle


def run(a, out):
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    devices = [a.device]
    write_json(out / 'admission.json', admission(devices, a.allocator_cap_gib))
    folder = Path(a.data)
    manifest = json.loads((folder / 'manifest.json').read_text())
    if digest(folder / 'prepared.pt') != manifest['prepared_sha256']:
        raise ValueError('Prepared checksum differs')
    data = torch.load(folder / 'prepared.pt', map_location='cpu', weights_only=True)
    model, codec, parent, parent_sha = load_candidate(a.source, a.codec, a.init, devices)
    if a.data_expansion:
        validate_expansion(manifest, data, parent)
    elif parent['prepared_sha256'] != manifest['prepared_sha256']:
        raise ValueError('Prepared lineage differs')
    if model.cached_prefix or model.head_kind != 'untied' or not model.head.input_norm:
        raise ValueError('Requires uncached independent normalized classifiers')
    policy = prefix_policy(a.runtime_audit, parent, manifest['prepared_sha256'])
    shared_limit = policy['maximum_shared_prefix']
    weight_policy = storage_policy(a.matrix_audit, a.weight_audit, a.reproducibility_audit, parent, manifest['prepared_sha256'])
    placement = placement_policy(a.weight_audit, a.placement_audit, parent, manifest['prepared_sha256'],
                                 a.matrix_audit, a.reproducibility_audit) if a.placement_audit else None
    admitted_devices = placement['relocation']['physical_devices'] if placement else weight_policy['physical_devices']
    if devices != admitted_devices:
        raise ValueError('Physical placement has not been qualified')
    packing_policy = qualified(a.packed_audit, parent, manifest['prepared_sha256'], packing=True) if a.packed_audit else None
    if packing_policy and a.pack_tokens > 1900:
        raise ValueError('Packing token budget exceeds the conservative audited limit')
    candidates, teacher_filter = filter_teacher_actions(data['prepared'], manifest)
    write_json(out / 'teacher-curriculum.json', teacher_filter)
    prepared, curriculum = exact_training_curriculum(codec, candidates, devices[-1])
    write_json(out / 'curriculum.json', curriculum)
    sampler = ResponseSampler(prepared)
    write_json(out / 'sampling.json', sampler.report)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    clipping = logical_gradient_groups(trainable, parent['layer_split'])
    heads = [p for n, p in trainable if n.startswith(('head.', 'stop_head.'))]
    matrices = [p for n, p in trainable if (n.startswith('backbone.adapters.') and p.ndim == 2) or n == 'project.weight']
    used = {id(p) for p in heads + matrices}
    other = [p for _, p in trainable if id(p) not in used]
    matrix_optimizer = FP32MasterMuon if a.fp32_master_muon else torch.optim.Muon
    optimizer = OptimizerBundle([
        torch.optim.AdamW(heads, lr=a.head_lr, weight_decay=.01, foreach=False),
        torch.optim.AdamW(other, lr=a.body_lr, weight_decay=.01, foreach=False),
        matrix_optimizer(matrices, lr=a.body_lr, weight_decay=.01, adjust_lr_fn='match_rms_adamw')])
    names = ('train_compact.py', 'compact_training_policy.py', 'distillation_curriculum.py', 'gguf_training_weights.py',
             'gguf_training_backbone.py', 'packed_training.py', 'train_accumulated.py', 'train_independent.py',
             'train.py', 'train_geometry.py', 'backbone.py', 'geometry.py', 'codec.py',
             'generate_candidate.py', 'coding_eval.py', 'training_runtime.py',
             'shared_training.py', 'vendor/veomni_shared_prefix.py', 'master_muon.py')
    implementation = {name: digest(Path(__file__).parent / name) for name in names}
    identity = {**parent, 'training_recipe': ('independent-classifiers-compact-master-v22' if a.fp32_master_muon
                                           else 'independent-classifiers-compact-v20'),
        'initialization': {'checkpoint_sha256': parent_sha, 'scope': 'Retain learned weights; fresh optimizer unless explicitly continued'},
        'prepared_sha256': manifest['prepared_sha256'], 'seed': a.seed, 'planned_steps': a.steps,
        'effective_batch_examples': a.accumulation, 'head_lr': a.head_lr, 'body_lr': a.body_lr, 'head_warmup': 0,
        'conditioning_corruption': {'maximum': a.corruption, 'schedule': 'linear from maximum at first level to zero at last; clean labels'},
        'loss': 'example mean of (level-mean code CE + 0.1 final-level end BCE)',
        'optimizer': {'name': 'Muon hidden matrices + AdamW classifiers/interfaces', 'head_lr': a.head_lr, 'body_lr': a.body_lr,
                      'fp32_master_muon': a.fp32_master_muon, 'explicit_legacy_migration': a.migrate_muon},
        'execution_policy': policy, 'compact_execution': {'weight_policy': weight_policy,
            'placement_policy': placement,
            'packing_policy': packing_policy, 'pack_tokens': a.pack_tokens, 'physical_devices': devices,
            'clipping': 'Canonical reference layer_split groups, independent of physical placement',
            'resident_frozen_weight_bytes': sum(row['bytes'] for row in model.backbone.transfers)},
        'implementation_sha256': implementation, 'allocator_cap_gib': a.allocator_cap_gib,
        'teacher_curriculum': teacher_filter,
        'quality_ready': False, 'evaluation': 'Unchanged development evaluator; no teacher fallback'}
    identity.pop('continuation', None)
    if a.data_expansion:
        identity['dataset_expansion'] = {k: manifest[k] for k in
            ('base_prepared_sha256', 'addition_manifest_sha256', 'frozen_evaluation_sha256', 'counts')}
    start, history = 0, []
    if a.continue_optimizer:
        for key, value in [('effective_batch_examples', a.accumulation), ('head_lr', a.head_lr),
                           ('body_lr', a.body_lr), ('seed', a.seed)]:
            if parent.get(key) != value:
                raise ValueError('Optimizer continuation changes learning setting: ' + key)
        if parent.get('conditioning_corruption', {}).get('maximum') != a.corruption:
            raise ValueError('Optimizer continuation changes corruption')
        checkpoint = torch.load(a.init, weights_only=True, map_location='cpu')
        if digest(a.init) != parent_sha:
            raise ValueError('Continuation checkpoint changed')
        if a.migrate_muon:
            migrate_bundle(optimizer, checkpoint['optimizer'])
            identity['initialization']['optimizer_migration'] = (
                'Retain AdamW states and Muon momentum; initialize FP32 masters from current rounded model; '
                'old discarded update bits are unrecoverable')
        else:
            optimizer.load_state_dict(checkpoint['optimizer'])
        set_rng(checkpoint['rng'])
        start, history = checkpoint['step'], checkpoint['history']
        identity['initialization']['scope'] = 'Retain learned weights, optimizer, RNG and completed update count'
        del checkpoint
    if a.resume:
        checkpoint = torch.load(out / 'latest.pt', map_location='cpu', weights_only=True)
        if checkpoint['identity'] != identity or set(checkpoint['model']) != set(model.trainable_state()):
            raise ValueError('Resume identity differs')
        with torch.no_grad():
            for name, parameter in trainable:
                parameter.copy_(checkpoint['model'][name].to(parameter))
        optimizer.load_state_dict(checkpoint['optimizer'])
        set_rng(checkpoint['rng'])
        start, history = checkpoint['step'], checkpoint['history']
        del checkpoint
    else:
        for name in implementation:
            target = out / 'implementation' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(__file__).parent / name, target)
    if start >= a.steps:
        raise ValueError('No remaining updates')
    write_json(out / 'identity.json', identity)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('results/tokenizer', local_files_only=True, trust_remote_code=False)
    for step in range(start, a.steps):
        headroom(devices)
        if shutil.disk_usage(out).free < 2 << 30:
            raise RuntimeError('Keep 2 GiB free for checkpoints')
        model.train()
        optimizer.zero_grad()
        began, loss_value = time.monotonic(), 0.
        batch = []
        for _ in range(a.accumulation):
            row, condition = sampler.sample(random)
            target = torch.tensor([encode(row['target'], codec.config.length)], device=devices[-1])
            with torch.no_grad():
                codes, _ = codec.targets(target)
            batch.append({'row': row, 'condition': condition, 'codes': codes,
                          'inputs': corrupted_inputs(codec, codes, a.corruption),
                          'prompt': torch.tensor([condition['ids']])})
        accuracies, pack_sizes = [0.] * len(codec.books), []
        groups = packs(batch, a.pack_tokens, codec.config.levels, shared_limit) if packing_policy else ([x] for x in batch)
        for group in groups:
            pack_sizes.append(len(group))
            with execution(model, 0, 'reference'):
                if len(group) == 1 and not 64 <= group[0]['prompt'].shape[1] <= shared_limit:
                    x = group[0]
                    for level, inp in enumerate(x['inputs']):
                        logits, end, _ = model(inp, x['prompt'], level, return_aux=True)
                        loss = example_loss(logits, x['codes'][level], end, x['condition']['is_final_chunk'],
                                            level, len(codec.books), a.accumulation)
                        if not torch.isfinite(loss):
                            raise RuntimeError('Nonfinite training loss')
                        loss_value += float(loss.detach())
                        accuracies[level] += float((logits.argmax(-1) == x['codes'][level]).float().mean()) / a.accumulation
                        loss.backward()
                        del logits, end, loss
                    continue
                if len(group) > 1:
                    outputs = forward_examples(model, [(x['inputs'], x['prompt']) for x in group])
                else:
                    x = group[0]
                    values = forward_levels(model, x['inputs'], x['prompt'])
                    outputs = [values]
                    del values
                joint = 0
                for x, values in zip(group, outputs):
                    for level, (logits, end, _) in enumerate(values):
                        loss = example_loss(logits, x['codes'][level], end, x['condition']['is_final_chunk'],
                                            level, len(codec.books), a.accumulation)
                        if not torch.isfinite(loss):
                            raise RuntimeError('Nonfinite training loss')
                        loss_value += float(loss.detach())
                        accuracies[level] += float((logits.argmax(-1) == x['codes'][level]).float().mean()) / a.accumulation
                        joint = joint + loss
                joint.backward()
                del joint, outputs, values, logits, end, loss
        for adapter in model.backbone.adapters:
            grad = adapter.up.weight.grad
            if grad is None or not torch.isfinite(grad).all() or grad.norm() == 0:
                raise RuntimeError('Missing/nonfinite/zero adapter gradient')
        for group in clipping:
            torch.nn.utils.clip_grad_norm_(group, 1., error_if_nonfinite=True, foreach=False)
        optimizer.step()
        examples = [{'id': x['row']['id'], 'source': x['row']['source'], 'prompt_tokens': x['prompt'].shape[1],
                     'distilled': bool(x['row'].get('distilled', False))} for x in batch]
        record = {'step': step + 1, 'loss': loss_value, 'code_accuracy': accuracies,
                  'seconds': time.monotonic() - began, 'examples': examples, 'packed_examples': pack_sizes,
                  'peak_allocated_bytes': {d: torch.cuda.max_memory_allocated(d) for d in devices}}
        history.append(record)
        print(json.dumps(record), flush=True)
        evaluate = (step + 1) % a.eval_every == 0 or step + 1 == a.steps
        if (step + 1) % a.save_every == 0 or evaluate:
            save(out / 'latest.pt', {'identity': identity, 'model': model.trainable_state(),
                'optimizer': cpu_tree(optimizer.state_dict()), 'rng': rng_state(), 'step': step + 1, 'history': history})
        write_json(out / 'progress.json', {'step': step + 1, 'training_complete': step + 1 == a.steps,
            'training_examples_consumed': (step + 1) * a.accumulation, 'history_tail': history[-8:], 'quality_ready': False})
        if evaluate:
            model.eval()
            write_json(out / f'chunks-{step+1}.json', evaluate_candidate(model, codec, prepared, devices[-1]))
            evaluate_model(model, codec, tokenizer, identity['prompt_tokens'], digest(out / 'latest.pt'),
                           out / f'coding-{step+1}.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'codec', 'init', 'data', 'output', 'runtime-audit', 'weight-audit', 'matrix-audit', 'reproducibility-audit'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--packed-audit')
    parser.add_argument('--placement-audit')
    parser.add_argument('--fp32-master-muon', action='store_true')
    parser.add_argument('--migrate-muon', action='store_true', help='Explicit legacy BF16-Muon state migration')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--pack-tokens', type=int, default=1900)
    parser.add_argument('--head-lr', type=float, default=.0003)
    parser.add_argument('--body-lr', type=float, default=.00003)
    parser.add_argument('--corruption', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=19001)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--allocator-cap-gib', type=float, default=11.25)
    parser.add_argument('--data-expansion', action='store_true')
    parser.add_argument('--continue-optimizer', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.accumulation, args.eval_every, args.save_every, args.pack_tokens) < 1:
        parser.error('Positive schedule bounds required')
    if args.migrate_muon and not (args.fp32_master_muon and args.continue_optimizer):
        parser.error('Muon migration requires FP32 masters and optimizer continuation')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=args.resume)
    with (output / '.training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, output)
