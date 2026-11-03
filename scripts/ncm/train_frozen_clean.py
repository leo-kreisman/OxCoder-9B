"""Continue original-backbone NCM training with canonical labels and corrupted conditioning.

An explicit v31 import preserves optimizer, global step, sampling and RNG.
Execution changes are versioned and numerically audited, not called bit-exact.
"""
import argparse
import json
import math
from pathlib import Path
import random
import time
import torch
from transformers import AutoTokenizer

from .backbone import digest
from .codec import encode, decode, PAD
from .coding_eval import TASKS, extract, check_case
from .generate_candidate import generate
from .gguf_training_backbone import load_candidate
from . import frozen_tower as tower
from .frozen_tower_training import load_corpus
from .frozen_ragged_qualification import qualify
from .frozen_clean_objective import backward_batch
from .frozen_multi_objective import trajectory
from .frozen_clean_resume import import_parent_state
from .frozen_tower_optimizer import TowerOptimizer
from .frozen_multi_inference import GenerationAdapter
from .train_geometry import ResponseSampler
from .train import admission, headroom, save, cpu_tree, write_json


@torch.no_grad()
def evaluate(model, codec, tokenizer, checkpoint_sha, output):
    model.eval()
    adapter = GenerationAdapter(model)
    records = []
    for name, prompt, cases in TASKS:
        answer = generate(adapter, codec, tokenizer, [
            {'role': 'system', 'content': 'Return only Python code defining the requested function.'},
            {'role': 'user', 'content': prompt}], 2048, 8)
        answer['runtime'] = 'Frozen-context original OxCoder backbone; packed weights, FP16 CUDA; native NCM'
        checks = []
        if answer['stop_reason'] == 'predicted_end':
            code = extract(answer['text'])
            checks = [check_case(code, name, args, want, preserve_arguments=name == 'merge_intervals')
                      for args, want in cases]
        passed = len(checks) == len(cases) and all(c['passed'] for c in checks)
        records.append({'task': name, 'generation': answer, 'checks': checks, 'passed': passed})
        report = {'checkpoint_sha256': checkpoint_sha, 'tasks': records,
            'passed': sum(r['passed'] for r in records), 'evaluated': len(records),
            'quality_ready': False, 'autoregressive_answer_fallback': False,
            'scope': 'Unchanged 12 elementary coding tasks; fresh-process replay required before acceptance'}
        write_json(output, report)
        print(json.dumps({'task': name, 'passed': passed}), flush=True)
    return report


@torch.no_grad()
def validate(model, codec, rows):
    model.eval()
    records = []
    for row, condition in rows:
        prompt = torch.tensor([condition['ids']])
        target = torch.tensor([encode(row['target'], codec.config.length)], device=codec.position.device)
        result = trajectory(model, codec, [prompt])
        output = codec.decode_latents(result['latent']).argmax(-1)
        try:
            text = decode(output[0].tolist())
        except (ValueError, UnicodeError):
            text = None
        records.append({'id': row['id'], 'free_exact': text == row['target'], 'generated_text': text,
            'free_nonpad_byte_accuracy': float((output == target)[target != PAD].float().mean()),
            'stop_probability': float(result['stop_logits'].sigmoid()[0])})
    return {'records': records, 'windows': len(records), 'free_exact_windows': sum(r['free_exact'] for r in records),
        'mean_byte_accuracy': sum(r['free_nonpad_byte_accuracy'] for r in records) / len(records),
        'quality_ready': False, 'scope': 'Frozen validation windows; some contain gold earlier response chunks'}


def optimizer_qualification(path, policy, parent_sha):
    report = json.loads(Path(path).read_text())
    if (not report.get('complete') or not report.get('passed') or report['execution'] != policy
            or report['checkpoint_sha256'] != parent_sha or len(report.get('records', [])) != 2):
        raise ValueError('Passing optimizer smoke for this initialization required')
    for name, sha in report['implementation_sha256'].items():
        if digest(Path(__file__).with_name(name)) != sha:
            raise ValueError('Optimizer implementation changed: ' + name)
    if any(not r['passed'] for r in report['records']):
        raise ValueError('Optimizer smoke failed')
    return report


def run(a):
    if min(a.steps, a.batch, a.warmup, a.save_every, a.eval_every) < 1 or a.session_steps < 0:
        raise ValueError('Positive schedule dimensions required')
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=a.resume or a.evaluate_only)
    limits = admission([a.device], 9)
    model, codec, parent, parent_sha = load_candidate(a.source, a.codec, a.init, [a.device])
    policy = qualify(a.execution_audit, parent['source_sha256'], parent['codec_sha256'])
    tower.set_compute_dtype(model.backbone, torch.float16)
    tokenizer = AutoTokenizer.from_pretrained('results/tokenizer', local_files_only=True, trust_remote_code=False)
    if a.evaluate_only:
        if parent.get('architecture') != 'frozen-context-original-oxcoder-native-ncm-v32':
            raise ValueError('Evaluation requires a trained v32 checkpoint')
        evaluate(model, codec, tokenizer, parent_sha, out / 'coding.json')
        return
    smoke = optimizer_qualification(a.optimizer_audit, policy, parent_sha)
    training, heldout, corpus = load_corpus(a.selection, a.manifest, parent['source_sha256'], parent['codec_sha256'])
    if corpus != smoke['corpus']:
        raise ValueError('Corpus differs from optimizer smoke')
    optimizer = TowerOptimizer(model, loss_scale=smoke['loss_scale'])
    sampler = ResponseSampler(training)
    rng = random.Random(a.seed)
    names = ('train_frozen_clean.py', 'frozen_clean_objective.py', 'train_independent.py', 'frozen_tower_ragged.py', 'frozen_ragged_objective.py', 'frozen_tower_multi.py', 'frozen_ragged_qualification.py',
        'frozen_multi_objective.py', 'frozen_multi_inference.py', 'frozen_clean_resume.py', 'train_frozen_tower.py', 'frozen_tower.py', 'frozen_tower_batch.py', 'frozen_tower_cache.py',
        'frozen_tower_training.py', 'frozen_tower_objective.py', 'frozen_tower_optimizer.py',
        'gguf_training_backbone.py', 'gguf_training_weights.py', 'backbone.py', 'codec.py',
        'geometry.py', 'coding_eval.py', 'generate_candidate.py', 'train_geometry.py', 'train_accumulated.py',
        'cpu_adamw.py', 'master_muon.py')
    identity = {**parent, 'architecture': 'frozen-context-original-oxcoder-native-ncm-v32',
        'initialization_sha256': parent_sha, 'initialization_policy': 'Explicit objective-change branch from v31; preserve optimizer, sampler, RNG and global schedule',
        'execution': policy, 'optimizer_audit_sha256': digest(a.optimizer_audit), 'corpus': corpus,
        'training_recipe': 'Frozen prompt; original 32 layers; canonical labels with corrupted conditioning',
        'compute_dtype': 'float16', 'body_lr': 1e-4, 'head_lr': 1e-3,
        'loss_scale': optimizer.loss_scale, 'effective_batch_examples': a.batch,
        'planned_steps': a.steps, 'warmup': a.warmup, 'seed': a.seed,
        'on_policy_probability': 0., 'conditioning_corruption_maximum': .15, 'autoregressive_answer_fallback': False, 'quality_ready': False,
        'implementation_sha256': {n: digest(Path(__file__).with_name(n)) for n in names}}
    start = 0
    if a.resume:
        checkpoint = torch.load(out / 'latest.pt', weights_only=True, map_location='cpu')
        if checkpoint['identity'] != identity:
            raise ValueError('Resume identity changed')
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    p.copy_(checkpoint['model'][name].to(p))
        optimizer.load_state_dict(checkpoint['optimizer'])
        rng.setstate(checkpoint['rng']['python'])
        torch.set_rng_state(checkpoint['rng']['torch'])
        torch.cuda.set_rng_state(checkpoint['rng']['cuda'], a.device)
        start = checkpoint['step']
        del checkpoint
    if not a.resume:
        start = import_parent_state(a.init, parent, model, optimizer, rng, a.device, corpus, a)
    write_json(out / 'identity.json', identity)
    write_json(out / 'admission.json', limits)
    write_json(out / 'sampling.json', sampler.report)
    source_state = tower.source_versions(model.backbone)
    for step in range(start + 1, a.steps + 1):
        headroom([a.device])
        began = time.monotonic()
        optimizer.zero_grad()
        factor = min(1., step / a.warmup) * (.1 + .9 * .5 * (1 + math.cos(math.pi * max(0, step - a.warmup) / max(1, a.steps - a.warmup))))
        optimizer.learning_rates(1e-4 * factor, 1e-3 * factor)
        selected = [sampler.sample(rng) for _ in range(a.batch)]
        # Advance the legacy conditioning draws to keep the response sampler
        # aligned with the v31 arm at the same update number.
        if step > a.warmup:
            for _ in selected:
                rng.random()
        records = backward_batch(model, codec, selected, optimizer.loss_scale, .15)
        norm = optimizer.step()
        if tower.source_versions(model.backbone) != source_state or any(p.grad is not None for p in codec.parameters()):
            raise RuntimeError('Frozen source/codec invariant violated')
        torch.cuda.synchronize(a.device)
        progress = {'step': step, 'loss': sum(r['loss'] for r in records), 'gradient_norm': norm,
            'seconds': time.monotonic() - began, 'conditioning_corruption_maximum': .15, 'label_policy': 'canonical',
            'examples': [{'id': row['id'], 'source': row['source'], 'distilled': bool(row.get('distilled'))} for row, _ in selected],
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(a.device), 'quality_ready': False}
        with (out / 'training.jsonl').open('a') as f:
            f.write(json.dumps(progress) + '\n')
        write_json(out / 'progress.json', progress)
        print(json.dumps(progress), flush=True)
        session_end = a.session_steps > 0 and step - start >= a.session_steps
        if step % a.save_every == 0 or step % a.eval_every == 0 or step == a.steps or session_end:
            state = {'identity': identity, 'model': model.trainable_state(),
                'optimizer': cpu_tree(optimizer.state_dict()), 'step': step,
                'rng': {'python': rng.getstate(), 'torch': torch.get_rng_state(),
                        'cuda': torch.cuda.get_rng_state(a.device)}}
            save(out / 'latest.pt', state)
            del state
            if (out / 'STOP_AFTER_CHECKPOINT').exists():
                return
        if step % a.eval_every == 0 or step == a.steps:
            snapshot = out / f'evaluated-{step}.pt'
            # Independent immutable checkpoint; latest remains replaceable.
            import shutil
            shutil.copyfile(out / 'latest.pt', snapshot)
            sha = digest(snapshot)
            write_json(out / f'validation-{step}.json', validate(model, codec, heldout))
            result = evaluate(model, codec, tokenizer, sha, out / f'coding-{step}.json')
            if result['passed'] == result['evaluated'] == 12:
                write_json(out / 'candidate.json', {'checkpoint': str(snapshot.resolve()),
                    'checkpoint_sha256': sha, 'passed': 12, 'quality_ready': False,
                    'phase': 'fresh_process_replay_required'})
                return
        if session_end:
            write_json(out / 'session.json', {'step': step, 'phase': 'yielded_after_checkpoint',
                'planned_steps': a.steps, 'quality_ready': False})
            return


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'codec', 'init', 'execution-audit', 'output'):
        p.add_argument('--' + key, required=True)
    for key in ('selection', 'manifest', 'optimizer-audit'):
        p.add_argument('--' + key)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--session-steps', type=int, default=0,
                   help='Yield after this many updates, retaining full resume state; zero runs the remaining schedule')
    p.add_argument('--seed', type=int, default=29004)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--evaluate-only', action='store_true')
    a = p.parse_args()
    if not a.evaluate_only and any(getattr(a, key) is None for key in ('selection', 'manifest', 'optimizer_audit')):
        p.error('Training requires selection, manifest and optimizer audit')
    run(a)
