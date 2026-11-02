"""Full-gradient NCM adaptation with several independent examples per update.

Uses the same frozen OxCoder backbone, independent code heads, codec, and
audited shared-prefix execution as v18. This is a new learning stage, not a
resource-only continuation: input data, effective batch, and optimizer state
are explicitly recorded. Neither teacher inference nor evaluation answers are
used at generation time.
"""
import argparse
import fcntl
import json
from pathlib import Path
import random
import shutil
import time

import torch
from torch.nn import functional as F

from .anvil import OptimizerBundle
from .backbone import digest
from .codec import encode
from .coding_eval import evaluate_model
from .generate_candidate import load_candidate
from .train import (admission, headroom, write_json, save, cpu_tree, rng_state,
                    set_rng, exact_training_curriculum, evaluate_candidate)
from .train_geometry import ResponseSampler
from .train_independent import corrupted_inputs, validate_expansion
from .training_runtime import validate_audit, execution


def example_loss(logits, codes, end, is_final, level, levels, accumulation):
    """Equal mean over examples, then equal mean over concept levels."""
    if accumulation < 1 or levels < 1 or not 0 <= level < levels:
        raise ValueError('Invalid loss normalization')
    loss = F.cross_entropy(logits.flatten(0, 1), codes.flatten()) / levels
    if level == levels - 1:
        loss = loss + .1 * F.binary_cross_entropy_with_logits(
            end, torch.full_like(end, float(is_final)))
    return loss / accumulation


def run(a, out):
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    devices = ['cuda:0', 'cuda:1']
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
    policy = validate_audit(a.runtime_audit, parent, manifest['prepared_sha256'], 0, 'reference', a.shared_prefix_training)
    prepared, curriculum = exact_training_curriculum(codec, data['prepared'], devices[-1])
    write_json(out / 'curriculum.json', curriculum)
    sampler = ResponseSampler(prepared)
    write_json(out / 'sampling.json', sampler.report)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    heads = [p for n, p in trainable if n.startswith(('head.', 'stop_head.'))]
    matrices = [p for n, p in trainable if (n.startswith('backbone.adapters.') and p.ndim == 2) or n == 'project.weight']
    used = {id(p) for p in heads + matrices}
    other = [p for _, p in trainable if id(p) not in used]
    optimizer = OptimizerBundle([
        torch.optim.AdamW(heads, lr=a.head_lr, weight_decay=.01, foreach=False),
        torch.optim.AdamW(other, lr=a.body_lr, weight_decay=.01, foreach=False),
        torch.optim.Muon(matrices, lr=a.body_lr, weight_decay=.01, adjust_lr_fn='match_rms_adamw')])
    names = ('train_accumulated.py', 'train_independent.py', 'train.py', 'train_geometry.py',
             'backbone.py', 'geometry.py', 'codec.py', 'generate_candidate.py', 'coding_eval.py',
             'training_runtime.py', 'shared_training.py', 'vendor/veomni_shared_prefix.py')
    implementation = {name: digest(Path(__file__).parent / name) for name in names}
    identity = {**parent, 'training_recipe': 'independent-classifiers-accumulated-v19',
        'initialization': {'checkpoint_sha256': parent_sha,
                           'scope': 'retain all trained weights; fresh optimizer for explicit new learning stage'},
        'prepared_sha256': manifest['prepared_sha256'], 'seed': a.seed,
        'planned_steps': a.steps, 'effective_batch_examples': a.accumulation,
        'head_lr': a.head_lr, 'body_lr': a.body_lr, 'head_warmup': 0,
        'conditioning_corruption': {'maximum': a.corruption,
            'schedule': 'linear from maximum at first level to zero at last; clean labels'},
        'loss': 'example mean of (level-mean code CE + 0.1 final-level end BCE)',
        'optimizer': {'name': 'Muon hidden matrices + AdamW classifiers/interfaces',
                      'head_lr': a.head_lr, 'body_lr': a.body_lr},
        'execution_policy': policy, 'implementation_sha256': implementation,
        'allocator_cap_gib': a.allocator_cap_gib, 'quality_ready': False,
        'evaluation': 'unchanged reused development diagnostic; never training targets'}
    identity.pop('continuation', None)
    if a.data_expansion:
        identity['dataset_expansion'] = {k: manifest[k] for k in
            ('base_prepared_sha256', 'addition_manifest_sha256', 'frozen_evaluation_sha256', 'counts')}
    start, history = 0, []
    if a.resume:
        checkpoint = torch.load(out / 'latest.pt', map_location='cpu', weights_only=True)
        if checkpoint['identity'] != identity or set(checkpoint['model']) != set(model.trainable_state()):
            raise ValueError('Resume identity or parameters differ')
        with torch.no_grad():
            for name, p in trainable:
                p.copy_(checkpoint['model'][name].to(p))
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
        if shutil.disk_usage(out).free < 2 << 30:
            raise RuntimeError('Keep 2 GiB free for checkpoints')
        model.train()
        optimizer.zero_grad()
        began, loss_value, examples = time.monotonic(), 0., []
        accuracies = [0.] * len(codec.books)
        for microstep in range(a.accumulation):
            headroom(devices)
            row, c = sampler.sample(random)
            target = torch.tensor([encode(row['target'], codec.config.length)], device=devices[-1])
            with torch.no_grad():
                codes, _ = codec.targets(target)
            inputs = corrupted_inputs(codec, codes, a.corruption)
            prompt = torch.tensor([c['ids']])
            shared = a.shared_prefix_training and prompt.shape[1] >= 64
            with execution(model, 0, 'reference'):
                if shared:
                    from .shared_training import forward_levels
                    outputs = forward_levels(model, inputs, prompt)
                joint = 0
                for level, inp in enumerate(inputs):
                    logits, end, _ = outputs[level] if shared else model(inp, prompt, level, return_aux=True)
                    loss = example_loss(logits, codes[level], end, c['is_final_chunk'],
                                        level, len(codes), a.accumulation)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite training loss')
                    loss_value += float(loss.detach())
                    accuracies[level] += float((logits.argmax(-1) == codes[level]).float().mean()) / a.accumulation
                    if shared:
                        joint = joint + loss
                    else:
                        loss.backward()
                if shared:
                    joint.backward()
                    del outputs, joint
            # Release references before constructing the next example's graph.
            del loss, logits, end, inp
            examples.append({'id': row['id'], 'source': row['source'], 'prompt_tokens': len(c['ids']),
                             'distilled': bool(row.get('distilled', False)), 'shared_prefix': shared})
        for adapter in model.backbone.adapters:
            gradient = adapter.up.weight.grad
            if gradient is None or not torch.isfinite(gradient).all() or gradient.norm() == 0:
                raise RuntimeError('Missing/nonfinite/zero adapter gradient')
        for device in devices:
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable if str(p.device) == device],
                                          1., error_if_nonfinite=True, foreach=False)
        optimizer.step()
        record = {'step': step + 1, 'loss': loss_value, 'code_accuracy': accuracies,
                  'seconds': time.monotonic() - began, 'examples': examples,
                  'peak_allocated_bytes': {d: torch.cuda.max_memory_allocated(d) for d in devices}}
        history.append(record)
        print(json.dumps(record), flush=True)
        evaluate = (step + 1) % a.eval_every == 0 or step + 1 == a.steps
        if (step + 1) % a.save_every == 0 or evaluate:
            save(out / 'latest.pt', {'identity': identity, 'model': model.trainable_state(),
                'optimizer': cpu_tree(optimizer.state_dict()), 'rng': rng_state(),
                'step': step + 1, 'history': history})
        write_json(out / 'progress.json', {'step': step + 1, 'training_complete': step + 1 == a.steps,
            'training_examples_consumed': (step + 1) * a.accumulation,
            'history_tail': history[-8:], 'full_history': 'latest.pt and the JSONL stdout log',
            'quality_ready': False})
        if evaluate:
            model.eval()
            write_json(out / f'chunks-{step+1}.json', evaluate_candidate(model, codec, prepared, devices[-1]))
            evaluate_model(model, codec, tokenizer, identity['prompt_tokens'], digest(out / 'latest.pt'),
                           out / f'coding-{step+1}.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'codec', 'init', 'data', 'output', 'runtime-audit'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--head-lr', type=float, default=.0003)
    parser.add_argument('--body-lr', type=float, default=.00003)
    parser.add_argument('--corruption', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=19001)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--allocator-cap-gib', type=float, default=11.25)
    parser.add_argument('--shared-prefix-training', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--data-expansion', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.accumulation, args.eval_every, args.save_every) < 1:
        parser.error('Positive schedule bounds required')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=args.resume)
    with (output / '.training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, output)
