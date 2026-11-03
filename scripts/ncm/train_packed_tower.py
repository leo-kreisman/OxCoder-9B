"""Qualified packed-gradient continuation with unchanged native-NCM learning."""
import argparse
import json
import math
from pathlib import Path
import random
import shutil
import time

import torch
from transformers import AutoTokenizer

from .backbone import digest
from .gguf_training_backbone import load_candidate
from . import frozen_tower as tower
from .cached_tower_corpus import load_corpus
from .frozen_ragged_qualification import qualify as qualify_execution
from .frozen_clean_objective import backward_batch
from .compact_curriculum_policy import CompactSampler
from .resident_tower_optimizer import ResidentTowerOptimizer
from .packed_training_policy import validate_parent, qualify, qualify_reader
from .resident_training_policy import restore
from .interleaved_context import installed
from .packed_gradient import installed as packed_gradient
from .exact_checkpoint import restore_trainable
from .train_compact_tower import evaluate, validate
from .train import admission, headroom, save, cpu_tree, write_json


def run(a):
    if min(a.steps, a.batch, a.warmup, a.save_every, a.eval_every) < 1 or a.session_steps < 0:
        raise ValueError('Invalid schedule dimensions')
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=a.resume or a.evaluate_only)
    limits = admission([a.device], 12)
    model, codec, parent, parent_sha = load_candidate(a.source, a.codec, a.init, [a.device])
    execution = qualify_execution(a.execution_audit, parent['source_sha256'], parent['codec_sha256'])
    tower.set_compute_dtype(model.backbone, torch.float16)
    restore_trainable(model, a.init, parent, parent_sha)
    tokenizer = AutoTokenizer.from_pretrained('results/tokenizer', local_files_only=True, trust_remote_code=False)
    if a.evaluate_only:
        if parent.get('architecture') != 'frozen-context-original-oxcoder-native-ncm-v35':
            raise ValueError('Expected packed-gradient runtime checkpoint')
        with packed_gradient(model.backbone), installed():
            evaluate(model, codec, tokenizer, parent_sha, out / 'coding.json')
        return
    audit = qualify(a.runtime_audit, parent_sha)
    reader_audit = qualify_reader(a.reader_audit, a.selection_cache, parent['corpus'])
    training, heldout, corpus = load_corpus(a.selection, a.manifest, parent['source_sha256'], parent['codec_sha256'], a.selection_cache)
    if audit['corpus'] != corpus:
        raise ValueError('Runtime audit corpus differs')
    validate_parent(parent, corpus, a)
    optimizer = ResidentTowerOptimizer(model)
    sampler = CompactSampler(training)
    rng = random.Random(a.seed)
    implementations = dict(parent['implementation_sha256'])
    for name in ('train_packed_tower.py', 'packed_training_policy.py', 'packed_gradient.py',
                 'selection_reader.py', 'cached_tower_corpus.py', 'exact_checkpoint.py'):
        implementations[name] = digest(Path(__file__).with_name(name))
    identity = {**parent, 'architecture': 'frozen-context-original-oxcoder-native-ncm-v35',
                'initialization_sha256': parent_sha,
                'initialization_policy': 'Preserve v34 weights, optimizer, RNG, sampling and global schedule; change qualified execution and serialization only',
                'parent_runtime_audit_sha256': parent['runtime_audit_sha256'],
                'runtime_audit_sha256': digest(a.runtime_audit),
                'reader_audit_sha256': digest(a.reader_audit),
                'selection_cache_manifest_sha256': reader_audit['cache_manifest_sha256'],
                'runtime_policy': 'Packed-only linear autograd without layer recomputation; resident heads; interleaved prefix; exact JSON selection',
                'implementation_sha256': implementations, 'execution': execution, 'quality_ready': False}
    start = restore(out / 'latest.pt' if a.resume else a.init,
                    identity if a.resume else parent, model, optimizer, rng, a.device, legacy=False)
    write_json(out / 'identity.json', identity)
    write_json(out / 'admission.json', limits)
    write_json(out / 'sampling.json', sampler.report)
    frozen = tower.source_versions(model.backbone)
    with packed_gradient(model.backbone), installed():
        for step in range(start + 1, a.steps + 1):
            headroom([a.device])
            began = time.monotonic()
            optimizer.zero_grad()
            factor = min(1., step / a.warmup) * (.1 + .9 * .5 * (1 + math.cos(math.pi * max(0, step - a.warmup) / max(1, a.steps - a.warmup))))
            optimizer.learning_rates(1e-4 * factor, 1e-3 * factor)
            selected = [sampler.sample(rng) for _ in range(a.batch)]
            if step > a.warmup:
                for _ in selected:
                    rng.random()
            records = backward_batch(model, codec, selected, optimizer.loss_scale, .15)
            torch.cuda.synchronize(a.device)
            backward_finished = time.monotonic()
            norm = optimizer.step()
            if tower.source_versions(model.backbone) != frozen or any(p.grad is not None for p in codec.parameters()):
                raise RuntimeError('Frozen source/codec invariant violated')
            torch.cuda.synchronize(a.device)
            progress = {'step': step, 'loss': sum(r['loss'] for r in records), 'gradient_norm': norm,
                        'seconds': time.monotonic() - began, 'conditioning_corruption_maximum': .15,
                        'label_policy': 'canonical', 'runtime_policy': identity['runtime_policy'],
                        'code_accuracy': [sum(r['code_accuracy'][level] for r in records) / len(records)
                                          for level in range(len(codec.books))],
                        'objective_and_backward_seconds': backward_finished - began,
                        'optimizer_and_invariant_seconds': time.monotonic() - backward_finished,
                        'examples': [{'id': row['id'], 'source': row['source'], 'distilled': bool(row.get('distilled')),
                                      'compact': row.get('curriculum') == 'compact-code-v33'} for row, _ in selected],
                        'peak_allocated_bytes': torch.cuda.max_memory_allocated(a.device), 'quality_ready': False}
            with (out / 'training.jsonl').open('a') as stream:
                stream.write(json.dumps(progress) + '\n')
            write_json(out / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
            session_end = (a.session_steps > 0 and step - start >= a.session_steps) or step == a.steps
            if step % a.save_every == 0 or step % a.eval_every == 0 or step == a.steps or session_end:
                state = {'identity': identity, 'model': model.trainable_state(),
                         'optimizer': cpu_tree(optimizer.state_dict()), 'step': step,
                         'rng': {'python': rng.getstate(), 'torch': torch.get_rng_state(),
                                 'cuda': torch.cuda.get_rng_state(a.device)}}
                save(out / 'latest.pt', state)
                del state
                if (out / 'STOP_AFTER_CHECKPOINT').exists():
                    write_json(out / 'stopped.json', {'step': step, 'checkpoint_sha256': digest(out / 'latest.pt'),
                                                    'quality_ready': False})
                    return
            if step % a.eval_every == 0 or step == a.steps:
                snapshot = out / f'evaluated-{step}.pt'
                shutil.copyfile(out / 'latest.pt', snapshot)
                checksum = digest(snapshot)
                write_json(out / f'validation-{step}.json', validate(model, codec, heldout))
                result = evaluate(model, codec, tokenizer, checksum, out / f'coding-{step}.json')
                if result['passed'] == result['evaluated'] == 12:
                    write_json(out / 'candidate.json', {'checkpoint': str(snapshot.resolve()),
                                                      'checkpoint_sha256': checksum, 'passed': 12,
                                                      'quality_ready': False, 'phase': 'fresh_process_replay_required'})
                    return
            if session_end:
                write_json(out / 'session.json', {'step': step, 'phase': 'yielded_after_checkpoint',
                                                'planned_steps': a.steps, 'quality_ready': False})
                return


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'codec', 'init', 'execution-audit', 'output'):
        p.add_argument('--' + name, required=True)
    for name in ('selection', 'manifest', 'runtime-audit', 'reader-audit', 'selection-cache'):
        p.add_argument('--' + name)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--session-steps', type=int, default=500)
    p.add_argument('--seed', type=int, default=29004)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--evaluate-only', action='store_true')
    a = p.parse_args()
    if not a.evaluate_only and any(getattr(a, key) is None for key in ('selection', 'manifest', 'runtime_audit', 'reader_audit', 'selection_cache')):
        p.error('Training requires selection, manifest, runtime audit, reader audit and selection cache')
    run(a)
