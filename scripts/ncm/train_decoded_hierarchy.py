"""Student-conditioned NCM with hard-hierarchy decoded byte supervision.

The original corpus, optimizer and global schedule are preserved. This is a
learning experiment, not a qualified conversion or a guaranteed coding model.
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
from .gguf_training_backbone import load_candidate
from . import frozen_tower as tower
from .frozen_tower_training import load_corpus
from .frozen_ragged_qualification import qualify
from .decoded_hierarchy import backward_batch
from .decoded_hierarchy_policy import validate_parent, qualify_objective
from .deterministic_student_policy import enable
from .resident_training_policy import restore
from .exact_checkpoint import restore_trainable
from .frozen_tower_optimizer import TowerOptimizer
from .train_geometry import ResponseSampler
from .train import admission, headroom, save, cpu_tree, write_json


from .train_frozen_clean import evaluate, validate


def run(a):
    if min(a.steps, a.batch, a.warmup, a.save_every, a.eval_every) < 1 or a.session_steps < 0:
        raise ValueError('Positive schedule dimensions required')
    deterministic = enable()
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
    restore_trainable(model, a.init, parent, parent_sha)
    tokenizer = AutoTokenizer.from_pretrained('results/tokenizer', local_files_only=True, trust_remote_code=False)
    for name, checksum in parent['implementation_sha256'].items():
        if digest(Path(__file__).with_name(name)) != checksum:
            raise ValueError('Checkpoint implementation changed: ' + name)
    if a.evaluate_only:
        if parent.get('architecture') != 'frozen-context-original-oxcoder-native-ncm-v40':
            raise ValueError('Evaluation requires a trained v40 checkpoint')
        if parent.get('deterministic_execution') != deterministic:
            raise ValueError('Cold evaluation execution policy differs')
        evaluate(model, codec, tokenizer, parent_sha, out / 'coding.json')
        return
    smoke = qualify_objective(a.optimizer_audit, policy, parent_sha)
    training, heldout, corpus = load_corpus(a.selection, a.manifest, parent['source_sha256'], parent['codec_sha256'])
    if corpus != smoke['corpus']:
        raise ValueError('Corpus differs from optimizer smoke')
    optimizer = TowerOptimizer(model, loss_scale=smoke['loss_scale'])
    sampler = ResponseSampler(training)
    rng = random.Random(a.seed)
    validate_parent(parent, corpus, a)
    names = ('train_decoded_hierarchy.py', 'decoded_hierarchy.py', 'decoded_hierarchy_policy.py', 'train_deterministic_student.py', 'deterministic_student_policy.py', 'canonical_onpolicy.py', 'canonical_onpolicy_policy.py', 'exact_checkpoint.py', 'resident_training_policy.py', 'train_frozen_clean.py', 'frozen_clean_objective.py', 'train_independent.py', 'frozen_tower_ragged.py', 'frozen_ragged_objective.py', 'frozen_tower_multi.py', 'frozen_ragged_qualification.py',
        'frozen_multi_objective.py', 'frozen_multi_inference.py', 'frozen_clean_resume.py', 'train_frozen_tower.py', 'frozen_tower.py', 'frozen_tower_batch.py', 'frozen_tower_cache.py',
        'frozen_tower_training.py', 'frozen_tower_objective.py', 'frozen_tower_optimizer.py',
        'gguf_training_backbone.py', 'gguf_training_weights.py', 'backbone.py', 'codec.py',
        'geometry.py', 'coding_eval.py', 'generate_candidate.py', 'train_geometry.py', 'train_accumulated.py',
        'cpu_adamw.py', 'master_muon.py')
    identity = {**parent, 'architecture': 'frozen-context-original-oxcoder-native-ncm-v40',
        'initialization_sha256': parent_sha, 'initialization_policy': 'Explicit decoded-hierarchy branch from v38; retain canonical labels, data, optimizer, sampler, RNG and global schedule; execute on GPU1',
        'execution': policy, 'deterministic_execution': deterministic, 'optimizer_audit_sha256': digest(a.optimizer_audit), 'corpus': corpus,
        'training_recipe': 'Frozen prompt; original 32 layers; canonical student-conditioned objective plus 0.1 hard-hierarchy byte CE on student rows',
        'compute_dtype': 'float16', 'body_lr': 1e-4, 'head_lr': 1e-3,
        'loss_scale': optimizer.loss_scale, 'effective_batch_examples': a.batch,
        'planned_steps': a.steps, 'warmup': a.warmup, 'seed': a.seed,
        'byte_loss_weight': .1, 'byte_loss_scope': 'all-level straight-through hard codes, student-conditioned rows only, mean over whole batch', 'on_policy_probability': .5, 'conditioning_corruption_maximum': .15, 'autoregressive_answer_fallback': False, 'quality_ready': False,
        'implementation_sha256': {n: digest(Path(__file__).with_name(n)) for n in names}}
    start = restore(out / 'latest.pt' if a.resume else a.init,
                    identity if a.resume else parent, model, optimizer, rng, a.device)
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
        # The same legacy RNG draws now select student versus teacher inputs.
        flags = [rng.random() < .5 for _ in selected] if step > a.warmup else [False] * len(selected)
        records = backward_batch(model, codec, selected, optimizer.loss_scale, .15, flags)
        torch.cuda.synchronize(a.device)
        backward_finished = time.monotonic()
        norm = optimizer.step()
        if tower.source_versions(model.backbone) != source_state or any(p.grad is not None for p in codec.parameters()):
            raise RuntimeError('Frozen source/codec invariant violated')
        torch.cuda.synchronize(a.device)
        progress = {'step': step, 'loss': sum(r['loss'] for r in records), 'gradient_norm': norm,
            'seconds': time.monotonic() - began, 'conditioning_corruption_maximum': .15, 'label_policy': 'canonical', 'student_conditioned_examples': sum(flags),
            'byte_ce': [r['byte_ce'] for r in records],
            'code_accuracy': [sum(r['code_accuracy'][level] for r in records) / len(records)
                              for level in range(len(codec.books))],
            'objective_and_backward_seconds': backward_finished - began,
            'optimizer_and_invariant_seconds': time.monotonic() - backward_finished,
            'examples': [{'id': row['id'], 'source': row['source'], 'distilled': bool(row.get('distilled'))} for row, _ in selected],
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(a.device), 'quality_ready': False}
        with (out / 'training.jsonl').open('a') as f:
            f.write(json.dumps(progress) + '\n')
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
