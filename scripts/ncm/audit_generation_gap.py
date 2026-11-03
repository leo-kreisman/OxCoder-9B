"""Separate conditional prediction accuracy from autonomous concept generation.

Gold code prefixes are deliberately privileged diagnostics, never acceptance
results. No updates are made and no coding-gate prompts or answers are loaded.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch

from .backbone import digest
from .codec import encode, decode, PAD, resize
from .gguf_training_backbone import load_candidate
from . import frozen_tower as tower
from . import frozen_tower_multi as multi
from . import frozen_tower_ragged as ragged
from .frozen_multi_objective import level_output
from .frozen_tower_training import load_corpus
from .train import admission, write_json
from .exact_checkpoint import restore_trainable


def selected_rows(rows, count):
    # Take distinct response groups in deterministic hash order. Prefer initial
    # chunks: they cannot contain gold earlier response chunks in the prompt.
    ordered = sorted(rows, key=lambda x: (not x[0]['id'].endswith('byte-0'), x[0]['id']))
    seen, selected = set(), []
    for row, condition in ordered:
        group = row['group']
        if group in seen:
            continue
        seen.add(group)
        selected.append((row, condition))
        if len(selected) == count:
            break
    return selected


def byte_metrics(codec, latent, target):
    output = codec.decode_latents(latent).argmax(-1)
    try:
        text = decode(output[0].tolist())
    except (ValueError, UnicodeError):
        text = None
    mask = target != PAD
    return {'valid_byte_sequence': text is not None,
            'exact': text is not None and text == decode(target[0].tolist()),
            'exact_including_padding': bool(torch.equal(output, target)),
            'nonpad_byte_accuracy': float((output == target)[mask].float().mean()),
            'text': text}


@torch.no_grad()
def diagnostic(model, codec, row, condition):
    prompt = torch.tensor([condition['ids']])
    target = torch.tensor([encode(row['target'], codec.config.length)], device=codec.position.device)
    labels, inputs = codec.targets(target)
    state = multi.prepare(model.backbone, [prompt])
    logits = ragged.forward_levels(model, inputs, [prompt], state)
    teacher = [(value[0].argmax(-1) == label).float().mean().item() for value, label in zip(logits, labels)]
    record = {'id': row['id'], 'split': row['split'], 'source': row['source'],
              'target_bytes': len(row['target'].encode('utf-8')),
              'initial_chunk': row['id'].endswith('byte-0'),
              'teacher_conditioned_code_accuracy': teacher, 'rollouts': []}
    for gold_levels in range(len(labels) + 1):
        z = torch.zeros_like(inputs[-1])
        code_accuracy = []
        for level, (book, size) in enumerate(zip(codec.books, codec.config.levels)):
            if level < gold_levels:
                selected = labels[level]
            else:
                prediction, _, _ = level_output(model, codec.resample(z, size), state, level)
                selected = prediction.argmax(-1)
            code_accuracy.append(float((selected == labels[level]).float().mean()))
            z += resize(book(selected), codec.config.length)
        record['rollouts'].append({'gold_prefix_levels': gold_levels,
            'privileged_diagnostic': gold_levels > 0, 'code_accuracy': code_accuracy,
            **byte_metrics(codec, z, target)})
    return record


def run(a):
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'complete': False, 'quality_ready': False, 'records': [],
              'scope': 'No updates; autonomous versus privileged gold-concept-prefix diagnostics'}
    supervisor = 'ncm-frozen-clean-pipeline-v32.service'
    worker = 'ncm-frozen-clean-worker-v32.service'
    paused = False

    def active(unit):
        return subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit]).returncode == 0

    try:
        if a.wait_for_session:
            if not active(worker) or not active(supervisor):
                raise RuntimeError('Expected a live canonical training session')
            subprocess.run(['systemctl', '--user', 'kill', '--kill-who=main', '--signal=STOP', supervisor], check=True)
            paused = True
            report['phase'] = 'waiting_for_saved_canonical_session'
            write_json(output / 'report.json', report)
            while active(worker):
                time.sleep(10)
        if Path('results/ncm-frozen-clean-training-v32/candidate.json').exists():
            report['phase'] = 'existing_candidate_takes_priority'
            return
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        report['admission'] = admission([a.device], 8)
        model, codec, parent, checksum = load_candidate(a.source, a.codec, a.checkpoint, [a.device])
        for name, sha in parent['implementation_sha256'].items():
            if digest(Path(__file__).with_name(name)) != sha:
                raise ValueError('Parent implementation changed: ' + name)
        tower.set_compute_dtype(model.backbone, torch.float16)
        restore_trainable(model, a.checkpoint, parent, checksum)
        model.eval()
        training, heldout, corpus = load_corpus(a.selection, a.manifest, parent['source_sha256'], parent['codec_sha256'])
        if corpus != parent['corpus']:
            raise ValueError('Corpus differs')
        report.update(phase='diagnosing', checkpoint_sha256=checksum, corpus=corpus,
                      implementation_sha256=digest(__file__),
                      exact_restore_sha256=digest(Path(__file__).with_name('exact_checkpoint.py')),
                      trainable_checkpoint_values_restored_exactly=True)
        for items in (training, heldout):
            for row, condition in selected_rows(items, a.examples):
                result = diagnostic(model, codec, row, condition)
                report['records'].append(result)
                write_json(output / 'report.json', report)
                print(json.dumps({'id': row['id'], 'split': row['split'],
                    'autonomous_byte_accuracy': result['rollouts'][0]['nonpad_byte_accuracy']}), flush=True)
        summary = []
        for split in sorted({r['split'] for r in report['records']}):
            records = [r for r in report['records'] if r['split'] == split]
            summary.append({'split': split, 'examples': len(records),
                'teacher_conditioned_code_accuracy': [statistics.mean(r['teacher_conditioned_code_accuracy'][i] for r in records)
                                                     for i in range(len(codec.books))],
                'by_gold_prefix': [{'gold_prefix_levels': n,
                    'mean_byte_accuracy': statistics.mean(r['rollouts'][n]['nonpad_byte_accuracy'] for r in records),
                    'valid_sequences': sum(r['rollouts'][n]['valid_byte_sequence'] for r in records),
                    'exact_windows': sum(r['rollouts'][n]['exact'] for r in records)} for n in range(len(codec.books) + 1)]})
        report.update(complete=True, summary=summary)
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        write_json(output / 'report.json', report)
        if paused:
            subprocess.run(['systemctl', '--user', 'kill', '--kill-who=main', '--signal=CONT', supervisor], check=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'codec', 'checkpoint', 'selection', 'manifest', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--examples', type=int, default=16)
    p.add_argument('--wait-for-session', action='store_true')
    a = p.parse_args()
    if a.examples < 1:
        p.error('examples must be positive')
    run(a)
