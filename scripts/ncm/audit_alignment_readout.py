"""Paired train/validation concept readout from fixed teacher/student features.

Neither arm generates code. Gold target context in the teacher makes its arm a
representation upper-bound control, not an inference path or acceptance score.
"""
import argparse
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import digest
from .train import write_json


def read_features(folder):
    manifest = json.loads((folder / 'manifest.json').read_text())
    identity = json.loads((folder / 'identity.json').read_text())
    if not manifest['complete'] or digest(folder / 'identity.json') != manifest['identity_sha256']:
        raise ValueError('Feature inventory is incomplete or identity changed')
    rows = {'train': [], 'validation': []}
    groups = {}
    for entry in manifest['records']:
        path = folder / entry['path']
        if digest(path) != entry['sha256']:
            raise ValueError('Feature checksum differs')
        row = torch.load(path, map_location='cpu', weights_only=True)
        split, group = entry['split'], entry['group']
        if (row['row']['split'] != split or row['row']['id'] != entry['id'] or row['row']['group'] != group
                or groups.setdefault(group, split) != split or split not in rows):
            raise ValueError('Feature split or identity differs')
        if any(len(row[field]) != len(identity['levels']) for field in
               ('student_features', 'teacher_targets', 'valid', 'codes')):
            raise ValueError('Paired student features are missing')
        for level, size in enumerate(identity['levels']):
            for field in ('student_features', 'teacher_targets'):
                tensor = row[field][level]
                if tensor.shape != (size, 4096) or not torch.isfinite(tensor).all():
                    raise ValueError('Nonfinite or incorrectly shaped feature')
            if (row['valid'][level].shape != (size,) or not row['valid'][level].any()
                    or row['codes'][level].shape != (1, size)
                    or (row['codes'][level] < 0).any() or (row['codes'][level] >= 4096).any()):
                raise ValueError('Invalid code or alignment mask')
        if split == 'train' and not row['codec_exact']:
            continue
        rows[split].append(row)
    if not rows['train'] or not rows['validation']:
        raise ValueError('Both data splits are required')
    return rows, identity


def features(rows, field, level, device):
    h = torch.stack([row[field][level] for row in rows]).to(device).float()
    h = h * h.square().mean(-1, keepdim=True).clamp_min(1e-6).rsqrt()
    codes = torch.cat([row['codes'][level] for row in rows]).to(device)
    mask = torch.stack([row['valid'][level] for row in rows]).to(device)
    return h, codes, mask


@torch.no_grad()
def evaluate(heads, rows, field, device):
    result = []
    for level, head in enumerate(heads):
        count = correct = loss = exact = 0
        for start in range(0, len(rows), 8):
            h, target, mask = features(rows[start:start+8], field, level, device)
            logits = head(h)
            got = logits.argmax(-1)
            count += int(mask.sum())
            correct += int(((got == target) & mask).sum())
            loss += float(F.cross_entropy(logits[mask], target[mask], reduction='sum'))
            exact += int(((got == target) | ~mask).all(-1).sum())
        result.append({'level': level, 'valid_codes': count, 'accuracy': correct / count,
                       'cross_entropy': loss / count, 'exact_code_windows': exact,
                       'windows': len(rows)})
    return result


def run(a):
    torch.set_num_threads(2)
    free, total = torch.cuda.mem_get_info(a.device)
    cap = min(int(2.5 * (1 << 30)), free - (3 << 30))
    if cap < 2 << 30:
        raise RuntimeError('Insufficient GPU reserve for readout control')
    torch.cuda.set_per_process_memory_fraction(cap / total, a.device)
    folder, out = Path(a.features), Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    rows, identity = read_features(folder)
    report = {'features_manifest_sha256': digest(folder / 'manifest.json'),
        'feature_identity': identity, 'train_rows': len(rows['train']),
        'validation_rows': len(rows['validation']), 'validation_filtered': False,
        'steps': a.steps, 'batch_size': a.batch_size, 'seed': a.seed,
        'objective': 'mean level nonpadding code CE; original codec targets',
        'initialization': 'identically seeded random linear classifiers in both arms',
        'optimizer': 'AdamW lr=0.001 weight_decay=0.01, norm clipping=1',
        'scope': 'fixed features with gold earlier concepts/target tokens; not free generation',
        'implementation_sha256': digest(__file__), 'arms': {}, 'complete': False, 'quality_ready': False}
    for arm, field in [('student', 'student_features'), ('teacher', 'teacher_targets')]:
        rng = random.Random(a.seed)
        torch.manual_seed(a.seed)
        heads = nn.ModuleList([nn.Linear(4096, 4096) for _ in identity['levels']]).to(a.device)
        optimizer = torch.optim.AdamW(heads.parameters(), lr=.001, weight_decay=.01, foreach=False)
        record = {'before': evaluate(heads, rows['validation'], field, a.device), 'history': []}
        report['arms'][arm] = record
        write_json(out / 'report.json', report)
        begin = time.monotonic()
        for step in range(a.steps):
            batch = [rng.choice(rows['train']) for _ in range(a.batch_size)]
            optimizer.zero_grad(set_to_none=True)
            value = 0.
            for level, head in enumerate(heads):
                h, target, mask = features(batch, field, level, a.device)
                loss = F.cross_entropy(head(h)[mask], target[mask]) / len(heads)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite readout loss')
                loss.backward()
                value += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(heads.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if (step + 1) % 200 == 0 or step + 1 == a.steps:
                item = {'step': step + 1, 'training_loss': value, 'seconds': time.monotonic() - begin,
                        'validation': evaluate(heads, rows['validation'], field, a.device)}
                record['history'].append(item)
                write_json(out / 'report.json', report)
                print(json.dumps({'arm': arm, **item}), flush=True)
        record['training'] = evaluate(heads, rows['train'], field, a.device)
        del optimizer, heads, h, target, mask, loss
        torch.cuda.empty_cache()
    report['complete'] = True
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(a.device)
    write_json(out / 'report.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=21005)
    args = parser.parse_args()
    if min(args.steps, args.batch_size) < 1:
        parser.error('Positive training bounds required')
    run(args)
