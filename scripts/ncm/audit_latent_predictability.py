"""Probe final-level predictability from gold earlier-level latents alone.

This small, train-only-fitted probe isolates information lost by the transferred
backbone interface. It has no prompt access and is not a standalone generator.
Development labels are scored only; they never receive optimizer updates.
"""
import argparse
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F
from .codec import Codec, CodecConfig, encode, PAD
from .backbone import digest
from .train import save, write_json


class LatentProbe(nn.Module):
    def __init__(self, width, length, codes):
        super().__init__()
        self.position = nn.Parameter(torch.randn(length, width) * .02)
        self.network = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 256),
                                     nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, codes))

    def forward(self, inputs):
        return self.network(inputs + self.position[:inputs.shape[1]])


@torch.no_grad()
def evaluate(model, codec, rows, device):
    loss_sum = correct = count = exact = byte_correct = byte_count = 0
    for start in range(0, len(rows), 8):
        batch = rows[start:start+8]
        ids = torch.tensor([encode(r['target'], codec.config.length) for r in batch], device=device)
        codes, inputs = codec.targets(ids)
        logits = model(inputs[-1])
        predicted = logits.argmax(-1)
        loss_sum += float(F.cross_entropy(logits.flatten(0, 1), codes[-1].flatten(), reduction='sum'))
        correct += int((predicted == codes[-1]).sum())
        count += codes[-1].numel()
        output = codec.reconstruct_codes(codes[:-1] + [predicted]).argmax(-1)
        mask = ids != PAD
        byte_correct += int(((output == ids) & mask).sum())
        byte_count += int(mask.sum())
        exact += int(((output == ids) | ~mask).all(-1).sum())
    return {'windows': len(rows), 'code_loss': loss_sum / count,
            'code_accuracy': correct / count, 'nonpad_accuracy': byte_correct / byte_count,
            'exact_nonpad_windows': exact,
            'scope': 'gold earlier concepts; no prompt; not free generation'}


def run(a):
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    free, total = torch.cuda.mem_get_info(a.device)
    cap = min(768 << 20, free - (3 << 30))
    if cap <= 0:
        raise RuntimeError('Insufficient GPU reserve')
    torch.cuda.set_per_process_memory_fraction(cap / total, a.device)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    ck = torch.load(a.codec, weights_only=True, map_location='cpu')
    codec = Codec(CodecConfig(**ck['config'])).to(a.device)
    codec.load_state_dict(ck['model'])
    codec.eval().requires_grad_(False)
    path = Path(a.data) / 'prepared.pt'
    manifest = json.loads((Path(a.data) / 'manifest.json').read_text())
    if digest(path) != manifest['prepared_sha256']:
        raise ValueError('Prepared checksum differs')
    rows = torch.load(path, weights_only=True, map_location='cpu')['prepared']
    training = [r for r, _ in rows if r['split'] == 'train']
    development = [r for r, _ in rows if r['split'] == 'validation']
    del rows
    rng = random.Random(a.seed)
    rng.shuffle(training)
    rng.shuffle(development)
    training, development = training[:4096], development[:256]
    if {r['group'] for r in training} & {r['group'] for r in development}:
        raise ValueError('Training/development group overlap')
    model = LatentProbe(codec.config.width, codec.config.length, codec.config.codes).to(a.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01, foreach=False)
    report = {'codec_sha256': digest(a.codec), 'prepared_sha256': digest(path),
        'implementation_sha256': digest(__file__), 'seed': a.seed, 'steps': a.steps,
        'training_ids': [r['id'] for r in training], 'development_ids': [r['id'] for r in development],
        'before': evaluate(model, codec, development, a.device), 'history': [],
        'quality_ready': False, 'complete': False,
        'scope': 'Final-level conditional probe, not OxCoder inference or coding acceptance'}
    write_json(out / 'report.json', report)
    began = time.monotonic()
    for step in range(a.steps):
        batch = [rng.choice(training) for _ in range(8)]
        ids = torch.tensor([encode(r['target'], codec.config.length) for r in batch], device=a.device)
        with torch.no_grad():
            codes, inputs = codec.targets(ids)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs[-1])
        loss = F.cross_entropy(logits.flatten(0, 1), codes[-1].flatten())
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite probe loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % 250 == 0 or step + 1 == a.steps:
            row = {'step': step + 1, 'loss': float(loss.detach()), 'seconds': time.monotonic() - began,
                   'development': evaluate(model, codec, development, a.device)}
            report['history'].append(row)
            write_json(out / 'report.json', report)
            print(json.dumps(row), flush=True)
    save(out / 'probe.pt', {'model': model.state_dict(), 'codec_sha256': report['codec_sha256'],
                          'scope': report['scope'], 'quality_ready': False})
    report['complete'] = True
    report['checkpoint_sha256'] = digest(out / 'probe.pt')
    write_json(out / 'report.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('codec', 'data', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--seed', type=int, default=19003)
    a = p.parse_args()
    if a.steps < 1:
        p.error('Positive steps required')
    run(a)
