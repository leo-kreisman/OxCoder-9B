"""Compare concept CE with frozen-decoder auxiliary losses on a small probe.

All variants receive identical training examples and initialization. Validation
groups are disjoint and used only for scoring. Earlier concepts are gold: this
is a training-objective diagnostic, not free generation or a coding result.
"""
import argparse
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from .audit_latent_predictability import LatentProbe, evaluate
from .codec import Codec, CodecConfig, encode
from .backbone import digest
from .train import write_json


def decoded_loss(codec, logits, earlier, target, straight_through=False):
    probabilities = logits.softmax(-1)
    expected = probabilities @ codec.books[-1].weight
    if straight_through:
        discrete = codec.books[-1](logits.argmax(-1))
        expected = expected + (discrete - expected).detach()
    output = codec.decode_latents(earlier + expected)
    return F.cross_entropy(output.flatten(0, 1), target.flatten())


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
    checkpoint = torch.load(a.codec, map_location='cpu', weights_only=True)
    codec = Codec(CodecConfig(**checkpoint['config'])).to(a.device)
    codec.load_state_dict(checkpoint['model'])
    codec.eval().requires_grad_(False)
    path = Path(a.data) / 'prepared.pt'
    manifest = json.loads((Path(a.data) / 'manifest.json').read_text())
    if digest(path) != manifest['prepared_sha256']:
        raise ValueError('Prepared data checksum differs')
    prepared = torch.load(path, map_location='cpu', weights_only=True)['prepared']
    train = [r for r, _ in prepared if r['split'] == 'train']
    validation = [r for r, _ in prepared if r['split'] == 'validation']
    rng = random.Random(a.seed)
    rng.shuffle(train)
    rng.shuffle(validation)
    train, validation = train[:4096], validation[:256]
    if {r['group'] for r in train} & {r['group'] for r in validation}:
        raise ValueError('Training and validation groups overlap')
    del prepared
    initial = LatentProbe(codec.config.width, codec.config.length, codec.config.codes)
    initial_state = {name: value.clone() for name, value in initial.state_dict().items()}
    del initial
    report = {'codec_sha256': digest(a.codec), 'prepared_sha256': digest(path),
              'implementation_sha256': digest(__file__), 'steps_per_variant': a.steps,
              'seed': a.seed, 'byte_loss_weight': a.weight,
              'training_ids': [r['id'] for r in train],
              'validation_ids': [r['id'] for r in validation],
              'variants': {}, 'complete': False, 'quality_ready': False,
              'scope': 'Conditional training probe with gold earlier concepts; not student generation'}
    for mode in ('ce', 'soft_decoder', 'straight_through_decoder'):
        rng = random.Random(a.seed + 1)
        model = LatentProbe(codec.config.width, codec.config.length, codec.config.codes).to(a.device)
        model.load_state_dict(initial_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01, foreach=False)
        history = []
        began = time.monotonic()
        for step in range(a.steps):
            batch = [rng.choice(train) for _ in range(8)]
            target = torch.tensor([encode(r['target'], codec.config.length) for r in batch], device=a.device)
            with torch.no_grad():
                codes, inputs = codec.targets(target)
            logits = model(inputs[-1])
            ce = F.cross_entropy(logits.flatten(0, 1), codes[-1].flatten())
            loss = ce
            if mode != 'ce':
                loss = loss + a.weight * decoded_loss(codec, logits, inputs[-1], target,
                                                     mode == 'straight_through_decoder')
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite diagnostic loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if (step + 1) % 500 == 0 or step + 1 == a.steps:
                row = {'step': step + 1, 'loss': float(loss.detach()), 'code_ce': float(ce.detach()),
                       'seconds': time.monotonic() - began,
                       'validation': evaluate(model, codec, validation, a.device)}
                history.append(row)
                print(json.dumps({'mode': mode, **row}), flush=True)
                report['variants'][mode] = history
                write_json(out / 'report.json', report)
        del model, optimizer, logits, ce, loss
    report['complete'] = True
    write_json(out / 'report.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('codec', 'data', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--seed', type=int, default=19004)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--weight', type=float, default=.2)
    args = parser.parse_args()
    if args.steps < 1 or args.weight < 0:
        parser.error('Invalid diagnostic schedule')
    run(args)
