"""Training-only codec diagnostic: can residual re-quantization repair bad codes?

No backbone training or evaluation-task answers are used. The repaired path is
an oracle diagnostic using training text, not a generation method or inference
fallback. Its results cannot count toward the coding acceptance gate.
"""
import argparse
import json
from pathlib import Path
import random

import torch
from .codec import Codec, CodecConfig, encode, resize, PAD
from .backbone import digest
from .train import write_json


@torch.no_grad()
def repair_codes(codec, latent, gold, masks, replacements):
    cumulative = torch.zeros_like(latent)
    repaired = []
    for level, (size, book) in enumerate(zip(codec.config.levels, codec.books)):
        residual = codec.resample(latent - cumulative, size)
        flat, weight = residual.flatten(0, 1).float(), book.weight.float()
        distances = flat.square().sum(-1, keepdim=True) + weight.square().sum(-1) - 2 * flat @ weight.T
        wanted = distances.argmin(-1).reshape(latent.shape[0], size)
        actual = torch.where(masks[level], replacements[level], wanted)
        repaired.append(actual)
        cumulative = cumulative + resize(book(actual), codec.config.length)
    return repaired


@torch.no_grad()
def run(a):
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    if a.device.startswith('cuda'):
        free, total = torch.cuda.mem_get_info(a.device)
        cap = min(768 << 20, free - (3 << 30))
        if cap <= 0:
            raise RuntimeError('Insufficient GPU reserve')
        torch.cuda.set_per_process_memory_fraction(cap / total, a.device)
    ck = torch.load(a.codec, weights_only=True, map_location='cpu')
    codec = Codec(CodecConfig(**ck['config'])).to(a.device)
    codec.load_state_dict(ck['model'])
    codec.eval()
    path = Path(a.data) / 'prepared.pt'
    manifest = json.loads((Path(a.data) / 'manifest.json').read_text())
    if digest(path) != manifest['prepared_sha256']:
        raise ValueError('Prepared data checksum differs')
    rows = [(r, c) for r, c in torch.load(path, weights_only=True, map_location='cpu')['prepared']
            if r['split'] == 'train']
    random.Random(a.seed).shuffle(rows)
    rows = rows[:a.examples]
    records = []
    for start in range(0, len(rows), 8):
        batch = rows[start:start+8]
        ids = torch.tensor([encode(r['target'], codec.config.length) for r, _ in batch], device=a.device)
        latent = codec.encode_latents(ids)
        _, gold, _, _ = codec.quantize(latent)
        masks, replacements = [], []
        for level, codes in enumerate(gold):
            probability = a.corruption * (1 - level / (len(gold) - 1))
            masks.append(torch.rand(codes.shape, device=a.device) < probability)
            replacements.append(torch.randint(codec.config.codes, codes.shape, device=a.device))
        corrupt = [torch.where(m, r, g) for m, r, g in zip(masks, replacements, gold)]
        repaired = repair_codes(codec, latent, gold, masks, replacements)
        mask = ids != PAD
        outputs = {name: codec.reconstruct_codes(codes).argmax(-1) for name, codes in
                   [('gold', gold), ('corrupted_with_clean_labels', corrupt), ('oracle_requantized', repaired)]}
        for i, (row, _) in enumerate(batch):
            scores = {name: {'nonpad_accuracy': float((got[i][mask[i]] == ids[i][mask[i]]).float().mean()),
                             'exact_nonpad': bool(((got[i] == ids[i]) | ~mask[i]).all())}
                      for name, got in outputs.items()}
            records.append({'id': row['id'], 'source': row['source'], 'split': row['split'], 'scores': scores})
    report = {'codec_sha256': digest(a.codec), 'prepared_sha256': digest(path),
        'implementation_sha256': digest(__file__), 'seed': a.seed, 'corruption': a.corruption,
        'scope': 'oracle codec recovery on training targets; not generation and not coding evaluation',
        'quality_ready': False, 'records': records, 'summary': {}}
    for name in outputs:
        report['summary'][name] = {'windows': len(records),
            'exact_nonpad_windows': sum(r['scores'][name]['exact_nonpad'] for r in records),
            'mean_nonpad_accuracy': sum(r['scores'][name]['nonpad_accuracy'] for r in records) / len(records)}
    report['complete'] = True
    write_json(a.output, report)
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('codec', 'data', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--seed', type=int, default=19002)
    p.add_argument('--examples', type=int, default=128)
    p.add_argument('--corruption', type=float, default=.5)
    a = p.parse_args()
    if a.examples < 1 or not 0 <= a.corruption <= 1:
        p.error('Invalid diagnostic bounds')
    run(a)
