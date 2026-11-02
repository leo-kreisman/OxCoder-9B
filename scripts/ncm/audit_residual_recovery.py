"""Oracle diagnostic: can later residual codebooks repair early prediction errors?

Targets are supplied only to this diagnostic. Its oracle outputs are never
model answers, training inputs, or executable evaluation candidates.
"""
import argparse
import json
from pathlib import Path

import torch

from .backbone import digest
from .codec import Codec, CodecConfig, PAD, resize
from .context_cache import FeatureCache
from .context_decoder import ContextConceptDecoder, DecoderConfig
from .probe_context_checkpoint import select
from .train import admission, write_json


@torch.no_grad()
def residual_targets(codec, target_latents, predicted, keep):
    if not 0 <= keep <= len(codec.books): raise ValueError('Invalid preserved prefix')
    z = torch.zeros_like(target_latents)
    chosen = []
    for level, (book, size) in enumerate(zip(codec.books, codec.config.levels)):
        if level < keep:
            code = predicted[level]
        else:
            residual = codec.resample(target_latents - z, size).float()
            flat, weights = residual.flatten(0, 1), book.weight.float()
            distances = flat.square().sum(-1, keepdim=True) + weights.square().sum(-1) - 2 * flat @ weights.T
            code = distances.argmin(-1).reshape(target_latents.shape[0], size)
        chosen.append(code)
        z = z + resize(book(code), codec.config.length)
    return chosen, z


def run(a):
    torch.set_num_threads(2); admission([a.device], 8)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    checksum = digest(a.checkpoint)
    checkpoint = torch.load(a.checkpoint, weights_only=True, map_location='cpu')
    if digest(a.checkpoint) != checksum: raise ValueError('Changing checkpoint')
    identity = checkpoint['identity']
    if digest(a.codec) != identity['codec_sha256']: raise ValueError('Wrong codec')
    config = DecoderConfig(**identity['config'])
    model = ContextConceptDecoder(config).to(a.device)
    model.load_state_dict(checkpoint['model']); model.eval().requires_grad_(False)
    c = torch.load(a.codec, weights_only=True, map_location='cpu')
    codec = Codec(CodecConfig(**c['config'])).to(a.device)
    codec.load_state_dict(c['model']); codec.eval().requires_grad_(False)
    cache = FeatureCache(a.cache, config)
    if cache.sha != identity['feature_manifest_sha256']: raise ValueError('Wrong cache')
    indices = select(cache.records, 16) + [i for i, r in enumerate(cache.records) if r['split'] == 'validation']
    records = []
    with torch.no_grad():
        for index in indices:
            payload = cache.get(index)
            hidden = payload['hidden'][None].to(a.device)
            valid = torch.ones(hidden.shape[:2], dtype=torch.bool, device=a.device)
            target = payload['bytes'][None].to(a.device)
            target_latents = codec.encode_latents(target)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                free, predictions, _ = model.generate_chunk(codec, hidden, valid)
            nonpad = target != PAD
            oracle = []
            for keep in range(len(codec.books)):
                _, z = residual_targets(codec, target_latents, predictions, keep)
                output = codec.decode_latents(z).argmax(-1)
                oracle.append({'preserved_predicted_levels': keep,
                    'nonpad_byte_accuracy': float((output == target)[nonpad].float().mean()),
                    'exact_nonpad': bool(((output == target) | ~nonpad).all())})
            records.append({'id': cache.records[index]['id'], 'split': cache.records[index]['split'],
                'free_byte_accuracy': float((free == target)[nonpad].float().mean()), 'oracle': oracle})
            report = {'checkpoint_sha256': checksum, 'feature_manifest_sha256': cache.sha,
                'codec_sha256': identity['codec_sha256'], 'implementation_sha256': digest(__file__),
                'records': records, 'complete': len(records) == len(indices), 'quality_ready': False,
                'scope': 'Target-informed oracle residual recovery, not a generator or inference option'}
            if report['complete']:
                report['summary'] = {}
                for split in ('train', 'validation'):
                    rows = [r for r in records if r['split'] == split]
                    report['summary'][split] = {'count': len(rows),
                        'free_byte_accuracy': sum(r['free_byte_accuracy'] for r in rows) / len(rows),
                        'oracle_accuracy_by_preserved_prefix': [sum(r['oracle'][i]['nonpad_byte_accuracy'] for r in rows) / len(rows) for i in range(len(codec.books))],
                        'oracle_exact_by_preserved_prefix': [sum(r['oracle'][i]['exact_nonpad'] for r in rows) for i in range(len(codec.books))]}
            write_json(out / 'report.json', report)
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('codec', 'checkpoint', 'cache', 'output'): p.add_argument('--' + name, required=True)
    p.add_argument('--device', default='cuda:0')
    run(p.parse_args())
