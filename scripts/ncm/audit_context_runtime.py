"""Compare cached training conditioning with fresh standalone source encoding."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from .backbone import digest
from .context_cache import FeatureCache
from .context_decoder import DecoderConfig
from .context_runtime import load
from .train import admission, write_json


def run(a):
    torch.set_num_threads(2)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    admitted = admission([a.device], 8)
    runtime, codec, identity, checksum = load(a.source, a.codec, a.checkpoint, a.device)
    cache = FeatureCache(a.cache, DecoderConfig(**identity['config']))
    if cache.sha != identity['feature_manifest_sha256']:
        raise ValueError('Runtime audit cache differs from training')
    indices = sorted((i for i, r in enumerate(cache.records) if r['split'] == 'validation'),
                     key=lambda i: hashlib.sha256(('runtime-v25:' + cache.records[i]['id']).encode()).hexdigest())[:a.count]
    if len(indices) != a.count or a.count < 1: raise ValueError('Insufficient validation rows')
    records = []
    with torch.no_grad():
        for i in indices:
            item = cache.get(i)
            hidden = item['hidden'][None].to(a.device)
            ids = torch.tensor([item['condition']['ids']], dtype=torch.long)
            fresh = runtime.backbone(runtime.backbone.embed_prompt(ids))
            valid = torch.ones(hidden.shape[:2], dtype=torch.bool, device=a.device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                expected = runtime.decoder.generate_chunk(codec, hidden, valid)
                actual = runtime.decoder.generate_chunk(codec, fresh, valid)
            relative = float((fresh.float() - hidden.float()).norm() / hidden.float().norm().clamp_min(1e-20))
            record = {'id': cache.records[i]['id'], 'prompt_tokens': hidden.shape[1],
                'cached_feature_exact': bool(torch.equal(fresh, hidden)), 'feature_relative_l2': relative,
                'concepts_exact': all(torch.equal(x, y) for x, y in zip(expected[1], actual[1])),
                'bytes_exact': bool(torch.equal(expected[0], actual[0])),
                'stop_max_abs': float((expected[2] - actual[2]).abs().max())}
            record['passed'] = (relative <= .001 and record['concepts_exact'] and record['bytes_exact']
                                and record['stop_max_abs'] <= 1e-5)
            records.append(record)
            report = {'checkpoint_sha256': checksum, 'cache_manifest_sha256': cache.sha,
                'source_sha256': identity['source_sha256'], 'codec_sha256': identity['codec_sha256'],
                'admission': admitted, 'records': records, 'complete': len(records) == a.count,
                'passed': len(records) == a.count and all(r['passed'] for r in records), 'quality_ready': False,
                'implementation_sha256': digest(__file__),
                'scope': 'Conditioning/reload numerical parity only; outputs are not asserted to be correct code'}
            write_json(out / 'report.json', report)
            print(json.dumps(record), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'codec', 'checkpoint', 'cache', 'output'): p.add_argument('--' + name, required=True)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--count', type=int, default=16)
    run(p.parse_args())
