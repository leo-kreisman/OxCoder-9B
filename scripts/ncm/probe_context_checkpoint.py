"""Fixed training-chunk diagnostic for separating underfit from overfit.

This probe never updates parameters and cannot qualify a model for release.
Later chunks retain gold earlier-response conditioning, unlike complete coding
generation. Only the unchanged executable evaluation measures the twelve tasks.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def select(records, count=16):
    if count < 1: raise ValueError('Positive probe count required')
    buckets = defaultdict(list)
    for index, row in enumerate(records):
        if row['split'] == 'train':
            key = hashlib.sha256(('train-probe-v25:' + row['id']).encode()).hexdigest()
            buckets[row['source']].append((key, index, row['group']))
    for values in buckets.values(): values.sort(reverse=True)
    selected, groups = [], set()
    while len(selected) < count:
        previous = len(selected)
        for source in sorted(buckets):
            while buckets[source]:
                _, index, group = buckets[source].pop()
                if group not in groups:
                    selected.append(index); groups.add(group); break
            if len(selected) == count: break
        if len(selected) == previous: raise ValueError('Insufficient distinct training conversations')
    return selected


def run(a):
    import torch
    from .backbone import digest
    from .codec import Codec, CodecConfig
    from .context_cache import FeatureCache
    from .context_decoder import ContextConceptDecoder, DecoderConfig
    from .train import admission, write_json
    from .train_context_decoder import validation
    torch.set_num_threads(2)
    admission([a.device], 8)
    checksum = digest(a.checkpoint)
    state = torch.load(a.checkpoint, weights_only=True, map_location='cpu')
    if digest(a.checkpoint) != checksum: raise ValueError('Checkpoint changed while loading')
    identity = state['identity']
    if digest(a.codec) != identity['codec_sha256']: raise ValueError('Codec differs')
    config = DecoderConfig(**identity['config'])
    cache = FeatureCache(a.cache, config)
    if cache.sha != identity['feature_manifest_sha256']: raise ValueError('Feature cache differs')
    model = ContextConceptDecoder(config).to(a.device)
    model.load_state_dict(state['model']); model.eval().requires_grad_(False)
    c = torch.load(a.codec, weights_only=True, map_location='cpu')
    codec = Codec(CodecConfig(**c['config'])).to(a.device)
    codec.load_state_dict(c['model']); codec.eval().requires_grad_(False)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    indices = select(cache.records, a.count)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'selection.json', {'checkpoint_sha256': checksum, 'feature_manifest_sha256': cache.sha,
        'ids': [cache.records[i]['id'] for i in indices], 'implementation_sha256': digest(__file__)})
    report = validation(model, codec, cache, indices, a.device)
    report.update(checkpoint_sha256=checksum, feature_manifest_sha256=cache.sha, step=state['step'],
                  scope='Training-only conditional chunk probe; never executable coding acceptance', quality_ready=False)
    write_json(out / 'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'codec', 'cache', 'output'): p.add_argument('--' + name, required=True)
    p.add_argument('--count', type=int, default=16); p.add_argument('--device', default='cuda:0')
    run(p.parse_args())
