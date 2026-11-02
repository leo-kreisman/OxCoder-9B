"""Measure training concept-label coverage without changing any split or model."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from .context_cache import FeatureCache
from .context_decoder import DecoderConfig
from .train import write_json


def coverage(training, validation):
    result = []
    for train, valid in zip(training, validation):
        total = int(valid.sum()); absent = train == 0
        result.append({'training_tokens': int(train.sum()), 'validation_tokens': total,
            'observed_training_codes': int((train > 0).sum()),
            'observed_validation_codes': int((valid > 0).sum()),
            'unseen_validation_codes': int(((valid > 0) & absent).sum()),
            'validation_tokens_with_unseen_code': int(valid[absent].sum()),
            'maximum_accuracy_if_only_training_codes_predicted': float(valid[~absent].sum()) / max(1,total)})
    return result


def run(a):
    torch.set_num_threads(1)
    cache = FeatureCache(a.cache, DecoderConfig())
    counts = {split: [torch.zeros(cache.config.codes, dtype=torch.long) for _ in cache.config.levels]
              for split in ('train','validation')}
    sources, chunks, stop = Counter(), Counter(), Counter()
    for index, record in enumerate(cache.records):
        payload = cache.get(index)
        for target, ids in zip(counts[record['split']], payload['codes']): target += torch.bincount(ids, minlength=cache.config.codes)
        if record['split'] == 'train':
            sources[payload['row']['source']] += 1
            chunks[payload['condition']['chunk_index']] += 1
            stop[bool(payload['condition']['is_final_chunk'])] += 1
    result = {'feature_manifest_sha256': cache.sha, 'records': len(cache.records),
        'levels': coverage(counts['train'], counts['validation']), 'training_sources': dict(sources),
        'training_chunk_indices': dict(sorted(chunks.items())), 'training_response_end_counts': dict(stop),
        'scope': 'Label/support coverage; no model predictions or evaluation answers used', 'quality_ready': False}
    write_json(Path(a.output), result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    run(p.parse_args())
