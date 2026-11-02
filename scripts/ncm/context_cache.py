"""Validate and read prompt-only feature shards for the native NCM decoder."""
import argparse
from collections import Counter, OrderedDict
import json
import os
from pathlib import Path

import torch

from .backbone import digest
from .codec import encode
from .train import write_json


def merge(selection, shards, output):
    selected = Path(selection)
    metadata = json.loads((selected / 'manifest.json').read_text())
    selection_sha = digest(selected / 'selection.pt')
    if selection_sha != metadata['selection_sha256']:
        raise ValueError('Selection checksum mismatch')
    items = torch.load(selected / 'selection.pt', weights_only=True, map_location='cpu')['prepared']
    expected = {i: (r['id'], r['group'], r['split']) for i, (r, _) in enumerate(items)}
    seen, numbers, records, rejected, provenance = set(), set(), [], [], []
    common = None
    out = Path(output).resolve()
    for folder in map(Path, shards):
        identity = json.loads((folder / 'identity.json').read_text())
        report = json.loads((folder / 'manifest.json').read_text())
        if not report.get('complete') or digest(folder / 'identity.json') != report['identity_sha256']:
            raise ValueError('Incomplete or altered feature shard')
        if identity['selection_sha256'] != selection_sha or identity['shards'] != len(shards):
            raise ValueError('Shard selection or partition differs')
        number = identity['shard']
        if number in numbers or not 0 <= number < len(shards):
            raise ValueError('Duplicate or invalid shard number')
        numbers.add(number)
        shared = {k: v for k, v in identity.items() if k not in ('shard', 'shards')}
        if common is not None and common != shared:
            raise ValueError('Feature shard identities differ')
        common = shared
        for record in report['records'] + report['rejected']:
            index = record['index']
            if (index in seen or index not in expected or index % len(shards) != number
                    or record['id'] != expected[index][0] or record['split'] != expected[index][2]):
                raise ValueError('Wrong, duplicate or misassigned feature row')
            seen.add(index)
        for record in report['records']:
            if record['group'] != expected[record['index']][1]:
                raise ValueError('Feature conversation identity differs')
            path = folder / record['path']
            if digest(path) != record['sha256']:
                raise ValueError('Feature payload checksum mismatch')
            records.append({**record, 'path': os.path.relpath(path.resolve(), out.parent)})
        if any(r['split'] != 'train' or r['reason'] != 'codec reconstruction not exact' for r in report['rejected']):
            raise ValueError('Only inexact training codec targets may be excluded')
        rejected.extend(report['rejected'])
        provenance.append({'path': os.path.relpath(folder.resolve(), out.parent),
                           'identity_sha256': report['identity_sha256'],
                           'manifest_sha256': digest(folder / 'manifest.json')})
    if seen != set(expected) or numbers != set(range(len(shards))):
        raise ValueError('Feature extraction did not account for every selected row')
    counts = dict(Counter(r['split'] for r in records))
    if counts.get('validation') != metadata['counts']['validation']:
        raise ValueError('Validation rows lost')
    training = [r for r in records if r['split'] == 'train']
    if ({r['source'] for r in training} != set(metadata['training_sources'])
            or not any(r['distilled'] for r in training)):
        raise ValueError('Training cache lost a source or all distillation')
    result = {'identity': common, 'selection_manifest_sha256': digest(selected / 'manifest.json'),
              'records': sorted(records, key=lambda x: x['index']), 'rejected': rejected,
              'counts': counts, 'shards': provenance, 'complete': True, 'quality_ready': False}
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    write_json(out, result)
    return result


class FeatureCache:
    def __init__(self, manifest, config, maximum_cached_bytes=512 << 20):
        self.path = Path(manifest)
        self.sha = digest(self.path)
        self.metadata = json.loads(self.path.read_text())
        if not self.metadata.get('complete'):
            raise ValueError('Incomplete merged feature cache')
        self.config = config
        self.records = self.metadata['records']
        self.limit = maximum_cached_bytes
        self.cached, self.cached_bytes = OrderedDict(), 0
        seen, groups = set(), {}
        for r in self.records:
            if r['id'] in seen or groups.setdefault(r['group'], r['split']) != r['split']:
                raise ValueError('Duplicate features or leaking conversations')
            seen.add(r['id'])
            if r['split'] not in ('train', 'validation'):
                raise ValueError('Unknown feature split')

    def get(self, index):
        if index in self.cached:
            self.cached.move_to_end(index)
            return self.cached[index][0]
        r = self.records[index]
        path = self.path.parent / r['path']
        if digest(path) != r['sha256']:
            raise ValueError('Cached feature changed')
        payload = torch.load(path, weights_only=True, map_location='cpu')
        h, codes, condition, row = (payload[k] for k in ('hidden', 'codes', 'condition', 'row'))
        if (row['id'] != r['id'] or row['group'] != r['group'] or row['split'] != r['split']
                or h.dtype != torch.bfloat16 or h.shape != (r['prompt_tokens'], self.config.source_width)
                or len(condition['ids']) != h.shape[0] or not 0 < h.shape[0] <= self.config.maximum_prompt
                or not torch.isfinite(h).all()
                or len(codes) != len(self.config.levels)):
            raise ValueError('Invalid prompt feature payload')
        for size, code in zip(self.config.levels, codes):
            if code.dtype != torch.long or code.shape != (size,) or not bool(((code >= 0) & (code < self.config.codes)).all()):
                raise ValueError('Invalid concept labels')
        if (payload['bytes'].tolist() != encode(row['target'], self.config.levels[-1])
                or payload['codec_exact'] != r['codec_exact']
                or (r['split'] == 'train' and not payload['codec_exact'])):
            raise ValueError('Invalid frozen codec targets')
        size = h.numel() * h.element_size() + sum(x.numel() * x.element_size() for x in codes)
        if size <= self.limit:
            while self.cached and self.cached_bytes + size > self.limit:
                _, (_, old_size) = self.cached.popitem(last=False)
                self.cached_bytes -= old_size
            self.cached[index] = payload, size
            self.cached_bytes += size
        return payload


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', required=True)
    p.add_argument('--shards', nargs='+', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = merge(a.selection, a.shards, a.output)
    print(json.dumps({'counts': result['counts'], 'rejected': len(result['rejected'])}), flush=True)
