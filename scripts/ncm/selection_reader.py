"""Exact JSON serialization cache for admitted selection.pt envelopes only."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from .prepared_reader import digest, identical


def restore(data):
    if not isinstance(data, dict) or list(data) != ['prepared'] or not isinstance(data['prepared'], list):
        raise ValueError('Expected selection envelope containing only prepared rows')
    if any(not isinstance(pair, (list, tuple)) or len(pair) != 2
           or any(not isinstance(item, dict) for item in pair) for pair in data['prepared']):
        raise ValueError('Invalid selection row/condition pairs')
    data['prepared'] = [tuple(pair) for pair in data['prepared']]
    return data


def load(directory, expected_sha, max_bytes=512 << 20):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if (manifest.get('format') != 'ncm-selection-json-v35'
            or manifest.get('source_sha256') != expected_sha
            or manifest.get('exact_type_value_comparison') is not True):
        raise ValueError('Unqualified selection serialization')
    path = directory / 'data.json'
    if not 0 < path.stat().st_size <= max_bytes:
        raise ValueError('Selection JSON exceeds size bound')
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != manifest['json_sha256']:
        raise ValueError('Selection JSON checksum differs')
    result = restore(json.loads(encoded))
    if len(result['prepared']) != manifest['rows']:
        raise ValueError('Selection row count differs')
    return result


def prepare(source, expected_sha, output):
    import torch
    source, output = Path(source), Path(output)
    if digest(source) != expected_sha:
        raise ValueError('Selection source checksum differs')
    started = time.monotonic()
    original = torch.load(source, weights_only=True, map_location='cpu')
    seconds = time.monotonic() - started
    if digest(source) != expected_sha:
        raise ValueError('Selection changed during conversion')
    encoded = json.dumps(original, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()
    restored = restore(json.loads(encoded))
    if not identical(original, restored):
        raise ValueError('Selection types, values or ordering changed')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'data.json').write_bytes(encoded)
    manifest = {'format': 'ncm-selection-json-v35', 'source_sha256': expected_sha,
                'json_sha256': hashlib.sha256(encoded).hexdigest(), 'rows': len(restored['prepared']),
                'json_bytes': len(encoded), 'exact_type_value_comparison': True,
                'torch_weights_only_seconds': seconds, 'implementation_sha256': digest(__file__),
                'quality_ready': False}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'sha256', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.source, a.sha256, a.output)), flush=True)
