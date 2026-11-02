"""Checksummed JSON cache for internally prepared training data.

The one-time conversion uses torch.load(weights_only=True). Repeated reads use
JSON, never unrestricted pickle. No active trainer silently switches readers.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def envelope(data):
    if (not isinstance(data, dict) or set(data) != {'prepared', 'data_manifest', 'prompt_tokens'}
            or not isinstance(data['prepared'], list) or not isinstance(data['data_manifest'], dict)
            or type(data['prompt_tokens']) is not int):
        raise ValueError('Unexpected prepared-data envelope')
    if any(not isinstance(pair, (tuple, list)) or len(pair) != 2
           or any(not isinstance(x, dict) for x in pair) for pair in data['prepared']):
        raise ValueError('Expected row/condition pairs')
    return data


def restore(data):
    envelope(data)
    data['prepared'] = [tuple(pair) for pair in data['prepared']]
    return data


def identical(left, right):
    """One-time type-and-value comparison, including tuple versus list."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return list(left) == list(right) and all(identical(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(identical(a, b) for a, b in zip(left, right))
    if type(left) is float:
        return math.isfinite(left) and left == right
    if left is None or type(left) in (bool, int, str):
        return left == right
    raise ValueError('Unsupported prepared-data type')


def load(directory, expected_source_sha256, max_bytes=512 << 20):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if (manifest.get('format') != 'ncm-prepared-json-v1' or not manifest.get('exact_type_value_comparison')
            or manifest.get('source_sha256') != expected_source_sha256):
        raise ValueError('Unqualified or mismatched prepared cache')
    path = directory / 'data.json'
    if not 0 < path.stat().st_size <= max_bytes:
        raise ValueError('Prepared JSON exceeds its size bound')
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != manifest['json_sha256']:
        raise ValueError('Prepared JSON checksum differs')
    data = restore(json.loads(encoded))
    if len(data['prepared']) != manifest['rows']:
        raise ValueError('Prepared JSON row count differs')
    return data


def prepare(source, expected_sha256, output):
    import torch
    source, output = Path(source), Path(output)
    if digest(source) != expected_sha256:
        raise ValueError('Prepared source checksum differs')
    started = time.monotonic()
    data = envelope(torch.load(source, weights_only=True, map_location='cpu'))
    seconds = time.monotonic() - started
    if digest(source) != expected_sha256:
        raise ValueError('Source changed during conversion')
    encoded = json.dumps(data, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()
    restored = restore(json.loads(encoded))
    if not identical(data, restored):
        raise ValueError('JSON round trip changes prepared types, values, or ordering')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'data.json').write_bytes(encoded)
    report = {'format': 'ncm-prepared-json-v1', 'source_sha256': expected_sha256,
        'json_sha256': hashlib.sha256(encoded).hexdigest(), 'json_bytes': len(encoded),
        'rows': len(data['prepared']), 'exact_type_value_comparison': True,
        'torch_weights_only_seconds': seconds, 'implementation_sha256': digest(__file__),
        'scope': 'Prepared data only; no model/optimizer/checkpoint deserialization', 'quality_ready': False}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'sha256', 'output']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.sha256, args.output)), flush=True)
