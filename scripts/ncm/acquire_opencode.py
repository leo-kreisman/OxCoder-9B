"""Acquire a bounded, pinned Python instruction supplement with holdout exclusion.

Publisher test scores are a selection signal, not independently verified labels.
Downloaded programs and tests are never executed by this acquisition command.
"""
import argparse
import ast
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import urllib.request

from .acquire_creator_corpora import archived_digest
from .admit_creator_corpora import (AdmissionFilter, benchmark_hashes, read_rows,
                                   shared_instruction_inventory)
from .distill_admitted import write_json
from .open_corpus import download, file_sha, sha

DATASET = 'nvidia/OpenCodeInstruct'
REVISION = '8f3ba5bafe4d6e8db46082cf7ae6741bc370604d'
SHARD = 'data/train-00000-of-00050.parquet'
SHARD_SHA = '342757e0c6b706c8f68cf0a867ead5c417d0273726453dbcfc934f5c3c6ca891'


def candidate(original, max_bytes):
    try:
        score = float(original['average_test_score'])
        statuses = json.loads(original['tests_execution_status'])
    except (TypeError, ValueError, KeyError):
        return None, 'invalid_publisher_verification'
    if score != 1.0 or not isinstance(statuses, list) or not statuses or any(x != 'pass' for x in statuses):
        return None, 'publisher_test_failures'
    prompt, target = original.get('input'), original.get('output')
    if not isinstance(prompt, str) or not isinstance(target, str) or not prompt.strip() or not target.strip():
        return None, 'empty_or_nontext'
    if len(prompt.encode()) > 8192 or not 32 <= len(target.encode()) <= max_bytes:
        return None, 'length'
    blocks = re.findall(r'```(?:python|py)\s*\n(.*?)```', target, re.S | re.I)
    code = blocks[0] if len(blocks) == 1 else target if not blocks else None
    if code is None:
        return None, 'multiple_python_blocks'
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None, 'not_parseable_python'
    if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for n in tree.body):
        return None, 'no_top_level_definition'
    group = sha(DATASET + ':' + str(original['id']))
    messages = [{'role': 'user', 'content': prompt}]
    return {'id': group + ':answer', 'group': group, 'split': 'train',
        'source': DATASET, 'source_revision': REVISION, 'source_file': SHARD,
        'original_id': original['id'], 'messages': messages,
        'prompt': json.dumps(messages, sort_keys=True, ensure_ascii=False),
        'target': target, 'distilled': False, 'targets_are_verified': False,
        'publisher_test_score': score, 'publisher_test_count': len(statuses),
        'admission': 'opencode-python-supplement-v19'}, None


def run(a):
    import pyarrow.parquet as pq
    frozen, admitted, out = Path(a.frozen), Path(a.admitted), Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    fm = json.loads((frozen / 'manifest.json').read_text())
    am = json.loads((admitted / 'manifest.json').read_text())
    documents, archive = frozen / 'documents.jsonl', admitted / 'examples.jsonl.gz'
    if file_sha(documents) != fm['documents_sha256']:
        raise ValueError('Frozen corpus checksum differs')
    if (not am['admission_complete'] or file_sha(archive) != am['examples_archive_sha256']
        or archived_digest(archive)[0] != am['examples_uncompressed_sha256']
        or file_sha(frozen / 'manifest.json') != am['frozen_manifest_sha256']):
        raise ValueError('Creator source admission differs')
    shard = out / 'source.parquet'
    record = download(DATASET, REVISION, SHARD, shard, 140 << 20)
    if record['sha256'] != SHARD_SHA:
        raise ValueError('Pinned source shard checksum differs')
    url = f'https://huggingface.co/datasets/{DATASET}/raw/{REVISION}/README.md'
    with urllib.request.urlopen(url, timeout=60) as stream:
        card = stream.read(1 << 20)
    (out / 'SOURCE-CARD.md').write_bytes(card)
    inventory = shared_instruction_inventory([documents, archive])
    filters = AdmissionFilter(benchmark_hashes(fm['benchmark_exclusions'], out), inventory)
    for row in read_rows(documents):
        filters.register(row, protect=row['split'] != 'train')
    for row in read_rows(archive):
        filters.register(row)
    counts = Counter()
    destination = out / 'examples.jsonl.gz'
    with gzip.open(destination, 'wt', encoding='utf-8', compresslevel=3) as output:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=128):
            for original in batch.to_pylist():
                counts['source_rows'] += 1
                row, reason = candidate(original, a.max_target_bytes)
                if reason is None:
                    reason = filters.consider(row)
                if reason is not None:
                    counts['rejected:' + reason] += 1
                else:
                    output.write(json.dumps(row, ensure_ascii=False) + '\n')
                    counts['admitted'] += 1
                if counts['admitted'] >= a.maximum:
                    break
            write_json(out / 'progress.json', {'counts': dict(counts), 'complete': False})
            if counts['admitted'] >= a.maximum:
                break
    uncompressed, size = archived_digest(destination)
    report = {'dataset': DATASET, 'revision': REVISION, 'license': 'CC-BY-4.0',
        'attribution': 'NVIDIA Corporation, OpenCodeInstruct (Ahmad et al., 2025)',
        'source': record, 'source_card_sha256': file_sha(out / 'SOURCE-CARD.md'),
        'source_url': f'https://huggingface.co/datasets/{DATASET}/tree/{REVISION}',
        'frozen_manifest_sha256': file_sha(frozen / 'manifest.json'),
        'creator_manifest_sha256': file_sha(admitted / 'manifest.json'),
        'examples_archive_sha256': file_sha(destination),
        'examples_uncompressed_sha256': uncompressed, 'examples_uncompressed_bytes': size,
        'counts': dict(counts), 'maximum': a.maximum, 'max_target_bytes': a.max_target_bytes,
        'implementation_sha256': file_sha(__file__),
        'admission_complete': True, 'targets_are_verified': False,
        'quality_ready': False, 'used_in_training': False,
        'selection': 'first pinned shard; publisher all-tests-pass; parseable Python; bounded length; lexical and conversation holdout exclusion',
        'limitations': 'No independent execution verification. Conservative lexical filtering cannot prove semantic decontamination.'}
    write_json(out / 'manifest.json', report)
    write_json(out / 'progress.json', {'counts': dict(counts), 'complete': True})
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('frozen', 'admitted', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--maximum', type=int, default=10000)
    parser.add_argument('--max-target-bytes', type=int, default=8192)
    args = parser.parse_args()
    if args.maximum < 1 or args.max_target_bytes < 32:
        parser.error('Invalid bounds')
    run(args)
