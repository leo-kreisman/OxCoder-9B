"""Bounded, pinned source-code acquisition. Downloaded code is never executed."""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import gzip
import json
from pathlib import Path
import random
import re
import shutil
import urllib.request

STACK = 'HuggingFaceCode/stack-v3-train'
REVISION = '1f61b735bc0a5698345ce2196730f24bfa467f33'
PARROT = 'codeparrot/codeparrot-clean'
PARROT_REVISION = '35a59fb025bc0a102f7d96eac09d145b896d487b'
BENCHMARKS = {
    'openai/openai_humaneval': '7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544',
    'google-research-datasets/mbpp': '4bb6404fdc6cacfda99d4ac4205087b89d32030c',
}


class DownloadLimit(RuntimeError):
    pass


def sha(x):
    return hashlib.sha256(x.encode() if isinstance(x, str) else x).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def tokens(text):
    return re.findall(r'\w+|[^\w\s]', text)


def shingles(text, n=32):
    ts = tokens(text)
    return {sha('\x00'.join(ts[i:i+n]))[:24] for i in range(max(0, len(ts)-n+1))}


def split_for(group):
    n = int(sha(group)[:8], 16) % 100
    return 'validation' if n < 5 else 'test' if n < 10 else 'train'


def download(repo, revision, name, dest, limit=600 << 20):
    if shutil.disk_usage(dest.parent).free < limit + (3 << 30):
        raise RuntimeError('Keep 3 GiB free plus download allowance')
    url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}'
    size = 0
    with urllib.request.urlopen(url, timeout=60) as r, open(dest, 'wb') as f:
        if int(r.headers.get('Content-Length', '0')) > limit:
            raise DownloadLimit('Shard exceeds hard byte bound')
        while b := r.read(1 << 20):
            size += len(b)
            if size > limit:
                raise DownloadLimit('Download exceeded hard byte bound')
            f.write(b)
    return {'file': name, 'sha256': file_sha(dest), 'bytes': size}


def benchmark_filter(root):
    import pyarrow.parquet as pq
    hashes, records = set(), []
    for repo, rev in BENCHMARKS.items():
        with urllib.request.urlopen(f'https://huggingface.co/api/datasets/{repo}/revision/{rev}', timeout=30) as r:
            info = json.load(r)
        names = [x['rfilename'] for x in info['siblings'] if x['rfilename'].endswith('.parquet')
                 and not x['rfilename'].startswith('sanitized/')]
        for name in names:
            path = root / 'benchmark.parquet'
            record = download(repo, rev, name, path, 10 << 20)
            for row in pq.read_table(path).to_pylist():
                for key in ('prompt', 'canonical_solution', 'code', 'text', 'test'):
                    if isinstance(row.get(key), str):
                        hashes.update(shingles(row[key]))
                if 'canonical_solution' in row:
                    hashes.update(shingles(row.get('prompt', '') + row['canonical_solution']))
            records.append({'dataset': repo, 'revision': rev, **record})
            path.unlink()
    return hashes, records


class NearDuplicates:
    """Conservative SimHash filter; not a guarantee against all contamination."""
    def __init__(self):
        self.bands = defaultdict(list)
        self.exact = set()

    def duplicate(self, text):
        words = tokens(text)
        key = sha('\x00'.join(words))
        if key in self.exact:
            return True
        counts = Counter(words)
        import numpy as np
        bits = np.unpackbits(np.frombuffer(b''.join(hashlib.blake2b(w.encode(), digest_size=8).digest()
                             for w in counts), dtype=np.uint8)).reshape(-1,64).astype(np.int64)
        sums = np.asarray(list(counts.values())) @ (2*bits-1)
        signature = int.from_bytes(np.packbits(sums >= 0).tobytes(), 'big')
        keys = [(i, (signature >> (16*i)) & 65535) for i in range(4)]
        candidates = {x for k in keys for x in self.bands[k]}
        if any((signature ^ x).bit_count() <= 3 for x in candidates):
            return True
        self.exact.add(key)
        for k in keys:
            self.bands[k].append(signature)
        return False


def eligible(file):
    if file.get('language') != 'Python' or file.get('license_type') != 'permissive':
        return False
    if file.get('is_vendor') or file.get('is_generated'):
        return False
    if re.search(r'human.?eval|bigcodebench|livecodebench|(?:^|/)mbpp', file.get('file_path', ''), re.I):
        return False
    text = file.get('content', '')
    if not isinstance(text, str) or not 128 <= len(text.encode()) <= 32768:
        return False
    try:
        ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return False
    return True


def acquire(a):
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    tmp = out / 'download'
    tmp.mkdir()
    counter = Tokenizer.from_file(a.counter_tokenizer)
    benchmarks, benchmark_records = benchmark_filter(tmp)
    with urllib.request.urlopen(f'https://huggingface.co/api/datasets/{STACK}/revision/{REVISION}', timeout=30) as r:
        info = json.load(r)
    names = sorted(x['rfilename'] for x in info['siblings'] if x['rfilename'].endswith('.parquet'))
    random.Random(a.seed).shuffle(names)
    dedup = NearDuplicates()
    stats, records, groups = Counter(), [], set()
    destination = out / 'documents.jsonl'
    def accept(row, writer):
        text = row['text']
        if benchmarks.intersection(shingles(text)):
            stats['benchmark_rejected'] += 1
            return False
        if dedup.duplicate(text):
            stats['duplicate_rejected'] += 1
            return False
        row['text_sha256'] = sha(text)
        row['counter_tokens'] = len(counter.encode(text, add_special_tokens=False).ids)
        writer.write(json.dumps(row, ensure_ascii=False) + '\n')
        stats[row['split'] + '_documents'] += 1
        stats[row['kind'] + '_tokens'] += row['counter_tokens']
        stats[row['kind'] + '_documents'] += 1
        return True
    with destination.open('w') as writer:
        # Preserve existing complete-conversation splits. No tools in traces run.
        for corpus in a.assistant_corpus:
            manifest = json.loads((Path(corpus)/'manifest.json').read_text())
            source = Path(corpus)/'examples.jsonl'
            if file_sha(source) != manifest['examples_sha256']:
                raise ValueError('Assistant corpus checksum mismatch')
            records.append({'local_corpus': str(source), 'sha256': file_sha(source),
                            'sources': manifest.get('sources'), 'kind': 'assistant'})
            with source.open() as stream:
                for line in stream:
                    r = json.loads(line)
                    accept({'id': r['id'], 'group': r['group'], 'split': r['split'],
                            'source': r['source'], 'kind': 'assistant', 'text': r['target'],
                            'prompt': r.get('prompt', ''),
                            'distilled': r.get('distilled', False)}, writer)
        for name in names[:a.max_shards]:
            path = tmp / 'current.parquet'
            try:
                record = download(STACK, REVISION, name, path)
            except DownloadLimit:
                path.unlink(missing_ok=True)
                records.append({'file':name, 'dataset':STACK, 'revision':REVISION, 'skipped':'over 600 MiB download limit'})
                continue
            before = stats['code_tokens']
            for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
                for repo in batch.to_pylist():
                    group = repo['repo_path'].lower()
                    if group in groups:
                        continue
                    groups.add(group)
                    files = list(repo['files'])
                    random.Random(sha(group)).shuffle(files)
                    kept = 0
                    for f in files:
                        if not eligible(f):
                            continue
                        kept += int(accept({'id': f['content_id'], 'group': group,
                            'split': split_for(group), 'source': STACK, 'kind': 'code',
                            'text': f['content'], 'repo': repo['repo_path'], 'commit': repo['commit_id'],
                            'path': f['file_path'], 'licenses': f.get('detected_licenses'),
                            'license_type': f['license_type']}, writer))
                        if kept >= 8 or stats['code_tokens'] >= a.tokens:
                            break
                    if stats['code_tokens'] >= a.tokens:
                        break
                if stats['code_tokens'] >= a.tokens:
                    break
            record['accepted_code_tokens'] = stats['code_tokens'] - before
            record.update(dataset=STACK, revision=REVISION)
            records.append(record)
            path.unlink()
            writer.flush()
            progress = dict(stats)
            (out/'progress.json').write_text(json.dumps(progress, indent=2))
            print(json.dumps({'shard': name, **progress}), flush=True)
            if stats['code_tokens'] >= a.tokens:
                break
        # Whole-repository Stack shards have low Python yield. Supplement using
        # the explicitly documented Python corpus instead of fetching terabytes.
        if stats['code_tokens'] < a.tokens:
            with urllib.request.urlopen(f'https://huggingface.co/api/datasets/{PARROT}/revision/{PARROT_REVISION}', timeout=30) as r:
                parrot_info = json.load(r)
            parrot_files = sorted(x['rfilename'] for x in parrot_info['siblings'] if x['rfilename'].endswith('.json.gz'))
            random.Random(a.seed).shuffle(parrot_files)
            per_repo = Counter()
            for name in parrot_files[:a.max_python_shards]:
                path = tmp/'python.json.gz'
                record = download(PARROT, PARROT_REVISION, name, path)
                before = stats['code_tokens']
                with gzip.open(path, 'rt') as f:
                    for line in f:
                        row = json.loads(line)
                        group = row['repo_name'].lower()
                        license = row.get('license', '')
                        if license not in ('mit', 'apache-2.0', 'bsd-2-clause', 'bsd-3-clause', 'isc', 'unlicense', 'cc0-1.0'):
                            continue
                        if per_repo[group] >= 8 or group in groups:
                            continue
                        converted = {'language': 'Python', 'license_type': 'permissive',
                                     'file_path': row['path'], 'content': row['content'],
                                     'is_generated': row.get('autogenerated', False)}
                        if not eligible(converted):
                            continue
                        per_repo[group] += int(accept({'id': sha(row['content']), 'group': group,
                            'split': split_for(group), 'source': PARROT, 'kind': 'code',
                            'text': row['content'], 'repo': row['repo_name'], 'path': row['path'],
                            'licenses': [license], 'license_type': 'permissive'}, writer))
                        if stats['code_tokens'] >= a.tokens:
                            break
                record.update(dataset=PARROT, revision=PARROT_REVISION,
                              accepted_code_tokens=stats['code_tokens']-before)
                records.append(record)
                path.unlink()
                writer.flush()
                (out/'progress.json').write_text(json.dumps(dict(stats), indent=2))
                print(json.dumps({'shard': name, **dict(stats)}), flush=True)
                if stats['code_tokens'] >= a.tokens:
                    break
    manifest = {'dataset': STACK, 'revision': REVISION, 'seed': a.seed,
        'documents_sha256': file_sha(destination), 'stats': dict(stats), 'sources': records,
        'benchmark_exclusions': benchmark_records, 'counter_tokenizer_sha256': file_sha(a.counter_tokenizer),
        'requested_code_tokens': a.tokens, 'token_budget_reached': stats['code_tokens'] >= a.tokens,
        'split_policy': 'code repository hash 90/5/5; assistants retain prior conversation splits',
        'dedup': 'global whitespace-token exact + SimHash Hamming <=3; upstream near-dedup',
        'decontamination': '32-lexeme overlaps with HumanEval and full MBPP; benchmark paths excluded; not exhaustive',
        'status': 'complete', 'quality_ready': False}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if min(stats[s+'_documents'] for s in ('train', 'validation', 'test')) < 1000:
        raise RuntimeError('Corpus too small for 1000-document evaluation splits')
    if not manifest['token_budget_reached']:
        raise RuntimeError('Acquisition bound reached before requested token budget; inspect manifest')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--counter-tokenizer', default='results/tokenizer/tokenizer.json')
    p.add_argument('--assistant-corpus', action='append', required=True)
    p.add_argument('--tokens', type=int, default=50_000_000)
    p.add_argument('--max-shards', type=int, default=2)
    p.add_argument('--max-python-shards', type=int, default=12)
    p.add_argument('--seed', type=int, default=37)
    acquire(p.parse_args())
