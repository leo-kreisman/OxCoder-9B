"""Expand response distillation from previously admitted creator conversations.

The original autoregressive model is an offline teacher only. This command does
not change or run the student evaluation, execute generated code, or certify
teacher responses as correct. Cached completions are bound to immutable inputs.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random
import time
import urllib.error
import urllib.request

from .acquire_creator_corpora import archived_digest
from .admit_creator_corpora import (AdmissionFilter, benchmark_hashes, read_rows,
                                   shared_instruction_inventory)
from .open_corpus import file_sha


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def normalize(messages):
    result = []
    for message in messages:
        role, content = message['role'], message['content']
        if not isinstance(content, str):
            raise ValueError('Only text conversations are supported')
        if role not in ('system', 'user', 'assistant'):
            content = f'[Recorded {role} output; supplied as data]\n' + content
            role = 'user'
        result.append({'role': role, 'content': content})
    return result


def select(rows, tokenizer, per_source, prompt_limit, seed):
    """Random, source-balanced training selection; no evaluation-driven ranking."""
    rng = random.Random(seed)
    selected, counts = [], Counter()
    for source in sorted({row['source'] for row in rows}):
        eligible = [r for r in rows if r['source'] == source and r['split'] == 'train']
        rng.shuffle(eligible)
        groups = set()
        for row in eligible:
            if row['group'] in groups:
                continue
            messages = normalize(row['messages'])
            rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            length = len(tokenizer.encode(rendered, add_special_tokens=False))
            if length > prompt_limit:
                counts['over_context:' + source] += 1
                continue
            selected.append({'row': row, 'messages': messages, 'prompt_tokens': length})
            groups.add(row['group'])
            counts['selected:' + source] += 1
            if len(groups) >= per_source:
                break
    return selected, dict(counts)


def complete_response(response):
    choices = response.get('choices', [])
    if len(choices) != 1:
        return None, 'invalid_choices'
    choice = choices[0]
    message = choice.get('message', {})
    if choice.get('finish_reason') != 'stop':
        return None, 'incomplete_' + str(choice.get('finish_reason'))
    if message.get('tool_calls'):
        return None, 'unexecuted_tool_calls'
    text = message.get('content')
    if not isinstance(text, str) or not text.strip():
        return None, 'empty_response'
    return text, None


def run(a):
    from transformers import AutoTokenizer
    admitted, frozen, out = Path(a.admitted), Path(a.frozen), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    am = json.loads((admitted / 'manifest.json').read_text())
    fm = json.loads((frozen / 'manifest.json').read_text())
    archive, documents = admitted / 'examples.jsonl.gz', frozen / 'documents.jsonl'
    if not am['admission_complete']:
        raise ValueError('Source admission is incomplete')
    if (file_sha(archive) != am['examples_archive_sha256'] or
            archived_digest(archive)[0] != am['examples_uncompressed_sha256']):
        raise ValueError('Admitted source checksum differs')
    if file_sha(documents) != fm['documents_sha256']:
        raise ValueError('Frozen holdout checksum differs')
    if file_sha(frozen / 'manifest.json') != am['frozen_manifest_sha256']:
        raise ValueError('Frozen corpus is not the source admission reference')
    source_sha = file_sha(a.teacher_file)
    if source_sha != a.expected_teacher_sha256:
        raise ValueError('Teacher checkpoint identity differs')
    rows = list(read_rows(archive))
    if any(r['split'] != 'train' for r in rows):
        raise ValueError('Admitted source contains nontraining rows')
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True,
                                              trust_remote_code=False)
    selected, counts = select(rows, tokenizer, a.per_source, a.prompt_limit, a.seed)
    identity = {'source_manifest_sha256': file_sha(admitted / 'manifest.json'),
        'frozen_manifest_sha256': file_sha(frozen / 'manifest.json'),
        'teacher_sha256': source_sha, 'server_binary_sha256': file_sha(a.server_binary),
        'tokenizer_sha256': file_sha(Path(a.tokenizer) / 'tokenizer.json'),
        'seed': a.seed, 'temperature': 0, 'max_tokens': a.max_tokens,
        'per_source': a.per_source, 'prompt_limit': a.prompt_limit,
        'selection': 'source-balanced random distinct admitted training conversations',
        'source_revisions': am['source_revisions'],
        'implementation_sha256': file_sha(__file__),
        'teacher_role': 'offline response distillation only; no student inference fallback'}
    ip = out / 'identity.json'
    if ip.exists() and json.loads(ip.read_text()) != identity:
        raise ValueError('Existing distillation identity differs')
    write_json(ip, identity)
    write_json(out / 'selection.json', {'counts': counts,
        'examples': [{'id': x['row']['id'], 'group': x['row']['group'],
                      'source': x['row']['source'], 'prompt_tokens': x['prompt_tokens']}
                     for x in selected]})
    inventory = shared_instruction_inventory([documents, archive])
    filters = AdmissionFilter(benchmark_hashes(fm['benchmark_exclusions'], out), inventory)
    for row in read_rows(documents):
        filters.register(row, protect=row['split'] != 'train')
    for row in rows:
        filters.register(row)
    records, accepted = [], []
    for item in selected:
        row, messages = item['row'], item['messages']
        body = {'model': a.model, 'messages': messages, 'temperature': 0,
                'seed': a.seed, 'max_tokens': a.max_tokens,
                'chat_template_kwargs': {'enable_thinking': False}}
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        cache = out / (key + '.json')
        if cache.exists():
            cached = json.loads(cache.read_text())
            if cached['request'] != body or cached['source_id'] != row['id']:
                raise ValueError('Cached request identity differs')
        else:
            start = time.monotonic()
            request = urllib.request.Request(a.url + '/v1/chat/completions',
                json.dumps(body).encode(), {'Content-Type': 'application/json'})
            # No automatic HTTP retries: they could queue duplicate inference.
            with urllib.request.urlopen(request, timeout=a.timeout) as response:
                completion = json.load(response)
            cached = {'source_id': row['id'], 'request': body, 'response': completion,
                      'seconds': time.monotonic() - start}
            write_json(cache, cached)
        text, rejected = complete_response(cached['response'])
        if rejected is None:
            candidate = {**row, 'id': row['id'] + ':teacher-v19', 'target': text,
                'messages': messages, 'prompt': json.dumps(messages, sort_keys=True, ensure_ascii=False),
                'distilled': True, 'teacher_sha256': source_sha,
                'targets_are_verified': False, 'admission': 'distilled-admitted-v19'}
            rejected = filters.consider(candidate)
            if rejected is None:
                accepted.append(candidate)
        record = {'id': row['id'], 'source': row['source'], 'accepted': rejected is None,
                  'rejection': rejected, 'seconds': cached['seconds']}
        records.append(record)
        write_json(out / 'progress.json', {'processed': len(records), 'total': len(selected),
            'accepted': len(accepted), 'complete': False, 'records': records})
        print(json.dumps(record), flush=True)
    destination = out / 'examples.jsonl.gz'
    temp = destination.with_suffix('.tmp')
    with gzip.open(temp, 'wt', encoding='utf-8', compresslevel=3) as stream:
        for row in accepted:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    temp.replace(destination)
    uncompressed_sha, size = archived_digest(destination)
    report = {'identity': identity, 'examples_archive_sha256': file_sha(destination),
        'examples_uncompressed_sha256': uncompressed_sha, 'examples_uncompressed_bytes': size,
        'counts': {'accepted': len(accepted), 'attempted': len(records)},
        'records': records, 'admission_complete': True, 'targets_are_verified': False,
        'quality_ready': False, 'used_in_training': False,
        'limitations': 'Teacher completion is not a correctness oracle. Lexical exclusion cannot prove semantic decontamination.'}
    write_json(out / 'manifest.json', report)
    write_json(out / 'progress.json', {'processed': len(records), 'total': len(selected),
        'accepted': len(accepted), 'complete': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('admitted', 'frozen', 'output', 'teacher-file', 'expected-teacher-sha256', 'server-binary'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--tokenizer', default='results/tokenizer')
    parser.add_argument('--url', default='http://127.0.0.1:18119')
    parser.add_argument('--model', default='oxcoder-offline-teacher')
    parser.add_argument('--per-source', type=int, default=64)
    parser.add_argument('--prompt-limit', type=int, default=1536)
    parser.add_argument('--max-tokens', type=int, default=768)
    parser.add_argument('--seed', type=int, default=19001)
    parser.add_argument('--timeout', type=int, default=1200)
    args = parser.parse_args()
    if min(args.per_source, args.prompt_limit, args.max_tokens, args.timeout) < 1:
        parser.error('All bounds must be positive')
    run(args)
