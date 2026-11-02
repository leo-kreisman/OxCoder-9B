"""Bounded public-data acquisition with source revisions and conversation splits.

Downloaded text is data only. No dataset scripts, tools or generated code execute.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import HfApi

DATASETS = {
    'TeichAI/Ox-Alpha-10k':'ae678ee78880e14251a553a0d0dc8ec6bd6dad6c',
    'TeichAI/DeepSeek-v4-Pro-Agent':'abbf7b71b145633b60beb21ddc0d158582fd80dd',
}


def text_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if part.get('type') in ('text', 'thinking'):
                parts.append(part.get('text', part.get('thinking', '')))
            elif part.get('type') in ('toolCall', 'tool_use'):
                parts.append(json.dumps(part, sort_keys=True, ensure_ascii=False))
        return '\n'.join(parts)
    return ''


def pairs(messages):
    history = []
    for m in messages:
        content = text_content(m.get('content', ''))
        role = m.get('role', '')
        if role == 'assistant' and content and history:
            yield json.dumps(history, ensure_ascii=False, sort_keys=True), content
        if content:
            history.append({'role': role, 'content': content})


def chunks(text, limit):
    """UTF-8 safe, explicit non-overlapping chunks; no silent token truncation."""
    current, size = [], 0
    for char in text:
        n = len(char.encode('utf-8'))
        if n > limit:
            raise ValueError('Window too small for Unicode')
        if size + n > limit:
            yield ''.join(current)
            current, size = [], 0
        current.append(char)
        size += n
    if current:
        yield ''.join(current)


def group_split(identity):
    value = int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16) % 100
    return 'test' if value < 10 else 'validation' if value < 20 else 'train'


def download_lines(repo, revision, filename, max_lines, max_bytes):
    url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}'
    rows, raw = [], bytearray()
    with urllib.request.urlopen(url, timeout=60) as stream:
        for _ in range(max_lines):
            line = stream.readline(max_bytes - len(raw) + 1)
            if not line:
                break
            if len(raw) + len(line) > max_bytes:
                raise ValueError('Dataset byte bound exceeded; reduce sample size')
            raw.extend(line)
            if line.strip():
                rows.append(json.loads(line))
    return rows, {'file':filename, 'rows_read':len(rows), 'bytes_read':len(raw),
                  'read_prefix_sha256':hashlib.sha256(raw).hexdigest()}


def prepare(output, length=128, alpha_rows=256, agent_files=24, per_conversation=12):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    examples, sources, seen = [], [], set()
    def add(conversation, repo, filename, messages):
        # Group by content: duplicate conversations cannot leak across sources.
        content_key = hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        split = group_split(content_key)
        count = 0
        for prompt, target in pairs(messages):
            previous = ''
            for index, chunk in enumerate(chunks(target, length - 1)):
                key = hashlib.sha256(chunk.encode()).hexdigest()
                if key in seen:
                    previous += chunk
                    continue
                seen.add(key)
                examples.append({'id':f'{conversation}:{index}:{count}', 'group':content_key,
                    'split':split, 'source':repo, 'source_file':filename,
                    'prompt':prompt + '\nContinue this assistant response exactly:\n' + previous[-2048:],
                    'target':chunk, 'chunked':True})
                previous += chunk
                count += 1
                if count >= per_conversation:
                    return
    for repo, revision in DATASETS.items():
        files = ['train.jsonl'] if 'Alpha' in repo else sorted(
            x.rfilename for x in HfApi().dataset_info(repo, revision=revision).siblings
            if x.rfilename.endswith('.jsonl'))[:agent_files]
        records = []
        for filename in files:
            rows, record = download_lines(repo, revision, filename,
                alpha_rows if 'Alpha' in repo else 2000, 32 << 20)
            records.append(record)
            if 'Alpha' in repo:
                for index, row in enumerate(rows):
                    add(f'{repo}:{index}', repo, filename, row['messages'])
            else:
                messages = [r['message'] for r in rows if isinstance(r.get('message'), dict)]
                add(f'{repo}:{filename}', repo, filename, messages)
        sources.append({'dataset':repo, 'revision':revision, 'files':records})
    counts = {s:sum(r['split']==s for r in examples) for s in ('train','validation','test')}
    if min(counts.values()) == 0:
        raise ValueError(f'Empty split; increase acquisition: {counts}')
    path = output / 'examples.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in examples))
    manifest = {'sources':sources, 'length':length, 'counts':counts,
        'examples_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'split_policy':'conversation-content hash, 80/10/10; globally deduplicated target chunks',
        'scope':'bounded pilot subset; chunks are not complete coding tasks',
        'synthetic_teacher':'these datasets are not OxCoder-generated ground truth'}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--length', type=int, default=128)
    p.add_argument('--alpha-rows', type=int, default=256)
    p.add_argument('--agent-files', type=int, default=24)
    a = p.parse_args()
    prepare(a.output, a.length, a.alpha_rows, a.agent_files)
