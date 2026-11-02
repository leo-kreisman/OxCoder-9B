"""Append admitted instruction/teacher archives while preserving frozen evaluation."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .acquire_creator_corpora import archived_digest
from .admit_creator_corpora import read_rows
from .open_corpus import file_sha
from .prepare_creator_expansion import add_response, frozen_evaluation_hash
from .train import save, write_json


def preserve_provenance(rows, original):
    for row, conditioning in rows:
        row['distilled'] = bool(original.get('distilled', False))
        row['expansion'] = 'admitted-training-v19'
        for key in ('teacher_sha256', 'source_revision', 'original_id'):
            if key in original:
                row[key] = original[key]
    return rows


def run(a):
    import torch
    from transformers import AutoTokenizer
    base, out = Path(a.base), Path(a.output)
    bm = json.loads((base / 'manifest.json').read_text())
    if file_sha(base / 'prepared.pt') != bm['prepared_sha256']:
        raise ValueError('Base prepared checksum differs')
    additions = []
    for path in a.addition:
        folder = Path(path)
        am = json.loads((folder / 'manifest.json').read_text())
        archive = folder / 'examples.jsonl.gz'
        if (not am['admission_complete'] or am['targets_are_verified'] or
            file_sha(archive) != am['examples_archive_sha256'] or
            archived_digest(archive)[0] != am['examples_uncompressed_sha256']):
            raise ValueError('Addition has invalid identity: ' + str(folder))
        additions.append({'directory': str(folder), 'manifest_sha256': file_sha(folder / 'manifest.json'),
                          'archive_sha256': file_sha(archive)})
    data = torch.load(base / 'prepared.pt', map_location='cpu', weights_only=True)
    prepared = data['prepared']
    before = frozen_evaluation_hash(prepared)
    groups = {}
    for row, _ in prepared:
        if groups.setdefault(row['group'], row['split']) != row['split']:
            raise ValueError('Base conversation split conflict')
    known = {r['id'] for r, _ in prepared}
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True, trust_remote_code=False)
    length = data['data_manifest']['length']
    counts, base_count = Counter(), len(prepared)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'additions.json', additions)
    for entry in additions:
        for original in read_rows(Path(entry['directory']) / 'examples.jsonl.gz'):
            rows, stats = add_response(original, groups, tokenizer, length,
                                        data['prompt_tokens'], a.chunks_per_response)
            rows = preserve_provenance(rows, original)
            counts['responses_considered'] += 1
            counts['responses_selected'] += bool(rows)
            counts.update(stats)
            for row, conditioning in rows:
                if row['id'] in known:
                    raise ValueError('Duplicate example in training expansion')
                prepared.append((row, conditioning))
                known.add(row['id'])
                groups[row['group']] = 'train'
                counts['source:' + row['source']] += 1
                counts['distilled_chunks'] += row['distilled']
            if counts['responses_considered'] % 250 == 0:
                write_json(out / 'progress.json', {'counts': dict(counts), 'complete': False})
    if not counts['selected_chunks'] or frozen_evaluation_hash(prepared) != before:
        raise ValueError('No selected training chunks or altered evaluation')
    save(out / 'prepared.pt', data)
    report = {'prepared_sha256': file_sha(out / 'prepared.pt'),
        'base_prepared_sha256': bm['prepared_sha256'], 'base_rows': base_count,
        'addition_manifest_sha256': file_sha(out / 'additions.json'), 'additions': additions,
        'counts': dict(counts), 'frozen_evaluation_sha256': before, 'evaluation_changed': False,
        'context_tokens': data['prompt_tokens'], 'byte_length': length,
        'chunks_per_response': a.chunks_per_response,
        'source_tokenizer_sha256': file_sha(Path(a.tokenizer) / 'tokenizer.json'),
        'implementation_sha256': {name: file_sha(Path(__file__).parent / name) for name in
            ('prepare_training_expansion.py', 'prepare_creator_expansion.py', 'conditioning.py')},
        'scope': 'Training-only addition; teacher provenance retained; codec reconstruction admission still required',
        'training_started': False, 'quality_ready': False}
    write_json(out / 'manifest.json', report)
    write_json(out / 'progress.json', {'counts': dict(counts), 'complete': True})
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--addition', action='append', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tokenizer', default='results/tokenizer')
    parser.add_argument('--chunks-per-response', type=int, default=8)
    args = parser.parse_args()
    if args.chunks_per_response < 1:
        parser.error('Positive chunk limit required')
    run(args)
