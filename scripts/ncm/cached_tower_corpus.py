"""v35 serialization-only alternative to frozen_tower_training.load_corpus.

Admission checks are deliberately identical to the pinned original reader.
Changing that module would invalidate active checkpoints, so this experimental
entry point remains separate and requires a whole-corpus comparison before use.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path

from .backbone import digest
from .selection_reader import load


def load_corpus(selection, manifest, source_sha, codec_sha, cache):
    selected = Path(selection) / 'selection.pt'
    report = json.loads(Path(manifest).read_text())
    identity = report['identity']
    if (not report.get('complete') or not identity.get('complete_training_responses_only')
            or identity['source_sha256'] != source_sha or identity['codec_sha256'] != codec_sha
            or digest(selected) != identity['selection_sha256']):
        raise ValueError('Wrong or incomplete corpus admission')
    prepared = load(cache, identity['selection_sha256'])['prepared']
    admitted, seen, groups = [], set(), {}
    for record in report['records']:
        row, condition = prepared[record['index']]
        if (row['id'] in seen or any(row[k] != record[k] for k in ('id', 'group', 'split', 'source'))
                or bool(row.get('distilled')) != bool(record['distilled'])
                or len(condition['ids']) != record['prompt_tokens']
                or not 0 < len(condition['ids']) <= 2048
                or groups.setdefault(row['group'], row['split']) != row['split']
                or (row['split'] == 'train' and not record['codec_exact'])):
            raise ValueError('Corpus identity, prompt bound, or split changed')
        seen.add(row['id'])
        admitted.append((row, condition))
    training = [item for item in admitted if item[0]['split'] == 'train']
    validation = [item for item in admitted if item[0]['split'] == 'validation']
    responses = defaultdict(list)
    for row, condition in training:
        responses[condition['response_id']].append(condition)
    for chunks in responses.values():
        chunks.sort(key=lambda c: c['chunk_index'])
        if ([c['chunk_index'] for c in chunks] != list(range(len(chunks)))
                or not chunks[-1]['is_final_chunk'] or any(c['is_final_chunk'] for c in chunks[:-1])):
            raise ValueError('Incomplete training response')
    sources = dict(Counter(row['source'] for row, _ in training))
    if (len(validation) != 64 or set(sources) != {'TeichAI/Ox-Alpha-10k',
            'TeichAI/DeepSeek-v4-Pro-Agent', 'nvidia/OpenCodeInstruct'}
            or not any(row.get('distilled') for row, _ in training)):
        raise ValueError('Missing required creator data, distillation, or held-out windows')
    return training, validation, {'admission_sha256': digest(manifest),
        'selection_sha256': identity['selection_sha256'], 'training_chunks': len(training),
        'validation_chunks': len(validation), 'complete_training_responses': len(responses),
        'sources': sources, 'distilled_chunks': sum(bool(r.get('distilled')) for r, _ in training)}
