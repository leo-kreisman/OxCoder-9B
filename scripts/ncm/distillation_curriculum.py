"""Recheck archived teacher responses without changing frozen evaluation rows."""
import gzip
import json
from pathlib import Path
import re

from .backbone import digest


def filter_teacher_actions(prepared, manifest):
    rejected = set()
    for entry in manifest.get('additions', []):
        archive = Path(entry['directory']) / 'examples.jsonl.gz'
        if digest(archive) != entry['archive_sha256']:
            raise ValueError('Training addition archive changed')
        with gzip.open(archive, 'rt', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if row.get('distilled') and re.search(r'<(?:tool_call(?:\s|>)|function=)', row['target']):
                    rejected.add(row['id'])
    kept, removed = [], []
    for row, condition in prepared:
        if row['split'] == 'train' and row.get('distilled') and condition['response_id'] in rejected:
            removed.append(row['id'])
        else:
            kept.append((row, condition))
    return kept, {'rejected_responses': sorted(rejected), 'rejected_chunks': removed,
                  'evaluation_filtered': False, 'reason': 'offline teacher emitted an unexecuted textual tool call'}
