"""Prepare a training addition while retaining inherited teacher filter inputs.

The standard preparer appends unchanged rows and preserves frozen evaluation.
This wrapper also carries the base manifest's known source archives forward,
so the trainer still excludes previously rejected teacher action responses.
"""
import argparse
import json
from pathlib import Path

from .open_corpus import file_sha
from .prepare_training_expansion import run as prepare
from .train import write_json


def combined_archives(inherited, added):
    result, seen = [], {}
    for entry in [*inherited, *added]:
        path = str(Path(entry['directory']).resolve())
        if path in seen:
            if seen[path] != entry:
                raise ValueError('Conflicting source archive declarations')
            continue
        archive = Path(path) / 'examples.jsonl.gz'
        if file_sha(archive) != entry['archive_sha256']:
            raise ValueError('Inherited or added source archive changed')
        if file_sha(Path(path) / 'manifest.json') != entry['manifest_sha256']:
            raise ValueError('Inherited or added source manifest changed')
        seen[path] = entry
        result.append(entry)
    return result


def run(a):
    base = Path(a.base)
    base_sha = file_sha(base / 'manifest.json')
    previous = json.loads((base / 'manifest.json').read_text())
    # Validate inherited declarations before running the more expensive prepare.
    combined_archives(previous.get('additions', []), [])
    prepare(a)
    if file_sha(base / 'manifest.json') != base_sha:
        raise ValueError('Base manifest changed during preparation')
    out = Path(a.output)
    current = json.loads((out / 'manifest.json').read_text())
    current['new_additions'] = current['additions']
    current['additions'] = combined_archives(previous.get('additions', []), current['new_additions'])
    current['inherited_archive_manifest_sha256'] = base_sha
    current['archive_lineage_scope'] = 'All archive declarations carried by the immediate base, plus new additions'
    current['implementation_sha256']['prepare_curated_expansion.py'] = file_sha(__file__)
    write_json(out / 'additions.json', current['additions'])
    current['addition_manifest_sha256'] = file_sha(out / 'additions.json')
    write_json(out / 'manifest.json', current)
    print(json.dumps({'prepared_sha256': current['prepared_sha256'],
                      'inherited_archives': len(previous.get('additions', [])),
                      'new_archives': len(current['new_additions']), 'complete': True}), flush=True)


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
