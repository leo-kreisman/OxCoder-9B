"""Materialize only unchanged, publisher-tested offline teacher responses.

This is a training-data filter, not a model quality gate. The publisher's
original answer and the distilled answer must both pass the checks. No text,
prompt, split, response ID, or test is repaired to obtain a passing result.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from .acquire_creator_corpora import archived_digest
from .distill_admitted import complete_response, write_json
from .open_corpus import file_sha


def checked_records(report, identity):
    if report.get('complete') is not True:
        raise ValueError('Verification is incomplete')
    inventory = {x['id']: x for x in identity['snapshot']}
    if len(inventory) != len(identity['snapshot']):
        raise ValueError('Duplicate snapshot ID')
    records = {x['id']: x for x in report['records']}
    if len(records) != len(report['records']) or set(records) != set(inventory):
        raise ValueError('Verification does not cover exactly its snapshot')
    accepted = {}
    for key, row in records.items():
        if row['split'] != 'train' or row['cache_sha256'] != inventory[key]['sha256']:
            raise ValueError('Verification identity differs')
        if row.get('passed') is not True:
            continue
        if row['reason'] != 'passed_publisher_checks':
            raise ValueError('Passing record has no publisher check evidence')
        original, teacher = row.get('original_checks', []), row.get('teacher_checks', [])
        if not original or len(original) != len(teacher):
            raise ValueError('Missing original-answer control or teacher checks')
        if not all(x.get('passed') is True and x.get('assertions_executed', 0) > 0
                   for x in original + teacher):
            raise ValueError('Passing record includes failed or unexecuted assertions')
        accepted[key] = (row, inventory[key])
    if not accepted:
        raise ValueError('No publisher-tested teacher responses')
    return accepted


def run(a):
    source, verification, out = Path(a.source), Path(a.verification), Path(a.output)
    manifest = json.loads((source / 'manifest.json').read_text())
    identity = json.loads((verification / 'identity.json').read_text())
    report = json.loads((verification / 'report.json').read_text())
    if (not manifest['admission_complete'] or
            file_sha(source / 'examples.jsonl.gz') != manifest['examples_archive_sha256'] or
            archived_digest(source / 'examples.jsonl.gz')[0] != manifest['examples_uncompressed_sha256']):
        raise ValueError('Source archive is incomplete or changed')
    if (file_sha(verification / 'identity.json') != report['identity_sha256'] or
            file_sha(source / 'identity.json') != identity['teacher_identity_sha256']):
        raise ValueError('Verification or teacher identity changed')
    accepted = checked_records(report, identity)
    kept, seen = [], set()
    with gzip.open(source / 'examples.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for row in map(json.loads, stream):
            suffix = ':teacher-v19'
            if not row['id'].endswith(suffix):
                raise ValueError('Unknown distilled response ID schema')
            key = row['id'][:-len(suffix)]
            if key not in accepted:
                continue
            if key in seen or row['split'] != 'train' or not row.get('distilled'):
                raise ValueError('Duplicate or nontraining/nonteacher response')
            proof, snapshot = accepted[key]
            path = source / snapshot['cache_file']
            if path.name != snapshot['cache_file'] or file_sha(path) != snapshot['sha256']:
                raise ValueError('Teacher cache changed or path is not a filename')
            cached = json.loads(path.read_text())
            text, rejected = complete_response(cached['response'])
            if cached['source_id'] != key or rejected is not None or text != row['target']:
                raise ValueError('Archived answer differs from the checked teacher answer')
            kept.append({**row, 'publisher_checks': {
                'report_sha256': file_sha(verification / 'report.json'),
                'cache_sha256': proof['cache_sha256'], 'tests_sha256': proof['tests_sha256'],
                'original_answer_control_passed': True, 'teacher_answer_passed': True,
                'scope': 'publisher tests only; not a correctness proof'}})
            seen.add(key)
    if seen != set(accepted):
        raise ValueError('Checked answers missing from completed teacher archive')
    out.mkdir(parents=True, exist_ok=False)
    archive = out / 'examples.jsonl.gz'
    with gzip.open(archive, 'wt', encoding='utf-8', compresslevel=3) as stream:
        for row in kept:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    uncompressed, size = archived_digest(archive)
    result = {'admission_complete': True, 'targets_are_verified': False,
        'quality_ready': False, 'used_in_training': False,
        'examples_archive_sha256': file_sha(archive),
        'examples_uncompressed_sha256': uncompressed, 'examples_uncompressed_bytes': size,
        'source_manifest_sha256': file_sha(source / 'manifest.json'),
        'verification_identity_sha256': file_sha(verification / 'identity.json'),
        'verification_report_sha256': file_sha(verification / 'report.json'),
        'implementation_sha256': file_sha(__file__),
        'license': identity['license'], 'attribution': identity['attribution'],
        'counts': {'source_admitted': manifest['counts']['accepted'],
                   'checked_snapshot': len(report['records']), 'selected': len(kept)},
        'verification_counts': dict(Counter(x['reason'] for x in report['records'])),
        'scope': 'Training-only unchanged responses passing local publisher tests and original-answer controls',
        'limitations': 'Tests are incomplete and can be wrong; this is not the separate 12-task model evaluation'}
    write_json(out / 'manifest.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'verification', 'output'):
        parser.add_argument('--' + name, required=True)
    run(parser.parse_args())
