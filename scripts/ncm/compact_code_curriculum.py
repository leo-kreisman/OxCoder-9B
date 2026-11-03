"""Prepare a tested, training-only short-program curriculum without changing active data.

Remove presentation overhead, retaining AST structure apart from docstrings.
Publisher checks must pass before and after transformation. No evaluation answer
is read for generation or selected as a training target.
"""
import argparse
import ast
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re

from .acquire_creator_corpora import archived_digest
from .admit_creator_corpora import AdmissionFilter, benchmark_hashes, read_rows, shared_instruction_inventory
from .distill_admitted import write_json
from .open_corpus import file_sha
from .verify_distillation import extract_python, decode_test, execute


class RemoveDocstrings(ast.NodeTransformer):
    def clean(self, node):
        self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:] or [ast.Pass()]
        return node

    visit_Module = clean
    visit_FunctionDef = clean
    visit_AsyncFunctionDef = clean
    visit_ClassDef = clean


def compact_python(text):
    code = extract_python(text)
    tree = ast.parse(code)
    if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for n in tree.body):
        raise ValueError('No top-level definition')
    # Tests remain necessary: static inspection cannot prove arbitrary Python
    # insensitive to formatting or docstring removal.
    forbidden = {'__doc__', 'getdoc', 'getsource', 'getsourcelines'}
    for node in ast.walk(tree):
        if ((isinstance(node, ast.Name) and node.id in forbidden)
                or (isinstance(node, ast.Attribute) and node.attr in forbidden)
                or (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and '__doc__' in node.value)):
            raise ValueError('Explicit documentation/source introspection')
    tree = ast.fix_missing_locations(RemoveDocstrings().visit(tree))
    result = ast.unparse(tree) + '\n'
    if ast.dump(ast.parse(result)) != ast.dump(tree):
        raise ValueError('AST changed beyond documented transformation')
    return result


def candidate(row, maximum_bytes):
    if row['split'] != 'train' or row['source'] != 'nvidia/OpenCodeInstruct':
        raise ValueError('Requires admitted NVIDIA training rows')
    messages = [dict(m) for m in row['messages']]
    if not messages or messages[-1]['role'] != 'user' or not isinstance(messages[-1]['content'], str):
        return None, 'unsupported_prompt'
    prompt = '\n'.join(str(m.get('content', '')) for m in messages)
    if re.search(r'\b(docstrings?|documentation|comments?|explanations?)\b', prompt, re.I):
        return None, 'presentation_requested'
    try:
        code = compact_python(row['target'])
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return None, 'unsupported_code'
    if not 1 <= len(code.encode()) <= maximum_bytes:
        return None, 'length'
    messages[-1]['content'] += '\n\nReturn only Python code. Omit comments, docstrings, and explanatory text.'
    return {**row, 'id': row['id'] + ':compact-v33', 'messages': messages,
        'prompt': json.dumps(messages, sort_keys=True, ensure_ascii=False), 'target': code,
        'targets_are_verified': False, 'admission': 'compact-code-v33',
        'transformation': {'parent_id': row['id'],
            'parent_target_sha256': hashlib.sha256(row['target'].encode()).hexdigest(),
            'method': 'AST unparse after docstring removal; comments and fences removed',
            'maximum_utf8_bytes': maximum_bytes}}, None


def run(a):
    import pyarrow.parquet as pq
    original, distilled, frozen, out = map(Path, (a.admitted, a.distilled, a.frozen, a.output))
    out.mkdir(parents=True, exist_ok=False)
    sources = []
    for folder in (original, distilled):
        manifest = json.loads((folder / 'manifest.json').read_text())
        archive = folder / 'examples.jsonl.gz'
        if (not manifest.get('admission_complete')
                or file_sha(archive) != manifest['examples_archive_sha256']
                or archived_digest(archive)[0] != manifest['examples_uncompressed_sha256']):
            raise ValueError('Changed or incomplete admitted source')
        sources.append({'directory': str(folder), 'manifest_sha256': file_sha(folder / 'manifest.json'),
                        'archive_sha256': file_sha(archive)})
    om = json.loads((original / 'manifest.json').read_text())
    fm = json.loads((frozen / 'manifest.json').read_text())
    if (file_sha(original / 'source.parquet') != om['source']['sha256']
            or file_sha(frozen / 'documents.jsonl') != fm['documents_sha256']
            or file_sha(frozen / 'manifest.json') != om['frozen_manifest_sha256']):
        raise ValueError('Publisher or frozen exclusion source changed')
    inventory = shared_instruction_inventory([frozen / 'documents.jsonl', original / 'examples.jsonl.gz',
                                              distilled / 'examples.jsonl.gz'])
    filters = AdmissionFilter(benchmark_hashes(fm['benchmark_exclusions'], original), inventory)
    for row in read_rows(frozen / 'documents.jsonl'):
        if filters.groups.setdefault(row['group'], row['split']) != row['split']:
            raise ValueError('Frozen group conflict')
        if row['split'] != 'train':
            filters.register(row, protect=True)
    selected, counts = defaultdict(list), Counter()
    for folder in (original, distilled):
        for row in read_rows(folder / 'examples.jsonl.gz'):
            item, reason = candidate(row, a.maximum_bytes)
            counts['considered'] += 1
            if reason is None:
                reason = filters.consider(item)
            if reason:
                counts['rejected:' + reason] += 1
                continue
            selected[row['original_id']].append((row, item))
    identity = {'sources': sources, 'publisher_sha256': om['source']['sha256'],
        'frozen_manifest_sha256': file_sha(frozen / 'manifest.json'), 'maximum_bytes': a.maximum_bytes,
        'implementation_sha256': {n: file_sha(Path(__file__).with_name(n)) for n in
            ('compact_code_curriculum.py', 'verify_distillation.py', 'admit_creator_corpora.py')},
        'selection': 'All admitted training responses satisfying fixed syntax, prompt, length and exclusion rules',
        'quality_ready': False, 'training_started': False}
    write_json(out / 'identity.json', identity)
    pending_count = sum(map(len, selected.values()))
    seen = set()
    with gzip.open(out / 'examples.jsonl.gz.tmp', 'wt', encoding='utf-8', compresslevel=3) as dest, \
            (out / 'checks.jsonl').open('w') as checks:
        for batch in pq.ParquetFile(original / 'source.parquet').iter_batches(
                batch_size=256, columns=['id', 'output', 'unit_tests']):
            for source in batch.to_pylist():
                if source['id'] not in selected:
                    continue
                if source['id'] in seen:
                    raise ValueError('Duplicate publisher ID')
                seen.add(source['id'])
                try:
                    tests = json.loads(source['unit_tests'])
                    if not isinstance(tests, list) or not 1 <= len(tests) <= 100:
                        raise ValueError('Invalid test inventory')
                    tests = [decode_test(t) for t in tests]
                    original_code = extract_python(source['output'])
                    controls = [execute(original_code, t) for t in tests]
                    if not all(c['passed'] and c.get('assertions_executed', 0) > 0 for c in controls):
                        raise ValueError('Publisher original fails local checks')
                except (SyntaxError, ValueError, TypeError) as error:
                    for parent, _ in selected[source['id']]:
                        counts['rejected:publisher_control'] += 1
                        checks.write(json.dumps({'parent_id': parent['id'], 'passed': False,
                                                 'reason': str(error)[:300]}) + '\n')
                    continue
                test_sha = hashlib.sha256(json.dumps(tests, ensure_ascii=False).encode()).hexdigest()
                for parent, item in selected[source['id']]:
                    parent_code = extract_python(parent['target'])
                    before = controls if parent_code == original_code else [execute(parent_code, t) for t in tests]
                    after = [execute(item['target'], t) for t in tests]
                    passed = all(c['passed'] and c.get('assertions_executed', 0) > 0 for c in before + after)
                    proof = {'parent_id': parent['id'], 'id': item['id'], 'split': 'train',
                        'publisher_id': source['id'], 'tests_sha256': test_sha, 'passed': passed,
                        'original_checks': controls, 'parent_checks': before, 'compact_checks': after}
                    checks.write(json.dumps(proof) + '\n'); checks.flush()
                    counts['admitted' if passed else 'rejected:transformation_checks'] += 1
                    if passed:
                        item['publisher_checks'] = {'tests_sha256': test_sha, 'original_answer_control_passed': True,
                            'parent_response_passed': True, 'compact_response_passed': True,
                            'scope': 'Publisher assertions only; not a correctness or semantic-equivalence proof'}
                        dest.write(json.dumps(item, ensure_ascii=False) + '\n')
                        counts['distilled_admitted'] += bool(item.get('distilled'))
                        counts['one_window_admitted'] += len(item['target'].encode()) <= 127
            write_json(out / 'progress.json', {'counts': dict(counts), 'pending_candidates': pending_count,
                                               'publisher_groups_checked': len(seen), 'complete': False})
            if len(seen) == len(selected):
                break
    if seen != set(selected) or not counts['admitted']:
        raise ValueError('Incomplete publisher join or empty curriculum')
    archive = out / 'examples.jsonl.gz'
    (out / 'examples.jsonl.gz.tmp').replace(archive)
    uncompressed, size = archived_digest(archive)
    result = {'admission_complete': True, 'targets_are_verified': False, 'training_started': False,
        'quality_ready': False, 'examples_archive_sha256': file_sha(archive),
        'examples_uncompressed_sha256': uncompressed, 'examples_uncompressed_bytes': size,
        'identity_sha256': file_sha(out / 'identity.json'), 'checks_sha256': file_sha(out / 'checks.jsonl'),
        'counts': dict(counts), 'license': om['license'], 'attribution': om['attribution'],
        'scope': 'Short-program training supplement; both creator sources remain in the base corpus',
        'limitations': 'Assertions are incomplete. Formatting and __doc__ change. No codec admission or training yet.'}
    write_json(out / 'manifest.json', result)
    write_json(out / 'progress.json', {'counts': dict(counts), 'complete': True})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('admitted', 'distilled', 'frozen', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--maximum-bytes', type=int, default=254)
    args = parser.parse_args()
    if args.maximum_bytes < 1:
        parser.error('Positive byte bound required')
    run(args)
