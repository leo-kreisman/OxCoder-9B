"""Independently execute publisher checks on an immutable teacher-cache snapshot.

Only admitted training conversations are inspected. The original published
answer must pass the same sandbox checks before its teacher answer is graded.
Passing these checks is not a proof of general correctness. No source/evaluation
archive is modified and generated programs cannot access the host workspace.
"""
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

from .distill_admitted import complete_response, write_json
from .open_corpus import file_sha


def extract_python(text):
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text, re.S | re.I)
    if len(blocks) > 1:
        raise ValueError('Multiple code blocks')
    code = blocks[0] if blocks else text.strip()
    ast.parse(code)
    return code


def decode_test(text):
    """Decode escaped file newlines only outside Python string literals.

    Some publisher rows serialize a test file with literal backslash-n between
    statements. Blind unicode_escape decoding would corrupt Unicode and escapes
    inside test strings. First accept valid Python unchanged; otherwise repair
    only outside strings/comments, and require a parseable result with asserts.
    """
    try:
        tree = ast.parse(text)
        normalized = text
    except SyntaxError:
        out, index, quote, comment = [], 0, None, False
        while index < len(text):
            char = text[index]
            if quote:
                if text.startswith(quote, index):
                    out.append(quote); index += len(quote); quote = None; continue
                if char == '\\' and index + 1 < len(text):
                    out.append(text[index:index+2]); index += 2; continue
                out.append(char); index += 1; continue
            if char == '\\' and index + 1 < len(text) and text[index+1] in 'ntr':
                value = {'n': '\n', 't': '\t', 'r': '\r'}[text[index+1]]
                out.append(value); index += 2
                if value in '\n\r':
                    comment = False
                continue
            if char in '\n\r':
                comment = False
            if char == '#' and not comment:
                comment = True
            if char in "\"'" and not comment:
                quote = char * 3 if text.startswith(char * 3, index) else char
                out.append(quote); index += len(quote); continue
            out.append(char); index += 1
        if quote:
            raise ValueError('Unterminated publisher test string')
        normalized = ''.join(out)
        tree = ast.parse(normalized)
    if not any(isinstance(node, ast.Assert) for node in ast.walk(tree)):
        raise ValueError('Publisher test has no assert statement')
    return normalized


WORKER = r'''
import ast,contextlib,io,json,random,resource,sys
resource.setrlimit(resource.RLIMIT_CPU,(2,2))
resource.setrlimit(resource.RLIMIT_AS,(256*1024*1024,256*1024*1024))
resource.setrlimit(resource.RLIMIT_FSIZE,(262144,262144))
resource.setrlimit(resource.RLIMIT_NOFILE,(32,32))
resource.setrlimit(resource.RLIMIT_NPROC,(1,1))
request=json.load(sys.stdin)
counter=[0]
def mark():counter[0]+=1
class Track(ast.NodeTransformer):
 def visit_Assert(self,node):
  return [ast.copy_location(ast.Expr(ast.Call(ast.Name('__publisher_assert_observed__',ast.Load()),[],[])),node),node]
try:
 random.seed(0)
 namespace={'__name__':'candidate'}
 with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
  exec(compile(request['code'],'<candidate>','exec'),namespace)
  tests=dict(namespace)
  tests['__publisher_assert_observed__']=mark
  program=ast.fix_missing_locations(Track().visit(ast.parse(request['test'])))
  exec(compile(program,'<publisher-test>','exec'),tests)
 response={'passed':counter[0]>0,'assertions_executed':counter[0]}
except BaseException as error:
 response={'passed':False,'exception':type(error).__name__,'assertions_executed':counter[0]}
print(json.dumps(response))
'''


def execute(code, test):
    command = ['bwrap', '--unshare-all', '--die-with-parent', '--new-session', '--clearenv',
        '--ro-bind', '/usr', '/usr', '--ro-bind', '/lib', '/lib', '--ro-bind', '/lib64', '/lib64',
        '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp', '--chdir', '/tmp',
        '/usr/bin/python3', '-I', '-c', WORKER]
    with tempfile.TemporaryFile() as output:
        try:
            result = subprocess.run(command, input=json.dumps({'code': code, 'test': test}).encode(),
                                    stdout=output, stderr=output, timeout=5)
            output.seek(0)
            value = json.loads(output.read(262145))
            if result.returncode:
                return {'passed': False, 'reason': 'worker_exit', 'exit_code': result.returncode}
            if not isinstance(value, dict) or not isinstance(value.get('passed'), bool):
                raise ValueError('Invalid worker protocol')
            return value
        except subprocess.TimeoutExpired:
            return {'passed': False, 'reason': 'timeout'}
        except (ValueError, UnicodeError):
            return {'passed': False, 'reason': 'invalid_worker_output'}


def run(a):
    import pyarrow.parquet as pq
    admitted, cache, out = Path(a.admitted), Path(a.cache), Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((admitted / 'manifest.json').read_text())
    if (file_sha(admitted / 'examples.jsonl.gz') != manifest['examples_archive_sha256']
            or file_sha(admitted / 'source.parquet') != manifest['source']['sha256']):
        raise ValueError('Publisher source/admission checksum differs')
    with gzip.open(admitted / 'examples.jsonl.gz', 'rt') as stream:
        originals = {r['id']: r for r in map(json.loads, stream)}
    progress = json.loads((cache / 'progress.json').read_text())
    # A completed progress file omits its records; use the final admission record.
    if progress['complete']:
        progress = json.loads((cache / 'manifest.json').read_text())
    accepted = {r['id'] for r in progress['records'] if r['accepted']}
    selected = {}
    for path in sorted(cache.glob('*.json')):
        if not re.fullmatch(r'[0-9a-f]{64}', path.stem):
            continue
        row = json.loads(path.read_text())
        source_id = row.get('source_id')
        if source_id not in accepted:
            continue
        original = originals[source_id]
        if original['split'] != 'train':
            raise ValueError('Nontraining row in teacher verification')
        selected[original['original_id']] = {'original': original, 'cached': row,
            'cache_sha256': file_sha(path), 'cache_file': path.name}
    identity = {'admission_manifest_sha256': file_sha(admitted / 'manifest.json'),
        'teacher_identity_sha256': file_sha(cache / 'identity.json'),
        'implementation_sha256': file_sha(__file__), 'scope': 'publisher checks on a training-only snapshot',
        'license': manifest['license'], 'attribution': manifest['attribution'],
        'snapshot': [{'id': v['original']['id'], 'cache_file': v['cache_file'], 'sha256': v['cache_sha256']}
                     for v in selected.values()], 'quality_ready': False}
    write_json(out / 'identity.json', identity)
    records, counts = [], Counter()
    for batch in pq.ParquetFile(admitted / 'source.parquet').iter_batches(
            batch_size=256, columns=['id', 'output', 'unit_tests']):
        for source in batch.to_pylist():
            if source['id'] not in selected:
                continue
            item = selected.pop(source['id'])
            record = {'id': item['original']['id'], 'split': 'train', 'cache_sha256': item['cache_sha256'],
                      'publisher_id': source['id'], 'passed': False}
            try:
                original_code = extract_python(source['output'])
                raw_tests = json.loads(source['unit_tests'])
                if not isinstance(raw_tests, list) or not 1 <= len(raw_tests) <= 100:
                    raise ValueError('Invalid publisher test inventory')
                tests = [decode_test(test) for test in raw_tests]
                record['tests_sha256'] = hashlib.sha256(json.dumps(tests, ensure_ascii=False).encode()).hexdigest()
                control = [execute(original_code, test) for test in tests]
                record['original_checks'] = control
                if not all(x['passed'] for x in control):
                    record['reason'] = 'original_answer_does_not_pass_local_publisher_checks'
                else:
                    text, reason = complete_response(item['cached']['response'])
                    if reason:
                        raise ValueError(reason)
                    candidate = extract_python(text)
                    record['teacher_checks'] = [execute(candidate, test) for test in tests]
                    record['passed'] = all(x['passed'] for x in record['teacher_checks'])
                    record['reason'] = 'passed_publisher_checks' if record['passed'] else 'teacher_failed_publisher_checks'
            except (SyntaxError, ValueError, TypeError) as error:
                record['reason'] = 'unsupported_or_invalid_code_or_tests'
                record['error'] = str(error)[:500]
            counts[record['reason']] += 1
            records.append(record)
            write_json(out / 'report.json', {'identity_sha256': file_sha(out / 'identity.json'),
                'counts': dict(counts), 'records': records, 'complete': False, 'quality_ready': False})
        if not selected:
            break
    if selected or not records:
        raise ValueError('Incomplete publisher join or empty snapshot')
    write_json(out / 'report.json', {'identity_sha256': file_sha(out / 'identity.json'),
        'counts': dict(counts), 'records': records, 'complete': True, 'quality_ready': False,
        'limitation': 'Publisher tests are incomplete and may be wrong; this is not the separate 12-task evaluation'})
    print(json.dumps({'counts': dict(counts), 'complete': True}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('admitted', 'cache', 'output'):
        parser.add_argument('--' + name, required=True)
    run(parser.parse_args())
