import copy
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from scripts.ncm.acquire_creator_corpora import archived_digest
from scripts.ncm.curate_distillation import checked_records, run
from scripts.ncm.open_corpus import file_sha


class DistillationCurationTests(unittest.TestCase):
    def fixture(self):
        evidence = {'passed': True, 'assertions_executed': 1}
        record = {'id': 'a', 'split': 'train', 'cache_sha256': 'hash', 'passed': True,
                  'reason': 'passed_publisher_checks', 'tests_sha256': 'tests',
                  'original_checks': [evidence], 'teacher_checks': [evidence]}
        return {'complete': True, 'records': [record]}, {
            'snapshot': [{'id': 'a', 'sha256': 'hash', 'cache_file': 'cache.json'}]}

    def test_requires_executed_assertions_on_both_answers(self):
        for field in ('original_checks', 'teacher_checks'):
            for checks in ([], [{'passed': False, 'assertions_executed': 1}],
                           [{'passed': True, 'assertions_executed': 0}]):
                report, identity = self.fixture()
                report['records'][0][field] = checks
                with self.assertRaises(ValueError):
                    checked_records(report, identity)

    def test_incomplete_duplicate_and_changed_snapshots_are_rejected(self):
        report, identity = self.fixture()
        self.assertEqual(set(checked_records(report, identity)), {'a'})
        for change in ('incomplete', 'duplicate', 'hash', 'holdout'):
            r = copy.deepcopy(report)
            if change == 'incomplete': r['complete'] = False
            elif change == 'duplicate': r['records'].append(copy.deepcopy(r['records'][0]))
            elif change == 'hash': r['records'][0]['cache_sha256'] = 'changed'
            else: r['records'][0]['split'] = 'validation'
            with self.assertRaises(ValueError):
                checked_records(r, identity)

    def test_archive_preserves_answer_and_rejects_cache_tampering(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, verification = root / 'teacher', root / 'verification'
            source.mkdir(); verification.mkdir()
            target = 'def example(x):\n    return x+1\n'
            row = {'id': 'a:teacher-v19', 'target': target, 'split': 'train', 'distilled': True,
                   'messages': [{'role': 'user', 'content': 'Example training prompt'}]}
            with gzip.open(source / 'examples.jsonl.gz', 'wt') as stream:
                stream.write(json.dumps(row) + '\n')
            (source / 'identity.json').write_text('{}')
            cache = {'source_id': 'a', 'response': {'choices': [
                {'finish_reason': 'stop', 'message': {'content': target}}]}}
            (source / 'cache.json').write_text(json.dumps(cache))
            archive = source / 'examples.jsonl.gz'
            manifest = {'admission_complete': True, 'examples_archive_sha256': file_sha(archive),
                        'examples_uncompressed_sha256': archived_digest(archive)[0],
                        'counts': {'accepted': 1}}
            (source / 'manifest.json').write_text(json.dumps(manifest))
            report, identity = self.fixture()
            checksum = file_sha(source / 'cache.json')
            report['records'][0]['cache_sha256'] = checksum
            identity['snapshot'][0]['sha256'] = checksum
            identity.update(teacher_identity_sha256=file_sha(source / 'identity.json'),
                            license='fixture', attribution='fixture')
            (verification / 'identity.json').write_text(json.dumps(identity))
            report['identity_sha256'] = file_sha(verification / 'identity.json')
            (verification / 'report.json').write_text(json.dumps(report))
            args = SimpleNamespace(source=str(source), verification=str(verification), output=str(root / 'out'))
            run(args)
            with gzip.open(root / 'out/examples.jsonl.gz', 'rt') as stream:
                output = json.loads(stream.readline())
            self.assertEqual(output['target'], target)
            self.assertEqual(output['messages'], row['messages'])
            self.assertEqual(output['id'], row['id'])
            self.assertFalse(json.loads((root / 'out/manifest.json').read_text())['quality_ready'])
            (source / 'cache.json').write_text('{}')
            args.output = str(root / 'tampered')
            with self.assertRaises(ValueError):
                run(args)
            self.assertFalse((root / 'tampered').exists())


if __name__ == '__main__':
    unittest.main()
