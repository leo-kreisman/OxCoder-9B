import gzip
import json
from pathlib import Path
import tempfile
import unittest

from scripts.ncm.screen_compact_corpus import complete_chunks, prefer_teacher, verified_rows
from scripts.ncm.acquire_creator_corpora import archived_digest
from scripts.ncm.open_corpus import file_sha


class ScreenCompactCorpusTests(unittest.TestCase):
    def test_teacher_preference_is_within_group(self):
        rows = [{'id': 'a', 'group': 'one'}, {'id': 'b', 'group': 'one', 'distilled': True},
                {'id': 'c', 'group': 'two'}]
        self.assertEqual([r['id'] for r in prefer_teacher(rows)], ['b', 'c'])

    def test_missing_response_chunk_or_ending_is_not_accepted(self):
        def rows(indices, finals):
            return [({}, {'chunk_index': i, 'is_final_chunk': f}) for i, f in zip(indices, finals)]
        self.assertTrue(complete_chunks(rows([0, 1], [False, True])))
        self.assertTrue(complete_chunks(rows([0], [True])))
        for bad in [[], rows([1], [True]), rows([0, 2], [False, True]),
                    rows([0, 1], [False, False]), rows([0, 1], [True, True])]:
            self.assertFalse(complete_chunks(bad))

    def test_execution_proof_not_just_manifest_boolean(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = {'id': 'a', 'split': 'train', 'transformation': {'parent_id': 'p'},
                   'publisher_checks': {'tests_sha256': 'tests'}}
            proof = {'id': 'a', 'parent_id': 'p', 'split': 'train', 'passed': True, 'tests_sha256': 'tests',
                **{k: [{'passed': True, 'assertions_executed': 1}]
                   for k in ('original_checks', 'parent_checks', 'compact_checks')}}
            with gzip.open(root / 'examples.jsonl.gz', 'wt') as stream:
                stream.write(json.dumps(row) + '\n')
            (root / 'identity.json').write_text('{}')
            def write_manifest():
                (root / 'checks.jsonl').write_text(json.dumps(proof) + '\n')
                manifest = {'admission_complete': True, 'training_started': False,
                    'examples_archive_sha256': file_sha(root / 'examples.jsonl.gz'),
                    'examples_uncompressed_sha256': archived_digest(root / 'examples.jsonl.gz')[0],
                    'checks_sha256': file_sha(root / 'checks.jsonl'), 'identity_sha256': file_sha(root / 'identity.json')}
                (root / 'manifest.json').write_text(json.dumps(manifest))
            write_manifest()
            self.assertEqual(verified_rows(root)[0], [row])
            proof['compact_checks'][0]['assertions_executed'] = 0
            write_manifest()
            with self.assertRaises(ValueError):
                verified_rows(root)


if __name__ == '__main__':
    unittest.main()
