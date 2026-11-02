import unittest

from scripts.ncm.acquire_opencode import candidate, DATASET, REVISION
from scripts.ncm.prepare_training_expansion import preserve_provenance


class InstructionExpansionTests(unittest.TestCase):
    def example(self):
        return {'id': '123', 'input': 'Create a constant greeting function.',
                'output': '```python\ndef greeting():\n    return "Hello there!"\n```',
                'average_test_score': '1', 'tests_execution_status': '["pass", "pass"]'}

    def test_publisher_signal_is_not_independent_verification(self):
        row, reason = candidate(self.example(), 8192)
        self.assertIsNone(reason)
        self.assertEqual(row['source'], DATASET)
        self.assertEqual(row['source_revision'], REVISION)
        self.assertEqual(row['split'], 'train')
        self.assertFalse(row['targets_are_verified'])
        self.assertEqual(row['publisher_test_count'], 2)

    def test_rejects_failed_tests_and_syntax(self):
        for changes, wanted in [({'average_test_score': '.9'}, 'publisher_test_failures'),
                                ({'tests_execution_status': '["fail"]'}, 'publisher_test_failures'),
                                ({'tests_execution_status': 'bad'}, 'invalid_publisher_verification'),
                                ({'output': '```python\ndef greeting(:\n    return "Hello there!"\n```'}, 'not_parseable_python')]:
            row, reason = candidate({**self.example(), **changes}, 8192)
            self.assertIsNone(row)
            self.assertEqual(reason, wanted)

    def test_teacher_flag_and_hash_survive_chunk_preparation(self):
        rows = [({'id': 'x:byte-0', 'distilled': False}, {'ids': [1]})]
        preserve_provenance(rows, {'distilled': True, 'teacher_sha256': 'abc'})
        self.assertTrue(rows[0][0]['distilled'])
        self.assertEqual(rows[0][0]['teacher_sha256'], 'abc')
        self.assertEqual(rows[0][1], {'ids': [1]})


if __name__ == '__main__':
    unittest.main()
