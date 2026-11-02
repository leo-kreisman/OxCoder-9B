import gzip
import json
from pathlib import Path
import tempfile
import unittest

from scripts.ncm.backbone import digest
from scripts.ncm.distillation_curriculum import filter_teacher_actions


class TeacherCurriculumTests(unittest.TestCase):
    def test_whole_response_is_removed_but_original_data_and_holdout_are_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            archive = Path(d) / 'examples.jsonl.gz'
            with gzip.open(archive, 'wt') as f:
                for row in [{'id': 'bad', 'distilled': True, 'target': 'prefix\n<tool_call>bash {}</tool_call>'},
                            {'id': 'original', 'distilled': False, 'target': '<tool_call>bash {}</tool_call>'}]:
                    f.write(json.dumps(row) + '\n')
            rows = [({'id': str(i), 'split': split, 'distilled': distilled}, {'response_id': response})
                    for i, (split, distilled, response) in enumerate([
                        ('train', True, 'bad'), ('train', True, 'bad'),
                        ('validation', True, 'bad'), ('train', False, 'original'), ('train', True, 'good')])]
            manifest = {'additions': [{'directory': d, 'archive_sha256': digest(archive)}]}
            kept, report = filter_teacher_actions(rows, manifest)
            self.assertEqual([r['id'] for r, _ in kept], ['2', '3', '4'])
            self.assertEqual(report['rejected_chunks'], ['0', '1'])
            self.assertFalse(report['evaluation_filtered'])
            manifest['additions'][0]['archive_sha256'] = 'wrong'
            with self.assertRaises(ValueError):
                filter_teacher_actions(rows, manifest)
