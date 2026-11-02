import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.ncm.run_lr_comparison import compare, compare_training


class LearningRateComparisonTests(unittest.TestCase):
    def reports(self):
        identity = {k: 'same' for k in ['source_sha256', 'codec_sha256', 'prepared_sha256',
                    'head_lr', 'rank', 'head_kind', 'conditioning_corruption',
                    'effective_batch_examples', 'seed']}
        baseline = {'complete': True, 'selection_sha256': 'selection', 'implementation_sha256': 'implementation',
                    'checkpoint_sha256': 'baseline', 'identity': identity,
                    'records': [{'id': str(i), 'teacher_forced_ce': [4., 4.]} for i in range(64)],
                    'summary': {'mean_level_ce': 4., 'free_exact_windows': 0, 'free_valid_windows': 2,
                                'mean_free_nonpad_byte_accuracy': .1}}
        candidate = copy.deepcopy(baseline)
        candidate['checkpoint_sha256'] = 'candidate'
        candidate['records'] = [{'id': str(i), 'teacher_forced_ce': [3.8, 3.8]} for i in range(64)]
        candidate['summary']['mean_level_ce'] = 3.8
        scores = [{'evaluated': 12, 'tasks': [{}] * 12, 'checkpoint_sha256': name, 'passed': 0}
                  for name in ['baseline', 'candidate']]
        return baseline, candidate, *scores

    def test_selects_only_improvement_without_generation_regression(self):
        reports = self.reports()
        self.assertTrue(compare(*reports)['select_candidate'])
        reports[1]['summary']['mean_free_nonpad_byte_accuracy'] = .099
        self.assertFalse(compare(*reports)['select_candidate'])
        reports = self.reports()
        reports[1]['summary']['mean_level_ce'] = 3.999
        self.assertFalse(compare(*reports)['select_candidate'])

    def test_rejects_incomplete_mismatched_and_nonfinite_evidence(self):
        for change in ['selection', 'identity', 'row', 'score', 'nan', 'incomplete']:
            b, c, bs, cs = self.reports()
            if change == 'selection': c['selection_sha256'] = 'other'
            elif change == 'identity': c['identity']['prepared_sha256'] = 'other'
            elif change == 'row': c['records'][0]['id'] = 'other'
            elif change == 'score': cs['checkpoint_sha256'] = 'other'
            elif change == 'nan': c['records'][0]['teacher_forced_ce'][0] = float('nan')
            else: c['complete'] = False
            with self.assertRaises(ValueError): compare(b, c, bs, cs)

    def test_real_training_sequence_identity_and_coverage(self):
        rows = [{'step': i, 'loss': 4., 'examples': [{'id': str(i)}], 'allocation': [[0], [1]]}
                for i in range(351, 501)]
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a', Path(directory) / 'b'
            a.write_text('\n'.join(map(json.dumps, rows)))
            changed = [{**r, 'loss': 3.} for r in rows]
            b.write_text('\n'.join(map(json.dumps, changed)))
            self.assertTrue(compare_training(a, b)['same_examples_and_allocation'])
            changed[-1]['examples'] = [{'id': 'wrong'}]
            b.write_text('\n'.join(map(json.dumps, changed)))
            with self.assertRaises(ValueError): compare_training(a, b)
            b.write_text('\n'.join(map(json.dumps, rows[:-1])))
            with self.assertRaises(ValueError): compare_training(a, b)


if __name__ == '__main__':
    unittest.main()
