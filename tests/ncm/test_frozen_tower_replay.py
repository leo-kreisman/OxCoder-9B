import copy
import unittest
from scripts.ncm.run_frozen_tower_stage import verify_replay, EXPECTED_TASKS, EXPECTED_CASES


class FrozenReplayTests(unittest.TestCase):
    def test_replay_requires_all_checks_identical_text_and_identity(self):
        report = {'checkpoint_sha256': 'abc', 'passed': 12, 'evaluated': 12,
            'tasks': [{'task': name, 'passed': True, 'checks': [{'passed': True}] * count,
                'generation': {'text': name, 'autoregressive_fallback': False, 'stop_reason': 'predicted_end'}}
                for name, count in zip(EXPECTED_TASKS, EXPECTED_CASES)]}
        verify_replay(report, copy.deepcopy(report), 'abc')
        for mutate in (
            lambda r: r.update(checkpoint_sha256='wrong'),
            lambda r: r['tasks'][0]['generation'].update(text='different'),
            lambda r: r['tasks'][0]['generation'].update(autoregressive_fallback=True),
            lambda r: r['tasks'][0]['checks'][0].update(passed=False),
            lambda r: r['tasks'][0].update(task='1'),
            lambda r: r['tasks'][0]['checks'].pop(),
        ):
            changed = copy.deepcopy(report)
            mutate(changed)
            with self.assertRaises(ValueError):
                verify_replay(report, changed, 'abc')


if __name__ == '__main__':
    unittest.main()
