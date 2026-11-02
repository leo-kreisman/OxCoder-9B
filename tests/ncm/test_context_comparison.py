import copy
import unittest

from scripts.ncm.monitor_context_comparison import valid_report


class ComparisonAcceptanceTests(unittest.TestCase):
    def test_requires_complete_checkpoint_bound_twelve_and_no_ar_fallback(self):
        report = {'complete': True, 'evaluated': 12, 'checkpoint_sha256': 'abc',
            'tasks': [{'passed': True, 'generation': {'autoregressive_fallback': False}} for _ in range(12)],
            'passed': 12, 'passed_all': True}
        self.assertTrue(valid_report(report, 'abc'))
        self.assertFalse(valid_report(report, 'other'))
        for key, value in [('complete', False), ('evaluated', 11), ('passed', 11), ('passed_all', False)]:
            changed = copy.deepcopy(report); changed[key] = value
            self.assertFalse(valid_report(changed, 'abc'))
        changed = copy.deepcopy(report); changed['tasks'][0]['generation']['autoregressive_fallback'] = True
        self.assertFalse(valid_report(changed, 'abc'))
        changed = copy.deepcopy(report); changed['tasks'].pop()
        self.assertFalse(valid_report(changed, 'abc'))

    def test_failure_is_valid_evidence_but_never_acceptance(self):
        report = {'complete': True, 'evaluated': 12, 'checkpoint_sha256': 'abc',
            'tasks': [{'passed': False, 'generation': {'autoregressive_fallback': False}} for _ in range(12)],
            'passed': 0, 'passed_all': False}
        self.assertTrue(valid_report(report, 'abc'))
        self.assertFalse(report['passed_all'])


if __name__ == '__main__': unittest.main()
