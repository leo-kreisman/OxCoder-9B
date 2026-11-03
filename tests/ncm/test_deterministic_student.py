import copy
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from scripts.ncm.backbone import digest
from scripts.ncm import deterministic_student_policy as policy
from scripts.ncm import train_deterministic_student, train_frozen_clean


class DeterministicStudentTests(unittest.TestCase):
    def control(self):
        event = {'operation': 'supervision', 'labels': ['canonical'], 'inputs': ['same'],
                 'rng_before': {'cuda': 'start'}, 'rng_after': {'cuda': 'end'}}
        return {'complete': True, 'checkpoint_sha256': 'parent', 'implementation_sha256': {},
            'deterministic_control': {**policy.POLICY, 'implementation_sha256':
                digest(Path(policy.__file__).with_name('audit_canonical_determinism.py'))},
            'records': [{'arm': arm, 'losses': [1., 2.], 'events': [copy.deepcopy(event)],
                'gradient_comparison_to_first': {'exact': True, 'relative_l2': 0.,
                    'maximum_absolute': 0., 'unequal_tensors': 0},
                'source_unchanged': True, 'codec_has_gradient': False}
                for arm in ('reference_1', 'reference_2', 'disabled_1', 'disabled_2')]}

    def test_requires_reference_repeatability_as_well_as_candidate_parity(self):
        control = self.control()
        policy.validate_repeatability(control, 'parent')
        for index in range(4):
            altered = copy.deepcopy(control)
            altered['records'][index]['gradient_comparison_to_first']['relative_l2'] = 1e-12
            with self.assertRaises(ValueError):
                policy.validate_repeatability(altered, 'parent')
        for name in ('labels', 'inputs'):
            altered = copy.deepcopy(control)
            altered['records'][2]['events'][0][name] = ['changed']
            with self.assertRaises(ValueError):
                policy.validate_repeatability(altered, 'parent')
        with self.assertRaises(ValueError):
            policy.validate_repeatability(control, 'other')
        altered = copy.deepcopy(control)
        altered['deterministic_control']['torch_deterministic_algorithms'] = False
        with self.assertRaises(ValueError):
            policy.validate_repeatability(altered, 'parent')

    def test_runtime_requires_explicit_deterministic_setup(self):
        old = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark,
               torch.backends.cudnn.deterministic)
        try:
            with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ''}):
                with self.assertRaises(ValueError):
                    policy.enable()
            with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}):
                self.assertEqual(policy.enable(), policy.POLICY)
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
                self.assertFalse(torch.backends.cudnn.benchmark)
                self.assertTrue(torch.backends.cudnn.deterministic)
        finally:
            torch.use_deterministic_algorithms(old[0])
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = old[1:]

    def test_executable_acceptance_is_unchanged(self):
        self.assertIs(train_deterministic_student.evaluate, train_frozen_clean.evaluate)
        self.assertIs(train_deterministic_student.validate, train_frozen_clean.validate)


if __name__ == '__main__':
    unittest.main()
