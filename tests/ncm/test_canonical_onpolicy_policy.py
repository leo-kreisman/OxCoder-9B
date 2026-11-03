import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from scripts.ncm.canonical_onpolicy_policy import validate_parent, qualify_objective
from scripts.ncm import train_frozen_clean, train_canonical_onpolicy


class CanonicalStudentPolicyTests(unittest.TestCase):
    def test_parent_changes_only_conditioning_and_preserves_gate(self):
        parent = {'architecture': 'frozen-context-original-oxcoder-native-ncm-v32',
            'corpus': {}, 'planned_steps': 10000, 'warmup': 100, 'effective_batch_examples': 4,
            'seed': 29004, 'compute_dtype': 'float16', 'loss_scale': 64., 'body_lr': 1e-4,
            'head_lr': 1e-3, 'on_policy_probability': 0., 'conditioning_corruption_maximum': .15,
            'autoregressive_answer_fallback': False, 'implementation_sha256': {}}
        args = SimpleNamespace(steps=10000, warmup=100, batch=4, seed=29004)
        validate_parent(parent, {}, args)
        for changed in ({'corpus': {'other': True}}, {'head_lr': .01}, {'on_policy_probability': .5},
                        {'autoregressive_answer_fallback': True}):
            with self.assertRaises(ValueError):
                validate_parent({**parent, **changed}, {}, args)
        self.assertIs(train_canonical_onpolicy.evaluate, train_frozen_clean.evaluate)
        self.assertIs(train_canonical_onpolicy.validate, train_frozen_clean.validate)

    def test_objective_requires_exact_control_and_both_finite_update_modes(self):
        records = [{'flags': f, 'passed': True, 'finite_parameters': True,
                    'source_unchanged': True, 'codec_has_gradient': False}
                   for f in ([True, False, True, False], [True] * 4)]
        report = {'complete': True, 'passed': True, 'disabled_student_exact': True,
            'execution': {}, 'checkpoint_sha256': 'parent', 'batch_examples': 4, 'loss_scale': 64.,
            'records': records, 'implementation_sha256': {}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'audit.json'
            path.write_text(json.dumps(report))
            qualify_objective(path, {}, 'parent')
            for changed in ({'disabled_student_exact': False}, {'error': 'failed'},
                            {'records': records[:1]}, {'checkpoint_sha256': 'other'},
                            {'records': [dict(records[0], source_unchanged=False), records[1]]}):
                path.write_text(json.dumps({**report, **changed}))
                with self.assertRaises(ValueError):
                    qualify_objective(path, {}, 'parent')


if __name__ == '__main__':
    unittest.main()
