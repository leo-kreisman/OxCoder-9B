import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from scripts.ncm import decoded_hierarchy_policy as policy
from scripts.ncm.deterministic_student_policy import POLICY
from scripts.ncm.backbone import digest
from scripts.ncm import train_decoded_hierarchy, train_frozen_clean


class BytePolicyTests(unittest.TestCase):
    def test_parent_preserves_data_and_schedule(self):
        args = SimpleNamespace(steps=10000, warmup=100, batch=4, seed=29004)
        parent = dict(architecture='frozen-context-original-oxcoder-native-ncm-v38',
            corpus={'id': 'data'}, planned_steps=10000, warmup=100, effective_batch_examples=4,
            seed=29004, compute_dtype='float16', loss_scale=64., body_lr=1e-4, head_lr=1e-3,
            on_policy_probability=.5, conditioning_corruption_maximum=.15,
            deterministic_execution=POLICY, autoregressive_answer_fallback=False,
            implementation_sha256={})
        policy.validate_parent(parent, parent['corpus'], args)
        for key, value in (('architecture', 'other'), ('on_policy_probability', 0.),
                           ('planned_steps', 9999), ('autoregressive_answer_fallback', True)):
            with self.assertRaises(ValueError):
                policy.validate_parent({**parent, key: value}, parent['corpus'], args)

    def test_audit_requires_exact_control_frozen_parameters_and_flagged_loss(self):
        record = dict(passed=True, finite_parameters=True, source_unchanged=True, codec_has_gradient=False)
        report = dict(complete=True, passed=True, disabled_byte_exact=True, checkpoint_sha256='parent',
            execution={'id': 'runtime'}, deterministic_execution=POLICY,
            batch_examples=4, loss_scale=64., byte_weight=.1,
            records=[dict(record, flags=[True, False, True, False], byte_ce=[1., 0., 2., 0.]),
                     dict(record, flags=[True]*4, byte_ce=[1.]*4)],
            implementation_sha256={'decoded_hierarchy_policy.py': digest(Path(policy.__file__))})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'audit.json'
            path.write_text(json.dumps(report))
            self.assertEqual(policy.qualify_objective(path, report['execution'], 'parent'), report)
            variants = []
            for key, value in (('disabled_byte_exact', False), ('byte_weight', .2),
                               ('checkpoint_sha256', 'different'), ('complete', False)):
                variants.append({**report, key: value})
            bad = copy.deepcopy(report); bad['records'][0]['byte_ce'][1] = 1.; variants.append(bad)
            bad = copy.deepcopy(report); bad['records'][1]['codec_has_gradient'] = True; variants.append(bad)
            bad = copy.deepcopy(report); bad['records'][1]['byte_ce'][0] = float('nan'); variants.append(bad)
            for bad in variants:
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    policy.qualify_objective(path, report['execution'], 'parent')

    def test_training_imports_unchanged_acceptance_and_validation(self):
        self.assertIs(train_decoded_hierarchy.evaluate, train_frozen_clean.evaluate)
        self.assertIs(train_decoded_hierarchy.validate, train_frozen_clean.validate)


if __name__ == '__main__':
    unittest.main()
