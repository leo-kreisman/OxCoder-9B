import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts.ncm.selective_training_policy import validate_parent, qualify, qualify_reader
from scripts.ncm.backbone import digest
from scripts.ncm import train_compact_tower, train_selective_tower


class SelectiveTrainingPolicyTests(unittest.TestCase):
    def test_runtime_gate_requires_same_parent_complete_updates_and_speed(self):
        records = [{'source_unchanged': True, 'codec_has_gradient': False}] * 4
        report = {'complete': True, 'passed': True, 'speed_qualified': True,
                  'checkpoint_sha256': 'parent', 'implementation_sha256': {},
                  'arms': {name: {'records': records} for name in ('resident_checkpointed', 'packed_alternate_recompute')},
                  'parameter_comparison': {'relative_l2': 1e-6, 'maximum_absolute': 1e-5}, 'speed_ratio': 1.2}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            path.write_text(json.dumps(report))
            qualify(path, 'parent')
            with self.assertRaises(ValueError):
                qualify(path, 'different')
            for changed in ({'complete': False}, {'speed_ratio': 1.01}, {'error': 'interrupted'},
                            {'parameter_comparison': {'relative_l2': .1, 'maximum_absolute': .01}}):
                path.write_text(json.dumps({**report, **changed}))
                with self.assertRaises(ValueError):
                    qualify(path, 'parent')

    def test_reader_gate_checks_corpus_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text('{}')
            corpus = {'identity': 'fixed'}
            report = {'complete': True, 'passed': True, 'all_types_values_ordering_identical': True,
                      'identical_sampler_draws': 1000, 'corpus': corpus,
                      'cache_manifest_sha256': digest(root / 'manifest.json'), 'implementation_sha256': {}}
            path = root / 'audit.json'
            path.write_text(json.dumps(report))
            qualify_reader(path, root, corpus)
            with self.assertRaises(ValueError):
                qualify_reader(path, root, {'identity': 'other'})
            (root / 'manifest.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                qualify_reader(path, root, corpus)

    def test_parent_learning_policy_and_evaluation_are_unchanged(self):
        parent = {'architecture': 'frozen-context-original-oxcoder-native-ncm-v34',
                  'corpus': {}, 'effective_batch_examples': 4, 'planned_steps': 10000,
                  'warmup': 100, 'seed': 29004, 'compute_dtype': 'float16', 'body_lr': 1e-4,
                  'head_lr': 1e-3, 'loss_scale': 64., 'on_policy_probability': 0.,
                  'conditioning_corruption_maximum': .15, 'compact_sampling_probability': .5,
                  'autoregressive_answer_fallback': False, 'implementation_sha256': {}}
        args = SimpleNamespace(batch=4, steps=10000, warmup=100, seed=29004)
        validate_parent(parent, {}, args)
        with self.assertRaises(ValueError):
            validate_parent({**parent, 'head_lr': .01}, {}, args)
        self.assertIs(train_selective_tower.evaluate, train_compact_tower.evaluate)
        self.assertIs(train_selective_tower.validate, train_compact_tower.validate)


if __name__ == '__main__':
    unittest.main()
