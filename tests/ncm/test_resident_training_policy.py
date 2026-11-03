import copy
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import torch
from scripts.ncm.resident_training_policy import validate_parent, restore
from scripts.ncm.resident_tower_optimizer import ResidentTowerOptimizer
from scripts.ncm.frozen_tower_optimizer import TowerOptimizer
from scripts.ncm.interleaved_context import installed
from scripts.ncm import frozen_tower_multi, interleaved_prefix
from scripts.ncm import train_compact_tower, train_resident_tower
from test_resident_tower_optimizer import model, update


class ResidentTrainingPolicyTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(batch=4, steps=10000, warmup=100, seed=29004)
        self.parent = {'architecture': 'frozen-context-original-oxcoder-native-ncm-v33',
                       'corpus': {'identity': 'test'}, 'effective_batch_examples': 4,
                       'planned_steps': 10000, 'warmup': 100, 'seed': 29004,
                       'compute_dtype': 'float16', 'body_lr': 1e-4, 'head_lr': 1e-3,
                       'loss_scale': 64., 'on_policy_probability': 0.,
                       'conditioning_corruption_maximum': .15, 'compact_sampling_probability': .5,
                       'autoregressive_answer_fallback': False, 'implementation_sha256': {}}

    def test_runtime_continuation_rejects_batch_or_sampling_change(self):
        validate_parent(self.parent, self.parent['corpus'], self.args)
        for key, value in [('effective_batch_examples', 8), ('compact_sampling_probability', .7),
                           ('conditioning_corruption_maximum', 0.), ('autoregressive_answer_fallback', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_parent({**self.parent, key: value}, self.parent['corpus'], self.args)

    def test_restore_preserves_model_optimizer_and_rng(self):
        torch.manual_seed(34007)
        original = model()
        old_optimizer = TowerOptimizer(original)
        update(original, old_optimizer, .01)
        data_rng = random.Random(73)
        for _ in range(5):
            data_rng.random()
        state = {'identity': self.parent, 'step': 2000,
                 'model': {n: p.detach().clone() for n, p in original.named_parameters()},
                 'optimizer': copy.deepcopy(old_optimizer.state_dict()),
                 'rng': {'python': data_rng.getstate(), 'torch': torch.get_rng_state()}}
        expected_random = [data_rng.random() for _ in range(4)]
        expected_torch = torch.rand(4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.pt'
            torch.save(state, path)
            restored = model()
            new_optimizer = ResidentTowerOptimizer(restored)
            restored_rng = random.Random(2)
            step = restore(path, self.parent, restored, new_optimizer, restored_rng, 'cpu', legacy=True)
            self.assertEqual(step, 2000)
            self.assertEqual([restored_rng.random() for _ in range(4)], expected_random)
            torch.testing.assert_close(torch.rand(4), expected_torch, rtol=0, atol=0)
            update(original, old_optimizer, -.03)
            update(restored, new_optimizer, -.03)
            for left, right in zip(original.parameters(), restored.parameters()):
                torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_prefix_override_restores_after_exception_and_rejects_nesting(self):
        before = frozen_tower_multi.prepare
        with self.assertRaisesRegex(ValueError, 'intentional'):
            with installed():
                self.assertIs(frozen_tower_multi.prepare, interleaved_prefix.prepare)
                with self.assertRaises(RuntimeError):
                    with installed():
                        pass
                raise ValueError('intentional')
        self.assertIs(frozen_tower_multi.prepare, before)

    def test_quality_gate_is_the_existing_function(self):
        self.assertIs(train_resident_tower.evaluate, train_compact_tower.evaluate)
        self.assertIs(train_resident_tower.validate, train_compact_tower.validate)


if __name__ == '__main__':
    unittest.main()
