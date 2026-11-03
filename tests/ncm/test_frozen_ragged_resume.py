import copy
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from test_frozen_tower_optimizer import Toy
from scripts.ncm.frozen_tower_optimizer import TowerOptimizer
from scripts.ncm.frozen_ragged_resume import import_parent_state


class RaggedResumeTests(unittest.TestCase):
    def test_import_preserves_next_update_and_random_draws(self):
        torch.manual_seed(300)
        model, rng = Toy(), random.Random(20)
        opt = TowerOptimizer(model)
        x = torch.randn(2, 4)
        (model(x) * opt.loss_scale).backward()
        opt.step()
        args = SimpleNamespace(steps=10000, warmup=100, batch=4, seed=29004)
        corpus = {'training_chunks': 45157}
        parent = {'architecture': 'frozen-context-original-oxcoder-native-ncm-v30',
                  'corpus': corpus, 'planned_steps': args.steps, 'warmup': args.warmup,
                  'effective_batch_examples': args.batch, 'seed': args.seed,
                  'compute_dtype': 'float16', 'loss_scale': opt.loss_scale,
                  'body_lr': 1e-4, 'head_lr': 1e-3, 'on_policy_probability': .5,
                  'autoregressive_answer_fallback': False, 'implementation_sha256': {}}
        state = {'identity': parent, 'step': 100,
                 'model': {n: p.detach().clone() for n, p in model.named_parameters()},
                 'optimizer': copy.deepcopy(opt.state_dict()),
                 'rng': {'python': rng.getstate(), 'torch': torch.get_rng_state()}}
        next_python, next_torch = rng.random(), torch.rand(4)
        resumed, other_rng = Toy(), random.Random(99)
        other = TowerOptimizer(resumed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.pt'
            torch.save(state, path)
            step = import_parent_state(path, parent, resumed, other, other_rng, 'cpu', corpus, args)
            self.assertEqual(step, 100)
            self.assertEqual(next_python, other_rng.random())
            torch.testing.assert_close(next_torch, torch.rand(4), rtol=0, atol=0)
            for m, o in ((model, opt), (resumed, other)):
                o.zero_grad()
                (m(x) * o.loss_scale).backward()
                o.step()
            for p, q in zip(model.parameters(), resumed.parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
            args.steps += 1
            with self.assertRaises(ValueError):
                import_parent_state(path, parent, resumed, other, other_rng, 'cpu', corpus, args)


if __name__ == '__main__':
    unittest.main()
