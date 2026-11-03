import copy
import unittest
import torch
from torch import nn
from scripts.ncm.frozen_tower_optimizer import TowerOptimizer


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.adapters = nn.ModuleList([nn.Linear(4, 4, bias=False).half()])
        self.project = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)
        self.stop_head = nn.Linear(4, 1)

    def forward(self, x):
        h = self.backbone.adapters[0](self.project(x).half()).float()
        return self.head(h).square().sum() + self.stop_head(h).square().sum()


class TowerOptimizerTests(unittest.TestCase):
    def test_scaled_update_and_exact_resume(self):
        torch.manual_seed(29)
        model = Toy()
        opt = TowerOptimizer(model)
        x = torch.randn(2, 4)
        opt.zero_grad()
        (model(x) * opt.loss_scale).backward()
        self.assertGreater(opt.step(), 0)
        state, weights = copy.deepcopy(opt.state_dict()), copy.deepcopy(model.state_dict())
        resumed = Toy()
        resumed.load_state_dict(weights)
        other = TowerOptimizer(resumed)
        other.load_state_dict(state)
        for m, o in ((model, opt), (resumed, other)):
            o.zero_grad()
            (m(x) * o.loss_scale).backward()
            o.step()
        for p, q in zip(model.parameters(), resumed.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        self.assertTrue(all(p.dtype == torch.float32 for p in opt.matrix.masters))
        opt.learning_rates(2e-5, 3e-4)
        self.assertEqual(opt.matrix.inner.param_groups[0]['lr'], 2e-5)
        self.assertEqual(opt.head.inner.param_groups[0]['lr'], 3e-4)

    def test_nonfinite_gradient_cannot_partially_update(self):
        model = Toy()
        opt = TowerOptimizer(model)
        (model(torch.randn(2, 4)) * opt.loss_scale).backward()
        model.head.weight.grad.fill_(float('inf'))
        old = copy.deepcopy(model.state_dict())
        with self.assertRaises(FloatingPointError):
            opt.step()
        for n, p in model.state_dict().items():
            torch.testing.assert_close(p, old[n], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
