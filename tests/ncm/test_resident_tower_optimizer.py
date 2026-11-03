import copy
import unittest
import torch
from torch import nn
from scripts.ncm.frozen_tower_optimizer import TowerOptimizer
from scripts.ncm.resident_tower_optimizer import ResidentTowerOptimizer, ResidentAdamW


def model():
    m = nn.Module()
    m.head = nn.Linear(4, 8)
    m.stop_head = nn.Linear(4, 1)
    m.project = nn.Linear(4, 4)
    m.position = nn.Parameter(torch.zeros(2, 4))
    m.backbone = nn.Module()
    m.backbone.adapters = nn.ModuleList([nn.Linear(4, 4, bias=False)])
    return m


def update(m, opt, value):
    opt.zero_grad()
    for p in m.parameters():
        p.grad = torch.full_like(p, value * 64)
    opt.step()


class ResidentOptimizerTests(unittest.TestCase):
    def test_legacy_import_preserves_moments_and_updates(self):
        torch.manual_seed(34001)
        left = model()
        old = TowerOptimizer(left)
        update(left, old, .01)
        right = copy.deepcopy(left)
        new = ResidentTowerOptimizer(right)
        saved = copy.deepcopy(old.state_dict())
        new.import_legacy_state(saved)
        update(left, old, -.02)
        update(right, new, -.02)
        for a, b in zip(left.parameters(), right.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(old.head.copies, new.head.original):
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(old.head.inner.state[a][key], new.head.inner.state[b][key], rtol=0, atol=0)

    def test_native_checkpoint_resume_matches_uninterrupted(self):
        torch.manual_seed(34002)
        left = model()
        old = ResidentTowerOptimizer(left)
        update(left, old, .01)
        right = copy.deepcopy(left)
        new = ResidentTowerOptimizer(right)
        new.load_state_dict(copy.deepcopy(old.state_dict()))
        update(left, old, .03)
        update(right, new, .03)
        for a, b in zip(left.parameters(), right.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_migration_requires_explicit_entrypoint(self):
        m = model()
        old, new = TowerOptimizer(m), ResidentTowerOptimizer(m)
        with self.assertRaises(ValueError):
            new.load_state_dict(old.state_dict())
        with self.assertRaises(ValueError):
            new.import_legacy_state(new.state_dict())

    def test_nonfinite_gradient_does_not_update(self):
        p = nn.Parameter(torch.ones(2))
        opt = ResidentAdamW([p], lr=.01)
        p.grad = torch.tensor([float('nan'), 0.])
        with self.assertRaises(FloatingPointError):
            opt.step()
        torch.testing.assert_close(p, torch.ones(2), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
