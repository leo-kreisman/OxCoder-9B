import unittest
import torch

import test_frozen_tower as fixtures
from scripts.ncm import frozen_tower as tower
from scripts.ncm import frozen_tower_batch as batched


class FrozenTowerBatchTests(unittest.TestCase):
    def test_batch_matches_independent_levels_and_padding_cannot_leak(self):
        fixture = fixtures.FrozenTowerTests()
        fixture.setUp()
        model = fixture.model.train()
        for length in (5, 64, 69):
            prompt = fixture.prompt.repeat(1, 14)[:, :length]
            state = tower.prepare(model, prompt)
            sizes = (1, 2, 4)
            x = torch.randn(3, 4, 16, requires_grad=True)
            independent = [tower.cached(model, x[i:i + 1, :n], state) for i, n in enumerate(sizes)]
            actual = batched.cached_many(model, x, state, sizes)
            for i, n in enumerate(sizes):
                torch.testing.assert_close(actual[i:i + 1, :n], independent[i], rtol=1e-4, atol=1e-5)
            parameters = [x, *model.adapters.parameters()]
            expected_grad = torch.autograd.grad(sum(y.square().mean() for y in independent), parameters)
            actual_grad = torch.autograd.grad(sum(actual[i:i + 1, :n].square().mean() for i, n in enumerate(sizes)), parameters)
            for a, b in zip(actual_grad, expected_grad):
                torch.testing.assert_close(a, b, rtol=.02, atol=2e-5)
            for i, n in enumerate(sizes):
                self.assertEqual(int(torch.count_nonzero(actual_grad[0][i, n:])), 0)
            changed = x.detach().clone()
            changed[0, 1:] = 1000
            another = batched.cached_many(model, changed, state, sizes)
            torch.testing.assert_close(another[0, :1], actual[0, :1], rtol=0, atol=0)
            torch.testing.assert_close(another[1:], actual[1:], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
