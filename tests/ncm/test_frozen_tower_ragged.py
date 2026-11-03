import unittest
import torch
import test_frozen_tower_objective as fixtures
from scripts.ncm import frozen_tower_multi as padded
from scripts.ncm import frozen_tower_ragged as ragged
from scripts.ncm.codec import encode


class RaggedTests(unittest.TestCase):
    def test_values_gradients_and_level_isolation(self):
        f = fixtures.FrozenTowerObjectiveTests()
        f.setUp()
        model, codec = f.model.train(), f.codec
        prompts = [f.prompt, f.prompt.repeat(1, 14)[:, :69]]
        _, inputs = codec.targets(torch.tensor([encode('ab', 4), encode('xy', 4)]))
        state = padded.prepare(model.backbone, prompts)
        a = padded.forward_levels(model, inputs, prompts, state)
        b = ragged.forward_levels(model, inputs, prompts, state)
        for p, q in zip(a, b):
            for x, y in zip(p[:2], q[:2]):
                torch.testing.assert_close(x, y, rtol=1e-4, atol=2e-5)
        parameters = [p for p in model.parameters() if p.requires_grad]
        ga = torch.autograd.grad(sum(x.square().mean() for row in a for x in row[:2]), parameters)
        gb = torch.autograd.grad(sum(x.square().mean() for row in b for x in row[:2]), parameters)
        for x, y in zip(ga, gb):
            torch.testing.assert_close(x, y, rtol=.02, atol=3e-5)
        changed = [x.clone() for x in inputs]
        changed[0].add_(10)
        other = ragged.forward_levels(model, changed, prompts, state)
        for row_a, row_b in zip(b[1:], other[1:]):
            torch.testing.assert_close(row_a[0], row_b[0], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
