import unittest
import torch
import test_frozen_tower_objective as fixtures
from scripts.ncm import frozen_tower as tower
from scripts.ncm import frozen_tower_multi as multi
from scripts.ncm.codec import encode


class MultiPromptTests(unittest.TestCase):
    def test_unequal_prompt_lengths_and_gradients_match_independent_reference(self):
        fixture = fixtures.FrozenTowerObjectiveTests()
        fixture.setUp()
        model, codec = fixture.model.train(), fixture.codec
        prompts = [fixture.prompt, fixture.prompt.repeat(1, 14)[:, :69]]
        _, inputs = codec.targets(torch.tensor([encode('ab', 4), encode('xy', 4)]))
        expected = [[tower.level_output(model, inp[i:i+1], prompt, level, False)
                     for level, inp in enumerate(inputs)] for i, prompt in enumerate(prompts)]
        state = multi.prepare(model.backbone, prompts)
        actual = multi.forward_levels(model, inputs, prompts, state)
        for i in range(2):
            for level in range(3):
                for a, b in zip(actual[level][:2], expected[i][level][:2]):
                    torch.testing.assert_close(a[i:i+1], b, rtol=1e-4, atol=2e-5)
        parameters = [p for p in model.parameters() if p.requires_grad]
        ga = torch.autograd.grad(sum(p.square().mean() for row in expected for out in row for p in out[:2]) / 2, parameters)
        gb = torch.autograd.grad(sum(p.square().mean() for out in actual for p in out[:2]), parameters)
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b, rtol=.02, atol=2e-5)
        changed = [p.clone() for p in prompts]
        changed[0][0, 0] = 9
        with self.assertRaises(ValueError):
            multi.forward_levels(model, inputs, changed, state)


if __name__ == '__main__':
    unittest.main()
