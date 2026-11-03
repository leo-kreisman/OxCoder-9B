import unittest
import torch
import test_frozen_tower_objective as fixtures
from scripts.ncm import frozen_multi_objective as reference
from scripts.ncm import frozen_ragged_objective as unpadded


class RaggedObjectiveTests(unittest.TestCase):
    def test_mixed_conditioning_loss_gradients_and_frozen_parameters(self):
        f = fixtures.FrozenTowerObjectiveTests()
        f.setUp()
        model, codec = f.model, f.codec
        prompts = [f.prompt, f.prompt.repeat(1, 14)[:, :69]]
        items = [({'split': 'train', 'target': text},
                  {'ids': prompt[0].tolist(), 'is_final_chunk': bool(i)})
                 for i, (text, prompt) in enumerate(zip(('ab', 'xy'), prompts))]
        a = reference.backward_batch(model, codec, items, 64, [False, True])
        gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
        model.zero_grad(set_to_none=True)
        b = unpadded.backward_batch(model, codec, items, 64, [False, True])
        self.assertAlmostEqual(sum(x['loss'] for x in a), sum(x['loss'] for x in b), places=4)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(parameter.grad, gradients[name], rtol=.02, atol=1e-4, msg=name)
            else:
                self.assertIsNone(parameter.grad, name)
        self.assertTrue(all(p.grad is None for p in codec.parameters()))
        self.assertIs(unpadded.trajectory, reference.trajectory)
        with self.assertRaises(ValueError):
            unpadded.backward_batch(model, codec, [({'split': 'validation'}, {})], 64, [True])


if __name__ == '__main__':
    unittest.main()
