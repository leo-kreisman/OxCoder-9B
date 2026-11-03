import unittest
import torch
import test_frozen_tower_objective as fixtures
from scripts.ncm.codec import encode
from scripts.ncm import frozen_ragged_objective as reference
from scripts.ncm import frozen_clean_objective as clean


class CanonicalObjectiveTests(unittest.TestCase):
    def setUp(self):
        f = fixtures.FrozenTowerObjectiveTests()
        f.setUp()
        self.model, self.codec = f.model, f.codec
        prompts = [f.prompt, f.prompt.repeat(1, 14)[:, :69]]
        self.items = [({'split': 'train', 'target': text},
                       {'ids': prompt[0].tolist(), 'is_final_chunk': bool(i)})
                      for i, (text, prompt) in enumerate(zip(('ab', 'xy'), prompts))]

    def test_zero_corruption_matches_teacher_forced_loss_and_all_gradients(self):
        a = reference.backward_batch(self.model, self.codec, self.items, 64, [False, False])
        gradients = {n: p.grad.clone() for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.zero_grad(set_to_none=True)
        b = clean.backward_batch(self.model, self.codec, self.items, 64, 0.)
        self.assertAlmostEqual(sum(x['loss'] for x in a), sum(x['loss'] for x in b), places=5)
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                torch.testing.assert_close(p.grad, gradients[n], rtol=.002, atol=2e-5, msg=n)
            else:
                self.assertIsNone(p.grad, n)
        self.assertTrue(all(p.grad is None for p in self.codec.parameters()))

    def test_corruption_changes_only_conditioning_and_replays_rng(self):
        target = torch.tensor([encode('ab', 4), encode('xy', 4)])
        labels, original = clean.supervision(self.codec, target, 0.)
        state = torch.get_rng_state()
        a, x = clean.supervision(self.codec, target, 1.)
        torch.set_rng_state(state)
        b, y = clean.supervision(self.codec, target, 1.)
        for p, q, r in zip(labels, a, b):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
            torch.testing.assert_close(p, r, rtol=0, atol=0)
        for p, q in zip(x, y):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
            self.assertFalse(p.requires_grad)
        torch.testing.assert_close(x[0], original[0], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(p, q) for p, q in zip(x[1:], original[1:])))
        with self.assertRaises(ValueError):
            clean.supervision(self.codec, target, 1.1)
        with self.assertRaises(ValueError):
            clean.backward_batch(self.model, self.codec, [({'split': 'validation'}, {})], 64, .15)


if __name__ == '__main__':
    unittest.main()
