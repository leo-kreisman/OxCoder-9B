import unittest
import torch

from scripts.ncm.train_accumulated import example_loss


class AccumulationTests(unittest.TestCase):
    def test_accumulated_gradient_matches_equal_example_batch(self):
        torch.manual_seed(8)
        weight = torch.nn.Parameter(torch.randn(5, 11, dtype=torch.float64))
        stop = torch.nn.Parameter(torch.randn(5, 1, dtype=torch.float64))
        hidden = [torch.randn(3, length, 5, dtype=torch.float64) for length in (4, 16)]
        targets = [torch.randint(11, (3, length)) for length in (4, 16)]
        finals = [False, True, False]
        for example in range(3):
            for level in range(2):
                h = hidden[level][example:example+1]
                example_loss(h @ weight, targets[level][example:example+1],
                             (h.mean(1) @ stop).squeeze(-1), finals[example],
                             level, 2, 3).backward()
        got = [weight.grad.clone(), stop.grad.clone()]
        weight.grad = None
        stop.grad = None
        reference = 0
        for level in range(2):
            h = hidden[level]
            reference = reference + torch.nn.functional.cross_entropy(
                (h @ weight).flatten(0, 1), targets[level].flatten()) / 2
            if level == 1:
                reference = reference + .1 * torch.nn.functional.binary_cross_entropy_with_logits(
                    (h.mean(1) @ stop).squeeze(-1), torch.tensor(finals, dtype=torch.float64))
        reference.backward()
        self.assertTrue(torch.allclose(got[0], weight.grad, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(got[1], stop.grad, atol=1e-12, rtol=1e-12))

    def test_invalid_normalization_rejected(self):
        with self.assertRaises(ValueError):
            example_loss(None, None, None, False, 0, 4, 0)


if __name__ == '__main__':
    unittest.main()
