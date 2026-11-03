import unittest
from unittest.mock import patch
import torch
import test_frozen_clean_objective as fixtures
from scripts.ncm import canonical_onpolicy as reference
from scripts.ncm import decoded_hierarchy as candidate
from scripts.ncm.codec import Codec, CodecConfig, encode, resize


class HierarchyTests(unittest.TestCase):
    def test_interpolation_surrogate_matches_reference_derivative(self):
        for source, destination in ((1, 8), (3, 8), (4, 128), (16, 128), (64, 128), (128, 128)):
            x = torch.randn(2, source, 8, requires_grad=True)
            y = x.detach().clone().requires_grad_()
            upstream = torch.randn(2, destination, 8)
            reference_value = resize(x, destination)
            candidate_value = candidate.linear_surrogate(y, destination)
            torch.testing.assert_close(reference_value, candidate_value, rtol=2e-6, atol=2e-6)
            (reference_value * upstream).sum().backward()
            (candidate_value * upstream).sum().backward()
            torch.testing.assert_close(x.grad, y.grad, rtol=2e-5, atol=2e-6)

    def test_hard_forward_target_independence_and_all_level_gradients(self):
        torch.manual_seed(40001)
        codec = Codec(CodecConfig(length=8, width=8, heads=2, layers=1,
            codes=17, levels=(1, 3, 8))).eval().requires_grad_(False)
        logits = [torch.randn(2, n, 17, requires_grad=True) for n in (1, 3, 8)]
        targets = torch.tensor([encode('abc', 8), encode('xyz', 8)])
        loss, output = candidate.hierarchy_loss(codec, logits, targets, [True, False])
        z = torch.zeros(1, 8, 8, requires_grad=True)
        for values, book in zip(logits, codec.books):
            z = z + resize(book(values[:1].argmax(-1)), 8)
        torch.testing.assert_close(output, codec.decode_latents(z), rtol=0, atol=0)
        _, changed = candidate.hierarchy_loss(codec, logits, targets.flip(0), [True, False])
        torch.testing.assert_close(output, changed, rtol=0, atol=0)
        self.assertEqual(float(loss[1].detach()), 0.)
        loss.mean().backward()
        for value in logits:
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(value.grad[0].norm()), 0.)
            self.assertEqual(float(value.grad[1].norm()), 0.)
        self.assertTrue(all(p.grad is None for p in codec.parameters()))

    def test_disabled_byte_objective_and_unflagged_rows_are_exact_reference(self):
        for flags, weight in (([True, False], 0.), ([False, False], .1)):
            fixture = fixtures.CanonicalObjectiveTests(); fixture.setUp()
            model, codec, items = fixture.model, fixture.codec, fixture.items
            torch.manual_seed(40002)
            expected = reference.backward_batch(model, codec, items, 64., .15, flags)
            gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
            model.zero_grad(set_to_none=True)
            torch.manual_seed(40002)
            with patch.object(candidate, 'hierarchy_loss', side_effect=AssertionError('must bypass decoder')):
                actual = candidate.backward_batch(model, codec, items, 64., .15, flags, weight)
            self.assertEqual([r['loss'] for r in expected], [r['loss'] for r in actual])
            for name, p in model.named_parameters():
                if p.requires_grad:
                    torch.testing.assert_close(p.grad, gradients[name], rtol=0, atol=0, msg=name)

    def test_joint_objective_updates_only_trainable_parameters(self):
        fixture = fixtures.CanonicalObjectiveTests(); fixture.setUp()
        model, codec, items = fixture.model, fixture.codec, fixture.items
        rows = candidate.backward_batch(model, codec, items, 64., .15, [True, False])
        self.assertGreater(rows[0]['byte_ce'], 0)
        self.assertEqual(rows[1]['byte_ce'], 0)
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
            else:
                self.assertIsNone(p.grad, name)
        self.assertTrue(all(p.grad is None for p in codec.parameters()))
        with self.assertRaises(ValueError):
            candidate.backward_batch(model, codec, items, 64., .15, [True, False], float('nan'))


if __name__ == '__main__':
    unittest.main()
