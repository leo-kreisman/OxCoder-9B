import unittest

import torch

import test_frozen_clean_objective as fixtures
from scripts.ncm import frozen_clean_objective as reference
from scripts.ncm import canonical_onpolicy as candidate
from scripts.ncm import frozen_tower_multi as prefix
from scripts.ncm.codec import encode


class CanonicalOnPolicyTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.CanonicalObjectiveTests()
        fixture.setUp()
        self.model, self.codec, self.items = fixture.model, fixture.codec, fixture.items

    def test_disabled_student_conditioning_matches_original_gradients(self):
        torch.manual_seed(36001)
        expected = reference.backward_batch(self.model, self.codec, self.items, 64, .15)
        gradients = {n: p.grad.clone() for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.zero_grad(set_to_none=True)
        torch.manual_seed(36001)
        actual = candidate.backward_batch(self.model, self.codec, self.items, 64, .15, [False, False])
        self.assertEqual([r['loss'] for r in actual], [r['loss'] for r in expected])
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0, msg=name)

    def test_student_inputs_do_not_depend_on_targets_but_labels_remain_canonical(self):
        prompts = [torch.tensor([c['ids']]) for _, c in self.items]
        self.model.eval()
        state = prefix.prepare(self.model.backbone, prompts)
        first = torch.tensor([encode('ab', 4), encode('xy', 4)])
        second = torch.tensor([encode('cd', 4), encode('mn', 4)])
        labels_a, inputs_a = candidate.conditioning(self.model, self.codec, prompts, first, [True, True], .15, state)
        labels_b, inputs_b = candidate.conditioning(self.model, self.codec, prompts, second, [True, True], .15, state)
        expected_a, _ = reference.supervision(self.codec, first, 0.)
        expected_b, _ = reference.supervision(self.codec, second, 0.)
        for a, b, expected in zip(inputs_a, inputs_b, expected_a):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertFalse(a.requires_grad)
        for actual, expected in zip(labels_a + labels_b, expected_a + expected_b):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_student_conditioning_produces_finite_gradients_and_keeps_source_frozen(self):
        records = candidate.backward_batch(self.model, self.codec, self.items, 64, .15, [True, False])
        self.assertEqual([r['student_conditioned'] for r in records], [True, False])
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            else:
                self.assertIsNone(parameter.grad, name)
        self.assertTrue(all(p.grad is None for p in self.codec.parameters()))


if __name__ == '__main__':
    unittest.main()
