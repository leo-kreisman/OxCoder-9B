import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from scripts.ncm import packed_gradient as candidate
from scripts.ncm import frozen_tower_ragged as engine
from scripts.ncm.gguf_training_weights import FrozenGGUFLinear


class PackedGradientTests(unittest.TestCase):
    def test_forward_and_input_gradient_match_dense_with_only_packed_saved(self):
        torch.manual_seed(35001)
        weight = torch.randn(32, 256)
        packed = torch.zeros(32 * 144, dtype=torch.uint8)
        source = torch.randn(2, 4, 256)
        gradient = torch.randn(2, 4, 32)
        reference = source.clone().requires_grad_()
        expected = F.linear(reference, weight)
        expected.backward(gradient)
        saved = []
        x = source.clone().requires_grad_()
        with patch.object(candidate, 'unpack', return_value=weight) as unpack, \
             torch.autograd.graph.saved_tensors_hooks(lambda t: (saved.append(t), t)[1], lambda t: t):
            output = candidate.PackedInputGradient.apply(x, packed, (32, 256), 12, '')
            output.backward(gradient)
            self.assertEqual(unpack.call_count, 2)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
        self.assertEqual(len(saved), 1)
        self.assertIs(saved[0], packed)

    def test_scope_restores_instance_and_checkpoint_on_error(self):
        module = FrozenGGUFLinear(torch.zeros(144, dtype=torch.uint8), (1, 256), 12, '')
        backbone = nn.Sequential(module)
        previous = engine.checkpoint
        with self.assertRaisesRegex(ValueError, 'intentional'):
            with candidate.installed(backbone):
                self.assertIsNot(engine.checkpoint, previous)
                with self.assertRaises(RuntimeError):
                    with candidate.installed(backbone):
                        pass
                with patch.object(candidate, 'unpack', return_value=torch.ones(1, 256)):
                    x = torch.ones(2, 256, requires_grad=True)
                    backbone(x).sum().backward()
                    torch.testing.assert_close(x.grad, torch.ones_like(x))
                raise ValueError('intentional')
        self.assertIs(engine.checkpoint, previous)
        self.assertNotIn('forward', module.__dict__)
        self.assertFalse(hasattr(engine, '_packed_gradient_scope'))

    def test_in_place_weight_change_is_detected(self):
        packed = torch.zeros(144, dtype=torch.uint8)
        with patch.object(candidate, 'unpack', return_value=torch.ones(1, 256)):
            result = candidate.PackedInputGradient.apply(torch.ones(1, 256, requires_grad=True),
                                                         packed, (1, 256), 12, '')
            packed.add_(1)
            with self.assertRaisesRegex(RuntimeError, 'modified by an inplace'):
                result.sum().backward()

    def test_disabling_layer_recompute_preserves_full_objective_gradients(self):
        import test_frozen_clean_objective as fixtures
        from scripts.ncm.frozen_clean_objective import backward_batch
        fixture = fixtures.CanonicalObjectiveTests()
        fixture.setUp()
        model, codec, items = fixture.model, fixture.codec, fixture.items
        torch.manual_seed(35003)
        reference = backward_batch(model, codec, items, 64, .15)
        gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
        model.zero_grad(set_to_none=True)
        torch.manual_seed(35003)
        with candidate.installed(model.backbone):
            actual = backward_batch(model, codec, items, 64, .15)
        self.assertEqual(reference, actual)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0, msg=name)
            else:
                self.assertIsNone(parameter.grad)
        self.assertTrue(all(p.grad is None for p in codec.parameters()))


if __name__ == '__main__':
    unittest.main()
