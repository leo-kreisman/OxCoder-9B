import unittest
from unittest.mock import patch

import torch
from torch import nn
import test_frozen_tower_objective as fixtures
from scripts.ncm import frozen_tower_multi as reference
from scripts.ncm import interleaved_prefix as candidate
from scripts.ncm.gguf_training_weights import FrozenGGUFLinear


class InterleavedPrefixTests(unittest.TestCase):
    def test_independent_attention_and_recurrent_states_match(self):
        fixture = fixtures.FrozenTowerObjectiveTests()
        fixture.setUp()
        backbone = fixture.model.eval().backbone
        prompts = [fixture.prompt, fixture.prompt.repeat(1, 14)[:, :69]]
        expected = reference.prepare(backbone, prompts)
        actual = candidate.prepare(backbone, prompts)
        self.assertEqual(actual.lengths, expected.lengths)
        self.assertEqual(actual.versions, expected.versions)
        for left, right in zip(actual.cache.layers, expected.cache.layers):
            for name in ('keys', 'values', 'conv_states', 'recurrent_states'):
                if hasattr(left, name):
                    a, b = getattr(left, name), getattr(right, name)
                    if isinstance(a, dict):
                        self.assertEqual(a.keys(), b.keys())
                        for k in a:
                            torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)
                    else:
                        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_unpack_reuse_and_exception_restoration(self):
        module = FrozenGGUFLinear(torch.zeros(144, dtype=torch.uint8), (1, 256), 12, '')
        layer = nn.Sequential(module)
        statistics = {}
        calls = []

        def unpack(packed, shape, kind, name, dtype):
            calls.append(dtype)
            return torch.ones(shape, dtype=dtype)

        with patch.object(candidate, 'unpack', unpack), torch.no_grad():
            with self.assertRaisesRegex(ValueError, 'intentional'):
                with candidate.layer_weights(layer, statistics):
                    for _ in range(4):
                        torch.testing.assert_close(layer(torch.ones(2, 256)), torch.full((2, 1), 256.))
                    with self.assertRaises(RuntimeError):
                        with candidate.layer_weights(layer):
                            pass
                    raise ValueError('intentional')
        self.assertEqual(calls, [torch.float32])
        self.assertEqual(statistics, {'linear_calls': 4, 'unpacks': 1})
        self.assertNotIn('forward', module.__dict__)
        self.assertNotIn('_ncm_unpack_scope', module.__dict__)

    def test_gradient_path_is_rejected(self):
        with self.assertRaises(RuntimeError):
            with candidate.layer_weights(nn.Identity()):
                pass

    def test_empty_prompt_is_rejected(self):
        with self.assertRaises(ValueError):
            candidate.prepare(None, [])


if __name__ == '__main__':
    unittest.main()
