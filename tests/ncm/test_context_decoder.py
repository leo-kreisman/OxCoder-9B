import unittest

import torch

from scripts.ncm.context_decoder import ContextConceptDecoder, DecoderConfig


class ContextDecoderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(25)
        self.model = ContextConceptDecoder(DecoderConfig(source_width=16, latent_width=8,
            width=32, heads=4, layers=2, codes=17, levels=(1, 3, 8), maximum_prompt=16)).eval()
        self.inputs = [torch.randn(2, n, 8) for n in (1, 3, 8)]
        self.hidden = torch.randn(2, 6, 16)
        self.valid = torch.tensor([[True]*6, [True]*3+[False]*3])

    def test_parallel_training_matches_each_independent_generation_level(self):
        logits, stop = self.model(self.inputs, self.hidden, self.valid)
        memory, mask = self.model.prepare_memory(self.hidden, self.valid)
        for level, inputs in enumerate(self.inputs):
            got, end = self.model.forward_level(inputs, level, memory, mask)
            torch.testing.assert_close(got, logits[level], atol=1e-6, rtol=1e-5)
            if end is not None: torch.testing.assert_close(end, stop, atol=1e-6, rtol=1e-5)

    def test_level_isolation_and_padding_isolation(self):
        logits, _ = self.model(self.inputs, self.hidden, self.valid)
        changed = [self.inputs[0], self.inputs[1] * 100, self.inputs[2] - 100]
        altered = self.hidden.clone(); altered[1, 3:] = 500
        other, _ = self.model(changed, altered, self.valid)
        torch.testing.assert_close(other[0], logits[0], atol=1e-6, rtol=1e-5)
        self.assertFalse(torch.allclose(other[1], logits[1]))

    def test_prompt_effect_and_parameter_gradients(self):
        logits, stop = self.model(self.inputs, self.hidden, self.valid)
        other, _ = self.model(self.inputs, self.hidden.flip(0), self.valid)
        self.assertFalse(torch.allclose(logits[0], other[0]))
        loss = sum(x.square().mean() for x in logits) + stop.square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in self.model.parameters()))

    def test_rejects_empty_conditioning_and_oversized_context(self):
        with self.assertRaises(ValueError):
            self.model.prepare_memory(self.hidden, torch.zeros_like(self.valid))
        with self.assertRaises(ValueError):
            self.model.prepare_memory(torch.zeros(2, 17, 16), torch.ones(2, 17, dtype=torch.bool))

    def test_bfloat16_feature_storage_is_accepted_without_autocast(self):
        logits, stop = self.model(self.inputs, self.hidden.bfloat16(), self.valid)
        self.assertTrue(all(torch.isfinite(x).all() for x in logits))
        self.assertTrue(torch.isfinite(stop).all())


if __name__ == '__main__':
    unittest.main()
