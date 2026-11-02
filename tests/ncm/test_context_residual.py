import unittest
import torch

from scripts.ncm.codec import Codec, CodecConfig
from scripts.ncm.context_decoder import ContextConceptDecoder, DecoderConfig
from scripts.ncm.context_residual import conditioned_targets


class AdaptiveResidualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(33)
        self.codec = Codec(CodecConfig(length=8, width=8, heads=2, layers=0, codes=17, levels=(1, 3, 8))).eval().requires_grad_(False)
        self.model = ContextConceptDecoder(DecoderConfig(source_width=16, latent_width=8, width=32,
            heads=4, layers=2, codes=17, levels=(1, 3, 8), maximum_prompt=16)).eval()
        self.hidden = torch.randn(2, 5, 16); self.valid = torch.ones(2, 5, dtype=torch.bool)
        self.target = torch.randint(258, (2, 8))

    def test_clean_policy_matches_original_frozen_codec_labels_and_inputs(self):
        expected_codes, expected_inputs = self.codec.targets(self.target)
        inputs, codes, mask = conditioned_targets(self.model, self.codec, self.hidden, self.valid, self.target, 0.)
        self.assertFalse(mask.any())
        for a, b in zip(codes, expected_codes): self.assertTrue(torch.equal(a, b))
        for a, b in zip(inputs, expected_inputs): torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_on_policy_inputs_cannot_leak_current_or_future_target_text(self):
        first, labels, mask = conditioned_targets(self.model, self.codec, self.hidden, self.valid, self.target, 1.)
        other, changed, _ = conditioned_targets(self.model, self.codec, self.hidden, self.valid, (self.target + 91) % 258, 1.)
        self.assertTrue(mask.all())
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, other)))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(labels, changed)))
        self.assertTrue(all(not x.requires_grad for x in first))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_mixed_policy_is_reproducible_and_rejects_invalid_probability(self):
        torch.manual_seed(8)
        a = conditioned_targets(self.model, self.codec, self.hidden, self.valid, self.target, .5)
        torch.manual_seed(8)
        b = conditioned_targets(self.model, self.codec, self.hidden, self.valid, self.target, .5)
        self.assertTrue(torch.equal(a[2], b[2]))
        for x, y in zip(a[0], b[0]): self.assertTrue(torch.equal(x, y))
        with self.assertRaises(ValueError): conditioned_targets(self.model, self.codec, self.hidden, self.valid, self.target, 2.)


if __name__ == '__main__': unittest.main()
