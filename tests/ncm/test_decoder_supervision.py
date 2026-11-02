import unittest
import torch
from scripts.ncm.codec import Codec, CodecConfig
from scripts.ncm.audit_decoder_supervision import decoded_loss


class DecoderSupervisionTests(unittest.TestCase):
    def test_straight_through_forward_is_discrete_but_gradient_reaches_logits(self):
        torch.manual_seed(1)
        codec = Codec(CodecConfig(length=8, width=8, heads=2, layers=0, codes=16, levels=(2, 8)))
        codec.eval().requires_grad_(False)
        logits = torch.randn(2, 8, 16, requires_grad=True)
        earlier = torch.randn(2, 8, 8)
        target = torch.randint(258, (2, 8))
        loss = decoded_loss(codec, logits, earlier, target, straight_through=True)
        with torch.no_grad():
            expected = torch.nn.functional.cross_entropy(
                codec.decode_latents(earlier + codec.books[-1](logits.argmax(-1))).flatten(0, 1),
                target.flatten())
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in codec.parameters()))


if __name__ == '__main__':
    unittest.main()
