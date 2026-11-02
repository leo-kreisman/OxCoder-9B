import unittest
import torch
from torch.nn import functional as F
from scripts.ncm.codec import Codec, CodecConfig, PAD
from scripts.ncm.decoded_supervision import decoded_loss


class DecodedSupervisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(77)
        self.codec=Codec(CodecConfig(length=8,width=8,heads=2,layers=1,codes=17,levels=(1,3,8))).eval().requires_grad_(False)
        self.previous=torch.randn(2,8,8,requires_grad=True)
        self.logits=torch.randn(2,8,17,requires_grad=True)
        self.targets=torch.randint(256,(2,8));self.targets[:,-1]=PAD

    def test_exact_discrete_forward_nonzero_gradient_only_to_logits(self):
        loss,output=decoded_loss(self.codec,self.previous,self.logits,self.targets)
        # Use the same differentiable codec path; no_grad may dispatch a
        # different fused attention implementation with FP32 rounding changes.
        expected=self.codec.decode_latents(self.previous+self.codec.books[-1](self.logits.argmax(-1)))
        torch.testing.assert_close(output,expected,atol=0,rtol=0)
        loss.backward()
        self.assertTrue(torch.isfinite(self.logits.grad).all())
        self.assertGreater(float(self.logits.grad.norm()),0)
        self.assertIsNone(self.previous.grad)
        self.assertTrue(all(p.grad is None for p in self.codec.parameters()))

    def test_target_change_cannot_change_prediction_and_pad_is_ignored(self):
        loss,one=decoded_loss(self.codec,self.previous,self.logits,self.targets)
        _,two=decoded_loss(self.codec,self.previous,self.logits,(self.targets+1)%258)
        self.assertTrue(torch.equal(one,two))
        self.assertAlmostEqual(float(loss.detach()),float(F.cross_entropy(one[:,:-1].flatten(0,1),self.targets[:,:-1].flatten()).detach()),places=6)
        self.codec.head.weight.requires_grad_(True)
        with self.assertRaises(ValueError):decoded_loss(self.codec,self.previous,self.logits,self.targets)


if __name__ == '__main__':unittest.main()
