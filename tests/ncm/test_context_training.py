import unittest

import torch

from scripts.ncm.codec import Codec, CodecConfig
from scripts.ncm.context_decoder import ContextConceptDecoder, DecoderConfig
from scripts.ncm.train_context_decoder import previous_concepts, make_optimizer


class ContextTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(29)
        self.codec = Codec(CodecConfig(length=8, width=8, heads=2, layers=0,
                                      codes=17, levels=(1, 3, 8))).eval().requires_grad_(False)
        self.codes = [torch.randint(17, (2, n)) for n in (1, 3, 8)]

    def test_label_for_current_or_future_scale_cannot_enter_input(self):
        inputs = previous_concepts(self.codec, self.codes)
        altered = [self.codes[0], (self.codes[1] + 1) % 17, (self.codes[2] + 2) % 17]
        other = previous_concepts(self.codec, altered)
        self.assertTrue(torch.equal(inputs[0], torch.zeros_like(inputs[0])))
        self.assertTrue(torch.equal(inputs[1], other[1]))
        self.assertFalse(torch.equal(inputs[2], other[2]))
        self.assertTrue(all(not x.requires_grad for x in inputs))

    def test_activation_checkpoint_keeps_gradients_and_outputs(self):
        model = ContextConceptDecoder(DecoderConfig(source_width=16, latent_width=8,
            width=32, heads=4, layers=2, codes=17, levels=(1, 3, 8), maximum_prompt=16)).train()
        inputs = previous_concepts(self.codec, self.codes)
        hidden = torch.randn(2, 6, 16)
        valid = torch.ones(2, 6, dtype=torch.bool)
        logits, stop = model(inputs, hidden, valid)
        (sum(x.square().mean() for x in logits) + stop.square().mean()).backward()
        gradients = {n: p.grad.clone() for n, p in model.named_parameters()}
        model.zero_grad(set_to_none=True); model.checkpoint_blocks = True
        other, end = model(inputs, hidden, valid)
        (sum(x.square().mean() for x in other) + end.square().mean()).backward()
        for a, b in zip(logits, other): torch.testing.assert_close(a, b)
        for n, p in model.named_parameters(): torch.testing.assert_close(p.grad, gradients[n])

    def test_muon_excludes_output_heads_and_all_parameters_have_one_optimizer(self):
        model = ContextConceptDecoder(DecoderConfig(source_width=16, latent_width=8,
            width=32, heads=4, layers=2, codes=17, levels=(1, 3, 8), maximum_prompt=16))
        bundle = make_optimizer(model, .001)
        adam = {id(p) for g in bundle.optimizers[0].param_groups for p in g['params']}
        muon = {id(p) for g in bundle.optimizers[1].param_groups for p in g['params']}
        self.assertFalse(adam & muon)
        self.assertEqual(adam | muon, {id(p) for p in model.parameters()})
        self.assertIn(id(model.stop.weight), adam)
        self.assertIn(id(model.heads[0].weight), adam)
        self.assertIn(id(model.memory[1].weight), muon)


if __name__ == '__main__': unittest.main()
