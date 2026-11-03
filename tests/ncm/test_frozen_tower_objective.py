import unittest
import torch

from test_prefix import SmallHybrid
from scripts.ncm.backbone import ConceptGenerator
from scripts.ncm.codec import Codec, CodecConfig, encode
from scripts.ncm.frozen_tower_objective import trajectory, GenerationAdapter


class FrozenTowerObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(29002)
        source = SmallHybrid().eval().requires_grad_(False)
        source.adapters.requires_grad_(True)
        source.width = 16
        self.codec = Codec(CodecConfig(length=4, width=8, heads=2, layers=0,
                                      codes=16, levels=(1, 2, 4))).eval().requires_grad_(False)
        self.model = ConceptGenerator(source, self.codec.config, parallel_attention=True,
            termination=True, cached_prefix=False, head_kind='untied',
            codebooks=self.codec.books, head_input_norm=True).eval()
        self.prompt = torch.tensor([[1, 2, 3, 4, 5]])

    def test_targets_do_not_change_student_inputs_or_predictions(self):
        a = trajectory(self.model, self.codec, self.prompt,
                       target_bytes=torch.tensor([encode('ab', 4)]))
        b = trajectory(self.model, self.codec, self.prompt,
                       target_bytes=torch.tensor([encode('xy', 4)]))
        free = trajectory(self.model, self.codec, self.prompt)
        for key in ('inputs', 'decisions'):
            for x, y, z in zip(a[key], b[key], free[key]):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
                torch.testing.assert_close(x, z, rtol=0, atol=0)
                self.assertFalse(x.requires_grad)
        torch.testing.assert_close(a['latent'], free['latent'], rtol=0, atol=0)
        output, decisions, stop = GenerationAdapter(self.model).generate_chunk(self.codec, self.prompt)
        torch.testing.assert_close(output, self.codec.decode_latents(free['latent']).argmax(-1))
        torch.testing.assert_close(stop, free['stop_logits'].sigmoid())

    def test_teacher_forced_targets_equal_codec_and_require_labels(self):
        target = torch.tensor([encode('ab', 4)])
        codes, inputs = self.codec.targets(target)
        actual = trajectory(self.model, self.codec, self.prompt, target_bytes=target,
                            self_conditioned=False)
        for a, b in zip(actual['labels'], codes):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(actual['inputs'], inputs):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            trajectory(self.model, self.codec, self.prompt, self_conditioned=False)


if __name__ == '__main__':
    unittest.main()
