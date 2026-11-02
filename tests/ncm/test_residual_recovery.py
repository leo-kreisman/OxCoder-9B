import unittest
import torch
from scripts.ncm.codec import Codec, CodecConfig, resize
from scripts.ncm.audit_residual_recovery import residual_targets


class ResidualRecoveryTests(unittest.TestCase):
    def test_zero_preserved_levels_equals_original_codec_quantization(self):
        torch.manual_seed(33); torch.set_num_threads(1)
        codec = Codec(CodecConfig(length=8, width=8, heads=2, layers=0, codes=17, levels=(1, 3, 8)))
        target = torch.randint(258, (2, 8))
        z = codec.encode_latents(target)
        expected, _ = codec.targets(target)
        codes, _ = residual_targets(codec, z, [], 0)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(codes, expected)))
        altered = [(a + 1) % 17 for a in expected]
        got, total = residual_targets(codec, z, altered, 3)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(got, altered)))
        direct = sum(resize(book(code), 8) for book, code in zip(codec.books, altered))
        torch.testing.assert_close(total, direct)
        with self.assertRaises(ValueError): residual_targets(codec, z, altered, 4)


if __name__ == '__main__': unittest.main()
