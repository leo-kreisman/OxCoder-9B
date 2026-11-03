from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from scripts.ncm.exact_checkpoint import restore_trainable
from scripts.ncm.backbone import digest


class ExactCheckpointTests(unittest.TestCase):
    def test_fp16_values_survive_initial_bf16_rounding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.pt'
            original = nn.Linear(4, 2, bias=False).half()
            with torch.no_grad():
                original.weight.copy_(torch.tensor([[.1234, .2345, .3456, .4567]] * 2).half())
            identity = {'compute_dtype': 'float16'}
            torch.save({'identity': identity, 'model': {'weight': original.weight.detach().clone()}}, path)
            loaded = nn.Linear(4, 2, bias=False).bfloat16()
            loaded.load_state_dict(original.state_dict())
            loaded.half()
            self.assertFalse(torch.equal(loaded.weight, original.weight))
            restore_trainable(loaded, path, identity, digest(path))
            torch.testing.assert_close(loaded.weight, original.weight, rtol=0, atol=0)
            probe = torch.tensor([[1., 2., 3., 4.]]).half()
            torch.testing.assert_close(loaded(probe), original(probe), rtol=0, atol=0)

    def test_dtype_or_identity_mismatch_cannot_silently_round(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.pt'
            model = nn.Linear(4, 2, bias=False)
            torch.save({'identity': {}, 'model': {'weight': model.weight.detach().half()}}, path)
            with self.assertRaises(ValueError):
                restore_trainable(model, path, {}, digest(path))
            model.half()
            with self.assertRaises(ValueError):
                restore_trainable(model, path, {'different': True}, digest(path))


if __name__ == '__main__':
    unittest.main()
