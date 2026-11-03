from types import SimpleNamespace
import unittest
import numpy as np
import torch
from test_prefix import SmallHybrid
from scripts.ncm.audit_frozen_source import audit
from scripts.ncm.prefix import rotary, mask_for


class FrozenSourceAuditTests(unittest.TestCase):
    def test_source_only_parity_and_mismatch_reporting(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        model = SmallHybrid().eval().requires_grad_(False)
        # Nonzero adapters must not contaminate this source-only audit.
        with torch.no_grad():
            for adapter in model.adapters:
                adapter.up.weight.normal_()
            ids = torch.tensor([[1, 2, 3]])
            x = model.embed_prompt(ids)
            for layer in model.layers:
                mask = mask_for(3, 0, x.device, x.dtype) if layer.block_type == 'full_attention' else None
                x = layer(x, position_embeddings=rotary(model, x, 0, 0), attention_mask=mask)
            h = model.norm(x)[0, -1]
        weights = np.zeros((248320, 16), dtype=np.float32)
        weights[7] = h.numpy()
        model.source = SimpleNamespace(sha='test', tensors={'output.weight': SimpleNamespace(data=weights, tensor_type=0)})
        reference = {'source_sha256': 'test', 'prompts': [{'prompt': 'fixture', 'ids': [1, 2, 3], 'expected': [7]}]}
        result = audit(model, reference)
        self.assertTrue(result[0]['passed'])
        self.assertEqual(len(model.layers[-1]._forward_hooks), 0)
        reference['prompts'][0]['expected'] = [8]
        self.assertFalse(audit(model, reference)[0]['passed'])
        reference['source_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            audit(model, reference)


if __name__ == '__main__':
    unittest.main()
