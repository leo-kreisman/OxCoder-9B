import copy
import unittest

import torch

from scripts.ncm.cpu_adamw import CPUAdamW
from scripts.ncm.cpu_adamw import migrate_bundle
from scripts.ncm.master_muon import FP32MasterMuon
from scripts.ncm.anvil import OptimizerBundle


class CPUAdamWTests(unittest.TestCase):
    def test_bundle_migration_preserves_other_optimizers(self):
        parameters = [torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(3)),
                      torch.nn.Parameter(torch.ones(2, 3, dtype=torch.bfloat16))]
        original = OptimizerBundle([torch.optim.AdamW([parameters[0]]), torch.optim.AdamW([parameters[1]]),
                                    FP32MasterMuon([parameters[2]], lr=.0001)])
        for parameter in parameters: parameter.grad = torch.ones_like(parameter)
        original.step()
        copied = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
        moved = OptimizerBundle([CPUAdamW([copied[0]]), torch.optim.AdamW([copied[1]]),
                                 FP32MasterMuon([copied[2]], lr=.0001)])
        migrate_bundle(moved, copy.deepcopy(original.state_dict()))
        for parameter in parameters + copied: parameter.grad = torch.full_like(parameter, .2)
        original.step(); moved.step()
        for a, b in zip(parameters, copied): self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(original.optimizers[2].masters[0], moved.optimizers[2].masters[0]))

    def test_preserves_existing_moments_and_resumes_without_reset(self):
        torch.manual_seed(17)
        reference = torch.nn.Parameter(torch.randn(13, 7))
        standard = torch.optim.AdamW([reference], lr=.0003, weight_decay=.01, foreach=False)
        for _ in range(5):
            reference.grad = torch.randn_like(reference)
            standard.step()
        original = torch.nn.Parameter(reference.detach().clone())
        moved = CPUAdamW([original], lr=.0003, weight_decay=.01, foreach=False)
        moved.load_legacy_state(copy.deepcopy(standard.state_dict()))
        for step in range(8):
            gradient = torch.randn_like(reference)
            reference.grad = gradient.clone()
            original.grad = gradient.clone()
            standard.step(); moved.step()
            self.assertTrue(torch.equal(reference, original))
            if step == 3:
                saved = copy.deepcopy(moved.state_dict())
                restored = CPUAdamW([original], lr=.0003, weight_decay=.01, foreach=False)
                restored.load_state_dict(saved)
                moved = restored
        self.assertEqual(float(moved.inner.state[moved.copies[0]]['step']), 13)
        self.assertIs(moved.param_groups[0]['params'][0], original)
        moved.zero_grad()
        self.assertIsNone(original.grad)
        self.assertIsNone(moved.copies[0].grad)

    def test_rejects_precision_change_bad_moments_and_nonfinite_gradient(self):
        with self.assertRaises(ValueError): CPUAdamW([torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))])
        original = torch.nn.Parameter(torch.ones(3))
        moved = CPUAdamW([original])
        original.grad = torch.tensor([float('nan'), 0., 1.])
        with self.assertRaises(FloatingPointError): moved.step()
        original.grad = torch.ones_like(original)
        moved.step()
        saved = copy.deepcopy(moved.state_dict())
        saved['inner']['state'][0]['exp_avg'] = torch.zeros(4)
        with self.assertRaises(ValueError): moved.load_state_dict(saved)


if __name__ == '__main__':
    unittest.main()
