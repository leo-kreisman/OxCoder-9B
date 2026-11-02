import copy
import unittest
import torch
from scripts.ncm.master_muon import FP32MasterMuon


class MasterMuonTests(unittest.TestCase):
    def test_small_updates_accumulate_in_master(self):
        original = torch.nn.Parameter(torch.ones(4,4,dtype=torch.bfloat16))
        resident = torch.nn.Parameter(original.detach().clone())
        master = FP32MasterMuon([original], lr=.001, weight_decay=0.)
        direct = torch.optim.Muon([resident], lr=.001, weight_decay=0.)
        for _ in range(32):
            original.grad = torch.eye(4,dtype=torch.bfloat16)
            resident.grad = original.grad.clone()
            master.step(); direct.step()
        self.assertTrue(torch.equal(resident, torch.ones_like(resident)))
        self.assertFalse(torch.equal(original, resident))
        self.assertEqual(master.masters[0].dtype, torch.float32)
        self.assertEqual(master.inner.state[master.masters[0]]['momentum_buffer'].dtype, torch.float32)

    def test_resume_preserves_unrounded_master_and_momentum(self):
        torch.manual_seed(5)
        a = torch.nn.Parameter(torch.randn(4,4).bfloat16())
        oa = FP32MasterMuon([a], lr=.001)
        for _ in range(3):
            a.grad = torch.randn_like(a); oa.step()
        saved = copy.deepcopy(oa.state_dict())
        b = torch.nn.Parameter(torch.zeros_like(a))
        ob = FP32MasterMuon([b], lr=.001); ob.load_state_dict(saved)
        self.assertTrue(torch.equal(a,b))
        gradient = torch.randn_like(a)
        a.grad = gradient.clone(); b.grad = gradient.clone()
        oa.step(); ob.step()
        self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(oa.masters[0],ob.masters[0]))

    def test_legacy_migration_retains_momentum_in_fp32(self):
        a = torch.nn.Parameter(torch.randn(4,4).bfloat16())
        old = torch.optim.Muon([a], lr=.001)
        a.grad = torch.randn_like(a); old.step()
        expected = old.state[a]['momentum_buffer'].float().clone()
        new = FP32MasterMuon([a], lr=.001)
        new.load_legacy_muon_state(old.state_dict())
        self.assertTrue(torch.equal(expected,new.inner.state[new.masters[0]]['momentum_buffer']))
        self.assertTrue(torch.equal(new.masters[0],a.float()))

    def test_corrupt_master_is_rejected(self):
        a = torch.nn.Parameter(torch.ones(4,4).bfloat16())
        optimizer = FP32MasterMuon([a], lr=.001)
        state = copy.deepcopy(optimizer.state_dict())
        state['masters'][0][0,0] = float('nan')
        with self.assertRaises(ValueError):
            optimizer.load_state_dict(state)

    def test_zero_grad_clears_original_and_master(self):
        a = torch.nn.Parameter(torch.ones(4,4).bfloat16())
        optimizer = FP32MasterMuon([a], lr=.001)
        a.grad = torch.ones_like(a); optimizer.step(); optimizer.zero_grad()
        self.assertIsNone(a.grad)
        self.assertIsNone(optimizer.masters[0].grad)


if __name__ == '__main__':
    unittest.main()
