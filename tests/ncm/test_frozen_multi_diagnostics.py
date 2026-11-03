import unittest
import torch
import test_frozen_tower_objective as fixtures
from scripts.ncm.diagnose_frozen_multi import probe
from scripts.ncm.codec import encode


class DiagnosticIsolationTests(unittest.TestCase):
    def test_oracle_target_does_not_affect_free_generation(self):
        f = fixtures.FrozenTowerObjectiveTests()
        f.setUp()
        before = {n: p.detach().clone() for n, p in f.model.named_parameters()}
        a = probe(f.model, f.codec, f.prompt, torch.tensor([encode('ab', 4)]), 'ab')
        b = probe(f.model, f.codec, f.prompt, torch.tensor([encode('xy', 4)]), 'xy')
        self.assertEqual(a['free_generation']['text'], b['free_generation']['text'])
        self.assertEqual(a['free_stop_probability'], b['free_stop_probability'])
        self.assertEqual(len(a['clean_teacher_forced']), 3)
        for n, p in f.model.named_parameters():
            self.assertIsNone(p.grad)
            torch.testing.assert_close(p, before[n], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
