import unittest
import torch
from scripts.ncm import cached_recurrence_metadata as candidate
from scripts.ncm import frozen_tower_ragged as engine
from scripts.ncm.frozen_clean_objective import backward_batch
import test_frozen_clean_objective as fixtures


class CachedRecurrenceTests(unittest.TestCase):
    def test_boundaries_preserve_batch_order_and_reuse_with_bounded_storage(self):
        cache = candidate.Boundaries(capacity=2)
        cpu, device = cache.get(2, (4, 16, 64, 128), 'cpu')
        self.assertEqual(cpu.tolist(), [0, 4, 20, 84, 212, 216, 232, 296, 424])
        self.assertIs(cache.get(2, (4, 16, 64, 128), 'cpu')[1], device)
        self.assertEqual(cache.hits, 1)
        cache.get(1, (4,), 'cpu')
        cache.get(3, (16,), 'cpu')
        self.assertEqual(len(cache.entries), 2)
        self.assertIsNot(cache.get(2, (4, 16, 64, 128), 'cpu')[1], device)
        self.assertTrue(torch.equal(cpu, device))

    def test_metadata_mutation_and_bad_shapes_fail(self):
        cache = candidate.Boundaries()
        cpu, _ = cache.get(1, (4,), 'cpu')
        cpu.add_(1)
        with self.assertRaisesRegex(RuntimeError, 'mutated'):
            cache.get(1, (4,), 'cpu')
        for batch, lengths in ((0, (4,)), (1, ()), (1, (-1,)), (True, (4,)), (1, (True,))):
            with self.assertRaises(ValueError):
                cache.get(batch, lengths, 'cpu')

    def test_scope_restoration_and_unchanged_cpu_reference_gradients(self):
        f = fixtures.CanonicalObjectiveTests()
        f.setUp()
        torch.manual_seed(39001)
        expected = backward_batch(f.model, f.codec, f.items, 64, .15)
        gradients = {n: p.grad.clone() for n, p in f.model.named_parameters() if p.requires_grad}
        f.model.zero_grad(set_to_none=True)
        torch.manual_seed(39001)
        with candidate.installed():
            actual = backward_batch(f.model, f.codec, f.items, 64, .15)
            with self.assertRaises(RuntimeError):
                with candidate.installed():
                    pass
        self.assertIs(engine.recurrence, candidate.REFERENCE)
        self.assertEqual(actual, expected)
        for n, p in f.model.named_parameters():
            if p.requires_grad:
                torch.testing.assert_close(p.grad, gradients[n], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            with candidate.installed():
                raise ValueError('test cleanup')
        self.assertIs(engine.recurrence, candidate.REFERENCE)


if __name__ == '__main__':
    unittest.main()
