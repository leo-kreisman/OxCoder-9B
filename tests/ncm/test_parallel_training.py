from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.ncm.parallel_training import assignments, check_inventory, example_seed, inputs_for, sum_gradients
from scripts.ncm.codec import Codec, CodecConfig
from scripts.ncm.train_independent import corrupted_inputs


def reduction_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(41)
        model = torch.nn.Linear(3, 2)
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        x = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10
        y = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 20
        # Deliberately unequal counts of local backwards: one versus three.
        indices = [0] if rank == 0 else [1, 2, 3]
        for i in indices:
            ((model(x[i:i+1]) - y[i:i+1]).square().mean() / 4).backward()
        check_inventory(list(model.named_parameters()))
        sum_gradients(list(model.named_parameters()), bucket_bytes=8)
        actual = {n: p.grad.clone() for n, p in model.named_parameters()}
        reference = torch.nn.Linear(3, 2)
        reference.load_state_dict(state)
        (reference(x) - y).square().mean().backward()
        passed = all(torch.allclose(actual[n], p.grad, atol=1e-7, rtol=1e-6)
                     for n, p in reference.named_parameters())
        if not passed:
            raise AssertionError('Summed gradients differ from the global mean')
        # A missing gradient on one worker must fail on both, not deadlock.
        if rank == 0:
            model.bias.grad = None
        try:
            sum_gradients(list(model.named_parameters()))
        except FloatingPointError:
            Path(output, str(rank)).write_text('passed')
        else:
            raise AssertionError('Missing worker gradient was accepted')
    finally:
        dist.destroy_process_group()


class ParallelTrainingTests(unittest.TestCase):
    def test_balancing_preserves_batch_and_is_deterministic(self):
        self.assertEqual(assignments([64, 512, 1024, 2048], 2), [[3], [0, 1, 2]])
        self.assertEqual(assignments([64, 64, 64, 64], 2), [[0, 2], [1, 3]])
        with self.assertRaises(ValueError):
            assignments([64], 2)

    def test_noise_seed_does_not_depend_on_worker_assignment(self):
        self.assertEqual(example_seed(1, 20, 3), example_seed(1, 20, 3))
        self.assertEqual(len({example_seed(1, s, i) for s in range(4) for i in range(4)}), 16)

    def test_seeded_corruption_preserves_labels_and_global_rng(self):
        codec = Codec(CodecConfig(length=8, width=8, heads=2, layers=0, codes=16, levels=(1, 4, 8)))
        codes = [torch.arange(size)[None] for size in codec.config.levels]
        before = [value.clone() for value in codes]
        rng = torch.random.get_rng_state().clone()
        first, second = inputs_for(codec, codes, .5, 14), inputs_for(codec, codes, .5, 14)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(codes, before)))
        clean = inputs_for(codec, codes, 0, 14)
        reference = corrupted_inputs(codec, codes, 0)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(clean, reference)))
        changed = [value.clone() for value in codes]
        changed[-1].add_(1)
        unchanged_inputs = inputs_for(codec, changed, .5, 14)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, unchanged_inputs)))

    def test_global_mean_with_unequal_backward_counts_and_missing_gradient(self):
        with TemporaryDirectory() as directory:
            mp.spawn(reduction_worker, args=(str(Path(directory, 'rendezvous')), directory),
                     nprocs=2, join=True)
            self.assertEqual([Path(directory, str(r)).read_text() for r in range(2)], ['passed', 'passed'])


if __name__ == '__main__':
    unittest.main()
