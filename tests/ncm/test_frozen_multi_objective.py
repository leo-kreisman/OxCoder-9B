import unittest
import torch
from test_frozen_tower_objective import FrozenTowerObjectiveTests
from scripts.ncm import frozen_multi_objective as multi
from scripts.ncm import frozen_tower_objective as single
from scripts.ncm.frozen_tower_training import backward_example
from scripts.ncm.codec import encode


class MultiObjectiveTests(unittest.TestCase):
    def setUp(self):
        f = FrozenTowerObjectiveTests()
        f.setUp()
        self.model, self.codec = f.model, f.codec
        self.prompts = [f.prompt, f.prompt.repeat(1, 14)[:, :69]]
        self.targets = torch.tensor([encode('ab', 4), encode('xy', 4)])

    def test_mixed_teacher_student_rollout_matches_independent(self):
        actual = multi.trajectory(self.model, self.codec, self.prompts, self.targets, [False, True])
        for i, flag in enumerate([False, True]):
            expected = single.trajectory(self.model, self.codec, self.prompts[i],
                                        target_bytes=self.targets[i:i+1], self_conditioned=flag)
            for key in ('inputs', 'labels', 'decisions'):
                for a, b in zip(actual[key], expected[key]):
                    torch.testing.assert_close(a[i:i+1], b, rtol=1e-4, atol=1e-5)

    def test_student_predictions_independent_of_targets_and_other_rows(self):
        a = multi.trajectory(self.model, self.codec, self.prompts, self.targets)
        b = multi.trajectory(self.model, self.codec, self.prompts, self.targets.flip(0))
        free = multi.trajectory(self.model, self.codec, self.prompts)
        for key in ('inputs', 'decisions'):
            for x, y, z in zip(a[key], b[key], free[key]):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
                torch.testing.assert_close(x, z, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            multi.trajectory(self.model, self.codec, self.prompts, self.targets, [False])

    def test_full_training_gradient_matches_independent(self):
        items = [({'split': 'train', 'target': text}, {'ids': prompt[0].tolist(), 'is_final_chunk': bool(i)})
                 for i, (text, prompt) in enumerate(zip(('ab', 'xy'), self.prompts))]
        parameters = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        expected = [backward_example(self.model, self.codec, item, 2, 64, flag, True)
                    for item, flag in zip(items, [False, True])]
        grads = {n: p.grad.clone() for n, p in parameters.items()}
        self.model.zero_grad(set_to_none=True)
        actual = multi.backward_batch(self.model, self.codec, items, 64, [False, True])
        for n, p in parameters.items():
            torch.testing.assert_close(p.grad, grads[n], rtol=.02, atol=1e-4, msg=n)
        self.assertAlmostEqual(sum(r['loss'] for r in actual), sum(r['loss'] for r in expected), places=4)
        with self.assertRaises(ValueError):
            multi.backward_batch(self.model, self.codec, [({'split': 'validation'}, {})], 64, [True])


if __name__ == '__main__':
    unittest.main()
