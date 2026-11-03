import unittest
import torch
from scripts.ncm import selective_packed_gradient as candidate
from scripts.ncm import frozen_tower_ragged as engine
from scripts.ncm.frozen_clean_objective import backward_batch
import test_frozen_clean_objective as fixtures


class SelectivePackedGradientTests(unittest.TestCase):
    def test_all_selected_patterns_preserve_full_gradients(self):
        f = fixtures.CanonicalObjectiveTests()
        f.setUp()
        model, codec, items = f.model, f.codec, f.items
        torch.manual_seed(37001)
        expected = backward_batch(model, codec, items, 64, .15)
        gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
        original = engine.checkpoint
        for selected in ([], [0], list(range(len(model.backbone.layers)))):
            model.zero_grad(set_to_none=True)
            torch.manual_seed(37001)
            with candidate.installed(model.backbone, selected):
                actual = backward_batch(model, codec, items, 64, .15)
            self.assertEqual(actual, expected)
            self.assertIs(engine.checkpoint, original)
            for n, p in model.named_parameters():
                if p.requires_grad:
                    torch.testing.assert_close(p.grad, gradients[n], rtol=0, atol=0, msg=n)

    def test_unknown_dispatch_and_invalid_layer_fail_and_restore(self):
        f = fixtures.CanonicalObjectiveTests()
        f.setUp()
        original = engine.checkpoint
        with self.assertRaisesRegex(ValueError, 'Unrecognized'):
            with candidate.installed(f.model.backbone):
                engine.checkpoint(lambda x: x, torch.ones(1), use_reentrant=False)
        self.assertIs(engine.checkpoint, original)
        for invalid in ([100], [-1], [0, 0], [True]):
            with self.assertRaises(ValueError):
                with candidate.installed(f.model.backbone, invalid):
                    pass
        self.assertIs(engine.checkpoint, original)


if __name__ == '__main__':
    unittest.main()
