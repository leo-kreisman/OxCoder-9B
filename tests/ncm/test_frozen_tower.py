import copy
import unittest

import torch

from test_prefix import SmallHybrid
from scripts.ncm import frozen_tower as tower


class FrozenTowerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29001)
        torch.set_num_threads(1)
        self.model = SmallHybrid().eval().requires_grad_(False)
        self.model.adapters.requires_grad_(True)
        # Nonzero adapters distinguish the new frozen-context architecture
        # from the old path that also adapts every prompt position.
        for adapter in self.model.adapters:
            torch.nn.init.normal_(adapter.up.weight, std=.1)
        self.prompt = torch.tensor([[1, 2, 3, 4, 5]])

    def test_outputs_gradients_and_replay_match_uncached_frozen_context(self):
        self.model.train()
        for size in (1, 4):
            x = torch.randn(1, size, 16, requires_grad=True)
            state = tower.prepare(self.model, self.prompt, alignment=1)
            before = copy.deepcopy(state.cache.layers[0].recurrent_states[0])
            parameters = [x, *self.model.adapters.parameters()]
            expected = tower.reference(self.model, x, self.prompt)
            probe = torch.randn_like(expected)
            g1 = torch.autograd.grad((expected * probe).sum(), parameters)
            actual = tower.cached(self.model, x, state)
            g2 = torch.autograd.grad((actual * probe).sum(), parameters)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            for a, b in zip(g1, g2):
                torch.testing.assert_close(a, b, rtol=.02, atol=2e-5)
            torch.testing.assert_close(state.cache.layers[0].recurrent_states[0], before, rtol=0, atol=0)
            torch.testing.assert_close(tower.cached(self.model, x, state), actual, rtol=0, atol=0)

    def test_aligned_prefix_replays_tail_without_adapting_it(self):
        self.model.train()
        for length in (5, 64, 69):
            prompt = (self.prompt.repeat(1, 14))[:, :length]
            x = torch.randn(1, 4, 16, requires_grad=True)
            state = tower.prepare(self.model, prompt)
            self.assertEqual(state.cached_tokens, length // 64 * 64)
            a = tower.reference(self.model, x, prompt)
            b = tower.cached(self.model, x, state)
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
            params = [x, *self.model.adapters.parameters()]
            ga = torch.autograd.grad(a.square().mean(), params)
            gb = torch.autograd.grad(b.square().mean(), params)
            for u, v in zip(ga, gb):
                torch.testing.assert_close(u, v, rtol=.02, atol=2e-5)

    def test_cache_survives_adapter_update_but_not_source_update(self):
        state = tower.prepare(self.model, self.prompt)
        x = torch.randn(1, 4, 16)
        original = tower.cached(self.model, x, state)
        with torch.no_grad():
            self.model.adapters[0].up.weight.add_(.1)
        actual = tower.cached(self.model, x, state)
        torch.testing.assert_close(actual, tower.reference(self.model, x, self.prompt), rtol=1e-4, atol=1e-5)
        self.assertFalse(torch.equal(actual, original))
        with torch.no_grad():
            self.model.layers[0].mlp.gate_proj.weight.add_(.01)
        with self.assertRaises(ValueError):
            tower.cached(self.model, x, state)

    def test_prompt_never_invokes_adapters(self):
        calls = []
        handles = [a.register_forward_hook(lambda *args: calls.append(True)) for a in self.model.adapters]
        try:
            tower.prepare(self.model, self.prompt, alignment=1)
            self.assertEqual(calls, [])
        finally:
            for handle in handles:
                handle.remove()

    def test_refuses_trainable_source_or_foreign_cache(self):
        state = tower.prepare(self.model, self.prompt)
        other = copy.deepcopy(self.model)
        with self.assertRaises(ValueError):
            tower.cached(other, torch.randn(1, 4, 16), state)
        self.model.layers[0].mlp.gate_proj.weight.requires_grad_(True)
        with self.assertRaises(ValueError):
            tower.prepare(self.model, self.prompt)


if __name__ == '__main__':
    unittest.main()
