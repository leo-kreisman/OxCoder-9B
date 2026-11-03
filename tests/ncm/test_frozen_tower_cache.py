import gc
import unittest
import weakref
import torch

import test_frozen_tower as fixtures
from scripts.ncm import frozen_tower as tower
from scripts.ncm.frozen_tower_cache import fork_layer
from scripts.ncm.prefix import fork_layer as old_fork


class FrozenCacheLifetimeTests(unittest.TestCase):
    def test_release_does_not_wait_for_cyclic_gc_and_state_remains_independent(self):
        fixture = fixtures.FrozenTowerTests()
        fixture.setUp()
        state = tower.prepare(fixture.model, fixture.prompt, alignment=1)
        enabled = gc.isenabled()
        gc.disable()
        try:
            control = old_fork(state.cache, 0)
            control_ref = weakref.ref(control.layers[0])
            del control
            self.assertIsNotNone(control_ref())  # demonstrate the former cycle
            fork = fork_layer(state.cache, 0)
            ref = weakref.ref(fork.layers[0])
            original = state.cache.layers[0].recurrent_states[0].clone()
            value = torch.ones_like(original, requires_grad=True)
            returned = fork.layers[0].update_recurrent_state(value)
            self.assertIs(returned, value)
            torch.testing.assert_close(state.cache.layers[0].recurrent_states[0], original, rtol=0, atol=0)
            del fork
            self.assertIsNone(ref())
        finally:
            gc.collect()
            if enabled:
                gc.enable()
        self.assertIsNone(control_ref())


if __name__ == '__main__':
    unittest.main()
