"""Independent hybrid cache forks without a layer-to-bound-method cycle.

A bound replacement method stored on its own instance keeps the layer alive
until cyclic GC runs. Its transient convolution/recurrent tensors can retain
the autograd graph and large unpacked weight tensors. A weak receiver permits
ordinary reference counting to release them immediately after use.
"""
import weakref
from .prefix import fork_layer as original_fork


def fork_layer(cache, index, batch_size=1):
    fork = original_fork(cache, index, batch_size)
    layer = fork.layers[index]
    if hasattr(layer, 'recurrent_states'):
        receiver = weakref.ref(layer)

        def replace_recurrent(new, state_idx=0, **kwargs):
            owner = receiver()
            if owner is None:
                raise RuntimeError('Recurrent cache receiver has been released')
            owner.recurrent_states[state_idx] = new
            owner.is_recurrent_states_initialized[state_idx] = True
            return new

        layer.update_recurrent_state = replace_recurrent
    return fork
