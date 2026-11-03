"""Packed input gradients with explicitly selected layer recomputation.

Runtime experiment only. Keep the scope through backward and serialize access
within a process. Existing checkpointed execution is restored on every exit.
"""
from contextlib import contextmanager
import inspect
from .packed_gradient import installed as packed_gradient
from . import frozen_tower_ragged as engine


@contextmanager
def installed(backbone, checkpoint_layers=None):
    count = len(backbone.layers)
    selected = tuple(range(0, count, 2)) if checkpoint_layers is None else tuple(checkpoint_layers)
    if (len(set(selected)) != len(selected)
            or any(type(i) is not int or not 0 <= i < count for i in selected)):
        raise ValueError('Unique valid layer indices required')
    selected = frozenset(selected)
    with packed_gradient(backbone, recompute_layers=True):
        original = engine.checkpoint
        def dispatch(function, x, *, use_reentrant):
            parameter = inspect.signature(function).parameters.get('index')
            if (use_reentrant or parameter is None or type(parameter.default) is not int
                    or not 0 <= parameter.default < count):
                raise ValueError('Unrecognized source-layer checkpoint dispatch')
            if parameter.default in selected:
                return original(function, x, use_reentrant=False)
            return function(x)
        engine.checkpoint = dispatch
        try:
            yield
        finally:
            engine.checkpoint = original
