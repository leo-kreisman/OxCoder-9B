"""Process-local installation of the separately audited prefix implementation."""
from contextlib import contextmanager
from . import frozen_tower_multi
from .interleaved_prefix import prepare, serial_prepare


@contextmanager
def installed():
    if frozen_tower_multi.prepare is not serial_prepare:
        raise RuntimeError('Another prefix override is installed')
    frozen_tower_multi.prepare = prepare
    try:
        yield
    finally:
        frozen_tower_multi.prepare = serial_prepare
