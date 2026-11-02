"""Non-invasive feature capture for separately identified alignment training."""
from contextlib import contextmanager


@contextmanager
def capture_concept_features(head):
    """Observe head inputs without changing logits, parameters, or state keys.

    Use one forward per level inside this scope. Saved tensors keep the student
    graph for an auxiliary training loss; the hook is removed even on failure.
    """
    captured = {}

    def capture(module, arguments):
        if len(arguments) < 2:
            raise ValueError('Expected hidden states and concept level')
        hidden, level = arguments[:2]
        if level in captured:
            raise ValueError('Duplicate concept level in capture scope')
        captured[level] = hidden

    handle = head.register_forward_pre_hook(capture)
    try:
        yield captured
    finally:
        handle.remove()
