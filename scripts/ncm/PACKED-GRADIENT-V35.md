# Packed-only backward state experiment

Standard `F.linear` with a frozen dequantized matrix still needs that matrix to
compute its input gradient. Retaining every dense matrix across all 32 source
layers would exceed the training memory budget. The current runtime therefore
checkpoints entire layers: it runs the layer once, recomputes its forward pass
during backward, and then computes input gradients.

`packed_gradient.py` tests a different tradeoff. Its custom autograd function
retains only the immutable packed GGUF byte tensor. Forward performs the same
unpack and `F.linear`; backward unpacks the same bytes again and computes
`gradient @ weight`. It does not compute or estimate weight gradients. This
removes the need to retain dense source weights and may allow ordinary training
activations to remain resident without recomputing every layer. Activation
memory increases relative to layer checkpointing and requires measurement.

This is an execution experiment. It neither changes quantization nor moves
weights to SSD. It is not automatically enabled in the v34 training run.

## Numerical evidence

The CUDA matrix audit tested nine representative real OxCoder source matrices,
including Q4_K/Q6_K weights, recurrent value-head permutations, attention and
MLP projections. Each was tested in FP16 and BF16 at four and 212 positions:
36 cases in total. All forward values and input gradients were bit-exact with
the existing implementation; packed tensors remained unchanged. Autograd saved
only the packed bytes. The allocator peak was 180,289,536 bytes for this
isolated audit, not for the full model.

The machine-readable evidence is
[packed-gradient.json](validation/v35-controls/packed-gradient.json). A separate
CPU hybrid-model control verifies unchanged full-objective gradients when
layer recomputation is disabled. The complete NCM unit suite passed 272 tests.

## Whole-model admission

`audit_packed_runtime.py` waits for a saved v34 session and pauses only that
experiment's supervisor. The running worker finishes normally. The audit loads
the immutable update-2500 checkpoint, restores identical full optimizer state
and compares four real updates using the resident checkpointed runtime against
four updates with packed-only autograd and no layer recomputation.

Both arms use the same training examples, RNG seed, canonical labels, corruption
and optimizer. Prompts of lengths 64, 69, 512 and 2048 exercise the qualified
shapes; the report marks synthetic extension/truncation explicitly. All updates
are discarded. Qualification requires finite unchanged-source execution, no
codec gradients, final trainable-parameter relative L2 below 1e-4, maximum
absolute difference below 2e-4 and more than 5% median speed improvement over
the last three updates. A 12 GiB allocator cap and host/GPU reserves constrain
this training experiment. They do not establish 8 GiB deployment suitability.

An optional additional profiled update records operator costs; it is excluded
from the timing comparison. Any exception marks the audit incomplete and
failed. Cleanup resumes the original supervisor, including when qualification
fails. A passing matrix audit alone cannot admit the optimization to training.

The custom backward supports first-order input gradients only. Scope changes
are process-local, serialized and restored on exceptions. Enter the packed
gradient context outside the prefix cache context and retain it through
backward. The existing no-gradient prefix cache may temporarily replace and
restore individual forwards while constructing prefix states.

No quality claim follows from these checks. The twelve autonomous executable
tasks and identical fresh-process replay remain required for acceptance.
