# Layer-major frozen-prompt prefill experiment

This is an isolated execution audit, not an adopted training implementation or
a usable NCM checkpoint. The active canonical and compact-curriculum workers
continue to use their pinned original execution modules.

## Repeated work

The packed source is frozen. For a four-example batch, the existing prefill
processes all 32 source layers for prompt one, then repeats that traversal for
each remaining prompt. Each packed linear projection unpacks its source bytes
to a temporary FP16 matrix on every call. Even though the weight is identical,
four independent prompts therefore cause four unpacks of each projection.

`interleaved_prefix.py` changes the loop nesting for prefill only: process all
prompts through layer one, release its temporary matrices, then proceed to
layer two. Each layer's unpacked projection is reused across prompts of the
same dtype and device. Nothing is retained across batches or persisted to SSD.
The existing frozen GGUF bytes remain the authoritative source. The peak extra
memory is bounded by one source layer's unpacked projections, plus the live
independent prompt activations. The real peak must be measured on full shapes.

Each prompt keeps its original embedding, rotary positions, causal attention
mask and separate hybrid recurrent/attention cache. Matrix multiplication
shapes are unchanged; this does not pad multiple prompts into a single GEMM.
Single-prompt execution delegates to the original implementation. Only after
all layers finish are the independent states combined using the existing
multi-prompt cache layout.

## Isolation and checks

Temporary forward replacements are instance-local and supported only during
serialized, no-gradient prefill. Nested scopes, concurrent use and gradient
execution are rejected. `finally` restores original methods and releases
temporary weights, including when a source layer raises. Frozen source tensor
identities and versions must remain unchanged. This helper must not wrap
training-side concept forwards or activation-checkpoint replay.

The CPU tests check exact attention and recurrent state equality for unequal
prompt lengths, single-unpack reuse across repeated projection calls, cleanup
after exceptions and gradient-path rejection. They do not qualify CUDA speed
or full-model memory. The full NCM CPU test suite passed 258 tests after this
addition.

`audit_interleaved_prefix.py` loads the actual original 32-layer source and a
saved NCM checkpoint. It compares all attention keys/values, convolution states
and recurrent states against the existing prefill for `(64,69,128,256)`,
`(64,512,1024,2048)` and a single 256-token prompt. Every saved tensor must be
bit-exact and finite. Timings exclude warmup, synchronize CUDA, retain every
sample and report peak allocated memory. Prompt length adjustment is recorded
as synthetic extension/truncation; this is an execution test, not a quality
evaluation.

```bash
"$PYTHON" -m scripts.ncm.audit_interleaved_prefix \
  --source "$SOURCE_GGUF" --codec "$CODEC_CHECKPOINT" \
  --checkpoint results/ncm-compact-tower-v33/evaluated-1500.pt \
  --selection results/ncm-compact-training-v33 \
  --output results/ncm-interleaved-prefix-audit-v34 \
  --device cuda:1 --repeats 5 --wait-for-session
```

The optional Linux experiment handover pauses only the named compact supervisor,
lets its independently managed worker finish its session, and waits for the
older native evaluator to release GPU1. Both supervisors are resumed in
`finally`; the service also carries an `ExecStopPost` recovery action. GPU0's
canonical training and unrelated applications are left running. The audit
retains no optimizer updates and does not alter any active training code.

Adoption requires exact full-shape states, a measured improvement for both
four-prompt cases, acceptable memory, and a separately identified continuation
preserving model, optimizer, RNG, sampling and global schedule. Passing this
audit alone cannot claim 12/12 coding results or an inference speedup.
