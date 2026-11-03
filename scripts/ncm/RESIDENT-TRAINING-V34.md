# Qualified execution changes for full-source NCM training

This stage continues the v33 compact-code curriculum with two execution changes:
layer-major reuse of temporary unpacked prefix weights and GPU-resident FP32
AdamW state for the concept classifiers and stop head. It preserves the source
model, codec, trainable tensors, training corpus, sampling, optimizer moments,
random-number states and global learning-rate schedule. It does not change the
twelve-task executable quality gate. Runtime qualification is not model acceptance.

## Evidence before adoption

The reference system has two RTX 5060 Ti 16 GB devices. These measurements use
CUDA device 1; they are neither Apple Metal measurements nor output-token rates.

The prefix-only audit used the v33 update-1500 checkpoint. Five timed repetitions
followed warm-up. All attention, convolution and recurrent prefix tensors were
bit-exact with the reference implementation:

| Independent prompt lengths | Reference median | Reuse median | Ratio |
| --- | ---: | ---: | ---: |
| 64, 69, 128, 256 | 0.5996 s | 0.3719 s | 1.612 |
| 64, 512, 1024, 2048 | 1.7347 s | 1.5650 s | 1.108 |
| 256, single prompt | 0.1924 s | 0.1920 s | 1.002 |

The second group explicitly truncates or extends two training prompts to test
those shapes. The largest measured allocator peak was 5,560,389,632 bytes.
Temporary layer weight reuse cost approximately 226 MB in the first group.
See [the complete prefix report](validation/v34-controls/prefix-execution.json).

A separate head-optimizer audit loaded all 67,129,345 real FP32 head parameters
and saved AdamW moments, then applied four synthetic gradient updates, including
a zero-gradient update. CPU AdamW took a median 0.7139 s versus 0.0297 s for
resident GPU AdamW. Maximum parameter difference was 2.98e-8; first moments were
bit-exact and maximum second-moment difference was 7.11e-15. This is an isolated
optimizer measurement. It is not a 24x whole-training speedup. All audit updates
were discarded. See [the optimizer report](validation/v34-controls/head-residency.json).

## Full-step admission

`audit_resident_runtime.py` must run before adoption. It loads one immutable v33
checkpoint and restores identical model weights, full optimizer moments and RNG
seed separately for each arm. Both arms run four real updates on the same
training examples, with prompt lengths 64, 69, 512 and 2048. The reference uses
serial prefix execution and CPU head AdamW. The candidate uses layer-major
prefix execution and resident head AdamW. All updates are discarded.

Admission requires final parameter relative L2 difference below 1e-4, maximum
absolute difference below 2e-4, unchanged frozen source tensors and no codec
gradients. The median of the last three updates must improve by more than 5%.
The report records implementation hashes and the exact initialization checkpoint
hash. A different checkpoint or changed implementation requires another audit.
GPU arithmetic can differ slightly from CPU arithmetic; the training trajectory
is not claimed to be bit-exact after migration.

## State-preserving continuation

`ResidentTowerOptimizer` retains the existing FP32 Muon master parameters for
matrix adapters and existing interface AdamW. Only the classifiers' and stop
head's AdamW parameters and moments move from host to GPU. Both implementations
use the existing global gradient scale and clipping. Legacy state import is an
explicit operation; loading an unrelated optimizer format fails.

The v34 trainer imports the v33 evaluation and validation functions directly.
It retains batch size four, the 50% compact/original sampling mixture, canonical
code labels, 0.15 maximum conditioning corruption, seed 29004, warm-up 100 and
the existing 10,000-update global schedule. Migration at update 2000 continues
at 2001. Each 500-update session saves full training state; every 250 updates
also produces an immutable checkpoint and autonomous coding evaluation.

The original OxCoder weights remain packed and frozen. Temporary FP16 matrices
exist during computation. The codec remains frozen. This implementation does
not use SSD weight offloading, an AR answer fallback or a teacher at inference.

## Coordinator and failure recovery

`run_resident_training_stage.py` first verifies that both previous v28 native
training arms have saved completion/stop markers. It records pending evaluations
in `deferred-native-evaluations.json`, stops only their obsolete monitor, and
leaves all checkpoints and reports intact. An already running evaluator is
allowed to finish. This prevents old monitor cleanup from launching another
full-model evaluator into the new worker's GPU allocation.

The coordinator waits for the v33 update-2000 session, checks the progress,
session and evaluated-checkpoint identities, and gives an existing passing
candidate priority. While the full-step audit owns the device, it retires the
paused v33 supervisor. It never resumes that supervisor immediately before
stopping it, which would allow a duplicate worker to launch.

Only a complete, numerically passing, faster audit on that exact checkpoint
admits v34 training. Failed qualification restarts the original v33 supervisor
from its existing saved state. The new worker cannot launch while an old worker
or evaluator owns its device. Named worker services have 18 GiB host-memory and
zero host-swap limits; the trainer admits a GPU allocator budget of at most
9 GiB while maintaining resource headroom. These are training limits, not an
8 GiB deployment qualification.

A candidate passes only after all twelve executable tasks pass. A separate
process then reloads the same immutable checkpoint and must reproduce all
outputs and checks. Completing the schedule, reducing loss, passing numerical
audits or improving throughput cannot set quality acceptance. Broader coding
and deployment-memory qualification remains separate from this elementary gate.

## Validation

The repository's NCM unit suite passed 268 tests after this implementation.
Coverage includes exact CPU state migration and subsequent updates, native
checkpoint restoration, Python/Torch RNG restoration, strict optimizer-format
handling, nonfinite-gradient rejection, restoration after prefix-context errors,
unchanged evaluation function identity and coordinator handover boundaries.
The full-step GPU audit is separately required; unit tests do not replace it.
