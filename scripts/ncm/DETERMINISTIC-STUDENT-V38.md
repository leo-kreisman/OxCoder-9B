# Deterministic student-conditioning qualification and continuation

The v36 optimizer audit failed its exact-gradient prerequisite and did not
start training. Investigation showed that the original objective itself did
not reproduce exact gradients on repeated CUDA execution. No coding test,
gradient tolerance, or acceptance requirement was weakened to proceed.

## Isolating the discrepancy

`audit_canonical_repeatability.py` loads the immutable v32 update-3000 parent
with exact saved FP16 trainable values. It runs the original objective twice
and disabled student conditioning twice, resetting the same random states
before each pass. It performs no optimizer updates and uses only admitted
training examples, with resource probes at 64, 69, 512 and 2048 prompt tokens.

The [ordinary-execution diagnostic](validation/v38-controls/repeatability.json)
found identical losses, canonical labels and corrupted inputs in every pass.
Nevertheless, the original objective's second backward pass differed from its
first in 66 trainable tensors: relative L2 0.0004572, maximum absolute 0.015625
on loss-scaled gradients. Disabled-student differences were of similar size.
This establishes backward nondeterminism in the tested execution, not an
identified individual kernel defect. Frozen source versions and codec gradients
were checked throughout. The [initial failed audit](validation/v38-controls/initial-student-audit.json)
remains preserved.

`audit_canonical_determinism.py` repeats the same experiment with:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Set this before Python initializes CUDA. Inside the process, enable
`torch.use_deterministic_algorithms(True)`, set `cudnn.deterministic=True`,
and disable `cudnn.benchmark`. TF32 remains disabled as in the parent.
The [deterministic diagnostic](validation/v38-controls/deterministic-repeatability.json)
then produced identical losses and **bit-exact gradients for every trainable
parameter in all four passes**. Labels, corrupted inputs and random-state
fingerprints also matched. This is a numerical control, not a quality result.

## Admission and execution

`deterministic_student_policy.py` requires the complete four-pass diagnostic,
matching parent and implementation hashes, exact reference repeatability,
exact disabled-student parity, unchanged supervision and frozen-source checks.
An audit with even a nonzero reported gradient difference is rejected.

`audit_deterministic_student.py` then runs the original full-shape optimizer
qualification under those same deterministic settings: exact disabled-mode
control followed by two discarded updates, first with mixed student flags and
then with all examples student-conditioned. All trainable groups must change
finitely; source and codec remain frozen. These updates never initialize the
actual training worker. The final report embeds the repeatability evidence
and records the execution policy and implementation hashes.

Only after that audit passes does `run_deterministic_student_stage.py` launch
`train_deterministic_student.py`. The worker restores the untouched v32
update-3000 checkpoint, full optimizer and random states. It retains the
original 45,157-chunk training selection, 64 validation windows, both creator
datasets, admitted distillation, batch four, canonical target labels and
global 10,000-update learning-rate schedule. Half of training examples use
their own detached native NCM rollout as conditioning; other examples retain
the original corrupted teacher conditioning. See
[the objective description](STUDENT-CONDITIONING-V36.md).

The objective and deterministic runtime are explicitly identified as the v38
experiment. The rejected v36 run is not relabeled as passing. CUDA device 0
retains the existing checkpointed source execution and host-resident head
optimizer. GPU1's compact-data v34 training continues independently.

The full CPU suite passed 297 tests before launch, including rejection of
changed reference gradients, supervision fingerprints, execution policy and
parent identity. Actual full-shape GPU qualification remains separate.

The [full GPU optimizer audit](validation/v38-controls/optimizer.json) subsequently
passed. Disabled-mode losses and all gradients matched exactly. Both discarded
student-conditioned updates changed adapters, interfaces and classifiers
finitely, with unchanged source tensors and no codec gradients. Peak allocated
GPU memory was 6,487,657,472 bytes. The mixed and all-student control updates
took 6.36 and 5.44 seconds respectively; these include audit bookkeeping and
are not inference speeds or a representative training throughput benchmark.
The coordinator admitted actual v38 training from the untouched parent only
after this report completed.

## Operation, recovery and acceptance

- Supervisor: `ncm-deterministic-student-pipeline-v38.service`.
- Audit/training child: `ncm-deterministic-student-worker-v38.service`.
- Audit report: `results/ncm-deterministic-student-audit-v38/report.json`.
- Checkpoints and training logs: `results/ncm-deterministic-student-v38/`.
- Coordinator state and child logs: `results/ncm-deterministic-student-pipeline-v38/`.

Workers reserve application memory, cap host allocation at 18 GiB, disable
host swap and run bounded six-hour sessions. These are training constraints;
they are not an eight-GiB inference qualification. No OxCoder model weights
are streamed from SSD in this path. Failed initial qualification resumes or
restarts the original v32 supervisor from its saved state.

Save every 100 updates, evaluate every 250, and yield full state every 1000.
The evaluator and all twelve executable task definitions are the same functions
used by the parent. A candidate must pass all twelve, then reproduce the same
outputs and passes after exact checkpoint restoration in a fresh process.
Cold replay enables the identical deterministic execution policy and checks
that policy against checkpoint metadata. No AR answer fallback or evaluation
target training is used. Numerical qualification alone never sets quality ready.
