# Canonical student-conditioned continuation

This is an experimental training branch. Passing numerical qualification does
not establish coding quality. Acceptance still requires all twelve executable
coding tasks and identical outputs after a fresh-process checkpoint reload.

## Evidence motivating the experiment

The update-2500 v32 checkpoint was restored exactly after selecting FP16
execution. The diagnostic compared autonomous generation against privileged
correct concept prefixes on sixteen distinct training responses and sixteen
validation responses. It performed no updates and did not load coding-gate
examples. See [the recorded summary](validation/v36-controls/generation-gap.json).

| Split | Autonomous byte accuracy | Three correct earlier levels | All four correct levels |
| --- | ---: | ---: | ---: |
| Training | 0.2373 | 0.9312 | 1.0000 |
| Validation | 0.0852 | 0.7174 | 1.0000 |

All-gold reconstruction was exact for 32/32 diagnostic windows. Neither the
autonomous path nor the three-gold-level path produced an exact window. These
are byte reconstruction diagnostics, **not coding success rates**. Twelve of
sixteen validation examples included earlier gold response chunks in their
prompt; they are not whole-response generation tests. The sample is small.

This establishes a substantial conditioning gap; it does not establish that
exposure bias is the only problem. The first-level prediction error, train to
validation gap, representation quality and available training volume remain
material. No training improvement is claimed before measurement.

## Controlled change

`canonical_onpolicy.py` assigns each example a student-conditioning flag. For
flagged examples, a detached native NCM rollout receives only the prompt and
previous student concept predictions. It receives no target bytes or labels.
The gradient-bearing forward pass trains on those intermediate inputs. For
other examples, conditioning follows the existing canonical-label objective
with the same maximum uniform corruption probability of 0.15.

Both paths use **unchanged canonical codec labels**. Unlike the older adaptive
residual-label experiment, mistakes in the rollout do not change the target
code IDs. Code cross-entropy is averaged over positions, examples and levels;
the final-level stop loss retains its 0.1 coefficient. The student flag has
probability 0.5. The legacy random draws that were previously discarded now
select this flag, preserving the response-sampler random stream.

The branch starts from the fully saved and evaluated v32 update-3000 checkpoint.
It preserves the original training selection (45,157 chunks, 8,408 complete
responses), the 64 validation windows, both creator datasets, 903 admitted
distillation chunks, frozen packed OxCoder source, frozen codec, all optimizer
moments, FP32 Muon masters, global learning-rate schedule, batch size four and
random generator state. No source weights are streamed from SSD during this
OxCoder training path. CPU-resident optimizer state is separate from model
inference memory.

The compact-corpus run on GPU1 remains a separate experiment. Its different data
mixture prevents attributing cross-run quality differences solely to the
conditioning objective. Checkpoints from v32 at matching updates are the
appropriate historical control where available.

## Qualification before adoption

1. Run `python -m unittest discover -s tests/ncm` in the configured environment.
   CPU tests cover exact disabled-mode gradients, target-independent student
   inputs, fixed canonical labels, finite mixed-mode gradients, unchanged
   evaluator functions and rejection of changed parent/audit identities.
2. `run_canonical_onpolicy_stage.py` pauses only the v32 supervisor and waits
   for its worker to finish saving and evaluating update 3000. A previously
   passing candidate takes priority. No unrelated application is stopped.
3. `audit_canonical_onpolicy.py` restores the exact checkpoint, then compares
   the original objective against disabled student conditioning on real
   packed-weight CUDA execution. Losses and **all trainable gradients must
   match exactly**. Inputs exercise 64, 69, 512 and 2048 prompt tokens; repeated
   or truncated prompts are resource probes, not quality benchmarks.
4. The audit performs two discarded optimizer updates, first with mixed flags,
   then with all examples student-conditioned. Parameters and gradients must
   remain finite; adapters, interfaces and classifiers must all change. Source
   versions must remain unchanged and the codec must receive no gradients.
5. Adoption requires the same checkpoint hash, corpus, implementation hashes
   and complete passing report. Failure resumes the old supervisor. The real
   training worker restores the untouched parent, never audit-updated weights.

The unit tests passed 289/289 before launch. Full-shape CUDA qualification is
recorded separately under `results/ncm-canonical-student-audit-v36/report.json`;
its status must be inspected before describing the branch as qualified.

## Execution and artifacts

The saved update-3000 parent completed its unchanged evaluation before
handover: 0/12 executable tasks and 0/64 exact validation windows. Its mean
nonpadding byte accuracy was 0.097282. The
[parent evaluation summary](validation/v36-controls/parent-quality.json)
records the checkpoint and full-report hashes. These values define the
pre-intervention boundary; they are not student-conditioned results.

The coordinator runs `scripts.ncm.train_canonical_onpolicy` on CUDA device 0.
It uses the existing checkpointed gradient engine and host-resident AdamW head
moments, avoiding an unqualified runtime change on the memory-shared GPU.
Allocator admission reserves four GiB of device headroom. The worker has an
18 GiB host-memory ceiling, no host swap allowance and a six-hour session limit.
These are **training limits**, not an eight-GiB deployment qualification.

- Coordinator: `ncm-canonical-student-pipeline-v36.service`.
- Worker/audit: `ncm-canonical-student-worker-v36.service`.
- Coordinator log and state: `results/ncm-canonical-student-pipeline-v36/`.
- Qualification: `results/ncm-canonical-student-audit-v36/report.json`.
- Training/checkpoints: `results/ncm-canonical-student-v36/`.

Save every 100 updates, run the unchanged executable gate every 250, and yield
with full state every 1000 updates. Global schedule ends at update 10000. An
exhausted schedule is a research checkpoint, not acceptance. Resume must retain
all identities, optimizer state and random streams. Never edit implementation
files pinned by a live checkpoint; introduce a new versioned branch instead.

`candidate.json` requires twelve passes but still sets `quality_ready: false`.
The coordinator reloads the checkpoint in a new process and calls the same
`train_frozen_clean.evaluate` function. `exact_checkpoint.restore_trainable`
restores original saved values after setting the final compute dtype, avoiding
an FP16 to BF16 to FP16 round trip. `verify_replay` requires the same checkpoint,
all twelve passes, identical generated outputs and original execution checks.
Only then can `acceptance.json` be written. No target lookup, memorized test
answers, autoregressive answer fallback or changed checker is admitted.

Even twelve elementary passes are not evidence of broad professional coding
quality. A separate untouched coding benchmark and deployment measurements
remain necessary for such a claim.

## Public training evidence checked on 2026-11-03

The [official training-code issue](https://github.com/HazyResearch/ncm/issues/10)
still records an intended release, without an attached training recipe. The
[coding-agent discussion](https://github.com/HazyResearch/ncm/issues/17) contains
community loss logs but does not establish a checkpoint passing executable
coding tests. Neither is evidence of a ready-made OxCoder conversion.

A [public XSum-derived dialogue experiment](https://huggingface.co/danielharkin21/ncm-chat-experimental-generator)
reports lower held-out concept loss after 4,000 updates, while its author
explicitly describes repetitive or incorrect replies. Its training script is
useful implementation evidence; its model is not a coding replacement.
[Independent NCM controls](https://github.com/Quantix-eng/ncm-experiment-notes)
likewise explicitly avoid claiming a model-performance breakthrough. These
sources reinforce keeping reconstruction, conditional loss and autonomous
executable quality as separate measurements.
