# Hard-hierarchy byte supervision

This experiment adds a training signal to the deterministic student-conditioned
version 38 continuation. It retains the original frozen OxCoder tower, canonical
concept labels, frozen codec and native next-concept inference. It is an
experimental checkpoint lineage, not a qualified coding release.

## Hypothesis and comparison

Concept cross entropy penalizes choosing the wrong code ID. It does not directly
measure the resulting decoded text. A discrete mistake can have a different
effect depending on the other selected concepts and the codec decoder. The new
auxiliary objective asks whether gradients from those decoded errors improve
autonomous generation while canonical concept supervision remains present.

The parent is version 38 update 3250, SHA-256
`8548c54f432897999731a4eeaedce0add083eed7b90317377be7e5dfb3500007`.
It had zero exact validation windows out of 64, mean nonpadding byte accuracy
0.1144287110, and zero of twelve executable coding tasks passed. These are the
pre-intervention measurements, not results of the new objective.

Version 38 continues independently on GPU0. Version 40 restores this same
parent, including optimizer moments, FP32 Muon masters, sampler and random
states, on GPU1. Both devices are RTX 5060 Ti cards. Compare evaluations at the
same global update and verify sampled example IDs and flags before attributing
differences to the loss. GPU placement and competing application load make
wall-clock comparisons uncontrolled; this is a quality experiment.

The earlier version 28 decoded-text experiment used a separate newly trained
129M-parameter decoder with frozen source features, adaptive residual labels,
and gradients only through the final code level. Version 40 instead uses the
original 32-layer source tower with adapters, canonical labels, and byte-loss
gradients into all four code predictions on student-conditioned examples. A
failure of one recipe does not establish success or failure of this different
recipe. Neither is an official pretrained-AR-to-NCM conversion method.

## Objective and information boundary

For each student-conditioned example and concept level `l`, let `p_l` be the
softmax of its logits and `E_l` the frozen codebook. Construct:

```
hard_l = E_l[argmax(logits_l)]
soft_l = p_l @ E_l
z_l = resize(hard_l) + (linear_resize(soft_l) - stop_gradient(linear_resize(soft_l)))
byte_logits = frozen_codec_decoder(sum_l z_l)
```

The forward latent uses hard codes, not a soft mixture. The subtraction supplies
a straight-through **surrogate** gradient; it is not the derivative of argmax
or an unbiased estimator. Every level's logits receive that gradient. Earlier
student rollouts remain detached, so this does not backpropagate through
discrete choices or through the construction of later-level inputs.

The loss is the existing equal-weight mean level CE plus 0.1 response-stop BCE,
plus **0.1 byte CE**. Byte CE is averaged over non-PAD positions separately for
each student-conditioned example, then over the entire batch. Teacher-conditioned
rows contribute zero auxiliary loss. EOS is supervised. Approximately half of
examples are student-conditioned under the unchanged sampler policy.

Target bytes are loss labels only. They never enter a student-conditioned
rollout or construct its conditioning. The codec's encoder, codebooks, decoder
and original packed source weights remain frozen. The source adapters,
projection, positional/level interfaces, code classifiers and termination head
are the same trainable parameters as before. No additional inference module,
autoregressive verifier, answer fallback or benchmark lookup is introduced.

CUDA's native linear-interpolation backward rejects deterministic execution.
The auxiliary path therefore computes its backward linear operator as a small
matrix multiplication. The matrix is obtained by applying the original resize
operator to the identity basis. Hard forward interpolation still uses the
original codec routine. CPU tests compare the surrogate's derivative with the
native derivative, including the actual 4/16/64/128-to-128 shapes. A full-size
isolated CUDA codec backward completed with finite gradients at every level
under the deterministic policy. This is a numerical check, not a speed result.

## Preserved data and optimization

The original admitted corpus remains 45,157 training chunks from 8,408 complete
responses, including 903 admitted distillation chunks, with both creator
sources (`TeichAI/Ox-Alpha-10k` and `TeichAI/DeepSeek-v4-Pro-Agent`) and NVIDIA
OpenCodeInstruct. All 64 fixed validation windows remain. The compact-corpus
selection from version 34 is not imported into this branch.

Batch size remains four, global schedule ends at update 10000, and learning
rates, warmup, weight decay, clipping and optimizer implementation are unchanged.
The run restores the original 3250-update state rather than audit-updated
weights. It uses CPU-resident AdamW head moments and the original checkpointed
gradient runtime. This is training-state placement, not SSD weight offloading.

The fixed coefficient 0.1 is chosen before the branch's executable results.
Zero weight must reproduce the version 38 loss and **every gradient exactly**.
When no rows are student-conditioned the auxiliary decoder is bypassed and the
same exact-reference requirement applies. No gate targets enter optimization.

## Qualification and operation

1. Run the repository NCM unit suite. It checks hard forward values, target
   independence, frozen codec parameters, nonzero finite gradients at all
   levels, exclusion of teacher-conditioned rows, deterministic interpolation
   derivatives and unchanged evaluator function identities.
2. Wait for version 34 to finish saving and evaluating its update-4000 session.
   Pause only its supervisor while its worker finishes. An existing passing
   candidate takes priority. Version 38 continues training independently.
3. Run `audit_decoded_hierarchy` against the immutable version 38 parent. Its
   four resource-probe prompts span lengths 64, 69, 512 and 2048. These prompts
   are training records extended/truncated for allocation testing, not quality
   examples. The disabled auxiliary must match the reference loss and all
   gradients exactly. Two discarded optimizer updates exercise mixed and all
   student-conditioned batches, checking finite updates, unchanged source
   weights and absence of codec gradients.
4. Only a complete passing report with matching parent, runtime and source
   hashes admits training. Audit failure resumes the saved version 34 run.
   Successful qualification retires its supervisor and preserves all artifacts.
5. Save every 100 updates, evaluate every 250, and yield full state every 1000.
   The unchanged executable evaluator requires twelve successful programs.
   A candidate must reproduce identical outputs after an exact checkpoint
   restore in a new process before acceptance is recorded.

The coordinator uses the established pinned CUDA environment:

```bash
python3 -S -m scripts.ncm.run_decoded_hierarchy_stage \
  --root /absolute/path/to/OxCoder-9B \
  --python /absolute/path/to/training/python \
  --source /absolute/path/to/OxCoder-9B.Q4_K_M.gguf
```

It creates a user-systemd worker with an 18 GiB host-memory limit, no host swap,
two PyTorch CPU threads and a six-hour session limit. GPU admission reserves
four GiB and runtime headroom checks remain enabled. These are training limits,
not an eight-GiB deployment measurement. Deterministic settings include
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic PyTorch/cuDNN, disabled cuDNN
benchmarking and disabled TF32.

Artifacts and units:

| Purpose | Location |
| --- | --- |
| Coordinator | `ncm-decoded-hierarchy-pipeline-v40.service` |
| Worker | `ncm-decoded-hierarchy-worker-v40.service` |
| Numerical audit | `results/ncm-decoded-hierarchy-audit-v40/report.json` |
| Supervisor state/log | `results/ncm-decoded-hierarchy-pipeline-v40/` |
| Training/checkpoints | `results/ncm-decoded-hierarchy-v40/` |
| Comparison continuation | `results/ncm-deterministic-student-v38/` |

Each update records byte CE per example, canonical code accuracy, total loss,
sample identities, flags, gradient norm and allocation peak. Auxiliary CE is
zero on excluded examples; this must not be interpreted as their reconstruction
quality. Checkpoint implementation hashes reject silent objective edits.
Resumption preserves model, optimizer and random state. Inference contains no
byte-loss calculation and imports the unchanged validation and coding routines.

Even a twelve-task acceptance requires separate broader coding evaluation and
Mac memory/latency measurements before a professional or deployment claim.
