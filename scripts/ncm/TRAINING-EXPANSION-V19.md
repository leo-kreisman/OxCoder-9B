# Training-data expansion and accumulated NCM updates

This stage keeps the transferred OxCoder backbone and the frozen multiscale
codec. It adds Python instruction data, expands offline response distillation,
and supports several independent training examples per optimizer update.
The implementation is experimental. Dataset admission, successful training,
codec reconstruction, and executable coding acceptance are separate results.

The coding acceptance target remains all 12 tasks in `coding_eval.py`, using
the student's own coarse-to-fine generation. Original OxCoder responses never
substitute for student outputs. The diagnostic has been reused during research;
it is development data, not an untouched final benchmark.

## Starting artifacts

Use the environment and source-model checksums from
[the v18 execution audit](TRAINING-ACCELERATION-V18.md). The source is the
validated OxCoder Q4_K_M GGUF, reconstructed into a resident BF16 training
backbone. Training uses two CUDA devices; these commands are not Mac inference
instructions or an 8 GB deployment measurement.

Required inputs:

| Input | Purpose |
|---|---|
| `results/ncm-contextual-decoder-v9/best.pt` | Frozen codec and codebooks |
| `results/ncm-contextual-expansion-v17` | Existing creator data, earlier distillation, frozen evaluation |
| `results/ncm-creator-admitted-v15b` | Additional admitted creator conversations |
| `results/ncm-open-corpus-v11c` | Frozen conversation/benchmark exclusion reference |
| `results/tokenizer` | Original OxCoder tokenizer and chat template |
| An immutable evaluated v18 checkpoint | Initialization for the next learning stage |

Large local artifacts are not included in the Git source tree. Their manifests
and predecessor documentation specify provenance; do not replace them with
similarly named checkpoints from another codec or model.

## 1. Acquire a pinned Python instruction supplement

[NVIDIA OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct)
contains coding prompts, responses, generated unit tests, and publisher-reported
test outcomes. The dataset is published under CC BY 4.0. Preserve attribution
to NVIDIA Corporation and the dataset authors, Ahmad et al. (2025), together
with the source revision and card when distributing derived data.

`acquire_opencode.py` pins revision
`8f3ba5bafe4d6e8db46082cf7ae6741bc370604d` and downloads only
`data/train-00000-of-00050.parquet`, 133,476,478 bytes, SHA256:

```text
342757e0c6b706c8f68cf0a867ead5c417d0273726453dbcfc934f5c3c6ca891
```

Admission requires a reported average test score of 1.0 and a nonempty list
containing only reported passes. It also requires parseable Python with a
top-level function or class, a nonempty prompt, and bounded response length.
AST parsing does not execute code. Publisher test outcomes are not independent
verification, so `targets_are_verified` remains false.

The existing admission filter excludes frozen conversation groups, exact and
near duplicate targets, and benchmark/held-out lexical overlap. Common
system/developer wrappers are treated as described in
[creator admission](CREATOR-EXPANSION-V15.md). These filters are conservative
and do not prove semantic decontamination.

```bash
python -m scripts.ncm.acquire_opencode \
  --frozen results/ncm-open-corpus-v11c \
  --admitted results/ncm-creator-admitted-v15b \
  --output results/ncm-opencode-v19 \
  --maximum 10000 --max-target-bytes 8192
```

The completed acquisition admitted 10,000 responses from 36,773 examined rows.
It excluded 21,697 rows with publisher test failures, 4,739 near duplicates,
287 benchmark/holdout overlaps, and 50 exact existing targets. The original
creator datasets remain in the base inventory.

## 2. Expand offline teacher distillation

Run the original GGUF in a separate local OpenAI-compatible server. While the
student occupies both GPUs, a CPU-only teacher can collect responses without
claiming additional accelerator memory. Use a separate port and one slot:

```bash
CUDA_VISIBLE_DEVICES= llama-server \
  --model /path/to/OxCoder-9B.Q4_K_M.gguf \
  --alias oxcoder-offline-teacher \
  --host 127.0.0.1 --port 18119 \
  --gpu-layers 0 --parallel 1 --ctx-size 4096 \
  --threads 4 --threads-batch 4 \
  --batch-size 256 --ubatch-size 128 --reasoning off
```

Bind the collection to the model checksum and the actual server executable:

```bash
python -m scripts.ncm.distill_admitted \
  --admitted results/ncm-creator-admitted-v15b \
  --frozen results/ncm-open-corpus-v11c \
  --output results/ncm-distillation-v19 \
  --teacher-file /path/to/OxCoder-9B.Q4_K_M.gguf \
  --expected-teacher-sha256 a171c091c67c3547c45ec58f1b85a888d73725315fde71bc929ff9f8c255bb6f \
  --server-binary /path/to/llama-server \
  --per-source 64 --prompt-limit 1536 --max-tokens 768
```

Selection is randomized within each source, using distinct admitted training
conversations and a recorded seed. Oversized prompts are skipped rather than
truncated. The full rendered prompt must fit the specified prompt limit.
Recorded tool messages become text observations; no tool is executed.

Each request is cached under its content hash. A rerun with the same identity
reuses completed requests. Changes to the model, tokenizer, selection, source
manifest, or implementation require a new output directory. There are no
automatic HTTP retries that could queue duplicate requests behind a slow one.

Only nonempty text responses ending with `finish_reason: stop` are eligible.
Truncated outputs and unexecuted tool calls are rejected. Eligible teacher
responses undergo the same benchmark/holdout exclusion before admission.
The final archive is written after collection completes. `progress.json` is
not an admissible replacement for a completed manifest.

This is response distillation: the teacher's text is encoded by the NCM codec
to form student training targets. It is not token-logit matching between
incompatible vocabularies, and it adds no teacher dependency to student
inference. Teacher completion alone does not establish answer correctness.

## 3. Prepare conditioning without changing evaluation

Append completed admitted archives to the base inventory:

```bash
python -m scripts.ncm.prepare_training_expansion \
  --base results/ncm-contextual-expansion-v17 \
  --addition results/ncm-opencode-v19 \
  --addition results/ncm-distillation-v19 \
  --output results/ncm-expanded-training-v19 \
  --chunks-per-response 8
```

The teacher archive must be complete before running this combined command.
For a separate instruction-only control, omit that `--addition` and choose a
different output directory. This control was prepared as
`results/ncm-opencode-prepared-v19`:

- 59,896 added chunks from all 10,000 admitted responses;
- no prompt exceeded the 2,048-token limit;
- 52,583 existing base rows retained;
- frozen evaluation SHA256 unchanged:
  `cd6ce9b9a86447ab4250a2ffb405eebe99334729a370d537e68fcab9d2f4611d`.

Each chunk retains its source, conversation group, teacher flag, and teacher
checkpoint hash where applicable. Conditioning includes the complete original
chat prompt and all preceding assistant chunks. Chunk selection limits training
exposure per response; it never removes the preceding history from a selected
chunk's prompt. No new held-out rows are appended or silently moved to training.

The trainer subsequently runs the existing exact-reconstruction filter on
training chunks only. All selected development errors remain present.

## 4. Qualify the execution path on the new inventory

Copy the selected v18 checkpoint to an immutable filename and record its hash.
Use that same file for both the runtime audit and training initialization:

```bash
python -m scripts.ncm.audit_training_runtime \
  --source /path/to/OxCoder-9B.Q4_K_M.gguf \
  --codec results/ncm-contextual-decoder-v9/best.pt \
  --checkpoint /path/to/immutable-initialization.pt \
  --data results/ncm-expanded-training-v19 \
  --output results/ncm-runtime-audit-v19.json \
  --keep 0 --shared --lengths 64,69,512,2048 --repeats 3
```

The reference and shared-prefix implementations must agree within the existing
logit and gradient tolerances, including separate adapter, classifier, and
interface gradient groups. Do not relax tolerances to make a new checkpoint
pass. If the shared implementation fails, the trainer supports
`--no-shared-prefix-training` using the qualified reference records.

The audit measures fixed-weight forward/backward execution, not optimizer
updates or coding quality. Requalify after changing the backbone or relevant
runtime code. A v18 report cannot be relabeled as an audit of a new dataset.

## 5. Train with gradient accumulation

First run a separate two-update smoke stage, then inspect finite gradients,
memory headroom, checkpoint serialization, and evaluation completion. A smoke
stage is not the long run or a coding acceptance result.

```bash
python -m scripts.ncm.train_accumulated \
  --source /path/to/OxCoder-9B.Q4_K_M.gguf \
  --codec results/ncm-contextual-decoder-v9/best.pt \
  --init /path/to/immutable-initialization.pt \
  --data results/ncm-expanded-training-v19 \
  --output results/ncm-accumulated-smoke-v19 \
  --runtime-audit results/ncm-runtime-audit-v19.json \
  --data-expansion --steps 2 --accumulation 4 --eval-every 2
```

After the smoke succeeds, start the planned learning stage from the immutable
initialization, in a new directory:

```bash
python -m scripts.ncm.train_accumulated \
  --source /path/to/OxCoder-9B.Q4_K_M.gguf \
  --codec results/ncm-contextual-decoder-v9/best.pt \
  --init /path/to/immutable-initialization.pt \
  --data results/ncm-expanded-training-v19 \
  --output results/ncm-accumulated-v19 \
  --runtime-audit results/ncm-runtime-audit-v19.json \
  --data-expansion --steps 5000 --accumulation 4 \
  --head-lr 0.0003 --body-lr 0.00003 \
  --corruption 0.5 --eval-every 250 --save-every 10
```

Each update samples four independent training examples. For each example,
the loss is the mean cross-entropy over the four concept levels plus 0.1 times
the final-level termination BCE. Each example's loss is divided by four before
backpropagation. Clipping and the optimizer step occur only after all four
examples. Thus accumulation averages gradients; it does not multiply the
learning rate or retain four full activation graphs.

Muon updates the residual-adapter and input-projection matrices. AdamW updates
classifiers, termination, and other interfaces. The new stage retains model
weights and explicitly resets optimizer state because its data and effective
batch change. An exact interrupted-run resume uses the same command plus
`--resume`, restoring model, optimizer, and RNG state. Configuration or source
changes are rejected rather than silently treated as a resume.

Full prompt gradients are retained. Shared-prefix computation applies only to
eligible prompts of at least 64 tokens; shorter prompts use the reference
four-pass path. Inference and the coding checker are unchanged.

## 6. Inspect evidence and enforce acceptance

`progress.json` contains the current update, consumed-example count and recent
history. Full history remains in `latest.pt` and the JSONL stdout log. Do not
rewrite or replace old evaluation reports when advancing to another stage.

At each evaluation checkpoint, inspect:

1. `chunks-N.json`: teacher-forced concept accuracy, actual generated text,
   UTF-8/EOS validity, and termination behavior;
2. `coding-N.json`: the unchanged 12-task executable development diagnostic;
3. the matching checkpoint SHA256 and preserved evaluated snapshot;
4. dataset and runtime identities, plus memory and nonfinite-gradient failures.

Lower training loss and faster updates do not satisfy the 12/12 gate. A
teacher-forced reconstruction, original AR-model score, or small diagnostic
probe cannot substitute for actual student generation. A 12/12 result would
still need serialization/reload parity and broader independent qualification
before being described as professional or deployment-ready.

## Diagnostic controls

`audit_residual_repair.py` tested 128 training windows with the selected frozen
codec. Gold codes reconstructed all 128 exactly. Replacing earlier codes under
the current corruption schedule, while retaining clean later labels, preserved
122 exact windows and 99.8962% mean non-padding accuracy. An oracle that
re-quantized residuals after corruption recovered all 128. This oracle uses
the known training text and is not a permitted inference path. The experiment
does not support treating earlier-code corruption alone as the main reason for
the generator's poor text output.

`audit_latent_predictability.py` fits a small final-level classifier from gold
earlier-level latents, without the prompt or OxCoder backbone. Training uses
4,096 training windows; scoring uses 256 distinct validation windows. It is an
interface diagnostic, not a replacement model. Its outputs must never be
counted as free generation or coding-task passes.

## Resource isolation

The local preparation/teacher jobs run in separate bounded user services, with
swap disabled for those services. The CPU teacher has an 8 GiB cgroup limit;
distillation has 5 GiB; acquisition has 4 GiB. These limits protect concurrent
training, and do not describe the combined machine memory footprint. GPU
diagnostics additionally reserve 3 GiB and cap their own PyTorch allocation.

Do not launch another full-backbone GPU process while the training service
already occupies both devices. Wait for a saved checkpoint and release the
owned training process before running a full runtime audit or GPU teacher.
Unrelated inference services must remain untouched. Record server launch
settings if changing the offline teacher's execution backend.

## Validation

The CPU suite passed 98 tests after adding this stage's acquisition,
distillation, provenance, and accumulated-loss tests. The gradient control
compares accumulated gradients with an equal-example batched objective in
float64. It covers the termination term and unequal level lengths. These are
implementation checks, not evidence that the student passed the coding gate.

Run CPU tests without the optional GPU-kernel import directory:

```bash
PYTHONPATH=results/ncm-deps:results/ncm-data-deps \
  python -m unittest discover -s tests/ncm
```

## Sequential stage coordinator

`run_expanded_stage.py` coordinates the prepared v19 experiment on a Linux
workstation with a user systemd manager and two CUDA devices. It is deliberately
specific to the recorded v18/v19 artifact paths and unit names. It is not a
general installer, a Mac runtime, or an unbounded claim that any training run
will converge. Supply absolute paths for the repository, training interpreter,
llama.cpp server, and source GGUF:

```bash
python3 -S -m scripts.ncm.run_expanded_stage \
  --root /absolute/path/to/OxCoder-9B \
  --python /absolute/path/to/training-environment/bin/python \
  --server /absolute/path/to/llama-server \
  --source /absolute/path/to/OxCoder-9B.Q4_K_M.gguf
```

The standard-library coordinator uses `-S` to avoid unrelated site-package
modules named `scripts`. Child training jobs use the explicitly supplied
training interpreter and the repository's pinned dependency directories.
Run the coordinator in a durable session or a separate user service. Do not
start two copies against the same output directory or experiment unit names.

Its sequence is:

1. Wait for the existing `ncm-independent-v18.service` to finish. Require its
   complete 2,000-update report and preserve an initialization checkpoint whose
   hash matches the final coding evaluation.
2. Stop only the two named v19 CPU distillation services. Require at least
   9 GiB of free GPU-1 memory, then start the same offline teacher checkpoint
   on GPU 1. Resume the collector from its immutable request cache. Record the
   new server command; CPU/GPU arithmetic need not be bit-identical, and the
   source checkpoint identity does not imply identical backend execution.
3. Finish all selected teacher requests and require a completed admission
   manifest. Stop the GPU teacher before loading the student.
4. Combine the pinned OpenCodeInstruct admission and teacher-response admission
   with the existing prepared corpus. Retain the frozen evaluation inventory.
5. Audit the four-level forward/backward implementation at 64, 69, 512, and
   2,048 prompt tokens. Prefer shared-prefix execution only when all four
   measurements pass. Otherwise require all four reference measurements to
   pass. The trainer independently rechecks source hashes and the numerical
   thresholds; the coordinator cannot override that gate.
6. Run a two-update smoke test with four accumulated examples per update. Then
   launch the separately initialized 5,000-update learning stage with the same
   accumulation. The smoke test's optimizer updates are not silently carried
   into the full stage.
7. Preserve evaluated student checkpoints by checksum. If a complete 12/12
   development result appears, stop the owned training service, reload that
   immutable checkpoint in a new process, and rerun the unchanged evaluator.
   Record acceptance only if the replay is also 12/12 for the same checksum.

`results/ncm-pipeline-v19/progress.json` records the current phase; launch
manifests record every command and resource limit. Student and teacher jobs
have distinct units and logs. The coordinator never stops unrelated services,
uses a teacher response during student evaluation, or changes evaluation
answers. A finite stage ending without acceptance is explicitly marked
`stage_complete_requires_next_experiment`; it is not a release. A failed
reload likewise does not become acceptance. The orchestrator's five unit tests
exercise incomplete audits, reference fallback, immutable snapshots, partial
evaluations, and changed checkpoints.
