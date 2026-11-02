# Complete-response corpus and decoded-text supervision

This stage retains the frozen OxCoder prompt encoder and native multiscale
concept decoder from [version 25](CONTEXT-DECODER-V25.md). It expands the
training corpus and compares two training objectives. It does not constitute a
qualified NCM checkpoint. Acceptance remains all twelve unchanged executable
tasks followed by identical answers from a fresh process.

## Evidence motivating the stage

The original 4,094-chunk pilot overfit. Baseline snapshots at updates 500,
1,000, 1,500 and 2,000 and the adaptive-residual continuation at update 1,500
all passed **0/12** coding tasks. Baseline training stopped at update 2,000;
the adaptive branch stopped at a saved boundary. Those checkpoints and reports
are preserved. Lower training loss did not establish useful generation.

`audit_context_coverage.py` found that only 17 of 8,192 final-level validation
code positions used labels absent from the training pilot. No validation code
was unseen in the earlier levels. Thus, missing class labels alone cannot
explain the failure. This diagnostic does not establish adequate semantic
coverage, nor does it make validation examples eligible for training.

## Corpus construction and leakage boundary

`complete_response_selection.py` uses the existing prepared training corpus:

- Creator sources: `TeichAI/Ox-Alpha-10k` and
  `TeichAI/DeepSeek-v4-Pro-Agent`, with their pinned revisions retained in the
  preparation manifests.
- Python supplement: `nvidia/OpenCodeInstruct`, revision
  `8f3ba5bafe4d6e8db46082cf7ae6741bc370604d`.
- Cached OxCoder teacher responses admitted by the existing publisher-test
  checks. The earlier verification stage admitted 188 complete responses;
  selection here additionally imposes the response-length limit.

The selector keeps complete responses of at most eight contiguous chunks. Each
chunk contains at most 127 UTF-8 bytes followed by EOS. When an eligible checked
teacher response exists for a conversation, it takes precedence over that
conversation's original answer. Existing malformed-teacher-action filtering
remains enabled. Eligibility never depends on the coding exam's answers or on
student predictions on that exam.

The selection contains 45,512 training chunks from 8,470 complete responses,
including 912 distilled chunks. Source counts are 1,580 Agent, 1,549 Alpha and
42,383 OpenCodeInstruct chunks. These counts are **before codec screening**.
The same 64 validation windows are copied unchanged from the earlier frozen
selection. Conversations cannot cross splits.

Selection checksum:

```
3e8927cfe7dcbbde0fd13e50be5e95515e55d960e8076f052d5abb0ac839f124
```

The source encoder receives the request and, for later training chunks, earlier
response chunks. It never receives the current target chunk or its concept
codes. Standalone generation supplies its own earlier generated chunks instead.
This remaining training/inference difference is not eliminated by the
self-conditioning of concept levels within a chunk.

## Reusable source features

`extract_context_reuse.py` checks source, codec, selection and implementation
identities. An old feature is reused only after checksum verification and exact
comparison of its training row and conditioning record. Qualified payloads are
hardlinked, preserving bytes without duplicating storage. All validation
features must come from the earlier qualified cache.

New features use all 32 original frozen OxCoder layers, with no adapters.
They are BF16 prompt states of width 4,096, capped at 2,048 prompt tokens.
Current-target codec reconstruction must be exact at all non-padding positions
for a new training chunk to be admitted. After shard merge,
`complete_context_cache.py` removes an **entire training response** if any of its
selected chunks failed that check. It preserves every validation record and
records the original merged manifest as provenance. Final training counts must
be read from this derived manifest, not inferred from the selection counts.

Storage before reuse is approximately 101.3 GiB. This is an offline training
feature cache; it is not inference-time SSD expert offloading. Each extraction
worker checks disk headroom and saves resumable progress. Resume requires
unchanged identities and valid payload checksums. Do not edit the extractor or
its hashed dependencies while extraction is running.

### Numerical qualification of extraction changes

`audit_feature_batching.py` compared sixteen fixed validation prompts with the
original cache and decoder. The gate required relative feature L2 error at most
0.001, identical generated codes and bytes, and stop-probability difference at
most 0.00001.

| Prompt batch | Source-encoding time | Exact generated code sequences | Result |
|---|---:|---:|---|
| 1 | 9.718 s | 16/16 | Passed |
| 2 | 8.923 s | 2/16 | Rejected; maximum relative L2 0.03295 |
| 4 | 24.141 s | 2/16 | Rejected; maximum relative L2 0.02986 |

Only batch one is used. These timings measure source encoding for this audit;
they are neither generation throughput nor end-to-end training speed.

`audit_encoder_graph.py` separately captured exact-length CUDA graphs without
padding or batching changes. Four original prompts and four changed inputs
matched eager features bit for bit. Replay medians were only approximately
1.01–1.03 times faster, while graph capture cost 0.52–0.82 seconds per shape.
This did not justify replacing the running extraction path. Graph replay is not
enabled in the corpus extractor.

## Matched training arms

Both arms initialize **model weights only** from the immutable version 25
update-500 snapshot:

```
2635052a085f45780e1fd5e17066b974d5c3584a45fdd719ee1d5abe02c5abb8
```

Both start a fresh optimizer, shuffled sampler, RNG sequence and learning-rate
schedule for the changed corpus. This is not an optimizer-preserving resume
from version 25. Within a new arm, ordinary interruption/resume does restore
all those states exactly.

Shared settings:

| Setting | Value |
|---|---|
| Decoder | 129,306,113 trainable parameters; width 768, 12 blocks |
| Concept levels | 4, 16, 64, 128 positions; 4,096 codes per book |
| Source encoder and codec | Frozen |
| Global batch / microbatch | 32 / 8 |
| Planned updates | 10,000 |
| Peak learning rate / warmup | 0.0002 / 100 updates |
| Optimizer | Muon hidden matrices; AdamW other parameters |
| Parameter and optimizer precision | FP32; BF16 model autocast |
| Self-conditioned residual probability | 0.5 per example |
| Training dropout | 0.1 on attention and feed-forward outputs |
| Seed | 28001 |
| Saves / immutable evaluated snapshots | Every 100 / 500 updates |

Dropout is disabled for detached student rollouts, validation and inference.
It adds no checkpoint parameters and changes no inference computation. The
adaptive correction labels follow [version 26](ADAPTIVE-RESIDUAL-V26.md).

The sole planned difference between arms is the decoded-text auxiliary weight:
zero for `ncm-context-control-v28`, and 0.1 for `ncm-context-decoded-v28`.

### Decoded-text auxiliary objective

For final-level logits `p`, frozen last codebook `E`, and the accumulated
previous-level latent `z`, the training-only operation is:

```
hard = E[argmax(p)]
soft = softmax(p) @ E
selected = detach(hard) + (soft - detach(soft))
byte_logits = frozen_codec_decoder(detach(z) + selected)
auxiliary_loss = cross_entropy(byte_logits, target_bytes, ignore=PAD)
```

The forward uses discrete predicted codes. The backward uses a
straight-through surrogate, which is not the exact derivative of argmax.
Gradients reach the concept logits, not the frozen codec or previous-level
inputs. EOS is supervised; PAD is excluded. There is no new byte predictor or
autoregressive answer path at inference.

The three-update CUDA smoke used eight long training prompts, including 2,048
prompt positions. It produced finite losses and gradients, no codec gradients,
and a peak PyTorch allocation of 2,541,813,248 bytes. This is an objective and
memory check, not a quality result or whole-process 8 GB deployment measurement.
The pipeline binds the smoke report to hashes of its implementation files.

## Running and resuming

Use the pinned environment described in the version 25 document. From the
repository root, create the selection in a new directory:

```bash
python -m scripts.ncm.complete_response_selection \
  --data results/ncm-curated-training-v23 \
  --json-cache results/ncm-prepared-json-v24 \
  --validation results/ncm-validation-selection-v24 \
  --output results/ncm-complete-selection-v27
```

Launch one extraction worker on each selected GPU, changing both `--shard` and
`--device` together. The audit report must qualify the selected batch size:

```bash
python -m scripts.ncm.extract_context_reuse \
  --source /path/to/OxCoder-9B.Q4_K_M.gguf \
  --codec results/ncm-contextual-decoder-v9/best.pt \
  --selection results/ncm-complete-selection-v27 \
  --reuse results/ncm-context-pipeline-v25/features.json \
  --audit results/ncm-feature-batch-audit-v27/report.json \
  --output results/ncm-complete-features-v27-0 \
  --device cuda:0 --shard 0 --shards 2 --batch 1
```

The qualified single-worker path is also available with `--shards 1`. Append
`--resume` only with the original inputs, shard assignment and implementation.
Never merge partially complete shards.

`run_complete_context_stage.py` is the concrete two-GPU systemd orchestration
for the recorded v27/v28 experiment. It waits for the specifically named
extraction services, verifies complete manifests, merges and qualifies the
corpus, starts the two arms, then invokes the serial evaluator. Its service
names, output directories and environment paths are experiment-specific; inspect
them before deploying on another workstation. The standalone trainer accepts
explicit paths and does not require systemd:

```bash
python -m scripts.ncm.train_context_corpus \
  --cache results/ncm-complete-pipeline-v28/features.json \
  --codec results/ncm-contextual-decoder-v9/best.pt \
  --init results/ncm-context-training-v25/evaluated-500.pt \
  --output results/ncm-context-decoded-v28 \
  --device cuda:1 --byte-loss 0.1 --steps 10000
```

For the matched control, change output directory, selected device and
`--byte-loss 0`; retain all other settings. Training services use a 12 GiB
host-memory limit and no swap. These are training service limits, not a claim
that combined source-plus-decoder inference fits the deployment allowance.

## Validation and acceptance

Unit tests cover complete-response selection, split isolation, removal of
partially rejected responses, feature batching, training-only dropout,
straight-through gradients, and exact interrupted optimizer/sampler/RNG
continuation. The 204-test NCM suite passed before this stage was launched.

Every 500 updates, canonical teacher-forced metrics and free chunk decoding are
recorded on the same 64 windows. These diagnostics include gold prior chunks
in some prompts and therefore do not establish complete-answer correctness.

`monitor_context_comparison.py` evaluates immutable checkpoints with fresh
source-model inference and the unchanged twelve-task executable evaluator.
It checks report completeness, checkpoint identity, all task counts, and absence
of autoregressive fallback. A first 12/12 triggers another fresh-process run;
identical generated answers are required before `acceptance.json` is written.
No training loss, oracle reconstruction, speed audit, or service-completion
status substitutes for that result. Even a successful gate covers twelve
elementary functions, not general professional coding competence.
