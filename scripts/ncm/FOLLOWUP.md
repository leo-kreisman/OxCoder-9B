# Combined data, distillation and optimizer experiment

This continues [the initial failed conversion](RESULTS.md). It targets
architectural adaptation of OxCoder into a next-concept generator. It does not
train a speculative drafter, change the deployed GGUF, or produce a Mac runtime.
All measurements are local Linux/CUDA experiments. The 8,000,000,000-byte
deployment limit remains a separate, unpassed requirement.

Measured outcomes are in [FOLLOWUP-RESULTS.md](FOLLOWUP-RESULTS.md).

## Corpus and provenance

The original GGUF, inverse transforms and autoregressive audit remain as
documented in [README.md](README.md). The new corpus reads 1,000 complete Alpha
conversations and 64 agent trace files at these revisions:

| Source | Revision |
|---|---|
| `TeichAI/Ox-Alpha-10k` | `ae678ee78880e14251a553a0d0dc8ec6bd6dad6c` |
| `TeichAI/DeepSeek-v4-Pro-Agent` | `abbf7b71b145633b60beb21ddc0d158582fd80dd` |

The corpus contains 1,216 complete assistant responses: 1,000 Alpha and 216 agent
responses; 939 train, 122 validation, 155 test. These are bounded subsets, not
either entire dataset. At most four assistant turns are spread across each
agent trace. Tool records remain text, never commands executed by the preparer.
Eight oversized examples were rejected. Complete targets are retained without
silent truncation. Conversation-content hashes preserve the earlier split rule;
complete targets are deduplicated. Source revisions, file hashes, selections and
the final JSONL hash are recorded. These demonstrations are not certified correct
programs or a reconstruction of OxCoder's original training mixture.

```bash
python -m scripts.ncm.corpus --output results/ncm-corpus-v2 \
  --alpha-rows 1000 --agent-files 64
```

Use the isolated environment and pinned requirements in the main guide. The
follow-up additionally requires Hydra/OmegaConf, now in `requirements.txt`.
Commands assume the repository root and an activated environment. `results/`
contains ignored local artifacts, not uploaded release assets.

## Official codec audit

Source: [HazyResearch/ncm](https://github.com/HazyResearch/ncm), commit
`c70274273f489abc73832b2b98584a8483b6b203` (MIT).
Weights: [hazyresearch/ncm-checkpoints](https://huggingface.co/hazyresearch/ncm-checkpoints),
revision `d342ce8aaea58830eef3b8ad473664673e41f692`.

`fetch_official.py` downloads only XSum/WMT14 codecs and configurations, checks
pinned SHA256 hashes and checks out the reviewed source revision. It refuses
mismatched existing files/checkouts. The wrapper refuses a dirty upstream tree.
Checkpoints use `weights_only=True`; no upstream pickle datasets are loaded.
Upstream Python is imported as code: review any replacement revision first.

```bash
python -m scripts.ncm.fetch_official
python -m scripts.ncm.official_codec --artifacts results/ncm-official/xsum \
  --data results/ncm-corpus-v2 --output results/ncm-official/xsum/screen.json
python -m scripts.ncm.official_codec --artifacts results/ncm-official/wmt14 \
  --data results/ncm-corpus-v2 --output results/ncm-official/wmt14/screen.json
python -m scripts.ncm.audit_official --artifacts results/ncm-official/xsum \
  --output results/ncm-official/xsum/scheduler-audit.json
```

| Codec | Pre-tokenizer | Window | Concept levels |
|---|---|---:|---|
| XSum | GPT-2 | 48 tokens | 4, 6, 10, 14, 20, 26, 32, 40, 48 |
| WMT14 | Qwen3-0.6B | 64 tokens | 1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64 |

Both have width 384 and 4,096 entries per codebook. GPT-2 tokenizer revision:
`607a30d783dfa663caf39e06633721c8d4cfcd7e`; Qwen tokenizer revision:
`c1899de289a04d12100db370d81485cdf75e47ca`. LM1B was rejected because its
BERT-uncased tokenizer normalizes case/whitespace. The actual release inventory
did not contain the OpenWebText codec mentioned in the repository; this
experiment does not assume it is downloadable.

Evaluation splits whole responses into fixed windows, pads only the final one,
reconstructs every window, concatenates token IDs and decodes once. Comparison
is byte-exact. Predictions after the known document length do not count: this
tests reconstruction, not a generator's ability to terminate. Six probes cover
identifiers, whitespace, operators, JSON, numeric strings, Unicode and lookalike
hash characters. A GPT-2 warning about sequences longer than 1,024 is harmless:
the codec receives only 48-token windows, never a full document as one input.

The screen requires every complete response and probe to match exactly, plus
99.99% token accuracy. Even passing would require a larger untouched evaluation
and executable coding tests. No follow-up test responses have been scored.

The scheduler audit supplies target codes as an oracle. It checks all inputs
against upstream `idx_to_ncm_input`, verifies current/future codes cannot affect
earlier inputs, and checks sequential residual assembly against reconstruction.
It passed with zero input difference. This verifies plumbing, not generation.

## Cached OxCoder response distillation

Start a separate teacher server using the unchanged, SHA-verified Q4_K_M model.
Choose an available port/GPU; do not stop or reuse unrelated inference services.
GPU index 1 reflects the measured two-GPU workstation, not a hardware requirement.

```bash
export OXCODER_GGUF=/absolute/path/to/OxCoder-9B.Q4_K_M.gguf
CUDA_VISIBLE_DEVICES=1 /path/to/llama-server \
  -m "$OXCODER_GGUF" --host 127.0.0.1 --port 18109 \
  -c 4096 -np 1 -ngl 99 -fa on --reasoning off --alias oxcoder-ncm-teacher
# In another terminal, after the server is ready:
python -m scripts.ncm.distill --data results/ncm-corpus-v2 \
  --output results/ncm-distill-v2 --teacher-file "$OXCODER_GGUF" \
  --tokenizer results/tokenizer --per-source 8 --max-tokens 768
```

`results/tokenizer` is the pinned original OxCoder tokenizer from the backbone
workflow. Teacher weights remain resident; this is not SSD offloading. The
collector picks distinct training conversations with the shortest eligible
complete prompts. Oversized prompts are rejected, not truncated. This favors
simple prompts and is not a representative benchmark. Non-chat roles become
user messages marked as recorded output. No environment or tools execute.

Responses retain source/group/split, request identity, finish reason, timing and
teacher identity. Reruns reuse matching cached requests. Only normally terminated,
nonempty text is included; length-truncated and tool-call responses are excluded.
The pilot retained eight of sixteen attempts: three agent and five Alpha.
They are unverified teacher demonstrations, not corrected ground truth or
successful task executions. This is response-level distillation, without teacher
logits or hidden-state matching. Stop only this experiment's teacher when done;
the actual run did so before training and left application servers running.

## Frozen-code decoder adaptation and optimizer comparison

```bash
python -m scripts.ncm.adapt_codec cache \
  --artifacts results/ncm-official/xsum --data results/ncm-corpus-v2 \
  --teacher results/ncm-distill-v2 --cache results/ncm-official/xsum/coding-cache.pt
bash scripts/ncm/run_limited.sh python -m scripts.ncm.adapt_codec train \
  --artifacts results/ncm-official/xsum --data results/ncm-corpus-v2 \
  --cache results/ncm-official/xsum/coding-cache.pt \
  --output results/ncm-codec-adamw-v2 --optimizer adamw --steps 1000 --eval-every 250
bash scripts/ncm/run_limited.sh python -m scripts.ncm.adapt_codec train \
  --artifacts results/ncm-official/xsum --data results/ncm-corpus-v2 \
  --cache results/ncm-official/xsum/coding-cache.pt \
  --output results/ncm-codec-anvil-v2 --optimizer anvil --steps 1000 --eval-every 250
```

The cache contains 7,229 windows / 325,657 non-padding tokens: 6,061 Alpha-source
and 1,168 agent-source windows, including 49 teacher windows. At most eight
windows are spread over each response. The teacher share is small; this pilot
cannot isolate its benefit. Validation/test windows never enter training.

Both arms use identical initialization, seed, sampled batches, batch size 8 and
1,000 updates. Only the decoder trains. Encoder, quantizer and residual transforms
stay frozen, preserving cache validity. Dropout is disabled. Parameters/state
use FP32; forward operations use BF16 autocast. Loss ignores padding; evaluation
uses FP32. New directories are mandatory unless `--resume` restores matching
optimizer/RNG states. AdamW uses 1e-4, ANVIL matrices 0.002. This is one controlled
configuration, not a learning-rate sweep.

The portable optimizer is **ANVIL2-derived**, from
[modded-nanogpt PR 360](https://github.com/KellerJordan/modded-nanogpt/pull/360),
commit `4f5270e5753b432a10d012bef3b2c4ca2d933707`; see
[its MIT notice](ANVIL-LICENSE.txt). It retains twin velocity rails, six spectral
maps, per-lane normalization and sign-aligned decay. Vocabulary heads, embeddings
and non-matrices keep AdamW. Differences include ordinary PyTorch matmuls, FP32
spectral math/master weights, fixed rail coefficients from step one, no
distributed banks, custom Triton or captured CUDA graphs. It does not reproduce
the full eight-H100 recipe.

The cited **39.914 seconds was training on eight H100s**, not inference latency,
NCM conversion time or a promise for this machine. That trainer also changes
architecture/data and includes a large hashed table. Its asserted eight-process
setup is not imported here. `training_seconds` measures synchronized
forward/backward/update work only, excluding downloads/loading/evaluation/saving.
It is neither end-to-end conversion time nor output tokens/s. Choose methods by
held-out quality at a fixed total budget, not the update counter alone.

## Joint adaptation control

```bash
bash scripts/ncm/run_limited.sh python -m scripts.ncm.adapt_codec train \
  --artifacts results/ncm-official/xsum --data results/ncm-corpus-v2 \
  --cache results/ncm-official/xsum/coding-cache.pt \
  --output results/ncm-codec-joint-v2 --optimizer adamw --joint \
  --steps 2000 --eval-every 500
```

This starts from the released codec, not a decoder-adapted checkpoint. It trains
encoder/decoder/residual transforms; books use upstream EMA. Codes are recomputed
from cached input tokens each batch: old code IDs are not targets after encoder
changes. Dropout/fine-level dropping are disabled. Loss is CE plus upstream VQ
loss. Snapshots include full codec/EMA state. Generator caches must be rebuilt
after selecting the final encoder, quantizer and residual transforms.

## Expanded byte-codec controls

```bash
python -m scripts.ncm.byte_followup --data results/ncm-corpus-v2 \
  --teacher results/ncm-distill-v2 --output results/ncm-byte-data-v3
bash scripts/ncm/run_limited.sh python -m scripts.ncm.train codec \
  --data results/ncm-byte-data-v3 --output results/ncm-byte-codec-v3 \
  --devices cuda:1 --steps 10000 --eval-every 1000 --batch 16 --lr 0.0003
bash scripts/ncm/run_limited.sh python -m scripts.ncm.train codec \
  --data results/ncm-byte-data-v3 --output results/ncm-byte-tokenwise-v3 \
  --devices cuda:1 --steps 10000 --eval-every 1000 --batch 16 --lr 0.0003 --layers 0
# Continue the promising architecture to 30,000 TOTAL updates:
bash scripts/ncm/run_limited.sh python -m scripts.ncm.train codec \
  --data results/ncm-byte-data-v3 --output results/ncm-byte-tokenwise-v3 \
  --devices cuda:1 --steps 30000 --eval-every 2000 --batch 16 --lr 0.0003 --layers 0 --resume
python -m scripts.ncm.evaluate_documents \
  --checkpoint results/ncm-byte-tokenwise-v3/best.pt --data results/ncm-corpus-v2 \
  --output results/ncm-byte-tokenwise-v3/documents.json
```

This corpus creates explicit UTF-8-safe 127-byte chunks plus EOS, spreading at
most 16 training chunks per response. Evaluation chunks take precedence in
cross-split exact-target deduplication. It has 14,002 training chunks (72 teacher),
7,118 validation and 8,638 test chunks. Test content is used only for duplicate
exclusion, never loss/selection/performance. Both controls start fresh.

`--layers 0` removes contextual attention from the codec but retains learned byte
embeddings, normalization, positions, five residual VQ levels and an output
classifier. This is a reconstruction ablation, not an official NCM checkpoint or
an unreported lossless side channel. Codebooks still learn; tests check gradients.
Whole-response evaluation joins decoded chunks with strict UTF-8/EOS checks.
Chunk exactness cannot authorize a professional coding claim. Reconstruction
knows chunk boundaries; free-running generation is harder.

After the tokenwise run, a separate capacity control starts from fresh weights
with 4,096 rather than 512 entries per book. Keep the same corpus/seed/lr/batch
and use `--layers 0 --codes 4096 --steps 10000 --eval-every 2000` in a new
`results/ncm-byte-widebook-v3` directory. The 512-entry run nearly filled its
finest books, but occupancy alone does not establish that extra capacity helps.
The recorded byte confusion audit checks actual errors rather than inferring
their cause from the loss:

```bash
python -m scripts.ncm.audit_byte_errors \
  --checkpoint results/ncm-byte-tokenwise-v3/best.pt --data results/ncm-byte-data-v3 \
  --output results/ncm-byte-tokenwise-v3/errors.json
```

## Promotion criteria

1. Preserve the original runtime and failed experiment records.
2. Pass byte-exact complete-response/probe reconstruction; freeze the codec,
   then evaluate untouched complete programs once.
3. Rebuild concept targets after changing encoder/books/residual transforms.
4. Then train OxCoder adapters/interfaces on creator demonstrations and verified
   teacher outputs. The original trainer supports the byte codec. The official
   wrapper adds an audited residual scheduler, but a full official-codec generator
   training/data/termination path has not been validated here.
5. Require generated-code execution, hidden tests, tool protocol correctness,
   long-answer termination and comparisons with unchanged OxCoder. Codec
   reconstruction, oracle scheduling and nonzero gradients are insufficient.
6. Measure useful tokens/s, whole-request latency and total memory only after
   quality passes. Native packed 4-bit NCM inference and Metal remain separate work.

A fast optimizer is not a reason to run a generator against corrupted targets.
Failure calls for changing representation/data coverage, not weakening the gate.
No checkpoint from this experiment is a production replacement for OxCoder.
