# OxCoder9B: constrained llama-server speed sweep

Measured on 2026-11-04: one RTX 5060 Ti, Linux, original Q4_K_M checkpoint,
llama.cpp commit `002a12ad25503a93501b2e188c360029830a241a`.
These measurements do not predict M3 Metal throughput.

| Configuration | Median output tokens/s, 3 requests | Sampled combined memory peak, GiB |
|---|---:|---:|
| Flash Attention off | 67.43 | 6.453 |
| Flash Attention on | 68.10 | 6.204 |
| Flash Attention on, microbatch 64 | 68.02 | 6.190 |
| Prompt lookup, n=3, draft length=16 | 60.68 | See raw report |

Each request generated up to 256 tokens, temperature zero, seed 123, no prompt
cache reuse, one slot, 4096 configured context. The prompt was a short merge-sort
request. This is a short-context throughput measurement, not a populated 4K or
128K context result. Generation included extra prose following the first code
block. Syntax checks are recorded; functional correctness and chat-template
behavior require a further workload evaluation. Do not describe this as a
professional coding benchmark or as a 12-task NCM result.

## Memory and loading

The driver samples process RSS plus per-process CUDA allocation every 250 ms
and terminates the server if their sum exceeds 8 GiB. A systemd cgroup separately
limits host memory to 8 GiB and disables host swap. The combined measurement is
sampled, not a hardware-enforced combined CUDA/host allocation limit; brief
unsampled peaks are possible. Driver memory is outside the server RSS sum.

Default mmap loading exceeded the sampled combined budget at roughly 11.2 GB
before readiness. All those attempts were rejected. Using `--load-mode none
--no-host` avoided the observed loading duplication and completed the sweep.
Weights remain resident during generation; there is no SSD weight offloading.
Host RAM and GPU memory are separate in this Linux test, unlike Apple unified
memory. The loading result must therefore be remeasured on Metal.

## Reproduction

Run from the repository root with a CUDA llama-server build exposing the flags
below. The driver currently selects physical CUDA device 1; adjust
`CUDA_VISIBLE_DEVICES` in the script for a different test machine.

```bash
python3 -S scripts/inference/sweep.py \
  --server /absolute/path/to/llama-server \
  --model /absolute/path/to/OxCoder-9B.Q4_K_M.gguf \
  --output /absolute/path/to/new-results
```

The measured common server options are `--load-mode none --no-host -ngl 99
--split-mode none -c 4096 -np 1 -b 256 -ub 128 -t 4`. Configurations are sequential;
existing report files are skipped, so use a new output directory after changing
the implementation. Every report contains the complete command and outputs.
The source checkpoint is the existing 5,629,117,536-byte Q4_K_M artifact with
SHA-256 `a171c091c67c3547c45ec58f1b85a888d73725315fde71bc929ff9f8c255bb6f`.

Raw reports: [evaluation/oxcoder-8g-cuda](evaluation/oxcoder-8g-cuda).

## Research decisions

[Strata](https://github.com/ro99/strata) targets MoE expert movement and residency.
OxCoder9B is dense, so expert caching cannot directly accelerate its layers.
[llama.cpp speculative decoding](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
offers prompt lookup without an additional drafter. It was slower in this measured
short generation workload. Larger gains on copying/editing workloads must be
measured separately. External DFlash/DSpark or MTP require checkpoint/backend
compatibility and memory qualification; support for base Qwen is insufficient
proof of compatible OxCoder weights. No speculative speedup is assumed here.
