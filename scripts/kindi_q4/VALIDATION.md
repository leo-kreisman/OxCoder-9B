# Native Kindi Q4 validation — 2026-10-25

## Test environment

Linux x86-64, Intel i7-8700K,64GB physical system RAM, Kingston NVMe storage. GPU tests used one otherwise idle RTX 5060 Ti 16GB (physical GPU1); the other GPU and existing applications were left running. The GPU's full 16GB was **not** treated as the inference budget. Guarded CUDA runs had a 1.7GB host cgroup cap, swap disabled and a monitor stopping the job when host cgroup plus physical GPU usage exceeded 8,000,000,000 bytes. CPU tests had a hard 8GB host cap. Two CPU cores of quota and low CPU/I/O priority were used.

The CUDA runs are **not M3 measurements**. Dedicated GPU memory and PCIe transfers differ from Apple unified memory. No comparison with a user's17tokens/s Mac result is valid without running both implementations on that Mac at identical settings.

## Weight conversion and numerical checks

- Source: `OxCoder-9B.Q4_K_M.gguf`,5,629,117,536 bytes.
- Source SHA256: `a171c091c67c3547c45ec58f1b85a888d73725315fde71bc929ff9f8c255bb6f`.
- Packed quantized weights:5,610,356,736 bytes.
- Packed weights SHA256: `78ca4abe91895cc5e52e57e78d2cda1641e72e84f8cf7d1a7a6400b2aa49246f`.
- All 427 tensor records converted and validated. Original Q4_K/Q6_K blocks survive a reverse-layout byte comparison; small recurrent gates become exact float32 values.
- Independent GGML and NumPy block decoders agree exactly. Native matrix products agree within the documented numerical tolerance.
- Full-model CPU and CUDA greedy verification:16/16 code-continuation tokens and17/17 chat tokens, including EOS, match the independently generated llama.cpp fixture. This is limited reference coverage, not an assertion of universal token identity.
- An early wrong-output prototype exposed Bonsai's incompatible recurrent-head permutation. It was fixed before accepting throughput results. That prototype's timing is excluded.

## Measured performance

All values below are short-context, batch-one measurements. The resident64 test starts with 9 actual prompt tokens and emits 64 tokens at a 512-token capacity. The SSD ablation uses the same one-token `def` prompt and four output tokens in both modes.

| Execution | Decode tokens/s | Time to first token | Accounted memory peak | Evidence |
|---|---:|---:|---:|---|
| Native CPU AVX2,2 threads,16 outputs |1.10|7.57s|See separate CPU verification resource report; original probe lacked physical-memory sampling|`native-code-avx.json`|
| Native CUDA resident,64 outputs |25.66|4.16s|6.79GB|`resident64.json`, `resident64-resources.json`|
| Native CUDA resident,4-output ablation |25.71|3.91s|6.99GB|`resident-short.json`|
| Native CUDA1.6GB tensor cache, protected embedding/head |0.37|3.90s|3.07GB|`stream-pinned.json`, `stream-pinned2-resources.json`|
| Native CUDA1.6GB cache, original unprotected LRU |0.27|4.01s|RSS+CUDA-pool sample2.04GB; no physical-GPU monitor in this earlier run|`gpu-stream.json`|

The 64-output resident completion took 6.62seconds after engine initialization; initialization took 0.32seconds with compiled kernels already cached and size-only integrity checks. Cold process startup still loads weights lazily during first-token processing. Decode tokens/s measures `(output_count-1)/(last_token_time-first_token_time)`.

The resident and streamed four-token runs produced identical token IDs. Protected boundary tensors reduced logical reads from 22.44GB to 18.22GB across four tokens; physical storage reads in the protected run were 18.223GB. The corresponding resident run read approximately 5.61GB once, then reused the weights. That distinction explains the large speed difference. It is not evidence that SSD can provide resident-memory throughput.

The final CPU reference run passed both prompts under the hard 8GB host cap, with a sampled cgroup peak of 5.21GB. Cgroup file-cache charges can belong to another cgroup when pages were already cached, so that peak is not a complete physical-residency estimate for a shared mmap file.

Recorded CUDA cgroup events show no OOM or swap use. The monitor sampled every 0.5seconds and includes the GPU's pre-existing graphics allocations. It does not establish an absolute hardware-enforced combined memory ceiling between samples.

## API and cache validation

`server_smoke.py` passed HTTP completion, exact repeated output, SSE streaming, and restoration of an on-disk prefix after a process restart. The generated response was the expected square function. A cache restore defect involving host-to-device array assignment was found and fixed; a CPU snapshot round-trip regression test covers the same abstraction.

- First request:27 new prompt tokens,5.88seconds prompt processing,16 output tokens in0.62seconds.
- Repeated request:20 cached tokens and7 new tokens;0.97seconds prompt processing, including saving the disk opening. Another same-session streaming request processed the suffix in about0.3seconds.
- After restart:20 cached tokens and7 new tokens, but5.19seconds prompt processing because model weights had to be loaded again. Persistent prefix state does not make cold weight loading disappear.
- Accounted peak across the API/restart test:7.17GB.

These are tiny correctness probes. They do not demonstrate long-context coding-agent performance, tool-call accuracy, multi-user serving or professional coding quality.

## Decision and remaining gates

Use resident mode for this dense model when the **complete** footprint fits. Forced SSD streaming works and has measurable real disk traffic, but is too slow here for comfortable interactive coding. The storage transformation has not reduced the model's information content or its total weight size.

The native Metal bridge is source-only experimental work: no macOS compilation, parity test or speed result is available. Its CPU/GPU synchronization can be expensive. It must pass the README's Mac gates before being offered as a faster replacement for llama.cpp. No 128K populated-context test, speculative decoding implementation, Edge0 MoE port or Metal SSD streaming validation is included.

Machine-readable evidence is in `validation/`. Full experiment logs and larger diagnostic files are in the workspace's `results/kindi-q4-20261025/`. Source SHA256 inventory accompanies the bundle; model weights are excluded from the source archive.
