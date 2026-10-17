# OxCoder-9B — Q4_K_M GGUF

Unmodified text-generation GGUF from [prithivMLmods/OxCoder-9B-GGUF](https://huggingface.co/prithivMLmods/OxCoder-9B-GGUF/tree/d108751ef79e645b07d04030427b6b67b284d07e),
converted from [OrionLLM/OxCoder-9B](https://huggingface.co/OrionLLM/OxCoder-9B).
This release contains the Q4_K_M quantization, not the original full-precision checkpoint.
No training or re-quantization was performed for this mirror.

## Download and reconstruct on macOS or Linux

Download `download.sh` from this release into an empty folder and run:

```sh
bash download.sh
```

Requires `curl` and Python 3. The script downloads the six parts and supporting
files, then verifies and joins them. Alternatively download all six `.partNNN`
assets, `model-manifest.json` and `reassemble.py`, then run `python3 reassemble.py`.
Load the resulting `OxCoder-9B.Q4_K_M.gguf` into a compatible GGUF runtime.

The parts are raw byte splits, not independently loadable GGUF shards.
Five parts are 1 GiB each; the last holds the remainder. Keep about 11 GiB free
on disk for both parts and the reconstructed file. Interrupted downloads resume.
After successful verification, the parts may be removed manually.

Weights: 5,629,117,536 bytes (approximately 5.24 GiB). Runtime memory
also includes context state and working buffers; this release makes no measured
RAM, speed or quality claim for a Mac. No vision projector is included.

Quantized source revision: `d108751ef79e645b07d04030427b6b67b284d07e`.
Full-file SHA-256: `a171c091c67c3547c45ec58f1b85a888d73725315fde71bc929ff9f8c255bb6f`.

Apache-2.0 according to the source model cards; see LICENSE and NOTICE.
Original and converter model cards are included unmodified. This mirror is not
an official release by OrionLLM or prithivMLmods.

[Download release](https://github.com/leo-kreisman/OxCoder-9B/releases/tag/model-oxcoder-9b-q4-k-m)
