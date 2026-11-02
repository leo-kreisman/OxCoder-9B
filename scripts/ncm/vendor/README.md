# VeOmni shared-prefix layout

`veomni_shared_prefix.py` is an unmodified copy of
`veomni/models/transformers/qwen3_5/shared_prefix.py` from
[ByteDance-Seed/VeOmni](https://github.com/ByteDance-Seed/VeOmni), revision
`c6099b48c4b5ee336971d85398d792f2f3118ec5`. Copyright ByteDance; Apache-2.0,
included in `VEOMNI-LICENSE`. No VeOmni training framework is installed or
implicitly imported. The local adapter uses its explicit layout and recurrent
state branching; it supplies NCM-specific masks and continuous concept inputs.

This is experimental until full-model output, gradient, memory and training
tests pass. VeOmni's reported RL speedups are not measurements of this adapter.
