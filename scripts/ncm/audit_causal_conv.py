"""Audit optional causal-conv1d against the current HF fallback before adoption.

This is an isolated operator test. It does not enable the extension in a running
training job, establish model quality, or measure end-to-end training speed.
"""
import argparse
import inspect
import statistics
import time

import torch

from .train import write_json


def difference(actual, expected):
    a, b = actual.detach().float(), expected.detach().float()
    return {'relative_l2': float((a-b).norm()/b.norm().clamp_min(1e-12)),
            'max_absolute': float((a-b).abs().max()),
            'finite': bool(torch.isfinite(a).all() and torch.isfinite(b).all())}


def run(device, lengths, repeats):
    import causal_conv1d
    from transformers.models.qwen3_5.modeling_qwen3_5 import causal_conv1d_fn
    reference = inspect.unwrap(causal_conv1d_fn)
    optimized = causal_conv1d.causal_conv1d_fn
    torch.manual_seed(17002)
    records = []
    for dtype in (torch.float32, torch.bfloat16):
        for length in lengths:
            # OxCoder's Q/K/V convolution has 8,192 channels and width four.
            x = torch.randn(1, length, 8192, device=device, dtype=dtype).transpose(1, 2)
            weight = torch.randn(8192, 4, device=device, dtype=dtype) * .1
            probe = torch.randn_like(x)
            outputs = []
            for function in (reference, optimized):
                timings = []
                for repeat in range(repeats+1):
                    xx = x.detach().clone().requires_grad_(True)
                    ww = weight.detach().clone().requires_grad_(True)
                    torch.cuda.synchronize(device)
                    begin = time.perf_counter()
                    y = function(xx, ww, activation='silu')
                    gradients = torch.autograd.grad((y.float()*probe.float()).mean(), (xx, ww))
                    torch.cuda.synchronize(device)
                    if repeat:
                        timings.append(time.perf_counter()-begin)
                outputs.append((y.detach(), [g.detach() for g in gradients], statistics.median(timings)))
            errors = [difference(outputs[1][0], outputs[0][0])]
            errors += [difference(a, b) for a, b in zip(outputs[1][1], outputs[0][1])]
            # Set in advance. BF16 fusion changes rounding; it is not bit-exact.
            tolerance = 1e-5 if dtype == torch.float32 else .02
            records.append({'dtype': str(dtype), 'length': length,
                'output_and_gradient_errors': errors, 'relative_l2_tolerance': tolerance,
                'passed': all(e['finite'] and e['relative_l2'] < tolerance for e in errors),
                'reference_forward_backward_seconds': outputs[0][2],
                'extension_forward_backward_seconds': outputs[1][2],
                'operator_speed_ratio': outputs[0][2]/outputs[1][2]})
    return {'torch': str(torch.__version__), 'causal_conv1d': causal_conv1d.__version__,
        'gpu': torch.cuda.get_device_name(device), 'records': records,
        'passed': all(r['passed'] for r in records), 'quality_ready': False,
        'adopted_in_training': False,
        'scope': 'isolated convolution outputs and gradients; whole-backbone audit still required',
        'timing_scope': 'operator only; other GPU activity may affect timings'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--lengths', default='64,512,2048')
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if a.repeats < 1:
        p.error('At least one measured repeat required')
    lengths = [int(n) for n in a.lengths.split(',')]
    if not lengths or min(lengths) < 1 or max(lengths) > 4096:
        p.error('Lengths must be within 1..4096 for this bounded audit')
    torch.set_num_threads(2)
    report = run(a.device, lengths, a.repeats)
    write_json(a.output, report)
    if not report['passed']:
        raise SystemExit('Convolution comparison failed; do not adopt this backend')


if __name__ == '__main__':
    main()
