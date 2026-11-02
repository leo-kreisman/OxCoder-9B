"""Compare optional FLA training kernels with HF's unwrapped reference.

Run in an isolated environment before changing a training stage's backend.
This tests an operator, not whole-model quality or inference throughput.
"""
import argparse
import inspect
import time
import torch
from .train import write_json


def run(device, lengths):
    import fla
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    reference = inspect.unwrap(torch_chunk_gated_delta_rule)
    records = []
    torch.manual_seed(321)
    for length in lengths:
        shape = (1, length, 32, 128)
        base = [torch.randn(shape, device=device, dtype=torch.bfloat16) for _ in range(3)]
        base += [-torch.rand(shape[:-1], device=device),
                 torch.rand(shape[:-1], device=device, dtype=torch.bfloat16)]
        probe = torch.randn(shape, device=device, dtype=torch.bfloat16)
        results = []
        for function in (reference, chunk_gated_delta_rule):
            timings = []
            for repeat in range(4):
                inputs = [x.detach().clone().requires_grad_(True) for x in base]
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                output, _ = function(*inputs, use_qk_l2norm_in_kernel=True)
                loss = (output.float() * probe.float()).sum() / output.numel()
                gradients = torch.autograd.grad(loss, inputs)
                torch.cuda.synchronize(device)
                if repeat:
                    timings.append(time.perf_counter() - started)
            results.append((output.detach(), [x.detach() for x in gradients], sum(timings)/len(timings)))
        def difference(actual, expected):
            a, b = actual.float(), expected.float()
            return {'relative_l2': float((a-b).norm()/b.norm().clamp_min(1e-12)),
                    'max_absolute': float((a-b).abs().max()),
                    'finite': bool(torch.isfinite(a).all())}
        output_error = difference(results[1][0], results[0][0])
        gradient_errors = [difference(a,b) for a,b in zip(results[1][1], results[0][1])]
        passed = all(e['finite'] and e['relative_l2'] < .02 for e in [output_error]+gradient_errors)
        records.append({'length': length, 'output_error': output_error,
                        'gradient_errors': gradient_errors, 'passed': passed,
                        'reference_forward_backward_seconds': results[0][2],
                        'fla_forward_backward_seconds': results[1][2]})
    return {'fla': fla.__version__, 'torch': str(torch.__version__),
            'gpu': torch.cuda.get_device_name(device), 'records': records,
            'passed': all(r['passed'] for r in records),
            'scope': 'isolated BF16 Gated-DeltaNet operator and gradients; not end-to-end speed',
            'relative_l2_tolerance': .02, 'quality_ready': False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--lengths', default='64,128,512')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    report = run(a.device, [int(x) for x in a.lengths.split(',')])
    write_json(a.output, report)
    if not report['passed']:
        raise SystemExit('Kernel comparison failed')


if __name__ == '__main__':
    main()
