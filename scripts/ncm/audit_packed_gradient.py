"""Check packed-only saved autograd state against original GGUF linears."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .backbone import Source, mapping, digest
from .gguf_training_weights import FrozenGGUFLinear
from .packed_gradient import PackedInputGradient
from .train import write_json


def run(a):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    free, total = torch.cuda.mem_get_info(a.device)
    cap = min(1 << 30, free - (3 << 30))
    if cap <= 0:
        raise RuntimeError('Insufficient GPU reserve')
    torch.cuda.set_per_process_memory_fraction(cap / total, a.device)
    source = Source(a.source)
    report = {'complete': False, 'passed': False, 'quality_ready': False,
              'source_sha256': source.sha, 'records': [],
              'scope': 'Representative real packed matrices; input gradients only; no training update',
              'implementation_sha256': {name: digest(Path(__file__).with_name(name)) for name in
                  ('packed_gradient.py', 'gguf_training_weights.py', 'audit_packed_gradient.py')}}
    names = [(0, name) for name in ('mlp.gate_proj.weight', 'mlp.down_proj.weight',
             'linear_attn.in_proj_qkv.weight', 'linear_attn.in_proj_z.weight',
             'linear_attn.in_proj_a.weight', 'linear_attn.in_proj_b.weight', 'linear_attn.out_proj.weight')]
    names += [(3, 'self_attn.q_proj.weight'), (3, 'self_attn.o_proj.weight')]
    for layer, name in names:
        tensor = source.tensors[f'blk.{layer}.' + mapping(name)]
        kind = int(tensor.tensor_type)
        if kind not in (12, 14):
            raise ValueError('Audit matrix must use packed Q4_K or Q6_K')
        packed = torch.from_numpy(np.array(tensor.data, dtype=np.uint8, copy=True).reshape(-1)).to(a.device)
        shape = tuple(reversed(tensor.shape.tolist()))
        module = FrozenGGUFLinear(packed, shape, kind, name)
        version = packed._version
        for dtype in (torch.float16, torch.bfloat16):
            for count in (4, 212):
                torch.manual_seed(35002)
                x = torch.randn(1, count, shape[1], dtype=dtype, device=a.device, requires_grad=True)
                expected = module(x)
                probe = torch.randn_like(expected)
                reference_gradient = torch.autograd.grad(expected, x, probe)[0]
                saved = []
                with torch.autograd.graph.saved_tensors_hooks(
                        lambda t: (saved.append((str(t.dtype), t.numel() * t.element_size())), t)[1], lambda t: t):
                    actual = PackedInputGradient.apply(x, packed, shape, kind, name)
                    gradient = torch.autograd.grad(actual, x, probe)[0]
                row = {'layer': layer, 'name': name, 'type': kind, 'dtype': str(dtype), 'positions': count,
                       'forward_bit_exact': torch.equal(actual, expected),
                       'input_gradient_bit_exact': torch.equal(gradient, reference_gradient),
                       'finite': bool(torch.isfinite(actual).all() and torch.isfinite(gradient).all()),
                       'packed_unchanged': packed._version == version,
                       'saved_tensors': saved, 'dense_weight_bytes_avoided': 2 * shape[0] * shape[1]}
                row['passed'] = (row['forward_bit_exact'] and row['input_gradient_bit_exact'] and row['finite']
                                 and row['packed_unchanged'] and saved == [('torch.uint8', packed.numel())])
                report['records'].append(row)
                write_json(a.output, report)
                print(row, flush=True)
                del x, expected, probe, reference_gradient, actual, gradient
        del module, packed
    report.update(complete=True, passed=all(r['passed'] for r in report['records']),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device))
    write_json(a.output, report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:1')
    run(p.parse_args())
