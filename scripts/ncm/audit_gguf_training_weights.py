"""Check immutable GGUF unpacking and input gradients against resident weights."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from .backbone import Source, mapping, to_hf, digest, config, Qwen3_5DecoderLayer
from .gguf_training_weights import FrozenGGUFLinear, unpack
from .train import write_json


def run(a):
    torch.set_num_threads(1)
    free, total = torch.cuda.mem_get_info(a.device)
    cap = min(1 << 30, free - (3 << 30))
    if cap <= 0:
        raise RuntimeError('Insufficient GPU reserve')
    torch.cuda.set_per_process_memory_fraction(cap / total, a.device)
    source = Source(a.source)
    report = {'source_sha256': source.sha, 'records': [], 'complete': False,
              'implementation_sha256': digest(Path(__file__).with_name('gguf_training_weights.py')),
              'scope': 'Frozen matrix unpacking and input gradients only; not full model or coding acceptance',
              'quality_ready': False}
    names = [(0, name) for name in (
        'mlp.gate_proj.weight', 'mlp.down_proj.weight', 'linear_attn.in_proj_qkv.weight',
        'linear_attn.in_proj_z.weight', 'linear_attn.in_proj_a.weight',
        'linear_attn.in_proj_b.weight', 'linear_attn.out_proj.weight')]
    names += [(3, 'self_attn.q_proj.weight'), (3, 'self_attn.o_proj.weight')]
    if a.all_layers:
        names = []
        cfg = config()
        for index in range(32):
            with torch.device('meta'):
                layer = Qwen3_5DecoderLayer(cfg, index)
            for name, parameter in layer.named_parameters():
                module, attr = name.rsplit('.', 1)
                if isinstance(layer.get_submodule(module), torch.nn.Linear) and attr == 'weight':
                    names.append((index, name))
            del layer
    report['all_linear_layers'] = a.all_layers
    report['expected_matrices'] = len(names)
    for layer, name in names:
        source_name = f'blk.{layer}.' + mapping(name)
        tensor = source.tensors[source_name]
        kind = int(tensor.tensor_type)
        if kind not in (12, 14):
            report['records'].append({'name': source_name, 'type': kind, 'scope': 'Retain ordinary resident matrix'})
            continue
        reference = to_hf(name, source.read(source_name)).to(a.device, dtype=torch.bfloat16)
        packed = torch.from_numpy(np.array(tensor.data, dtype=np.uint8, copy=True).reshape(-1)).to(a.device)
        shape = tuple(reversed(tensor.shape.tolist()))
        started = time.monotonic()
        value = unpack(packed, shape, kind, name)
        torch.cuda.synchronize(a.device)
        equal = torch.equal(value, reference)
        weight_error = float((value.float() - reference.float()).abs().max())
        mismatches = int((value != reference).sum())
        del value
        module = FrozenGGUFLinear(packed, shape, kind, name)
        torch.manual_seed(19006)
        x = torch.randn(1, 16, shape[1], device=a.device, dtype=torch.bfloat16, requires_grad=True)
        expected = torch.nn.functional.linear(x, reference)
        output = module(x)
        probe = torch.randn_like(output)
        expected_grad = torch.autograd.grad((expected * probe).sum(), x)[0]
        actual_grad = torch.autograd.grad((output * probe).sum(), x)[0]
        row = {'name': source_name, 'type': kind, 'shape': shape,
               'unpacked_weight_bit_exact': equal, 'weight_max_absolute': weight_error,
               'weight_mismatches': mismatches,
               'forward_bit_exact': torch.equal(output, expected),
               'input_gradient_bit_exact': torch.equal(actual_grad, expected_grad),
               'packed_bytes': packed.numel(), 'resident_bf16_bytes': reference.numel() * 2,
               'seconds_including_first_compilation': time.monotonic() - started}
        row['passed'] = row['unpacked_weight_bit_exact'] and row['forward_bit_exact'] and row['input_gradient_bit_exact']
        report['records'].append(row)
        write_json(a.output, report)
        print(row, flush=True)
        del reference, packed, module, x, expected, output, probe, expected_grad, actual_grad
    report['complete'] = True
    report['passed'] = all(row.get('passed', True) for row in report['records'])
    write_json(a.output, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--all-layers', action='store_true')
    run(parser.parse_args())
