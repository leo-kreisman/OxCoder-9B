"""Compare CPU and CUDA AdamW using saved NCM interface weights and moments."""
import argparse
import copy
from pathlib import Path
import time

import torch

from .backbone import digest
from .cpu_adamw import CPUAdamW
from .train import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    if free < 7 << 30:
        raise RuntimeError('Optimizer audit requires 7 GiB free and uses at most 3 GiB')
    torch.cuda.set_per_process_memory_fraction((3 << 30) / total, device)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    source_sha = digest(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if digest(args.checkpoint) != source_sha or checkpoint['optimizer']['classes'][0] != 'AdamW':
        raise ValueError('Source identity or optimizer differs')
    names = [name for name in checkpoint['model'] if name.startswith(('head.', 'stop_head.'))]
    values = [checkpoint['model'][name] for name in names]
    saved = checkpoint['optimizer']['states'][0]
    identity = {key: checkpoint['identity'][key] for key in
                ['source_sha256', 'codec_sha256', 'head_kind', 'head_input_norm', 'head_lr']}
    keys = saved['param_groups'][0]['params']
    if len(keys) != len(values) or any(v.dtype != torch.float32 for v in values):
        raise ValueError('Unexpected head inventory or precision')
    moments = [saved['state'][key]['exp_avg'].clone() for key in keys]
    if any(p.shape != m.shape for p, m in zip(values, moments)):
        raise ValueError('Head and moment shapes differ')
    resident = [torch.nn.Parameter(p.to(device=device, copy=True)) for p in values]
    offloaded = [torch.nn.Parameter(p.to(device=device, copy=True)) for p in values]
    reference = torch.optim.AdamW(resident, foreach=False)
    reference.load_state_dict(copy.deepcopy(saved))
    candidate = CPUAdamW(offloaded, foreach=False)
    candidate.load_legacy_state(copy.deepcopy(saved))
    del checkpoint
    records = []
    for step in range(8):
        for a, b, moment in zip(resident, offloaded, moments):
            # Same training-derived direction and deterministic nonconstant
            # scale in both arms; no benchmark answers enter this control.
            gradient = (moment * (1 + .1 * (step % 3))).to(device)
            a.grad = gradient
            b.grad = gradient
        torch.cuda.synchronize(); began = time.monotonic()
        reference.step()
        torch.cuda.synchronize(); resident_seconds = time.monotonic() - began
        began = time.monotonic()
        candidate.step()
        torch.cuda.synchronize(); cpu_seconds = time.monotonic() - began
        numerator = denominator = parameter_norm = 0.
        maximum = 0.
        for a, b, original in zip(resident, offloaded, values):
            actual, expected = b.detach().cpu(), a.detach().cpu()
            difference = (actual - expected).double()
            numerator += float(difference.square().sum())
            denominator += float((expected - original).double().square().sum())
            parameter_norm += float(expected.double().square().sum())
            maximum = max(maximum, float(difference.abs().max()))
        relative_update = (numerator / max(denominator, 1e-30))**.5
        relative_parameter = (numerator / max(parameter_norm, 1e-30))**.5
        records.append({'step': step + 1, 'resident_seconds': resident_seconds, 'cpu_seconds': cpu_seconds,
                        'relative_update_l2': relative_update, 'relative_parameter_l2': relative_parameter,
                        'maximum_absolute_error': maximum,
                        'passed': relative_update < .001 and relative_parameter < .000002})
        if step == 3:
            state = copy.deepcopy(candidate.state_dict())
            candidate = CPUAdamW(offloaded, foreach=False)
            candidate.load_state_dict(state)
    def gpu_bytes(optimizer):
        return sum(value.numel() * value.element_size() for state in optimizer.state.values()
                   for value in state.values() if isinstance(value, torch.Tensor) and value.is_cuda)
    write_json(out / 'report.json', {
        'checkpoint_sha256': source_sha, 'names': names, 'records': records,
        'model_identity': identity,
        'resident_optimizer_gpu_bytes': gpu_bytes(reference),
        'offloaded_optimizer_gpu_bytes': gpu_bytes(candidate.inner),
        'implementation_sha256': {name: digest(Path(__file__).parent / name)
                                  for name in ['cpu_adamw.py', 'audit_cpu_adamw.py']},
        'passed': all(row['passed'] for row in records), 'complete': True,
        'quality_ready': False, 'scope': 'Optimizer numerics, migration and reload; not generation quality',
        'gradient_source': 'Saved training first moments, scaled deterministically; not new model backward gradients'})


if __name__ == '__main__':
    main()
