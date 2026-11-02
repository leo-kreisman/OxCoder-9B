"""Synthetic full-shape microbatch audit; no coding-quality measurement."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch
from torch.nn import functional as F

from .backbone import digest
from .context_decoder import ContextConceptDecoder, DecoderConfig
from .train_context_decoder import make_optimizer
from .train import admission, write_json


def run(a):
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    paused = False
    try:
        if a.pause_feature_unit:
            if not a.pause_feature_unit.startswith('ncm-context-features-v25-'):
                raise ValueError('Only this experiment feature worker may be suspended')
            subprocess.run(['systemctl', '--user', 'kill', '--kill-who=main', '--signal=STOP', a.pause_feature_unit], check=True)
            paused = True
            time.sleep(2)
        torch.set_num_threads(2); torch.manual_seed(25002)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        admitted = admission([a.device], 8)
        model = ContextConceptDecoder(DecoderConfig()).to(a.device).train()
        model.checkpoint_blocks = True
        hidden = torch.randn(8, 2048, 4096, device=a.device, dtype=torch.bfloat16)
        valid = torch.ones(8, 2048, device=a.device, dtype=torch.bool)
        inputs = [torch.randn(8, n, 128, device=a.device) for n in model.config.levels]
        targets = [torch.randint(4096, (8, n), device=a.device) for n in model.config.levels]
        ending = torch.tensor([0., 1.] * 4, device=a.device)
        reference = None
        results = []
        def backward(micro):
            model.zero_grad(set_to_none=True)
            total = 0.
            for offset in range(0, 8, micro):
                stop = offset + micro
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits, end = model([x[offset:stop] for x in inputs], hidden[offset:stop], valid[offset:stop])
                    loss = sum(F.cross_entropy(p.flatten(0, 1), q[offset:stop].flatten()) for p, q in zip(logits, targets)) / len(targets)
                    loss = loss + .1 * F.binary_cross_entropy_with_logits(end, ending[offset:stop])
                (loss * micro / 8).backward()
                total += float(loss.detach()) * micro / 8
            torch.cuda.synchronize(a.device)
            return total
        for micro in (2, 4, 8):
            torch.cuda.reset_peak_memory_stats(a.device)
            backward(micro)
            elapsed = []
            for _ in range(2):
                start = time.monotonic(); loss = backward(micro); elapsed.append(time.monotonic() - start)
            grads = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()}
            if reference is None: reference, reference_loss = grads, loss
            numerator, denominator = 0., 0.
            for name, value in grads.items():
                if not torch.isfinite(value).all(): raise FloatingPointError('Nonfinite audited gradient')
                numerator += float((value - reference[name]).double().square().sum())
                denominator += float(reference[name].double().square().sum())
            relative = (numerator / max(denominator, 1e-30)) ** .5
            row = {'microbatch': micro, 'global_batch': 8, 'prompt_tokens': 2048,
                   'forward_backward_seconds': elapsed, 'median_seconds': statistics.median(elapsed),
                   'loss': loss, 'relative_gradient_l2': relative,
                   'relative_loss_delta': abs(loss - reference_loss) / abs(reference_loss),
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated(a.device)}
            row['passed'] = relative <= .02 and row['relative_loss_delta'] <= .01
            results.append(row)
            write_json(out / 'report.json', {'complete': False, 'results': results, 'quality_ready': False})
            print(json.dumps(row), flush=True)
            if grads is not reference: del grads
            gc.collect()
        optimizer = make_optimizer(model, .0003)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        start = time.monotonic(); optimizer.step(); torch.cuda.synchronize(a.device)
        optimizer_seconds = time.monotonic() - start
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite parameters after audited optimizer update')
        report = {'complete': True, 'config': asdict(model.config), 'admission': admitted,
                  'trainable_parameters': sum(p.numel() for p in model.parameters()),
                  'results': results, 'optimizer_first_step_seconds': optimizer_seconds,
                  'peak_with_optimizer_bytes': torch.cuda.max_memory_allocated(a.device),
                  'implementation_sha256': {n: digest(Path(__file__).parent / n) for n in
                      ('context_decoder.py', 'train_context_decoder.py', 'audit_context_batch.py')},
                  'quality_ready': False, 'scope': 'Synthetic maximum-context compute/gradient audit; excludes feature loading and source encoding; not end-to-end training or generation speed',
                  'thresholds': {'relative_gradient_l2': .02, 'relative_loss_delta': .01},
                  'all_passed': all(x['passed'] for x in results)}
        write_json(out / 'report.json', report)
        print(json.dumps({'complete': True, 'all_passed': report['all_passed']}), flush=True)
    finally:
        if paused:
            subprocess.run(['systemctl', '--user', 'kill', '--kill-who=main', '--signal=CONT', a.pause_feature_unit], check=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True); p.add_argument('--device', default='cuda:1')
    p.add_argument('--pause-feature-unit')
    run(p.parse_args())
