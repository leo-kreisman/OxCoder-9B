"""Repeat the objective diagnostic with deterministic PyTorch execution."""
import argparse
import json
import os
from pathlib import Path
import torch
from .audit_canonical_repeatability import run
from .backbone import digest
from .train import write_json


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'codec', 'checkpoint', 'selection', 'manifest', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before process startup')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        run(args)
    finally:
        path = Path(args.output) / 'report.json'
        if path.exists():
            report = json.loads(path.read_text())
            report['deterministic_control'] = {'torch_deterministic_algorithms': True,
                'cublas_workspace_config': ':4096:8', 'cudnn_deterministic': True,
                'implementation_sha256': digest(__file__)}
            write_json(path, report)
