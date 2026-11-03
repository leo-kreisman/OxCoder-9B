"""Require repeatable disabled-objective gradients and audited student updates."""
import os
from pathlib import Path
import torch
from .backbone import digest
from .canonical_onpolicy_policy import validate_parent, qualify_objective as original_qualification


POLICY = {'torch_deterministic_algorithms': True, 'cublas_workspace_config': ':4096:8',
          'cudnn_deterministic': True, 'cudnn_benchmark': False}


def enable():
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != POLICY['cublas_workspace_config']:
        raise ValueError('CUBLAS_WORKSPACE_CONFIG=:4096:8 is required before startup')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return dict(POLICY)


def validate_repeatability(report, parent_sha):
    rows = report.get('records', [])
    control = report.get('deterministic_control', {})
    if (report.get('complete') is not True or report.get('error')
            or report.get('checkpoint_sha256') != parent_sha
            or [r.get('arm') for r in rows] != ['reference_1', 'reference_2', 'disabled_1', 'disabled_2']
            or any(control.get(k) != POLICY[k] for k in
                   ('torch_deterministic_algorithms', 'cublas_workspace_config', 'cudnn_deterministic'))):
        raise ValueError('Matching deterministic repeatability control required')
    first = rows[0]
    supervision = next(e for e in first['events'] if e['operation'] == 'supervision')
    for row in rows:
        delta = row['gradient_comparison_to_first']
        event = next(e for e in row['events'] if e['operation'] == 'supervision')
        if (delta.get('exact') is not True or delta.get('relative_l2') != 0
                or delta.get('maximum_absolute') != 0 or delta.get('unequal_tensors') != 0
                or row['losses'] != first['losses'] or event != supervision
                or row.get('source_unchanged') is not True or row.get('codec_has_gradient') is not False):
            raise ValueError('Repeated reference or disabled-student execution is not exact')
    for name, checksum in report['implementation_sha256'].items():
        if digest(Path(__file__).with_name(name)) != checksum:
            raise ValueError('Repeatability implementation changed: ' + name)
    if digest(Path(__file__).with_name('audit_canonical_determinism.py')) != control['implementation_sha256']:
        raise ValueError('Deterministic diagnostic implementation changed')


def qualify_objective(path, execution, parent_sha):
    report = original_qualification(path, execution, parent_sha)
    if report.get('deterministic_execution') != POLICY:
        raise ValueError('Objective audit must use the training deterministic policy')
    validate_repeatability(report['repeatability_control'], parent_sha)
    return report
