"""Admission for the versioned hard-hierarchy byte-loss experiment."""
import json
import math
from pathlib import Path
from .backbone import digest
from .deterministic_student_policy import POLICY


def validate_parent(parent, corpus, args):
    expected = {'architecture': 'frozen-context-original-oxcoder-native-ncm-v38',
        'corpus': corpus, 'planned_steps': args.steps, 'warmup': args.warmup,
        'effective_batch_examples': args.batch, 'seed': args.seed,
        'compute_dtype': 'float16', 'loss_scale': 64., 'body_lr': 1e-4, 'head_lr': 1e-3,
        'on_policy_probability': .5, 'conditioning_corruption_maximum': .15,
        'deterministic_execution': POLICY, 'autoregressive_answer_fallback': False}
    if any(parent.get(k) != v for k, v in expected.items()):
        raise ValueError('Byte supervision changes parent corpus or schedule')
    for name, checksum in parent['implementation_sha256'].items():
        if digest(Path(__file__).with_name(name)) != checksum:
            raise ValueError('Parent implementation changed: ' + name)


def qualify_objective(path, execution, parent_sha):
    report = json.loads(Path(path).read_text())
    records = report.get('records', [])
    if (report.get('complete') is not True or report.get('passed') is not True
            or report.get('disabled_byte_exact') is not True
            or report.get('checkpoint_sha256') != parent_sha
            or report.get('execution') != execution or report.get('deterministic_execution') != POLICY
            or report.get('batch_examples') != 4 or report.get('loss_scale') != 64.
            or report.get('byte_weight') != .1 or 'error' in report
            or [r.get('flags') for r in records] != [[True, False, True, False], [True] * 4]):
        raise ValueError('Matching full-shape byte-objective qualification required')
    for row in records:
        if (row.get('passed') is not True or row.get('finite_parameters') is not True
                or row.get('source_unchanged') is not True or row.get('codec_has_gradient') is not False
                or len(row.get('byte_ce', [])) != 4):
            raise ValueError('Byte-objective optimizer update failed')
        for flag, value in zip(row['flags'], row['byte_ce']):
            if not math.isfinite(value) or (value <= 0 if flag else value != 0):
                raise ValueError('Byte loss must apply only to student-conditioned rows')
    for name, checksum in report['implementation_sha256'].items():
        if digest(Path(__file__).with_name(name)) != checksum:
            raise ValueError('Audited byte-objective implementation changed: ' + name)
    return report
