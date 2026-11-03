"""Admission policy for a controlled canonical student-conditioning branch."""
import json
from pathlib import Path
from .backbone import digest


def validate_parent(parent, corpus, args):
    expected = {'architecture': 'frozen-context-original-oxcoder-native-ncm-v32',
        'corpus': corpus, 'planned_steps': args.steps, 'warmup': args.warmup,
        'effective_batch_examples': args.batch, 'seed': args.seed,
        'compute_dtype': 'float16', 'loss_scale': 64., 'body_lr': 1e-4, 'head_lr': 1e-3,
        'on_policy_probability': 0., 'conditioning_corruption_maximum': .15,
        'autoregressive_answer_fallback': False}
    if any(parent.get(k) != v for k, v in expected.items()):
        raise ValueError('Student conditioning changes parent corpus or schedule')
    for name, sha in parent['implementation_sha256'].items():
        if digest(Path(__file__).with_name(name)) != sha:
            raise ValueError('Parent implementation changed: ' + name)


def qualify_objective(path, execution, parent_sha):
    report = json.loads(Path(path).read_text())
    if (not report.get('complete') or not report.get('passed') or not report.get('disabled_student_exact')
            or report.get('execution') != execution or report.get('checkpoint_sha256') != parent_sha
            or report.get('batch_examples') != 4 or report.get('loss_scale') != 64.
            or len(report.get('records', [])) != 2):
        raise ValueError('Full-shape objective audit for this parent required')
    if ([r.get('flags') for r in report['records']] != [[True, False, True, False], [True] * 4]
            or any(not r.get('passed') or not r.get('finite_parameters')
                   or not r.get('source_unchanged') or r.get('codec_has_gradient')
                   for r in report['records']) or 'error' in report):
        raise ValueError('Student-conditioning update qualification failed')
    for name, sha in report['implementation_sha256'].items():
        if digest(Path(__file__).with_name(name)) != sha:
            raise ValueError('Audited objective implementation changed: ' + name)
    return report
