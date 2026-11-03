"""Admission gate for batched prompt execution; no quality claims."""
import json
from pathlib import Path
from .backbone import digest


def qualify(path, source_sha, codec_sha):
    report = json.loads(Path(path).read_text())
    if (not report.get('complete') or not report.get('passed') or not report.get('speed_qualified')
            or report.get('source_sha256') != source_sha or report.get('codec_sha256') != codec_sha
            or report.get('compute_dtype') != 'float16' or report.get('prefix_alignment') != 1):
        raise ValueError('Passing FP16 multi-prompt numerical and speed audit required')
    if [[e['prompt_tokens'] for e in r['examples']] for r in report['records']] != [[64, 69, 128, 256], [64, 512, 1024, 2048]]:
        raise ValueError('Incomplete shape coverage')
    for r in report['records']:
        groups = [r['gradient_comparison'], *r['gradient_groups'].values()]
        if (not r['passed'] or any(not c['finite'] or c['relative_l2'] >= .05 or c['cosine'] <= .99 for c in groups)
                or not r['logit_comparison']['finite'] or r['logit_comparison']['relative_l2'] >= .02
                or r['speed_ratio_over_v29'] <= 1):
            raise ValueError('Numerical or speed gate failed')
    for name, sha in report['implementation_sha256'].items():
        if digest(Path(__file__).with_name(name)) != sha:
            raise ValueError('Execution implementation changed: ' + name)
    return {'audit_sha256': digest(path), 'batch_levels': True, 'batch_prompts': True,
            'prefix_alignment': 1, 'compute_dtype': 'float16',
            'selection_sha256': report['selection_sha256']}
