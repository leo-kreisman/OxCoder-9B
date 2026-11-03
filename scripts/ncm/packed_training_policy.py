"""Strict qualification for packed-gradient execution and cached corpus loading."""
import json
from pathlib import Path

from .backbone import digest
from .resident_training_policy import validate_parent as validate_v33


def validate_parent(parent, corpus, args):
    if parent.get('architecture') != 'frozen-context-original-oxcoder-native-ncm-v34':
        raise ValueError('Expected a resident-runtime v34 parent')
    # The learning-policy fields are identical. Only bypass the older helper's
    # architecture tag after explicitly checking the required newer tag here.
    validate_v33({**parent, 'architecture': 'frozen-context-original-oxcoder-native-ncm-v33'}, corpus, args)


def check_implementation(report):
    for name, checksum in report['implementation_sha256'].items():
        if digest(Path(__file__).with_name(name)) != checksum:
            raise ValueError('Qualified implementation changed: ' + name)


def qualify(path, parent_sha):
    report = json.loads(Path(path).read_text())
    if (report.get('complete') is not True or report.get('passed') is not True
            or report.get('speed_qualified') is not True or report.get('error')
            or report.get('checkpoint_sha256') != parent_sha
            or set(report.get('arms', {})) != {'resident_checkpointed', 'packed_no_recompute'}):
        raise ValueError('Incomplete or unrelated packed-runtime qualification')
    for arm in report['arms'].values():
        if len(arm['records']) != 4 or any(not r['source_unchanged'] or r['codec_has_gradient'] for r in arm['records']):
            raise ValueError('Missing full-update qualification records')
    difference = report['parameter_comparison']
    if not (difference['relative_l2'] < 1e-4 and difference['maximum_absolute'] < 2e-4 and report['speed_ratio'] > 1.05):
        raise ValueError('Numerical or speed threshold failed')
    check_implementation(report)
    return report


def qualify_reader(path, cache, corpus):
    report = json.loads(Path(path).read_text())
    if (report.get('complete') is not True or report.get('passed') is not True
            or report.get('all_types_values_ordering_identical') is not True
            or report.get('identical_sampler_draws') != 1000 or report.get('corpus') != corpus
            or report.get('cache_manifest_sha256') != digest(Path(cache) / 'manifest.json')):
        raise ValueError('Unqualified or unrelated corpus serialization')
    check_implementation(report)
    return report
