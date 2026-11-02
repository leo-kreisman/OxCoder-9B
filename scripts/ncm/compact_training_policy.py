"""Qualification and logical gradient groups for compact-weight NCM training."""
import json
from pathlib import Path

import torch

from .backbone import digest
from .training_runtime import validate_audit


def prefix_policy(path, parent, prepared_sha):
    """Limit prefix sharing to the context range that passed unchanged gates."""
    reference = validate_audit(path, parent, prepared_sha, 0, 'reference', False)
    report = json.loads(Path(path).read_text())
    records = [r for r in report['records'] if r['runtime'].get('shared_prefix_training')
               and r['runtime'].get('keep_per_device') == 0
               and r['runtime'].get('convolution') == 'reference']
    if len(records) != 4 or {r['requested_prompt_tokens'] for r in records} != {64, 69, 512, 2048}:
        raise ValueError('Shared-prefix audit coverage is incomplete')
    def passed(r):
        g, l = r.get('gradient_comparison', {}), r.get('logit_comparison', {})
        groups = r.get('gradient_groups', {})
        return (r.get('passed') and g.get('finite') and l.get('finite')
                and g.get('relative_l2', 1) < .05 and g.get('cosine', 0) > .99
                and l.get('relative_l2', 1) < .02
                and set(groups) == {'adapters', 'classifiers', 'interfaces'}
                and all(v.get('finite') and v.get('relative_l2', 1) < .05
                        and v.get('cosine', 0) > .99 for v in groups.values()))
    maximum = 0
    if all(passed(r) for r in records if r['requested_prompt_tokens'] <= 512):
        maximum = 2048 if all(passed(r) for r in records) else 512
    return {**reference, 'shared_prefix_training': maximum > 0,
            'maximum_shared_prefix': maximum, 'outside_range': 'independent reference levels'}


def logical_gradient_groups(named_parameters, layer_split):
    if sum(layer_split) != 32 or min(layer_split) < 1:
        raise ValueError('Invalid canonical layer split')
    locations = [i for i, count in enumerate(layer_split) for _ in range(count)]
    groups = [[] for _ in layer_split]
    for name, parameter in named_parameters:
        if name.startswith('backbone.adapters.'):
            group = locations[int(name.split('.')[2])]
        elif name.startswith(('head.', 'stop_head.')):
            group = len(layer_split) - 1
        elif name == 'position' or name.startswith(('project.', 'level.')):
            group = 0
        else:
            raise ValueError('Unclassified trainable parameter: ' + name)
        groups[group].append(parameter)
    return groups


def qualified(path, parent, prepared_sha, packing=False, _placement=False):
    report = json.loads(Path(path).read_text())
    if not report.get('complete') or report.get('prepared_sha256') != prepared_sha:
        raise ValueError('Incomplete or different-data runtime audit')
    for name in ('source_sha256', 'codec_sha256', 'rank', 'head_kind', 'concept_input_norm', 'layer_split'):
        if report['model_identity'].get(name) != parent.get(name):
            raise ValueError('Runtime audit model identity differs: ' + name)
    implementation = ['gguf_training_weights.py', 'gguf_training_backbone.py', 'backbone.py',
                      'shared_training.py', 'training_runtime.py', 'vendor/veomni_shared_prefix.py']
    if packing:
        implementation.append('packed_training.py')
    for name in implementation:
        if report['implementation_sha256'].get(name) != digest(Path(__file__).parent / name):
            raise ValueError('Runtime implementation changed: ' + name)
    if packing:
        records = [r for r in report['records'] if r.get('packed')]
        if (len(records) != 3 or not report.get('compact_weights') or
                {tuple(r['requested_prompt_tokens']) for r in records} !=
                {(64, 69), (64, 69, 128, 256), (512, 1024)}):
            raise ValueError('Packing audit has incomplete boundary coverage')
    else:
        records = [r for r in report['records'] if r.get('compact_weights')]
        if len(records) != 4 or {r['requested_prompt_tokens'] for r in records} != {64, 69, 512, 2048}:
            raise ValueError('Weight audit has incomplete context coverage')
    for record in records:
        g, l = record.get('gradient_comparison', {}), record.get('logit_comparison', {})
        groups = record.get('gradient_groups', {})
        if (not record.get('passed') or not g.get('finite') or not l.get('finite') or
                g.get('relative_l2', 1) >= .05 or g.get('cosine', 0) <= .99 or l.get('relative_l2', 1) >= .02 or
                set(groups) != {'adapters', 'classifiers', 'interfaces'} or
                any(not v.get('finite') or v.get('relative_l2', 1) >= .05 or v.get('cosine', 0) <= .99
                    for v in groups.values())):
            raise ValueError('Runtime failed output/gradient gate')
        if not packing and not _placement and not (record.get('bit_exact_outputs') and record.get('bit_exact_gradients')):
            raise ValueError('Frozen-weight storage must be bit-exact at the complete-model level')
        if _placement and not record.get('bit_exact_outputs'):
            raise ValueError('Device-placement outputs must remain bit-exact')
    return {'audit_sha256': digest(path), 'audited_checkpoint_sha256':
            report.get('checkpoint_sha256', report.get('source_checkpoint_sha256')),
            'packing': packing, 'physical_devices': records[0].get('devices', report.get('devices'))}


def storage_policy(matrix_path, runtime_path, reproducibility_path, parent, prepared_sha):
    """Exact matrices plus output parity; control inherent backward variability."""
    matrices = json.loads(Path(matrix_path).read_text())
    if (not matrices.get('complete') or not matrices.get('passed') or not matrices.get('all_linear_layers')
            or matrices.get('source_sha256') != parent['source_sha256']
            or matrices.get('implementation_sha256') != digest(Path(__file__).with_name('gguf_training_weights.py'))
            or matrices.get('expected_matrices') != 248  # 24 recurrent x 8 + 8 attention x 7
            or len(matrices['records']) != matrices['expected_matrices']
            or len({r['name'] for r in matrices['records']}) != matrices['expected_matrices']
            or {int(r['name'].split('.')[1]) for r in matrices['records']} != set(range(32))
            or any(not all(r.get(k) for k in ('passed', 'unpacked_weight_bit_exact', 'forward_bit_exact',
                                            'input_gradient_bit_exact')) for r in matrices['records'])):
        raise ValueError('All frozen linear matrices must pass exact weight/output/input-gradient checks')
    baseline = json.loads(Path(reproducibility_path).read_text())
    if (not baseline.get('complete') or not baseline.get('passed') or baseline['prepared_sha256'] != prepared_sha
            or len(baseline['records']) != 4
            or {r['prompt_tokens'] for r in baseline['records']} != {64, 69, 512, 2048}):
        raise ValueError('Reference reproducibility control is incomplete')
    for name in ('source_sha256', 'codec_sha256', 'rank', 'head_kind', 'concept_input_norm', 'layer_split'):
        if baseline['model_identity'].get(name) != parent.get(name):
            raise ValueError('Reference reproducibility model differs')
    for name in ('backbone.py', 'shared_training.py', 'training_runtime.py', 'vendor/veomni_shared_prefix.py'):
        if baseline['implementation_sha256'].get(name) != digest(Path(__file__).parent / name):
            raise ValueError('Reference reproducibility implementation changed')
    if not any(not r['bit_exact_gradients'] for r in baseline['records']):
        # A deterministic reference must continue to satisfy the strict gate.
        runtime = qualified(runtime_path, parent, prepared_sha)
    else:
        for record in baseline['records']:
            groups = record.get('gradient_groups', {})
            g = record.get('gradient_comparison', {})
            if (not record.get('passed') or not record.get('bit_exact_outputs')
                    or not g.get('finite') or g.get('relative_l2', 1) >= .05 or g.get('cosine', 0) <= .99
                    or set(groups) != {'adapters', 'classifiers', 'interfaces'}
                    or any(not v.get('finite') or v.get('relative_l2', 1) >= .05 or v.get('cosine', 0) <= .99
                           for v in groups.values())):
                raise ValueError('Reference backward variability exceeds the unchanged numerical gate')
        runtime = qualified(runtime_path, parent, prepared_sha, _placement=True)
    if runtime['audited_checkpoint_sha256'] != baseline['source_checkpoint_sha256']:
        raise ValueError('Reference reproducibility checkpoint differs')
    return {**runtime, 'matrix_audit_sha256': digest(matrix_path),
            'reference_reproducibility_sha256': digest(reproducibility_path),
            'scope': 'Every frozen linear matrix exact; full outputs exact; backward uses unchanged numerical gates'}


def placement_policy(storage_path, placement_path, parent, prepared_sha, matrix_path=None, reproducibility_path=None):
    """Separate exact storage conversion from bounded device-relocation rounding."""
    storage = (storage_policy(matrix_path, storage_path, reproducibility_path, parent, prepared_sha)
               if matrix_path and reproducibility_path else qualified(storage_path, parent, prepared_sha))
    moved = qualified(placement_path, parent, prepared_sha, _placement=True)
    if storage['physical_devices'] != ['cuda:0', 'cuda:1'] or moved['physical_devices'] != ['cuda:1']:
        raise ValueError('Placement control must compare canonical two-device storage with GPU 1')
    if storage['audited_checkpoint_sha256'] != moved['audited_checkpoint_sha256']:
        raise ValueError('Placement controls use different checkpoints')
    return {'storage': storage, 'relocation': moved,
            'scope': 'Exact frozen weights/outputs; relocation gradients use unchanged .05 relative-L2/.99 cosine gates'}


def packs(examples, token_budget, levels, maximum_prefix=2048):
    """Preserve sample order; long/short-prefix examples run singly."""
    if token_budget < 1:
        raise ValueError('Positive compact-token budget required')
    pending, used = [], 0
    for example in examples:
        length = example['prompt'].shape[1]
        cost = length + sum(levels)
        if length < 64 or length > maximum_prefix or cost > token_budget:
            if pending:
                yield pending
                pending, used = [], 0
            yield [example]
            continue
        if pending and used + cost > token_budget:
            yield pending
            pending, used = [], 0
        pending.append(example)
        used += cost
    if pending:
        yield pending
