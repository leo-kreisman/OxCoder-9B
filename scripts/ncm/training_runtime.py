"""Opt-in execution changes; never change the NCM objective or parameter set.

The context is process-global and intended for a single training model. It must
remain active through backward, because checkpoint replay uses the same ops.
Default execution is untouched. No prompt gradients are detached or cached.
"""
from contextlib import contextmanager
import inspect
import json
from pathlib import Path


def retained_layers(device_indices, per_device):
    if not isinstance(per_device, int) or per_device < 0:
        raise ValueError('Retained layer count must be a nonnegative integer')
    selected = []
    for device in sorted(set(device_indices)):
        indices = [i for i, d in enumerate(device_indices) if d == device]
        if per_device > len(indices):
            raise ValueError('Retained layer count exceeds device allocation')
        # Evenly spread saved activations rather than concentrating on one GPU.
        selected.extend(indices[(j * len(indices)) // per_device]
                        for j in range(per_device))
    return selected


def validate_audit(path, model_identity, prepared_sha, keep, convolution, shared):
    """An opt-in runtime must pass the actual model's immutable numerical gate."""
    from .backbone import digest
    report=json.loads(Path(path).read_text())
    if not report.get('complete') or report.get('prepared_sha256')!=prepared_sha:
        raise ValueError('Runtime audit is incomplete or uses different data')
    audited=report.get('model_identity',{})
    for key in ('source_sha256','codec_sha256','rank','head_kind','concept_input_norm','layer_split'):
        if audited.get(key)!=model_identity.get(key):
            raise ValueError('Runtime audit model identity differs: '+key)
    for name in ('training_runtime.py','shared_training.py','vendor/veomni_shared_prefix.py','backbone.py'):
        if report.get('implementation_sha256',{}).get(name)!=digest(Path(__file__).parent/name):
            raise ValueError('Runtime audit implementation differs: '+name)
    records=[r for r in report['records'] if r['runtime'].get('keep_per_device')==keep
             and r['runtime'].get('convolution')==convolution
             and r['runtime'].get('shared_prefix_training',False)==shared]
    if {r['requested_prompt_tokens'] for r in records}!={64,69,512,2048}:
        raise ValueError('Runtime audit must cover short, unaligned and maximum-context prompts')
    for r in records:
        g=r.get('gradient_comparison',{});l=r.get('logit_comparison',{})
        groups=r.get('gradient_groups',{})
        if (not r.get('passed') or not g.get('finite') or not l.get('finite')
            or g.get('relative_l2',1)>=.05 or g.get('cosine',0)<=.99 or l.get('relative_l2',1)>=.02
            or set(groups)!={'adapters','classifiers','interfaces'}
            or any(not v.get('finite') or v.get('relative_l2',1)>=.05 or v.get('cosine',0)<=.99 for v in groups.values())):
            raise ValueError('Runtime failed full-model output/gradient gate')
    return {'keep_per_device':keep,'convolution':convolution,'shared_prefix_training':shared,
            'audit_sha256':digest(path),'audited_checkpoint_sha256':report['checkpoint_sha256'],
            'prompt_gradients':'retained','minimum_shared_prefix':64}


@contextmanager
def execution(model, keep_per_device=0, convolution='reference'):
    from . import backbone
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
    if convolution not in ('reference', 'causal-conv1d'):
        raise ValueError('Unknown convolution backend')
    selected = retained_layers(model.backbone.layer_device_indices, keep_per_device)
    identities = {id(model.backbone.layers[i]) for i in selected}
    original_checkpoint = backbone.checkpoint
    original_conv = hf.causal_conv1d_fn

    def checkpoint(function, *args, **kwargs):
        defaults = getattr(function, '__defaults__', None)
        if defaults and id(defaults[0]) in identities:
            return function(*args)
        return original_checkpoint(function, *args, **kwargs)

    if convolution == 'causal-conv1d':
        from causal_conv1d import causal_conv1d_fn
        hf.causal_conv1d_fn = causal_conv1d_fn
    else:
        # Installing the optional extension can change HF's decorated dispatch.
        # Pin the fallback explicitly so a baseline cannot silently use it.
        hf.causal_conv1d_fn = inspect.unwrap(original_conv)
    backbone.checkpoint = checkpoint
    try:
        yield {'keep_per_device': keep_per_device, 'retained_layers': selected,
               'convolution': convolution, 'prompt_gradients': 'retained'}
    finally:
        backbone.checkpoint = original_checkpoint
        hf.causal_conv1d_fn = original_conv
