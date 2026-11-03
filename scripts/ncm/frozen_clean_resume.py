"""Import complete training state; reject silent resets or schedule changes."""
from pathlib import Path
import torch
from .backbone import digest


def import_parent_state(path, parent, model, optimizer, rng, device, corpus, args):
    if parent.get('architecture') != 'frozen-context-original-oxcoder-native-ncm-v31':
        raise ValueError('Continuation requires an original-backbone v31 training checkpoint')
    expected = {'corpus': corpus, 'planned_steps': args.steps, 'warmup': args.warmup,
                'effective_batch_examples': args.batch, 'seed': args.seed, 'compute_dtype': 'float16',
                'loss_scale': optimizer.loss_scale, 'body_lr': 1e-4, 'head_lr': 1e-3,
                'on_policy_probability': .5, 'autoregressive_answer_fallback': False}
    if any(parent.get(k) != v for k, v in expected.items()):
        raise ValueError('Parent recipe, corpus or numerical policy changed')
    for name, sha in parent['implementation_sha256'].items():
        if digest(Path(__file__).with_name(name)) != sha:
            raise ValueError('Parent implementation changed: ' + name)
    state = torch.load(path, weights_only=True, map_location='cpu')
    if state['identity'] != parent or not 0 < state['step'] < args.steps:
        raise ValueError('Invalid parent training state')
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(parameters) != set(state['model']):
        raise ValueError('Trainable parameter inventory changed')
    with torch.no_grad():
        for name, p in parameters.items():
            p.copy_(state['model'][name].to(p))
    optimizer.load_state_dict(state['optimizer'])
    rng.setstate(state['rng']['python'])
    torch.set_rng_state(state['rng']['torch'])
    if str(device).startswith('cuda'):
        torch.cuda.set_rng_state(state['rng']['cuda'], device)
    return state['step']
