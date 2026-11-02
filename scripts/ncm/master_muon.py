"""Muon with FP32 master parameters and momentum, BF16 forward parameters.

PyTorch's Muon updates parameters in their resident dtype. Retaining a separate
FP32 master prevents repeated sub-ULP updates from disappearing in BF16. This
changes optimizer numerics and requires an explicitly identified training stage.
"""
import torch


class FP32MasterMuon:
    def __init__(self, params, **kwargs):
        self.original = list(params)
        if not self.original or any(p.ndim != 2 or not p.is_floating_point() for p in self.original):
            raise ValueError('Muon master parameters must be nonempty floating matrices')
        if len({id(p) for p in self.original}) != len(self.original):
            raise ValueError('Duplicate master parameter')
        self.masters = [torch.nn.Parameter(p.detach().float().clone()) for p in self.original]
        self.inner = torch.optim.Muon(self.masters, **kwargs)

    @property
    def param_groups(self):
        # OptimizerBundle needs original parameter ownership, not the masters.
        return [{**self.inner.param_groups[0], 'params': self.original}]

    def zero_grad(self, set_to_none=True):
        self.inner.zero_grad(set_to_none=set_to_none)
        for parameter in self.original:
            if parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    parameter.grad.zero_()

    @torch.no_grad()
    def step(self):
        for parameter, master in zip(self.original, self.masters):
            master.grad = parameter.grad.detach().float().clone() if parameter.grad is not None else None
            if master.grad is not None and not torch.isfinite(master.grad).all():
                raise FloatingPointError('Nonfinite master Muon gradient')
        self.inner.step()
        for parameter, master in zip(self.original, self.masters):
            parameter.copy_(master)

    def state_dict(self):
        return {'format': 'fp32-master-muon-v1', 'original_dtypes': [str(p.dtype) for p in self.original],
                'masters': [p.detach() for p in self.masters], 'inner': self.inner.state_dict()}

    @torch.no_grad()
    def load_state_dict(self, state):
        if (state.get('format') != 'fp32-master-muon-v1'
                or state.get('original_dtypes') != [str(p.dtype) for p in self.original]
                or len(state['masters']) != len(self.masters)):
            raise ValueError('Master optimizer layout differs')
        for saved, master in zip(state['masters'], self.masters):
            if saved.dtype != torch.float32 or saved.shape != master.shape or not torch.isfinite(saved).all():
                raise ValueError('Invalid FP32 master checkpoint')
        self.inner.load_state_dict(state['inner'])
        for saved, master, parameter in zip(state['masters'], self.masters, self.original):
            master.copy_(saved.to(master))
            parameter.copy_(master)

    def load_legacy_muon_state(self, state):
        """Explicit migration: retain old momentum, initialize missing masters.

        Previously lost low bits cannot be recovered. The rounded current model
        initializes the masters; future momentum and parameter updates use FP32.
        """
        if len(state['param_groups']) != 1:
            raise ValueError('Only a single legacy Muon group is supported')
        keys = state['param_groups'][0]['params']
        if len(keys) != len(self.masters):
            raise ValueError('Legacy Muon inventory differs')
        for key, master in zip(keys, self.masters):
            saved = state['state'].get(key, {}).get('momentum_buffer')
            if saved is not None and (saved.shape != master.shape or not torch.isfinite(saved).all()):
                raise ValueError('Legacy momentum shape or finiteness differs')
        self.inner.load_state_dict(state)


def migrate_bundle(bundle, state):
    expected = ['AdamW', 'AdamW', 'Muon']
    actual = [type(optimizer).__name__ for optimizer in bundle.optimizers]
    if (state.get('format') != 'mixed-optimizers-v1' or state.get('classes') != expected
            or actual != ['AdamW', 'AdamW', 'FP32MasterMuon'] or len(state['states']) != 3):
        raise ValueError('Unsupported optimizer migration')
    for optimizer, saved in zip(bundle.optimizers[:2], state['states'][:2]):
        optimizer.load_state_dict(saved)
    bundle.optimizers[2].load_legacy_muon_state(state['states'][2])
