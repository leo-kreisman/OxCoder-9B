"""Explicit GPU-resident head AdamW state, retaining the existing Muon policy.

CPU and CUDA AdamW have small FP32 rounding differences. Legacy state import
is explicit; this optimizer must not silently replace a pinned running job.
"""
import torch
from .frozen_tower_optimizer import TowerOptimizer


class ResidentAdamW:
    def __init__(self, params, **kwargs):
        self.original = list(params)
        if (not self.original or any(p.dtype != torch.float32 for p in self.original)
                or len({id(p) for p in self.original}) != len(self.original)
                or len({p.device for p in self.original}) != 1):
            raise ValueError('Unique FP32 parameters on one device are required')
        self.inner = torch.optim.AdamW(self.original, **kwargs)

    def zero_grad(self, set_to_none=True):
        self.inner.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self):
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.original):
            raise FloatingPointError('Nonfinite resident AdamW gradient')
        self.inner.step()

    def state_dict(self):
        return {'format': 'resident-adamw-v34', 'shapes': [list(p.shape) for p in self.original],
                'dtypes': [str(p.dtype) for p in self.original], 'inner': self.inner.state_dict()}

    def _load(self, state, required_format):
        if (state.get('format') != required_format
                or state.get('shapes') != [list(p.shape) for p in self.original]
                or state.get('dtypes') != [str(p.dtype) for p in self.original]):
            raise ValueError('Resident AdamW state layout differs')
        groups = state['inner']['param_groups']
        if len(groups) != 1 or len(groups[0]['params']) != len(self.original):
            raise ValueError('Resident AdamW group inventory differs')
        for key, parameter in zip(groups[0]['params'], self.original):
            for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                moment = state['inner']['state'].get(key, {}).get(name)
                if moment is not None and (moment.shape != parameter.shape or moment.dtype != torch.float32
                                           or not torch.isfinite(moment).all()):
                    raise ValueError('Invalid resident AdamW moment')
        self.inner.load_state_dict(state['inner'])

    def load_state_dict(self, state):
        self._load(state, 'resident-adamw-v34')

    def import_cpu_state(self, state):
        self._load(state, 'cpu-adamw-v1')


class ResidentTowerOptimizer(TowerOptimizer):
    def __init__(self, model, body_lr=1e-4, head_lr=1e-3, loss_scale=64.):
        super().__init__(model, body_lr, head_lr, loss_scale)
        self.head = ResidentAdamW(self.heads, lr=head_lr, weight_decay=.01, foreach=False)

    def state_dict(self):
        state = super().state_dict()
        state['format'] = 'frozen-tower-resident-head-v34'
        return state

    def _load(self, state, required_format, legacy):
        if (state.get('format') != required_format or state['names'] != self.names
                or state['loss_scale'] != self.loss_scale):
            raise ValueError('Resident tower optimizer identity differs')
        if legacy:
            self.head.import_cpu_state(state['head'])
        else:
            self.head.load_state_dict(state['head'])
        self.interface.load_state_dict(state['interface'])
        self.matrix.load_state_dict(state['matrix'])

    def load_state_dict(self, state):
        self._load(state, 'frozen-tower-resident-head-v34', False)

    def import_legacy_state(self, state):
        self._load(state, 'frozen-tower-scaled-optimizer-v29', True)
