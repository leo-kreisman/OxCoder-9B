"""AdamW for FP32 interfaces with optimizer arithmetic/state in host memory.

Forward parameters and gradients remain on their original device. CPU copies
receive the same gradients and return updated FP32 parameters each step. This
is optimizer offloading during training, not model or SSD inference offloading.
"""
import torch


class CPUAdamW:
    def __init__(self, params, **kwargs):
        self.original = list(params)
        if not self.original or any(p.dtype != torch.float32 for p in self.original):
            raise ValueError('CPU AdamW supports nonempty FP32 parameter groups only')
        if len({id(p) for p in self.original}) != len(self.original):
            raise ValueError('Duplicate optimizer parameter')
        self.copies = [torch.nn.Parameter(p.detach().to(device='cpu', copy=True)) for p in self.original]
        self.inner = torch.optim.AdamW(self.copies, **kwargs)

    @property
    def param_groups(self):
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
        for original, copied in zip(self.original, self.copies):
            copied.grad = original.grad.detach().to(device='cpu', copy=True) if original.grad is not None else None
            if copied.grad is not None and not torch.isfinite(copied.grad).all():
                raise FloatingPointError('Nonfinite CPU AdamW gradient')
        self.inner.step()
        for original, copied in zip(self.original, self.copies):
            original.copy_(copied)

    def state_dict(self):
        return {'format': 'cpu-adamw-v1', 'shapes': [list(p.shape) for p in self.original],
                'dtypes': [str(p.dtype) for p in self.original], 'inner': self.inner.state_dict()}

    @torch.no_grad()
    def load_legacy_state(self, state):
        if len(state['param_groups']) != 1 or len(state['param_groups'][0]['params']) != len(self.original):
            raise ValueError('Optimizer parameter inventory differs')
        for index, original in zip(state['param_groups'][0]['params'], self.original):
            for name in ['exp_avg', 'exp_avg_sq', 'max_exp_avg_sq']:
                value = state['state'].get(index, {}).get(name)
                if value is not None and (value.shape != original.shape or value.dtype != torch.float32
                                          or not torch.isfinite(value).all()):
                    raise ValueError('Invalid FP32 optimizer moment')
        self.inner.load_state_dict(state)
        # Model weights, restored by the caller, are the authoritative FP32
        # parameter copy. Avoid storing a duplicate copy in every checkpoint.
        for original, copied in zip(self.original, self.copies):
            copied.copy_(original)

    def load_state_dict(self, state):
        if (state.get('format') != 'cpu-adamw-v1'
                or state.get('shapes') != [list(p.shape) for p in self.original]
                or state.get('dtypes') != [str(p.dtype) for p in self.original]):
            raise ValueError('CPU AdamW checkpoint layout differs')
        self.load_legacy_state(state['inner'])


def migrate_bundle(bundle, state):
    if (state.get('format') != 'mixed-optimizers-v1'
            or state.get('classes') != ['AdamW', 'AdamW', 'FP32MasterMuon']
            or [type(o).__name__ for o in bundle.optimizers] != ['CPUAdamW', 'AdamW', 'FP32MasterMuon']
            or len(state.get('states', [])) != 3):
        raise ValueError('Unsupported CPU-optimizer migration')
    bundle.optimizers[0].load_legacy_state(state['states'][0])
    for optimizer, saved in zip(bundle.optimizers[1:], state['states'][1:]):
        optimizer.load_state_dict(saved)
