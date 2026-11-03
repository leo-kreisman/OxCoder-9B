"""Scaled gradients, FP32 Muon masters, and host-resident head AdamW state.

Unscaling and clipping of low-precision parameter gradients occur on FP32
masters, avoiding a round trip through FP16 before the optimizer update.
"""
import math
import torch

from .cpu_adamw import CPUAdamW
from .master_muon import FP32MasterMuon


class TowerOptimizer:
    def __init__(self, model, body_lr=1e-4, head_lr=1e-3, loss_scale=64.):
        if not math.isfinite(loss_scale) or loss_scale < 1:
            raise ValueError('Finite loss scale >= 1 required')
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.names = [n for n, _ in named]
        self.parameters = [p for _, p in named]
        self.heads = [p for n, p in named if n.startswith(('head.', 'stop_head.'))]
        matrices = [p for n, p in named if (n.startswith('backbone.adapters.') and p.ndim == 2) or n == 'project.weight']
        used = {id(p) for p in self.heads + matrices}
        self.other = [p for _, p in named if id(p) not in used]
        if any(p.dtype != torch.float32 for p in self.heads + self.other):
            raise ValueError('Non-matrix interface parameters must remain FP32')
        self.head = CPUAdamW(self.heads, lr=head_lr, weight_decay=.01, foreach=False)
        self.interface = torch.optim.AdamW(self.other, lr=body_lr, weight_decay=.01, foreach=False)
        self.matrix = FP32MasterMuon(matrices, lr=body_lr, weight_decay=.01,
                                     adjust_lr_fn='match_rms_adamw')
        self.loss_scale = loss_scale

    def zero_grad(self):
        for opt in (self.head, self.interface, self.matrix):
            opt.zero_grad(set_to_none=True)

    def learning_rates(self, body, head):
        for opt, lr in ((self.matrix.inner, body), (self.interface, body), (self.head.inner, head)):
            for group in opt.param_groups:
                group['lr'] = lr

    @torch.no_grad()
    def step(self):
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.parameters):
            raise FloatingPointError('Missing or nonfinite scaled gradient; no optimizer updated')
        for original, master in zip(self.matrix.original, self.matrix.masters):
            master.grad = original.grad.detach().float().clone().div_(self.loss_scale)
        for p in self.heads + self.other:
            p.grad.div_(self.loss_scale)
        update_parameters = self.heads + self.other + self.matrix.masters
        norm = torch.nn.utils.clip_grad_norm_(update_parameters, 1., error_if_nonfinite=True, foreach=False)
        self.head.step()
        self.interface.step()
        self.matrix.inner.step()
        for original, master in zip(self.matrix.original, self.matrix.masters):
            original.copy_(master)
        return float(norm)

    def state_dict(self):
        return {'format': 'frozen-tower-scaled-optimizer-v29', 'names': self.names,
                'loss_scale': self.loss_scale, 'head': self.head.state_dict(),
                'interface': self.interface.state_dict(), 'matrix': self.matrix.state_dict()}

    def load_state_dict(self, state):
        if (state.get('format') != 'frozen-tower-scaled-optimizer-v29'
                or state['names'] != self.names or state['loss_scale'] != self.loss_scale):
            raise ValueError('Frozen-tower optimizer identity changed')
        self.head.load_state_dict(state['head'])
        self.interface.load_state_dict(state['interface'])
        self.matrix.load_state_dict(state['matrix'])
