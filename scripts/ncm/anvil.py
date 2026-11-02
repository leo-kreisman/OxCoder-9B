"""Portable ANVIL2-derived matrix optimizer, not the eight-H100 speedrun trainer.

Adapted from KellerJordan/modded-nanogpt commit
4f5270e5753b432a10d012bef3b2c4ca2d933707. See ANVIL-LICENSE.txt.
Uses ordinary torch matmuls, FP32 state/master parameters and fixed twin-rail
coefficients from step one. No distributed banks, Triton or captured tail graphs.
The spectral calculation is FP32 here; upstream uses BF16. This is an ablation
of the optimizer idea, not a numerically identical implementation of its recipe.
"""
import torch

MAPS=(
 (3.923798038567,-6.095026865488,3.905234618423),
 (3.278126713798,-3.328923386476,.989127286973),
 (3.505298394150,-5.137358782410,1.968325560615),
 (2.815058591845,-3.685181239622,1.417196497642),
 (2.245503932403,-2.443826979899,.963091710461),
 (2.256537145403,-2.166840097229,.929501253245))


def spectral_map(x):
    tall=x.shape[-2]>x.shape[-1]
    x=x.float().mT if tall else x.float()
    x=x/(x.norm()*1.05+1e-6)
    for a,b,c in MAPS:
        gram=x@x.mT
        x=a*x+(b*gram+c*(gram@gram))@x
    return x.mT if tall else x


class Anvil(torch.optim.Optimizer):
    def __init__(self,params,lr=.002,momentum=.95,beta2=.95,weight_decay=0.):
        super().__init__(params,dict(lr=lr,momentum=momentum,beta2=beta2,weight_decay=weight_decay))
        for group in self.param_groups:
            if any(p.ndim!=2 for p in group['params']):raise ValueError('ANVIL matrix parameters must be 2D')

    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():loss=closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:continue
                g=p.grad.float()
                if not torch.isfinite(g).all():raise FloatingPointError('Nonfinite ANVIL gradient')
                s=self.state[p];dim=-1 if p.shape[0]>=p.shape[1] else -2
                if not s:
                    s.update(fast=torch.zeros_like(g),slow=torch.zeros_like(g),master=p.detach().float().clone(),
                             energy=torch.zeros_like(g.mean(dim,keepdim=True)),step=0)
                s['step']+=1;s['fast'].lerp_(g,.15);s['slow'].lerp_(g,.02)
                velocity=.4385*s['fast']+.5615*s['slow']
                update=spectral_map(torch.lerp(g,velocity,group['momentum']))
                power=update.square().mean(dim,keepdim=True)
                s['energy'].lerp_(power,1-group['beta2'])
                gain=s['energy'].clamp_min(1e-10).rsqrt()
                # Preserve total update norm after per-lane equalization.
                ratio=(power.sum()/(power*gain.square()).sum().clamp_min(1e-20)).sqrt()
                update=update*gain*ratio
                master=s['master'];aligned=(s['slow']*master)>=0
                master.sub_(group['lr']*(update+group['weight_decay']*master*aligned))
                p.copy_(master)
        return loss


def make_optimizers(module,method,lr,anvil_lr,muon_lr=None,weight_decay=0.):
    if method not in ('adamw','anvil','muon'):raise ValueError('Unknown optimizer method')
    embedding_ids={id(p) for m in module.modules() if isinstance(m,torch.nn.Embedding) for p in m.parameters()}
    matrices=[];others=[]
    for name,p in module.named_parameters():
        if not p.requires_grad:continue
        # Vocabulary projections and embedding tables retain AdamW in both arms.
        hidden=p.ndim==2 and not any(s in name for s in ('output_layer','head','embedding','wte','wpe'))
        if method=='muon':
            hidden=hidden and id(p) not in embedding_ids and not any(s in name for s in ('position','level.weight'))
        if method in ('anvil','muon') and hidden:
            matrices.append(p)
        else:others.append(p)
    if method=='muon' and not matrices:
        raise ValueError('No trainable hidden matrices for Muon; embeddings/heads alone require AdamW')
    result=[torch.optim.AdamW(others,lr=lr,weight_decay=weight_decay)] if others else []
    if matrices:
        if method=='muon':
            result.append(torch.optim.Muon(matrices,lr=lr if muon_lr is None else muon_lr,
                weight_decay=weight_decay,momentum=.95,nesterov=True,ns_steps=5,adjust_lr_fn='match_rms_adamw'))
        else:result.append(Anvil(matrices,lr=anvil_lr,weight_decay=weight_decay))
    return result


class OptimizerBundle:
    """One checkpoint/step interface for disjoint Muon and AdamW parameter groups."""
    def __init__(self,optimizers):
        self.optimizers=optimizers
        ids=[id(p) for o in optimizers for group in o.param_groups for p in group['params']]
        if len(ids)!=len(set(ids)):raise ValueError('An optimizer parameter occurs in multiple groups')

    def zero_grad(self,set_to_none=True):
        for o in self.optimizers:o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.optimizers:o.step()

    def state_dict(self):
        return {'format':'mixed-optimizers-v1','classes':[type(o).__name__ for o in self.optimizers],
                'states':[o.state_dict() for o in self.optimizers]}

    def load_state_dict(self,state):
        if state.get('format')!='mixed-optimizers-v1' or state['classes']!=[type(o).__name__ for o in self.optimizers]:
            raise ValueError('Optimizer checkpoint layout differs')
        if len(state['states'])!=len(self.optimizers):raise ValueError('Optimizer checkpoint count differs')
        for o,s in zip(self.optimizers,state['states']):o.load_state_dict(s)
