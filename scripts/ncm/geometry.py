"""Per-level heads tied to the frozen residual codebooks.

Distance scoring follows HazyResearch/ncm's _TiedCodebookHead (MIT; see
NCM-UPSTREAM-LICENSE.txt). This variant also exposes continuous residuals for
supervision and supports training-only code-frequency initialization.
"""
import torch
from torch import nn


class IndependentConceptHeads(nn.Module):
    """Separate unconstrained code classifiers; no codebook-distance scoring.

    No residual vector is predicted. Callers requiring a continuous residual
    loss must select a different head instead of treating logits as latents.
    """
    def __init__(self,width,codebooks,input_norm=False):
        super().__init__()
        if not codebooks:raise ValueError('At least one codebook is required')
        self.input_norm=input_norm
        self.classifiers=nn.ModuleList([nn.Linear(width,b.num_embeddings) for b in codebooks])

    def forward(self,hidden,level,return_latent=False):
        if self.input_norm:
            hidden=hidden*hidden.square().mean(-1,keepdim=True).clamp_min(1e-6).rsqrt()
        logits=self.classifiers[level](hidden)
        return (logits,None) if return_latent else logits


class TiedResidualHeads(nn.Module):
    def __init__(self,width,codebooks,input_norm=False,nearest_code=False):
        super().__init__()
        self.input_norm=input_norm
        self.nearest_code=nearest_code
        self.projections=nn.ModuleList([nn.Linear(width,b.weight.shape[1],bias=nearest_code) for b in codebooks])
        self.biases=nn.ParameterList([nn.Parameter(torch.zeros(b.num_embeddings),requires_grad=not nearest_code) for b in codebooks])
        self.log_temperature=nn.Parameter(torch.zeros(len(codebooks)))
        for i,b in enumerate(codebooks):
            self.register_buffer(f'book_{i}',b.weight.detach().clone(),persistent=False)

    def forward(self,hidden,level,return_latent=False):
        if self.input_norm:
            hidden=hidden*hidden.square().mean(-1,keepdim=True).clamp_min(1e-6).rsqrt()
        residual=self.projections[level](hidden)
        logits=self.score_residual(residual,level)
        return (logits,residual) if return_latent else logits

    def score_residual(self,residual,level):
        book=getattr(self,f'book_{level}')
        square=book.square().sum(-1)
        distance=(2*residual@book.T-square)/square.mean().clamp_min(1e-6)
        return distance*self.log_temperature[level].clamp(-8,6).exp()+self.biases[level]

    @torch.no_grad()
    def initialize_priors(self,counts):
        if self.nearest_code:raise ValueError('Nearest-code heads do not use class priors; calibrate residual means instead')
        if len(counts)!=len(self.biases):raise ValueError('Wrong prior count')
        for bias,count in zip(self.biases,counts):
            if count.shape!=bias.shape or (count<0).any():raise ValueError('Invalid code frequency counts')
            p=(count.float()+1)/(count.sum()+count.numel())
            bias.copy_(p.log().to(bias))

    @torch.no_grad()
    def calibrate(self,counts):
        """Initialize using admitted training frequencies.

        Both modes scale distances by the frequency-weighted mean code norm.
        Legacy mode initializes logits to empirical priors with a compensating
        class bias. Nearest-code mode has no class priors: its projection offset
        starts at the weighted mean residual, and class biases stay frozen zero.
        No code is removed and only training frequencies are used.
        """
        if len(counts)!=len(self.biases):raise ValueError('Wrong prior count')
        report=[]
        for i,(bias,count) in enumerate(zip(self.biases,counts)):
            if count.shape!=bias.shape or (count<0).any() or count.sum()<=0:raise ValueError('Invalid training counts')
            square=getattr(self,f'book_{i}').square().sum(-1)
            scale=(square*count.to(square)).sum()/count.sum()
            temperature=(square.mean()/scale.clamp_min(1e-6)).clamp(min=torch.exp(torch.tensor(-8.)).item(),max=400.)
            self.log_temperature[i].copy_(temperature.log())
            self.projections[i].weight.zero_()
            if self.nearest_code:
                mean=(getattr(self,f'book_{i}')*count.to(square)[:,None]).sum(0)/count.sum()
                self.projections[i].bias.copy_(mean)
                bias.zero_()
            else:
                probability=(count.float()+1)/(count.sum()+count.numel())
                bias.copy_(probability.log().to(bias)+temperature*square/square.mean().clamp_min(1e-6))
            report.append({'level':i,'all_code_mean_square_norm':float(square.mean()),
                           'training_weighted_square_norm':float(scale),'temperature':float(temperature),
                           'observed_codes':int((count>0).sum()),'nearest_code':self.nearest_code})
        return report
