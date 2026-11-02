"""Detached hybrid prefix states, forked independently for every concept level.

Training deliberately omits gradients through prompt processing. Target-side
adapters still receive gradients. A state is invalid after any adapter update.
"""
import copy
import types
from dataclasses import dataclass
import torch
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import DynamicCache


@dataclass
class PrefixState:
    cache: object
    tokens: torch.Tensor
    owner: int
    versions: tuple


def versions(backbone):
    return tuple(p._version for a in backbone.adapters for p in a.parameters())


def mask_for(q, past, device, dtype, parallel=False, valid=None):
    valid=q if valid is None else valid
    if parallel:
        mask=torch.zeros(q,past+q,device=device,dtype=dtype)
    else:
        keys=torch.arange(past+q,device=device)
        queries=torch.arange(past,past+q,device=device)
        mask=torch.zeros(q,past+q,device=device,dtype=dtype).masked_fill(keys[None,:]>queries[:,None],float('-inf'))
    mask[:,past+valid:]=float('-inf')
    return mask[None,None]


def rotary(backbone,x,device_index,start):
    pos=torch.arange(start,start+x.shape[1],device=x.device)[None,None,:].expand(3,x.shape[0],-1)
    return backbone.rotaries[device_index](x,pos)


@torch.no_grad()
def prepare(backbone,tokens,chunk_size=512):
    if tokens.shape[0]!=1 or tokens.shape[1]<1 or chunk_size<1:
        raise ValueError('Prefix preparation requires one nonempty prompt and a positive chunk size')
    cache=DynamicCache(config=backbone.config)
    for start in range(0,tokens.shape[1],chunk_size):
        x=backbone.embed_prompt(tokens[:,start:start+chunk_size])
        for i,(layer,adapter) in enumerate(zip(backbone.layers,backbone.adapters)):
            di=backbone.layer_device_indices[i]
            x=x.to(device=backbone.devices[di],dtype=layer.mlp.gate_proj.weight.dtype)
            rope=rotary(backbone,x,di,start)
            mask=mask_for(x.shape[1],start,x.device,x.dtype) if layer.block_type=='full_attention' else None
            x=adapter(layer(x,position_embeddings=rope,attention_mask=mask,past_key_values=cache))
    return PrefixState(cache,tokens.detach().cpu().clone(),id(backbone),versions(backbone))


def fork_layer(cache,index,batch_size=1):
    fork=copy.copy(cache);fork.layers=list(cache.layers)
    fork.layers[index]=copy.deepcopy(cache.layers[index])
    if batch_size!=1:
        layer=fork.layers[index]
        def expand(tensor):
            if tensor is None:return None
            if tensor.shape[0]!=1:raise ValueError('Expected a single cached prefix')
            return tensor.repeat(batch_size,*([1]*(tensor.ndim-1)))
        for name in ('keys','values'):
            if hasattr(layer,name):setattr(layer,name,expand(getattr(layer,name)))
        for name in ('conv_states','recurrent_states'):
            if hasattr(layer,name):setattr(layer,name,{k:expand(v) for k,v in getattr(layer,name).items()})
    def replace_recurrent(self,new,state_idx=0,**kwargs):
        # The original HF cache copies into the initial-state tensor in place.
        # Autograd needs that tensor unchanged for backward.
        self.recurrent_states[state_idx]=new
        self.is_recurrent_states_initialized[state_idx]=True
        return new
    if hasattr(fork.layers[index],'recurrent_states'):
        fork.layers[index].update_recurrent_state=types.MethodType(replace_recurrent,fork.layers[index])
    return fork


def forward(backbone,x,state,parallel=True):
    if state.owner!=id(backbone) or state.versions!=versions(backbone):
        raise ValueError('Prefix belongs to another backbone or stale adapter weights')
    size=x.shape[1];past=state.tokens.shape[1]
    # FLA's recurrent one-token path is inference-only. Padding a masked dummy
    # position selects its differentiable chunk path. It cannot affect real
    # outputs: DeltaNet is causal and attention cannot read the dummy key.
    if size==1:
        x=torch.cat((x,torch.zeros_like(x)),dim=1)
    for i,(layer,adapter) in enumerate(zip(backbone.layers,backbone.adapters)):
        di=backbone.layer_device_indices[i]
        x=x.to(device=backbone.devices[di],dtype=layer.mlp.gate_proj.weight.dtype)
        rope=rotary(backbone,x,di,past)
        mask=mask_for(x.shape[1],past,x.device,x.dtype,parallel,size) if layer.block_type=='full_attention' else None
        def step(h,layer=layer,adapter=adapter,rope=rope,mask=mask,index=i):
            # A fresh layer state is also created during checkpoint recompute.
            cache=fork_layer(state.cache,index)
            return adapter(layer(h,position_embeddings=rope,attention_mask=mask,past_key_values=cache))
        x=checkpoint(step,x,use_reentrant=False) if backbone.training and torch.is_grad_enabled() else step(x)
    return backbone.norm(x[:,:size])


def forward_many(backbone,x,state,lengths,parallel=True):
    """Independent target sequences sharing one detached prefix, padded in batch.

    Batching does not allow attention or recurrent state to cross concept levels.
    Outputs beyond each supplied length are padding and must not enter losses.
    """
    if state.owner!=id(backbone) or state.versions!=versions(backbone):
        raise ValueError('Prefix belongs to another backbone or stale adapter weights')
    if x.shape[0]!=len(lengths) or min(lengths)<1 or max(lengths)>x.shape[1]:
        raise ValueError('Invalid batched target lengths')
    past=state.tokens.shape[1];batch=x.shape[0]
    if x.shape[1]==1:x=torch.cat((x,torch.zeros_like(x)),dim=1)
    for i,(layer,adapter) in enumerate(zip(backbone.layers,backbone.adapters)):
        di=backbone.layer_device_indices[i]
        x=x.to(device=backbone.devices[di],dtype=layer.mlp.gate_proj.weight.dtype)
        rope=rotary(backbone,x,di,past)
        mask=torch.cat([mask_for(x.shape[1],past,x.device,x.dtype,parallel,n) for n in lengths]) if layer.block_type=='full_attention' else None
        def step(h,layer=layer,adapter=adapter,rope=rope,mask=mask,index=i):
            cache=fork_layer(state.cache,index,batch_size=batch)
            return adapter(layer(h,position_embeddings=rope,attention_mask=mask,past_key_values=cache))
        x=checkpoint(step,x,use_reentrant=False) if backbone.training and torch.is_grad_enabled() else step(x)
    return backbone.norm(x)
