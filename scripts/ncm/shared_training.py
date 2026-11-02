"""Differentiable shared prompt for four independent NCM level branches.

Unlike the inference prefix cache, no prompt state or adapter gradient is
detached. GatedDeltaNet forks only on 64-token chunk boundaries and replays the
tail, following VeOmni's layout. Full attention keeps NCM's bidirectional
within-level targets; different levels cannot see one another.
"""
from contextlib import contextmanager
import torch
from .vendor.veomni_shared_prefix import SharedPrefixPlan, _Group


def layout(prefix, sizes, device):
    if prefix < 64 or not sizes or min(sizes)<1:
        raise ValueError('Shared training requires a prefix of at least 64 tokens')
    lengths=[prefix+n for n in sizes];starts=[];total=0
    for n in lengths:starts.append(total);total+=n
    return SharedPrefixPlan([_Group(starts,lengths,prefix)],total,4,torch.device(device))


def attention_mask(prefix, sizes, device, dtype):
    length=prefix+sum(sizes)
    mask=torch.full((length,length),float('-inf'),device=device,dtype=dtype)
    mask[:prefix,:prefix]=torch.full((prefix,prefix),float('-inf'),device=device,dtype=dtype).triu(1)
    start=prefix
    for size in sizes:
        mask[start:start+size,:prefix]=0
        mask[start:start+size,start:start+size]=0
        start+=size
    return mask[None,None]


@contextmanager
def branched_mixers(plan, prefix, sizes):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
    original_conv=hf.causal_conv1d_fn
    original_gdn=hf.torch_chunk_gated_delta_rule

    def conv(x, weight, bias=None, activation=None, **kwargs):
        # Preserve the selected convolution backend. Each branch gets only its
        # own suffix and the last width-1 prefix positions as causal history.
        history=weight.shape[-1]-1
        outputs=[original_conv(x[:,:,:prefix],weight,bias,activation=activation)]
        start=prefix
        for size in sizes:
            branch=torch.cat((x[:,:,max(0,prefix-history):prefix],x[:,:,start:start+size]),dim=2)
            outputs.append(original_conv(branch,weight,bias,activation=activation)[:,:,-size:])
            start+=size
        return torch.cat(outputs,dim=2)

    def gdn(query,key,value,g,beta,**kwargs):
        if kwargs.pop('initial_state',None) is not None or kwargs.pop('output_final_state',False):
            raise ValueError('Shared training cannot use inference state')
        if kwargs.pop('cu_seqlens',None) is not None:
            raise ValueError('Nested sequence packing is unsupported')
        output=plan.gated_delta_rule(original_gdn,query,key,value,g,beta,**kwargs)
        return output,None

    hf.causal_conv1d_fn=conv;hf.torch_chunk_gated_delta_rule=gdn
    try:yield
    finally:
        hf.causal_conv1d_fn=original_conv;hf.torch_chunk_gated_delta_rule=original_gdn


def forward_levels(model,inputs,prompt):
    from . import backbone as implementation
    if model.cached_prefix or not model.parallel_attention or prompt.shape[0]!=1:
        raise ValueError('Shared training requires uncached parallel-attention NCM, batch one')
    backbone=model.backbone;sizes=model.cfg.levels;prefix=prompt.shape[1]
    if len(inputs)!=len(sizes):raise ValueError('Wrong level inventory')
    first=model.project.weight.device
    parts=[backbone.embed_prompt(prompt).to(first,dtype=model.project.weight.dtype)]
    for level,(inp,size) in enumerate(zip(inputs,sizes)):
        if inp.shape[0]!=1 or inp.shape[1]!=size:raise ValueError('Wrong concept input shape')
        x=model.project(inp.to(first))+model.level.weight[level]+model.position[:size]
        if model.concept_input_norm is not None:
            x=torch.nn.functional.normalize(x,dim=-1,eps=1e-6)*model.concept_input_norm
        parts.append(x)
    x=torch.cat(parts,dim=1)
    positions=list(range(prefix))
    for size in sizes:positions.extend(range(prefix,prefix+size))
    plans={};masks={};pos={}
    for device in backbone.devices:
        plans[str(device)]=layout(prefix,sizes,device)
        masks[str(device)]=attention_mask(prefix,sizes,device,torch.bfloat16)
        pos[str(device)]=torch.tensor(positions,device=device)[None,None,:].expand(3,1,-1)
    for i,(layer,adapter) in enumerate(zip(backbone.layers,backbone.adapters)):
        di=backbone.layer_device_indices[i];device=backbone.devices[di]
        x=x.to(device,dtype=torch.bfloat16)
        rope=backbone.rotaries[di](x,pos[str(device)])
        mask=masks[str(device)] if i%4==3 else None
        plan=plans[str(device)]
        # Bind every replay dependency, including the device-local branch plan.
        def step(h,layer=layer,adapter=adapter,rope=rope,mask=mask,plan=plan):
            with branched_mixers(plan,prefix,sizes):
                return adapter(layer(h,position_embeddings=rope,attention_mask=mask,past_key_values=None))
        x=implementation.checkpoint(step,x,use_reentrant=False) if model.training and torch.is_grad_enabled() else step(x)
    x=backbone.norm(x).float();result=[];start=prefix
    for level,size in enumerate(sizes):
        hidden=x[:,start:start+size];start+=size
        logits,residual=model.head(hidden,level,return_latent=True)
        end=model.stop_head(hidden.mean(1)).squeeze(-1)
        result.append((logits,end,residual))
    return result
