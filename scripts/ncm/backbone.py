"""Resident, frozen OxCoder backbone with trainable residual adapters.

Uses HF's Qwen3.5 implementation. Each level is a separate forward pass;
recurrent state is NEVER carried between levels or examples. All positions at
a level predict codes together from known earlier-level inputs. Keeping the
causal hybrid mixer is an intentional departure from upstream bidirectional NCM.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding, Qwen3_5RMSNorm)

SOURCE_SHA256 = 'a171c091c67c3547c45ec58f1b85a888d73725315fde71bc929ff9f8c255bb6f'
SOURCE_REVISION = '07ba04e2149bace1b43c66448471a70378a7a708'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def inverse_heads(tensor, axis, nk=16, nv=32, dim=128):
    """Undo llama.cpp's [value_per_key, key_head, dim] storage ordering."""
    x = tensor.movedim(axis, 0)
    shape = x.shape
    return x.reshape(nv // nk, nk, dim, *shape[1:]).transpose(0, 1).reshape(shape).movedim(0, axis).contiguous()


def mapping(name):
    common = {'input_layernorm.weight': 'attn_norm.weight',
              'post_attention_layernorm.weight': 'post_attention_norm.weight',
              'mlp.gate_proj.weight': 'ffn_gate.weight', 'mlp.up_proj.weight': 'ffn_up.weight',
              'mlp.down_proj.weight': 'ffn_down.weight',
              'self_attn.q_proj.weight': 'attn_q.weight', 'self_attn.k_proj.weight': 'attn_k.weight',
              'self_attn.v_proj.weight': 'attn_v.weight', 'self_attn.o_proj.weight': 'attn_output.weight',
              'self_attn.q_norm.weight': 'attn_q_norm.weight', 'self_attn.k_norm.weight': 'attn_k_norm.weight',
              'linear_attn.in_proj_qkv.weight': 'attn_qkv.weight',
              'linear_attn.in_proj_z.weight': 'attn_gate.weight',
              'linear_attn.in_proj_a.weight': 'ssm_alpha.weight',
              'linear_attn.in_proj_b.weight': 'ssm_beta.weight',
              'linear_attn.out_proj.weight': 'ssm_out.weight',
              'linear_attn.A_log': 'ssm_a', 'linear_attn.dt_bias': 'ssm_dt.bias',
              'linear_attn.conv1d.weight': 'ssm_conv1d.weight', 'linear_attn.norm.weight': 'ssm_norm.weight'}
    return common[name]


def to_hf(name, x):
    """Invert documented GGUF transforms; quantization error remains."""
    if name == 'linear_attn.in_proj_qkv.weight':
        x = torch.cat((x[:4096], inverse_heads(x[4096:], 0)))
    elif name == 'linear_attn.conv1d.weight':
        x = torch.cat((x[:4096], inverse_heads(x[4096:], 0))).unsqueeze(1)
    elif name == 'linear_attn.in_proj_z.weight':
        x = inverse_heads(x, 0)
    elif name in ('linear_attn.in_proj_a.weight', 'linear_attn.in_proj_b.weight'):
        x = inverse_heads(x, 0, dim=1)
    elif name == 'linear_attn.out_proj.weight':
        x = inverse_heads(x, 1)
    elif name in ('linear_attn.A_log', 'linear_attn.dt_bias'):
        x = inverse_heads(x.flatten(), 0, dim=1)
        if name.endswith('A_log'):
            if not (x < 0).all():
                raise ValueError('Invalid folded -exp(A_log)')
            x = (-x).log()
    elif name.endswith('norm.weight') and name != 'linear_attn.norm.weight':
        x = x.flatten() - 1  # HF RMSNormZeroCentered, GGUF stores multiplier.
    return x


class Source:
    def __init__(self, path):
        from gguf import GGUFReader
        self.path = str(Path(path).resolve())
        self.sha = digest(path)
        if self.sha != SOURCE_SHA256:
            raise ValueError('Unvalidated source: this loader requires the published OxCoder Q4_K_M SHA256')
        self.reader = GGUFReader(path, 'r')
        self.tensors = {t.name: t for t in self.reader.tensors}
        # A bounded resident CPU copy: no repeated embedding reads from disk.
        self.embedding_data = self.tensors['token_embd.weight'].data.copy()

    def read(self, name):
        from gguf.quants import dequantize
        t = self.tensors[name]
        a = dequantize(t.data, t.tensor_type).reshape(tuple(reversed(t.shape.tolist())))
        return torch.from_numpy(np.array(a, dtype=np.float32, copy=True))

    def embeddings(self, ids):
        from gguf.quants import dequantize
        t = self.tensors['token_embd.weight']
        flat = ids.detach().cpu().numpy().reshape(-1)
        if len(flat) == 0 or flat.min() < 0 or flat.max() >= 248320:
            raise ValueError('Invalid OxCoder prompt token IDs')
        a = dequantize(self.embedding_data[flat], t.tensor_type)
        return torch.from_numpy(a.copy()).reshape(*ids.shape, 4096)


def config():
    return Qwen3_5TextConfig(hidden_size=4096, intermediate_size=12288, num_hidden_layers=32,
        num_attention_heads=16, num_key_value_heads=4, head_dim=256,
        linear_num_key_heads=16, linear_num_value_heads=32, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_conv_kernel_dim=4, vocab_size=248320,
        layer_types=['full_attention' if i % 4 == 3 else 'linear_attention' for i in range(32)],
        rope_parameters={'rope_type':'default', 'rope_theta':1e7, 'partial_rotary_factor':.25,
                         'mrope_section':[11,11,10], 'mrope_interleaved':True})


class Adapter(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(torch.nn.functional.silu(self.down(x)))


def concept_attention_mask(length,prefix_length,device,dtype):
    if not 0<=prefix_length<=length:raise ValueError('Invalid prefix length')
    mask=torch.zeros((length,length),device=device,dtype=dtype)
    mask[:prefix_length,:prefix_length]=torch.full((prefix_length,prefix_length),float('-inf'),device=device,dtype=dtype).triu(1)
    mask[:prefix_length,prefix_length:]=float('-inf')
    return mask[None,None]


class OxBackbone(nn.Module):
    def __init__(self, source, devices=('cuda:0','cuda:1'), rank=8, progress=print,layer_split=None):
        super().__init__()
        self.width = 4096
        self.devices = list(map(torch.device, devices))
        if layer_split is None:
            self.layer_device_indices=[min(i*len(self.devices)//32,len(self.devices)-1) for i in range(32)]
        else:
            if len(layer_split)!=len(self.devices) or sum(layer_split)!=32 or min(layer_split)<=0:
                raise ValueError('Layer split must allocate all 32 layers across devices')
            self.layer_device_indices=[d for d,n in enumerate(layer_split) for _ in range(n)]
        self.config = config()
        self.source = source
        self.config._attn_implementation = 'sdpa'
        self.layers = nn.ModuleList()
        self.adapters = nn.ModuleList()
        self.rotaries = nn.ModuleList([Qwen3_5TextRotaryEmbedding(self.config).to(d) for d in self.devices])
        self.transfers = []
        for i in range(32):
            device = self.devices[self.layer_device_indices[i]]
            with torch.device('meta'):
                layer = Qwen3_5DecoderLayer(self.config, i)
            for name, p in list(layer.named_parameters()):
                src = f'blk.{i}.' + mapping(name)
                value = to_hf(name, source.read(src))
                if tuple(value.shape) != tuple(p.shape):
                    raise ValueError(f'{src}: shape {value.shape} != {p.shape}')
                # Preserve scalar/norm precision. Large weights resident BF16.
                dtype = torch.float32 if value.ndim == 1 else torch.bfloat16
                module_name, attr = name.rsplit('.', 1)
                module = layer.get_submodule(module_name)
                setattr(module, attr, nn.Parameter(value.to(device=device, dtype=dtype), requires_grad=False))
                self.transfers.append(dict(source=src, target=f'layers.{i}.{name}',
                    shape=list(value.shape), parameters=value.numel(), dtype=str(dtype), device=str(device)))
                del value
            self.layers.append(layer)
            self.adapters.append(Adapter(4096, rank).to(device=device, dtype=torch.bfloat16))
            progress(json.dumps(dict(phase='load_backbone', layer=i+1, layers=32)), flush=True)
        self.norm = Qwen3_5RMSNorm(4096, 1e-6).to(self.devices[-1])
        self.norm.weight = nn.Parameter((source.read('output_norm.weight').flatten()-1).to(self.devices[-1]), requires_grad=False)
        self.source_identity = {'sha256':source.sha, 'path':source.path,
            'reused_parameters':sum(t['parameters'] for t in self.transfers)+4096,
            'excluded':['token_embd.weight','output.weight','vision components absent in GGUF'],
            'architecture':'hybrid-causal-mixer-next-concept-v1'}

    def embed_prompt(self, ids):
        return self.source.embeddings(ids).to(self.devices[0], dtype=torch.bfloat16)

    def forward(self, x, prefix_length=None):
        length = x.shape[1]
        pos = torch.arange(length, device=x.device)[None, None, :].expand(3, x.shape[0], -1)
        for i, (layer, adapter) in enumerate(zip(self.layers, self.adapters)):
            di = self.layer_device_indices[i]
            x = x.to(self.devices[di], dtype=torch.bfloat16)
            rope = self.rotaries[di](x, pos.to(x.device))
            mask = None
            if i % 4 == 3:
                mask = torch.full((length, length), float('-inf'), device=x.device, dtype=x.dtype).triu(1)[None,None]
                if prefix_length is not None:mask=concept_attention_mask(length,prefix_length,x.device,x.dtype)
            # Bind closures: backward recomputes the SAME layer, not the last loop variable.
            def step(h, layer=layer, adapter=adapter, rope=rope, mask=mask):
                return adapter(layer(h, position_embeddings=rope, attention_mask=mask, past_key_values=None))
            x = checkpoint(step, x, use_reentrant=False) if self.training and torch.is_grad_enabled() else step(x)
        return self.norm(x)


class ConceptGenerator(nn.Module):
    def __init__(self, backbone, codec_config, parallel_attention=False,termination=False,concept_input_norm=None,cached_prefix=False,head_kind='linear',codebooks=None,head_input_norm=False,nearest_code=False):
        super().__init__()
        self.backbone = backbone
        self.cfg = codec_config
        self.parallel_attention=parallel_attention
        if concept_input_norm is not None and (not math.isfinite(concept_input_norm) or concept_input_norm<=0):
            raise ValueError('Concept input norm must be finite and positive')
        self.concept_input_norm=concept_input_norm
        self.cached_prefix=cached_prefix
        self.head_kind=head_kind
        first, last = backbone.devices[0], backbone.devices[-1]
        self.project = nn.Linear(codec_config.width, backbone.width).to(first)
        self.level = nn.Embedding(len(codec_config.levels), backbone.width).to(first)
        self.position = nn.Parameter(torch.randn(codec_config.length, backbone.width, device=first) * .02)
        if head_kind=='linear':self.head = nn.Linear(backbone.width, codec_config.codes).to(last)
        elif head_kind=='geometry':
            from .geometry import TiedResidualHeads
            if codebooks is None:raise ValueError('Geometry heads require matching frozen codebooks')
            self.head=TiedResidualHeads(backbone.width,codebooks,input_norm=head_input_norm,nearest_code=nearest_code).to(last)
        elif head_kind=='untied':
            from .geometry import IndependentConceptHeads
            if codebooks is None:raise ValueError('Untied heads require the codebook inventory')
            if nearest_code:raise ValueError('Untied classifiers cannot use nearest-code scoring')
            self.head=IndependentConceptHeads(backbone.width,codebooks,input_norm=head_input_norm).to(last)
        else:raise ValueError('Unknown concept head')
        self.stop_head=nn.Linear(backbone.width,1).to(last) if termination else None

    def prepare_prefix(self,prompt_ids):
        if not self.cached_prefix:return None
        from .prefix import prepare
        return prepare(self.backbone,prompt_ids)

    def forward_levels(self,inputs,prompt_ids,prefix_state=None,detach_backbone=False):
        """Teacher-forced levels evaluated as independent members of one batch."""
        if not self.cached_prefix:raise ValueError('Batched levels require prefix caching')
        if len(inputs)!=len(self.cfg.levels) or prompt_ids.shape[0]!=1:raise ValueError('Invalid level batch')
        from .prefix import forward_many,forward as cached_forward
        if prefix_state is None:prefix_state=self.prepare_prefix(prompt_ids)
        if not torch.equal(prefix_state.tokens,prompt_ids.cpu()):raise ValueError('Prefix prompt differs')
        def features():
            xs=[]
            for i,(inp,size) in enumerate(zip(inputs,self.cfg.levels)):
                if inp.shape[1]!=size or inp.shape[0]!=1:raise ValueError('Invalid level input shape')
                x=self.project(inp.to(self.project.weight.device))+self.level.weight[i]+self.position[:size]
                if self.concept_input_norm is not None:x=torch.nn.functional.normalize(x,dim=-1,eps=1e-6)*self.concept_input_norm
                xs.append(torch.nn.functional.pad(x,(0,0,0,self.cfg.levels[-1]-size)))
            # Preserve the one-position level's exact padded-two execution path.
            # Padding it to the finest level changed BF16 target gradients beyond
            # the full-model numerical gate. Only the remaining levels batch.
            first=cached_forward(self.backbone,xs[0][:,:self.cfg.levels[0]],prefix_state,self.parallel_attention)
            first=torch.nn.functional.pad(first,(0,0,0,self.cfg.levels[-1]-self.cfg.levels[0]))
            if len(xs)==1:return first.float()
            rest=forward_many(self.backbone,torch.cat(xs[1:]),prefix_state,self.cfg.levels[1:],self.parallel_attention)
            return torch.cat((first,rest)).float()
        if detach_backbone:
            with torch.no_grad():hidden=features()
        else:hidden=features()
        result=[]
        for i,size in enumerate(self.cfg.levels):
            h=hidden[i:i+1,:size]
            if self.head_kind in ('geometry','untied'):logits,residual=self.head(h,i,return_latent=True)
            else:logits=self.head(h);residual=None
            stop=self.stop_head(h.mean(1)).squeeze(-1) if self.stop_head is not None else None
            result.append((logits,stop,residual))
        return result

    def hidden_features(self,earlier_latents,prompt_ids,level,prefix_state=None):
        size = self.cfg.levels[level]
        if earlier_latents.shape[1] != size:
            raise ValueError('Wrong level inputs')
        device = self.project.weight.device
        x = self.project(earlier_latents.to(device)) + self.level.weight[level] + self.position[:size]
        if self.concept_input_norm is not None:
            x=torch.nn.functional.normalize(x,dim=-1,eps=1e-6)*self.concept_input_norm
        if self.cached_prefix:
            from .prefix import forward
            if prefix_state is None:prefix_state=self.prepare_prefix(prompt_ids)
            if not torch.equal(prefix_state.tokens,prompt_ids.cpu()):raise ValueError('Prefix prompt differs')
            h=forward(self.backbone,x,prefix_state,self.parallel_attention)
        else:
            prefix = self.backbone.embed_prompt(prompt_ids).to(x.dtype)
            joined=torch.cat((prefix,x),dim=1)
            h = self.backbone(joined,prefix_length=prefix.shape[1]) if self.parallel_attention else self.backbone(joined)
        return h[:,-size:].float()

    def forward(self, earlier_latents, prompt_ids, level,return_stop=False,prefix_state=None,return_aux=False):
        target=self.hidden_features(earlier_latents,prompt_ids,level,prefix_state)
        if self.head_kind in ('geometry','untied'):logits,residual=self.head(target,level,return_latent=True)
        else:logits=self.head(target);residual=None
        if return_aux:
            stop=self.stop_head(target.mean(dim=1)).squeeze(-1) if self.stop_head is not None else None
            return logits,stop,residual
        if return_stop:
            if self.stop_head is None:raise ValueError('Termination head not enabled')
            return logits,self.stop_head(target.mean(dim=1)).squeeze(-1)
        return logits

    @torch.no_grad()
    def generate_chunk(self,codec,prompt_ids):
        if self.stop_head is None:raise ValueError('Termination head not enabled')
        from .codec import resize
        z=torch.zeros(prompt_ids.shape[0],self.cfg.length,self.cfg.width,device=codec.position.device)
        prefix_state=self.prepare_prefix(prompt_ids)
        codes=[]
        for i,(size,book) in enumerate(zip(self.cfg.levels,codec.books)):
            if i==len(self.cfg.levels)-1:
                logits,stop=self(codec.resample(z,size),prompt_ids,i,return_stop=True,prefix_state=prefix_state)
            else:logits=self(codec.resample(z,size),prompt_ids,i,prefix_state=prefix_state)
            ids=logits.argmax(-1).to(z.device);z+=resize(book(ids),self.cfg.length);codes.append(ids)
        return codec.decode_latents(z).argmax(-1),codes,stop.sigmoid()

    @torch.no_grad()
    def generate(self, codec, prompt_ids):
        if hasattr(codec, 'generate_with'):
            return codec.generate_with(self, prompt_ids)
        from .codec import resize
        z = torch.zeros(prompt_ids.shape[0], self.cfg.length, self.cfg.width, device=codec.position.device)
        codes = []
        for i, (size, book) in enumerate(zip(self.cfg.levels, codec.books)):
            ids = self(codec.resample(z, size), prompt_ids, i).argmax(-1).to(z.device)
            z += resize(book(ids), self.cfg.length)
            codes.append(ids)
        return codec.decode_latents(z).argmax(-1), codes

    def trainable_state(self):
        return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if p.requires_grad}
