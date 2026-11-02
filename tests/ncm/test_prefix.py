import copy
import unittest
import torch
from torch import nn
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer,Qwen3_5TextRotaryEmbedding,Qwen3_5RMSNorm
from scripts.ncm.backbone import Adapter,concept_attention_mask
from scripts.ncm import prefix


class SmallHybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=Qwen3_5TextConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=1,head_dim=8,linear_num_key_heads=2,
            linear_num_value_heads=2,linear_key_head_dim=4,linear_value_head_dim=4,
            layer_types=['linear_attention','full_attention'],vocab_size=32,
            rope_parameters={'rope_type':'default','rope_theta':1e4,'partial_rotary_factor':1.,'mrope_section':[1,1,2]})
        self.config._attn_implementation='sdpa'
        self.devices=[torch.device('cpu')];self.layer_device_indices=[0,0]
        self.layers=nn.ModuleList([Qwen3_5DecoderLayer(self.config,i) for i in range(2)])
        self.adapters=nn.ModuleList([Adapter(16,4) for _ in range(2)])
        self.rotaries=nn.ModuleList([Qwen3_5TextRotaryEmbedding(self.config)])
        self.norm=Qwen3_5RMSNorm(16,1e-6);self.embedding=nn.Embedding(32,16)

    def embed_prompt(self,ids):return self.embedding(ids)

    def full(self,tokens,target):
        x=torch.cat((self.embed_prompt(tokens).detach(),target),dim=1)
        for layer,adapter in zip(self.layers,self.adapters):
            mask=concept_attention_mask(x.shape[1],tokens.shape[1],x.device,x.dtype) if layer.block_type=='full_attention' else None
            x=adapter(layer(x,position_embeddings=prefix.rotary(self,x,0,0),attention_mask=mask))
        return self.norm(x[:,tokens.shape[1]:])


class PrefixTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19);torch.set_num_threads(1)
        self.model=SmallHybrid().eval();self.tokens=torch.tensor([[1,2,3,4,5]])

    def test_prefill_chunking_and_forks_match_full_forward(self):
        for n in [1,4]:
            target=torch.randn(1,n,16)
            full=self.model.full(self.tokens,target)
            state=prefix.prepare(self.model,self.tokens,chunk_size=2)
            before=copy.deepcopy(state.cache.layers[0].recurrent_states[0])
            a=prefix.forward(self.model,target,state);b=prefix.forward(self.model,target,state)
            torch.testing.assert_close(a,full,rtol=1e-4,atol=1e-5)
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            torch.testing.assert_close(state.cache.layers[0].recurrent_states[0],before,rtol=0,atol=0)

    def test_target_gradients_and_checkpoint_recompute(self):
        self.model.train()
        for n in [1,4]:
            x=torch.randn(1,n,16,requires_grad=True)
            expected=self.model.full(self.tokens,x).square().sum()
            g1=torch.autograd.grad(expected,x)[0]
            state=prefix.prepare(self.model,self.tokens,chunk_size=3)
            actual=prefix.forward(self.model,x,state).square().sum()
            g2=torch.autograd.grad(actual,x)[0]
            torch.testing.assert_close(g1,g2,rtol=.02,atol=2e-5)

    def test_reject_stale_state_after_adapter_update(self):
        state=prefix.prepare(self.model,self.tokens)
        with torch.no_grad():self.model.adapters[0].up.weight.add_(.01)
        with self.assertRaises(ValueError):prefix.forward(self.model,torch.randn(1,4,16),state)

    def test_batched_levels_match_outputs_gradients_and_isolate_padding(self):
        self.model.train();lengths=[1,2,4]
        x=torch.randn(3,4,16,requires_grad=True)
        state=prefix.prepare(self.model,self.tokens,chunk_size=2)
        expected=[prefix.forward(self.model,x[i:i+1,:n],state) for i,n in enumerate(lengths)]
        g1=torch.autograd.grad(sum(t.square().mean() for t in expected),x)[0]
        actual=prefix.forward_many(self.model,x,state,lengths)
        for i,n in enumerate(lengths):torch.testing.assert_close(actual[i:i+1,:n],expected[i],rtol=1e-4,atol=1e-5)
        g2=torch.autograd.grad(sum(actual[i:i+1,:n].square().mean() for i,n in enumerate(lengths)),x)[0]
        torch.testing.assert_close(g1,g2,rtol=.02,atol=2e-5)
        for i,n in enumerate(lengths):self.assertEqual(int(torch.count_nonzero(g2[i,n:])),0)
        changed=x.detach().clone();changed[0,1:]=10000
        other=prefix.forward_many(self.model,changed,state,lengths)
        torch.testing.assert_close(actual[0,:1],other[0,:1],rtol=0,atol=0)
        torch.testing.assert_close(actual[1:],other[1:],rtol=0,atol=0)

    def test_generator_batched_heads_match_and_warmup_detaches_backbone(self):
        from scripts.ncm.codec import Codec,CodecConfig,encode
        from scripts.ncm.backbone import ConceptGenerator
        self.model.width=16
        codec=Codec(CodecConfig(length=4,width=8,heads=2,layers=0,codes=16,levels=(1,2,4))).eval().requires_grad_(False)
        model=ConceptGenerator(self.model,codec.config,parallel_attention=True,termination=True,
                               cached_prefix=True,head_kind='geometry',codebooks=codec.books,head_input_norm=True)
        _,inputs=codec.targets(torch.tensor([encode('ab',4)]))
        state=model.prepare_prefix(self.tokens)
        expected=[model(x,self.tokens,i,prefix_state=state,return_aux=True) for i,x in enumerate(inputs)]
        actual=model.forward_levels(inputs,self.tokens,prefix_state=state,detach_backbone=True)
        for a,b in zip(actual,expected):
            for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=1e-4,atol=1e-5)
        sum(r[0].square().mean() for r in actual).backward()
        self.assertIsNone(model.project.weight.grad)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertGreater(model.head.projections[-1].weight.grad.norm(),0)


if __name__=='__main__':unittest.main()
