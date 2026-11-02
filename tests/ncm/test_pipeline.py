import tempfile
import unittest
import json
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import torch
from torch import nn
from scripts.ncm.codec import Codec, CodecConfig, encode, decode, resize
from scripts.ncm.data import chunks, group_split, pairs
from scripts.ncm.backbone import inverse_heads, to_hf, ConceptGenerator
from scripts.ncm.train import load_data, train_generator
from scripts.ncm.backbone import digest


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.width=16;self.devices=[torch.device('cpu')]
        self.embedding=nn.Embedding(258,16)
        self.embedding.requires_grad_(False)
        self.layer=nn.Linear(16,16)

    def embed_prompt(self,x):return self.embedding(x)
    def forward(self,x):return self.layer(x + x.cumsum(1)/torch.arange(1,x.shape[1]+1)[None,:,None])


class PipelineTests(unittest.TestCase):
    def test_concept_normalization_preserves_prompt_and_gradient(self):
        backbone=TinyBackbone();captured=[]
        hook=backbone.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
        model=ConceptGenerator(backbone,self.cfg,concept_input_norm=.8)
        prompt=torch.tensor([[1,2,3]])
        logits=model(torch.randn(1,4,16),prompt,1)
        torch.testing.assert_close(captured[0][:,:3],backbone.embed_prompt(prompt))
        torch.testing.assert_close(captured[0][:,3:].norm(dim=-1),torch.full((1,4),.8))
        logits.square().mean().backward()
        self.assertGreater(float(model.project.weight.grad.norm()),0.)
        hook.remove()
        with self.assertRaises(ValueError):ConceptGenerator(backbone,self.cfg,concept_input_norm=float('nan'))

    def test_byte_loss_updates_generator_without_updating_codec(self):
        from scripts.ncm.train import decoded_byte_loss
        codec=Codec(self.cfg).eval().requires_grad_(False)
        target=torch.tensor([encode('x=1',8)])
        _,inputs=codec.targets(target)
        logits=torch.randn(1,8,16,requires_grad=True)
        loss=decoded_byte_loss(codec,logits,inputs[-1],target)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.norm()),0.)
        self.assertTrue(all(p.grad is None for p in codec.parameters()))
        with self.assertRaises(ValueError):decoded_byte_loss(codec,logits[:,:4],inputs[-1],target)

    def test_curriculum_separates_source_teacher_and_end_cases(self):
        from scripts.ncm.train import curriculum_buckets
        cases=[({'source':'a','distilled':False},{'is_final_chunk':False}),
               ({'source':'a','distilled':True},{'is_final_chunk':False}),
               ({'source':'b','distilled':False},{'is_final_chunk':False}),
               ({'source':'a','distilled':False},{'is_final_chunk':True})]
        buckets=curriculum_buckets(cases+[cases[0]])
        self.assertEqual(sorted(map(len,buckets)),[1,1,1,2])
        self.assertEqual(sum(map(len,buckets)),5)

    def test_parallel_level_mask_preserves_prefix(self):
        from scripts.ncm.backbone import concept_attention_mask,OxBackbone
        m=concept_attention_mask(5,3,'cpu',torch.float32)[0,0]
        self.assertTrue(torch.isneginf(m[:3,3:]).all())
        self.assertTrue(torch.isneginf(m[0,1:3]).all())
        self.assertTrue((m[3:,:]==0).all())
        with self.assertRaises(ValueError):OxBackbone(None,['cpu','cpu'],layer_split=[2,2])

    def test_chunk_termination_has_gradient_and_inference_output(self):
        codec=Codec(self.cfg);model=ConceptGenerator(TinyBackbone(),self.cfg,termination=True)
        ids=torch.tensor([encode('abc',8)]);_,inputs=codec.targets(ids)
        logits,stop=model(inputs[-1],ids,2,return_stop=True)
        torch.nn.functional.binary_cross_entropy_with_logits(stop,torch.ones_like(stop)).backward()
        self.assertGreater(float(model.stop_head.weight.grad.norm()),0.)
        output,codes,probability=model.generate_chunk(codec,ids)
        self.assertEqual(tuple(output.shape),(1,8));self.assertEqual(len(codes),3)
        self.assertTrue(((probability>=0)&(probability<=1)).all())

    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(3)
        self.cfg=CodecConfig(length=8,width=16,heads=2,layers=1,codes=16,levels=(1,4,8))

    def test_utf8_and_whitespace_roundtrip(self):
        text='x = "é"\n\treturn 1\n'
        self.assertEqual(decode(encode(text,40)),text)
        self.assertEqual(''.join(chunks(text,7)),text)
        with self.assertRaises(ValueError):encode(text,4)
        with self.assertRaises(ValueError):decode([255,256])
        with self.assertRaises(ValueError):decode([97,98])
        with self.assertRaises(ValueError):decode([97,257,256])

    def test_no_target_level_leakage(self):
        codec=Codec(self.cfg)
        z=torch.randn(2,8,16)
        _, codes, inputs, _=codec.quantize(z)
        self.assertEqual(int(torch.count_nonzero(inputs[0])),0)
        expected=torch.zeros_like(z)
        for book, size, ids, observed in zip(codec.books,self.cfg.levels,codes,inputs):
            torch.testing.assert_close(observed,resize(expected,size))
            expected=expected+resize(book(ids),8)

    def test_codec_gradient_and_code_decode(self):
        codec=Codec(self.cfg)
        ids=torch.tensor([encode('a=1',8),encode('b=2',8)])
        loss,logits,codes,_=codec(ids)
        loss.backward()
        self.assertGreater(float(codec.embedding.weight.grad.norm()),0)
        self.assertTrue(all(book.weight.grad is not None for book in codec.books))
        torch.testing.assert_close(logits,codec.reconstruct_codes(codes),atol=1e-6,rtol=1e-5)

    def test_generator_train_and_generate_all_levels(self):
        codec=Codec(self.cfg);codec.requires_grad_(False)
        model=ConceptGenerator(TinyBackbone(),self.cfg)
        ids=torch.tensor([encode('abc',8)])
        codes,inputs=codec.targets(ids)
        logits=model(inputs[2],ids,2)
        torch.nn.functional.cross_entropy(logits.flatten(0,1),codes[2].flatten()).backward()
        self.assertGreater(float(model.project.weight.grad.norm()),0)
        output,generated=model.generate(codec,ids)
        self.assertEqual(tuple(output.shape),(1,8))
        self.assertEqual([x.shape[1] for x in generated],[1,4,8])

    def test_geometry_generator_checkpoint_roundtrip_and_gradients(self):
        import copy
        codec=Codec(self.cfg).eval().requires_grad_(False)
        backbone=TinyBackbone()
        model=ConceptGenerator(backbone,self.cfg,termination=True,head_kind='geometry',
                               codebooks=codec.books,head_input_norm=True)
        ids=torch.tensor([encode('abc',8)]);codes,inputs=codec.targets(ids)
        logits,stop,residual=model(inputs[-1],ids,2,return_aux=True)
        loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),codes[-1].flatten())
        loss.backward()
        self.assertGreater(model.project.weight.grad.norm(),0)
        self.assertGreater(model.head.projections[-1].weight.grad.norm(),0)
        other=ConceptGenerator(copy.deepcopy(backbone),self.cfg,termination=True,head_kind='geometry',
                               codebooks=codec.books,head_input_norm=True)
        state=model.trainable_state()
        with torch.no_grad():
            for name,v in other.named_parameters():
                if v.requires_grad:v.copy_(state[name])
        torch.testing.assert_close(model(inputs[-1],ids,2),other(inputs[-1],ids,2),rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in codec.parameters()))

    def test_oracle_zero_does_not_use_target_codes(self):
        from scripts.ncm.diagnose_generator import oracle_prefix
        codec=Codec(self.cfg).eval();model=ConceptGenerator(TinyBackbone(),self.cfg).eval()
        prompt=torch.tensor([[1,2]])
        a=[torch.zeros(1,s,dtype=torch.long) for s in self.cfg.levels]
        b=[torch.ones(1,s,dtype=torch.long) for s in self.cfg.levels]
        torch.testing.assert_close(oracle_prefix(model,codec,prompt,a,0)[0],oracle_prefix(model,codec,prompt,b,0)[0])
        all_known=oracle_prefix(model,codec,prompt,b,len(b))[0]
        torch.testing.assert_close(all_known,codec.reconstruct_codes(b).argmax(-1))

    def test_nearest_geometry_restores_frozen_bias_and_residual_byte_gradient(self):
        import copy
        codec=Codec(self.cfg).eval().requires_grad_(False)
        backbone=TinyBackbone()
        model=ConceptGenerator(backbone,self.cfg,termination=True,head_kind='geometry',
                               codebooks=codec.books,head_input_norm=True,nearest_code=True)
        ids=torch.tensor([encode('abc',8)]);_,inputs=codec.targets(ids)
        logits,stop,residual=model(inputs[-1],ids,2,return_aux=True)
        continuous=codec.decode_latents(inputs[-1]+residual)
        torch.nn.functional.cross_entropy(continuous.flatten(0,1),ids.flatten()).backward()
        self.assertGreater(model.head.projections[-1].weight.grad.norm(),0)
        self.assertGreater(model.project.weight.grad.norm(),0)
        state=model.trainable_state()
        self.assertFalse(any(n.startswith('head.biases.') for n in state))
        self.assertTrue(all(p.grad is None for p in codec.parameters()))
        other=ConceptGenerator(copy.deepcopy(backbone),self.cfg,termination=True,head_kind='geometry',
                               codebooks=codec.books,head_input_norm=True,nearest_code=True)
        self.assertEqual(set(state),set(other.trainable_state()))
        with torch.no_grad():
            for name,v in other.named_parameters():
                if v.requires_grad:v.copy_(state[name])
        self.assertTrue(all(torch.count_nonzero(b)==0 for b in other.head.biases))
        torch.testing.assert_close(logits,other(inputs[-1],ids,2),rtol=0,atol=0)

    def test_uncached_warmup_detaches_backbone_but_trains_head(self):
        codec=Codec(self.cfg).eval().requires_grad_(False)
        model=ConceptGenerator(TinyBackbone(),self.cfg,head_kind='geometry',codebooks=codec.books)
        ids=torch.tensor([encode('abc',8)]);codes,inputs=codec.targets(ids)
        self.assertIsNone(model.prepare_prefix(ids))
        with torch.no_grad():features=model.hidden_features(inputs[-1],ids,2)
        logits=model.head(features,2)
        torch.nn.functional.cross_entropy(logits.flatten(0,1),codes[-1].flatten()).backward()
        self.assertGreater(model.head.projections[-1].weight.grad.norm(),0)
        self.assertIsNone(model.project.weight.grad)
        self.assertIsNone(model.backbone.layer.weight.grad)

    def test_inverse_head_permutation(self):
        original=torch.arange(32*128*3).reshape(32*128,3)
        gguf=original.reshape(16,2,128,3).transpose(0,1).reshape(32*128,3)
        self.assertTrue(torch.equal(original,inverse_heads(gguf,0)))
        original=original.T.contiguous();gguf=gguf.T.contiguous()
        self.assertTrue(torch.equal(original,inverse_heads(gguf,1)))

    def test_norm_and_decay_inversion(self):
        norm=torch.rand(256)
        torch.testing.assert_close(to_hf('self_attn.q_norm.weight',norm+1),norm)
        log=torch.arange(32)/32
        folded=(-log.exp()).reshape(16,2,1).transpose(0,1).flatten()
        torch.testing.assert_close(to_hf('linear_attn.A_log',folded),log)
        with self.assertRaises(ValueError):to_hf('linear_attn.A_log',torch.ones(32))

    def test_split_and_agent_roles(self):
        self.assertEqual(group_split('same conversation'),group_split('same conversation'))
        messages=[{'role':'user','content':'Do x'}, {'role':'assistant','content':[{'type':'toolCall','name':'bash','arguments':{'command':'echo test'}}]},
                  {'role':'toolResult','content':'test'}, {'role':'assistant','content':'done'}]
        p=list(pairs(messages))
        self.assertEqual(len(p),2)
        self.assertIn('toolResult',p[-1][0])
        self.assertNotIn('done',p[-1][0])

    def test_checkpoint_roundtrip(self):
        codec=Codec(self.cfg).eval()
        ids=torch.tensor([encode('foo',8)])
        expected=codec(ids)[1]
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'codec.pt';torch.save(codec.state_dict(),path)
            other=Codec(self.cfg).eval();other.load_state_dict(torch.load(path,weights_only=True))
            torch.testing.assert_close(expected,other(ids)[1],atol=0,rtol=0)

    def test_data_checksum_and_group_leak_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); data=root/'examples.jsonl'; manifest=root/'manifest.json'
            rows=[{'group':'a','split':'train'}, {'group':'a','split':'test'}]
            data.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            manifest.write_text(json.dumps({'examples_sha256':digest(data)}))
            with self.assertRaisesRegex(ValueError,'leaks'):load_data(root)
            data.write_text('changed')
            with self.assertRaisesRegex(ValueError,'checksum'):load_data(root)

    def test_failed_codec_blocks_quality_training_before_loading_backbone(self):
        with tempfile.TemporaryDirectory() as root:
            codec=Path(root)/'failed.pt'
            torch.save({'identity':{'data_sha256':'abc'}, 'validation':{'passed':False}},codec)
            args=SimpleNamespace(devices='cpu', data='unused', codec=str(codec), smoke=False)
            with patch('scripts.ncm.train.admission'), patch('scripts.ncm.train.load_data',return_value=([],{'examples_sha256':'abc'})):
                with self.assertRaisesRegex(RuntimeError,'gate failed'):train_generator(args)


if __name__=='__main__':unittest.main()
