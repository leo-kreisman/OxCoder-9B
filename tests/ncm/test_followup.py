import copy
import json
import tempfile
from pathlib import Path
import unittest
import torch
from scripts.ncm.anvil import Anvil, spectral_map, MAPS, make_optimizers,OptimizerBundle
from scripts.ncm.corpus import assistant_examples
from scripts.ncm.distill import normalize


class FollowupTests(unittest.TestCase):
    def test_exact_curriculum_keeps_evaluation_failures(self):
        from types import SimpleNamespace
        from scripts.ncm.train import exact_training_curriculum
        class Codec:
            config=SimpleNamespace(length=8)
            def eval(self):return self
            def targets(self,ids):return [ids],None
            def reconstruct_codes(self,codes):
                ids=codes[0].clone();ids[ids[:,0]==ord('b'),0]=ord('x')
                return torch.nn.functional.one_hot(ids,258).float()
        def row(i,text,split,source):return ({'id':i,'target':text,'split':split,'source':source},{})
        examples=[row('a','a','train','one'),row('b','b','train','one'),
                  row('c','c','train','two'),row('v','b','validation','one')]
        selected,report=exact_training_curriculum(Codec(),examples,'cpu')
        self.assertEqual([r['id'] for r,c in selected],['a','c','v'])
        self.assertEqual(report['rejected_ids'],['b']);self.assertFalse(report['quality_ready'])

    def test_complete_conditioning_never_truncates_or_leaks_target(self):
        from scripts.ncm.conditioning import condition_chunk,index_responses
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):return 'QUESTION:'+messages[0]['content']+'ANSWER:'
            def encode(self,text,**kwargs):return list(text.encode())
        original={'id':'r','group':'g','split':'train','target':'éabc',
                  'messages':[{'role':'user','content':'write code'}]}
        indexed=index_responses([original],[])
        row={'id':'r:byte-1','group':'g','split':'train','target':'bc'}
        c=condition_chunk(row,indexed,Tokenizer(),4,100)
        self.assertTrue(c['is_final_chunk'])
        self.assertEqual(bytes(c['ids']).decode(),'QUESTION:write codeANSWER:éa')
        self.assertIsNone(condition_chunk(row,indexed,Tokenizer(),4,3))
        with self.assertRaises(ValueError):condition_chunk({**row,'target':'wrong'},indexed,Tokenizer(),4,100)
        with self.assertRaises(ValueError):index_responses([original],[{**original,'id':'teacher','split':'test'}])

    def test_coverage_batch_has_one_end_then_padding(self):
        from scripts.ncm.calibrate_codec import coverage_batch
        ids=coverage_batch(128,16,torch.Generator().manual_seed(1))
        self.assertTrue(((ids==256).sum(-1)==1).all())
        for r in ids:
            n=int((r==256).nonzero()[0]);self.assertTrue((r[:n]<256).all());self.assertTrue((r[n+1:]==257).all())

    def test_revival_only_changes_unused_fine_codes(self):
        from scripts.ncm.calibrate_codec import revive_fine_codes,coverage_batch
        from scripts.ncm.codec import Codec,CodecConfig
        m=Codec(CodecConfig(length=8,width=16,layers=0,codes=16,levels=(1,4,8)))
        o=torch.optim.AdamW(m.parameters());rng=torch.Generator().manual_seed(7)
        ids=coverage_batch(4,8,rng);loss,*_=m(ids);loss.backward();o.step()
        before=m.books[-1].weight.detach().clone();usage=torch.ones(16,dtype=torch.long);usage[3]=0
        count=revive_fine_codes(m,o,ids,usage,rng)
        self.assertEqual(count,1)
        torch.testing.assert_close(before[usage!=0],m.books[-1].weight[usage!=0],rtol=0,atol=0)
        self.assertEqual(int(torch.count_nonzero(o.state[m.books[-1].weight]['exp_avg'][3])),0)

    def test_codec_weights_only_initialization(self):
        from scripts.ncm.codec import Codec,CodecConfig
        from scripts.ncm.train import initialize_codec
        cfg=CodecConfig(length=8,width=16,layers=1,codes=16,levels=(1,4,8))
        a=Codec(cfg);b=Codec(cfg)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'checkpoint.pt'
            torch.save({'config':a.metadata(),'model':a.state_dict(),'step':12},p)
            identity=initialize_codec(b,p)
            self.assertEqual(identity['source_step'],12)
            for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
            wrong=Codec(CodecConfig(length=8,width=16,layers=0,codes=16,levels=(1,4,8)))
            with self.assertRaises(ValueError):initialize_codec(wrong,p)

    def test_muon_partition_and_resume(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.books=torch.nn.ModuleList([torch.nn.Embedding(12,4)])
                self.level=torch.nn.Embedding(3,4)
                self.position=torch.nn.Parameter(torch.randn(8,4))
                self.project=torch.nn.Linear(4,6)
                self.head=torch.nn.Linear(6,12)
        m=Model();opts=make_optimizers(m,'muon',.0001,0.)
        self.assertIsInstance(opts[1],torch.optim.Muon)
        matrix_ids={id(p) for g in opts[1].param_groups for p in g['params']}
        self.assertEqual(matrix_ids,{id(m.project.weight)})
        b=OptimizerBundle(opts)
        for p in m.parameters():p.grad=torch.randn_like(p)
        b.step()
        n=copy.deepcopy(m);restored=OptimizerBundle(make_optimizers(n,'muon',.0001,0.))
        restored.load_state_dict(copy.deepcopy(b.state_dict()))
        for p,q in zip(m.parameters(),n.parameters()):
            p.grad=torch.randn_like(p);q.grad=p.grad.clone()
        b.step();restored.step()
        for p,q in zip(m.parameters(),n.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
        with self.assertRaises(ValueError):OptimizerBundle([opts[0],opts[0]])

    def test_muon_rejects_embedding_only_model(self):
        m=torch.nn.Module();m.books=torch.nn.Embedding(12,4);m.head=torch.nn.Linear(4,12)
        with self.assertRaises(ValueError):make_optimizers(m,'muon',.0001,0.)

    def test_complete_gate_cannot_be_bypassed_by_chunk_score(self):
        from scripts.ncm.train import require_complete_gate
        with self.assertRaises(RuntimeError):require_complete_gate(None,'codec',{})
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'gate.json'
            value={'checkpoint_sha256':'codec','corpus_sha256':'corpus','passed':False,
                   'validation':{'passed':False},'probes':{'passed':True}}
            p.write_text(json.dumps(value))
            with self.assertRaises(RuntimeError):require_complete_gate(p,'codec',{'complete_corpus_sha256':'corpus'})
            value['passed']=value['validation']['passed']=True;p.write_text(json.dumps(value))
            require_complete_gate(p,'codec',{'complete_corpus_sha256':'corpus'})
            with self.assertRaises(ValueError):require_complete_gate(p,'another',{'complete_corpus_sha256':'corpus'})

    def test_tokenwise_codec_has_real_quantization_gradients(self):
        from scripts.ncm.codec import Codec,CodecConfig
        model=Codec(CodecConfig(length=8,width=16,heads=4,layers=0,codes=16,levels=(1,4,8)))
        loss,*_=model(torch.randint(258,(2,8)));loss.backward()
        self.assertIsInstance(model.encoder,torch.nn.Identity)
        self.assertTrue(all(b.weight.grad.abs().sum()>0 for b in model.books))
        self.assertGreater(float(model.embedding.weight.grad.abs().sum()),0.)

    def test_spectral_transpose_and_reference(self):
        torch.manual_seed(3)
        for shape in [(3,7),(7,3),(4,4)]:
            x=torch.randn(shape)
            want=x/(x.norm()*1.05+1e-6)
            for a,b,c in MAPS:
                g=want@want.T
                want=a*want+(b*g+c*(g@g))@want
            torch.testing.assert_close(spectral_map(x),want,rtol=2e-4,atol=2e-5)
            torch.testing.assert_close(spectral_map(x.T),spectral_map(x).T,rtol=2e-4,atol=2e-5)

    def test_zero_gradient_and_checkpoint_resume(self):
        p=torch.nn.Parameter(torch.randn(7,3));o=Anvil([p]);initial=p.detach().clone()
        p.grad=torch.zeros_like(p);o.step();torch.testing.assert_close(p,initial)
        for _ in range(3):p.grad=torch.randn_like(p);o.step()
        q=torch.nn.Parameter(p.detach().clone());other=Anvil([q]);other.load_state_dict(copy.deepcopy(o.state_dict()))
        g=torch.randn_like(p);p.grad=g;q.grad=g;o.step();other.step()
        torch.testing.assert_close(p,q,rtol=0,atol=0)
        self.assertTrue(torch.isfinite(p).all())

    def test_nonfinite_rejected(self):
        p=torch.nn.Parameter(torch.ones(2,2));o=Anvil([p]);p.grad=torch.full_like(p,float('nan'))
        with self.assertRaises(FloatingPointError):o.step()
        with self.assertRaises(ValueError):Anvil([torch.nn.Parameter(torch.ones(2))])

    def test_head_uses_adamw_in_both_arms(self):
        m=torch.nn.Module();m.block=torch.nn.Linear(4,4);m.output_layer=torch.nn.Linear(4,7,bias=False)
        opts=make_optimizers(m,'anvil',.0001,.002)
        self.assertEqual(len(opts),2)
        self.assertTrue(any(p is m.output_layer.weight for g in opts[0].param_groups for p in g['params']))
        self.assertTrue(any(p is m.block.weight for g in opts[1].param_groups for p in g['params']))

    def test_full_response_and_split_provenance(self):
        text='x'*3000
        rows=assistant_examples([{'role':'user','content':'code please'},
            {'role':'assistant','content':text},{'role':'tool','content':'exit 0'},
            {'role':'assistant','content':'done'}],'source','file')
        self.assertEqual(rows[0]['target'],text)
        self.assertEqual(rows[0]['group'],rows[1]['group'])
        self.assertEqual(rows[0]['split'],rows[1]['split'])
        normalized=normalize(rows[1]['messages'])
        self.assertEqual(normalized[-1]['role'],'user')
        self.assertIn('Recorded tool output',normalized[-1]['content'])


if __name__=='__main__':unittest.main()
