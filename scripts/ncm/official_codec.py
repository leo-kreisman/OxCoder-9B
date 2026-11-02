"""Audited wrapper for released NCM codecs. Text fidelity is measured separately."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from .backbone import digest
from .train import admission, load_data, write_json

UPSTREAM_COMMIT='c70274273f489abc73832b2b98584a8483b6b203'
HF_REVISION='d342ce8aaea58830eef3b8ad473664673e41f692'
PRETOKENIZER_REVISIONS={'gpt2':'607a30d783dfa663caf39e06633721c8d4cfcd7e',
    'Qwen/Qwen3-0.6B':'c1899de289a04d12100db370d81485cdf75e47ca'}
ARTIFACTS={
 'xsum':{'tokenizer.pt':'1f806ddaf8e023d1933e95efeabcaaf99f795576c0a05045f1a0f9389ab34fa7',
         'config.yaml':'5dee6b7339645a22362f695fbadb27df6e5254db2d3be0bff8516138d95da493'},
 'wmt14':{'tokenizer.pt':'74f04176a3cd985e4142761bd7464ea6edb44dc0025c40302b5e45ea0d865702',
          'config.yaml':'6401af2a14d41c6582c3181a0b683e8970c1eeefe8cf7afaf74f1597da3fcd72'}}


class OfficialCodec(nn.Module):
    def __init__(self, upstream, artifact_dir, device='cuda:1'):
        super().__init__()
        upstream=Path(upstream).resolve();artifact_dir=Path(artifact_dir).resolve()
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip()
        if revision!=UPSTREAM_COMMIT:raise ValueError('Unreviewed upstream revision')
        if subprocess.check_output(['git','status','--porcelain'],cwd=upstream,text=True).strip():
            raise ValueError('Upstream implementation has local changes')
        sys.path.insert(0,str(upstream))
        from models.vqvae import VQVAE
        artifact_hashes={name:digest(artifact_dir/name) for name in ('tokenizer.pt','config.yaml')}
        if artifact_hashes not in ARTIFACTS.values():raise ValueError('Unreviewed official codec artifact hashes')
        cfg=OmegaConf.load(artifact_dir/'config.yaml')
        self.model=VQVAE(**cfg.tokenizer.vqvae)
        ck=torch.load(artifact_dir/'tokenizer.pt',weights_only=True,map_location='cpu')
        self.model.load_state_dict(ck['model'],strict=True)
        self.model.to(device);self.model.eval();self.model.requires_grad_(False)
        tc=cfg.tokenizer.pre_tokenizer
        if tc.name!='hf':raise ValueError('Only audited HF pre-tokenizers supported')
        if tc.model_id not in PRETOKENIZER_REVISIONS:raise ValueError('Unaudited or lossy pre-tokenizer')
        self.tokenizer=AutoTokenizer.from_pretrained(tc.model_id,revision=PRETOKENIZER_REVISIONS[tc.model_id],trust_remote_code=False)
        special=dict(tc.special_tokens)
        # Match upstream special-token handling, including GPT-2 added PAD.
        self.tokenizer.eos_token=special['eos_token']
        pad=special['pad_token']
        if pad!=self.tokenizer.eos_token and self.tokenizer.convert_tokens_to_ids(pad) in (None,self.tokenizer.unk_token_id,self.tokenizer.eos_token_id):
            self.tokenizer.add_special_tokens({'pad_token':pad})
        else:self.tokenizer.pad_token=pad
        levels=tuple(self.model.vq.levels)
        sizes=[e.num_embeddings for e in self.model.vq.embeddings]
        if len(set(sizes))!=1:raise ValueError('Generator requires uniform codebook cardinality')
        self.config=SimpleNamespace(length=self.model.context_length,width=self.model.embed_dim,
                                    levels=levels,codes=sizes[0])
        self.identity={'kind':'official-ncm-codec','upstream':revision,'release_revision':HF_REVISION,
            'weights_sha256':artifact_hashes['tokenizer.pt'],'config_sha256':artifact_hashes['config.yaml'],
            'pre_tokenizer':tc.model_id,'levels':levels,'window_tokens':self.config.length}

    @property
    def device(self):return next(self.model.parameters()).device

    def encode_text(self,text):return self.tokenizer.encode(text,add_special_tokens=False)

    def decode_text(self,ids):return self.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)

    def windows(self,ids):
        n=self.config.length;p=self.tokenizer.pad_token_id
        return torch.tensor([ids[i:i+n]+[p]*max(0,i+n-len(ids)) for i in range(0,len(ids),n)],dtype=torch.long)

    @torch.no_grad()
    def targets(self,ids):
        flat=self.model.encode(ids.to(self.device))
        codes=list(flat.split(self.config.levels,dim=1))
        first=torch.zeros(ids.shape[0],self.config.levels[0],self.config.width,device=self.device)
        inputs=[first]+list(self.model.vq.idx_to_ncm_input(flat).split(self.config.levels[1:],dim=1))
        return codes,inputs

    @torch.no_grad()
    def reconstruct(self,ids):return self.model.decode(self.model.encode(ids.to(self.device))).argmax(-1)

    @torch.no_grad()
    def generate_with(self,generator,prompt_ids):
        """Use upstream residual transforms, not the pilot's linear interpolation."""
        batch=prompt_ids.shape[0]
        z=torch.zeros(batch,self.config.length,self.config.width,device=self.device)
        inputs=torch.zeros(batch,self.config.levels[0],self.config.width,device=self.device)
        codes=[]
        for level,book in enumerate(self.model.vq.embeddings):
            ids=generator(inputs,prompt_ids,level).argmax(-1).to(self.device)
            codes.append(ids)
            z,inputs=self.model.vq.get_next_autoregressive_input(level,z,book(ids))
        return self.model.decode(torch.cat(codes,dim=1)).argmax(-1),codes


PROBES=[
 'def sumValues(xs):\n    return sum(x for x in xs if x is not None)\n',
 'const API_KEY = "AbCd0123";\nlet api_key = "abcd0123";\n',
 '{"id":"0O1lI","price":0.000001,"enabled":false,"x":null}\n',
 'def café(値):\n\treturn "é🧪" + str(値)\n',
 'if (a != b && x <= 0xFF) { return a << 2; }\r\n',
 'hash = "c1e0123456789abcdef"\nother = "cle0123456789abcdef"\n',
]


@torch.no_grad()
def evaluate(codec,documents):
    reports=[];tokens=correct=windows=exact_windows=0
    for ident,text in documents:
        ids=codec.encode_text(text)
        pre_exact=codec.decode_text(ids)==text
        pieces=codec.windows(ids);pred=[];row_correct=0;row_exact=0
        for i in range(len(pieces)):
            got=codec.reconstruct(pieces[i:i+1])[0].tolist()
            expected=ids[i*codec.config.length:(i+1)*codec.config.length]
            got=got[:len(expected)]
            n=sum(a==b for a,b in zip(got,expected));row_correct+=n;row_exact+=int(got==expected);pred.extend(got)
        rendered=codec.decode_text(pred)
        reports.append({'id':ident,'text_sha256':__import__('hashlib').sha256(text.encode()).hexdigest(),
            'tokens':len(ids),'pre_tokenizer_exact':pre_exact,'correct_tokens':row_correct,
            'exact_windows':row_exact,'windows':len(pieces),'exact_text':rendered==text})
        tokens+=len(ids);correct+=row_correct;windows+=len(pieces);exact_windows+=row_exact
    n=len(reports)
    return {'documents':n,'tokens':tokens,'token_accuracy':correct/tokens,'exact_window_rate':exact_windows/windows,
        'exact_document_rate':sum(r['exact_text'] for r in reports)/n,
        'pre_tokenizer_exact_rate':sum(r['pre_tokenizer_exact'] for r in reports)/n,
        'passed':correct/tokens>=.9999 and all(r['exact_text'] for r in reports),
        'gate':'all screened documents byte-exact and token accuracy >= 99.99%; screening is not a release guarantee',
        'rows':reports}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',default='results/ncm-upstream');p.add_argument('--artifacts',required=True)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:1');p.add_argument('--per-source',type=int,default=16)
    p.add_argument('--checkpoint',help='Optional adapted decoder or complete codec checkpoint')
    a=p.parse_args();torch.set_num_threads(2);admission([a.device],4)
    codec=OfficialCodec(a.upstream,a.artifacts,a.device)
    if a.checkpoint:
        checkpoint=torch.load(a.checkpoint,weights_only=True,map_location='cpu')
        if checkpoint['identity']['codec']!=codec.identity:raise ValueError('Adapted checkpoint belongs to another codec')
        if 'model' in checkpoint:codec.model.load_state_dict(checkpoint['model'])
        else:codec.model.decoder.load_state_dict(checkpoint['decoder'])
        codec.model.eval()
    rows,manifest=load_data(a.data);chosen=[]
    for source in sorted({r['source'] for r in rows}):
        subset=sorted([r for r in rows if r['source']==source and r['split']=='validation'],key=lambda r:r['id'])[:a.per_source]
        chosen.extend((r['id'],r['target']) for r in subset)
    report={'identity':codec.identity,'data_sha256':manifest['examples_sha256'],
        'probes':evaluate(codec,[(f'probe-{i}',s) for i,s in enumerate(PROBES)]),
        'validation':evaluate(codec,chosen),'quality_ready':False}
    if a.checkpoint:
        report['checkpoint_sha256']=digest(a.checkpoint)
        report['selected_step']=checkpoint['step']
    report['codec_gate_passed']=report['probes']['passed'] and report['validation']['passed']
    if Path(a.output).exists():raise ValueError('Report already exists')
    write_json(a.output,report)
    print(json.dumps({k:({x:y for x,y in v.items() if x!='rows'} if isinstance(v,dict) else v) for k,v in report.items()},indent=2))


if __name__=='__main__':main()
