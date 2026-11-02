"""Evaluate byte-codec reconstruction on whole held-out responses and syntax probes."""
import argparse
import hashlib
import torch
from .codec import Codec,CodecConfig,encode,decode,PAD
from .data import chunks
from .official_codec import PROBES
from .adapt_codec import documents
from .backbone import digest
from .train import load_data,write_json,admission


@torch.no_grad()
def evaluate(codec,docs,device):
    rows=[];correct=total=exact_chunks=total_chunks=valid=0
    for ident,text in docs:
        pieces=list(chunks(text,codec.config.length-1));ids=torch.tensor([encode(t,codec.config.length) for t in pieces],device=device)
        outputs=[]
        for start in range(0,len(ids),16):
            codes,_=codec.targets(ids[start:start+16]);outputs.append(codec.reconstruct_codes(codes).argmax(-1))
        got=torch.cat(outputs);mask=ids!=PAD
        correct+=int(((got==ids)&mask).sum());total+=int(mask.sum())
        matches=int(((got==ids)|~mask).all(-1).sum());exact_chunks+=matches;total_chunks+=len(pieces)
        try:rendered=''.join(decode(p) for p in got.tolist());is_valid=True
        except (ValueError,UnicodeError):rendered=None;is_valid=False
        valid+=is_valid
        rows.append({'id':ident,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'chunks':len(pieces),'exact_chunks':matches,'valid_utf8_and_eos':is_valid,'exact_text':rendered==text})
    return {'documents':len(rows),'nonpad_symbol_accuracy':correct/total,'exact_chunk_rate':exact_chunks/total_chunks,
        'exact_document_rate':sum(r['exact_text'] for r in rows)/len(rows),'valid_document_rate':valid/len(rows),
        'passed':all(r['exact_text'] for r in rows) and correct/total>=.9999,'rows':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:1');p.add_argument('--per-source',type=int,default=16)
    a=p.parse_args();torch.set_num_threads(2);admission([a.device],4)
    ck=torch.load(a.checkpoint,weights_only=True,map_location='cpu')
    config=ck.get('config',ck['identity']['config']);codec=Codec(CodecConfig(**config)).to(a.device)
    codec.load_state_dict(ck['model']);codec.eval();rows,manifest=load_data(a.data)
    report={'checkpoint_sha256':digest(a.checkpoint),'corpus_sha256':manifest['examples_sha256'],
        'checkpoint_step':ck['step'],'validation':evaluate(codec,documents(rows,a.per_source),a.device),
        'probes':evaluate(codec,[(f'probe-{i}',s) for i,s in enumerate(PROBES)],a.device),
        'quality_ready':False,'scope':'byte-exact whole response reconstruction, not generation or coding quality'}
    report['passed']=report['validation']['passed'] and report['probes']['passed']
    write_json(a.output,report)
