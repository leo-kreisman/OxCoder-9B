"""Count byte confusions on validation data without exporting source text."""
import argparse
from collections import Counter
import torch
from .codec import Codec,CodecConfig,encode,PAD
from .backbone import digest
from .train import load_data,write_json,admission


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    a=p.parse_args();torch.set_num_threads(2);admission([a.device],4)
    ck=torch.load(a.checkpoint,weights_only=True,map_location='cpu');m=Codec(CodecConfig(**ck['config'])).to(a.device)
    m.load_state_dict(ck['model']);m.eval();rows,manifest=load_data(a.data)
    if ck['identity']['data_sha256']!=manifest['examples_sha256']:raise ValueError('Data/checkpoint mismatch')
    rows=[r for r in rows if r['split']=='validation'];bad=Counter();counts=Counter()
    for i in range(0,len(rows),32):
        ids=torch.tensor([encode(r['target'],m.config.length) for r in rows[i:i+32]],device=a.device)
        codes,_=m.targets(ids);pred=m.reconstruct_codes(codes).argmax(-1)
        for expected,got in zip(ids.flatten().tolist(),pred.flatten().tolist()):
            if expected==PAD:continue
            counts[expected]+=1
            if expected!=got:bad[(expected,got)]+=1
    write_json(a.output,{'checkpoint_sha256':digest(a.checkpoint),'checkpoint_step':ck['step'],
        'data_sha256':manifest['examples_sha256'],'errors':sum(bad.values()),'symbols':sum(counts.values()),
        'top_confusions':[{'expected':x,'got':y,'errors':n,'expected_occurrences':counts[x]}
                          for (x,y),n in bad.most_common(30)]})


if __name__=='__main__':main()
