"""Oracle residual classification control; never a generated-code benchmark."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import torch
from .codec import Codec,CodecConfig
from .geometry import TiedResidualHeads
from .backbone import digest
from .train import write_json


def main():
    p=argparse.ArgumentParser()
    for k in ['codec','checkpoint','counts','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    c=torch.load(a.codec,map_location='cpu',weights_only=True)
    codec=Codec(CodecConfig(**c['config']));codec.load_state_dict(c['model']);codec.eval()
    # The trainer atomically replaces latest.pt. Hash and load the same byte
    # snapshot so a concurrent save cannot misattribute this control result.
    raw=Path(a.checkpoint).read_bytes();checkpoint_sha=hashlib.sha256(raw).hexdigest()
    ck=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True);identity=ck['identity']
    del raw
    if identity['codec_sha256']!=digest(a.codec) or identity.get('head_kind')!='geometry':
        raise ValueError('Checkpoint does not match this geometry codec')
    raw=Path(a.counts).read_bytes();counts_sha=hashlib.sha256(raw).hexdigest()
    counts=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    del raw
    if len(counts)!=len(codec.books):raise ValueError('Code-count level inventory differs')
    for count,book in zip(counts,codec.books):
        if count.shape!=(book.num_embeddings,) or not torch.isfinite(count).all() or (count<0).any() or count.sum()<=0:
            raise ValueError('Invalid admitted code counts')
    width=ck['model']['head.projections.0.weight'].shape[1]
    head=TiedResidualHeads(width,codec.books,input_norm=identity.get('head_input_norm',False),
                           nearest_code=identity.get('nearest_code',False))
    states={name[5:]:v for name,v in ck['model'].items() if name.startswith('head.')}
    expected={name for name,v in head.named_parameters() if v.requires_grad}
    if set(states)!=expected:raise ValueError('Head inventory differs')
    with torch.no_grad():
        for name,v in head.named_parameters():
            if v.requires_grad:v.copy_(states[name])
        records=[]
        for i,(book,count) in enumerate(zip(codec.books,counts)):
            used=(count>0).nonzero().flatten();correct=vector_correct=weighted=0
            for ids in used.split(128):
                got=head.score_residual(book.weight[ids],i).argmax(-1)
                good=got==ids
                correct+=int(good.sum());weighted+=float(count[ids][good].sum())
                vector_correct+=int(torch.isclose(book.weight[got],book.weight[ids],rtol=1e-5,atol=1e-5).all(-1).sum())
            records.append({'level':i,'observed_codes':len(used),'oracle_code_identity_accuracy':correct/len(used),
                            'oracle_vector_accuracy':vector_correct/len(used),
                            'count_weighted_identity_accuracy':weighted/float(count.sum())})
    report={'checkpoint_sha256':checkpoint_sha,'counts_sha256':counts_sha,
            'checkpoint_step':ck['step'],'records':records,'quality_ready':False,
            'scope':'exact code vectors supplied to classifier; no generator reasoning, coding quality or inference-speed claim'}
    write_json(a.output,report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
