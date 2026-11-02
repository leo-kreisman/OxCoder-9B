"""Full-shape CUDA gradient/memory smoke for decoded-text native NCM training."""
import argparse
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F

from .backbone import digest
from .codec import Codec,CodecConfig
from .context_cache import FeatureCache
from .context_decoder import ContextConceptDecoder,DecoderConfig
from .context_regularization import install_dropout
from .context_residual import conditioned_targets
from .decoded_supervision import decoded_loss
from .train_context_decoder import batch,make_optimizer
from .train import admission,write_json


def run(a):
    torch.set_num_threads(2);torch.manual_seed(28001);admission([a.device],8)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    state=torch.load(a.checkpoint,weights_only=True,map_location='cpu');identity=state['identity']
    if digest(a.codec)!=identity['codec_sha256']:raise ValueError('Wrong codec')
    config=DecoderConfig(**identity['config']);model=ContextConceptDecoder(config).to(a.device)
    model.load_state_dict(state['model']);model.checkpoint_blocks=True;install_dropout(model,.1)
    ck=torch.load(a.codec,weights_only=True,map_location='cpu');codec=Codec(CodecConfig(**ck['config'])).to(a.device)
    codec.load_state_dict(ck['model']);codec.eval().requires_grad_(False)
    cache=FeatureCache(a.cache,config);opt=make_optimizer(model,.0002)
    if cache.sha!=identity['feature_manifest_sha256']:raise ValueError('Wrong parent features')
    # Fixed long training prompts exercise maximum working memory. No eval rows.
    ids=sorted((i for i,r in enumerate(cache.records) if r['split']=='train'),
        key=lambda i:(-cache.records[i]['prompt_tokens'],cache.records[i]['id']))[:8]
    b=batch(cache,ids,a.device);records=[]
    for update in range(3):
        torch.cuda.synchronize(a.device);start=time.monotonic();opt.zero_grad();model.eval()
        inputs,labels,mask=conditioned_targets(model,codec,b['hidden'],b['valid'],b['bytes'],.5)
        model.train()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits,stop=model(inputs,b['hidden'],b['valid'])
            ce=sum(F.cross_entropy(x.flatten(0,1),y.flatten()) for x,y in zip(logits,labels))/len(labels)
            stop_loss=F.binary_cross_entropy_with_logits(stop,b['stop'])
        byte_loss,_=decoded_loss(codec,inputs[-1],logits[-1],b['bytes'])
        loss=ce+.1*stop_loss+.1*byte_loss
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
        loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step()
        torch.cuda.synchronize(a.device)
        records.append({'update':update+1,'loss':float(loss.detach()),'code_ce':float(ce.detach()),
            'byte_ce':float(byte_loss.detach()),'gradient_norm':norm,'seconds':time.monotonic()-start,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(a.device),
            'frozen_codec_has_gradient':any(p.grad is not None for p in codec.parameters())})
    report={'complete':True,'passed':all(not r['frozen_codec_has_gradient'] for r in records),
        'checkpoint_sha256':digest(a.checkpoint),'feature_manifest_sha256':cache.sha,'records':records,
        'implementation_sha256':{n:digest(Path(__file__).with_name(n)) for n in
            ('audit_decoded_objective.py','train_context_corpus.py','decoded_supervision.py','context_regularization.py')},
        'scope':'Three training-only update smoke; no output weights saved and no coding-quality claim', 'quality_ready':False}
    write_json(Path(a.output),report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('checkpoint','codec','cache','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--device',default='cuda:1');run(p.parse_args())
