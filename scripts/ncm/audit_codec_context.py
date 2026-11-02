"""Measure which input positions affect the first coarse latent query.

This is a continuous-encoder dependency probe, not a semantic quality score.
The frozen codec is never modified. A random projection avoids LayerNorm's
zero-sum direction obscuring otherwise nonzero gradients.
"""
import argparse
import json
import torch
from .codec import Codec,CodecConfig,encode,resize
from .backbone import digest
from .train import write_json


def dependency(codec,ids,area=False):
    captured=[]
    def capture(module,args,output):
        value=output.detach().requires_grad_(True)
        captured.append(value)
        return value
    hook=codec.embedding.register_forward_hook(capture)
    try:
        z=codec.encode_latents(ids)
        size=codec.config.levels[0]
        if area:
            coarse=torch.nn.functional.interpolate(z.transpose(1,2),size=size,mode='area').transpose(1,2)
        else:coarse=codec.resample(z,size)
        generator=torch.Generator(device='cpu').manual_seed(7341)
        direction=torch.randn(codec.config.width,generator=generator).to(z)
        gradient=torch.autograd.grad((coarse[0,0]*direction).sum(),captured[0])[0]
        norms=gradient[0].norm(dim=-1).detach().cpu()
        return {'positions_above_1e_9':(norms>1e-9).nonzero().flatten().tolist(),
                'gradient_norms':norms.tolist()}
    finally:hook.remove()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--codec',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(2)
    ck=torch.load(a.codec,map_location='cpu',weights_only=True)
    codec=Codec(CodecConfig(**ck['config']));codec.load_state_dict(ck['model'])
    codec.eval().requires_grad_(False)
    text=('def add(a, b):\n    return a + b\n'*8).encode()[:codec.config.length-1].decode()
    ids=torch.tensor([encode(text,codec.config.length)])
    actual=dependency(codec,ids)
    area=dependency(codec,ids,True)
    report={'codec_sha256':digest(a.codec),'codec_config':codec.metadata(),
            'actual_first_coarse_query':actual,'area_pooling_control':area,
            'input_positions':codec.config.length,'quality_ready':False,
            'scope':'One continuous-latent directional derivative; no generator, code-ID, reconstruction or coding score. Area is a counterfactual control, not an applied checkpoint conversion.'}
    write_json(a.output,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['actual_first_coarse_query','area_pooling_control']}))
    print('Actual reached positions:',actual['positions_above_1e_9'])
    print('Area-control reached positions:',len(area['positions_above_1e_9']))


if __name__=='__main__':main()
