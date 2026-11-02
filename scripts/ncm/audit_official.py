"""Check official-codec level scheduling against teacher-forced upstream codes."""
import argparse
import torch
from .official_codec import OfficialCodec,PROBES
from .train import write_json


@torch.no_grad()
def audit(codec):
    ids=codec.windows(codec.encode_text(PROBES[0]))[:1].to(codec.device)
    codes,inputs=codec.targets(ids);deltas=[]
    class Oracle:
        def __call__(self,latent,prompt,level):
            deltas.append(float((latent-inputs[level]).abs().max()))
            result=torch.full((*codes[level].shape,codec.config.codes),-1.,device=codec.device)
            return result.scatter_(-1,codes[level].unsqueeze(-1),1.)
    reconstructed,generated=codec.generate_with(Oracle(),torch.zeros(1,1,dtype=torch.long,device=codec.device))
    flat=torch.cat(codes,1);middle=len(codes)//2
    altered=list(codes);altered[middle]=(altered[middle]+1)%codec.config.codes
    changed=codec.model.vq.idx_to_ncm_input(torch.cat(altered,1)).split(codec.config.levels[1:],dim=1)
    earlier_equal=all(torch.equal(inputs[level],changed[level-1]) for level in range(1,middle+1))
    later_changed=any(not torch.equal(inputs[level],changed[level-1]) for level in range(middle+1,len(codes)))
    parity=torch.equal(reconstructed,codec.model.decode(flat).argmax(-1))
    result={'identity':codec.identity,'level_input_max_abs_errors':deltas,
        'teacher_codes_reproduced':all(torch.equal(a,b) for a,b in zip(codes,generated)),
        'current_or_future_codes_do_not_change_earlier_inputs':earlier_equal,
        'changed_code_affects_later_input':later_changed,'oracle_output_matches_reconstruction':parity,
        'scope':'scheduler correctness only; oracle supplies known codes, not a trained generator'}
    result['passed']=max(deltas)<1e-5 and earlier_equal and later_changed and parity and result['teacher_codes_reproduced']
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--upstream',default='results/ncm-upstream')
    p.add_argument('--artifacts',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    a=p.parse_args();torch.set_num_threads(2)
    result=audit(OfficialCodec(a.upstream,a.artifacts,a.device));write_json(a.output,result)
    if not result['passed']:raise RuntimeError('Official scheduler audit failed')
