import unittest
import torch
from scripts.ncm.shared_training import layout,attention_mask


def additive_recurrence(q,k,v,g,beta,initial_state=None,output_final_state=False,cu_seqlens=None,**kwargs):
    pieces=[];states=[]
    for i,(start,end) in enumerate(zip(cu_seqlens[:-1],cu_seqlens[1:])):
        part=q[:,int(start):int(end)].cumsum(1)
        if initial_state is not None:part=part+initial_state[i][None,None]
        pieces.append(part);states.append(part[0,-1])
    return torch.cat(pieces,1),torch.stack(states) if output_final_state else None


class SharedLayoutTests(unittest.TestCase):
    def test_prefix_gradient_sums_across_isolated_branches(self):
        for prefix in [64,69,128]:
            sizes=[4,16,64,128]
            plan=layout(prefix,sizes,'cpu')
            x=torch.randn(1,prefix+sum(sizes),1,2,dtype=torch.float64,requires_grad=True)
            actual=plan.gated_delta_rule(additive_recurrence,x,x,x,x,x)
            outputs=[x[:,:prefix].cumsum(1)];start=prefix
            for size in sizes:
                full=torch.cat([x[:,:prefix],x[:,start:start+size]],1).cumsum(1)
                outputs.append(full[:,-size:]);start+=size
            expected=torch.cat(outputs,1)
            torch.testing.assert_close(actual,expected)
            probe=torch.randn_like(actual)
            a=torch.autograd.grad((actual*probe).sum(),x,retain_graph=True)[0]
            b=torch.autograd.grad((expected*probe).sum(),x)[0]
            torch.testing.assert_close(a,b)

    def test_mask_forbids_cross_level_and_prefix_target_leaks(self):
        mask=attention_mask(64,[4,16], 'cpu',torch.float32)[0,0]
        self.assertTrue(torch.isneginf(mask[:64,64:]).all())
        self.assertTrue(torch.isneginf(mask[64:68,68:]).all())
        self.assertTrue(torch.isneginf(mask[68:,64:68]).all())
        self.assertTrue((mask[64:,:64]==0).all())
        self.assertTrue((mask[68:,68:]==0).all())
        with self.assertRaises(ValueError):layout(63,[4,16],'cpu')


if __name__=='__main__':unittest.main()
