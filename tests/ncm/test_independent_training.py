import unittest
import torch
from scripts.ncm.codec import Codec,CodecConfig
from scripts.ncm.train_independent import corrupted_inputs,validate_continuation,validate_expansion
from scripts.ncm.prepare_creator_expansion import frozen_evaluation_hash


class ConditioningTests(unittest.TestCase):
    def test_data_expansion_binds_parent_and_preserves_heldout_inventory(self):
        prepared=[({'group':'v','split':'validation'},{'ids':[1]}),
                  ({'group':'t','split':'train'},{'ids':[2]})]
        manifest={'base_prepared_sha256':'base','evaluation_changed':False,'base_rows':1,
                  'frozen_evaluation_sha256':frozen_evaluation_hash(prepared)}
        validate_expansion(manifest,{'prepared':prepared},{'prepared_sha256':'base'})
        with self.assertRaises(ValueError):validate_expansion(manifest,{'prepared':prepared},{'prepared_sha256':'other'})
        prepared[0][1]['ids']=[9]
        with self.assertRaises(ValueError):validate_expansion(manifest,{'prepared':prepared},{'prepared_sha256':'base'})

    def test_resource_continuation_cannot_change_data_or_learning_recipe(self):
        original={'source_sha256':'a','codec_sha256':'b','head_lr':.001,
                  'implementation_sha256':{'trainer':'old'},'seed':12}
        changed={**original,'allocator_cap_gib':11.25,'implementation_sha256':{'trainer':'new'}}
        validate_continuation(original,changed)
        for key,value in [('codec_sha256','different'),('head_lr',.01),('seed',13)]:
            with self.assertRaises(ValueError):validate_continuation(original,{**changed,key:value})

    def test_zero_corruption_preserves_inputs_and_nonzero_preserves_labels(self):
        torch.manual_seed(7)
        codec=Codec(CodecConfig(length=8,width=8,heads=2,layers=1,codes=16,levels=(1,2,4,8))).eval()
        target=torch.tensor([[65,66,67,256,257,257,257,257]])
        with torch.no_grad():codes,clean=codec.targets(target)
        originals=[x.clone() for x in codes]
        for a,b in zip(clean,corrupted_inputs(codec,codes,0)):
            torch.testing.assert_close(a,b)
        changed=corrupted_inputs(codec,codes,1)
        torch.testing.assert_close(changed[0],clean[0])
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(changed[1:],clean[1:])))
        for a,b in zip(codes,originals):torch.testing.assert_close(a,b)
        self.assertTrue(all(not x.requires_grad for x in changed))
        with self.assertRaises(ValueError):corrupted_inputs(codec,codes,1.1)


if __name__=='__main__':unittest.main()
