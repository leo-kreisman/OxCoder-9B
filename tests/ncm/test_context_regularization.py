import unittest
import torch
from scripts.ncm.context_decoder import ContextConceptDecoder,DecoderConfig
from scripts.ncm.context_regularization import install_dropout


class TrainingDropoutTests(unittest.TestCase):
    def test_eval_is_identical_and_checkpoint_inventory_is_unchanged(self):
        torch.set_num_threads(1);torch.manual_seed(81)
        m=ContextConceptDecoder(DecoderConfig(source_width=16,latent_width=8,width=32,heads=4,layers=2,codes=17,levels=(1,3,8)))
        inputs=[torch.randn(2,n,8) for n in (1,3,8)];hidden=torch.randn(2,5,16);valid=torch.ones(2,5,dtype=torch.bool)
        m.eval();before=m(inputs,hidden,valid);keys=list(m.state_dict());install_dropout(m,.2)
        after=m(inputs,hidden,valid)
        self.assertEqual(keys,list(m.state_dict()))
        for x,y in zip(before[0],after[0]):self.assertTrue(torch.equal(x,y))
        self.assertTrue(torch.equal(before[1],after[1]))
        m.train();trained=m(inputs,hidden,valid)
        self.assertTrue(any(not torch.equal(x,y) for x,y in zip(before[0],trained[0])))
        with self.assertRaises(ValueError):install_dropout(m,.2)


if __name__=='__main__':unittest.main()
