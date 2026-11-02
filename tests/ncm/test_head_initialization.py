import unittest
import torch
from scripts.ncm.geometry import IndependentConceptHeads,TiedResidualHeads
from scripts.ncm.initialize_independent import initialize_from_geometry


class InitializationTests(unittest.TestCase):
    def test_expansion_preserves_scores_with_and_without_projection_bias(self):
        torch.manual_seed(3)
        for nearest in (False,True):
            for normalized in (False,True):
                books=[torch.nn.Embedding(13,4),torch.nn.Embedding(11,4)]
                source=TiedResidualHeads(8,books,input_norm=normalized,nearest_code=nearest)
                with torch.no_grad():source.log_temperature.copy_(torch.tensor([1.,-.5]))
                before={k:v.clone() for k,v in source.state_dict().items()}
                target=IndependentConceptHeads(8,books,input_norm=normalized)
                initialize_from_geometry(target,source)
                x=torch.randn(2,7,8)
                for level in range(2):
                    torch.testing.assert_close(target(x,level),source(x,level),rtol=2e-5,atol=2e-5)
                    self.assertTrue(torch.equal(target(x,level).argmax(-1),source(x,level).argmax(-1)))
                for k,v in before.items():torch.testing.assert_close(v,source.state_dict()[k])
                target(x,0).sum().backward()
                self.assertIsNotNone(target.classifiers[0].weight.grad)
                self.assertIsNone(source.projections[0].weight.grad)

    def test_normalization_mismatch_is_rejected(self):
        books=[torch.nn.Embedding(3,2)]
        with self.assertRaises(ValueError):
            initialize_from_geometry(IndependentConceptHeads(4,books),TiedResidualHeads(4,books,input_norm=True))


if __name__=='__main__':unittest.main()
