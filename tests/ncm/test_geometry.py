import random
import unittest
import torch
from scripts.ncm.geometry import TiedResidualHeads,IndependentConceptHeads
from scripts.ncm.train_geometry import ResponseSampler


class GeometryTests(unittest.TestCase):
    def test_independent_heads_have_separate_gradients_and_no_residual(self):
        books=[torch.nn.Embedding(7,2),torch.nn.Embedding(5,2)]
        head=IndependentConceptHeads(4,books,input_norm=True)
        x=torch.randn(2,3,4,requires_grad=True)
        logits,residual=head(x,1,return_latent=True)
        self.assertIsNone(residual)
        self.assertEqual(logits.shape,(2,3,5))
        torch.testing.assert_close(logits,head(x*10,1))
        logits.square().mean().backward()
        self.assertIsNone(head.classifiers[0].weight.grad)
        self.assertGreater(head.classifiers[1].weight.grad.norm(),0)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(b.weight.grad is None for b in books))
        restored=IndependentConceptHeads(4,books,input_norm=True)
        restored.load_state_dict(head.state_dict())
        torch.testing.assert_close(head(x,0),restored(x,0))

    def test_distance_head_and_gradient(self):
        book=torch.nn.Embedding(3,2)
        with torch.no_grad():book.weight.copy_(torch.tensor([[0.,0.],[2.,0.],[0.,2.]]))
        head=TiedResidualHeads(2,[book])
        with torch.no_grad():head.projections[0].weight.copy_(torch.eye(2))
        x=torch.tensor([[[1.9,.1],[.1,1.9]]],requires_grad=True)
        logits,residual=head(x,0,True)
        self.assertEqual(logits.argmax(-1).tolist(),[[1,2]])
        torch.nn.functional.cross_entropy(logits.flatten(0,1),torch.tensor([1,2])).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(head.projections[0].weight.grad.norm(),0)
        self.assertIsNone(book.weight.grad)
        self.assertFalse(head.book_0.requires_grad)

    def test_sampler_excludes_validation_and_retains_sources(self):
        rows=[]
        for split in ['train','validation','test']:
            for source in ['a','b']:
                for distilled in [False,True]:
                    rows.append(({'id':f'{split}-{source}-{distilled}','split':split,'source':source,'distilled':distilled},
                                 {'coding_response':True,'response_id':f'{split}-{source}-{distilled}'}))
        sampler=ResponseSampler(rows);rng=random.Random(99)
        samples=[sampler.sample(rng)[0] for _ in range(200)]
        self.assertEqual({r['split'] for r in samples},{'train'})
        self.assertEqual({r['source'] for r in samples},{'a','b'})
        self.assertEqual({r['distilled'] for r in samples},{False,True})

    def test_normalized_head_resists_feature_scale_and_zero(self):
        head=TiedResidualHeads(4,[torch.nn.Embedding(8,2)],input_norm=True)
        x=torch.randn(2,3,4,requires_grad=True)
        torch.testing.assert_close(head(x,0),head(x*100,0))
        self.assertTrue(torch.isfinite(head(torch.zeros_like(x),0)).all())
        head(x,0).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_calibration_starts_at_training_priors_despite_norm_imbalance(self):
        book=torch.nn.Embedding(3,2)
        with torch.no_grad():book.weight.copy_(torch.tensor([[.001,.001],[12.,0.],[0.,12.]]))
        head=TiedResidualHeads(4,[book],input_norm=True)
        counts=torch.tensor([0.,100.,50.])
        report=head.calibrate([counts])
        logits=head(torch.randn(1,1,4)*100,0)
        expected=(counts+1)/(counts.sum()+3)
        torch.testing.assert_close(logits.softmax(-1)[0,0],expected)
        self.assertEqual(report[0]['observed_codes'],2)
        torch.nn.functional.cross_entropy(logits.flatten(0,1),torch.tensor([1])).backward()
        self.assertGreater(head.projections[0].weight.grad.norm(),0)

    def test_nearest_head_preserves_exact_residual_identity_despite_unequal_frequencies(self):
        book=torch.nn.Embedding(3,2)
        with torch.no_grad():book.weight.copy_(torch.tensor([[.001,.001],[12.,0.],[0.,12.]]))
        head=TiedResidualHeads(2,[book],nearest_code=True)
        head.calibrate([torch.tensor([100000.,1.,1.])])
        with torch.no_grad():
            head.projections[0].weight.copy_(torch.eye(2));head.projections[0].bias.zero_()
        self.assertEqual(head(book.weight[None],0).argmax(-1).tolist(),[[0,1,2]])
        self.assertFalse(head.biases[0].requires_grad)


if __name__=='__main__':unittest.main()
