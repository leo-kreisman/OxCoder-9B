import unittest
import torch
from scripts.ncm.representation_alignment import alignment_map, pool_teacher, cosine_alignment


class RepresentationAlignmentTests(unittest.TestCase):
    def test_causal_shift_and_padding(self):
        weights, valid = alignment_map('!abcd', 1, [(0,1),(1,3),(3,5)], 8)
        self.assertEqual(weights.argmax(-1)[:5].tolist(), [0,0,1,1,2])
        self.assertEqual(valid.tolist(), [True]*5+[False]*3)
        h = torch.tensor([[1.,0.],[0.,1.],[2.,2.]])
        targets, masks = pool_teacher(h, weights, valid, [2,8])
        self.assertTrue(torch.equal(targets[0], torch.tensor([[.5,.5],[2.,2.]])))
        self.assertTrue(torch.equal(masks[1], valid))

    def test_unicode_overlap_does_not_duplicate_byte_mass(self):
        # Two byte-fallback tokens share the character span of é.
        w, valid = alignment_map('!éz', 1, [(0,1),(1,2),(1,2),(2,3)], 8)
        self.assertTrue(torch.equal(w[:2,:2], torch.full((2,2), .5)))
        self.assertTrue(torch.equal(w[valid].sum(-1), torch.ones(4)))
        self.assertEqual(w[2].argmax().item(), 2)

    def test_token_can_cross_prefix_target_boundary(self):
        w, _ = alignment_map('! xyz', 3, [(0,1),(1,4),(4,5)], 8)
        self.assertEqual(w[:3].argmax(-1).tolist(), [0,1,2])

    def test_reject_gaps_and_missing_causal_context(self):
        for offsets in [[(0,1),(2,3)], [(0,3)]]:
            with self.assertRaises(ValueError):
                alignment_map('!ab', 1, offsets, 8)

    def test_only_student_has_gradients(self):
        student = torch.randn(1,4,7, requires_grad=True)
        teacher = torch.randn(1,4,7, requires_grad=True)
        valid = torch.tensor([[True,True,False,False]])
        cosine_alignment(student, teacher, valid).backward()
        self.assertIsNone(teacher.grad)
        self.assertEqual(student.grad[:,2:].abs().sum().item(), 0)
        self.assertGreater(student.grad[:,:2].abs().sum().item(), 0)

    def test_pooling_never_retains_teacher_graph(self):
        h = torch.randn(3,4, requires_grad=True)
        w, valid = alignment_map('!ab', 1, [(0,1),(1,2),(2,3)], 8)
        targets, _ = pool_teacher(h, w, valid, [2,8])
        self.assertTrue(all(not x.requires_grad for x in targets))


if __name__ == '__main__':
    unittest.main()
