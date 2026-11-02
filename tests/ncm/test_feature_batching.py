import unittest
import torch
from scripts.ncm.audit_feature_batching import encode_batch
from scripts.ncm.extract_context_reuse import groups


class ToyCausal:
    def embed_prompt(self, ids):return ids.float()[:,:,None]
    def __call__(self, hidden):return hidden.cumsum(1)


class BatchFeatureTests(unittest.TestCase):
    def test_right_padding_does_not_cross_examples_or_modify_real_tokens(self):
        source=ToyCausal();sequences=[[4,9],[1,3,5,7],[10]]
        actual=encode_batch(source,sequences)
        for row,values in zip(actual,sequences):
            expected=source(source.embed_prompt(torch.tensor([values])))[0]
            self.assertTrue(torch.equal(row,expected))
        with self.assertRaises(ValueError):encode_batch(source,[[]])

    def test_length_groups_preserve_inventory_and_token_budget(self):
        inputs=[(i,{}, {'ids':list(range(n))}) for i,n in enumerate([1800,100,900,110,2000,500,120])]
        batches=list(groups(inputs,4,4096))
        self.assertEqual(sorted(x[0] for g in batches for x in g),list(range(len(inputs))))
        for group in batches:
            self.assertLessEqual(len(group),4)
            self.assertLessEqual(len(group)*max(len(x[2]['ids']) for x in group),4096)


if __name__=='__main__':unittest.main()
