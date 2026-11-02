import unittest
import torch
from scripts.ncm.audit_context_coverage import coverage


class CoverageTests(unittest.TestCase):
    def test_weighted_token_coverage_separates_rare_missing_codes(self):
        r=coverage([torch.tensor([2,0,7,0])],[torch.tensor([1,1,8,0])])[0]
        self.assertEqual(r['unseen_validation_codes'],1)
        self.assertEqual(r['validation_tokens_with_unseen_code'],1)
        self.assertEqual(r['maximum_accuracy_if_only_training_codes_predicted'],.9)
        self.assertEqual(r['observed_training_codes'],2)


if __name__ == '__main__': unittest.main()
