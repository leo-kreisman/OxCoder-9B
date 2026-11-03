import unittest

from scripts.ncm.audit_generation_gap import selected_rows, diagnostic


class GenerationGapTests(unittest.TestCase):
    def test_selection_is_response_disjoint_and_prefers_initial_chunks(self):
        rows = [({'id': 'c:byte-1', 'group': 'a'}, {}),
                ({'id': 'b:byte-0', 'group': 'a'}, {}),
                ({'id': 'a:byte-0', 'group': 'b'}, {})]
        self.assertEqual([r['id'] for r, _ in selected_rows(rows, 2)], ['a:byte-0', 'b:byte-0'])

    def test_oracle_modes_are_labeled_and_do_not_update_model_or_codec(self):
        import torch
        import test_frozen_clean_objective as fixtures
        fixture = fixtures.CanonicalObjectiveTests()
        fixture.setUp()
        model, codec = fixture.model.eval(), fixture.codec
        row, condition = fixture.items[0]
        row = {**row, 'id': 'fixture:byte-0', 'source': 'fixture'}
        versions = [(id(p), p._version) for p in list(model.parameters()) + list(codec.parameters())]
        result = diagnostic(model, codec, row, condition)
        self.assertEqual([r['gold_prefix_levels'] for r in result['rollouts']], list(range(len(codec.books) + 1)))
        self.assertFalse(result['rollouts'][0]['privileged_diagnostic'])
        self.assertTrue(all(r['privileged_diagnostic'] for r in result['rollouts'][1:]))
        self.assertEqual(result['rollouts'][-1]['code_accuracy'], [1.] * len(codec.books))
        self.assertEqual(versions, [(id(p), p._version) for p in list(model.parameters()) + list(codec.parameters())])
        self.assertTrue(all(p.grad is None for p in list(model.parameters()) + list(codec.parameters())))


if __name__ == '__main__':
    unittest.main()
