import copy
import unittest

from scripts.ncm.compact_code_curriculum import compact_python, candidate
from scripts.ncm.verify_distillation import execute


class CompactCodeCurriculumTests(unittest.TestCase):
    def row(self):
        return {'id': 'training:answer', 'group': 'training', 'split': 'train',
            'source': 'nvidia/OpenCodeInstruct', 'distilled': True, 'teacher_sha256': 'teacher',
            'messages': [{'role': 'user', 'content': 'Implement scale(x) returning three times x.'}],
            'target': '```python\ndef scale(x):\n    """Multiply a number."""\n    # Explain the operation\n    return x * 3\n```'}

    def test_removes_only_presentation_and_retains_runtime_literals(self):
        code = compact_python('def value():\n    """Doc."""\n    s = "# literal"\n    return s\n')
        self.assertNotIn('Doc.', code)
        self.assertIn('# literal', code)
        scope = {}
        exec(code, scope)
        self.assertEqual(scope['value'](), '# literal')

    def test_empty_documented_body_remains_valid(self):
        code = compact_python('class Holder:\n    """Doc."""\n')
        self.assertIn('pass', code)
        compile(code, '<owned-fixture>', 'exec')

    def test_doc_introspection_is_rejected(self):
        for code in ['def scale(x):\n return scale.__doc__\n',
                     'def scale(x):\n return getattr(x, "__doc__")\n',
                     'def scale(x):\n return inspect.getsource(x)\n']:
            with self.assertRaises(ValueError):
                compact_python(code)

    def test_nontraining_is_never_eligible(self):
        row = self.row(); row['split'] = 'validation'
        with self.assertRaises(ValueError):
            candidate(row, 254)

    def test_source_and_teacher_provenance_survive_without_mutation(self):
        row = self.row(); before = copy.deepcopy(row)
        changed, reason = candidate(row, 254)
        self.assertIsNone(reason)
        self.assertEqual(row, before)
        self.assertEqual(changed['teacher_sha256'], 'teacher')
        self.assertTrue(changed['distilled'])
        self.assertEqual(changed['group'], row['group'])
        self.assertEqual(changed['transformation']['parent_id'], row['id'])
        self.assertFalse(changed['targets_are_verified'])
        self.assertIn('Omit comments', changed['messages'][-1]['content'])

    def test_prompt_contract_and_byte_limit_are_enforced(self):
        row = self.row(); row['messages'][0]['content'] += ' Include a docstring.'
        self.assertEqual(candidate(row, 254), (None, 'presentation_requested'))
        self.assertEqual(candidate(self.row(), 1), (None, 'length'))

    def test_publisher_checks_detect_doc_sensitive_transformation(self):
        original = 'def scale(x):\n    """Doc."""\n    return x * 3\n'
        compact = compact_python(original)
        functional = 'assert scale(4) == 12\nassert scale(-2) == -6\n'
        self.assertTrue(execute(original, functional)['passed'])
        self.assertTrue(execute(compact, functional)['passed'])
        documentation = 'assert scale.__doc__ == "Doc."\n'
        self.assertTrue(execute(original, documentation)['passed'])
        self.assertFalse(execute(compact, documentation)['passed'])


if __name__ == '__main__':
    unittest.main()
