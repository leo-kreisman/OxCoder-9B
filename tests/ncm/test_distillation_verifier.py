import unittest
from scripts.ncm.verify_distillation import decode_test, extract_python, execute


class DistillationVerifierTests(unittest.TestCase):
    def test_valid_python_and_string_escapes_are_preserved(self):
        text = "assert f('é\\n') == 2\n"
        self.assertEqual(decode_test(text), text)

    def test_encoded_file_newlines_preserve_unicode_and_string_literals(self):
        text = r"\nassert f('é\n') == 2\n"
        self.assertEqual(decode_test(text), "\nassert f('é\\n') == 2\n")

    def test_comments_and_quotes(self):
        text = r"\n# don't parse this quote\nassert f('x') == 1\n"
        self.assertEqual(decode_test(text), "\n# don't parse this quote\nassert f('x') == 1\n")

    def test_no_assert_is_rejected(self):
        with self.assertRaises(ValueError):
            decode_test('f(1)')

    def test_multiple_blocks_are_rejected(self):
        with self.assertRaises(ValueError):
            extract_python('```python\nx=1\n```\n```python\nx=2\n```')

    def test_sandbox_executes_actual_assertions(self):
        self.assertTrue(execute('def f(x): return x+1', 'assert f(1)==2')['passed'])
        self.assertFalse(execute('def f(x): return x', 'assert f(1)==2')['passed'])
        # A definition-only test must not pass without executing its assertion.
        self.assertFalse(execute('def f(x): return x', 'def test_it():\n assert f(1)==1')['passed'])


if __name__ == '__main__':
    unittest.main()
