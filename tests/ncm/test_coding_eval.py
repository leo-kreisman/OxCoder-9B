import shutil
import sys
import unittest
from scripts.ncm.coding_eval import check_case


@unittest.skipUnless(sys.platform=='linux' and shutil.which('bwrap'),'Linux bubblewrap required')
class ExecutionTests(unittest.TestCase):
    def test_correct_and_incorrect_are_graded_outside_worker(self):
        self.assertTrue(check_case('def add(a,b): return a+b','add',[2,3],5)['passed'])
        self.assertFalse(check_case('def add(a,b): return a-b','add',[2,3],5)['passed'])

    def test_early_exit_cannot_report_a_pass(self):
        self.assertFalse(check_case('import os\nos._exit(0)','add',[2,3],5)['passed'])

    def test_mutation_and_filesystem_isolation(self):
        code='def sort_it(items):\n items.sort()\n return items'
        self.assertFalse(check_case(code,'sort_it',[[2,1]],[1,2],preserve_arguments=True)['passed'])
        code='def read_host():\n return open("/home/scribe/Projects/OxCoder-9B/README.md").read()'
        self.assertTrue(check_case(code,'read_host',[],{'raises':'FileNotFoundError'})['passed'])


if __name__=='__main__':unittest.main()
