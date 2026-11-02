import unittest
import json
from scripts.ncm.admit_creator_corpora import AdmissionFilter,instruction_keys
from scripts.ncm.coding_eval import TASKS
from scripts.ncm.open_corpus import shingles
from scripts.ncm.audit_oxcoder_heads import select_examples


def row(group,split,text,prompt=''):
    return {'group':group,'split':split,'target':text,'prompt':prompt}


class AdmissionTests(unittest.TestCase):
    def test_shared_instruction_wrapper_is_not_treated_as_a_heldout_answer(self):
        scaffold=' '.join('policy'+str(i) for i in range(50))
        prompt=lambda text:json.dumps([{'role':'developer','content':scaffold},{'role':'user','content':text}])
        keys=instruction_keys(prompt('question'))
        f=AdmissionFilter(set(),keys)
        f.register(row('v','validation','a reserved answer',prompt('reserved question')),protect=True)
        self.assertIsNone(f.consider(row('t','train','def new_function(): return 67291',prompt('new question'))))
        # A benchmark inside that wrapper is still protected, as is a target
        # containing the wrapper itself. Only prompt-to-prompt boilerplate is removed.
        f=AdmissionFilter(shingles(scaffold),keys)
        self.assertEqual(f.consider(row('t','train','new target',prompt('new'))),'benchmark_or_holdout_shingle')
        f=AdmissionFilter(set(),keys);f.register(row('v','test',scaffold),protect=True)
        self.assertEqual(f.consider(row('t','train',scaffold+' more')),'benchmark_or_holdout_shingle')
        self.assertEqual(f.consider(row('t','train','different output',prompt('new'))),'benchmark_or_holdout_shingle')

    def test_prompt_containing_heldout_target_is_excluded(self):
        protected=' '.join('heldout'+str(i) for i in range(40))
        f=AdmissionFilter(set());f.register(row('v','validation',protected),protect=True)
        self.assertEqual(f.consider(row('new','train','def answer(): return 1',protected)),
                         'benchmark_or_holdout_shingle')
        self.assertEqual(f.consider(row('v','train','some different content')),'protected_group')

    def test_benchmark_exact_and_reserved_splits_cannot_enter_training(self):
        text=' '.join('benchmark'+str(i) for i in range(40))
        f=AdmissionFilter(shingles(text))
        self.assertEqual(f.consider(row('a','train',text)),'benchmark_or_holdout_shingle')
        self.assertEqual(f.consider(row('b','test','unseen')),'reserved_nontraining')
        self.assertEqual(f.consider(row('c','train','output',TASKS[0][1])),'local_coding_task_prompt')
        f.register(row('old','train','def unique_fixture(x): return x + 891'))
        self.assertEqual(f.consider(row('d','train','def unique_fixture(x): return x + 891')),'existing_exact_target')

    def test_conflicting_frozen_group_fails(self):
        f=AdmissionFilter(set());f.register(row('same','test','abc'))
        with self.assertRaisesRegex(ValueError,'split conflict'):f.register(row('same','train','xyz'))

    def test_head_control_selects_training_only_and_unique_responses(self):
        prepared=[]
        for split in ('train','validation','test'):
            for i in range(8):
                r={**row(str(i),split,'long target '*5),'source':'fixture','id':str(i)}
                prepared.append((r,{'ids':[1]*10,'response_id':str(i)}))
        selected=select_examples(prepared,8,16,3)
        self.assertEqual({r['split'] for r,c in selected},{'train'})
        self.assertEqual(len({c['response_id'] for r,c in selected}),8)
        with self.assertRaises(ValueError):select_examples(prepared,9,16,3)


if __name__=='__main__':unittest.main()
