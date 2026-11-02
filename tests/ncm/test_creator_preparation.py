import unittest
from scripts.ncm.prepare_creator_expansion import add_response,frozen_evaluation_hash


class Tokenizer:
    def apply_chat_template(self,messages,**kw):return 'Q:'+messages[0]['content']+'A:'
    def encode(self,text,**kw):return list(text.encode())


class PreparationTests(unittest.TestCase):
    def test_expansion_preserves_complete_prefix_and_utf8_chunks(self):
        original={'id':'answer','group':'g','split':'train','source':'fixture',
            'target':'éabcdefghi','messages':[{'role':'user','content':'q'}]}
        rows,stats=add_response(original,{},Tokenizer(),4,100,2)
        self.assertEqual(len(rows),2)
        last,c=rows[-1]
        self.assertEqual(bytes(c['ids']).decode(),'Q:qA:éabcdefg')
        self.assertEqual(last['target'],'hi');self.assertTrue(c['is_final_chunk'])
        self.assertEqual(stats['total_chunks'],4)
        self.assertEqual(add_response(original,{},Tokenizer(),4,1,8)[0],[])
        with self.assertRaises(ValueError):add_response(original,{'g':'test'},Tokenizer(),4,100,8)
        with self.assertRaises(ValueError):add_response({**original,'split':'test'},{},Tokenizer(),4,100,8)

    def test_evaluation_hash_ignores_train_append_but_detects_holdout_mutation(self):
        original=[({'split':'validation','target':'original'},{'ids':[1,2]})]
        key=frozen_evaluation_hash(original)
        original.append(({'split':'train','target':'new'},{'ids':[3,4]}))
        self.assertEqual(key,frozen_evaluation_hash(original))
        original[0][1]['ids'][0]=9
        self.assertNotEqual(key,frozen_evaluation_hash(original))


if __name__=='__main__':unittest.main()
