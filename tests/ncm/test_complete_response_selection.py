import unittest
from scripts.ncm.complete_response_selection import select


def rows(response,group,length,split='train',distilled=False,complete=True):
    return [({'id':f'{response}:{i}','group':group,'split':split,'distilled':distilled},
        {'response_id':response,'chunk_index':i,'is_final_chunk':complete and i==length-1}) for i in range(length)]


class CompleteResponseTests(unittest.TestCase):
    def test_keeps_every_chunk_and_prefers_teacher_without_validation(self):
        data=rows('base','a',3)+rows('teacher','a',2,distilled=True)+rows('other','b',4)+rows('validation','c',2,split='validation')
        chosen=select(data,4)
        self.assertEqual([r['id'] for r,_ in chosen],['teacher:0','teacher:1','other:0','other:1','other:2','other:3'])

    def test_incomplete_long_or_gapped_response_is_not_admitted(self):
        data=rows('incomplete','a',2,complete=False)+rows('long','b',9)+rows('gap','c',3)[1:]
        self.assertEqual(select(data,8),[])

    def test_duplicates_or_split_leakage_raise(self):
        x=rows('one','a',2)
        with self.assertRaises(ValueError):select(x+x)
        with self.assertRaises(ValueError):select(x+rows('val','a',1,split='validation'))


if __name__=='__main__':unittest.main()
