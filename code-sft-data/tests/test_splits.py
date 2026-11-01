import unittest
from build_splits import canonical_body,unresolved_names,choose

class SplitTests(unittest.TestCase):
    def row(self,i,seed=None,body=None,task=None):
        return dict(source_id=i,source_row=i,source_seed_sha1=seed or str(i),canonical_body_sha256=body or str(i),instruction_sha256=task or str(i))
    def test_deterministic_nested_and_disjoint_families(self):
        rows=[self.row(i,seed=str(i//3)) for i in range(1600)]
        train,test,_=choose(rows,1000,100,42)
        again,again_test,_=choose(rows,1000,100,42)
        self.assertEqual(train,again); self.assertEqual(test,again_test)
        self.assertEqual(len(train),1000);self.assertEqual(len(test),100)
        self.assertEqual(train[:100],again[:1000][:100])
        self.assertFalse({r['family_id'] for r in train}&{r['family_id'] for r in test})
        self.assertFalse({r['source_seed_sha1'] for r in train}&{r['source_seed_sha1'] for r in test})
    def test_deduplication_and_insufficient_pool(self):
        rows=[self.row(0,body='same'),self.row(1,body='same'),self.row(2)]
        with self.assertRaises(ValueError):choose(rows,2,1,42)
        a,b,stats=choose(rows,1,1,42)
        self.assertEqual(stats['duplicate_bodies_removed'],1)
    def test_renamed_solutions_match_but_operations_differ(self):
        a='def f(x):\n    y=x+1\n    return y\n'
        b='def g(a):\n    z=a+1\n    return z\n'
        self.assertEqual(canonical_body(a),canonical_body(b))
        self.assertNotEqual(canonical_body(a),canonical_body(b.replace('+1','-1')))
    def test_unknown_names_caught_without_running_code(self):
        self.assertEqual(unresolved_names('def f(x):\n return missing_library.solve(x)\n'),['missing_library'])
        self.assertEqual(unresolved_names('def f(x):\n return sorted(x)\n'),[])
        self.assertEqual(unresolved_names('def f(x):\n return f(x-1) if x else 0\n'),[])
if __name__=='__main__':unittest.main()
