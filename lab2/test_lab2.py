"""Содержательные проверки ограничений, операторов, бюджета и воспроизводимости."""
import unittest
import numpy as np
from lab2 import Problem, generate_data, variation, run

class ScheduleTests(unittest.TestCase):
    def setUp(self):self.p=Problem(generate_data())
    def test_size_and_conflicts(self):
        self.assertEqual(len(self.p.data['events']),21)
        self.assertEqual(len(self.p.data['conflicts']),27)
        self.assertEqual(self.p.evaluate(self.p.witness)['violations'],0)
    def test_bad_plan_detected(self):
        v=self.p.evaluate(np.zeros(21,dtype=int))
        self.assertEqual(v['overcap'],17)
        self.assertEqual(v['night_overcap'],5)
        self.assertEqual(v['conflicts'],27)
    def test_repair_100_random_inputs(self):
        rng=np.random.default_rng(321)
        for _ in range(100):
            x,_=self.p.repair(rng.integers(0,7,21),rng)
            self.assertEqual(self.p.evaluate(x)['violations'],0)
    def test_swap_preserves_counts_without_crossover(self):
        # Если родители одинаковы, кроссовер не меняет генотип.
        rng=np.random.default_rng(87)
        for _ in range(50):
            child=variation(self.p.witness,self.p.witness,rng,'swap')
            self.assertTrue(np.array_equal(np.bincount(child,minlength=7),np.bincount(self.p.witness,minlength=7)))
    def test_reproducibility_and_exact_partial_budget(self):
        for method in ['penalty','repair','random']:
            a=run(self.p,method,'replace',42,187)
            b=run(self.p,method,'replace',42,187)
            self.assertEqual(a['evaluations'],187)
            self.assertEqual(a['solution'],b['solution'])
            self.assertEqual(a['best'],b['best'])
            self.assertEqual(a['history'],b['history'])
            self.assertEqual(self.p.evaluate(a['solution'])['violations'],0)
            values=[h['best'] for h in a['history']]
            self.assertTrue(all(v>=w for v,w in zip(values,values[1:])))
    def test_validation(self):
        with self.assertRaises(ValueError):self.p.evaluate([0]*20)
        with self.assertRaises(ValueError):run(self.p,'repair','swap',0,20)

if __name__=='__main__':unittest.main()
