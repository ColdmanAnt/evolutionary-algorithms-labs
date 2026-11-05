import json
import unittest
import numpy as np
from main import HERE, evaluate, reflect, run, migrate


class Invariants(unittest.TestCase):
    def setUp(self):
        self.c=json.loads((HERE/'config.json').read_text())
        self.c['evaluation_budget']=480

    def test_functions(self):
        self.assertAlmostEqual(evaluate(np.zeros((1,9)),'sphere')[0],0)
        self.assertAlmostEqual(evaluate(np.zeros((1,9)),'rastrigin')[0],0)
        self.assertAlmostEqual(evaluate(np.zeros((1,9)),'michalewicz')[0],0)
        self.assertAlmostEqual(evaluate(np.full((1,1),np.pi/2),'michalewicz')[0],-(np.sin(np.pi/4)**20))

    def test_reflection(self):
        a=reflect(np.array([-201.,-1.,0.,1.,2.,201.]),0,1)
        self.assertTrue(np.all((a>=0)&(a<=1)))
        np.testing.assert_allclose(a,[1,1,0,1,0,1])

    def test_budget_reproducibility_monotonicity(self):
        for m in ['panmixia','isolated','ring','fully_connected']:
            a,h=run('michalewicz',m,42,self.c)
            b,j=run('michalewicz',m,42,self.c)
            self.assertEqual(a['evaluations'],480)
            self.assertEqual(a['solution'],b['solution'])
            self.assertEqual(h,j)
            self.assertTrue(np.all(np.diff([x['best'] for x in h])<=1e-12))
            self.assertTrue(np.all((np.array(a['solution'])>=0)&(np.array(a['solution'])<=np.pi)))
            self.assertAlmostEqual(evaluate(np.array([a['solution']]),'michalewicz')[0],a['best'])

    def test_ring_uses_snapshot(self):
        pops=[np.array([[i],[100+i]],dtype=float) for i in range(4)]
        fits=[np.array([i,100+i],dtype=float) for i in range(4)]
        migrate(pops,fits,'ring')
        self.assertEqual(pops[1][1,0],0)
        self.assertEqual(pops[2][1,0],1)
        self.assertEqual(pops[3][1,0],2)
        self.assertEqual(pops[0][1,0],3)

    def test_invalid_budget(self):
        self.c['evaluation_budget']=481
        with self.assertRaises(ValueError):run('sphere','ring',1,self.c)


if __name__=='__main__':unittest.main()
