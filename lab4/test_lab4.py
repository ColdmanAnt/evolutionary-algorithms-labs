import json
import copy
import unittest
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import main as lab4
from audit_saved import mean_drift_guard, same_environment_match
from main import HERE, evaluate, reflect, run, migrate, validate_config, offspring, experiment


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

    def test_scalar_oracles_for_random_vectors(self):
        rng = np.random.default_rng(782)
        vectors = rng.uniform(0, math.pi, (40, 9))
        for name in lab4.FUNCTIONS:
            expected = []
            for x in vectors:
                if name == 'sphere': v = sum(float(t)**2 for t in x)
                elif name == 'rastrigin': v = sum(float(t)**2 - 10*math.cos(2*math.pi*float(t)) + 10 for t in x)
                else: v = -sum(math.sin(float(t))*math.sin((i+1)*float(t)**2/math.pi)**20 for i,t in enumerate(x))
                expected.append(v)
            np.testing.assert_allclose(evaluate(vectors,name), expected, rtol=1e-13, atol=1e-12)

    def test_invalid_config_values_fail_before_rng_or_output(self):
        bad = {'dimension':[0,1.5,True], 'total_population':[0,119,30.5], 'islands':[0,1,5],
               'evaluation_budget':[0,119,481,480.0], 'tournament':[0,2.5],
               'migration_interval':[0,-1], 'bootstrap_seed':[-1,True],
               'blx_alpha':[-.1,float('nan'),float('inf'),1j],
               'crossover_probability':[-.1,1.1,float('nan')],
               'mutation_probability':[-.1,1.1,float('nan')],
               'sigma_start':[-.1,float('inf')], 'sigma_end':[-.1,float('nan')],
               'functions':[[],['sphere','sphere'],['unknown']],
               'methods':[[],['ring'],['panmixia','unknown']],
               'seeds':[[],[1,1],[-1],[1.1],[True]]}
        for key, values in bad.items():
            for value in values:
                with self.subTest(key=key,value=value):
                    c=dict(self.c);c[key]=value
                    with patch.object(np.random,'default_rng',side_effect=AssertionError('RNG must not start')):
                        with self.assertRaises(ValueError):run('sphere','ring',1,c)
        with TemporaryDirectory() as tmp:
            output=Path(tmp)/'must_not_be_created'
            c=dict(self.c);c['seeds']=[]
            with self.assertRaises(ValueError):experiment(c,output)
            self.assertFalse(output.exists())

    def test_reflection_symmetry_periodicity_and_invalid_inputs(self):
        x=np.random.default_rng(8).uniform(-100,100,1000)
        lo,hi=-2.,3.
        y=reflect(x,lo,hi)
        np.testing.assert_allclose(reflect(y,lo,hi),y,atol=1e-13)
        np.testing.assert_allclose(reflect(x+100*(hi-lo)*2,lo,hi),y,atol=2e-13)
        np.testing.assert_allclose(reflect(2*lo-x,lo,hi),y,atol=1e-13)
        for a,b,points in [(0,0,[1]),(1,0,[1]),(0,np.inf,[1]),(0,1,[np.nan]),(False,1,[.1])]:
            with self.assertRaises(ValueError):reflect(np.array(points),a,b)
        with self.assertRaises(ValueError):lab4.bounds('unknown')

    def test_actual_evaluation_count_includes_init_not_migrants(self):
        c=dict(self.c);c['evaluation_budget']=1320
        for method in lab4.METHODS:
            batches=[]
            def counted(points,name):
                batches.append(len(points));return evaluate(points,name)
            with patch.object(lab4,'evaluate',side_effect=counted):
                result,history=run('sphere',method,37,c)
            self.assertEqual(sum(batches),1320)
            self.assertEqual(result['evaluations'],1320)
            self.assertEqual(result['migration_slots'],4 if method in ('ring','fully_connected') else 0)
            self.assertEqual([h['evaluations'] for h in history],list(range(120,1321,120)))

    def test_initial_population_is_paired_between_architectures(self):
        c=dict(self.c);c['evaluation_budget']=120
        results=[run('sphere',method,77,c) for method in lab4.METHODS]
        for result,history in results[1:]:
            self.assertEqual(result['best'],results[0][0]['best'])
            self.assertEqual(result['solution'],results[0][0]['solution'])
            self.assertEqual(history[0]['diversity'],results[0][1][0]['diversity'])

    def test_fully_connected_snapshot_and_no_worsening(self):
        pops=[np.array([[i],[10+i]],dtype=float) for i in range(4)]
        fits=[np.array([i,10+i],dtype=float) for i in range(4)]
        migrate(pops,fits,'fully_connected')
        self.assertEqual(pops[0][1,0],1)
        for i in range(1,4):self.assertEqual(pops[i][1,0],0)
        pops=[np.array([[i],[i]],dtype=float) for i in range(4)]
        fits=[np.array([i,i],dtype=float) for i in range(4)]
        before=pops[0].copy()
        migrate(pops,fits,'fully_connected')
        np.testing.assert_array_equal(pops[0],before)
        with self.assertRaises(ValueError):migrate(pops[:1],fits[:1],'ring')
        with self.assertRaises(ValueError):migrate(pops,fits,'typo')

    def test_offspring_without_variation_is_parent_and_does_not_mutate(self):
        rng=np.random.default_rng(41)
        p=rng.uniform(0,1,(30,3));original=p.copy();f=np.arange(30.)
        c=dict(self.c);c.update(crossover_probability=0,mutation_probability=0)
        children=offspring(p,f,rng,0,1,.3,c)
        np.testing.assert_array_equal(p,original)
        for child in children:self.assertTrue(np.any(np.all(p==child,axis=1)))

    def test_migration_rejects_integer_destinations_before_truncation(self):
        pops=[np.array([[2],[3]]),np.array([[.5],[1.]])]
        fits=[np.array([4.,9.]),np.array([.25,1.])]
        before=pops[0].copy()
        with self.assertRaises(ValueError):migrate(pops,fits,'ring')
        np.testing.assert_array_equal(pops[0],before)
        with self.assertRaises(ValueError):migrate([[[.1]],[[.2]]],[np.array([.01]),np.array([.04])],'ring')

    def test_short_custom_series_report_matches_actual_parameters(self):
        c=dict(self.c);c.update(evaluation_budget=120,dimension=2,bootstrap_seed=123,functions=['sphere','michalewicz'],methods=['panmixia','ring'],seeds=[3])
        with TemporaryDirectory() as tmp:
            experiment(c,Path(tmp))
            report=json.loads((Path(tmp)/'report_content.json').read_text())
            text=' '.join(p for section in report['sections'] for p in section.get('paragraphs',[]))
            self.assertIn('120 оценок',text)
            self.assertIn('1;',text)
            self.assertIn('короткий демонстрационный',text)
            self.assertIn('bootstrap_seed=123',text)
            self.assertNotIn('bootstrap_seed=88010',text)
            self.assertIn('размерности 2',text)
            self.assertNotIn('30 запускам',text)
            self.assertEqual([s['image'] for s in report['sections'] if s.get('image','').startswith('convergence_')],['convergence_sphere.png','convergence_michalewicz.png'])
            for section in report['sections']:
                if section.get('image'):self.assertTrue((Path(tmp)/section['image']).exists(),section['image'])

    def test_saved_audit_rejects_nan_and_duplicate_runs(self):
        import csv
        from audit_saved import verify_saved
        with (HERE/'runs.csv').open() as stream:rows=list(csv.DictReader(stream))
        for kind in ('nan','duplicate'):
            with self.subTest(kind=kind), TemporaryDirectory() as tmp:
                root=Path(tmp)
                (root/'config.json').write_text((HERE/'config.json').read_text())
                damaged=[dict(r) for r in rows]
                if kind=='nan':damaged[0]['best']='nan'
                else:damaged[1]=dict(damaged[0])
                with (root/'runs.csv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=damaged[0]);writer.writeheader();writer.writerows(damaged)
                with self.assertRaises(ValueError):verify_saved(root=root)


class AuditHelpers(unittest.TestCase):
    def test_mean_guard_uses_archive_sample_standard_error(self):
        # Для сбалансированной серии +/-1 сумма квадратов равна n:
        # 3 * sample_std / sqrt(n) = 3 / sqrt(n-1).
        for n in [2, 8, 30]:
            reference=np.array([-1.,1.]*(n//2))
            result=mean_drift_guard(reference,np.zeros(n))
            self.assertAlmostEqual(result['threshold'],3/math.sqrt(n-1),places=14)
            self.assertEqual(result['mean_delta'],0.)
            self.assertIn('not proof',result['interpretation'])

    def test_mean_guard_accepts_boundary_and_rejects_next_float_beyond(self):
        reference=np.array([-1.,1.])
        # Здесь фиксированный порог равен ровно 3, без оценки по повторной серии.
        for sign in (-1.,1.):
            with self.subTest(sign=sign):
                accepted=mean_drift_guard(reference,np.full(2,sign*3.))
                self.assertEqual(accepted['threshold'],3.)
                self.assertEqual(accepted['mean_delta'],sign*3.)
                outside=np.nextafter(sign*3.,sign*np.inf)
                with self.assertRaisesRegex(ValueError,'дрейф среднего'):
                    mean_drift_guard(reference,np.full(2,outside))

    def test_mean_guard_cannot_expand_threshold_using_repeat_variance(self):
        reference=np.array([-1.,1.])
        changed_variance=mean_drift_guard(reference,np.array([-1e9,1e9]))
        self.assertEqual(changed_variance['threshold'],3.)
        self.assertEqual(changed_variance['mean_delta'],0.)
        for repeated in [np.full(2,4.),np.array([-1e9+4.,1e9+4.])]:
            with self.assertRaises(ValueError):mean_drift_guard(reference,repeated)

    def test_mean_guard_zero_variance_and_singleton_use_fixed_floor(self):
        for reference in [np.zeros(2),np.zeros(1)]:
            for sign in (-1.,1.):
                accepted=mean_drift_guard(reference,np.full_like(reference,sign*1e-11))
                self.assertEqual(accepted['threshold'],1e-11)
                self.assertEqual(accepted['archive']['std'],0.)
                outside=np.nextafter(sign*1e-11,sign*np.inf)
                with self.assertRaises(ValueError):
                    mean_drift_guard(reference,np.full_like(reference,outside))

    def test_mean_guard_rejects_invalid_shapes_and_nonfinite_series(self):
        for reference,repeated in [([],[]),([1.],[1.,2.]),([[1.,2.]],[[1.,2.]]),
                                   (1.,1.),([1.],[[1.]])]:
            with self.subTest(reference=reference,repeated=repeated):
                with self.assertRaises(ValueError):mean_drift_guard(reference,repeated)
        for value in [np.nan,np.inf,-np.inf]:
            for reference,repeated in [([0.,value],[0.,0.]),([0.,0.],[0.,value])]:
                with self.subTest(value=value,reference=reference,repeated=repeated):
                    with self.assertRaisesRegex(ValueError,'неконечная серия'):
                        mean_drift_guard(reference,repeated)

    @staticmethod
    def sample_repeat():
        result=dict(function='sphere',method='ring',seed=7,solution=[.25,-.5],best=.3125,
                    evaluations=240,generations=1,migration_slots=0,diversity=.125,seconds=.01)
        history=[dict(generation=0,evaluations=120,best=1.,diversity=.25),
                 dict(generation=1,evaluations=240,best=.3125,diversity=.125)]
        return result,history

    def test_same_environment_match_ignores_elapsed_seconds_only(self):
        first,history=self.sample_repeat()
        second,second_history=copy.deepcopy(first),copy.deepcopy(history)
        second['seconds']=1000.
        same_environment_match(first,history,second,second_history)
        # Пропуск seconds также допустим: счётчик времени не часть результата поиска.
        del second['seconds']
        same_environment_match(first,history,second,second_history)

    def test_same_environment_match_requires_exact_metrics_and_discrete_metadata(self):
        first,history=self.sample_repeat()
        changes={'function':'rastrigin','method':'isolated','seed':8,
                 'solution':[np.nextafter(.25,np.inf),-.5],
                 'best':np.nextafter(first['best'],np.inf),'evaluations':241,
                 'generations':2,'migration_slots':1,
                 'diversity':np.nextafter(first['diversity'],np.inf)}
        for field,value in changes.items():
            with self.subTest(field=field):
                second=copy.deepcopy(first);second[field]=value
                with self.assertRaisesRegex(ValueError,field):
                    same_environment_match(first,history,second,copy.deepcopy(history))

    def test_same_environment_match_requires_exact_complete_history(self):
        first,history=self.sample_repeat()
        for field,value in [('generation',2),('evaluations',241),
                            ('best',np.nextafter(.3125,np.inf)),
                            ('diversity',np.nextafter(.125,np.inf))]:
            with self.subTest(field=field):
                changed=copy.deepcopy(history);changed[-1][field]=value
                with self.assertRaisesRegex(ValueError,'история'):
                    same_environment_match(first,history,copy.deepcopy(first),changed)
        for changed in [history[:-1],list(reversed(history))]:
            with self.assertRaisesRegex(ValueError,'история'):
                same_environment_match(first,history,copy.deepcopy(first),changed)


if __name__=='__main__':unittest.main()
