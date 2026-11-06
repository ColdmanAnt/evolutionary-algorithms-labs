import unittest
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import main as lab1_main
import numpy as np
from main import objective, reflect, sbx, run_ga, run_random

class Lab1Tests(unittest.TestCase):
    def test_function_values_and_bounds(self):
        self.assertAlmostEqual(float(objective(np.zeros(9))[0]),0.)
        x=np.random.default_rng(98).uniform(0,np.pi,(10000,9))
        values=objective(x)
        self.assertTrue(np.all((-9<=values)&(values<=0)))
        y=np.pi/2
        self.assertAlmostEqual(float(objective(np.array([y,y]))[0]),-1.0009765625,places=11)

    def test_reflection_multiple_periods(self):
        x=np.array([-100*np.pi,-3*np.pi,-.4,0,.4,np.pi,7*np.pi,1234.5])
        y=reflect(x)
        self.assertTrue(np.all((y>=0)&(y<=np.pi)))
        np.testing.assert_allclose(y[3:6],x[3:6],atol=1e-14)
        self.assertAlmostEqual(y[2],.4)

    def test_sbx_identity_at_no_crossing(self):
        a=np.ones((20,9));b=np.zeros((20,9))
        np.testing.assert_array_equal(sbx(a,b,np.random.default_rng(3),probability=0),a)

    def test_reproducibility_budget_elitism(self):
        a=run_ga(27,.1,population=40,budget=413)
        b=run_ga(27,.1,population=40,budget=413)
        self.assertEqual(a['x'],b['x']);self.assertEqual(a['history'],b['history'])
        self.assertEqual(a['evaluations'],413)
        self.assertTrue(a['feasible'])
        self.assertTrue(all(x[2]>=y[2] for x,y in zip(a['history'],a['history'][1:])))
        self.assertAlmostEqual(a['best'],float(objective(np.array(a['x']))[0]))

    def test_random_budget_and_reproduction(self):
        a=run_random(21,population=30,budget=413)
        b=run_random(21,population=30,budget=413)
        self.assertEqual(a['x'],b['x']);self.assertEqual(a['evaluations'],413)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):run_ga(1,.1,population=20)
        with self.assertRaises(ValueError):run_ga(1,-.1)
        with self.assertRaises(ValueError):run_ga(1,.1,budget=20)

    def test_report_output_isolation_and_default_paths(self):
        # Пробный каталог не должен изменять отчёт сохранённого эксперимента.
        summaries = [dict(algorithm=name,best=-1.,mean=-1.,median=-1.,
                          std_sample=0.,worst=-1.,mean_seconds=.01)
                     for name in ['GA_sigma_002','GA_sigma_010','Random']]
        winner = dict(best=-1.,seed=100,x=[1.]*9)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)/'lab1'
            root.mkdir()
            archive_report = root/'report_content.json'
            archive_report.write_text('archive sentinel',encoding='utf-8')
            scratch = root/'scratch'
            scratch.mkdir()
            with patch.object(lab1_main,'ROOT',root):
                lab1_main.create_report({},summaries,{'value':-2.},'GA_sigma_002',winner,scratch)
                self.assertEqual(archive_report.read_text(encoding='utf-8'),'archive sentinel')
                trial = json.loads((scratch/'report_content.json').read_text(encoding='utf-8'))
                trial_images = [section['image'] for section in trial['sections'] if 'image' in section]
                self.assertTrue(trial_images)
                self.assertTrue(all(Path(image).parent == Path('.') for image in trial_images))
                default = root/'results'
                default.mkdir()
                lab1_main.create_report({},summaries,{'value':-2.},'GA_sigma_002',winner,default)
                archive = json.loads(archive_report.read_text(encoding='utf-8'))
                archive_images = [section['image'] for section in archive['sections'] if 'image' in section]
                self.assertEqual(archive_images,['results/'+image for image in trial_images])

    def test_objective_matches_independent_scalar_oracle(self):
        from audit_saved import scalar_objective
        points = np.random.default_rng(190).uniform(0, np.pi, (25, 9))
        for point, value in zip(points, objective(points)):
            self.assertAlmostEqual(float(value), scalar_objective(point), places=12)
        for invalid in [np.zeros((2,3,4)), [], [float('nan')], [float('inf')]]:
            with self.subTest(invalid=str(invalid)), self.assertRaises(ValueError):
                objective(invalid)

    def test_tournament_selects_minimum_and_copies_winners(self):
        class Candidates:
            def integers(self, *args):
                return np.array([[0,1,2], [3,2,1], [3,0,1]])
        population = np.arange(8).reshape(4,2).astype(float)
        chosen = lab1_main.tournament(population, np.array([4.,3.,1.,2.]), Candidates(), 3, 3)
        np.testing.assert_array_equal(chosen, population[[2,2,3]])
        chosen[0,0] = -999
        self.assertEqual(population[2,0], 4.)

    def test_sbx_known_quantiles_symmetry_and_parent_integrity(self):
        class Draws:
            def __init__(self): self.call = 0
            def random(self, shape):
                self.call += 1
                if self.call == 1:
                    return np.array([[0., .5, .75, .96875]]*2)
                if self.call == 2:
                    return np.array([[.75]*4, [.25]*4])
                return np.zeros((2,1))
        a=np.full((2,4),2.); b=np.zeros((2,4))
        child=sbx(a,b,Draws(),probability=1.,eta=1.)
        np.testing.assert_allclose(child, [[1.,2.,1+math.sqrt(2),5.], [1.,0.,1-math.sqrt(2),-3.]])
        np.testing.assert_allclose(child[0]+child[1],a[0]+b[0])
        np.testing.assert_array_equal(a,np.full((2,4),2.))
        np.testing.assert_array_equal(b,np.zeros((2,4)))

    def test_gaussian_mutation_only_changes_masked_coordinates(self):
        class Draws:
            def uniform(self, low, high, shape): return np.tile([.5,1.,1.5],(shape[0],1))
            def random(self, shape): return np.tile([.2,.7,.7],(shape[0],1))
            def normal(self, mean, scale, shape): return np.full(shape,.25)
        observed=[]
        def observe(points):
            observed.append(points.copy())
            return objective(points)
        def parents(pop, fit, rng, count, k): return pop[:count].copy()
        with patch.object(lab1_main.np.random,'default_rng',return_value=Draws()), \
             patch.object(lab1_main,'tournament',side_effect=parents), \
             patch.object(lab1_main,'sbx',side_effect=lambda a,b,*args:a.copy()), \
             patch.object(lab1_main,'objective',side_effect=observe):
            result=run_ga(5,.1,dimension=3,population=30,budget=31)
        self.assertEqual([len(batch) for batch in observed],[30,1])
        np.testing.assert_allclose(observed[1][0],[.75,1.,1.5])
        self.assertEqual(result['evaluations'],31)

    def test_counted_objective_rows_equal_budget_at_boundaries(self):
        for runner in [run_ga,run_random]:
            for budget in [30,31,59,60,61]:
                with self.subTest(runner=runner.__name__,budget=budget):
                    rows=[]
                    def observe(points):
                        rows.append(len(points))
                        self.assertTrue(np.all((points>=0)&(points<=np.pi)))
                        return objective(points)
                    with patch.object(lab1_main,'objective',side_effect=observe):
                        result=runner(99,.1,population=30,budget=budget) if runner is run_ga else runner(99,population=30,budget=budget)
                    self.assertEqual(sum(rows),budget)
                    self.assertEqual(result['evaluations'],budget)
                    self.assertEqual(result['history'][-1][1],budget)
                    self.assertTrue(all(a[2]>=b[2] for a,b in zip(result['history'],result['history'][1:])))

    def test_nonfinite_fractional_and_boolean_parameters_rejected(self):
        cases=[dict(dimension=2.5),dict(population=30.5),dict(budget=80.5),dict(elite=1.5),
               dict(tournament_k=1.5),dict(eta=float('nan')),dict(eta=float('inf')),
               dict(population=True),dict(budget=True),dict(elite=True),dict(tournament_k=True)]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):run_ga(1,.1,**kwargs)
        for scale in [float('nan'),float('inf'),float('-inf'),1e308,True]:
            with self.subTest(scale=scale),self.assertRaises(ValueError):run_ga(1,scale)
        for seed in [-1,1.5,True]:
            with self.subTest(seed=seed),self.assertRaises(ValueError):run_ga(seed,.1)
            with self.assertRaises(ValueError):run_random(seed)
        for kwargs in [dict(dimension=2.5),dict(population=1.5),dict(budget=80.5)]:
            with self.assertRaises(ValueError):run_random(1,**kwargs)
        for value in [float('nan'),float('inf')]:
            with self.assertRaises(ValueError):reflect(np.array([value]))
        for dimension in [0,1.5,True]:
            with self.assertRaises(ValueError):lab1_main.numerical_reference(dimension)

    def test_archive_config_mismatches_fail_before_writing(self):
        config=json.loads((lab1_main.ROOT/'config.json').read_text(encoding='utf-8'))
        for key,value in [('N',26),('variant',16),('mutation_probability',.5),('bounds',[-1,1]),
                          ('boundary_handling','clip'),('sigma_fractions',[.02,.2]),('seeds',[100]*20),
                          ('population',80.),('dimension',9.),('Q',True)]:
            with self.subTest(key=key),TemporaryDirectory() as temporary:
                invalid=dict(config);invalid[key]=value;output=Path(temporary)/'not_created'
                with self.assertRaises(ValueError):lab1_main.experiment(invalid,output)
                self.assertFalse(output.exists())
        with self.assertRaises(ValueError):lab1_main.validate_config([],archived=True)
        with self.assertRaises(ValueError):lab1_main.validate_config({},archived=True)

    def test_cli_rejects_ignored_and_nonfinite_options(self):
        for options in [['--sigma','.1'],['--seed','-1'],['--seed','1','--sigma','nan'],['--replot','--seed','1'],['--config','missing-config-audit.json']]:
            with self.subTest(options=options):
                result=subprocess.run([sys.executable,str(lab1_main.ROOT/'main.py'),*options],capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('error:',result.stderr)
                self.assertNotIn('Traceback',result.stderr)

    def test_saved_audit_detects_corrupted_numeric_artifacts(self):
        from audit_saved import verify_saved
        with TemporaryDirectory() as temporary:
            root=Path(temporary)/'lab1';(root/'results').mkdir(parents=True)
            shutil.copy(lab1_main.ROOT/'config.json',root/'config.json')
            names=['runs.csv','history.csv','summary.csv','summary.json','best_solution.json','reference.json']
            for name in names:shutil.copy(lab1_main.ROOT/'results'/name,root/'results'/name)
            # Проверяем именно подмену сохранённого результата, не реализацию ГА.
            target=root/'results'/'runs.csv'
            original=target.read_text(encoding='utf-8')
            target.write_text(original.replace(',24000,',',23999,',1),encoding='utf-8')
            with self.assertRaises(AssertionError):verify_saved(folder=root)
            target.write_text(original,encoding='utf-8')
            target=root/'results'/'summary.json'
            data=json.loads(target.read_text(encoding='utf-8'));data[0]['mean']+=.1
            target.write_text(json.dumps(data),encoding='utf-8')
            with self.assertRaises(AssertionError):verify_saved(folder=root)

if __name__=='__main__':unittest.main()
