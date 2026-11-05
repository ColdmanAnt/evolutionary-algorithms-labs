import unittest
import json
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

if __name__=='__main__':unittest.main()
