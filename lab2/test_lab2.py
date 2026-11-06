"""Содержательные проверки ограничений, операторов, бюджета и воспроизводимости."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from lab2 import Problem, generate_data, variation, run, experiment
from oracle import oracle_schedule
from audit_saved import verify_saved, SAVED_FILES

HERE = Path(__file__).resolve().parent

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

    def test_independent_oracle_for_1000_random_and_extreme_vectors(self):
        rng = np.random.default_rng(20261106)
        candidates = [self.p.witness] + [np.full(21, worker, dtype=int) for worker in range(7)]
        candidates += list(rng.integers(0, 7, size=(1000, 21)))
        for assignments in candidates:
            self.assertEqual(self.p.evaluate(assignments), oracle_schedule(self.p.data, assignments.tolist()))

    def test_penalty_bound_and_feasible_ordering(self):
        # Верхняя граница достигается в векторе с одним сотрудником и всеми c=9.
        data = generate_data()
        data['preference_costs'] = [[9] * 7 for _ in range(21)]
        p = Problem(data)
        bad = p.evaluate(np.zeros(21, dtype=int))
        feasible = p.evaluate(p.witness)
        self.assertEqual(bad['soft'], 1407)
        self.assertGreater(bad['score'], feasible['score'])
        self.assertLess(1407, 10000)

    def test_repair_properties_across_generated_instances(self):
        for data_seed in [0, 1, 17, 10010, 20261106]:
            p = Problem(generate_data(data_seed))
            rng = np.random.default_rng(data_seed)
            for _ in range(25):
                original = rng.integers(0, 7, 21)
                saved = original.copy()
                repaired, _ = p.repair(original, rng)
                np.testing.assert_array_equal(original, saved)
                self.assertEqual(oracle_schedule(p.data, repaired.tolist())['violations'], 0)
                repeated, fallback = p.repair(repaired, rng)
                np.testing.assert_array_equal(repeated, repaired)
                self.assertFalse(fallback)

    def test_repair_finite_fallback_is_independently_feasible_and_copied(self):
        data = generate_data()
        # В допустимом witness первые две смены получают 1 и 0. Жадный выбор
        # 0 для первой смены блокирует вторую, где только 0 доступен.
        data['witness'][0], data['witness'][1] = 1, 0
        data['availability'] = [[1] * 7 for _ in range(21)]
        data['availability'][0] = [1, 1, 0, 0, 0, 0, 0]
        data['availability'][1] = [1, 0, 0, 0, 0, 0, 0]
        p = Problem(data)
        class OrderedRng:
            calls = 0
            def permutation(self, size):
                self.calls += 1
                return np.arange(size)
        rng = OrderedRng()
        repaired, fallback = p.repair(np.zeros(21, dtype=int), rng)
        self.assertTrue(fallback)
        self.assertEqual(rng.calls, 12)
        self.assertEqual(oracle_schedule(data, repaired.tolist())['violations'], 0)
        repaired[0] = 6
        self.assertEqual(int(p.witness[0]), 1)

    def test_no_silent_fractional_or_string_genotype_coercion(self):
        bad_vectors = [[0.9] * 21, ['1'] * 21, [True] * 21,
                       [float('nan')] * 21, [float('inf')] * 21,
                       [-1] * 21, [7] * 21, [[0] * 21]]
        rng = np.random.default_rng(0)
        for bad in bad_vectors:
            with self.subTest(value=str(bad)[:30]):
                with self.assertRaises(ValueError): self.p.evaluate(bad)
                with self.assertRaises(ValueError): self.p.repair(bad, rng)
                with self.assertRaises(ValueError): variation(bad, self.p.witness, rng, 'swap')

    def test_variation_preserves_alphabet_and_never_mutates_parents(self):
        rng = np.random.default_rng(928)
        for operator in ['replace', 'swap']:
            for _ in range(100):
                a, b = rng.integers(0, 7, (2, 21))
                saved_a, saved_b = a.copy(), b.copy()
                child = variation(a, b, rng, operator)
                self.assertEqual(child.shape, (21,))
                self.assertTrue(np.all((child >= 0) & (child < 7)))
                np.testing.assert_array_equal(a, saved_a)
                np.testing.assert_array_equal(b, saved_b)

    def test_invalid_methods_operators_and_numeric_parameters_fail_early(self):
        for kwargs in [dict(method='typo'), dict(operator='typo'),
                       dict(operator='none'), dict(seed=-1), dict(seed=True),
                       dict(seed=1.5), dict(budget=59), dict(budget=True),
                       dict(budget=100.5), dict(pop_size=1), dict(pop_size=0),
                       dict(pop_size=-1), dict(pop_size=2.5), dict(pop_size=True)]:
            arguments = dict(problem=self.p, method='repair', operator='replace', seed=0, budget=120)
            arguments.update(kwargs)
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError): run(**arguments)
        with self.assertRaises(ValueError):
            variation(self.p.witness, self.p.witness, np.random.default_rng(1), 'unknown')

    def test_infeasible_or_malformed_data_rejected(self):
        invalid = []
        for field, value in [('witness', [0] * 21), ('availability', [[1] * 6] * 21),
                             ('availability', [[2] * 7] * 21), ('preference_costs', [[-1] * 7] * 21),
                             ('preference_costs', [[10] * 7] * 21), ('preference_costs', [[1.5] * 7] * 21),
                             ('max_shifts', 0), ('max_nights', -1), ('max_shifts', True),
                             ('conflicts', [[0, 0]]), ('conflicts', [[0, 21]]),
                             ('conflicts', [[0, 1], [1, 0]]), ('staff', ['same'] * 7),
                             ('events', [])]:
            data = generate_data()
            data[field] = value
            invalid.append(data)
        missing = generate_data(); del missing['witness']; invalid.append(missing)
        for data in invalid:
            with self.subTest(data=str(data)[:40]):
                with self.assertRaises(ValueError): Problem(data)

    def test_caller_data_mutation_cannot_invalidate_loaded_witness(self):
        data = generate_data()
        p = Problem(data)
        expected = oracle_schedule(p.data, p.witness.tolist())
        data['max_shifts'] = 0
        data['witness'][0] = 6
        data['availability'][0][p.witness[0]] = 0
        self.assertEqual(p.evaluate(p.witness), expected)

    def test_exact_instrumented_budget_at_generation_boundaries(self):
        class CountedProblem(Problem):
            def evaluate(self, x):
                self.calls = getattr(self, 'calls', 0) + 1
                return super().evaluate(x)
        p = CountedProblem(generate_data())
        for method, operator in [('penalty', 'replace'), ('penalty', 'swap'),
                                 ('repair', 'replace'), ('repair', 'swap'), ('random', 'none')]:
            for budget in [60, 61, 118, 119, 120, 121, 187]:
                p.calls = 0
                r = run(p, method, operator, 8, budget)
                self.assertEqual(p.calls, budget)
                self.assertEqual(r['history'][0]['best'], oracle_schedule(p.data, p.witness.tolist())['soft'])
                self.assertEqual(r['history'][-1]['evaluations'], budget)
                self.assertEqual(oracle_schedule(p.data, r['solution'])['violations'], 0)

    def test_invalid_experiment_never_writes_partial_output(self):
        with tempfile.TemporaryDirectory(dir=HERE, prefix='.validation-test-') as tmp:
            output = Path(tmp) / 'invalid'
            for runs, budget, seed in [(0, 6000, 0), (20, 59, 0), (20, 6000, -1)]:
                with self.assertRaises(ValueError): experiment(output, runs, budget, seed)
                self.assertFalse(output.exists())

    def test_cli_invalid_input_has_argparse_error_without_traceback(self):
        for arguments in [['single', '--seed', '-1'], ['single', '--budget', '0'],
                          ['experiment', '--runs', '0'], ['experiment', '--budget', '59'],
                          ['experiment', '--seed-start', '-1']]:
            completed = subprocess.run([sys.executable, str(HERE/'lab2.py'), *arguments],
                                       capture_output=True, text=True)
            self.assertEqual(completed.returncode, 2)
            self.assertIn('error:', completed.stderr)
            self.assertNotIn('Traceback', completed.stderr)

    def test_witness_rejected_even_under_python_optimized_mode(self):
        script = "from lab2 import *; d=generate_data(); d['witness']=[0]*21; Problem(d)"
        completed = subprocess.run([sys.executable, '-O', '-c', script], cwd=HERE,
                                   capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('ValueError', completed.stderr)

    def test_saved_independent_audit_and_corruption_detection(self):
        result = verify_saved()
        self.assertEqual(result['independent_saved_solution_checks'], 100)
        self.assertEqual(result['history_checkpoints'], 10100)
        with tempfile.TemporaryDirectory(dir=HERE, prefix='.audit-test-') as temporary:
            root = Path(temporary)
            for filename in SAVED_FILES: shutil.copyfile(HERE/filename, root/filename)
            original = (root/'solutions.json').read_text()
            corrupted = json.loads(original)
            corrupted[0]['solution'] = [0] * 21
            (root/'solutions.json').write_text(json.dumps(corrupted))
            with self.assertRaisesRegex(ValueError, 'infeasible saved solution'):
                verify_saved(root=root)
            (root/'solutions.json').write_text(original)
            summary = json.loads((root/'summary.json').read_text())
            summary[0]['mean'] += 1
            (root/'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'independent summary mismatch'):
                verify_saved(root=root)

if __name__=='__main__':unittest.main()
