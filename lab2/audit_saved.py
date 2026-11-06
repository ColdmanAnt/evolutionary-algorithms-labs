"""Независимая проверка сохранённой ЛР2; --replay воспроизводит все 100 запусков.

python audit_saved.py [--replay]
JSON выводится только после успешного завершения всех проверок. При нарушении
выдаётся описательное исключение и ненулевой код завершения. Данные не изменяются.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from oracle import oracle_schedule

HERE = Path(__file__).resolve().parent
SAVED_FILES = ('data.json', 'config.json', 'solutions.json', 'perrun.csv',
               'histories.csv', 'summary.json', 'summary.csv', 'examples.json',
               'best_solution.json', 'best_schedule.csv', 'report_content.json')


def require(condition, context):
    if not condition:
        raise ValueError(f'Lab2 audit: {context}')


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def close(first, second):
    return math.isclose(float(first), float(second), rel_tol=1e-12, abs_tol=1e-12)


def summary_values(values):
    ordered = sorted(values)
    count = len(values)
    mean = sum(values) / count
    median = ordered[count // 2] if count % 2 else sum(ordered[count//2-1:count//2+1]) / 2
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / (count - 1)) if count > 1 else 0.
    return dict(best=ordered[0], mean=mean, median=median, std=std, worst=ordered[-1])


def same_csv_value(serialized, value):
    if isinstance(value, bool):
        return serialized == str(value)
    if isinstance(value, (float, int)):
        return close(serialized, value)
    return serialized == value


def verify_saved(replay=False, root=None) -> dict:
    started = time.perf_counter()
    root = HERE if root is None else Path(root)
    objects = {name: json.loads((root/name).read_text(encoding='utf-8'))
               for name in SAVED_FILES if name.endswith('.json')}
    data, config, solutions = (objects[name] for name in ('data.json', 'config.json', 'solutions.json'))
    configurations = [('penalty', 'replace'), ('penalty', 'swap'),
                      ('repair', 'replace'), ('repair', 'swap'), ('random', 'none')]
    require(config['configs'] == [list(pair) for pair in configurations], 'factorial configuration plan mismatch')
    require(config['variant'] == 10 and config['N'] == 28 and config['Q'] == 1 and config['Y'] == 26,
            'individual variant mismatch')
    require(config['runs'] == 20 and config['seeds'] == list(range(20)), 'saved seed plan mismatch')
    require(config['budget'] == 6000 and config['population'] == 60, 'saved budget/population mismatch')
    require(len(data['events']) == 21 and len(data['staff']) == 7, 'minimum task size or weekly model mismatch')
    expected_conflicts = {(i, j) for i in range(21) for j in range(i+1, 21)
                          if data['events'][i]['day'] == data['events'][j]['day']
                          or (data['events'][i]['shift'] == 'Ночь'
                              and data['events'][j]['shift'] == 'Утро'
                              and data['events'][j]['day'] == data['events'][i]['day'] + 1)}
    require(len(data['conflicts']) == 27 and set(map(tuple, data['conflicts'])) == expected_conflicts,
            'conflict graph does not represent daily exclusion and night-next-morning rest')
    require(all(0 <= value <= 9 for row in data['preference_costs'] for value in row),
            'preference costs invalidate penalty bound')
    witness_eval = oracle_schedule(data, data['witness'])
    require(witness_eval['violations'] == 0, 'infeasible baseline witness')

    by_key = {}
    perrun = rows(root/'perrun.csv')
    require(len(solutions) == len(perrun) == 100, 'expected 100 saved solutions and run rows')
    for solution, run_row in zip(solutions, perrun):
        key = (solution['config'], solution['seed'])
        require(key not in by_key, f'duplicate run {key}')
        by_key[key] = solution
        require(solution['config'] == f"{solution['method']}_{solution['operator']}", f'run label mismatch {key}')
        evaluated = oracle_schedule(data, solution['solution'])
        require(evaluated['violations'] == 0, f'infeasible saved solution {key}: {evaluated}')
        require(evaluated['soft'] == solution['best'], f'saved objective mismatch {key}')
        require(solution['evaluations'] == config['budget'], f'budget mismatch {key}')
        require(solution['feasible_found'] is True, f'feasible status mismatch {key}')
        for rate in ('feasible_evaluation_fraction', 'final_population_feasible_fraction'):
            require(0 <= solution[rate] <= 1, f'invalid fraction {rate} in {key}')
        if solution['method'] in ('repair', 'random'):
            require(solution['feasible_evaluation_fraction'] == solution['final_population_feasible_fraction'] == 1,
                    f'repaired candidate infeasibility in {key}')
        for field, value in solution.items():
            if field != 'solution':
                require(field in run_row and same_csv_value(run_row[field], value),
                        f'perrun.csv disagreement for {key}, field {field}')
    expected_keys = {(f'{method}_{operator}', seed) for method, operator in configurations for seed in config['seeds']}
    require(set(by_key) == expected_keys, 'configurations or independent seeds missing')

    history_groups = defaultdict(list)
    for row in rows(root/'histories.csv'):
        key = (row['config'], int(row['seed']))
        history_groups[key].append(dict(evaluations=int(row['evaluations']), best=float(row['best']),
                                        feasible_population=float(row['feasible_population'])))
    require(set(history_groups) == expected_keys, 'history configurations or seeds missing')
    checkpoints = [1] + list(range(config['population'], config['budget'] + 1, config['population']))
    for key, history in history_groups.items():
        require([row['evaluations'] for row in history] == checkpoints, f'history budget checkpoints mismatch {key}')
        require(history[0]['best'] == witness_eval['soft'], f'different initial witness in {key}')
        require(history[-1]['best'] == by_key[key]['best'], f'history incumbent disagrees with final score {key}')
        require(history[-1]['feasible_population'] == by_key[key]['final_population_feasible_fraction'],
                f'final feasible population mismatch {key}')
        require(all(left['best'] >= right['best'] for left, right in zip(history, history[1:])),
                f'nonmonotone best feasible archive {key}')
        require(all(0 <= row['feasible_population'] <= 1 for row in history), f'population rate out of range {key}')
    summaries = objects['summary.json']
    summary_csv = rows(root/'summary.csv')
    require(len(summaries) == len(summary_csv) == 5, 'summary table length mismatch')
    independent_stats = {}
    for (method, operator), summary, csv_row in zip(configurations, summaries, summary_csv):
        label = f'{method}_{operator}'
        require(summary['config'] == label, f'summary label mismatch {label}')
        runs = [by_key[(label, seed)] for seed in config['seeds']]
        values = summary_values([run['best'] for run in runs])
        independent_stats[label] = values
        values.update(runs=len(runs), feasible_rate=1, budget=config['budget'],
                      mean_feasible_evaluations=sum(r['feasible_evaluation_fraction'] for r in runs)/len(runs),
                      mean_final_feasible_population=sum(r['final_population_feasible_fraction'] for r in runs)/len(runs),
                      mean_seconds=sum(r['seconds'] for r in runs)/len(runs),
                      fallback_total=sum(r['repair_fallbacks'] for r in runs))
        for field, value in values.items():
            require(close(summary[field], value), f'independent summary mismatch: {label}, {field}')
        for field, value in summary.items():
            require(field in csv_row and same_csv_value(csv_row[field], value), f'summary.csv mismatch: {label}, {field}')

    examples = objects['examples.json']
    feasible_examples, infeasible_examples = [], []
    for example in examples:
        evaluation = oracle_schedule(data, example['solution'])
        require(evaluation == example['evaluation'], f'example evaluation mismatch: {example["name"]}')
        (feasible_examples if evaluation['violations'] == 0 else infeasible_examples).append(example['solution'])
    require(len(feasible_examples) >= 2 and len(infeasible_examples) >= 2
            and feasible_examples[0] != feasible_examples[1], 'two distinct feasible and two infeasible examples required')
    best = objects['best_solution.json']
    best_key = (best['config'], best['seed'])
    require(best_key in by_key and best['solution'] == by_key[best_key]['solution'], 'best solution identity mismatch')
    require(best['evaluation'] == oracle_schedule(data, best['solution']), 'best solution evaluation mismatch')
    require(best['best'] == min(run['best'] for run in solutions), 'best solution is not minimum saved result')
    if 'history' in best:
        require(best['history'] == history_groups[best_key], 'best solution stored history mismatch')
    schedule = rows(root/'best_schedule.csv')
    require(len(schedule) == 21, 'domain schedule has missing shifts')
    for i, row in enumerate(schedule):
        worker = best['solution'][i]
        require(row['day'] == str(data['events'][i]['day'])
                and row['shift'] == data['events'][i]['shift'] and row['employee'] == data['staff'][worker]
                and int(row['preference_cost']) == data['preference_costs'][i][worker], f'domain schedule row mismatch {i}')

    sections = objects['report_content.json']['sections']
    report_stats = next(section['table'] for section in sections if section['heading'] == 'Итоговое качество по серии запусков')
    expected_rows = [[row['config'], row['best'], f"{row['mean']:.2f}", row['median'], f"{row['std']:.2f}", row['worst']]
                     for row in summaries]
    require(report_stats['rows'] == expected_rows, 'report statistics table mismatch')
    diagnostics = next(section['table'] for section in sections if section['heading'] == 'Допустимость, стоимость ограничений и сходимость')
    expected_diagnostics = [[r['config'], f"{100*r['mean_feasible_evaluations']:.1f}",
                             f"{100*r['mean_final_feasible_population']:.1f}", f"{r['mean_seconds']:.3f}", r['fallback_total']]
                            for r in summaries]
    require(diagnostics['rows'] == expected_diagnostics, 'report diagnostics table mismatch')
    text = ' '.join(paragraph for section in sections for paragraph in section.get('paragraphs', []))
    winner = min(summaries, key=lambda item: item['mean'])
    require(f"Наименьшее среднее S={winner['mean']:.2f}" in text
            and f"S={best['evaluation']['soft']}: предпочтения {best['evaluation']['preference']}" in text,
            'report quality conclusions do not match data')

    replayed, objective_calls = 0, 0
    if replay:
        from lab2 import Problem, generate_data, run
        require(generate_data(config['data_seed']) == data, 'data generator does not reproduce saved instance')
        class CountedProblem(Problem):
            def evaluate(self, assignments):
                self.calls = getattr(self, 'calls', 0) + 1
                return super().evaluate(assignments)
        problem = CountedProblem(data)
        for key, expected in by_key.items():
            problem.calls = 0  # Общая проверка witness при загрузке исключена у всех методов.
            actual = run(problem, expected['method'], expected['operator'], expected['seed'], config['budget'], config['population'])
            require(problem.calls == config['budget'], f'instrumented objective count mismatch {key}')
            objective_calls += problem.calls
            for field, value in expected.items():
                if field not in ('seconds', 'config'):
                    require(actual[field] == value, f'full replay mismatch {key}, field {field}')
            require(actual['history'] == history_groups[key], f'full replay history mismatch {key}')
            replayed += 1
    return dict(lab=2, audit_date='2026-11-06', passed=True,
                independent_saved_solution_checks=len(solutions), configurations=5, seeds_per_configuration=20,
                independent_summary_checks=5, history_checkpoints=sum(map(len, history_groups.values())),
                examples_checked=len(examples), model=dict(shifts=21, staff=7, conflict_pairs=27,
                                                         maximum_soft_score=1407, penalty=10000),
                replay_requested=bool(replay), replayed_runs=replayed, instrumented_search_objective_calls=objective_calls,
                objective_budget_per_run=config['budget'], independent_statistics=independent_stats,
                exclusions=['Runtime seconds are measured, not required to replay bitwise.',
                            'One deterministic witness check at Problem construction is outside each search budget.',
                            'Repair checks constraints incrementally; their overhead is included in runtime, not objective calls.'],
                artifact_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in SAVED_FILES},
                elapsed_seconds=time.perf_counter()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', action='store_true')
    arguments = parser.parse_args()
    print(json.dumps(verify_saved(replay=arguments.replay), ensure_ascii=False, indent=2))
