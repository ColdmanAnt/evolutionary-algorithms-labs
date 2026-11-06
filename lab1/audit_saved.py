"""Независимая проверка сохранённых результатов ЛР1; --replay повторяет все 60 запусков.

Успех: только JSON в stdout. Нарушение инварианта: исключение и ненулевой код.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
from datetime import datetime, timezone
import numpy as np
from main import run_ga, run_random

HERE = Path(__file__).resolve().parent
ALGORITHMS = {'GA_sigma_002': .02, 'GA_sigma_010': .10, 'Random': None}
ABSOLUTE_TOLERANCE = 2e-12
RELATIVE_TOLERANCE = 1e-12


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def scalar_objective(vector):
    """Скалярный oracle на math; не вызывает рабочую функцию objective."""
    return -math.fsum(math.sin(float(x)) * math.sin(i * float(x)**2 / math.pi)**20
                      for i, x in enumerate(vector, 1))


def close(a, b, message, *, tolerance=ABSOLUTE_TOLERANCE):
    require(math.isfinite(float(a)) and math.isfinite(float(b)) and
            math.isclose(float(a), float(b), rel_tol=RELATIVE_TOLERANCE, abs_tol=tolerance), message)



def compare_replay(result, saved_row, saved_history, *, expected_budget, saved_vector=None):
    """Точны ключи/счётчики; float сравниваются с объявленным допуском libm.

    Этот допуск относится только к новому вычислению на другой платформе.
    Согласованность сохранённых CSV/JSON проверяется отдельно без ослабления.
    """
    require(result['seed'] == int(saved_row['seed']), 'Повтор не совпал по seed')
    require(result['evaluations'] == expected_budget == int(saved_row['evaluations']) and
            result['generations'] == int(saved_row['generations']) and result['feasible'] is True,
            'Повтор не совпал по бюджету/поколениям/допустимости')
    close(result['best'], saved_row['best'], 'Повтор не совпал по fitness в пределах допуска')
    history = result['history']
    require(len(history) == len(saved_history), 'Повтор не совпал по длине истории')
    exact_checkpoints = 0; max_history_error = 0.
    for actual, saved in zip(history, saved_history):
        require(len(actual) == len(saved) == 3 and actual[:2] == saved[:2],
                'Повтор не совпал по номеру поколения/счётчику checkpoint')
        close(actual[2], saved[2], 'Повтор не совпал по checkpoint в пределах допуска')
        exact_checkpoints += actual[2] == saved[2]
        max_history_error = max(max_history_error, abs(actual[2]-saved[2]))
    vector_exact = 0; vector_within_tolerance = 0; max_vector_error = 0.
    if saved_vector is not None:
        require(len(result['x']) == len(saved_vector), 'Повтор не совпал по размерности сохранённого вектора')
        for actual, saved in zip(result['x'], saved_vector):
            close(actual, saved, 'Повтор не совпал с сохранённым вектором в пределах допуска')
            max_vector_error = max(max_vector_error, abs(actual-saved))
        vector_exact = int(result['x'] == saved_vector)
        vector_within_tolerance = 1
    return dict(exact_final_fitness_matches=int(result['best'] == float(saved_row['best'])),
                final_fitness_matches_within_tolerance=1,
                exact_history_checkpoint_matches=exact_checkpoints,
                history_checkpoints_within_tolerance=len(history),
                exact_history_run_matches=int(exact_checkpoints == len(history)),
                history_runs_within_tolerance=1,
                exact_saved_global_best_vector_matches=vector_exact,
                saved_global_best_vector_matches_within_tolerance=vector_within_tolerance,
                max_absolute_final_fitness_error=abs(result['best']-float(saved_row['best'])),
                max_absolute_history_error=max_history_error,
                max_absolute_saved_vector_coordinate_error=max_vector_error)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def verify_saved(replay=False, *, folder=HERE):
    folder = Path(folder)
    results = folder/'results'
    config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    require((config['N'], config['Q'], config['Y'], config['variant']) == (28, 1, 26, 10), 'Неверные метаданные варианта')
    require(config['variant'] == 1 + ((17*config['N'] + 7*config['Q'] + config['Y']) % 20), 'Неверная формула варианта')
    dimension = config['dimension']; population = config['population']; budget = config['budget']
    require((dimension, population, budget) == (5+(10 % 6), 80, 24000), 'Неверный архивный план')
    require(config['seeds'] == list(range(100, 120)), 'Неверный диапазон seeds')
    require(config['sigma_fractions'] == [.02, .10], 'Неверные масштабы мутации')
    require((config['elite'], config['crossover_probability'], config['sbx_eta'], config['tournament_k']) == (2, .9, 10, 3), 'Изменены архивные параметры операторов')
    require(config['mutation_probability'] == '1/dimension' and config['bounds'] == [0, 'pi'] and
            config['boundary_handling'] == 'reflection' and config['initialization'] == 'uniform', 'Изменена модель поиска')
    rows = read_csv(results/'runs.csv')
    require(len(rows) == 60, 'Ожидаются 60 строк запусков')
    by_key = {}
    for row in rows:
        key = row['algorithm'], int(row['seed'])
        require(key[0] in ALGORITHMS and key[1] in config['seeds'] and key not in by_key, 'Неверный или повторный ключ запуска')
        require(int(row['evaluations']) == budget and int(row['generations']) == 299, f'Неверный бюджет/поколения: {key}')
        require(int(row['feasible']) == 1, f'Недопустимый сохранённый результат: {key}')
        require(math.isfinite(float(row['best'])) and -dimension <= float(row['best']) <= 0, f'Неконечный/невозможный fitness: {key}')
        require(math.isfinite(float(row['seconds'])) and float(row['seconds']) >= 0, 'Неверное сохранённое время')
        by_key[key] = row
    expected_keys = {(algorithm, seed) for algorithm in ALGORITHMS for seed in config['seeds']}
    require(set(by_key) == expected_keys, 'Не все комбинации алгоритм/seed представлены')
    histories = collections.defaultdict(list)
    for row in read_csv(results/'history.csv'):
        key = row['algorithm'], int(row['seed'])
        require(key in by_key, 'Журнал содержит неизвестный запуск')
        histories[key].append((int(row['generation']), int(row['evaluations']), float(row['best'])))
    require(set(histories) == expected_keys, 'Журналы некоторых запусков отсутствуют')
    for key, history in histories.items():
        require(len(history) == 300, f'Неполная история: {key}')
        previous = float('inf')
        for generation, (saved_generation, evaluations, best) in enumerate(history):
            require(saved_generation == generation and evaluations == (generation+1)*population, f'Неверный checkpoint: {key}')
            require(math.isfinite(best) and -dimension <= best <= 0 and best <= previous, f'Нарушена монотонность/границы: {key}')
            previous = best
        require(history[-1][2] == float(by_key[key]['best']), f'Итог не совпадает с журналом: {key}')
    # Общая инициализация проверяется скалярным oracle независимо от ГА.
    for seed in config['seeds']:
        initial = np.random.default_rng(seed).uniform(0, math.pi, (population, dimension))
        initial_best = min(scalar_objective(x) for x in initial)
        for algorithm in ALGORITHMS:
            close(histories[(algorithm, seed)][0][2], initial_best, f'Неверная инициализация: {(algorithm, seed)}')
    summaries_json = json.loads((results/'summary.json').read_text(encoding='utf-8'))
    summaries_csv = read_csv(results/'summary.csv')
    require(len(summaries_json) == len(summaries_csv) == 3, 'Неверное число агрегатов')
    for container in [summaries_json, summaries_csv]:
        require({row['algorithm'] for row in container} == set(ALGORITHMS), 'Неверные методы в агрегатах')
        for row in container:
            group = [item for item in rows if item['algorithm'] == row['algorithm']]
            values = [float(item['best']) for item in group]
            expected = dict(best=min(values), mean=statistics.mean(values), median=statistics.median(values),
                            std_sample=statistics.stdev(values), worst=max(values), feasible_share=1.,
                            mean_seconds=statistics.mean(float(item['seconds']) for item in group))
            for key, value in expected.items():
                close(row[key], value, f'Неверный агрегат {row["algorithm"]}/{key}')
            require(int(row['runs']) == 20 and int(row['evaluations']) == budget, 'Неверный размер/бюджет серии в агрегатах')
    winner = json.loads((results/'best_solution.json').read_text(encoding='utf-8'))
    key = winner['algorithm'], winner['seed']
    require(key in by_key and winner['best'] == min(float(row['best']) for row in rows), 'Неверный глобальный лучший результат')
    require(winner['best'] == float(by_key[key]['best']), 'Лучшее решение не соответствует своему запуску')
    require(winner['evaluations'] == budget and winner['generations'] == 299 and winner['feasible'] is True, 'Неверные метаданные лучшего решения')
    vector = winner['x']
    require(len(vector) == dimension and all(math.isfinite(x) and 0 <= x <= math.pi for x in vector), 'Недопустимый лучший вектор')
    close(scalar_objective(vector), winner['best'], 'Формула не совпадает с лучшим вектором')
    require([list(h) for h in histories[key]] == winner['history'], 'Журнал в best_solution.json не совпадает с CSV')
    reference = json.loads((results/'reference.json').read_text(encoding='utf-8'))
    require(reference['lower_bound'] == -dimension and reference['status'] == 'numerical_reference_not_certified_global_optimum', 'Неверный статус численного ориентира')
    require(len(reference['x']) == dimension and all(math.isfinite(x) and 0 <= x <= math.pi for x in reference['x']), 'Недопустимый вектор ориентира')
    close(scalar_objective(reference['x']), reference['value'], 'Формула не совпадает с ориентиром')
    require(-dimension <= reference['value'] <= winner['best'], 'Неверный порядок нижней границы/ориентира/лучшего')
    # Другая (сдвинутая) сетка: проверка качества ориентира, не сертификат глобальности.
    grid = (np.arange(200003) + .5) * (math.pi/200003)
    grid_sum = 0.
    for i, x in enumerate(reference['x'], 1):
        candidates = -np.sin(grid) * np.sin(i*grid*grid/math.pi)**20
        grid_best = min(0., float(candidates.min()))
        own = -math.sin(x)*math.sin(i*x*x/math.pi)**20
        require(own <= grid_best + 1e-9, f'Ориентир хуже независимой сетки по координате {i}')
        grid_sum += grid_best
    replays = 0; replay_checkpoints = 0; vector_hashes = {}; max_oracle_delta = 0.
    replay_metrics = dict(exact_final_fitness_matches=0, final_fitness_matches_within_tolerance=0,
        exact_history_checkpoint_matches=0, history_checkpoints_within_tolerance=0,
        exact_history_run_matches=0, history_runs_within_tolerance=0,
        exact_saved_global_best_vector_matches=0, saved_global_best_vector_matches_within_tolerance=0,
        max_absolute_final_fitness_error=0., max_absolute_history_error=0.,
        max_absolute_saved_vector_coordinate_error=0.)
    if replay:
        for algorithm, sigma in ALGORITHMS.items():
            for seed in config['seeds']:
                kwargs = dict(dimension=dimension, population=population, budget=budget)
                result = run_random(seed, **kwargs) if sigma is None else run_ga(seed, sigma,
                    elite=config['elite'], crossover=config['crossover_probability'], eta=config['sbx_eta'],
                    tournament_k=config['tournament_k'], **kwargs)
                key = algorithm, seed
                comparison = compare_replay(result, by_key[key], histories[key], expected_budget=budget,
                    saved_vector=winner['x'] if key == (winner['algorithm'], winner['seed']) else None)
                for name, value in comparison.items():
                    replay_metrics[name] = max(replay_metrics[name], value) if name.startswith('max_') else replay_metrics[name] + value
                require(len(result['x']) == dimension and all(0 <= x <= math.pi for x in result['x']), f'Неверный повторный вектор: {key}')
                delta = abs(scalar_objective(result['x']) - result['best'])
                max_oracle_delta = max(max_oracle_delta, delta)
                close(scalar_objective(result['x']), result['best'], f'Scalar oracle не совпал с повторным вектором: {key}')
                vector_hashes[f'{algorithm}:{seed}'] = hashlib.sha256(json.dumps(result['x']).encode()).hexdigest()
                replays += 1; replay_checkpoints += len(result['history'])
    return dict(status='passed', executed_at_utc=datetime.now(timezone.utc).isoformat(),
        environment=dict(python=platform.python_version(), numpy=np.__version__, system=platform.system(), machine=platform.machine()),
        variant=10, dimension=dimension, saved_runs_checked=len(rows), history_checkpoints_checked=sum(map(len,histories.values())),
        saved_summary_records_checked=6, initial_populations_checked=20, initial_candidates_scalar_oracle_checked=1600,
        saved_best_vectors_checked=1, numerical_reference=dict(scalar_formula_matches=True, shifted_grid_points_per_coordinate=200003,
            grid_best_sum=grid_sum, saved_value=reference['value'], certified_global_minimum=False),
        replay=dict(requested=bool(replay), completed_runs=replays, history_checkpoints_compared=replay_checkpoints,
            comparison_tolerance=dict(absolute=ABSOLUTE_TOLERANCE, relative=RELATIVE_TOLERANCE),
            **replay_metrics,
            replayed_vectors_scalar_oracle_checked=replays, max_scalar_oracle_absolute_difference=max_oracle_delta,
            final_vector_sha256=vector_hashes),
        numeric_artifact_sha256={str(path.relative_to(folder)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [folder/'config.json', *[results/name for name in ['runs.csv','history.csv','summary.csv','summary.json','best_solution.json','reference.json']]]},
        limitations=['Only the overall winning vector was archived; the other 59 final vectors are recomputed and checked against their fitness, not compared with unavailable original vectors.',
            'Replay floating values use abs_tol=2e-12 and rel_tol=1e-12 for cross-platform libm rounding; discrete counters and saved-artifact consistency remain strict. Exact and within-tolerance match counts are reported separately.',
            'The shifted-grid reference check is numerical evidence, not a rigorous global-optimality certificate.',
            'Runtime is excluded from replay comparisons; saved mean runtimes are recomputed only from the archived rows.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify_saved(replay=args.replay), ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
