"""Независимая проверка сохранённых данных ЛР3; --replay повторяет всю серию.

Запуск из корня: python lab3/audit_saved.py [--replay]
При успехе stdout содержит только JSON. Несовпадение завершает процесс с ошибкой.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np

from lab3 import (ACTIVATIONS, DATA_SEED, EPOCHS, LEARNING_RATE, POPULATION,
                  WIDTHS, Gene, make_data, nondominated, search,
                  select_representatives, train_model)

ROOT = Path(__file__).resolve().parent
ATOL = 2e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message, errors=None):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(actual.shape == expected.shape, message + ': shape mismatch')
    require(np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)), message + ': non-finite value')
    error = float(np.max(np.abs(actual - expected))) if actual.size else 0.
    if errors is not None:
        errors.append(error)
    require(np.allclose(actual, expected, rtol=ATOL, atol=ATOL), message + f': mismatch (max abs error {error:.3g})')


def csv_rows(directory, filename):
    with (directory / filename).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def record(row):
    result = dict(row)
    for key in ['seed', 'parameters', 'evaluations', 'distinct_trainings', 'training_steps']:
        if key in result:
            result[key] = int(result[key])
    for key in ['penalty', 'fitness', 'validation_loss', 'validation_accuracy', 'test_loss', 'test_accuracy', 'seconds']:
        if key in result:
            result[key] = float(result[key])
    if isinstance(result.get('gene'), str):
        result['gene'] = json.loads(result['gene'])
    return result


def identity(item):
    return int(item['seed']), item['key']


def compare_record(actual, expected, context, errors):
    for key in ['seed', 'gene', 'key', 'parameters']:
        require(actual[key] == expected[key], f'{context}: {key} mismatch')
    for key in ['validation_loss', 'validation_accuracy']:
        close(actual[key], expected[key], f'{context}: {key}', errors)


def oracle_front(records):
    """Независимый двумерный oracle: группировка по цене, проход по минимумам.

    Равные точки сохраняются, поскольку строгое доминирование их не исключает.
    """
    grouped = defaultdict(list)
    for item in records:
        grouped[item['parameters']].append(item)
    best_at_smaller_size, result = float('inf'), []
    for size in sorted(grouped):
        best_here = min(r['validation_loss'] for r in grouped[size])
        if best_here < best_at_smaller_size:
            result.extend(r for r in grouped[size] if r['validation_loss'] == best_here)
        best_at_smaller_size = min(best_at_smaller_size, best_here)
    return result


def independent_logits(gene, weights, x):
    """Forward pass без predict_logits/binary_loss из основной реализации."""
    columns = [i for i in range(4) if (gene['mask'] >> i) & 1]
    features = x[:, columns]
    h = WIDTHS[gene['width_index']]
    if h == 0:
        return features.dot(weights['w']) + weights['b'].item()
    affine = features.dot(weights['w1']) + weights['b1']
    hidden = np.tanh(affine) if gene['activation'] == 0 else np.fmax(affine, 0.)
    return hidden.dot(weights['w2']) + weights['b2'].item()


def independent_metrics(labels, logits):
    # logaddexp обеспечивает устойчивую BCE, в том числе для уверенных ошибок.
    loss = float(np.sum(np.logaddexp(0., logits) - labels * logits) / len(labels))
    return loss, float(np.count_nonzero((logits >= 0.) == labels) / len(labels))


def training_gradient_oracle():
    """Центральные конечные разности против реальных шагов train_model.

    Проверяется полная регуляризованная training-цель, включая bias и L2.
    Производная выводится из фактически выполненного шага, без копирования backprop.
    """
    rng = np.random.default_rng(3072026)
    x = rng.normal(size=(9, 4))
    y = np.array([0., 1., 0., 1., 1., 0., 0., 1., 1.])
    eps, largest_error, coordinates = 1e-6, 0., 0
    for gene in [Gene(15, 0, 0), Gene(7, 1, 0), Gene(7, 1, 1)]:
        for steps_before in (0, 1):
            base = train_model(gene, x, y, 73, epochs=steps_before)
            after = train_model(gene, x, y, 73, epochs=steps_before + 1)

            def objective(weights):
                logits = independent_logits({'mask': gene.mask, 'width_index': gene.width_index,
                                             'activation': gene.activation}, weights, x)
                regularization = .5e-4 * sum(float(np.sum(v*v)) for k, v in weights.items() if k.startswith('w'))
                return independent_metrics(y, logits)[0] + regularization

            before_loss = objective(base)
            require(objective(after) < before_loss, 'gradient oracle: a training step did not decrease regularized loss')
            for key, value in base.items():
                analytical = (value - after[key]) / LEARNING_RATE
                numerical = np.zeros_like(value)
                for index in np.ndindex(value.shape):
                    plus, minus = {k:v.copy() for k,v in base.items()}, {k:v.copy() for k,v in base.items()}
                    plus[key][index] += eps
                    minus[key][index] -= eps
                    numerical[index] = (objective(plus) - objective(minus)) / (2*eps)
                    coordinates += 1
                largest_error = max(largest_error, float(np.max(np.abs(analytical - numerical))))
                require(np.allclose(analytical, numerical, rtol=2e-5, atol=5e-8),
                        f'gradient oracle mismatch: {gene}, step {steps_before + 1}, {key}')
    return {'architectures': ['linear', 'tanh', 'relu'], 'steps_per_architecture': 2,
            'finite_difference_coordinates': coordinates, 'epsilon': eps,
            'max_absolute_gradient_error': largest_error,
            'regularized_loss_decreased_in_all_six_steps': True}


def verify_saved(replay=False, *, directory=None):
    directory = Path(directory) if directory is not None else ROOT
    errors = []
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    results = json.loads((directory / 'results.json').read_text(encoding='utf-8'))
    expected_config = {'variant': 10, 'N': 28, 'Q': 1, 'Y': 26, 'track': 'B', 'group': 517,
                       'data_seed': DATA_SEED, 'population': POPULATION, 'fitness_budget': 240,
                       'epochs_per_training': EPOCHS, 'learning_rate': LEARNING_RATE, 'l2': 1e-4,
                       'seeds': list(range(100, 120)), 'penalties': [0., .004],
                       'widths': list(WIDTHS), 'activations': list(ACTIVATIONS),
                       'train_count': 280, 'validation_count': 100, 'test_count': 100,
                       'tournament': 3, 'mask_bit_mutation': .12, 'width_mutation': .35,
                       'activation_mutation': .15, 'elitism': 1}
    for key, value in expected_config.items():
        require(config[key] == value, f'config.json: {key} differs from implemented archival experiment')
    require(1 + ((17*config['N'] + 7*config['Q'] + config['Y']) % 20) == config['variant'], 'variant formula mismatch')

    rows = csv_rows(directory, 'data.csv')
    require(len(rows) == 480 and {int(r['id']) for r in rows} == set(range(480)), 'data.csv: IDs/row count mismatch')
    rows.sort(key=lambda r: int(r['id']))
    x = np.array([[float(r[f'x{i}']) for i in range(1, 5)] for r in rows])
    y = np.array([float(r['y']) for r in rows])
    indices_csv = {s: np.array([i for i, r in enumerate(rows) if r['split'] == s]) for s in ['train', 'validation', 'test']}
    generated_x, generated_z, generated_y, indices, generated_mean, generated_scale = make_data(config['data_seed'])
    require(np.array_equal(x, generated_x) and np.array_equal(y, generated_y), 'data.csv differs from recorded data seed')
    for name, count in [('train', 280), ('validation', 100), ('test', 100)]:
        require(len(indices_csv[name]) == count and set(indices_csv[name]) == set(indices[name]), f'data.csv: {name} split mismatch')
    mean = np.sum(x[indices_csv['train']], axis=0) / len(indices_csv['train'])
    scale = np.sqrt(np.sum((x[indices_csv['train']] - mean)**2, axis=0) / len(indices_csv['train']))
    scale = np.maximum(scale, 1e-12)
    close(config['standardization_mean'], mean, 'config.json: train-only scaler mean', errors)
    close(config['standardization_scale'], scale, 'config.json: train-only scaler scale', errors)
    close(generated_mean, mean, 'generator/scaler mean', errors)
    close(generated_scale, scale, 'generator/scaler scale', errors)
    z = (x - mean) / scale

    runs = [record(r) for r in csv_rows(directory, 'runs.csv')]
    require(len(runs) == 40, 'runs.csv: expected all 40 saved runs')
    run_map = {(r['penalty'], r['seed']): r for r in runs}
    expected_keys = {(p, s) for p in config['penalties'] for s in config['seeds']}
    require(len(run_map) == 40 and set(run_map) == expected_keys, 'runs.csv: configuration/seed set mismatch')
    for run in runs:
        gene = Gene(**run['gene'])
        require(gene.valid() and gene == gene.canonical() and run['key'] == gene.key(), 'runs.csv: invalid/noncanonical genotype or key')
        require(run['evaluations'] == 240 and run['parameters'] == gene.parameters, 'runs.csv: budget/parameter count mismatch')
        require(run['training_steps'] == run['distinct_trainings'] * EPOCHS and 1 <= run['distinct_trainings'] <= 135,
                'runs.csv: training/cache accounting mismatch')
        close(run['fitness'], run['validation_loss'] + run['penalty']*run['parameters'], 'runs.csv: fitness formula', errors)
        require(np.isfinite(run['seconds']) and run['seconds'] >= 0, 'runs.csv: invalid elapsed time')

    histories = defaultdict(list)
    for row in csv_rows(directory, 'history.csv'):
        histories[(float(row['penalty']), int(row['seed']))].append(
            {'evaluations': int(row['evaluations']), 'best_fitness': float(row['best_fitness']),
             'distinct_trainings': int(row['distinct_trainings'])})
    require(set(histories) == expected_keys, 'history.csv: configuration/seed set mismatch')
    for key, history in histories.items():
        require([h['evaluations'] for h in history] == list(range(1, 241)), f'history.csv {key}: budget sequence mismatch')
        require(all(a['best_fitness'] >= b['best_fitness'] for a, b in zip(history, history[1:])), f'history.csv {key}: incumbent worsened')
        require(history[0]['distinct_trainings'] == 1, f'history.csv {key}: initial cache count mismatch')
        require(all(b['distinct_trainings'] - a['distinct_trainings'] in (0, 1) for a, b in zip(history, history[1:])),
                f'history.csv {key}: invalid cache counter')
        require(history[-1]['distinct_trainings'] == run_map[key]['distinct_trainings'], f'history.csv {key}: final cache count mismatch')
        close(history[-1]['best_fitness'], run_map[key]['fitness'], f'history.csv {key}: final fitness', errors)

    evaluated = [record(r) for r in csv_rows(directory, 'evaluated_architectures.csv')]
    evaluated_map = {identity(r): r for r in evaluated}
    require(len(evaluated) == len(evaluated_map), 'evaluated_architectures.csv: repeated seed/genotype key')
    for item in evaluated:
        gene = Gene(**item['gene'])
        require(gene.valid() and gene == gene.canonical() and gene.key() == item['key'] and gene.parameters == item['parameters'],
                'evaluated_architectures.csv: invalid genotype/parameter count')
        require(item['seed'] in config['seeds'] and np.isfinite(item['validation_loss']) and 0 <= item['validation_accuracy'] <= 1,
                'evaluated_architectures.csv: invalid seed/validation metrics')
    for run in runs:
        require(identity(run) in evaluated_map, 'run winner is absent from evaluated architectures')
        compare_record(run, evaluated_map[identity(run)], 'winner/evaluated cache', errors)
    archive = [record(r) for r in csv_rows(directory, 'pareto_archive.csv')]
    require(len(archive) == results['archive_count'], 'archive_count mismatch')
    expected_front = oracle_front(evaluated)
    require({identity(r) for r in archive} == {identity(r) for r in expected_front}, 'Pareto archive differs from independent sorted-scan oracle')
    require({identity(r) for r in nondominated(evaluated)} == {identity(r) for r in expected_front}, 'nondominated implementation differs from oracle')
    require(len(archive) == len({identity(r) for r in archive}), 'Pareto archive repeats same seed/genotype')
    for item in archive:
        compare_record(item, evaluated_map[identity(item)], 'archive/evaluated cache', errors)

    before_test = json.loads((directory / 'selection_before_test.json').read_text(encoding='utf-8'))
    require(before_test['test_used_for_selection'] is False, 'selection_before_test.json: test selection flag')
    selected = select_representatives(archive)
    require(len(before_test['selected']) == 3 and [identity(r) for r in before_test['selected']] == [identity(r) for r in selected],
            'selection_before_test.json: representative rule mismatch')
    for item in before_test['selected']:
        require(not any(k.startswith('test_') for k in item), 'test metrics present in pre-test selection')
        compare_record(item, evaluated_map[identity(item)], 'pre-test selection/cache', errors)
    representatives = [record(r) for r in csv_rows(directory, 'representatives.csv')]
    require(len(representatives) == 3 and len({r['parameters'] for r in representatives}) == 3, 'three different model complexities required')
    require([identity(r) for r in representatives] == [identity(r) for r in selected], 'representatives.csv: selection mismatch')
    require(len(results['representatives']) == 3, 'results.json: representative count mismatch')
    representative_weights = {}
    for i, (item, item_json) in enumerate(zip(representatives, results['representatives']), 1):
        compare_record(item, item_json, f'representative {i}: CSV/JSON', errors)
        model_path = directory / f'model_{i}.npz'
        with np.load(model_path, allow_pickle=False) as saved:
            weights = {k: saved[k].copy() for k in saved.files if k not in ['gene', 'mean', 'scale']}
            require(saved['gene'].tolist() == [item['gene'][k] for k in ['mask', 'width_index', 'activation']], f'{model_path.name}: gene mismatch')
            close(saved['mean'], mean, f'{model_path.name}: train-only mean', errors)
            close(saved['scale'], scale, f'{model_path.name}: train-only scale', errors)
            model_z = (x - saved['mean']) / saved['scale']
        require(sum(a.size for a in weights.values()) == item['parameters'], f'{model_path.name}: number of saved weights mismatch')
        representative_weights[identity(item)] = weights
        for name in ['validation', 'test']:
            logits = independent_logits(item['gene'], weights, model_z[indices_csv[name]])
            loss, acc = independent_metrics(y[indices_csv[name]], logits)
            close(loss, item[f'{name}_loss'], f'{model_path.name}: {name}_loss', errors)
            close(acc, item[f'{name}_accuracy'], f'{model_path.name}: {name}_accuracy', errors)
            close(item[f'{name}_loss'], item_json[f'{name}_loss'], f'{model_path.name}: results.json {name}_loss', errors)
            close(item[f'{name}_accuracy'], item_json[f'{name}_accuracy'], f'{model_path.name}: results.json {name}_accuracy', errors)

    require(len(results['summaries']) == 2 and {s['penalty'] for s in results['summaries']} == set(config['penalties']),
            'results.json: expected one summary for each penalty')
    for summary in results['summaries']:
        subset = [r for r in runs if r['penalty'] == summary['penalty']]
        require(len(subset) == 20, 'results.json: summary configuration mismatch')
        for metric in ['fitness', 'validation_loss', 'test_loss', 'test_accuracy', 'parameters', 'seconds', 'distinct_trainings']:
            values = np.array([r[metric] for r in subset])
            maximize = metric == 'test_accuracy'
            oracle = {'best': float(max(values) if maximize else min(values)), 'mean': float(np.sum(values)/len(values)),
                      'median': float(np.median(values)), 'std': float(np.sqrt(np.sum((values-values.mean())**2)/(len(values)-1))),
                      'worst': float(min(values) if maximize else max(values))}
            for statistic, value in oracle.items():
                close(summary[metric][statistic], value, f'results.json {metric}/{statistic}', errors)

    # Baseline-метрики также пересчитываются, без готовых предикторов основной программы.
    require(len(results['baselines']) == 2, 'results.json: baseline count mismatch')
    probability = float(np.count_nonzero(generated_y[indices['train']]) / len(indices['train']))
    constant_logits = np.full(100, np.log(probability/(1-probability)))
    loss, acc = independent_metrics(generated_y[indices['test']], constant_logits)
    require(results['baselines'][0]['parameters'] == 1, 'constant baseline size mismatch')
    close(results['baselines'][0]['test_loss'], loss, 'constant baseline test_loss', errors)
    close(results['baselines'][0]['test_accuracy'], acc, 'constant baseline test_accuracy', errors)
    linear_gene = Gene(15, 0, 0)
    linear_weights = train_model(linear_gene, generated_z[indices['train']], generated_y[indices['train']], 100)
    require(results['baselines'][1]['parameters'] == 5, 'linear baseline size mismatch')
    for name in ['validation', 'test']:
        logits = independent_logits({'mask':15, 'width_index':0, 'activation':0}, linear_weights, generated_z[indices[name]])
        loss, acc = independent_metrics(generated_y[indices[name]], logits)
        close(results['baselines'][1][name+'_loss'], loss, f'linear baseline {name}_loss', errors)
        if name == 'test':
            close(results['baselines'][1]['test_accuracy'], acc, 'linear baseline test_accuracy', errors)

    gradient_check = training_gradient_oracle()
    replay_info = {'performed': bool(replay), 'runs': 0, 'history_checkpoints': 0,
                   'actual_trainings': 0, 'max_absolute_metric_difference': None,
                   'seconds_compared': False}
    if replay:
        replay_errors, union = [], {}
        for penalty in config['penalties']:
            for seed in config['seeds']:
                expected = run_map[(penalty, seed)]
                actual, history, cache, models = search(seed, penalty, 240, generated_z, generated_y, indices)
                compare_record(actual, expected, f'replay {penalty}/{seed}', replay_errors)
                for key in ['evaluations', 'distinct_trainings', 'training_steps']:
                    require(actual[key] == expected[key], f'replay {penalty}/{seed}: {key} mismatch')
                close(actual['fitness'], expected['fitness'], f'replay {penalty}/{seed}: fitness', replay_errors)
                for a, b in zip(history, histories[(penalty, seed)]):
                    require(a['evaluations'] == b['evaluations'] and a['distinct_trainings'] == b['distinct_trainings'],
                            f'replay {penalty}/{seed}: history counters mismatch')
                    close(a['best_fitness'], b['best_fitness'], f'replay {penalty}/{seed}: history fitness', replay_errors)
                require(len(history) == len(histories[(penalty, seed)]) == 240, 'replay history length mismatch')
                gene = expected['gene']
                logits = independent_logits(gene, models[expected['key']], generated_z[indices['test']])
                test_loss, test_accuracy = independent_metrics(generated_y[indices['test']], logits)
                close(test_loss, expected['test_loss'], f'replay {penalty}/{seed}: winner test_loss', replay_errors)
                close(test_accuracy, expected['test_accuracy'], f'replay {penalty}/{seed}: winner test_accuracy', replay_errors)
                for item in cache:
                    union.setdefault(identity(item), item)
                    if identity(item) in representative_weights:
                        saved_weights = representative_weights[identity(item)]
                        require(set(models[item['key']]) == set(saved_weights), 'replay representative weight keys mismatch')
                        for key, weights in saved_weights.items():
                            close(models[item['key']][key], weights,
                                  f'replay representative {identity(item)} weights {key}', replay_errors)
                replay_info['runs'] += 1
                replay_info['history_checkpoints'] += len(history)
                replay_info['actual_trainings'] += actual['distinct_trainings']
        require(set(union) == set(evaluated_map), 'replay evaluated architecture union mismatch')
        for key, item in union.items():
            compare_record(item, evaluated_map[key], f'replay evaluated architecture {key}', replay_errors)
        replay_info['max_absolute_metric_difference'] = max(replay_errors, default=0.)
        errors.extend(replay_errors)

    hashed_files = sorted({*directory.glob('*.csv'), *directory.glob('*.npz'),
                           *(p for p in directory.glob('*.json') if p.name != 'audit.json'),
                           directory/'lab3.py', directory/'audit_saved.py', directory/'test_lab3.py'})
    return {'status': 'passed', 'lab': 3, 'variant': 10, 'track': 'B',
            'environment': {'python': platform.python_version(), 'numpy': importlib.metadata.version('numpy')},
            'saved_runs_checked': len(runs), 'saved_history_checkpoints_checked': sum(map(len, histories.values())),
            'data_rows_checked': len(rows), 'split_sizes': {k:len(v) for k,v in indices_csv.items()},
            'scaler_train_only': True, 'held_out_test_excluded_from_selection': True,
            'saved_models_independently_recomputed': 3, 'three_distinct_complexities': [r['parameters'] for r in representatives],
            'baseline_models_independently_checked': 2,
            'evaluated_architectures_checked': len(evaluated), 'archive_points_checked': len(archive),
            'pareto_oracle': 'independent grouped sorted scan; equal objective points retained',
            'budgets': {'candidate_calls_per_run': 240, 'cache_hits_counted': True,
                        'actual_trainings_total_across_runs': sum(r['distinct_trainings'] for r in runs),
                        'gradient_steps_total_across_runs': sum(r['training_steps'] for r in runs),
                        'equal_candidate_calls_do_not_imply_equal_training_cost': True},
            'gradient_oracle': gradient_check, 'max_absolute_numeric_difference': max(errors, default=0.),
            'replay': replay_info,
            'notes': ['The audit checks the fixed archival plan, not arbitrary custom output folders.',
                      'Test content never enters search, fitness, training, archive, or representative selection.',
                      'No numerical experiment artifacts were modified by this audit.'],
            'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in hashed_files}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', action='store_true', help='повторить все 40 запусков и 9600 точек истории')
    args = parser.parse_args()
    print(json.dumps(verify_saved(args.replay), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
