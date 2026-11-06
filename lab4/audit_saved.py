"""Независимая проверка CSV/JSON и полное воспроизведение островных серий."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import main as algorithm


def require(condition, message):
    if not condition:
        raise ValueError("ЛР4: " + message)


def close(actual, expected, label):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)) and
            math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-11),
            f"{label}: {actual} != {expected}")


def read_csv(root, name):
    with (root/name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def scalar_objective(vector, function):
    """Оракул использует math и скалярные суммы, не production evaluate."""
    if function == "sphere":
        return math.fsum(float(x)**2 for x in vector)
    if function == "rastrigin":
        return math.fsum(float(x)**2 - 10*math.cos(2*math.pi*float(x)) + 10 for x in vector)
    return -math.fsum(math.sin(float(x))*math.sin(i*float(x)**2/math.pi)**20
                      for i, x in enumerate(vector, 1))


def describe(values):
    values = np.asarray(values, dtype=float)
    return dict(best=float(values.min()), mean=float(values.mean()), median=float(np.median(values)),
                std=float(values.std(ddof=1)) if len(values)>1 else 0., worst=float(values.max()))


def mean_drift_guard(reference, repeated):
    """Заранее фиксированный инженерный порог, НЕ тест эквивалентности."""
    reference=np.asarray(reference,dtype=float);repeated=np.asarray(repeated,dtype=float)
    require(reference.ndim==repeated.ndim==1 and reference.shape==repeated.shape and len(reference)>0,
            'несогласованные серии для проверки дрейфа среднего')
    require(np.all(np.isfinite(reference)) and np.all(np.isfinite(repeated)),'неконечная серия дрейфа')
    threshold=max(1e-11,3*float(np.std(reference,ddof=1))/math.sqrt(len(reference))) if len(reference)>1 else 1e-11
    delta=float(repeated.mean()-reference.mean())
    require(abs(delta)<=threshold,f'дрейф среднего {delta} превышает фиксированный порог {threshold}')
    return dict(mean_delta=delta,threshold=threshold,rule='abs(mean_delta) <= max(1e-11, 3*archive_sample_std/sqrt(n))',
                interpretation='fixed engineering guard; not proof of statistical or trajectory equivalence',
                archive=describe(reference),repeated=describe(repeated))


def same_environment_match(first, first_history, second, second_history):
    for field in ['function','method','seed','solution','best','evaluations','generations','migration_slots','diversity']:
        require(first[field]==second[field],f'недетерминированный повтор в одной среде: {field}')
    require(first_history==second_history,'недетерминированная история в одной среде')


def verify_saved(replay=False, root=None, strict_archive=False):
    replay=bool(replay or strict_archive)
    root = Path(root) if root is not None else algorithm.HERE
    started = perf_counter()
    config = json.loads((root/"config.json").read_text(encoding="utf-8"))
    algorithm.validate_config(config)
    rows = read_csv(root, "runs.csv")
    expected = {(fn, method, seed) for fn in config["functions"] for method in config["methods"] for seed in config["seeds"]}
    actual = {}
    total, budget = config["total_population"], config["evaluation_budget"]
    generations = budget//total-1
    for row in rows:
        key = row["function"], row["method"], int(row["seed"])
        require(key in expected and key not in actual, f"неожиданный/повторный запуск {key}")
        require(int(row["evaluations"]) == budget and int(row["generations"]) == generations, f"бюджет {key}")
        close(row["best"], row["best"], f"конечность fitness {key}")
        require(math.isfinite(float(row["diversity"])) and 0 <= float(row["diversity"]) <= .5+1e-12, f"разнообразие {key}")
        require(math.isfinite(float(row["seconds"])) and float(row["seconds"]) >= 0, f"время {key}")
        slots = generations//config["migration_interval"]*config["islands"] if row["method"] in ("ring", "fully_connected") else 0
        require(int(row["migration_slots"]) == slots, f"слоты миграции {key}")
        actual[key] = row
    require(set(actual) == expected, "неполная серия запусков")

    histories = defaultdict(list)
    for h in read_csv(root, "history.csv"):
        key = h["function"], h["method"], int(h["seed"])
        require(key in expected, f"неизвестный запуск в истории {key}")
        histories[key].append(h)
    require(set(histories) == expected, "неполный набор историй")
    for key, history in histories.items():
        require(len(history) == generations+1, f"длина истории {key}")
        previous = math.inf
        for g, point in enumerate(history):
            require(int(point["generation"]) == g and int(point["evaluations"]) == (g+1)*total, f"сетка бюджета/поколений {key}")
            value = float(point["best"])
            require(math.isfinite(value) and value <= previous+1e-11, f"рекорд/конечность {key}, g={g}")
            require(math.isfinite(float(point["diversity"])) and 0 <= float(point["diversity"]) <= .5+1e-12, f"разнообразие истории {key}")
            previous = value
        close(history[-1]["best"], actual[key]["best"], f"итог истории {key}")
        close(history[-1]["diversity"], actual[key]["diversity"], f"итог разнообразия {key}")
    for fn in config["functions"]:
        for seed in config["seeds"]:
            base=histories[(fn,"panmixia",seed)][0]
            for method in config["methods"]:
                for field in ("best","diversity"):
                    close(histories[(fn,method,seed)][0][field],base[field],f"общая начальная популяция {fn}/{seed}/{method}/{field}")

    expected_groups = {(fn,m) for fn in config["functions"] for m in config["methods"]}
    summaries = {}
    for row in read_csv(root, "summary.csv"):
        key = row["function"],row["method"]
        require(key in expected_groups and key not in summaries, f"неожиданная/повторная сводка {key}")
        group = [actual[(key[0],key[1],s)] for s in config["seeds"]]
        for field, value in describe([float(r["best"]) for r in group]).items():
            close(row[field],value,f"summary {key}/{field}")
        close(row["mean_seconds"],np.mean([float(r["seconds"]) for r in group]),f"summary time {key}")
        require(int(row["runs"]) == len(group) and int(row["evaluations"]) == budget,f"summary counts {key}")
        summaries[key]=row
    require(set(summaries) == expected_groups,"неполная сводка")

    pairs = read_csv(root,"paired_comparison.csv")
    expected_pairs = {(fn,m) for fn,m in expected_groups if m != "panmixia"}
    keyed_pairs = {(p["function"],p["method"]):p for p in pairs}
    require(len(keyed_pairs) == len(pairs) and set(keyed_pairs) == expected_pairs,"состав парных сравнений")
    bootstrap = np.random.default_rng(config["bootstrap_seed"])
    for fn in config["functions"]:
        base = np.array([float(actual[(fn,"panmixia",s)]["best"]) for s in config["seeds"]])
        for method in config["methods"]:
            if method == "panmixia":continue
            values=np.array([float(actual[(fn,method,s)]["best"]) for s in config["seeds"]])
            delta=values-base
            means=delta[bootstrap.integers(0,len(delta),(10000,len(delta)))].mean(axis=1)
            lo,hi=np.quantile(means,[.025,.975])
            wanted=dict(mean_delta_vs_panmixia=delta.mean(),ci95_low=lo,ci95_high=hi,
                        win_fraction=np.mean(delta < -1e-12),tie_fraction=np.mean(np.abs(delta)<=1e-12))
            for field,value in wanted.items():close(keyed_pairs[(fn,method)][field],value,f"pairs {fn}/{method}/{field}")

    best=json.loads((root/"best_solutions.json").read_text(encoding="utf-8"))
    require(set(best) == set(config["functions"]),"состав лучших решений")
    for fn, record in best.items():
        key=fn,record["method"],record["seed"]
        require(key in actual and record["function"]==fn,"метаданные лучшего решения")
        vector=np.asarray(record["solution"],dtype=float)
        lo,hi=(0.,math.pi) if fn=="michalewicz" else (-5.12,5.12)
        require(vector.shape==(config["dimension"],) and np.all(np.isfinite(vector)) and np.all((vector>=lo)&(vector<=hi)),f"границы лучшего {fn}")
        close(record["best"],scalar_objective(vector,fn),f"независимый fitness лучшего {fn}")
        close(record["best"],min(float(r["best"]) for k,r in actual.items() if k[0]==fn),f"глобальный рекорд серии {fn}")
        for field in ["best","evaluations","generations","migration_slots","diversity","seconds"]:
            close(record[field],actual[key][field],f"best/run {fn}/{field}")

    # Таблицы PDF строятся из этих JSON: сверяем их с CSV до пересборки PDF.
    report=json.loads((root/"report_content.json").read_text(encoding="utf-8"))
    table=next(s["table"] for s in report["sections"] if s["heading"]=="3. Экспериментальная методика")
    require(len(table["rows"])==len(summaries),"строки статистики отчёта")
    labels={v:k for k,v in algorithm.LABELS.items()}
    seen=set()
    for row in table["rows"]:
        key=row[0],labels[row[1]]
        require(key in summaries and key not in seen,"неизвестная/повторная конфигурация в отчёте")
        seen.add(key)
        for column,field in enumerate(["best","mean","median","std","worst"],2):
            # Исходный формат отчёта печатает шесть значащих цифр.
            require(math.isclose(float(row[column]),float(summaries[key][field]),rel_tol=6e-6,abs_tol=1e-10),f"таблица отчёта {key}/{field}")
    require(seen==expected_groups,"неполный набор конфигураций отчёта")
    pair_table=next(s["table"] for s in report["sections"] if s["heading"]=="6. Парные сравнения")
    seen=set()
    for row in pair_table["rows"]:
        key=row[0],labels[row[1]]
        require(key in keyed_pairs and key not in seen,"неизвестная/повторная пара в отчёте")
        seen.add(key)
        for col,field in enumerate(["mean_delta_vs_panmixia","ci95_low","ci95_high"],2):
            require(str(row[col])==f'{float(keyed_pairs[key][field]):.5g}',f"пара в отчёте {key}/{field}")
        require(row[5]==f'{float(keyed_pairs[key]["win_fraction"]):.1%}',f"доля побед в отчёте {key}")
    require(seen==expected_pairs,"неполные парные сравнения в отчёте")
    timing_table=next(s["table"] for s in report["sections"] if s["heading"].startswith("7."))
    seen=set()
    for row in timing_table["rows"]:
        fn=row[0]
        require(fn in config["functions"] and fn not in seen,"повтор/неизвестная функция времени")
        seen.add(fn)
        times=[float(summaries[(fn,m)]["mean_seconds"]) for m in config["methods"]]
        require(row[1]==f'{min(times):.3f}–{max(times):.3f}' and int(row[2])==budget,f"время/бюджет в отчёте {fn}")
    require(seen==set(config["functions"]),"неполная таблица времени")
    for s in report["sections"]:
        if s.get("image"):require((root/s["image"]).is_file(),f"нет рисунка {s['image']}")

    max_delta, counted, replayed = 0., 0, 0
    archive_equivalent=True;differences=[];rerun_values=defaultdict(list)
    if replay:
        original=algorithm.evaluate
        def instrument(points,fn):
            nonlocal counted
            counted+=len(points)
            return original(points,fn)
        for key,row in actual.items():
            before=counted
            with patch.object(algorithm,"evaluate",side_effect=instrument):
                reproduced,history=algorithm.run(*key,config)
            require(counted-before==budget,f"фактический счётчик оценок {key}")
            before=counted
            with patch.object(algorithm,"evaluate",side_effect=instrument):
                second,second_history=algorithm.run(*key,config)
            require(counted-before==budget,f'счётчик второго повтора {key}')
            same_environment_match(reproduced,history,second,second_history)
            for field,value in zip(['function','method','seed'],key):
                require(reproduced[field]==value,f'replay identity {key}/{field}')
            for field in ["evaluations","generations","migration_slots"]:
                require(reproduced[field]==int(row[field]),f"replay result {key}/{field}")
            lo,hi=(0.,math.pi) if key[0]=="michalewicz" else (-5.12,5.12)
            vector=np.asarray(reproduced["solution"])
            require(vector.shape==(config["dimension"],) and np.all(np.isfinite(vector)) and np.all((vector>=lo)&(vector<=hi)),f"replay bounds {key}")
            close(scalar_objective(reproduced["solution"],key[0]),reproduced["best"],f"oracle replay {key}")
            differences_for_run=[];first_different_generation=None;vector_within=True;vector_exact=True
            if key[0] in best and key[1:]==(best[key[0]]["method"],best[key[0]]["seed"]):
                differences_for_run.extend(abs(float(a)-float(b)) for a,b in zip(reproduced['solution'],best[key[0]]['solution']))
                vector_within=all(math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-11) for a,b in zip(reproduced['solution'],best[key[0]]['solution']))
                vector_exact=reproduced['solution']==best[key[0]]['solution']
                if strict_archive:require(vector_within,f"вектор рекорда {key}")
            require(len(history)==len(histories[key]),f"replay history length {key}")
            previous=math.inf;exact_checkpoints=0;within_checkpoints=0;first_nonexact_generation=None
            for expected_point,point in zip(histories[key],history):
                for field,value in zip(['function','method','seed'],key):
                    require(point[field]==value,f'replay history identity {key}/{field}')
                for field in ["generation","evaluations"]:
                    require(point[field]==int(expected_point[field]),f"replay history grid {key}/{field}")
                for field in ["best","diversity"]:
                    require(math.isfinite(float(point[field])),f'неконечная история {key}/{field}')
                    delta=abs(point[field]-float(expected_point[field]));differences_for_run.append(delta)
                    max_delta=max(max_delta,delta)
                    if first_nonexact_generation is None and point[field]!=float(expected_point[field]):
                        first_nonexact_generation=point['generation']
                    if first_different_generation is None and not math.isclose(point[field],float(expected_point[field]),rel_tol=1e-10,abs_tol=1e-11):
                        first_different_generation=point['generation']
                    if strict_archive:close(point[field],expected_point[field],f"replay history {key}/{point['generation']}/{field}")
                require(point['best']<=previous+1e-11 and 0<=point['diversity']<=.5+1e-12,f'replay monotonicity/diversity {key}')
                previous=point['best']
                exact_checkpoints+=all(point[f]==float(expected_point[f]) for f in ['best','diversity'])
                within_checkpoints+=all(math.isclose(point[f],float(expected_point[f]),rel_tol=1e-10,abs_tol=1e-11) for f in ['best','diversity'])
            for field in ['best','diversity']:
                require(reproduced[field]==history[-1][field],f'replay own final/history {key}/{field}')
                if strict_archive:close(reproduced[field],row[field],f'replay final {key}/{field}')
            final_within=all(math.isclose(reproduced[f],float(row[f]),rel_tol=1e-10,abs_tol=1e-11) for f in ['best','diversity'])
            differences_for_run.extend(abs(reproduced[f]-float(row[f])) for f in ['best','diversity'])
            equivalent=first_different_generation is None and vector_within and final_within
            archive_equivalent=archive_equivalent and equivalent
            max_delta=max(max_delta,max(differences_for_run,default=0.))
            differences.append(dict(function=key[0],method=key[1],seed=key[2],archive_best=float(row['best']),
                                    repeated_best=reproduced['best'],final_delta=reproduced['best']-float(row['best']),
                                    final_diversity_delta=reproduced['diversity']-float(row['diversity']),
                                    final_metrics_within_tolerance=final_within,
                                    max_absolute_difference=max(differences_for_run,default=0.),
                                    first_nonexact_generation=first_nonexact_generation,
                                    first_generation_outside_tolerance=first_different_generation,
                                    exact_history_checkpoint_matches=exact_checkpoints,
                                    history_checkpoints_within_tolerance=within_checkpoints,
                                    saved_global_vector_exact=vector_exact if key[1:]==(best[key[0]]['method'],best[key[0]]['seed']) else None,
                                    archive_equivalent=equivalent))
            rerun_values[key[:2]].append(reproduced['best'])
            replayed+=1
    guards={}
    if replay:
        for fn,method in sorted(expected_groups):
            reference=[float(row['best']) for key,row in actual.items() if key[:2]==(fn,method)]
            guards[fn+'/'+method]=mean_drift_guard(reference,rerun_values[(fn,method)])
    return dict(status="passed",runs_checked=len(actual),history_checkpoints_checked=sum(map(len,histories.values())),
                summary_groups_checked=len(summaries),paired_comparisons_checked=len(pairs),best_vectors_checked=len(best),
                replayed_runs=replayed,instrumented_objective_evaluations=counted,max_replay_numeric_error=max_delta,
                same_environment_exact_double_replays=replayed,strict_archive_requested=bool(strict_archive),
                archive_equivalent_within_declared_tolerance=archive_equivalent if replay else None,
                archive_comparisons=differences,mean_drift_guards=guards,
                archive_comparison_tolerance=dict(relative=1e-10,absolute=1e-11),
                compatibility_policy='https://numpy.org/doc/stable/reference/random/compatibility.html',
                replay_note="Сравниваются все сохранённые fitness/траектории и три архивных вектора; остальные векторы исходно не сохранялись.",
                seconds=perf_counter()-started)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay",action="store_true",help="Повторить все360 запусков без перезаписи данных")
    parser.add_argument('--strict-archive',action='store_true',help='Дополнительно требовать совпадения с архивной траекторией')
    args=parser.parse_args()
    print(json.dumps(verify_saved(args.replay or args.strict_archive,strict_archive=args.strict_archive),ensure_ascii=False,indent=2))
