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


def verify_saved(replay=False, root=None):
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
            close(reproduced["best"],row["best"],f"replay fitness {key}")
            for field in ["evaluations","generations","migration_slots","diversity"]:
                close(reproduced[field],row[field],f"replay result {key}/{field}")
            lo,hi=(0.,math.pi) if key[0]=="michalewicz" else (-5.12,5.12)
            vector=np.asarray(reproduced["solution"])
            require(vector.shape==(config["dimension"],) and np.all((vector>=lo)&(vector<=hi)),f"replay bounds {key}")
            close(scalar_objective(reproduced["solution"],key[0]),reproduced["best"],f"oracle replay {key}")
            if key[0] in best and key[1:]==(best[key[0]]["method"],best[key[0]]["seed"]):
                require(np.allclose(reproduced["solution"],best[key[0]]["solution"],rtol=1e-10,atol=1e-11),f"вектор рекорда {key}")
            require(len(history)==len(histories[key]),f"replay history length {key}")
            for expected_point,point in zip(histories[key],history):
                for field in ["generation","evaluations"]:
                    require(point[field]==int(expected_point[field]),f"replay history grid {key}/{field}")
                for field in ["best","diversity"]:
                    close(point[field],expected_point[field],f"replay history {key}/{point['generation']}/{field}")
                    max_delta=max(max_delta,abs(point[field]-float(expected_point[field])))
            replayed+=1
    return dict(status="passed",runs_checked=len(actual),history_checkpoints_checked=sum(map(len,histories.values())),
                summary_groups_checked=len(summaries),paired_comparisons_checked=len(pairs),best_vectors_checked=len(best),
                replayed_runs=replayed,instrumented_objective_evaluations=counted,max_replay_numeric_error=max_delta,
                replay_note="Сравниваются все сохранённые fitness/траектории и три архивных вектора; остальные векторы исходно не сохранялись.",
                seconds=perf_counter()-started)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay",action="store_true",help="Повторить все360 запусков без перезаписи данных")
    args=parser.parse_args()
    print(json.dumps(verify_saved(args.replay),ensure_ascii=False,indent=2))
