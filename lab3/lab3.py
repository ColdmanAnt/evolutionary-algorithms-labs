"""ЛР3, вариант 10, трек B. Эволюция архитектуры малого классификатора.

Все операторы поиска и обучение реализованы явно, без sklearn/оптимизаторов.
Запуск: python lab3.py --runs 20 --budget 240 --output .
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np

WIDTHS = (0, 2, 4, 8, 12)
ACTIVATIONS = ("tanh", "relu")
DATA_SEED = 1051728
EPOCHS = 280
LEARNING_RATE = 0.12
POPULATION = 24


def _integer(value, *, minimum=0):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value >= minimum


def _training_arrays(train_x, train_y):
    x, y = np.asarray(train_x, dtype=float), np.asarray(train_y, dtype=float)
    if x.ndim != 2 or x.shape[1] != 4 or len(x) == 0:
        raise ValueError("training features must be a nonempty n-by-4 matrix")
    if y.shape != (len(x),) or not np.all(np.isfinite(x)) or not np.all(np.isin(y, [0., 1.])):
        raise ValueError("training data must be finite and labels must be a binary n-vector")
    return x, y


@dataclass(frozen=True)
class Gene:
    mask: int
    width_index: int
    activation: int

    def canonical(self):
        """У линейной модели нет активации скрытого слоя."""
        return Gene(self.mask, self.width_index, self.activation if self.width_index else 0)

    def valid(self):
        return (_integer(self.mask, minimum=1) and self.mask <= 15 and
                _integer(self.width_index) and self.width_index < len(WIDTHS) and
                _integer(self.activation) and self.activation in (0, 1))

    @property
    def features(self):
        return [i for i in range(4) if self.mask & (1 << i)]

    @property
    def width(self):
        return WIDTHS[self.width_index]

    @property
    def parameters(self):
        d = len(self.features)
        return d + 1 if self.width == 0 else (d + 1) * self.width + self.width + 1

    def key(self):
        return f"{self.mask}:{self.width_index}:{self.canonical().activation}"


def make_data(seed=DATA_SEED):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(480, 4))
    # Первые два признака задают нелинейную границу; третий слабый, четвертый шумовой.
    latent = x[:, 0] ** 2 + x[:, 1] ** 2 + 0.35 * x[:, 2] + rng.normal(0, 0.12, len(x))
    y = (latent > 1.45).astype(float)
    order = rng.permutation(len(x))
    indices = {"train": order[:280], "validation": order[280:380], "test": order[380:]}
    # Никакие статистики validation/test не участвуют в подготовке признаков.
    mean, scale = fit_scaler(x[indices["train"]])
    z = (x - mean) / scale
    return x, z, y, indices, mean, scale


def fit_scaler(train_x):
    train_x = np.asarray(train_x, dtype=float)
    if train_x.ndim != 2 or len(train_x) == 0 or train_x.shape[1] == 0 or not np.all(np.isfinite(train_x)):
        raise ValueError("scaler requires a nonempty finite feature matrix")
    return np.mean(train_x, axis=0), np.maximum(np.std(train_x, axis=0), 1e-12)


def sigmoid(logits):
    return 1 / (1 + np.exp(-np.clip(logits, -40, 40)))


def binary_loss(y, logits):
    return float(np.mean(np.logaddexp(0, logits) - y * logits))


def train_model(gene, train_x, train_y, run_seed, epochs=EPOCHS):
    """Воспроизводимая инициализация зависит от seed запуска и самой архитектуры."""
    if not isinstance(gene, Gene) or not gene.valid():
        raise ValueError("invalid classifier genotype")
    if not _integer(run_seed) or not _integer(epochs):
        raise ValueError("seed and epochs must be nonnegative integers")
    train_x, train_y = _training_arrays(train_x, train_y)
    gene = gene.canonical()
    rng = np.random.default_rng(np.random.SeedSequence([run_seed, gene.mask, gene.width_index, gene.activation]))
    x = train_x[:, gene.features]
    n, d = x.shape
    if gene.width == 0:
        w = np.zeros(d)
        b = np.zeros(1)
        for _ in range(epochs):
            error = sigmoid(x @ w + b[0]) - train_y
            w -= LEARNING_RATE * (x.T @ error / n + 1e-4 * w)
            b[0] -= LEARNING_RATE * np.mean(error)
        return {"w": w, "b": b}
    h = gene.width
    w1 = rng.normal(0, 1 / np.sqrt(d), (d, h))
    b1 = np.zeros(h)
    w2 = rng.normal(0, 1 / np.sqrt(h), h)
    b2 = np.zeros(1)
    for _ in range(epochs):
        pre = x @ w1 + b1
        hidden = np.tanh(pre) if gene.activation == 0 else np.maximum(pre, 0)
        error = sigmoid(hidden @ w2 + b2[0]) - train_y
        grad_w2 = hidden.T @ error / n + 1e-4 * w2
        grad_b2 = np.mean(error)
        derivative = 1 - hidden ** 2 if gene.activation == 0 else (pre > 0)
        back = error[:, None] * w2 * derivative
        w1 -= LEARNING_RATE * (x.T @ back / n + 1e-4 * w1)
        b1 -= LEARNING_RATE * np.mean(back, axis=0)
        w2 -= LEARNING_RATE * grad_w2
        b2[0] -= LEARNING_RATE * grad_b2
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def predict_logits(gene, model, x):
    x = x[:, gene.features]
    if gene.width == 0:
        return x @ model["w"] + model["b"][0]
    hidden = x @ model["w1"] + model["b1"]
    hidden = np.tanh(hidden) if gene.activation == 0 else np.maximum(hidden, 0)
    return hidden @ model["w2"] + model["b2"][0]


def accuracy(y, logits):
    return float(np.mean((logits >= 0) == y))


def random_gene(rng):
    return Gene(int(rng.integers(1, 16)), int(rng.integers(5)), int(rng.integers(2))).canonical()


def crossover(a, b, rng):
    # Равномерный кроссовер отдельных битов маски и структурных генов.
    mask = sum((a.mask if rng.random() < 0.5 else b.mask) & (1 << i) for i in range(4))
    if mask == 0:
        mask = 1 << int(rng.integers(4))
    return Gene(mask, a.width_index if rng.random() < 0.5 else b.width_index,
                a.activation if rng.random() < 0.5 else b.activation).canonical()


def mutate(gene, rng):
    mask, width, activation = gene.mask, gene.width_index, gene.activation
    # Мутации маски, числа нейронов и нелинейности являются независимыми.
    for bit in range(4):
        if rng.random() < 0.12:
            mask ^= 1 << bit
    if mask == 0:
        mask = 1 << int(rng.integers(4))
    if rng.random() < 0.35:
        width = int(rng.choice([i for i in range(5) if i != width]))
    if rng.random() < 0.15:
        activation = 1 - activation
    return Gene(mask, width, activation).canonical()


def nondominated(records):
    """Внешний архив по (ошибка validation, число параметров)."""
    for record in records:
        if (not np.isfinite(record["validation_loss"]) or record["validation_loss"] < 0 or
                not _integer(record["parameters"], minimum=1)):
            raise ValueError("archive objectives must be finite nonnegative loss and positive integer size")
    result = []
    for record in records:
        dominated = any(other["validation_loss"] <= record["validation_loss"] and
                        other["parameters"] <= record["parameters"] and
                        (other["validation_loss"] < record["validation_loss"] or
                         other["parameters"] < record["parameters"])
                        for other in records)
        if not dominated:
            result.append(record)
    return sorted(result, key=lambda r: (r["parameters"], r["validation_loss"]))


def select_representatives(archive):
    """Выбор только по validation/сложности; test-метрики не читаются."""
    unique_complexities = {}
    for item in archive:
        p = item["parameters"]
        if p not in unique_complexities or item["validation_loss"] < unique_complexities[p]["validation_loss"]:
            unique_complexities[p] = item
    front = sorted(unique_complexities.values(), key=lambda r: r["parameters"])
    if len(front) < 3:
        raise ValueError("fewer than three archive complexities; increase --runs or --budget to prepare a complete report")
    return [front[0], front[len(front) // 2], min(front, key=lambda r: r["validation_loss"])]


def search(seed, penalty, budget, z, y, indices, population_size=POPULATION):
    if not _integer(seed) or not _integer(budget) or not _integer(population_size, minimum=4) or budget < population_size:
        raise ValueError("integer seed >= 0 and budget >= population >= 4 required")
    if not isinstance(penalty, (int, float, np.integer, np.floating)) or isinstance(penalty, (bool, np.bool_)) or not np.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be finite and nonnegative")
    z, y = np.asarray(z, dtype=float), np.asarray(y, dtype=float)
    if z.ndim != 2 or z.shape[1] != 4 or y.shape != (len(z),):
        raise ValueError("features must be n-by-4 and labels an n-vector")
    split_indices, normalized_indices = [], {}
    for name in ("train", "validation", "test"):
        part = np.asarray(indices.get(name, []))
        if (part.ndim != 1 or not np.issubdtype(part.dtype, np.integer) or len(part) == 0 or
                np.any(part < 0) or np.any(part >= len(z))):
            raise ValueError("train/validation/test indices must be nonempty integer vectors within the data")
        split_indices.extend(part.tolist())
        normalized_indices[name] = part
    if len(set(split_indices)) != len(split_indices):
        raise ValueError("train/validation/test splits must be disjoint and contain no repeated indices")
    indices = normalized_indices
    # Проверяем обучающую и validation-части; содержимое test не влияет на поиск.
    _training_arrays(z[indices["train"]], y[indices["train"]])
    _training_arrays(z[indices["validation"]], y[indices["validation"]])
    rng = np.random.default_rng(seed)
    train_x, train_y = z[indices["train"]], y[indices["train"]]
    val_x, val_y = z[indices["validation"]], y[indices["validation"]]
    cache, models, history = {}, {}, []
    evaluations = 0
    incumbent = None
    started = perf_counter()

    def evaluate(gene):
        nonlocal evaluations, incumbent
        gene = gene.canonical()
        evaluations += 1
        key = gene.key()
        if key not in cache:
            model = train_model(gene, train_x, train_y, seed)
            logits = predict_logits(gene, model, val_x)
            loss = binary_loss(val_y, logits)
            cache[key] = {"seed": seed, "gene": asdict(gene), "key": key,
                          "validation_loss": loss, "validation_accuracy": accuracy(val_y, logits),
                          "parameters": gene.parameters}
            models[key] = model
        result = cache[key]
        score = result["validation_loss"] + penalty * result["parameters"]
        if incumbent is None or score < incumbent[0]:
            incumbent = (score, gene)
        history.append({"evaluations": evaluations, "best_fitness": incumbent[0],
                        "distinct_trainings": len(cache)})
        return score

    population = [random_gene(rng) for _ in range(population_size)]
    scores = [evaluate(g) for g in population]
    while evaluations < budget:
        offspring, new_scores = [], []
        while len(offspring) < population_size - 1 and evaluations < budget:
            def parent():
                ids = rng.integers(len(population), size=3)
                return population[min(ids, key=lambda i: scores[i])]
            child = mutate(crossover(parent(), parent(), rng), rng)
            offspring.append(child)
            new_scores.append(evaluate(child))
        elite_index = int(np.argmin(scores))
        population = [population[elite_index]] + offspring
        scores = [scores[elite_index]] + new_scores
    score, winner = incumbent
    record = dict(cache[winner.key()])
    record.update(penalty=penalty, fitness=score, evaluations=evaluations,
                  distinct_trainings=len(cache), training_steps=len(cache) * EPOCHS,
                  seconds=perf_counter() - started)
    return record, history, list(cache.values()), models


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def json_dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def describe(values, maximize=False):
    a = np.array(values)
    return {"best": float(a.max() if maximize else a.min()), "mean": float(a.mean()), "median": float(np.median(a)),
            "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "worst": float(a.min() if maximize else a.max())}


def run_experiment(output, runs=20, budget=240):
    if not _integer(runs, minimum=1) or not _integer(budget, minimum=POPULATION):
        raise ValueError("runs must be a positive integer and budget an integer >= 24")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    x, z, y, indices, mean, scale = make_data()
    split_name = {int(i): name for name, values in indices.items() for i in values}
    write_csv(output / "data.csv", [{"id": i, **{f"x{k+1}": float(v) for k, v in enumerate(row)},
                                      "y": int(y[i]), "split": split_name[i]} for i, row in enumerate(x)])
    config = {"variant": 10, "N": 28, "Q": 1, "Y": 26, "track": "B", "group": 517,
              "author": "Холодков Антон Евгеньевич", "data_seed": DATA_SEED,
              "seeds": list(range(100, 100 + runs)), "population": POPULATION,
              "fitness_budget": budget, "epochs_per_training": EPOCHS,
              "learning_rate": LEARNING_RATE, "l2": 1e-4, "penalties": [0.0, 0.004],
              "widths": WIDTHS, "activations": ACTIVATIONS, "train_count": 280,
              "validation_count": 100, "test_count": 100, "standardization_mean": mean.tolist(),
              "standardization_scale": scale.tolist(), "tournament": 3,
              "mask_bit_mutation": 0.12, "width_mutation": 0.35, "activation_mutation": 0.15,
              "crossover": "uniform, probability 1", "elitism": 1,
              "memoization": "per run and configuration; candidate calls counted incl cache hits",
              "stopping": "exact candidate evaluation budget; final generation may be partial"}
    json_dump(output / "config.json", config)
    runs_rows, histories, all_records, all_models = [], [], [], {}
    for penalty in config["penalties"]:
        for seed in config["seeds"]:
            record, history, archive_records, models = search(seed, penalty, budget, z, y, indices)
            runs_rows.append(record)
            histories += [{"penalty": penalty, "seed": seed, **row} for row in history]
            for item in archive_records:
                identifier = (seed, item["key"])
                if identifier not in all_models:
                    all_records.append(item)
                    all_models[identifier] = models[item["key"]]
            print(f"lambda={penalty} seed={seed} fitness={record['fitness']:.4f} p={record['parameters']} fits={record['distinct_trainings']}", flush=True)
    # Выбор решений производится до любого обращения к тестовым целям.
    archive = nondominated(all_records)
    # Три представителя: линейный компактный, средний по сложности, лучший по validation.
    selected = select_representatives(archive)
    json_dump(output / "selection_before_test.json", {"selection_rule": "min complexity, median archive complexity, min validation loss",
              "selected": selected, "test_used_for_selection": False})
    # Только после фиксации выбора раскрываем test и измеряем обобщение.
    representatives = []
    for i, selected_record in enumerate(selected, start=1):
        item = dict(selected_record)
        gene = Gene(**item["gene"])
        model = all_models[(item["seed"], item["key"])]
        logits = predict_logits(gene, model, z[indices["test"]])
        item["test_loss"] = binary_loss(y[indices["test"]], logits)
        item["test_accuracy"] = accuracy(y[indices["test"]], logits)
        item["label"] = ["Компактный", "Компромиссный", "Минимальная ошибка validation"][i-1]
        representatives.append(item)
        np.savez(output / f"model_{i}.npz", **model, mean=mean, scale=scale,
                 gene=np.array([gene.mask, gene.width_index, gene.activation]))
    # Тест для победителей независимых запусков - только финальная оценка, без изменения выбора.
    for item in runs_rows:
        gene = Gene(**item["gene"])
        model = all_models[(item["seed"], item["key"])]
        logits = predict_logits(gene, model, z[indices["test"]])
        item["test_loss"] = binary_loss(y[indices["test"]], logits)
        item["test_accuracy"] = accuracy(y[indices["test"]], logits)
    train_y = y[indices["train"]]
    const_probability = float(np.mean(train_y))
    const_logits = np.full(len(indices["test"]), np.log(const_probability / (1-const_probability)))
    full_linear = Gene(15, 0, 0)
    linear_model = train_model(full_linear, z[indices["train"]], train_y, 100)
    linear_test = predict_logits(full_linear, linear_model, z[indices["test"]])
    linear_val = predict_logits(full_linear, linear_model, z[indices["validation"]])
    baselines = [
        {"model": "Константа по доле класса train", "parameters": 1,
         "test_loss": binary_loss(y[indices["test"]], const_logits),
         "test_accuracy": accuracy(y[indices["test"]], const_logits)},
        {"model": "Логистическая регрессия, все признаки", "parameters": 5,
         "validation_loss": binary_loss(y[indices["validation"]], linear_val),
         "test_loss": binary_loss(y[indices["test"]], linear_test),
         "test_accuracy": accuracy(y[indices["test"]], linear_test)}]
    summaries = []
    for penalty in config["penalties"]:
        rows = [r for r in runs_rows if r["penalty"] == penalty]
        summary = {"penalty": penalty}
        for metric in ["fitness", "validation_loss", "test_loss", "test_accuracy", "parameters", "seconds", "distinct_trainings"]:
            summary[metric] = describe([r[metric] for r in rows], maximize=(metric == "test_accuracy"))
        summaries.append(summary)
    write_csv(output / "runs.csv", runs_rows)
    write_csv(output / "history.csv", histories)
    write_csv(output / "evaluated_architectures.csv", all_records)
    write_csv(output / "pareto_archive.csv", archive)
    write_csv(output / "representatives.csv", representatives)
    json_dump(output / "results.json", {"summaries": summaries, "representatives": representatives,
                                        "baselines": baselines, "archive_count": len(archive)})
    make_plots(output, histories, all_records, archive, representatives)
    make_report(output, config, summaries, representatives, baselines, len(archive))


def make_plots(output, histories, all_records, archive, representatives):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cycler import cycler
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.prop_cycle": cycler(color=["#7d353b", "#707448", "#ad6b42", "#9b948a", "#34312d"]),
                         "text.color": "#34312d", "axes.labelcolor": "#34312d",
                         "axes.edgecolor": "#675d50", "xtick.color": "#34312d",
                         "ytick.color": "#34312d", "grid.color": "#9b948a",
                         "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, penalty in zip(axes, (0.0, 0.004)):
        rows = [r for r in histories if r["penalty"] == penalty]
        seeds = sorted({r["seed"] for r in rows})
        trajectories = np.array([[r["best_fitness"] for r in rows if r["seed"] == seed] for seed in seeds])
        xs = np.arange(1, trajectories.shape[1]+1)
        ax.fill_between(xs, trajectories.min(axis=0), trajectories.max(axis=0), color="#9b948a", alpha=.28, label="min-max по запускам")
        ax.plot(xs, trajectories.mean(axis=0), color="#7d353b", label="среднее")
        ax.set(title=f"Штраф λ = {penalty}", xlabel="Обращения к fitness", ylabel="Лучший fitness (min)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "convergence.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    ax.scatter([r["parameters"] for r in all_records], [r["validation_loss"] for r in all_records], s=8, alpha=.16, color="#9b948a", label="исследованные модели")
    front = sorted(archive, key=lambda r: r["parameters"])
    ax.plot([r["parameters"] for r in front], [r["validation_loss"] for r in front], "o-", color="#707448", ms=4, label="внешний архив")
    for i, r in enumerate(representatives, 1):
        ax.scatter(r["parameters"], r["validation_loss"], s=65, color="#7d353b" if i != 2 else "#ad6b42")
        ax.annotate(str(i), (r["parameters"], r["validation_loss"]), xytext=(6, 6), textcoords="offset points")
    ax.set(xlabel="Число обучаемых параметров", ylabel="BCE на validation", title="Качество и сложность архитектур")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "quality_complexity.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.axis("off")
    labels = ["Данные: train / validation / test; scaler только по train",
              "Генотип: маска признаков + число нейронов + активация",
              "Обучение на train: собственный градиентный спуск",
              "Fitness: BCE(validation) + λ × число параметров",
              "Турнир → структурный кроссовер → мутация → элитизм",
              "Внешний недоминируемый архив: ошибка / сложность",
              "Фиксация 3 моделей по validation → однократный test"]
    for i, label in enumerate(labels):
        yy = .94 - i * .14
        ax.text(.5, yy, label, ha="center", va="center", fontsize=9,
                bbox={"boxstyle":"round,pad=.5", "fc":"#f3eee5", "ec":"#675d50"})
        if i < len(labels)-1:
            ax.annotate("", (.5, yy-.105), (.5, yy-.04), arrowprops={"arrowstyle":"->", "color":"#34312d"})
    fig.tight_layout()
    fig.savefig(output / "algorithm_flow.png", dpi=170)
    plt.close(fig)


def make_report(output, config, summaries, reps, baselines, archive_count):
    def f(x):
        return f"{x:.5f}"
    rows=[]
    for s in summaries:
        m=s["validation_loss"]
        rows.append([str(s["penalty"]),f(m["best"]),f(m["mean"]),f(m["median"]),f(m["std"]),f(m["worst"])])
    rep_rows=[]
    for i,r in enumerate(reps,1):
        g=Gene(**r["gene"])
        rep_rows.append([str(i),str([k+1 for k in g.features]),str(g.width),"linear" if not g.width else ACTIVATIONS[g.activation],str(r["parameters"]),f(r["validation_loss"]),f(r["test_loss"]),f(r["test_accuracy"])])
    compact,best=reps[0],reps[-1]
    s0,s1=summaries
    sections=[
        {"heading":"1. Постановка задачи и вариант", "paragraphs":[
            "Вариант V = 1 + ((17×28 + 7×1 + 26) mod 20) = 10. Выбран трек B: настройка архитектуры малого классификатора. В методичке не задана формула распределения между треками A/B; выбор трека B следует подтвердить при защите, если преподаватель назначает трек отдельно.",
            "Цель: эволюционно подобрать структуру бинарного классификатора, сохраняя компромисс между качеством и сложностью. Архитектура, а не только маска признаков, является объектом поиска: допускается линейная модель либо один скрытый слой различной ширины и нелинейности."]},
        {"heading":"2. Данные и разделение", "paragraphs":[
            "Синтетический набор: 480 объектов, четыре признака x из N(0,1), seed = 1051728. Метка y = I[x1² + x2² + 0.35x3 + ε > 1.45], ε ~ N(0,0.12). Четвёртый признак шумовой. Случайное фиксированное разбиение: train 280, validation 100, test 100. Исходные значения, метки и принадлежность к части сохранены в data.csv.",
            "Средние и стандартные отклонения вычислены только на train. Веса обучаются только на train; fitness, ранжирование, архив и выбор трёх представителей используют только validation. Список выбранных решений сохранён в selection_before_test.json до раскрытия test. Тестовые метрики затем рассчитываются для зафиксированных решений и победителей запусков; они не меняют результат поиска."]},
        {"heading":"3. Формальная модель и представление", "paragraphs":[
            "Генотип g = (m,h,a), где m ∈ {1,…,15} - непустая четырёхбитная маска, h ∈ {0,2,4,8,12} - число скрытых нейронов, a ∈ {tanh,ReLU}. При h=0 активация канонизируется и модель является логистической регрессией. Фенотип - обученная сеть, выбранные признаки и фиксированный scaler.",
            "Для k выбранных признаков число параметров P = k+1 при h=0 и P=(k+1)h+h+1 при h>0. Вероятность класса 1 равна sigmoid(w·z+b) либо sigmoid(v·a(Wz+b)+c). Итого 15×(1+4×2)=135 синтаксически различных архитектур.",
            "Минимизируется Fλ(g)=BCEvalidation(g)+λP(g), BCE = mean(log(1+exp(logit))−y·logit). Сравниваются λ=0 и λ=0.004. Поиск не использует test. Ограничения: непустая маска, допустимая ширина и активация, один скрытый слой; операторы сохраняют синтаксическую корректность."]},
        {"heading":"4. Алгоритм и новые механизмы", "paragraphs":[
            "Начальная популяция состоит из 24 случайных генотипов. Родители выбираются турниром из трёх. Равномерный кроссовер смешивает биты маски и структурные гены. Биты мутируют с вероятностью 0.12 каждый, ширина с вероятностью 0.35, активация с вероятностью 0.15. Пустая маска восстанавливается одним случайным признаком. Элита - одна лучшая особь; остальные места заполняются потомками.",
            "Новый механизм - эволюция структуры сети и внешний недоминируемый архив по (BCEvalidation,P). A доминирует B, когда не хуже по обеим координатам и строго лучше хотя бы по одной. Архив позволяет интерпретировать компромиссы независимо от выбранного λ.",
            "Внутреннее обучение реализовано собственным полным градиентным спуском: 280 шагов, learning rate 0.12, L2=0.0001. Инициализация зависит от seed запуска и генотипа, поэтому оценка повторного генотипа в том же запуске детерминирована. Повторения кэшируются отдельно для каждого запуска и конфигурации.",
            f"Условия сравнения: seeds 100–{99+len(config['seeds'])}, по {len(config['seeds'])} независимых запусков для каждого λ; ровно {config['fitness_budget']} обращений к fitness, включая начальную популяцию и попадания в кэш. Последнее поколение может быть неполным. Меняется только λ. Число фактических обучений и градиентных шагов отдельно записано в runs.csv: равенство числа обращений не означает равенства wall time и обучений. Это ограничение вычислительного сравнения."] ,"image":"algorithm_flow.png"},
        {"heading":"5. Статистика серии: ошибка validation", "paragraphs":[
            "BCE следует минимизировать. В таблице приведены минимум, среднее, медиана, выборочное стандартное отклонение (ddof=1) и максимум ошибок победителей запусков. Fitness двух штрафов нельзя непосредственно сравнивать, поэтому основное сравнение качества использует исходную BCE."],
         "table":{"headers":["λ","Лучшее","Среднее","Медиана","s","Худшее"],"rows":rows}},
        {"heading":"6. Затраты и устойчивость", "paragraphs":[
            f"λ=0: среднее число параметров {s0['parameters']['mean']:.2f}, distinct trainings {s0['distinct_trainings']['mean']:.2f}, время {s0['seconds']['mean']:.3f} с (s={s0['seconds']['std']:.3f}). λ=0.004: параметры {s1['parameters']['mean']:.2f}, distinct trainings {s1['distinct_trainings']['mean']:.2f}, время {s1['seconds']['mean']:.3f} с (s={s1['seconds']['std']:.3f}). Все решения синтаксически допустимы; доля допустимых = 1.",
            "Истории содержат incumbent после каждого обращения к fitness. Средняя траектория и диапазон min-max отражают разброс между запусками, а не доверительный интервал. Независимы seed поиска/инициализации весов; набор данных фиксирован. Поэтому выводы характеризуют устойчивость на одном наборе, а не статистическую генеральную совокупность задач."],"image":"convergence.png"},
        {"heading":"7. Три решения и компромиссы", "paragraphs":[
            f"Объединённый архив содержит {archive_count} недоминируемых моделей. Представители выбираются без test: минимальная сложность, медианная сложность архива, минимальная BCEvalidation. Таблица: признаки нумеруются с 1; h=0 означает линейную модель. Все три модели сохранены в model_1.npz–model_3.npz вместе со scaler и генотипом."],
         "table":{"headers":["№","Признаки","h","a","P","BCEval","BCEtest","Acc test"],"rows":rep_rows}},
        {"heading":"8. Предметный смысл фронта", "paragraphs":[
            f"Компактная модель использует {compact['parameters']} параметров; её ошибка validation {compact['validation_loss']:.4f} и test accuracy {compact['test_accuracy']:.2%}. Минимум validation достигается моделью с {best['parameters']} параметрами: BCEvalidation {best['validation_loss']:.4f}, test accuracy {best['test_accuracy']:.2%}. Она лучше описывает нелинейную границу, но требует больше памяти и операций.",
            "Компромиссное решение соответствует промежуточной сложности. Линейная модель не выражает квадратичную границу, тогда как скрытые нелинейные нейроны могут её аппроксимировать. Меньшая validation-ошибка не гарантирует лучшую test-ошибку каждого отдельного представителя: это ожидаемое ограничение конечной выборки."],"image":"quality_complexity.png"},
        {"heading":"9. Базовые модели", "paragraphs":[
            "Константный прогноз использует долю положительного класса в train. Логистическая регрессия на всех четырёх признаках обучается тем же собственным алгоритмом и числом шагов. Базовые модели предназначены для оценки качества, их стоимость обучения не приравнивается бюджету эволюционного поиска."],
         "table":{"headers":["Модель","P","BCE test","Accuracy test"],"rows":[[r['model'],str(r['parameters']),f(r['test_loss']),f(r['test_accuracy'])] for r in baselines]}},
        {"heading":"10. Выводы и ограничения", "paragraphs":[
            f"Увеличение штрафа с 0 до 0.004 изменило среднюю сложность с {s0['parameters']['mean']:.2f} до {s1['parameters']['mean']:.2f} параметров и среднюю BCEvalidation с {s0['validation_loss']['mean']:.5f} до {s1['validation_loss']['mean']:.5f}. Это наблюдаемый обмен качества на компактность, а не доказательство превосходства одной настройки для всех приложений.",
            f"Лучший по validation представитель имеет test accuracy {best['test_accuracy']:.2%}; линейный baseline - {baselines[1]['test_accuracy']:.2%}. Результат оценивается по отложенному качеству и цене структуры. Для учебной демонстрации синтеза архитектуры это содержательный результат; гарантии глобального оптимума и прикладного качества отсутствуют.",
            "Пространство из 135 архитектур достаточно мало для полного перебора: эволюция здесь изучается как метод, а не доказывается экономически необходимой. Внешний архив объединяет 40 запусков, поэтому его лучшие validation-значения подвержены множественному выбору. Один train/validation/test split и 100 тестовых объектов ограничивают точность обобщения. Улучшения: несколько независимых наборов данных, вложенная кросс-валидация, больше повторов обучения весов, бюджет в градиентных шагах и расширение пространства архитектур."]},
        {"heading":"11. Проверка и вопросы к защите", "paragraphs":[
            "test_lab3.py проверяет допустимость операторов, формулу числа параметров, воспроизводимость обучения и поиска, точный бюджет, train-only scaler и алгоритм недоминирования. Проверки запускаются командой python -m unittest -v test_lab3.py.",
            "Почему fitness не равен accuracy? BCE различает уверенность вероятностных прогнозов и пригодна для градиентного обучения; штраф отражает цену сложности. Почему test не входит в fitness? Иначе результат нельзя считать независимой оценкой обобщения. Почему кэш допустим? Повторный генотип имеет те же данные, инициализацию и число шагов в одном запуске; бюджет обращений и фактических обучений разделён. Где нет математической гарантии? Поиск не гарантирует глобальный минимум; качество на новых данных является экспериментальной оценкой."]},
        {"heading":"12. Источник задания", "paragraphs":[
            "Душкин Р. В. Дисциплина «Эволюционное программирование и генетические алгоритмы». Комплекс лабораторных работ, с. 12–17: ЛР3, трек B, вариант 10. Внешние наборы данных и готовые реализации оптимизаторов не использованы. NumPy используется для линейной алгебры, Matplotlib - только для графиков."]}
    ]
    json_dump(output / "report_content.json", {"title":"Лабораторная работа № 3", "task":"Эволюционный синтез архитектуры малого классификатора. Вариант 10, трек B.", "sections":sections})


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=int,default=20)
    parser.add_argument("--budget",type=int,default=240)
    parser.add_argument("--output",type=Path,default=Path(__file__).parent)
    args=parser.parse_args()
    if args.runs<1 or args.budget<POPULATION:
        parser.error("runs>=1 and budget>=24 required")
    try:
        run_experiment(args.output,args.runs,args.budget)
    except ValueError as error:
        parser.error(str(error))
