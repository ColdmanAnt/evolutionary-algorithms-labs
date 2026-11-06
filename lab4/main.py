"""Островной ГА: собственные операторы, синхронная миграция и честный бюджет."""
from __future__ import annotations

import argparse
import csv
import json
import time
from numbers import Real
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
METHODS = ("panmixia", "isolated", "ring", "fully_connected")
LABELS = {"panmixia": "Панмиксия", "isolated": "Без миграции", "ring": "Кольцо", "fully_connected": "Полная связь"}
FUNCTIONS = ("michalewicz", "rastrigin", "sphere")


def validate_config(config: dict) -> None:
    """Проверить план до RNG, файлов вывода и потенциального деления на ноль."""
    if not isinstance(config, dict):
        raise ValueError("Конфигурация должна быть объектом")
    integers = {"dimension": 1, "total_population": 30, "islands": 2,
                "evaluation_budget": 30, "tournament": 1,
                "migration_interval": 1, "bootstrap_seed": 0}
    for key, minimum in integers.items():
        value = config.get(key)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{key}: требуется целое число >= {minimum}")
    for key in ("blx_alpha", "crossover_probability", "mutation_probability", "sigma_start", "sigma_end"):
        value = config.get(key)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{key}: требуется конечное неотрицательное число")
    if any(config[key] > 1 for key in ("crossover_probability", "mutation_probability")):
        raise ValueError("Вероятности операторов должны принадлежать [0, 1]")
    total, islands, budget = (config[k] for k in ("total_population", "islands", "evaluation_budget"))
    if total % islands or total // islands < 30 or budget < total or budget % total:
        raise ValueError("Острова >=30, делимость бюджета и размера популяции обязательны")
    for key, allowed in (("functions", FUNCTIONS), ("methods", METHODS)):
        values = config.get(key)
        if not isinstance(values, list) or not values or any(not isinstance(v, str) or v not in allowed for v in values) or len(values) != len(set(values)):
            raise ValueError(f"{key}: нужен непустой список различных допустимых имён")
    # Таблицы сравнений используют панмиксию как явно заданный контроль.
    if "panmixia" not in config["methods"] or len(config["methods"]) < 2:
        raise ValueError("Нужны панмиксия и хотя бы одна островная схема для сравнения")
    seeds = config.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(isinstance(s, (bool, np.bool_)) or not isinstance(s, (int, np.integer)) or s < 0 for s in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("seeds: нужны различные неотрицательные целые числа")


def evaluate(x: np.ndarray, function: str) -> np.ndarray:
    """Одна строка = одна оценка; функция не меняет переданные данные."""
    if function == "michalewicz":
        i = np.arange(1, x.shape[-1] + 1)
        return -(np.sin(x) * np.sin(i * x**2 / np.pi)**20).sum(axis=-1)
    if function == "rastrigin":
        return (x*x - 10*np.cos(2*np.pi*x) + 10).sum(axis=-1)
    if function == "sphere":
        return (x*x).sum(axis=-1)
    raise ValueError(function)


def bounds(function: str) -> tuple[float, float]:
    if function not in FUNCTIONS:
        raise ValueError("Неизвестная функция")
    return (0.0, np.pi) if function == "michalewicz" else (-5.12, 5.12)


def reflect(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Отражение корректно обрабатывает выход за границы на несколько периодов."""
    x = np.asarray(x, dtype=float)
    if isinstance(lo, (bool, np.bool_)) or isinstance(hi, (bool, np.bool_)) or not isinstance(lo, Real) or not isinstance(hi, Real) or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo or not np.all(np.isfinite(x)):
        raise ValueError("Нужны конечные координаты и границы lo < hi")
    width = hi-lo
    return lo + width - np.abs((x-lo) % (2*width) - width)


def offspring(pop: np.ndarray, fit: np.ndarray, rng: np.random.Generator,
              lo: float, hi: float, sigma: float, config: dict) -> np.ndarray:
    n, d = pop.shape
    tournaments = rng.integers(0, n, (2, n, config["tournament"]))
    indices = np.take_along_axis(tournaments, np.argmin(fit[tournaments], axis=-1)[..., None], axis=-1)[..., 0]
    a, b = pop[indices[0]], pop[indices[1]]
    alpha = config["blx_alpha"]
    low, high = np.minimum(a,b), np.maximum(a,b)
    extension = alpha*(high-low)
    crossed = rng.uniform(low-extension, high+extension)
    children = np.where(rng.random((n,1)) < config["crossover_probability"], crossed, a).copy()
    mask = rng.random((n,d)) < config["mutation_probability"]
    children += mask*rng.normal(0,sigma,children.shape)
    return reflect(children,lo,hi)


def migrate(populations: list[np.ndarray], fitnesses: list[np.ndarray], method: str) -> int:
    """Снимки предотвращают обход кольца за один такт. Копирование не требует оценки."""
    if method not in METHODS:
        raise ValueError("Неизвестная схема миграции")
    if method not in ("ring", "fully_connected"):
        return 0
    count = len(populations)
    if count < 2 or len(fitnesses) != count:
        raise ValueError("Для миграции нужны минимум два согласованных острова")
    if any(not isinstance(p, np.ndarray) or not isinstance(f, np.ndarray) or
           not np.issubdtype(p.dtype, np.floating) or not np.issubdtype(f.dtype, np.floating) or
           p.ndim != 2 or len(p) == 0 or f.shape != (len(p),) or not p.flags.writeable or not f.flags.writeable or
           not np.all(np.isfinite(p)) or not np.all(np.isfinite(f)) for p, f in zip(populations, fitnesses)) or len({p.shape[1] for p in populations}) != 1:
        raise ValueError("Некорректные популяции или оценки островов")
    champions = [p[np.argmin(f)].copy() for p,f in zip(populations,fitnesses)]
    values = [float(np.min(f)) for f in fitnesses]
    for dest in range(count):
        source = (dest-1) % count if method == "ring" else min((i for i in range(count) if i != dest), key=lambda i: values[i])
        worst = int(np.argmax(fitnesses[dest]))
        # Иммигрант допускается только при неухудшении заменяемой особи.
        if values[source] <= fitnesses[dest][worst]:
            populations[dest][worst] = champions[source]
            fitnesses[dest][worst] = values[source]
    return count


def run(function: str, method: str, seed: int, config: dict) -> tuple[dict,list[dict]]:
    validate_config(config)
    if method not in METHODS or function not in FUNCTIONS:
        raise ValueError("Неизвестный метод или функция")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed должен быть неотрицательным целым числом")
    total = config["total_population"]
    budget = config["evaluation_budget"]
    islands = 1 if method == "panmixia" else config["islands"]
    if total % islands or budget % total or budget < total or total//islands < 30:
        raise ValueError("Популяции >=30, делимость бюджета и размера популяции обязательны")
    rng = np.random.default_rng(seed)
    lo,hi = bounds(function)
    start = time.perf_counter()
    # Все методы получают один и тот же набор 120 стартовых точек при данном seed.
    initial = rng.uniform(lo,hi,(total,config["dimension"]))
    populations = [p.copy() for p in np.split(initial,islands)]
    fitnesses = [evaluate(p,function) for p in populations]
    evaluations = total
    history = []
    migrants = 0
    generations = budget//total-1

    def record(generation: int) -> None:
        history.append({"function":function,"method":method,"seed":seed,
                        "generation":generation,"evaluations":evaluations,
                        "best":float(min(np.min(f) for f in fitnesses)),
                        "diversity":float(np.std(np.concatenate(populations),axis=0).mean()/(hi-lo))})

    record(0)
    for generation in range(1,generations+1):
        progress = (generation-1)/max(1,generations-1)
        sigma = (hi-lo)*(config["sigma_start"]*(1-progress)+config["sigma_end"]*progress)
        for k,(p,f) in enumerate(zip(populations,fitnesses)):
            best = int(np.argmin(f))
            elite, elite_f = p[best].copy(), float(f[best])
            children = offspring(p,f,rng,lo,hi,sigma,config)
            child_f = evaluate(children,function)
            evaluations += len(children)
            worst = int(np.argmax(child_f))
            if elite_f < child_f[worst]:
                children[worst],child_f[worst] = elite,elite_f
            populations[k],fitnesses[k] = children,child_f
        if generation % config["migration_interval"] == 0:
            migrants += migrate(populations,fitnesses,method)
        record(generation)
    all_pop,all_fit = np.concatenate(populations),np.concatenate(fitnesses)
    idx = int(np.argmin(all_fit))
    result = {"function":function,"method":method,"seed":seed,"best":float(all_fit[idx]),
              "evaluations":evaluations,"seconds":time.perf_counter()-start,
              "generations":generations,"migration_slots":migrants,
              "diversity":history[-1]["diversity"],"solution":all_pop[idx].tolist()}
    if evaluations != budget:
        raise RuntimeError("Нарушен внутренний счётчик оценок")
    return result,history


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def stats(values) -> dict:
    a=np.array(values)
    return {"best":float(a.min()),"mean":float(a.mean()),"median":float(np.median(a)),
            "std":float(a.std(ddof=1)) if len(a)>1 else 0.0,"worst":float(a.max())}


def experiment(config: dict, output: Path) -> None:
    validate_config(config)
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    results,trajectories=[],[]
    for function in config["functions"]:
        for method in config["methods"]:
            for seed in config["seeds"]:
                result,history=run(function,method,seed,config)
                results.append(result);trajectories.extend(history)
            print(function,method,"mean",np.mean([r["best"] for r in results if r["function"]==function and r["method"]==method]),flush=True)
    write_csv(output/"runs.csv",[{k:v for k,v in r.items() if k != "solution"} for r in results])
    write_csv(output/"history.csv",trajectories)
    summary=[]
    for function in config["functions"]:
        for method in config["methods"]:
            group=[r for r in results if r["function"]==function and r["method"]==method]
            summary.append({"function":function,"method":method,**stats([r["best"] for r in group]),
                            "mean_seconds":float(np.mean([r["seconds"] for r in group])),"runs":len(group),"evaluations":config["evaluation_budget"]})
    write_csv(output/"summary.csv",summary)
    pairs=[]
    bootstrap=np.random.default_rng(config["bootstrap_seed"])
    for function in config["functions"]:
        base=np.array([r["best"] for r in results if r["function"]==function and r["method"]=="panmixia"])
        for method in config["methods"]:
            if method=="panmixia":continue
            other=np.array([r["best"] for r in results if r["function"]==function and r["method"]==method])
            delta=other-base
            means=delta[bootstrap.integers(0,len(delta),(10000,len(delta)))].mean(axis=1)
            lo,hi=np.quantile(means,[.025,.975])
            pairs.append({"function":function,"method":method,"mean_delta_vs_panmixia":float(delta.mean()),
                          "ci95_low":float(lo),"ci95_high":float(hi),"win_fraction":float(np.mean(delta < -1e-12)),"tie_fraction":float(np.mean(np.abs(delta)<=1e-12))})
    write_csv(output/"paired_comparison.csv",pairs)
    best={fn:min([r for r in results if r["function"]==fn],key=lambda r:r["best"]) for fn in config["functions"]}
    (output/"best_solutions.json").write_text(json.dumps(best,ensure_ascii=False,indent=2),encoding="utf-8")
    plots(results,trajectories,output,config)
    make_report(summary,pairs,output,config)


def plots(results,trajectories,output,config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.prop_cycle":plt.cycler(color=["#34312d","#9b948a","#7d353b","#707448"]),"text.color":"#34312d","axes.labelcolor":"#34312d","xtick.color":"#34312d","ytick.color":"#34312d","grid.color":"#c8c0b3"})
    for function in config["functions"]:
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
        for method in config["methods"]:
            h=[r for r in trajectories if r["function"]==function and r["method"]==method]
            a=np.array([r["best"] for r in h]).reshape(len(config["seeds"]),-1)
            diversity=np.array([r["diversity"] for r in h]).reshape(a.shape)
            x=np.array([r["evaluations"] for r in h[:a.shape[1]]])
            axes[0].plot(x,a.mean(axis=0),label=LABELS[method])
            axes[0].fill_between(x,a.min(axis=0),a.max(axis=0),alpha=.08)
            axes[1].plot(x,diversity.mean(axis=0),label=LABELS[method])
        axes[0].set(xlabel="Число оценок",ylabel="Лучшее f, среднее и min-max",title=function)
        axes[1].set(xlabel="Число оценок",ylabel="Среднее σ координат / ширина области",title="Разнообразие популяции")
        for ax in axes:ax.grid(alpha=.2)
        axes[1].legend(fontsize=8)
        fig.savefig(output/f"convergence_{function}.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,len(config["functions"]),figsize=(max(5, 3.7*len(config["functions"])),4),layout="constrained",squeeze=False)
    axes=axes[0]
    for ax,function in zip(axes,config["functions"]):
        values=[[r["best"] for r in results if r["function"]==function and r["method"]==method] for method in config["methods"]]
        ax.boxplot(values,tick_labels=[LABELS[m] for m in config["methods"]])
        ax.tick_params(axis="x",labelrotation=30);ax.set(title=function,ylabel="Итоговое f, меньше лучше");ax.grid(axis="y",alpha=.2)
    fig.savefig(output/"distributions.png",dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));ax.axis("off")
    labels=[f'Конфигурация + seed → {config["total_population"]} начальных точек', f'Панмиксия или {config["islands"]} острова × {config["total_population"]//config["islands"]}', "Турнир → BLX-α → мутация → отражение границ", f'Оценка {config["total_population"]} потомков; сохранение элиты', f'Каждые {config["migration_interval"]} поколений: снимок → миграция', "Логирование качества и разнообразия", f'{config["evaluation_budget"]} оценок → CSV, статистика и графики']
    for i,label in enumerate(labels):
        y=.94-i*.14
        ax.text(.5,y,label,ha="center",va="center",bbox=dict(boxstyle="round,pad=.5",fc="#f3eee5",ec="#675d50"),fontsize=11)
        if i<len(labels)-1:ax.annotate("",xy=(.5,y-.095),xytext=(.5,y-.045),arrowprops=dict(arrowstyle="->",color="#675d50"))
    fig.savefig(output/"algorithm_flow.png",dpi=150,bbox_inches="tight");plt.close(fig)


def make_report(summary,pairs,output,config):
    rows=[[s["function"],LABELS[s["method"]],*[f'{s[k]:.6g}' for k in ["best","mean","median","std","worst"]]] for s in summary]
    conclusions=[]
    for function in config["functions"]:
        subset=[s for s in summary if s["function"]==function]
        winner=min(subset,key=lambda s:s["mean"])
        conclusions.append(f'{function}: наименьшее среднее f={winner["mean"]:.6g} получено схемой «{LABELS[winner["method"]]}». Это вывод о данной серии и параметрах; универсального преимущества топологии он не устанавливает.')
    report={"title":"Лабораторная работа № 4. Островная модель генетического алгоритма", "task":"Вариант 10. Несколько популяций, синхронная миграция, сравнение топологий.","sections":[
      {"heading":"1. Постановка и заявка","paragraphs":["Исследуется минимизация трёх аналитических функций в размерности d=9. Пространство решений - ограниченный вещественный гиперкуб; генотип совпадает с фенотипом. Единственные жёсткие ограничения - координатные границы. Сфера: f=Σxᵢ². Растригин: f=Σ[xᵢ²−10cos(2πxᵢ)+10]. Михалевич: f=−Σsin(xᵢ)sin²⁰(i xᵢ²/π). Для Михалевича область [0,π]⁹, для остальных [-5.12,5.12]⁹.","Для сферы и Растригина известный минимум равен 0. Для Михалевича d=9 используется численный ориентир из ЛР1, а не утверждение о доказанной оптимальности. Сфера проверяет поведение на простой задаче; две мультимодальные функции проверяют устойчивость к локальным минимумам.","Файл proposal.txt содержит заявку, подготовленную до реализации проектной части. Согласование и приёмка преподавателем не подтверждены; они остаются процедурой учебного курса."]},
      {"heading":"2. Архитектура и операторы","paragraphs":["Сравниваются панмиксия (1×120), четыре изолированных острова, четыре острова в кольце и четыре острова с полносвязным обменом. Все методы используют одинаковые 120 начальных точек при одном seed. В панмиксии элита сохраняется одна на общую популяцию; в островной модели одна на каждый остров. Это часть сравниваемой архитектуры.","Турнир размера 3, кроссовер BLX-α с α=0.5 и вероятностью 0.9. Каждая координата мутирует с вероятностью 1/9. Нормальный шум имеет стандартное отклонение от 0.08 до 0.005 ширины области, линейно уменьшающееся по заранее заданному расписанию. Это расписание, а не самоадаптация по качеству. Выход за границы исправляется отражением, в том числе при многократном превышении ширины.","Каждый остров порождает 30 новых оцениваемых потомков. Лучший родитель при необходимости заменяет худшего потомка без повторной оценки. Каждые 10 поколений кольцо передаёт лучшего предшественника, полная связь выбирает лучшего из остальных островов. На остров выделяется одинаково один слот миграции. Иммигрант заменяет худшего при неухудшении его качества; источник берётся из снимка ДО обмена. Поэтому порядок обработки островов не создаёт каскадной миграции.","Без миграции схема служит контрольным сравнением: она отделяет эффект разделения популяции от эффекта обмена. Топология полной связи реализована как обмен чемпионами с выбором лучшего доступного отправителя; выводы относятся именно к этому правилу, а не ко всем схемам all-to-all."],"image":"algorithm_flow.png"},
      {"heading":"3. Экспериментальная методика","paragraphs":[f'Для каждой из 12 комбинаций выполнено {len(config["seeds"])} независимых запусков, seeds={config["seeds"][0]}…{config["seeds"][-1]}. Бюджет каждого запуска - ровно {config["evaluation_budget"]} оценок: 120 при инициализации и 249×120 оценок потомков. Копирование уже оценённых особей не расходует оценок. Все итоговые решения допустимы по построению.',"Запуски с одинаковым seed образуют пары для сравнения. Внутри каждой конфигурации seeds независимы; общие стартовые точки между конфигурациями уменьшают шум сравнения. Время измерено perf_counter и зависит от оборудования. Исходные строки запусков и поколений сохранены в runs.csv и history.csv.","Приводятся минимум, среднее, медиана, выборочное стандартное отклонение (ddof=1), максимум. Парные разности равны f(метод)−f(панмиксия), отрицательная разность означает улучшение. 95% percentile bootstrap-интервал среднего построен по 10000 перевыборкам пар с bootstrap_seed=88010. Это описательный интервал без коррекции множественных сравнений, не доказательство превосходства на любых задачах."],"table":{"headers":["Функция","Схема","Лучшее","Среднее","Медиана","s","Худшее"],"rows":rows}},
      {"heading":"4. Сходимость и устойчивость","paragraphs":["Линии показывают среднее лучшего значения, полупрозрачные области - минимум и максимум по 30 запускам. Разнообразие - среднее стандартное отклонение координат, нормированное на ширину области; оно вычисляется на общей популяции всех островов.",*conclusions],"image":"convergence_michalewicz.png"},
      {"heading":"Растригин","paragraphs":["Нулевой минимум позволяет напрямую интерпретировать остаточную ошибку. Структура островов меняет скорость обмена удачными координатами; сниженное разнообразие само по себе не доказывает высокое качество."],"image":"convergence_rastrigin.png"},
      {"heading":"Сфера","paragraphs":["На выпуклой сфере сохранение нескольких популяций может стоить части скорости сходимости: здесь нет необходимости преодолевать локальные минимумы. Это проверяется экспериментально, а не принимается как универсальное правило."],"image":"convergence_sphere.png"},
      {"heading":"5. Распределения результатов","paragraphs":["Boxplot отражает медиану, межквартильный размах, усы и выбросы. Он дополняет среднюю траекторию и позволяет увидеть нестабильность отдельных запусков."],"image":"distributions.png"},
      {"heading":"6. Парные сравнения","paragraphs":["Если интервал разности включает ноль, данная серия не даёт ясного направленного вывода по среднему. Даже при интервале с одной стороны от нуля результат ограничен выбранными функциями, seeds и бюджетом."],"table":{"headers":["Функция","Схема","Δ среднего","95% low","95% high","Доля побед"],"rows":[[p["function"],LABELS[p["method"]],f'{p["mean_delta_vs_panmixia"]:.5g}',f'{p["ci95_low"]:.5g}',f'{p["ci95_high"]:.5g}',f'{p["win_fraction"]:.1%}'] for p in pairs]}},
      {"heading":"7. Корректность, выводы и ограничения","paragraphs":["Автотесты проверяют формулы, сохранение допустимости отражением, воспроизводимость, ровный бюджет, неухудшение рекорда и использование синхронных снимков при миграции. Для выбранных best_solutions.json повторно проверяется соответствие вектора целевому значению.","Результат хорош как сравнительное исследование: конфигурации воспроизводимы, бюджет равен, видны распределения и механизмы обмена. Достижение глобального оптимума стохастическим алгоритмом не гарантируется. Эффект миграции зависит от задачи; наиболее сильная схема выбирается отдельно по каждой функции.","Ограничения: три функции и одна размерность, единственные интервал и объём миграции, линейное расписание масштаба мутации. Не оценивались распределённое исполнение и сетевые задержки: острова моделируются последовательно на одном CPU. Расход миграции по времени есть в измерениях, но оценок новых решений она не требует. Следующий этап - варьировать частоту и число мигрантов, размеры островов, независимые benchmark-наборы и бюджет.","Сценарий защиты на 5-7 минут приведён в README.md. Он включает архитектуру, доказуемые свойства реализации, показ статистики, разбор компромисса между обменом и разнообразием, ограничения."],"table":{"headers":["Функция","Среднее время по схемам, с: min–max","Оценок/запуск"],"rows":[[fn, f'{min(s["mean_seconds"] for s in summary if s["function"]==fn):.3f}–{max(s["mean_seconds"] for s in summary if s["function"]==fn):.3f}', config["evaluation_budget"]] for fn in config["functions"]]}}
    ]}
    # Пробные конфигурации тоже должны описывать фактически выполненную серию.
    # Архивный config.json остаётся неизменным; значения не зашиты в отчёт.
    dimension, total, islands = (config[k] for k in ("dimension", "total_population", "islands"))
    budget, repetitions = config["evaluation_budget"], len(config["seeds"])
    sections = report["sections"]
    sections[0]["paragraphs"][0] = (
        f'Минимизируются функции {", ".join(config["functions"])} в размерности d={dimension}. '
        'Генотип совпадает с фенотипом: вещественный вектор в ограниченном гиперкубе. '
        'Жёсткие ограничения - координатные границы. Сфера: f=Σxᵢ²; Растригин: f=Σ[xᵢ²−10cos(2πxᵢ)+10]; '
        f'Михалевич: f=−Σsin(xᵢ)sin²⁰(i xᵢ²/π). Область для Михалевича [0,π]^{dimension}, для остальных [-5.12,5.12]^{dimension}.'
    )
    if dimension != 9:
        sections[0]["paragraphs"][1] = (
            'Для сферы и Растригина известный минимум равен 0. '
            f'Для Михалевича в размерности {dimension} внешний численный ориентир здесь не рассчитывался; '
            'ориентир ЛР1 в другой размерности переносить нельзя. Глобальная оптимальность не гарантируется.'
        )
    sections[1]["paragraphs"][0] = (
        f'Сравниваются схемы: {", ".join(LABELS[m] for m in config["methods"])}. '
        f'Панмиксия использует 1×{total} особей, островные схемы - {islands}×{total//islands}. '
        f'При одном seed все методы получают одинаковые {total} начальных точек. '
        'Панмиксия сохраняет одну элиту на общую популяцию, островная модель - одну на каждый остров.'
    )
    sections[1]["paragraphs"][1] = (
        f'Турнир размера {config["tournament"]}, BLX-α с α={config["blx_alpha"]} и вероятностью {config["crossover_probability"]}. '
        f'Вероятность мутации координаты {config["mutation_probability"]:.6g}. Стандартное отклонение гауссовского шума '
        f'линейно меняется от {config["sigma_start"]} до {config["sigma_end"]} ширины области. '
        'Расписание задано заранее; это не самоадаптация по качеству. Границы восстанавливаются отражением.'
    )
    sections[1]["paragraphs"][2] = (
        f'Каждый остров порождает {total//islands} новых потомков. Лучший родитель при необходимости заменяет худшего '
        f'потомка без повторной оценки. Каждые {config["migration_interval"]} поколений кольцо передаёт чемпиона '
        'предшественника, полная связь выбирает лучшего чемпиона остальных островов. Один слот иммиграции '
        'на остров; замена только при неухудшении худшего. Все источники берутся из снимка до обмена.'
    )
    sections[2]["paragraphs"][0] = (
        f'Комбинаций функция×схема: {len(config["functions"])*len(config["methods"])}. '
        f'Независимых запусков каждой комбинации: {repetitions}; seeds={config["seeds"]}. '
        f'Ровно {budget} оценок на запуск: {total} начальных и {budget//total-1}×{total} новых. '
        'Копирование оценённых особей не требует оценки. Все результаты допустимы по построению. '
        + ('Это короткий демонстрационный прогон; для исследовательских выводов требуется не менее 30 seeds.' if repetitions < 30 else '')
    )
    sections[2]["paragraphs"][2] = (
        'Приводятся минимум, среднее, медиана, выборочное стандартное отклонение (ddof=1) и максимум. '
        'Парные разности равны f(метод)−f(панмиксия); отрицательная разность означает улучшение. '
        f'95% percentile bootstrap-интервал среднего строится по 10000 перевыборкам пар, bootstrap_seed={config["bootstrap_seed"]}. '
        'Это описательный интервал без коррекции множественных сравнений, не доказательство универсального превосходства.'
    )
    sections[3]["paragraphs"][0] = (
        f'Линия - среднее лучшего значения; полоса - минимум и максимум по {repetitions} запускам. '
        'Разнообразие - среднее стандартное отклонение координат общей популяции, нормированное на ширину области.'
    )
    sections[3]["image"] = f'convergence_{config["functions"][0]}.png'
    explanations = {
        "sphere": ("Сфера", "Выпуклая функция проверяет поведение без локальных минимумов. Разнообразие оценивается вместе с итоговой ошибкой."),
        "rastrigin": ("Растригин", "Нулевой минимум позволяет интерпретировать остаточную ошибку. Острова меняют скорость распространения полезных координат."),
        "michalewicz": ("Михалевич", "Узкие впадины и множество локальных минимумов усложняют поиск. Малое разнообразие само по себе не означает хорошего качества.")}
    convergence_sections = [sections[3]]
    for fn in config["functions"][1:]:
        heading, explanation = explanations[fn]
        convergence_sections.append(dict(heading=heading, paragraphs=[explanation], image=f'convergence_{fn}.png'))
    sections = sections[:3] + convergence_sections + sections[6:]
    sections[-1]["paragraphs"][2] = (
        f'Ограничения: {len(config["functions"])} функции и одна размерность d={dimension}; единственные интервал '
        'и объём миграции, линейное расписание масштаба мутации. Острова выполняются последовательно на одном CPU; '
        'сетевые задержки и параллельное ускорение не исследовались. Время включает обмен, но копирование оценённых '
        'мигрантов не требует новых оценок. Следующий этап - варьировать частоту и число мигрантов, размеры островов, '
        'функции и бюджет.'
    )
    for section in sections:
        if section["heading"] == "5. Распределения результатов":
            section["paragraphs"][0] = f'Показаны результаты {repetitions} запусков каждой конфигурации: медиана, квартильный размах, усы и выбросы.'
    report["sections"] = sections
    (output/"report_content.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=HERE/"config.json")
    parser.add_argument("--output",type=Path,default=HERE)
    parser.add_argument("--seed",type=int,help="Один seed для отладки вместо полной серии")
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding="utf-8"))
    if args.seed is not None:config["seeds"]=[args.seed]
    experiment(config,args.output)


if __name__=="__main__":main()
