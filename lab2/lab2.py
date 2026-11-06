#!/usr/bin/env python3
"""ЛР2: собственный ГА составления недельного графика дежурств."""
from __future__ import annotations
import argparse, copy, csv, json, math, numbers, statistics, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
STAFF = 7
SLOTS = 21
POP = 60
PENALTY = 10000


def nonnegative_integer(value, name, minimum=0):
    """Не допускаем bool, дроби и неявное округление параметров."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f'{name}: требуется целое число не меньше {minimum}')
    return int(value)


def schedule_vector(value):
    array = np.asarray(value)
    if (array.shape != (SLOTS,) or array.dtype.kind not in 'iu'
            or np.any(array < 0) or np.any(array >= STAFF)):
        raise ValueError('Нужно ровно 21 целочисленное назначение с индексами сотрудников 0..6')
    return array.astype(int, copy=False)


def generate_data(seed: int = 10010) -> dict:
    seed = nonnegative_integer(seed, 'seed')
    rng = np.random.default_rng(seed)
    witness = [(day + 2 * shift) % STAFF for day in range(7) for shift in range(3)]
    availability = (rng.random((SLOTS, STAFF)) < .80).astype(int)
    for event, employee in enumerate(witness):
        availability[event, employee] = 1
    costs = rng.integers(0, 10, (SLOTS, STAFF)).tolist()
    conflicts = [[3 * d + a, 3 * d + b] for d in range(7) for a in range(3) for b in range(a + 1, 3)]
    conflicts += [[3 * d + 2, 3 * (d + 1)] for d in range(6)]
    return dict(seed=seed, staff=[f'Сотрудник {i+1}' for i in range(STAFF)],
                events=[dict(id=i, day=i//3+1, shift=['Утро','День','Ночь'][i%3]) for i in range(SLOTS)],
                availability=availability.tolist(), preference_costs=costs,
                conflicts=conflicts, max_shifts=4, max_nights=2,
                witness=witness, assumptions='Одна позиция на смену; ночь и следующее утро несовместимы. Горизонт конечный: после воскресенья обязательств нет.')


class Problem:
    def __init__(self, data: dict):
        if not isinstance(data, dict):
            raise ValueError('Данные задачи должны быть словарём')
        required = {'staff', 'events', 'availability', 'preference_costs', 'conflicts',
                    'max_shifts', 'max_nights', 'witness'}
        if not required <= data.keys():
            raise ValueError(f'Отсутствуют поля данных: {sorted(required - data.keys())}')
        self.data = copy.deepcopy(data)
        if (not isinstance(data['staff'], list) or len(data['staff']) != STAFF
                or any(not isinstance(name, str) or not name.strip() for name in data['staff'])
                or len(set(data['staff'])) != STAFF):
            raise ValueError('Нужно 7 различных непустых имён сотрудников')
        expected_events = [dict(id=i, day=i//3+1, shift=['Утро','День','Ночь'][i%3]) for i in range(SLOTS)]
        if data['events'] != expected_events:
            raise ValueError('Нужно 21 событие в порядке дней и смен Утро/День/Ночь')
        availability = np.asarray(data['availability'])
        costs = np.asarray(data['preference_costs'])
        if (availability.shape != (SLOTS, STAFF) or availability.dtype.kind not in 'biu'
                or np.any((availability != 0) & (availability != 1))):
            raise ValueError('Доступность должна быть матрицей 21×7 из 0 и 1')
        if (costs.shape != (SLOTS, STAFF) or costs.dtype.kind not in 'iu'
                or np.any(costs < 0) or np.any(costs > 9)):
            raise ValueError('Предпочтения должны быть целочисленной матрицей 21×7 со значениями 0..9')
        max_shifts = nonnegative_integer(data['max_shifts'], 'max_shifts', 1)
        max_nights = nonnegative_integer(data['max_nights'], 'max_nights', 1)
        if max_shifts > SLOTS or max_nights > SLOTS // 3:
            raise ValueError('Лимиты смен превышают длину недельного горизонта')
        conflicts = np.asarray(data['conflicts'])
        if conflicts.size == 0:
            conflicts = np.empty((0, 2), dtype=int)
        if (conflicts.ndim != 2 or conflicts.shape[1] != 2 or conflicts.dtype.kind not in 'iu'
                or np.any(conflicts < 0) or np.any(conflicts >= SLOTS)
                or np.any(conflicts[:, 0] == conflicts[:, 1])):
            raise ValueError('Конфликты должны быть парами различных индексов событий 0..20')
        if len({tuple(sorted(map(int, pair))) for pair in conflicts}) != len(conflicts):
            raise ValueError('Конфликтные пары не должны повторяться')
        self.available = availability.astype(bool, copy=True)
        self.costs = costs.astype(int, copy=True)
        self.conflicts = conflicts.astype(int, copy=True)
        self.neighbours = [[] for _ in range(SLOTS)]
        for a, b in self.conflicts:
            self.neighbours[a].append(int(b)); self.neighbours[b].append(int(a))
        self.witness = schedule_vector(data['witness']).copy()
        if self.evaluate(self.witness)['violations'] != 0:
            raise ValueError('Исходный резервный план должен удовлетворять всем ограничениям')

    def evaluate(self, x) -> dict:
        x = schedule_vector(x)
        loads = np.bincount(x, minlength=STAFF)
        nights = np.bincount(x[2::3], minlength=STAFF)
        unavailable = int(np.count_nonzero(~self.available[np.arange(SLOTS), x]))
        overcap = int(np.maximum(loads - self.data['max_shifts'], 0).sum())
        night_overcap = int(np.maximum(nights - self.data['max_nights'], 0).sum())
        conflicts = int(np.count_nonzero(x[self.conflicts[:, 0]] == x[self.conflicts[:, 1]]))
        violations = unavailable + overcap + night_overcap + conflicts
        preference = int(self.costs[np.arange(SLOTS), x].sum())
        fairness = int(3 * np.square(loads - 3).sum() + 2 * np.square(nights - 1).sum())
        soft = preference + fairness
        return dict(soft=soft, preference=preference, fairness=fairness, violations=violations,
                    unavailable=unavailable, overcap=overcap, night_overcap=night_overcap,
                    conflicts=conflicts, score=soft + PENALTY * violations,
                    loads=loads.tolist(), nights=nights.tolist())

    def repair(self, original, rng) -> tuple[np.ndarray, bool]:
        """Жадное восстановление с сохранением исходных назначений, 12 попыток."""
        original = schedule_vector(original)
        for _ in range(12):
            x = np.full(SLOTS, -1, dtype=int)
            loads = np.zeros(STAFF, dtype=int); nights = np.zeros(STAFF, dtype=int)
            order = rng.permutation(SLOTS)
            for event in order:
                candidates = [w for w in range(STAFF)
                              if self.available[event,w] and loads[w] < self.data['max_shifts']
                              and (event % 3 != 2 or nights[w] < self.data['max_nights'])
                              and all(x[j] != w for j in self.neighbours[event])]
                if not candidates:
                    break
                preferred = int(original[event])
                employee = preferred if preferred in candidates else int(rng.choice(candidates))
                x[event] = employee; loads[employee] += 1
                if event % 3 == 2:
                    nights[employee] += 1
            else:
                return x, False
        # Гарантированный допустимый план приложен к синтетическим данным.
        return self.witness.copy(), True


def variation(a, b, rng, operator: str):
    if operator not in ('swap', 'replace'):
        raise ValueError('Оператор должен быть swap или replace')
    a, b = schedule_vector(a), schedule_vector(b)
    child = a.copy()
    if rng.random() < .9:
        mask = rng.random(SLOTS) < .5
        child[mask] = b[mask]
    if rng.random() < .8:
        i, j = rng.choice(SLOTS, size=2, replace=False)
        if operator == 'swap':
            child[i], child[j] = child[j], child[i]
        elif operator == 'replace':
            child[[i,j]] = rng.integers(0, STAFF, 2)
        else:
            raise ValueError(operator)
    return child


def run(problem: Problem, method: str, operator: str, seed: int, budget: int, pop_size: int = POP):
    if method not in ('penalty', 'repair', 'random'):
        raise ValueError('Метод должен быть penalty, repair или random')
    allowed = ('replace', 'swap', 'none') if method == 'random' else ('replace', 'swap')
    if operator not in allowed:
        raise ValueError('Неизвестный оператор; none допустим только для случайного baseline')
    seed = nonnegative_integer(seed, 'seed')
    budget = nonnegative_integer(budget, 'budget', 1)
    pop_size = nonnegative_integer(pop_size, 'pop_size', 2)
    if budget < pop_size:
        raise ValueError('Бюджет должен быть не меньше популяции')
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    evaluations = 0; fallback_count = 0; feasible_evaluations = 0
    best_x = None; best_soft = math.inf
    history = []
    def assess(x):
        nonlocal evaluations, feasible_evaluations, best_x, best_soft
        ev = problem.evaluate(x); evaluations += 1
        if ev['violations'] == 0:
            feasible_evaluations += 1
            if ev['soft'] < best_soft:
                best_soft = ev['soft']; best_x = x.copy()
        return ev
    def make(x):
        nonlocal fallback_count
        if method in ('repair', 'random'):
            x, fallback = problem.repair(x, rng); fallback_count += fallback
        return x
    # Одинаковое допустимое исходное решение у всех вариантов, включая baseline.
    witness = problem.witness.copy()
    witness_ev = assess(witness)
    history.append(dict(evaluations=evaluations, best=best_soft, feasible_population=1.0))
    if method == 'random':
        while evaluations < budget:
            x = make(rng.integers(0,STAFF,SLOTS)); assess(x)
            if evaluations % pop_size == 0 or evaluations == budget:
                history.append(dict(evaluations=evaluations,best=best_soft,feasible_population=1.0))
        final_feasible = 1.
    else:
        pop = [witness]
        vals = [witness_ev]
        for _ in range(pop_size - 1):
            x = make(rng.integers(0,STAFF,SLOTS)); pop.append(x); vals.append(assess(x))
        pop = np.asarray(pop)
        history.append(dict(evaluations=evaluations,best=best_soft,
                            feasible_population=sum(v['violations']==0 for v in vals)/len(vals)))
        while evaluations < budget:
            children=[]; childvals=[]
            for _ in range(min(pop_size, budget-evaluations)):
                parents=[]
                for __ in range(2):
                    candidates=rng.integers(0,len(pop),3)
                    winner=min(candidates,key=lambda i:vals[i]['score'])
                    parents.append(pop[winner])
                child = make(variation(*parents,rng,operator))
                children.append(child); childvals.append(assess(child))
            elite=min(range(len(vals)),key=lambda i:vals[i]['score'])
            # Одно старое элитное решение, остальные — лучшие новые потомки.
            choices=sorted(range(len(children)),key=lambda i:childvals[i]['score'])[:pop_size-1]
            if len(children)<pop_size-1:
                # Последний неполный шаг; слияние нужно только для отчёта популяции.
                merged=list(pop)+children; merged_vals=vals+childvals
                chosen=sorted(range(len(merged)),key=lambda i:merged_vals[i]['score'])[:pop_size]
                pop=np.asarray([merged[i] for i in chosen]); vals=[merged_vals[i] for i in chosen]
            else:
                pop=np.asarray([pop[elite]]+[children[i] for i in choices])
                vals=[vals[elite]]+[childvals[i] for i in choices]
            history.append(dict(evaluations=evaluations,best=best_soft,
                                feasible_population=sum(v['violations']==0 for v in vals)/len(vals)))
        final_feasible=sum(v['violations']==0 for v in vals)/len(vals)
    assert evaluations == budget and best_x is not None
    return dict(seed=seed,method=method,operator=operator,best=int(best_soft),
                evaluations=evaluations,seconds=time.perf_counter()-started,
                feasible_found=True, feasible_evaluation_fraction=feasible_evaluations/evaluations,
                final_population_feasible_fraction=final_feasible,
                repair_fallbacks=fallback_count,solution=best_x.tolist(),history=history)


def write_csv(path, rows):
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def experiment(out: Path, runs: int, budget: int, seed_start: int):
    runs = nonnegative_integer(runs, 'runs', 1)
    budget = nonnegative_integer(budget, 'budget', POP)
    seed_start = nonnegative_integer(seed_start, 'seed_start')
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    data=generate_data(); (out/'data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    problem=Problem(data)
    configs=[('penalty','replace'),('penalty','swap'),('repair','replace'),('repair','swap'),('random','none')]
    results=[]; histories=[]; best=None
    for method,operator in configs:
        for seed in range(seed_start,seed_start+runs):
            result=run(problem,method,operator,seed,budget)
            label=f'{method}_{operator}'
            if best is None or result['best']<best['best']:
                best=result | dict(config=label)
            histories.extend(dict(config=label,seed=seed,**h) for h in result.pop('history'))
            results.append(result | dict(config=label))
        print(f'{method}/{operator}: {runs} запусков завершены',flush=True)
    (out/'solutions.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    write_csv(out/'perrun.csv',[{k:v for k,v in r.items() if k!='solution'} for r in results])
    write_csv(out/'histories.csv',histories)
    summaries=[]
    for method,operator in configs:
        label=f'{method}_{operator}'; rows=[r for r in results if r['config']==label]
        values=[r['best'] for r in rows]
        summaries.append(dict(config=label,runs=len(rows),best=min(values),mean=statistics.mean(values),
                              median=statistics.median(values),std=statistics.stdev(values) if len(values)>1 else 0.,
                              worst=max(values),feasible_rate=statistics.mean(r['feasible_found'] for r in rows),
                              mean_feasible_evaluations=statistics.mean(r['feasible_evaluation_fraction'] for r in rows),
                              mean_final_feasible_population=statistics.mean(r['final_population_feasible_fraction'] for r in rows),
                              mean_seconds=statistics.mean(r['seconds'] for r in rows),budget=budget,
                              fallback_total=sum(r['repair_fallbacks'] for r in rows)))
    write_csv(out/'summary.csv',summaries)
    (out/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2))
    config=dict(variant=10,N=28,Q=1,Y=26,group=517,student='Холодков Антон Евгеньевич',
                runs=runs,seeds=list(range(seed_start,seed_start+runs)),budget=budget,population=POP,
                selection='tournament, k=3',crossover='uniform,p=0.9',mutation_probability=.8,
                mutation_loci=2,elitism=1,penalty=PENALTY,data_seed=data['seed'],configs=configs,
                stop='Точно budget вызовов Problem.evaluate в поиске; ремонт использует инкрементальные проверки ограничений, не фитнес.')
    (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    best['evaluation']=problem.evaluate(best['solution'])
    (out/'best_solution.json').write_text(json.dumps(best,ensure_ascii=False,indent=2))
    schedule=[dict(day=e['day'],shift=e['shift'],employee=data['staff'][best['solution'][e['id']]],
                   preference_cost=int(problem.costs[e['id'],best['solution'][e['id']]])) for e in data['events']]
    write_csv(out/'best_schedule.csv',schedule)
    witness=problem.witness
    rng=np.random.default_rng(888)
    second,_=problem.repair(rng.integers(0,STAFF,SLOTS),rng)
    bad1=witness.copy(); bad1[1]=bad1[0]
    bad2=np.zeros(SLOTS,dtype=int)
    examples=[dict(name=name,solution=x.tolist(),evaluation=problem.evaluate(x))
              for name,x in [('Допустимый исходный',witness),('Допустимый после ремонта',second),
                             ('Недопустимый: две смены в день',bad1),('Недопустимый: один сотрудник на всё',bad2)]]
    (out/'examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2))
    plots(out,results,histories,best,problem)
    report(out,summaries,best,problem,examples,config)


def plots(out,results,histories,best,problem):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from cycler import cycler
    from matplotlib.colors import ListedColormap
    palette = ['#34312d', '#7d353b', '#707448', '#ad6b42', '#b39a68']
    plt.rcParams.update({
        'font.family':'DejaVu Sans', 'font.size':10,
        'axes.spines.top':False, 'axes.spines.right':False,
        'axes.prop_cycle':cycler(color=palette),
        'figure.facecolor':'#ffffff', 'axes.facecolor':'#fffdf9',
        'text.color':'#34312d', 'axes.labelcolor':'#34312d',
        'axes.edgecolor':'#675d50', 'xtick.color':'#34312d', 'ytick.color':'#34312d',
        'grid.color':'#9b948a', 'legend.facecolor':'#fffdf9', 'legend.edgecolor':'#9b948a',
    })
    labels={'penalty_replace':'Штраф + замена','penalty_swap':'Штраф + обмен',
            'repair_replace':'Ремонт + замена','repair_swap':'Ремонт + обмен','random_none':'Случайный допустимый'}
    fig,ax=plt.subplots(figsize=(9,4.8))
    for label in labels:
        hs=[h for h in histories if h['config']==label]
        seeds=sorted({h['seed'] for h in hs})
        checkpoints=sorted({h['evaluations'] for h in hs})
        trajectories=np.array([[next(h['best'] for h in hs if h['seed']==s and h['evaluations']==c) for c in checkpoints] for s in seeds])
        mean=trajectories.mean(0)
        line=ax.plot(checkpoints,mean,label=labels[label])[0]
        ax.fill_between(checkpoints,trajectories.min(0),trajectories.max(0),color=line.get_color(),alpha=.10)
    ax.set(xlabel='Число вычислений фитнес-функции',ylabel='Лучший допустимый мягкий штраф',title=f'Сходимость: среднее и диапазон min–max по {len(seeds)} запускам')
    ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'convergence.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5))
    groups=[[r['best'] for r in results if r['config']==c] for c in labels]
    boxes = ax.boxplot(groups,tick_labels=list(labels.values()),showmeans=True,patch_artist=True,
                       medianprops=dict(color='#7d353b',linewidth=1.4),
                       meanprops=dict(marker='D',markerfacecolor='#34312d',markeredgecolor='#34312d',markersize=4),
                       whiskerprops=dict(color='#675d50'),capprops=dict(color='#675d50'),
                       flierprops=dict(marker='o',markerfacecolor='#fffdf9',markeredgecolor='#675d50',markersize=4))
    for box, color in zip(boxes['boxes'], palette):
        box.set(facecolor=color,edgecolor=color,alpha=.24,linewidth=1.3)
    ax.tick_params(axis='x',labelrotation=15,labelsize=8)
    ax.set(ylabel='Мягкий штраф (меньше — лучше)',title='Распределение итогового качества');ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/'quality.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5))
    x=np.asarray(best['solution']).reshape(7,3).T
    schedule_colors = ListedColormap(['#34312d','#7d353b','#707448','#ad6b42','#b39a68','#9b948a','#8f6f58'])
    im=ax.imshow(x,cmap=schedule_colors,vmin=-.5,vmax=6.5,aspect='auto')
    for i in range(3):
        for j in range(7):
            ax.text(j,i,f'Сотр. {x[i,j]+1}',ha='center',va='center',color='#fffdf9',fontsize=11,
                    bbox=dict(facecolor='#34312d',alpha=.50,edgecolor='none',pad=2))
    ax.set_xticks(range(7),['Пн','Вт','Ср','Чт','Пт','Сб','Вс']);ax.set_yticks(range(3),['Утро','День','Ночь'])
    ax.set_title(f"Лучшее расписание: мягкий штраф {best['best']}, нарушений 0")
    fig.tight_layout();fig.savefig(out/'schedule.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5.4));ax.axis('off')
    boxes=[('Данные + seed\nДоступность, 27 конфликтов, лимиты',(.50,.91)),
           ('60 векторов: 59 случайных\n+ один допустимый план',(.50,.73)),
           ('Турнир k=3 → равномерный кроссовер\nЗамена двух генов / обмен',(.50,.55)),
           ('Штраф 10000·H + S\nили ремонт → оценка S',(.50,.37)),
           ('Архив допустимого лучшего + элитизм 1\nЛимит 6000 оценок → результат',(.50,.17))]
    for text,pos in boxes:
        ax.text(*pos,text,ha='center',va='center',bbox=dict(boxstyle='round,pad=.6',fc='#f3eee5',ec='#675d50'),fontsize=11)
    for (_,a),(_,b) in zip(boxes,boxes[1:]):
        ax.annotate('',xy=(b[0],b[1]+.065),xytext=(a[0],a[1]-.065),arrowprops=dict(arrowstyle='->',color='#34312d',lw=1.5))
    fig.tight_layout();fig.savefig(out/'algorithm_flow.png',dpi=180);plt.close(fig)


def report(out,summaries,best,problem,examples,cfg):
    by={r['config']:r for r in summaries}; ranked=sorted(summaries,key=lambda r:r['mean'])
    best_avg=ranked[0]
    winner=best['evaluation']
    table=dict(headers=['Конфигурация','Лучшее','Среднее','Медиана','s','Худшее'],rows=[
        [r['config'],r['best'],f"{r['mean']:.2f}",r['median'],f"{r['std']:.2f}",r['worst']] for r in summaries])
    diagnostics=dict(headers=['Конфигурация','Допуст. оценок, %','Допуст. финал, %','Время, с','Резерв ремонта'],rows=[
        [r['config'],f"{100*r['mean_feasible_evaluations']:.1f}",f"{100*r['mean_final_feasible_population']:.1f}",f"{r['mean_seconds']:.3f}",r['fallback_total']] for r in summaries])
    exampletable=dict(headers=['Пример','Недоступность','Лимит смен','Лимит ночей','Конфл.','S'],rows=[
        [e['name'],e['evaluation']['unavailable'],e['evaluation']['overcap'],e['evaluation']['night_overcap'],e['evaluation']['conflicts'],e['evaluation']['soft']] for e in examples])
    sections=[
        dict(heading='Постановка и индивидуальный вариант',paragraphs=[
            'Вариант V = 1 + ((17·28 + 7·1 + 26) mod 20) = 10. Требуется составить недельный график дежурств с ограничениями по сменам.',
            'Синтетический экземпляр содержит 21 событие (по одной утренней, дневной и ночной смене в каждый из 7 дней), 7 сотрудников и 27 пар конфликтующих событий. Это 21 внутридневная пара и 6 пар «ночь — следующее утро».',
            'Данные сгенерированы numpy.random.default_rng(10010). Каждая доступность выбирается с вероятностью 0.80; назначения исходного допустимого плана дополнительно разрешаются. Индивидуальные предпочтения — целые штрафы 0..9. Полный набор данных: data.json. Доступность — единственная модель квалификации; отпуска, ставки и трудовое право здесь не моделируются.']),
        dict(heading='Формальная модель',paragraphs=[
            'Генотип x = (x₀, …, x₂₀), xᵢ ∈ {0, …, 6}. Фенотип — назначенный сотрудник на каждую из 21 смен. Все смены заполняются ровно один раз непосредственно представлением.',
            'Жёсткие ограничения: Aᵢ,xᵢ = 1; Lw = Σᵢ[xᵢ=w] ≤ 4; Nw = Σᵢ:ночь[xᵢ=w] ≤ 2; для каждой конфликтной пары (i,j) должно выполняться xᵢ ≠ xⱼ. Конечный горизонт не связывает воскресную ночь с новым понедельником.',
            'Число нарушений H = число недоступных назначений + Σw max(Lw−4,0) + Σw max(Nw−2,0) + число совпадений сотрудников в конфликтных парах.',
            'Минимизируется S(x) = Σᵢ cᵢ,xᵢ + 3Σw(Lw−3)² + 2Σw(Nw−1)² при H=0. Первое слагаемое отражает неудобство, второе — дисбаланс общего числа смен, третье — дисбаланс ночей. Все веса заданы до эксперимента.',
            'Штрафная оценка F(x)=S(x)+10000·H(x). Для любого вектора S≤1407: предпочтения≤189, общий дисбаланс≤3·378, ночной≤2·42; поэтому любой допустимый план лучше любого недопустимого при таком коэффициенте. В отчётное качество включается только допустимый архив.']),
        dict(heading='Алгоритм и обоснование операторов',paragraphs=[
            f'Популяция {POP}; турнирная селекция k=3; равномерный кроссовер с вероятностью 0.9; мутация с вероятностью 0.8; элитизм 1; остановка после ровно {cfg["budget"]} вычислений фитнес-функции с учётом инициализации. Инициализация: один общий исходный допустимый план и 59 случайных векторов. После кроссовера каждое значение остаётся индексом сотрудника.',
            'Мутация replace выбирает две разные позиции и независимо переназначает сотрудников. Она меняет загрузку, поэтому может улучшать баланс, но нарушать доступность. Мутация swap обменивает два назначения, сохраняя общую загрузку; ограничения ночей и конфликтов при этом могут нарушаться.',
            'В штрафном варианте исходный вектор оценивается без изменений. В ремонтном варианте смены обходятся в случайном порядке; исходное назначение сохраняется, если оно допустимо относительно уже построенной части, иначе выбирается случайный допустимый сотрудник. Учитываются доступность, лимиты общей/ночной загрузки и все уже назначенные конфликтующие смены.',
            'Если частичное построение попадает в тупик, оно начинается заново, максимум 12 попыток. После этого используется приложенный и проверенный допустимый исходный план. Таким образом, ремонт всегда завершается. Число резервных возвратов записано в таблице; ремонт не использует целевую функцию для выбора, но требует дополнительных проверок ограничений.',
            'Лучшее допустимое решение сохраняется отдельно от штрафной популяции. Замена поколения оставляет одну старую элиту и 59 лучших из 60 новых потомков. Все 60 потомков оцениваются, включая не вошедшего в популяцию.'],image='algorithm_flow.png'),
        dict(heading='Допустимые и недопустимые решения',paragraphs=[
            'Ниже приведены два различных допустимых плана и два недопустимых. Полные векторы и все проверки сохранены в examples.json. В первом недопустимом примере один сотрудник получает утреннюю и дневную смену одного дня; во втором все 21 смена назначены сотруднику 1.'],table=exampletable),
        dict(heading='План сравнения и бюджеты',paragraphs=[
            f'Исследован полный факторный план 2×2: обработка ограничений penalty/repair и мутация replace/swap. Остальные параметры одинаковы. Для каждой из четырёх конфигураций выполнено {cfg["runs"]} независимых запусков, seeds={cfg["seeds"][0]}..{cfg["seeds"][-1]}. Те же seeds образуют парные сравнения конфигураций; генераторы разных запусков независимы.',
            f'Baseline — случайное допустимое построение: новый равномерный вектор ремонтируется тем же механизмом и оценивается. Оно получает тот же исходный допустимый план и ровно {cfg["budget"]} оценок, как ГА. Равенство бюджета относится к полной целевой оценке; проверки ограничений в ремонте отдельно не подсчитываются, их стоимость отражена во времени. Детерминированная проверка исходного плана при загрузке не входит в поиск и одинакова для всех методов.',
            'В таблицах s — выборочное стандартное отклонение, ddof=1. Машиночитаемые исходные результаты: perrun.csv, solutions.json, histories.csv; параметры: config.json.']),
        dict(heading='Итоговое качество по серии запусков',paragraphs=[
            f'Наименьшее среднее S={best_avg["mean"]:.2f} получено конфигурацией {best_avg["config"]}. Минимальное найденное допустимое значение во всех сериях — {best["best"]} (seed={best["seed"]}, {best["config"]}). Во всех запусках найдено допустимое решение: общий исходный план гарантирует это с самого начала; доля сама по себе не доказывает превосходства метода.'],table=table,image='quality.png'),
        dict(heading='Допустимость, стоимость ограничений и сходимость',paragraphs=[
            'Доля допустимых оценок показывает, сколько фитнес-вызовов приходится на допустимые кандидаты. Доля допустимых особей в конечной популяции характеризует работу ограничений, а не просто наличие исходного допустимого плана.',
            f'При замене генов штрафной вариант даёт среднее {by["penalty_replace"]["mean"]:.2f}, ремонтный — {by["repair_replace"]["mean"]:.2f}; изменение отражает способ обработки ограничений при фиксированной мутации. При ремонте обмен даёт {by["repair_swap"]["mean"]:.2f}, замена — {by["repair_replace"]["mean"]:.2f}; это контролируемое сравнение операторов.',
            f'Случайное допустимое построение даёт среднее {by["random_none"]["mean"]:.2f}. Его следует считать серьёзным baseline: сравнение не предполагает, что ГА обязательно победит. Время измерено perf_counter на текущей машине; оно зависит от среды.',
            'На графике линия — среднее лучшего допустимого значения, полупрозрачный диапазон — min..max по независимым запускам. Это диапазон разброса, а не доверительный интервал. Архив делает каждую траекторию невозрастающей.'],table=diagnostics,image='convergence.png'),
        dict(heading='Лучший график дежурств',paragraphs=[
            f'Лучший план имеет S={winner["soft"]}: предпочтения {winner["preference"]}, штраф баланса {winner["fairness"]}. Все жёсткие нарушения равны нулю. Загрузка сотрудников: {winner["loads"]}; ночные смены: {winner["nights"]}. Полная предметная таблица сохранена в best_schedule.csv.',
            'Результат приемлем как план данной учебной модели: все смены закрыты, недоступные назначения отсутствуют, конфликты и лимиты соблюдены. Глобальная оптимальность не доказана, поскольку точный оптимум экземпляра не вычислялся.'],image='schedule.png'),
        dict(heading='Выводы, ограничения и ответы к защите',paragraphs=[
            f'В данном экземпляре лучший по среднему результат — {best_avg["config"]}. Вывод относится к одному синтетическому набору и бюджету {cfg["budget"]}, а не ко всем задачам составления расписаний. Для обобщения нужны другие seeds данных, большие размеры и дополнительные уровни плотности ограничений.',
            'Равномерный кроссовер сохраняет алфавит генов, но не допустимость. Обмен сохраняет общую загрузку, однако может нарушить ночные лимиты, доступность и отдых; поэтому его необходимо проверять/ремонтировать. Предпочтения мягкие, а доступность и пары конфликтов жёсткие.',
            'Почему нельзя судить по одному запуску: траектории стохастичны и различаются; таблица показывает центральную тенденцию и разброс. Преждевременная сходимость проявляется плато архивного лучшего и малым изменением качества; для диагностики разнообразия в продолжении работы полезно добавить среднее расстояние Хэмминга.',
            'При n сменах пространство назначений имеет размер 7^n (в этом экземпляре 7^21); с ростом числа смен увеличивается и стоимость проверки конфликтных пар. Возможные улучшения: согласованное сохранение блоков дней, цепочки обменов, локальный поиск допустимых переназначений, точное решение малых экземпляров для оценки разрыва и учёт полного календарного трудового законодательства.',
            f'На данном наборе два оператора при ремонте почти равны: разность средних составляет {abs(by["repair_replace"]["mean"]-by["repair_swap"]["mean"]):.2f}, что мало относительно выборочных стандартных отклонений {by["repair_replace"]["std"]:.2f} и {by["repair_swap"]["std"]:.2f}. Оснований объявлять обмен существенно лучше замены по этим данным нет.',
            'Математически гарантированы корректность алфавита, конечность ремонта, допустимость его результата за счёт резервного плана и точный бюджет оценок. Качество и преимущество конфигураций — экспериментальные результаты, не гарантия глобального минимума.'])]
    content=dict(title='Лабораторная работа № 2',subtitle='Генетический алгоритм для недельного расписания дежурств',task='Вариант 10 · 21 смена · 7 сотрудников · 27 конфликтов',sections=sections)
    (out/'report_content.json').write_text(json.dumps(content,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    ex=sub.add_parser('experiment');ex.add_argument('--runs',type=int,default=20);ex.add_argument('--budget',type=int,default=6000);ex.add_argument('--seed-start',type=int,default=0);ex.add_argument('--out',type=Path,default=HERE)
    one=sub.add_parser('single');one.add_argument('--method',choices=['penalty','repair','random'],default='repair');one.add_argument('--operator',choices=['replace','swap'],default='replace');one.add_argument('--seed',type=int,default=0);one.add_argument('--budget',type=int,default=6000)
    args=parser.parse_args()
    try:
        if args.command == 'experiment':
            nonnegative_integer(args.runs, '--runs', 1)
            nonnegative_integer(args.budget, '--budget', POP)
            nonnegative_integer(args.seed_start, '--seed-start')
        else:
            nonnegative_integer(args.seed, '--seed')
            nonnegative_integer(args.budget, '--budget', POP)
    except ValueError as error:
        parser.error(str(error))
    if args.command=='experiment':
        experiment(args.out,args.runs,args.budget,args.seed_start)
    else:
        p=Problem(generate_data());r=run(p,args.method,args.operator,args.seed,args.budget);r.pop('history');print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
