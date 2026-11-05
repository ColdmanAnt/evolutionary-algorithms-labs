"""ЛР1: собственный вещественный ГА для функции Михалевича."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from time import perf_counter
import numpy as np

ROOT = Path(__file__).resolve().parent


def objective(x: np.ndarray) -> np.ndarray:
    """Одна строка = одна особь; пакетная обработка не меняет счётчик оценок."""
    x = np.atleast_2d(x)
    i = np.arange(1, x.shape[1] + 1)
    return -np.sum(np.sin(x) * np.sin(i * x*x / np.pi)**20, axis=1)


def reflect(x: np.ndarray) -> np.ndarray:
    """Зеркальное восстановление [0, pi] при любом конечном выходе."""
    return np.pi - np.abs(np.mod(x, 2*np.pi) - np.pi)


def tournament(pop: np.ndarray, fit: np.ndarray, rng: np.random.Generator,
               count: int, k: int = 3) -> np.ndarray:
    candidates = rng.integers(0, len(pop), (count, k))
    winners = candidates[np.arange(count), np.argmin(fit[candidates], axis=1)]
    return pop[winners].copy()


def sbx(a: np.ndarray, b: np.ndarray, rng: np.random.Generator,
        probability: float = .9, eta: float = 10.) -> np.ndarray:
    """SBX, один ребёнок от пары; второй симметричный знак выбирается случайно."""
    u = rng.random(a.shape)
    beta = np.where(u <= .5, (2*u)**(1/(eta+1)),
                    (1/(2*(1-u)))**(1/(eta+1)))
    sign = np.where(rng.random(a.shape) < .5, -1., 1.)
    child = .5*((1+sign*beta)*a + (1-sign*beta)*b)
    crossing = rng.random((len(a), 1)) < probability
    return np.where(crossing, child, a)


def run_ga(seed: int, sigma_fraction: float, *, dimension: int = 9,
           population: int = 80, budget: int = 24000, elite: int = 2,
           crossover: float = .9, eta: float = 10., tournament_k: int = 3) -> dict:
    if dimension < 2 or population < 30 or budget < population or not 0 < elite < population:
        raise ValueError('Некорректные размерность, популяция, бюджет или элитизм')
    if sigma_fraction <= 0 or not 0 <= crossover <= 1 or eta <= 0 or tournament_k < 1:
        raise ValueError('Некорректные параметры операторов')
    rng = np.random.default_rng(seed)
    start = perf_counter()
    pop = rng.uniform(0, np.pi, (population, dimension))
    fit = objective(pop)
    used = population
    generation = 0
    history = [(generation, used, float(fit.min()))]
    while used < budget:
        n = min(population, budget-used)
        a = tournament(pop, fit, rng, n, tournament_k)
        b = tournament(pop, fit, rng, n, tournament_k)
        children = sbx(a, b, rng, crossover, eta)
        mask = rng.random(children.shape) < 1/dimension
        children += mask * rng.normal(0, sigma_fraction*np.pi, children.shape)
        children = reflect(children)
        child_fit = objective(children)
        used += n
        # При неполном последнем поколении недостающие места сохраняют родителей.
        best_parents = np.argsort(fit, kind='stable')[:max(elite, population-n)]
        combined = np.concatenate((pop[best_parents], children), axis=0)
        combined_fit = np.concatenate((fit[best_parents], child_fit))
        survivors = np.argsort(combined_fit, kind='stable')[:population]
        pop, fit = combined[survivors], combined_fit[survivors]
        generation += 1
        history.append((generation, used, float(fit.min())))
    best = int(np.argmin(fit))
    return dict(seed=seed, best=float(fit[best]), x=pop[best].tolist(),
                evaluations=used, generations=generation,
                feasible=bool(np.all((pop >= 0) & (pop <= np.pi))),
                seconds=perf_counter()-start, history=history)


def run_random(seed: int, *, dimension: int = 9, population: int = 80,
               budget: int = 24000) -> dict:
    if dimension < 2 or population < 1 or budget < population:
        raise ValueError('Некорректные параметры случайного поиска')
    rng = np.random.default_rng(seed)
    start = perf_counter()
    used, generation, best = 0, -1, float('inf')
    best_x = None
    history = []
    while used < budget:
        count = min(population, budget-used)
        x = rng.uniform(0, np.pi, (count, dimension))
        fit = objective(x)
        j = int(np.argmin(fit))
        if fit[j] < best:
            best, best_x = float(fit[j]), x[j].tolist()
        used += count
        generation += 1
        history.append((generation, used, best))
    return dict(seed=seed, best=best, x=best_x, evaluations=used,
                generations=generation, feasible=True,
                seconds=perf_counter()-start, history=history)


def numerical_reference(dimension: int) -> dict:
    """Внешний к ГА численный ориентир: сепарабельность + одномерная сетка.

    Не используется для инициализации, обучения или отбора ГА. Не является
    интервальным доказательством глобального минимума.
    """
    grid = np.linspace(0., np.pi, 100001)
    xs = []
    for i in range(1, dimension+1):
        f = lambda x: -np.sin(x)*np.sin(i*x*x/np.pi)**20
        values = f(grid)
        j = int(np.argmin(values))
        a, b = grid[max(j-1, 0)], grid[min(j+1, len(grid)-1)]
        ratio = (math.sqrt(5)-1)/2
        c, e = b-ratio*(b-a), a+ratio*(b-a)
        for _ in range(80):
            if f(c) < f(e):
                b, e = e, c
                c = b-ratio*(b-a)
            else:
                a, c = c, e
                e = a+ratio*(b-a)
        xs.append(float((a+b)/2))
    return dict(x=xs, value=float(objective(np.array(xs))[0]),
                method='100001 grid points per coordinate + 80 golden-section iterations',
                status='numerical_reference_not_certified_global_optimum',
                lower_bound=-dimension)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def experiment(config: dict, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    all_runs, rows, histories, summaries = {}, [], [], []
    expected = dict(dimension=9,population=80,budget=24000,elite=2,
                    crossover_probability=.9,sbx_eta=10,tournament_k=3,
                    seeds=list(range(100,120)))
    if any(config.get(key) != value for key,value in expected.items()):
        raise ValueError('Серия и её отчёт используют архивный план config.json; для отдельных изменённых параметров используйте --seed')
    if config['sigma_fractions'] != [.02, .10]:
        raise ValueError('Для архивной сравнительной серии заданы масштабы 0.02 и 0.10; иной масштаб проверяйте через --seed и --sigma')
    for name, scale in [('GA_sigma_002', config['sigma_fractions'][0]),
                        ('GA_sigma_010', config['sigma_fractions'][1]), ('Random', None)]:
        runs = []
        for seed in config['seeds']:
            kwargs = dict(dimension=config['dimension'], population=config['population'], budget=config['budget'])
            if scale is None:
                result = run_random(seed, **kwargs)
            else:
                result = run_ga(seed, scale, elite=config['elite'], crossover=config['crossover_probability'],
                                eta=config['sbx_eta'], tournament_k=config['tournament_k'], **kwargs)
            runs.append(result)
            rows.append(dict(algorithm=name, seed=seed, best=result['best'],
                             evaluations=result['evaluations'], generations=result['generations'],
                             feasible=int(result['feasible']), seconds=result['seconds']))
            histories.extend(dict(algorithm=name, seed=seed, generation=g, evaluations=n, best=v)
                             for g, n, v in result['history'])
        all_runs[name] = runs
        vals = np.array([r['best'] for r in runs])
        summaries.append(dict(algorithm=name, runs=len(runs), best=float(vals.min()),
                              mean=float(vals.mean()), median=float(np.median(vals)),
                              std_sample=float(vals.std(ddof=1)), worst=float(vals.max()),
                              feasible_share=float(np.mean([r['feasible'] for r in runs])),
                              mean_seconds=float(np.mean([r['seconds'] for r in runs])),
                              evaluations=config['budget']))
    reference = numerical_reference(config['dimension'])
    winner_name, winner = min(((name, r) for name, runs in all_runs.items() for r in runs), key=lambda t:t[1]['best'])
    save_json(output/'best_solution.json', dict(algorithm=winner_name, **winner))
    save_json(output/'reference.json', reference)
    save_json(output/'summary.json', summaries)
    write_csv(output/'runs.csv', rows)
    write_csv(output/'history.csv', histories)
    write_csv(output/'summary.csv', summaries)
    plot_results(all_runs, reference, output)
    create_report(config, summaries, reference, winner_name, winner, output)
    return dict(summaries=summaries, reference=reference, best_algorithm=winner_name, best=winner['best'])


def plot_results(all_runs: dict, reference: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from cycler import cycler
    colors = ['#7d353b', '#707448', '#9b948a']
    graphite, ivory = '#34312d', '#fbf8f1'
    plt.rcParams.update({
        'font.family':'DejaVu Sans', 'font.size':10,
        'axes.spines.top':False, 'axes.spines.right':False,
        'axes.prop_cycle':cycler(color=colors + ['#ad6b42']),
        'figure.facecolor':ivory, 'axes.facecolor':ivory,
        'savefig.facecolor':ivory, 'text.color':graphite,
        'axes.labelcolor':graphite, 'axes.edgecolor':graphite,
        'xtick.color':graphite, 'ytick.color':graphite,
        'grid.color':'#9b948a', 'legend.facecolor':'#f3eee5',
        'legend.edgecolor':'#c7bdb0', 'legend.framealpha':1.,
    })
    fig, ax = plt.subplots(figsize=(9, 4.7), constrained_layout=True)
    for (name, runs), color in zip(all_runs.items(), colors):
        v = np.array([[v for _,_,v in r['history']] for r in runs])
        x = [g for g,_,_ in runs[0]['history']]
        label = {'GA_sigma_002':'ГА: σ = 0,02π', 'GA_sigma_010':'ГА: σ = 0,10π', 'Random':'Случайный поиск'}[name]
        ax.plot(x, v.mean(axis=0), color=color, label=label)
        ax.fill_between(x, v.min(axis=0), v.max(axis=0), color=color, alpha=.13)
    ax.axhline(reference['value'], color=graphite, linestyle='--', linewidth=1, label='Численный ориентир')
    ax.set(xlabel='Поколение / пакет из 80 оценок', ylabel='Лучшее значение f(x)', title='Средняя сходимость и min–max по 20 запускам')
    ax.grid(alpha=.18)
    ax.legend(fontsize=9, loc='upper right')
    fig.savefig(output/'convergence.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    names = list(all_runs)
    boxes = ax.boxplot([[r['best'] for r in all_runs[n]] for n in names],
                       tick_labels=['ГА σ=0,02π','ГА σ=0,10π','Случайный'],
                       patch_artist=True,
                       boxprops={'edgecolor':graphite},
                       medianprops={'color':graphite,'linewidth':1.5},
                       whiskerprops={'color':graphite},
                       capprops={'color':graphite},
                       flierprops={'markeredgecolor':'#ad6b42'})
    for patch, color in zip(boxes['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(.28)
    ax.set(ylabel='Лучшее конечное значение', title='Распределение результатов: одинаковые 24 000 оценок')
    ax.grid(axis='y', alpha=.18)
    fig.savefig(output/'distribution.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5.6), constrained_layout=True)
    ax.set(xlim=(0,10), ylim=(0,10))
    ax.axis('off')
    boxes = [(8.9,'Seed и параметры → равномерная популяция'),(7.4,'Оценить 80 особей; сохранить элиту'),
             (5.9,'Турнир k=3 → SBX → гауссовская мутация'),(4.4,'Отразить координаты в [0, π]'),
             (2.9,'Оценить потомков → элита + лучшие потомки'),(1.4,'Бюджет исчерпан? → результат и журнал')]
    for y,t in boxes:
        ax.text(5,y,t,ha='center',va='center',bbox=dict(boxstyle='round,pad=.7',fc='#f3eee5',ec='#675d50'),fontsize=10)
        if y>1.4: ax.annotate('',xy=(5,y-1),xytext=(5,y-.42),arrowprops=dict(arrowstyle='->',color=graphite))
    ax.text(5,.45,'Если бюджет не исчерпан: повторить с шага «Турнир k=3»',
            ha='center',va='center',fontsize=9,color='#ad6b42')
    fig.savefig(output/'algorithm_flow.png', dpi=180)
    plt.close(fig)


def replot_saved(output: Path) -> None:
    """Перестроить рисунки по сохранённым данным без повторных экспериментов."""
    all_runs: dict[str, list[dict]] = {}
    by_key = {}
    with (output/'runs.csv').open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            record = dict(seed=int(row['seed']), best=float(row['best']), history=[])
            all_runs.setdefault(row['algorithm'], []).append(record)
            by_key[(row['algorithm'], int(row['seed']))] = record
    with (output/'history.csv').open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            by_key[(row['algorithm'], int(row['seed']))]['history'].append(
                (int(row['generation']), int(row['evaluations']), float(row['best'])))
    reference = json.loads((output/'reference.json').read_text(encoding='utf-8'))
    plot_results(all_runs, reference, output)


def create_report(config, summaries, reference, winner_name, winner, output):
    lookup = {s['algorithm']:s for s in summaries}
    a,b,r = (lookup[n] for n in ['GA_sigma_002','GA_sigma_010','Random'])
    superior = '0,02π' if a['mean'] < b['mean'] else '0,10π'
    sections = [
      dict(heading='1. Постановка задачи и вариант',paragraphs=[
          'Выполнил: Холодков Антон Евгеньевич, группа 517, номер по списку N = 28. Q = 1, Y = 26. V = 1 + ((17·28 + 7·1 + 26) mod 20) = 10. Размерность d = 5 + (10 mod 6) = 9.',
          'Найти минимум функции Михалевича f(x) = -Σ[i=1..9] sin(x_i)·sin²⁰(i·x_i²/π), где 0 ≤ x_i ≤ π. Все координаты вещественные. Генотип и фенотип совпадают: особь является девятимерным вектором, дополнительное декодирование не требуется.',
          'Каждое слагаемое лежит в [-1, 0], поэтому -9 ≤ f(x) ≤ 0. Функция сепарабельна, но сильно мультимодальна: степень 20 делает впадины узкими, а случайная точка редко одновременно попадает в хорошие интервалы всех координат. Локальный поиск может остаться в одной из неглобальных впадин. Специализированное разделение по координатам эффективно, поэтому ГА здесь исследуется как общий учебный метод, а не как лучший возможный решатель.']),
      dict(heading='2. Алгоритм и операторы',paragraphs=[
          'Начальная популяция: 80 независимых равномерных точек в [0, π]^9. Минимизация ведётся непосредственно по f; дополнительного масштабирования фитнеса нет. В турнире из трёх индексов с возвращением выбирается особь с наименьшим f.',
          'SBX (simulated binary crossover) применяется с вероятностью 0,9 к паре векторов, индекс распределения η = 10. Для каждой координаты u ~ U(0,1); β = (2u)^(1/(η+1)) при u ≤ 0,5 и β = [1/(2(1-u))]^(1/(η+1)) иначе. Потомок = 0,5·[(1+sβ)a + (1-sβ)b], s равновероятно равен -1 или +1. При отсутствии кроссовера копируется первый родитель. SBX сохраняет смысл вещественных координат и допускает исследование окрестностей родителей.',
          'Каждая координата мутирует независимо с вероятностью 1/9: добавляется N(0,σ²). Единственный изменяемый фактор в двух конфигурациях — σ: 0,02π или 0,10π. Все остальные параметры и набор seed одинаковы. Отражение x ← π - |(x mod 2π) - π| возвращает любые конечные координаты в область, не накапливая точки на границе.',
          'Два лучших родителя переходят в следующее поколение без повторной оценки; остальные места занимают лучшие оценённые потомки. За поколение оцениваются 80 новых особей. Инициализация также включена в бюджет: 80 + 299·80 = 24 000. Критерий остановки — ровно 24 000 вычислений функции. Реализация также поддерживает неполный последний пакет. Лучшее найденное значение никогда не ухудшается.'],image='algorithm_flow.png'),
      dict(heading='3. План эксперимента',paragraphs=[
          'Проведено по 20 независимых запусков каждой из двух конфигураций ГА и случайного поиска: всего 60 запусков. Использованы seed 100–119. Запуски внутри метода независимы; одинаковые seed между методами образуют парный дизайн. Случайный поиск генерирует 24 000 равномерных точек и сохраняет лучшую. Пакетная NumPy-обработка считает каждую строку отдельной оценкой; сравнение честно по числу целевых вычислений, а не по секундам.',
          'Параметры сохранены в config.json; runs.csv содержит отдельные результаты, history.csv — лучшие значения после каждого пакета, summary.csv — агрегаты. Стандартное отклонение выборочное (ddof = 1); допустимость проверена по всем особям конечных популяций. Время — однопроцессное фактическое время на текущем компьютере и зависит от оборудования.'],table=dict(headers=['Параметр','Значение'],rows=[['Популяция / элита','80 / 2'],['Бюджет на запуск','24 000'],['Число поколений после инициализации','299'],['Турнир / SBX η / p_c','3 / 10 / 0,9'],['p_m на координату','1/9'],['σ','0,02π либо 0,10π'],['Seed','100–119'],['Обработка ограничений','Отражение координат']])),
      dict(heading='4. Результаты серии',paragraphs=[
          'Все значения ниже относятся к минимизации: меньше — лучше. Доля допустимых результатов равна 1,00 для каждого метода. Каждый запуск использовал ровно 24 000 оценок. Показаны все серии, а не один удачный seed.'],table=dict(headers=['Метод','Лучшее','Среднее','Медиана','s','Худшее','Время, с'],rows=[[s['algorithm'],*[f"{s[k]:.5f}" for k in ['best','mean','median','std_sample','worst']],f"{s['mean_seconds']:.3f}"] for s in summaries])),
      dict(heading='5. Сходимость и влияние масштаба мутации',paragraphs=[
          f'Меньшее среднее значение в этой серии получил масштаб σ = {superior}. Среднее для σ=0,02π: {a["mean"]:.6f}; для σ=0,10π: {b["mean"]:.6f}; для случайного поиска: {r["mean"]:.6f}. Смена масштаба меняет одновременно возможность перескакивать между впадинами и точность удержания хороших координат; вывод относится к данной функции, бюджету и параметрам, а не к универсальному преимуществу одного масштаба.',
          'Линии показывают среднее лучшего-до-текущего значения; полупрозрачные полосы — минимум и максимум по 20 запускам. Это диапазон наблюдений, а не доверительный интервал. Для случайного поиска горизонтальная ось означает номер пакета из 80 оценок. Последовательность элитизма объясняет монотонность каждой траектории.'],image='convergence.png'),
      dict(heading='6. Устойчивость результатов',paragraphs=[
          'Диаграмма показывает распределения конечного качества при равном бюджете. Межквартильный разброс и выбросы следует оценивать вместе со средним и стандартным отклонением; единичный лучший результат не доказывает устойчивость. Серия из 20 запусков достаточна для требований лабораторной, но не обосновывает широкие выводы на других функциях.'],image='distribution.png'),
      dict(heading='7. Лучшее решение и ориентиры качества',paragraphs=[
          f'Лучший фактически найденный результат: {winner["best"]:.10f}, метод {winner_name}, seed = {winner["seed"]}. Вектор x = ['+', '.join(f'{x:.9f}' for x in winner['x'])+']. Вектор и журнал сохранены в best_solution.json.',
          f'Независимый численный ориентир равен {reference["value"]:.10f}. Он получен использованием сепарабельности: 100 001 узел на [0,π] для каждой координаты и 80 шагов метода золотого сечения в окрестности лучшего узла. Код ориентирующего расчёта написан самостоятельно и не участвует в работе ГА или случайной выборки. Это численная оценка высокого разрешения, а не строгий сертификат глобального оптимума; гарантированной нижней границей остаётся -9.',
          'Литературный ориентир: университетский каталог тестовых функций A. Hedar (Kyoto University) приводит для d=10 значение около -9,66015: https://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page2376.htm . Вывод для d=9 получен здесь, а не процитирован из таблицы: десятое независимое слагаемое достигает -1 при x_10=π/2, поэтому из приведённого округлённого значения следует ориентир около -8,66015. Он согласуется с нашим отдельным численным расчётом.',
          f'Разрыв лучшего результата до численного ориентира: {winner["best"]-reference["value"]:.6f}. Полученный результат оценивается по разрыву, устойчивости серии и преимуществу над равнобюджетной случайной выборкой. Даже близкое совпадение с ориентиром не даёт ГА математической гарантии глобальности.']),
      dict(heading='8. Проверка корректности, ограничения и защита',paragraphs=[
          'Автоматические проверки: известные значения и диапазон функции; восстановление границ при многократном выходе; воспроизводимость по seed; точное соблюдение нестандартного бюджета; монотонность лучшей траектории при элитизме; допустимость конечного решения. Команда: python -m unittest discover -s . -p "test_*.py".',
          'Ограничения: единственная функция и размерность; фиксированные параметры турнира и SBX; два масштаба мутации; нет статистического теста значимости и оценки переноса результатов. Улучшения: адаптивный масштаб, независимая тестовая серия seed, исследование размера популяции и рестартов при стагнации.',
          'К защите: объяснить, почему генотип совпадает с фенотипом; почему SBX и отражение согласованы с вещественной областью; как считается бюджет с сохранением элиты; чем наблюдаемый min–max отличается от доверительного интервала; почему численный ориентир и эмпирическое преимущество не являются гарантией глобального оптимума.'])
    ]
    content = dict(title='Лабораторная работа № 1',subtitle='Многомерная оптимизация функции Михалевича',task='Вариант 10; d = 9',sections=sections)
    save_json(output/'report_content.json',content)
    # Корневой сборщик читает только архивный каталог results/.
    # Пробный запуск с --output сохраняет все материалы внутри своего каталога.
    if output.resolve() == (ROOT/'results').resolve():
        package_content = json.loads(json.dumps(content,ensure_ascii=False))
        for section in package_content['sections']:
            if 'image' in section:
                section['image'] = (Path('results')/section['image']).as_posix()
        save_json(ROOT/'report_content.json',package_content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results')
    parser.add_argument('--seed',type=int,help='Один воспроизводимый запуск ГА')
    parser.add_argument('--sigma',type=float,default=.02,help='Масштаб мутации как доля π')
    parser.add_argument('--replot',action='store_true',help='Обновить только рисунки из сохранённых CSV/JSON')
    args = parser.parse_args()
    if args.replot:
        replot_saved(args.output)
        return
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.seed is not None:
        result = run_ga(args.seed,args.sigma,dimension=config['dimension'],population=config['population'],budget=config['budget'],
                        elite=config['elite'],crossover=config['crossover_probability'],eta=config['sbx_eta'],tournament_k=config['tournament_k'])
        print(json.dumps({k:v for k,v in result.items() if k!='history'},ensure_ascii=False,indent=2))
    else:
        print(json.dumps(experiment(config,args.output),ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
