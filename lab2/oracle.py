"""Независимая проверка расписания: обычные циклы Python, без production evaluate.

Ночные события определяются по названиям смен из данных, а не срезом 2::3.
Этот модуль используется тестами и проверкой сохранённых результатов.
"""
from numbers import Integral


def oracle_schedule(data, assignments):
    staff_count = len(data['staff'])
    event_count = len(data['events'])
    if (len(assignments) != event_count
            or any(isinstance(w, bool) or not isinstance(w, Integral)
                   or w < 0 or w >= staff_count for w in assignments)):
        raise ValueError('Oracle: неверный алфавит или длина расписания')
    loads, nights = [0] * staff_count, [0] * staff_count
    preference, unavailable = 0, 0
    for i, (event, worker) in enumerate(zip(data['events'], assignments)):
        loads[worker] += 1
        if event['shift'] == 'Ночь':
            nights[worker] += 1
        unavailable += int(data['availability'][i][worker] == 0)
        preference += data['preference_costs'][i][worker]
    overcap = sum(max(0, count - data['max_shifts']) for count in loads)
    night_overcap = sum(max(0, count - data['max_nights']) for count in nights)
    conflicts = sum(assignments[first] == assignments[second] for first, second in data['conflicts'])
    # Средние загрузки выводятся из количества событий и сотрудников.
    target = event_count // staff_count
    night_target = sum(event['shift'] == 'Ночь' for event in data['events']) // staff_count
    fairness = sum(3 * (count - target) ** 2 for count in loads)
    fairness += sum(2 * (count - night_target) ** 2 for count in nights)
    violations = unavailable + overcap + night_overcap + conflicts
    soft = preference + fairness
    return dict(soft=soft, preference=preference, fairness=fairness, violations=violations,
                unavailable=unavailable, overcap=overcap, night_overcap=night_overcap,
                conflicts=conflicts, score=soft + 10000 * violations,
                loads=loads, nights=nights)
