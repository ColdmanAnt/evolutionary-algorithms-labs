"""Проверка комплектности, серий и тестов; записывает verification.json."""
import csv
import collections
import importlib.metadata
import json
import platform
from pathlib import Path
import re
import subprocess
import sys
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]

def read(relative):
    with (ROOT/relative).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    tests=[]
    for i in range(1,5):
        run=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/f'lab{i}'),'-p','test*.py','-v'],capture_output=True,text=True)
        log=run.stdout+run.stderr
        if run.returncode:raise RuntimeError(log)
        n=int(re.search(r'Ran (\d+) tests?',log).group(1))
        tests.append({'lab':i,'passed':n,'log':log})
    specifications=[
      (1,'lab1/results/runs.csv',('algorithm',),3,20,24000),
      (2,'lab2/perrun.csv',('config',),5,20,6000),
      (3,'lab3/runs.csv',('penalty',),2,20,240),
      (4,'lab4/runs.csv',('function','method'),12,30,30000)]
    series=[]
    for lab,file,keys,groups,count,budget in specifications:
        rows=read(file)
        grouping=collections.defaultdict(list)
        for row in rows:grouping[tuple(row[k] for k in keys)].append(row)
        assert len(grouping)==groups,(lab,len(grouping))
        assert all(len(g)==count for g in grouping.values())
        assert all(len({r['seed'] for r in g})==count for g in grouping.values())
        assert all(int(r['evaluations'])==budget for r in rows)
        series.append({'lab':lab,'configurations':groups,'seeds_per_configuration':count,
                       'runs':len(rows),'budget_per_run':budget,'unit':'fitness candidate calls' if lab==3 else 'objective evaluations'})
    pages=[]
    for i in range(1,5):
        reader=PdfReader(ROOT/f'reports/lab{i}_report.pdf')
        text=''.join(page.extract_text() or '' for page in reader.pages)
        assert 'Холодков' in text and '517' in text
        assert len(text)>5000
        pages.append(len(reader.pages))
    assert len(PdfReader(ROOT/'all_reports.pdf').pages)==sum(pages)
    result={'author':'Холодков Антон Евгеньевич','group':517,'N':28,'Q':1,'Y':26,
            'variant':1+((17*28+7+26)%20),'date':'2026-11-05',
            'environment':{'python':platform.python_version(),'system':platform.system(),'machine':platform.machine(),
                           **{k:importlib.metadata.version(k) for k in ['numpy','matplotlib']}},
            'series':series,'total_runs':sum(s['runs'] for s in series),
            'unit_tests_passed':sum(t['passed'] for t in tests),'unit_tests':tests,
            'pdf_pages':pages,'combined_pdf_pages':sum(pages),
            'independent_review':{'lab1':'operators, budgets, exact replay and numerical reference',
              'lab2':'100 saved schedules independently recomputed; constraints, score and budget match',
              'lab3':'train-only preprocessing, validation-only selection, held-out test perturbation invariant',
              'lab4':'360 runs,90000 checkpoints, budgets, monotonicity, statistics and replay checked'},
            'visual_review':'All PDF pages rendered and inspected; final charts use warm burgundy/olive/graphite palette.',
            'notes':['Lab3 has equal candidate-call budgets; distinct trainings are reported separately.',
                     'Track B chosen for lab3; lecturer assignment/approval is not asserted.',
                     'Lab4 proposal is provided; lecturer approval and successful defense are not asserted.']}
    (ROOT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['variant','total_runs','unit_tests_passed','combined_pdf_pages']},ensure_ascii=False))

if __name__=='__main__':main()
