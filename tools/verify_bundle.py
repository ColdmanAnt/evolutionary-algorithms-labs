"""Проверка алгоритмов, данных и PDF; --replay воспроизводит все560 запусков."""
from __future__ import annotations

import argparse
import collections
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

from PIL import Image
from pypdf import PdfReader
from pypdf.generic import IndirectObject, StreamObject

ROOT=Path(__file__).resolve().parents[1]


def require(condition, message):
    # Проверки входных артефактов должны работать и с python -O.
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical_file_digest(path):
    data=path.read_bytes()
    if path.suffix in {'.py','.md','.json','.csv','.txt','.yml','.yaml'} or path.name=='.python-version':
        data=data.replace(b'\r\n',b'\n')
    return digest(data)


def compact(text):
    return ''.join(str(text).split())


def pdf_resource_signature(value, active=None):
    """Сравнить PDF-ресурсы по содержимому, а не номерам косвенных объектов.

    Включает шрифты, изображения, маски, цвета и вложенные Form XObjects.
    Потоки декодируются: отличие только способа сжатия не считается порчей.
    """
    active=set() if active is None else active
    if isinstance(value,IndirectObject):
        value=value.get_object()
    token=id(value)
    if isinstance(value,(dict,list)):
        if token in active:return {'cycle':True}
        active.add(token)
        try:
            if isinstance(value,dict):
                result={str(k):pdf_resource_signature(v,active) for k,v in sorted(value.items(),key=lambda kv:str(kv[0]))
                        if str(k) not in ('/Length','/Filter','/DecodeParms')}
                if isinstance(value,StreamObject):result['decoded_sha256']=digest(value.get_data())
                return result
            return [pdf_resource_signature(v,active) for v in value]
        finally:active.remove(token)
    if isinstance(value,bytes):return {'bytes':digest(value)}
    if isinstance(value,(int,float,bool)) or value is None:return value
    return str(value)


def pdf_page_signature(page):
    contents=page.get_contents()
    return dict(content_sha256=digest(contents.get_data() if contents is not None else b''),
                resources=pdf_resource_signature(page.get('/Resources',{})),
                media_box=[float(v) for v in page.mediabox],
                crop_box=[float(v) for v in page.cropbox],rotation=page.get('/Rotate',0))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def validate_variant(root):
    variant=json.loads((root/'variant.json').read_text(encoding='utf-8'))
    for field in ['N','Q','Y','V','lab1_dimension']:
        value=variant.get(field)
        require(type(value) is int,f'variant.json: {field} должен быть целым')
    require(variant['N']>=1 and variant['Q'] in (0,1,2) and 0<=variant['Y']<=99,'Некорректные N/Q/Y')
    group=re.search(r'(507|517|527)$',str(variant.get('group','')))
    require(group is not None and {'507':0,'517':1,'527':2}[group[0]]==variant['Q'],'Группа не соответствует Q')
    calculated=1+((17*variant['N']+7*variant['Q']+variant['Y'])%20)
    require(variant['V']==calculated,'Неверно вычислен вариант')
    require(variant['lab1_dimension']==5+calculated%6,'Неверно вычислена размерность')
    require(variant.get('lab3_track')=='B' and variant.get('lab4_topic')==10,'Трек/проект не соответствует комплекту')
    for i in (1,3):
        config=json.loads((root/f'lab{i}'/'config.json').read_text(encoding='utf-8'))
        for field in ['N','Q','Y']:
            require(config[field]==variant[field],f'ЛР{i}: неверное поле {field}')
        require(config['variant']==variant['V'],f'ЛР{i}: другой вариант')
    config=json.loads((root/'lab1/config.json').read_text(encoding='utf-8'))
    require(config['dimension']==variant['lab1_dimension'],'ЛР1: другая размерность')
    require(variant.get('author')=='Холодков Антон Евгеньевич','Неожиданный автор комплекта')
    return variant


def validate_series(root):
    specifications=[
        (1,'lab1/results/runs.csv',('algorithm',),3,20,24000),
        (2,'lab2/perrun.csv',('config',),5,20,6000),
        (3,'lab3/runs.csv',('penalty',),2,20,240),
        (4,'lab4/runs.csv',('function','method'),12,30,30000)]
    result=[]
    for lab,file,keys,groups,seeds,budget in specifications:
        rows=read_csv(root/file)
        grouped=collections.defaultdict(list)
        for row in rows:
            grouped[tuple(row[k] for k in keys)].append(row)
        require(len(grouped)==groups,f'ЛР{lab}: неполный набор конфигураций')
        require(all(len(g)==seeds and len({r['seed'] for r in g})==seeds for g in grouped.values()),f'ЛР{lab}: число/повторы seed')
        require(all(int(r['evaluations'])==budget for r in rows),f'ЛР{lab}: неверный бюджет')
        result.append(dict(lab=lab,configurations=groups,seeds_per_configuration=seeds,runs=len(rows),budget_per_run=budget,
                           unit='fitness candidate calls' if lab==3 else 'objective evaluations'))
    return result


def validate_markdown_links(root):
    checked=0
    for path in root.rglob('*.md'):
        if any(part in ('.git','.venv','reproduced','scratch_run') for part in path.relative_to(root).parts):
            continue
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if ':' in target or target.startswith('#'):
                continue
            resolved=(path.parent/target.split('#')[0]).resolve()
            require(resolved.is_relative_to(root.resolve()) and resolved.exists(),f'Неверная ссылка {path.relative_to(root)} → {target}')
            checked+=1
    return checked


def validate_pdfs(root):
    combined=PdfReader(root/'all_reports.pdf')
    combined_text=[compact(p.extract_text() or '') for p in combined.pages]
    offset=0
    counts=[]
    hashes=[]
    images=0
    for lab in range(1,5):
        source=root/f'lab{lab}'/'report_content.json'
        content=json.loads(source.read_text(encoding='utf-8'))
        target=root/'reports'/f'lab{lab}_report.pdf'
        reader=PdfReader(target)
        metadata=reader.metadata or {}
        require(metadata.get('/ReportContentSHA256')==digest(source.read_bytes()),f'ЛР{lab}: PDF устарел относительно report_content.json')
        asset_hashes={}
        for section in content['sections']:
            if section.get('image'):
                asset=(source.parent/section['image']).resolve()
                require(asset.is_relative_to(root.resolve()) and asset.is_file(),f'ЛР{lab}: нет рисунка {section["image"]}')
                with Image.open(asset) as image:image.verify()
                asset_hashes[section['image']]=digest(asset.read_bytes())
                images+=1
        assets_digest=digest(json.dumps(asset_hashes,sort_keys=True,ensure_ascii=False).encode())
        require(metadata.get('/ReportAssetsSHA256')==assets_digest,f'ЛР{lab}: PDF устарел относительно рисунков')
        # ReportLab рисует колонтитул до тела страницы; при переносе абзаца он
        # попадает между его фрагментами в extract_text и не является содержимым.
        body_pages=[re.sub(r'^Холодков А\. Е\. · 517 · ЛР\d · Вариант 10\n\d+\n','',p.extract_text() or '',count=1)
                    for p in reader.pages]
        text=''.join(compact(page) for page in body_pages)
        require('Холодков' in text and '517' in text and compact(content['title']) in text,f'ЛР{lab}: неверные реквизиты PDF')
        for section in content['sections']:
            require(compact(section['heading']) in text,f'ЛР{lab}: пропущен раздел {section["heading"]}')
            for paragraph in section.get('paragraphs',[]):
                require(compact(paragraph) in text,f'ЛР{lab}: текст PDF не соответствует разделу {section["heading"]}')
            table=section.get('table')
            if table:
                for row in [table['headers'],*table['rows']]:
                    for cell in row:
                        require(compact(cell) in text,f'ЛР{lab}: ячейка таблицы отсутствует в PDF: {cell}')
        texts=[compact(p.extract_text() or '') for p in reader.pages]
        require(combined_text[offset:offset+len(texts)]==texts,f'ЛР{lab}: страницы общего PDF отличаются от отдельного отчёта')
        for number,page in enumerate(reader.pages):
            require(pdf_page_signature(page)==pdf_page_signature(combined.pages[offset+number]),
                    f'ЛР{lab}: графика/шрифты/содержимое страницы {number+1} общего PDF отличаются')
        offset+=len(texts);counts.append(len(texts));hashes.append(digest(target.read_bytes()))
    require(offset==len(combined.pages),'В общем PDF лишние/недостающие страницы')
    require((combined.metadata or {}).get('/SourceReportsSHA256')==digest(json.dumps(hashes).encode()),'Общий PDF не соответствует отдельным файлам')
    return dict(individual_pages=counts,combined_pages=offset,source_and_asset_hashes_verified=True,
                paragraph_and_table_content_verified=True,combined_page_order_verified=True,
                combined_graphics_fonts_and_streams_verified=True,referenced_images_verified=images)


def execute(command,label):
    run=subprocess.run(command,capture_output=True,text=True)
    if run.returncode:
        raise ValueError(f'{label}: команда завершилась с кодом {run.returncode}\n{run.stdout}\n{run.stderr}')
    return run


def run_checks(root,replay=False):
    variant=validate_variant(root)
    tests=[]
    for directory in ['lab1','lab2','lab3','lab4','tools']:
        run=execute([sys.executable,'-m','unittest','discover','-s',str(root/directory),'-p','test*.py','-v'],directory)
        log=(run.stdout+run.stderr).replace(str(root),'REPOSITORY')
        match=re.search(r'Ran (\d+) tests?',log)
        require(match is not None and int(match[1])>0,f'{directory}: тесты не обнаружены')
        tests.append(dict(suite=directory,passed=int(match[1]),log=log))
    laboratory_audits={}
    for lab in range(1,5):
        command=[sys.executable,str(root/f'lab{lab}'/'audit_saved.py')]
        if replay:command.append('--replay')
        run=execute(command,f'Глубокая проверка ЛР{lab}')
        evidence=json.loads(run.stdout)
        require(evidence and evidence.get('status','passed')=='passed' and evidence.get('passed',True) is True,f'ЛР{lab}: аудит не подтвердил успех')
        laboratory_audits[f'lab{lab}']=evidence
        evidence['artifact_hash_convention']='Per-lab hashes describe raw working-copy bytes; CSV line endings may differ after Git checkout.'
        if replay:
            (root/f'lab{lab}'/'audit.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(f'ЛР{lab}: сохранённые данные'+(' и полная серия' if replay else '')+' проверены',file=sys.stderr,flush=True)
    series=validate_series(root)
    pdfs=validate_pdfs(root)
    links=validate_markdown_links(root)
    fingerprints={}
    for directory in ['lab1','lab2','lab3','lab4','tools','reports']:
        for p in sorted((root/directory).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.name!='audit.json':
                fingerprints[p.relative_to(root).as_posix()]=canonical_file_digest(p)
    for relative in ['variant.json','requirements.txt','requirements-dev.txt','.python-version',
                     '.github/workflows/verify.yml','README.md','all_reports.pdf']:
        fingerprints[relative]=canonical_file_digest(root/relative)
    # В проектном протоколе используется московское время UTC+3; не требует
    # внешней базы IANA/tzdata в стандартной установке Python на Windows.
    now=datetime.now(timezone(timedelta(hours=3),name='Europe/Moscow'))
    return dict(author=variant['author'],group=517,N=variant['N'],Q=variant['Q'],Y=variant['Y'],variant=variant['V'],
                date=now.date().isoformat(),checked_at=now.isoformat(),audit_mode='full_replay' if replay else 'saved_integrity',
                environment=dict(python=platform.python_version(),system=platform.system(),machine=platform.machine(),
                                 **{k:importlib.metadata.version(k) for k in ['numpy','matplotlib','reportlab','pypdf']}),
                series=series,total_runs=sum(s['runs'] for s in series),unit_tests_passed=sum(t['passed'] for t in tests),unit_tests=tests,
                laboratory_audits=laboratory_audits,pdf_pages=pdfs['individual_pages'],combined_pdf_pages=pdfs['combined_pages'],
                pdf_validation=pdfs,relative_markdown_links_checked=links,verified_file_sha256=fingerprints,
                verified_file_hash_convention='SHA256 of Git-normalized LF text; exact bytes for binary files.',
                notes=['Исходные времена запусков сохранены; повторные сравнения исключают wall time.',
                       'ЛР3: равны бюджеты вызовов fitness; фактические обучения и градиентные шаги учитываются отдельно.',
                       'Все560 запусков повторяются только с --replay; быстрый режим проверяет сохранённые данные и модели.',
                       'Отсутствующие исходные векторы ЛР1/ЛР4 нельзя сравнить задним числом: проверяются сохранённые рекорды, все fitness/истории и допустимость вновь полученных векторов.',
                       'ТрекB, согласование проектаЛР4 и успешная защита не являются утверждением о выполненных процедурах преподавателя.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',action='store_true',help='Повторить все560 записанных запусков, не меняя исходные результаты')
    parser.add_argument('--output',type=Path,default=ROOT/'verification.json',help='Путь протокола проверки')
    args=parser.parse_args()
    result=run_checks(ROOT,args.replay)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['variant','audit_mode','total_runs','unit_tests_passed','combined_pdf_pages']},ensure_ascii=False))


if __name__=='__main__':main()
