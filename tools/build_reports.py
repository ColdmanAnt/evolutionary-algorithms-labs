"""Сборка четырёх PDF из проверенных результатов и объединение отчётов."""
from pathlib import Path
import argparse
import json
import re
import html
import hashlib
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader, PdfWriter

ROOT=Path(__file__).resolve().parents[1]

def setup_fonts():
    import matplotlib
    folder=Path(matplotlib.get_data_path())/'fonts'/'ttf'
    pdfmetrics.registerFont(TTFont('Body',str(folder/'DejaVuSerif.ttf')))
    pdfmetrics.registerFont(TTFont('BodyBold',str(folder/'DejaVuSerif-Bold.ttf')))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='Body',boldItalic='BodyBold')

def safe(value):
    # ReportLab понимает XML-подобную разметку, входные строки трактуем как текст.
    return html.escape(str(value)).replace('\n','<br/>')

def build(lab):
    source=ROOT/f'lab{lab}'/'report_content.json'
    if not source.exists():raise FileNotFoundError(source)
    data=json.loads(source.read_text(encoding='utf-8'))
    target=ROOT/'reports'/f'lab{lab}_report.pdf'
    doc=SimpleDocTemplate(str(target),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,
                          topMargin=20*mm,bottomMargin=18*mm,
                          title=f'Лабораторная работа № {lab}. Вариант 10',author='Холодков Антон Евгеньевич')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TextRu',fontName='Body',fontSize=9.5,leading=14,spaceAfter=7))
    styles.add(ParagraphStyle(name='HeadingRu',fontName='BodyBold',fontSize=12,leading=16,spaceBefore=13,spaceAfter=8,keepWithNext=True))
    styles.add(ParagraphStyle(name='TitleRu',fontName='BodyBold',fontSize=19,leading=25,spaceAfter=12))
    styles.add(ParagraphStyle(name='SmallRu',fontName='Body',fontSize=8,leading=11,spaceAfter=6,textColor=colors.HexColor('#675d50')))
    styles.add(ParagraphStyle(name='CellRu',fontName='Body',fontSize=7.8,leading=10))
    styles.add(ParagraphStyle(name='CellHead',fontName='BodyBold',fontSize=7.6,leading=10,textColor=colors.white))
    story=[Paragraph('НИЯУ МИФИ · Эволюционное программирование и генетические алгоритмы',styles['SmallRu']),
           Spacer(1,7*mm),Paragraph(safe(data['title']),styles['TitleRu']),
           Paragraph(safe(data.get('task','')),styles['TextRu']),
           Paragraph('Холодков Антон Евгеньевич<br/>Группа 517 · N = 28 · Q = 1 · Y = 26 · V = 10<br/>6 ноября 2026 года',styles['TextRu']),
           Spacer(1,3*mm)]
    asset_hashes={}
    for section in data['sections']:
        story.append(Paragraph(safe(section['heading']),styles['HeadingRu']))
        for p in section.get('paragraphs',[]):story.append(Paragraph(safe(p),styles['TextRu']))
        t=section.get('table')
        if t:
            headers=t['headers'];rows=t['rows'];n=len(headers)
            # Ограничиваем влияние длинных предложений на ширину первого столбца.
            weights=[]
            for c in range(n):
                strings=[str(headers[c])]+[str(row[c]) for row in rows if c<len(row)]
                weights.append(min(24,max(7,max(len(s) for s in strings)**.70*2)))
            widths=[doc.width*w/sum(weights) for w in weights]
            formatted=[[Paragraph(safe(h),styles['CellHead']) for h in headers]]
            formatted += [[Paragraph(safe(v),styles['CellRu']) for v in row] for row in rows]
            table=Table(formatted,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#783d41')),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f5f1e9'),colors.white]),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),
                ('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),5),
                ('RIGHTPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#783d41'))]))
            story.extend([table,Spacer(1,8)])
        if section.get('image'):
            imagepath=source.parent/section['image']
            if not imagepath.exists():
                imagepath=ROOT/f'lab{lab}'/section['image']
            if not imagepath.exists():raise FileNotFoundError(imagepath)
            asset_hashes[section['image']]=hashlib.sha256(imagepath.read_bytes()).hexdigest()
            from PIL import Image as PILImage
            with PILImage.open(imagepath) as im:w,h=im.size
            factor=min(doc.width/w,108*mm/h)
            story.extend([Image(str(imagepath),width=w*factor,height=h*factor),Spacer(1,8)])
    story.append(Paragraph('Источник задания и состав комплекта',styles['HeadingRu']))
    story.append(Paragraph('Душкин Р. В. Дисциплина «Эволюционное программирование и генетические алгоритмы». Комплекс лабораторных работ. Методические материалы, предоставленные пользователем, 24 страницы. Индивидуальный вариант вычислен по формуле раздела 1.5. Исходный код, конфигурации, входные данные и необработанные результаты находятся в папке соответствующей работы.',styles['TextRu']))
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#d7d0c5'));canvas.setLineWidth(.4)
        canvas.line(doc.leftMargin,A4[1]-13*mm,A4[0]-doc.rightMargin,A4[1]-13*mm)
        canvas.setFont('Body',7);canvas.setFillColor(colors.HexColor('#675d50'))
        canvas.drawString(doc.leftMargin,10*mm,f'Холодков А. Е. · 517 · ЛР{lab} · Вариант 10')
        canvas.drawRightString(A4[0]-doc.rightMargin,10*mm,str(doc.page));canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    # Связать PDF с точным JSON и рисунками, чтобы проверки ловили устаревший отчёт.
    original=PdfReader(target)
    sealed=PdfWriter()
    sealed.append(original)
    sealed.add_metadata({str(k):str(v) for k,v in (original.metadata or {}).items()})
    sealed.add_metadata({
        '/ReportContentSHA256':hashlib.sha256(source.read_bytes()).hexdigest(),
        '/ReportAssetsSHA256':hashlib.sha256(json.dumps(asset_hashes,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
    })
    sealed.write(target)
    return target

def main():
    setup_fonts();(ROOT/'reports').mkdir(exist_ok=True)
    pdfs=[build(i) for i in range(1,5)]
    writer=PdfWriter()
    for i,p in enumerate(pdfs,1):writer.append(str(p),outline_item=f'Лабораторная работа № {i}')
    writer.add_metadata({'/Title':'Все лабораторные работы. Вариант 10','/Author':'Холодков Антон Евгеньевич'})
    writer.add_metadata({'/SourceReportsSHA256':hashlib.sha256(json.dumps(
        [hashlib.sha256(p.read_bytes()).hexdigest() for p in pdfs]).encode()).hexdigest()})
    writer.write(ROOT/'all_reports.pdf')
    for p in pdfs+[ROOT/'all_reports.pdf']:print(p,len(PdfReader(p).pages),'pages')

if __name__=='__main__':main()
