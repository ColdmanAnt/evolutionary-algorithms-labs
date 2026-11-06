"""Регрессии целостности комплекта: повреждённые данные и устаревшие PDF."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, NameObject
from verify_bundle import ROOT, validate_markdown_links, validate_pdfs, validate_series, validate_variant


def pdf_fixture(destination):
    (destination/'reports').mkdir()
    shutil.copy2(ROOT/'all_reports.pdf',destination/'all_reports.pdf')
    for lab in range(1,5):
        folder=destination/f'lab{lab}';folder.mkdir()
        source=ROOT/f'lab{lab}'/'report_content.json'
        shutil.copy2(source,folder/'report_content.json')
        shutil.copy2(ROOT/'reports'/f'lab{lab}_report.pdf',destination/'reports'/f'lab{lab}_report.pdf')
        for section in json.loads(source.read_text())['sections']:
            if section.get('image'):
                target=folder/section['image'];target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source.parent/section['image'],target)


class IntegrityTests(unittest.TestCase):
    def test_variant_matches_all_configs(self):
        self.assertEqual(validate_variant(ROOT)['V'],10)
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            shutil.copy2(ROOT/'variant.json',root/'variant.json')
            for i in [1,3]:
                (root/f'lab{i}').mkdir();shutil.copy2(ROOT/f'lab{i}'/'config.json',root/f'lab{i}'/'config.json')
            wrong=json.loads((root/'lab3/config.json').read_text());wrong['variant']=16
            (root/'lab3/config.json').write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError,'другой вариант'):validate_variant(root)

    def test_invalid_variant_rejected_with_optimization_enabled(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);wrong=json.loads((ROOT/'variant.json').read_text());wrong['N']=True
            (root/'variant.json').write_text(json.dumps(wrong))
            code='from pathlib import Path; from verify_bundle import validate_variant; validate_variant(Path(__import__("sys").argv[1]))'
            run=subprocess.run([sys.executable,'-O','-c',code,str(root)],cwd=ROOT/'tools',capture_output=True,text=True)
            self.assertNotEqual(run.returncode,0)
            self.assertIn('N должен быть целым',run.stderr)

    def test_links_reject_missing_and_outside_targets(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);readme=root/'README.md'
            readme.write_text('[missing](missing.pdf)')
            with self.assertRaisesRegex(ValueError,'Неверная ссылка'):validate_markdown_links(root)
            outside=root.parent/'outside.md'
            readme.write_text('[outside](../outside.md)')
            with self.assertRaisesRegex(ValueError,'Неверная ссылка'):validate_markdown_links(root)
            readme.write_text('[self](README.md) [web](https://github.com)')
            self.assertEqual(validate_markdown_links(root),1)

    def test_saved_pdf_sources_and_combined_pages_match(self):
        result=validate_pdfs(ROOT)
        self.assertEqual(sum(result['individual_pages']),result['combined_pages'])
        self.assertTrue(result['source_and_asset_hashes_verified'])

    def test_stale_report_json_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);pdf_fixture(root)
            p=root/'lab1/report_content.json';p.write_bytes(p.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'PDF устарел относительно report_content'):validate_pdfs(root)

    def test_stale_plot_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);pdf_fixture(root)
            p=root/'lab1/results/convergence.png';p.write_bytes(p.read_bytes()+b'changed asset')
            with self.assertRaisesRegex(ValueError,'PDF устарел относительно рисунков'):validate_pdfs(root)

    def test_same_page_count_does_not_allow_wrong_combined_order(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);pdf_fixture(root)
            original=PdfReader(root/'all_reports.pdf');writer=PdfWriter()
            for page in reversed(original.pages):writer.add_page(page)
            writer.add_metadata(dict(original.metadata or {}));writer.write(root/'all_reports.pdf')
            with self.assertRaisesRegex(ValueError,'страницы общего PDF отличаются'):validate_pdfs(root)

    def test_saved_series_reject_changed_budget(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'lab1/results').mkdir(parents=True)
            with (ROOT/'lab1/results/runs.csv').open() as f:rows=list(csv.DictReader(f))
            rows[0]['evaluations']='23999'
            with (root/'lab1/results/runs.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
            with self.assertRaisesRegex(ValueError,'неверный бюджет'):validate_series(root)

    def test_combined_pdf_cannot_drop_a_plot_while_preserving_text(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);pdf_fixture(root)
            original=PdfReader(root/'all_reports.pdf');writer=PdfWriter();changed=False
            for page in original.pages:
                if not changed:
                    content=ContentStream(page.get_contents(),original)
                    operations=[operation for operation in content.operations if operation[1]!=b'Do']
                    if len(operations)!=len(content.operations):
                        content.operations=operations
                        page[NameObject('/Contents')]=content;changed=True
                writer.add_page(page)
            self.assertTrue(changed)
            writer.add_metadata(dict(original.metadata or {}));writer.write(root/'all_reports.pdf')
            with self.assertRaisesRegex(ValueError,'графика/шрифты/содержимое'):validate_pdfs(root)


if __name__=='__main__':unittest.main()
