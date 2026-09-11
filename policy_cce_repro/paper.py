"""Generate only numerical paper tables and experiment-result figures offline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

from .offline import DataError, require
from .paper_common import numeric_rows, table_unit_tokens, verified_report_inputs, write_text_new


def display_contract():
    locations = (Path(__file__).resolve().parent.parent / 'metadata/paper-layout-v2.json',
                 Path(sys.prefix) / 'share/policy-cce-reproducibility/metadata/paper-layout-v2.json')
    for path in locations:
        if path.is_file():
            value = json.loads(path.read_text())
            require(value.get('schema') == 'policy_cce_paper_display_contract_v1', 'Unknown paper display contract')
            return value, hashlib.sha256(path.read_bytes()).hexdigest()
    raise DataError('Packaged paper display contract is missing')


def compare_display(tables, figures, contract):
    errors = []
    if set(tables) != set(contract['tables']):
        errors.append({'kind':'table_scope', 'actual':sorted(tables), 'expected':sorted(contract['tables'])})
    if set(figures) != set(contract['figures']):
        errors.append({'kind':'figure_scope', 'actual':sorted(figures), 'expected':sorted(contract['figures'])})
    for label, entry in tables.items():
        if label not in contract['tables']:
            continue
        actual = numeric_rows(entry['latex'])
        declaration = contract['tables'][label]
        expected = declaration['expected_numeric_rows']
        if actual != expected:
            errors.append({'kind':'printed_numeric_rows', 'label':label, 'actual':actual, 'expected':expected})
        labels = re.findall(r'\\label\{([^}]+)\}',entry['latex'])
        if labels != [label]:
            errors.append({'kind':'internal_table_label','label':label,'actual':labels})
        caption = re.search(r'\\(?:RevisionCaption|caption)\{([^}]+)\}',entry['latex'])
        if 'caption' in declaration and (not caption or ' '.join(caption[1].split()) != ' '.join(declaration['caption'].split())):
            errors.append({'kind':'table_caption','label':label,'actual':caption[1] if caption else None,'expected':declaration['caption']})
        if 'expected_unit_tokens' in declaration and table_unit_tokens(entry['latex']) != declaration['expected_unit_tokens']:
            errors.append({'kind':'table_units','label':label,'actual':table_unit_tokens(entry['latex']),'expected':declaration['expected_unit_tokens']})
    return errors


def validate_figure_inventory(out, contract):
    expected = {str(Path(item['asset_name']).with_suffix(ext))
                for item in contract['figures'].values() for ext in ('.pdf','.png')}
    folder = Path(out)/'figures'
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    require(actual==expected,'Figure folder must contain only the four declared result figures and their PNG companions')


def preview_tex(tables, figures, contract):
    lines = [r'\documentclass[10pt]{article}',r'\usepackage[a4paper,margin=20mm]{geometry}',
             r'\usepackage{newtxtext,newtxmath,booktabs,array,graphicx,xcolor,caption}',
             r'\usepackage{amsmath}',r'\captionsetup{font=small,labelfont=bf}',
             r'\newcommand{\RevisionCaption}[1]{\caption{#1}}',
             r'\newcommand{\TableNotes}[2][\linewidth]{\par\smallskip{\footnotesize #2\par}}',
             r'\pagestyle{plain}']
    lines.append(r'\makeatletter')
    for label,value in contract.get('reference_labels',{}).items():
        lines.append(f'\\newlabel{{{label}}}{{{{{value}}}{{0}}}}')
    lines.append(r'\makeatother')
    lines.append(r'\begin{document}')
    objects = [('table',label,entry) for label,entry in sorted(tables.items(),key=lambda p:contract['tables'][p[0]]['number'])]
    objects += [('figure',label,entry) for label,entry in sorted(figures.items(),key=lambda p:contract['figures'][p[0]]['number'])]
    for index,(kind,label,entry) in enumerate(objects):
        if index: lines.append(r'\clearpage')
        number = contract[kind+'s'][label]['number']
        lines.extend([r'\begin{center}',f'\\setcounter{{{kind}}}{{{number-1}}}'])
        if kind == 'table':
            latex = entry['latex']
            wide = r'\begin{table*}' in latex
            width = r'\textwidth' if wide else r'0.50\textwidth'
            latex = re.sub(r'\\begin\{table\*?\}(?:\[[^]]*\])?', '', latex)
            latex = re.sub(r'\\end\{table\*?\}', '', latex)
            lines += [f'\\begin{{minipage}}{{{width}}}',r'\captionsetup{type=table}',latex,r'\end{minipage}']
        else:
            name = contract['figures'][label]['asset_name']
            lines += [r'\begin{minipage}{\textwidth}',r'\captionsetup{type=figure}',
                      f'\\includegraphics[width=\\linewidth]{{figures/{name}}}',
                      f"\\caption{{{contract['figures'][label]['caption']}}}",r'\end{minipage}']
        lines.append(r'\end{center}')
    lines += [r'\end{document}','']
    return '\n'.join(lines)


def _json_default(value):
    if hasattr(value,'tolist'): return value.tolist()
    if isinstance(value,Path): return str(value)
    raise TypeError(f'Unsupported presentation evidence type: {type(value).__name__}')


def build_paper(dataset, output_dir, *, workers=2):
    """Recompute fixed-q statistics then generate the versioned display scope."""
    out = dataset.validate_output_dir(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    require(not any(out.iterdir()), 'Paper output directory must be new and empty')
    contract, contract_hash = display_contract()
    before = dataset.verify_all()
    from .recompute_audits import recompute_audits
    from .recompute_outcomes import recompute_outcomes
    audits = recompute_audits(dataset,out/'statistics/audits',workers=workers)
    outcomes = recompute_outcomes(dataset,out/'statistics/outcomes',workers=workers)
    verified_report_inputs(dataset,audits,outcomes)
    from .paper_benchmark import build_benchmark
    from .paper_cnc import build_cnc
    from .paper_sensitivity import build_sensitivity
    reports = {'benchmark':build_benchmark(dataset,audits,out),
               'cnc':build_cnc(dataset,audits,outcomes,out),
               'sensitivity':build_sensitivity(dataset,out)}
    tables, figures = {}, {}
    for name,report in reports.items():
        require(not set(tables)&set(report.get('tables',{})), 'Duplicate table generator')
        require(not set(figures)&set(report.get('figures',{})), 'Duplicate figure generator')
        tables.update(report.get('tables',{})); figures.update(report.get('figures',{}))
    errors = compare_display(tables,figures,contract)
    for name,report in reports.items():
        if report.get('mismatches'):
            errors.append({'kind':'numeric_evidence','builder':name,'mismatches':report['mismatches']})
    for label,entry in tables.items():
        require(label in contract['tables'], 'Out-of-scope table')
        name = label.split(':',1)[1]+'.tex'
        write_text_new(out/'tables'/name,entry['latex'])
    for label in figures:
        require(label in contract['figures'], 'Out-of-scope figure')
        require((out/'figures'/contract['figures'][label]['asset_name']).is_file(), 'Missing expected figure PDF')
    validate_figure_inventory(out,contract)
    write_text_new(out/'paper-preview.tex',preview_tex(tables,figures,contract))
    after = dataset.verify_all()
    require(before==after, 'Immutable inputs changed while generating paper artifacts')
    files = [{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,
              'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(out.rglob('*')) if p.is_file()]
    result = {'schema':'policy_cce_paper_artifacts_v1','status':'pass' if not errors else 'fail',
              'manuscript_commit':contract['manuscript_commit'],'display_contract_sha256':contract_hash,
              'data_manifest_sha256':dataset.manifest_sha256,'input_verification':after,
              'table_count':len(tables),'result_figure_count':len(figures),
              'excluded_diagrams':contract['excluded_diagrams'],'publication_channel':'GitHub only',
              'recomputed_audit_jobs':audits['matched_job_count'],'recomputed_outcome_cases':outcomes['case_count'],
              'errors':errors,'builders':reports,'files':files,
              'scope':'Offline fixed-q reanalysis and result presentation only; no new experiments or manuscript edits'}
    write_text_new(out/'paper-report.json',json.dumps(result,indent=2,sort_keys=True,allow_nan=False,default=_json_default)+'\n')
    return json.loads(json.dumps(result,default=_json_default,allow_nan=False))
