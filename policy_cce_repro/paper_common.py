"""Small presentation helpers; no scientific calculations or network access."""
from __future__ import annotations

from pathlib import Path
import re

from .offline import DataError, require


def table(caption, label, columns, body, notes='', wide=False):
    """Return a LaTeX float; body contains its header, midrules and data rows."""
    require(isinstance(body, str), 'Table body must be LaTeX text')
    environment = 'table*' if wide else 'table'
    width = r'\textwidth' if wide else r'\columnwidth'
    return (
        f'\\begin{{{environment}}}[!htbp]\n\\centering\n'
        f'\\caption{{{caption}}}\n\\label{{{label}}}\n'
        '\\footnotesize\n\\setlength{\\tabcolsep}{2pt}\n'
        f'\\begin{{tabular*}}{{{width}}}{{@{{\\extracolsep{{\\fill}}}}{columns}@{{}}}}\n'
        '\\toprule\n' + body.rstrip() + '\n\\bottomrule\n\\end{tabular*}\n'
        + (f'\\par\\smallskip{{\\footnotesize {notes}\\par}}\n' if notes else '')
        + f'\\end{{{environment}}}\n'
    )


def numeric_rows(latex):
    """Extract printed numbers from table data rows, excluding header/layout sizes.

    Used only to compare formatted output against a frozen manuscript display
    contract. These strings are never inputs to a scientific calculation.
    """
    rows = []
    inside = False
    data_started = False
    for raw in latex.splitlines():
        line = raw.strip()
        if re.search(r'\\begin\{tabular\*?\}', line):
            inside, data_started = True, False
            continue
        if re.search(r'\\end\{tabular\*?\}', line):
            inside = False
            continue
        if not inside:
            continue
        if line.startswith(r'\midrule'):
            data_started = True
            continue
        if not data_started or '&' not in line:
            continue
        if line.startswith((r'\multicolumn', r'\cmidrule', r'\addlinespace')) or r'\multicolumn' in line:
            continue
        if re.match(r'^(Outcome|Method|Mechanism|Condition|Family|Setting|Profit comparison)\s*&', line):
            continue
        line = re.sub(r'\\\\(?:\[[^]]*\])?\s*$', '', line)
        # Formatting parameters in an inline shortstack are not result values.
        line = re.sub(r'\\shortstack(?:\[[^]]*\])?', '', line)
        line = re.sub(r'(?<=\d)--(?=\d)', ' ', line)
        line = re.sub(r'\\geq\s*3', 'three', line)
        tokens = re.findall(r'(?<![A-Za-z0-9.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?', line)
        if tokens:
            rows.append([token.lstrip('+').replace(',', '') for token in tokens])
    return rows


def write_text_new(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(text)


def table_unit_tokens(latex):
    blocks = re.findall(r'\\begin\{tabular\*?\}([\s\S]*?)\\end\{tabular\*?\}', latex)
    text = '\n'.join(blocks)
    return {'percent':len(re.findall(r'\\%',text)),
            'percentage_points':len(re.findall(r'\bpp\b',text)),
            'minutes':len(re.findall(r'\bmin\b',text))}


def verified_report_inputs(dataset, audit_report, outcomes_report=None):
    require(audit_report.get('status') == 'pass' and audit_report.get('matched_job_count') == 276,
            'Paper figures require all 276 independent-audit replays to pass')
    require(audit_report.get('mismatch_field_count') == 0 and audit_report.get('failed_job_count') == 0,
            'Paper statistics contain unresolved audit differences')
    jobs = [entry['job_id'] for entry in audit_report['jobs']]
    require(len(jobs) == len(set(jobs)) == 276 and set(jobs) <= set(dataset.jobs), 'Audit replay scope mismatch')
    if outcomes_report is not None:
        require(outcomes_report.get('status') == 'pass' and outcomes_report.get('case_count') == 96
                and outcomes_report.get('mismatch_count') == 0, 'CNC outcome replay has unresolved differences')
