"""Protect the numerical display contract and exclusion of structural diagrams."""
import copy

import pytest

from policy_cce_repro.offline import DataError
from policy_cce_repro.paper import (
    build_paper, compare_display, display_contract, preview_tex,
    validate_figure_inventory,
)
from policy_cce_repro.paper_common import numeric_rows, table, write_text_new


def example(body):
    return table('A concise result caption.','tab:example','lrr',
                 'Outcome & Change & 95\\% CI\\\\\n\\midrule\n'+body)


def test_signed_effects_and_interval_endpoints_remain_distinct():
    tex=example('Profit & $+3.78$ & $[+3.75,\\,+3.81]$\\\\\n'
                'Assignment & $-0.02$ & $[-0.02,\\,-0.01]$\\\\')
    assert numeric_rows(tex)==[['3.78','3.75','3.81'],['-0.02','-0.02','-0.01']]


def test_range_dash_does_not_change_positive_standard_errors_to_negatives():
    tex=example('$(4,6)$ & 0.83--2.24 & 0.00--0.83 \\\\')
    assert numeric_rows(tex)==[['4','6','0.83','2.24','0.00','0.83']]


def test_geometry_commands_are_not_reported_as_results():
    tex=example('Orders with $\\geq3$ suppliers & $+9.66$ & $[+9.50,\\,+9.81]$\\\\[3pt]')
    assert numeric_rows(tex)==[['9.66','9.50','9.81']]


def test_two_panel_table_retains_both_sets_of_values():
    tex=example('Base & 1.00 & 1.00\\\\')+example('AUC2 vs AUC1 & 52 & 2\\\\')
    assert numeric_rows(tex)==[['1.00','1.00'],['52','2']]


def test_display_contract_keeps_mixed_definition_result_table_and_excludes_diagrams():
    contract,_=display_contract()
    assert len(contract['tables'])==13 and len(contract['figures'])==4
    assert 'tab:parameter_settings' in contract['tables']
    assert set(contract['excluded_diagrams']).isdisjoint(contract['figures'])
    assert len(contract['excluded_diagrams'])==4


def test_display_comparison_detects_rounding_drift():
    tex=example('Profit & 3.78 & 3.81\\\\')
    contract={'tables':{'tab:example':{'expected_numeric_rows':[['3.78','3.81']]}},'figures':{}}
    assert compare_display({'tab:example':{'latex':tex}},{},contract)==[]
    changed=tex.replace('3.78','3.8')
    assert compare_display({'tab:example':{'latex':changed}},{},contract)[0]['kind']=='printed_numeric_rows'


def test_scope_comparison_rejects_extra_structural_figure():
    contract={'tables':{},'figures':{}}
    errors=compare_display({}, {'fig:predeployment_framework':{}},contract)
    assert errors[0]['kind']=='figure_scope'


def test_preview_keeps_paper_numbers_without_embedding_structural_assets():
    tex=example('Profit & 3.78 & 3.81\\\\')
    contract={'tables':{'tab:example':{'number':13}},'figures':{}}
    preview=preview_tex({'tab:example':{'latex':tex}},{},contract)
    assert r'\setcounter{table}{12}' in preview
    assert r'\begin{table}' not in preview
    assert 'system.png' not in preview and 'cce_matrix_visual' not in preview


def test_text_output_is_never_overwritten(tmp_path):
    target=tmp_path/'tables/t.tex'
    write_text_new(target,'preserved')
    with pytest.raises(FileExistsError):write_text_new(target,'new')
    assert target.read_text()=='preserved'


def test_input_output_rejection_precedes_statistics(tmp_path):
    class RejectingDataset:
        def validate_output_dir(self,output):raise DataError('overlap')
        def verify_all(self):raise AssertionError('must not read or compute')
    with pytest.raises(DataError,match='overlap'):build_paper(RejectingDataset(),tmp_path)


@pytest.fixture
def labelled_display():
    """A declared table with units in the header and independent numeric values."""
    tex = table('A concise result caption.', 'tab:example', 'lrrr',
                r'Outcome & Change (\%) & Time (min) & Difference (pp)\\' + '\n'
                + r'\midrule' + '\n' + r'Profit & 3.78 & 4.25 & 0.02\\')
    contract = {'tables': {'tab:example': {
        'caption': 'A concise result caption.',
        'expected_numeric_rows': [['3.78', '4.25', '0.02']],
        'expected_unit_tokens': {'percent': 1, 'percentage_points': 1, 'minutes': 1},
    }}, 'figures': {}}
    assert compare_display({'tab:example': {'latex': tex}}, {}, contract) == []
    return tex, contract


@pytest.mark.parametrize('replacement', [
    r'\label{tab:unrelated}',
    '',
    r'\label{tab:example}\label{tab:example}',
])
def test_internal_table_label_must_appear_exactly_once(labelled_display, replacement):
    tex, contract = labelled_display
    changed = tex.replace(r'\label{tab:example}', replacement)
    assert numeric_rows(changed) == numeric_rows(tex)
    errors = compare_display({'tab:example': {'latex': changed}}, {}, contract)
    assert [e['kind'] for e in errors] == ['internal_table_label']


@pytest.mark.parametrize('replacement', [r'\caption{A different result.}', ''])
def test_caption_change_or_omission_fails_even_when_numbers_match(labelled_display, replacement):
    tex, contract = labelled_display
    changed = tex.replace(r'\caption{A concise result caption.}', replacement)
    assert numeric_rows(changed) == numeric_rows(tex)
    errors = compare_display({'tab:example': {'latex': changed}}, {}, contract)
    assert [e['kind'] for e in errors] == ['table_caption']


def test_caption_whitespace_and_revision_wrapper_are_presentation_only(labelled_display):
    tex, contract = labelled_display
    changed = tex.replace(r'\caption{A concise result caption.}',
                          '\\RevisionCaption{A concise\n   result caption.}')
    assert compare_display({'tab:example': {'latex': changed}}, {}, contract) == []


@pytest.mark.parametrize('original,replacement', [
    (r'Change (\%)', 'Change (pp)'),
    ('Time (min)', 'Time (s)'),
    ('Difference (pp)', r'Difference (\%)'),
    ('3.78', r'3.78\%'),
])
def test_unit_drift_fails_even_when_printed_numbers_match(labelled_display, original, replacement):
    tex, contract = labelled_display
    changed = tex.replace(original, replacement)
    assert numeric_rows(changed) == numeric_rows(tex)
    errors = compare_display({'tab:example': {'latex': changed}}, {}, contract)
    assert [e['kind'] for e in errors] == ['table_units']


def test_caption_and_notes_unit_mentions_do_not_change_table_unit_counts(labelled_display):
    tex, contract = labelled_display
    changed = tex.replace(r'\end{table}',
                          r'\par Notes: min, pp, and \% explain the units.\end{table}')
    assert compare_display({'tab:example': {'latex': changed}}, {}, contract) == []


@pytest.fixture
def figure_contract():
    names = ('solver_frontier.pdf', 'scalability.pdf',
             'cnc_mechanism_levels.pdf', 'cnc_policy_composition_original_style.pdf')
    return {'figures': {f'fig:result_{i}': {'asset_name': name} for i, name in enumerate(names)}}


def populate_figure_inventory(output, contract, omit=None):
    """File content is immaterial to the inventory gate; no image is rendered."""
    from pathlib import Path
    names = {str(Path(entry['asset_name']).with_suffix(extension))
             for entry in contract['figures'].values() for extension in ('.pdf', '.png')}
    for name in sorted(names - {omit}):
        write_text_new(output / 'figures' / name, 'synthetic inventory fixture')
    return names


def test_exact_four_result_pdfs_and_png_companions_pass(tmp_path, figure_contract):
    names = populate_figure_inventory(tmp_path, figure_contract)
    assert len(names) == 8
    assert validate_figure_inventory(tmp_path, figure_contract) is None


@pytest.mark.parametrize('missing', ['solver_frontier.pdf', 'solver_frontier.png'])
def test_missing_result_or_png_companion_fails(tmp_path, figure_contract, missing):
    populate_figure_inventory(tmp_path, figure_contract, omit=missing)
    with pytest.raises(DataError, match='declared result figures'):
        validate_figure_inventory(tmp_path, figure_contract)


@pytest.mark.parametrize('extra', ['system.png', 'cce_matrix.pdf', 'nested/route_capacity_model.png'])
def test_extra_structural_file_fails_even_if_all_results_are_present(tmp_path, figure_contract, extra):
    populate_figure_inventory(tmp_path, figure_contract)
    write_text_new(tmp_path / 'figures' / extra, 'excluded structural fixture')
    with pytest.raises(DataError, match='declared result figures'):
        validate_figure_inventory(tmp_path, figure_contract)


def test_absent_figure_directory_cannot_pass_inventory(tmp_path, figure_contract):
    with pytest.raises(DataError, match='declared result figures'):
        validate_figure_inventory(tmp_path, figure_contract)
