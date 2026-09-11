"""Parameter displays: full-grid accounting, immutable q, and numeric lineage."""
from __future__ import annotations

from collections import Counter
import copy
from dataclasses import asdict
from itertools import product
import socket
import subprocess

import pytest

from cmfg_cce.evaluation.independent_audit import distribution_hash
from cmfg_cce.experiments.revision_full_v1_spec import fractional_robustness_settings
from policy_cce_repro.offline import DataError, Dataset
from policy_cce_repro.paper_sensitivity import (
    CONDITIONS, FAMILY, MECHANISMS, METRICS, SEEDS, SELECTOR, SETTINGS, STRESS,
    build_sensitivity,
)


class FakeDataset:
    """Model already hash-validated Dataset results; never fake calculations."""
    validate_output_dir = Dataset.validate_output_dir

    def __init__(self, root):
        self.root = root
        self.jobs, self.results, self.references = {}, {}, {}
        self.reads = Counter()

    def result(self, job_id):
        self.reads[job_id] += 1
        return copy.deepcopy(self.results[job_id])

    def reference(self, name):
        return copy.deepcopy(self.references[name])


@pytest.fixture(autouse=True)
def no_external_execution(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('Sensitivity displays must remain local and calculation-only')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(subprocess, 'Popen', denied)


@pytest.fixture
def data(tmp_path):
    data = FakeDataset(tmp_path / 'input')
    q = {'support': [['A1'] * 4], 'probabilities': [1.0]}
    q['q_hash'] = distribution_hash(q['support'], q['probabilities'])
    rows, grid = [], {}
    for ci, condition in enumerate(CONDITIONS):
        for si, setting in enumerate(SETTINGS):
            for seed, (mi, mechanism) in product(SEEDS, enumerate(MECHANISMS)):
                jid = f'job-{ci}-{si}-{seed}-{mi}'
                profit = 100 + ci * 20 + si * 3 + seed + (2 if mi >= 2 else 0)
                if mi % 2:
                    profit += (10, -5, 0)[seed] * (2 if mi == 3 else 1)
                metrics = dict(zip(METRICS, (20 + ci + si + mi + seed / 10,
                                             profit, 0.5 + 0.01 * mi + 0.001 * seed,
                                             0.8 - 0.01 * si - 0.001 * seed), strict=True))
                row = {'job_id': jid, 'condition': condition, 'setting': setting,
                       'seed': seed, 'mechanism': mechanism, 'selector': SELECTOR,
                       'q_hash': q['q_hash'], **metrics}
                rows.append(row)
                grid[condition, setting, seed, mechanism] = row
                metadata = {'mechanism': mechanism, 'seed': seed, 'variant': condition + '__' + setting}
                data.jobs[jid] = {'family': FAMILY, 'metadata': metadata}
                data.results[jid] = {
                    'job_id': jid, 'family': FAMILY, 'N': 4, 'J': 6, 'policy_ids': ['A1'],
                    'train_rollouts': 200, 'audit_rollouts': 2000, 'outcome_rollouts': 500,
                    **metadata, 'condition': condition, 'robustness_setting': setting,
                    'distributions': {SELECTOR: copy.deepcopy(q)},
                    'formal_audit': {'distributions': {SELECTOR: {'q_hash_before': q['q_hash'],
                                                                  'q_hash_after': q['q_hash']}}},
                    'outcome_evaluation': {'distributions': {SELECTOR: {'q_hash': q['q_hash'],
                        'metrics': {m: {'estimate': v} for m, v in metrics.items()}}}},
                }
    definitions = [asdict(s) for s in fractional_robustness_settings()]
    contrasts, overall = [], []
    for low, high in ((0, 1), (2, 3)):
        pairs = []
        for condition, setting, seed in product(CONDITIONS, SETTINGS, SEEDS):
            a, b = (grid[condition, setting, seed, MECHANISMS[m]] for m in (low, high))
            pairs.append({'condition': condition, 'setting': setting, 'seed': seed,
                          'source_jobs': [a['job_id'], b['job_id']],
                          'difference': b[METRICS[1]] - a[METRICS[1]]})
        differences = [r['difference'] for r in pairs]
        stats = {'n': 54, 'positive_n': 18, 'negative_n': 18, 'zero_n': 18,
                 'mean': sum(differences) / 54, 'min': min(differences), 'max': max(differences)}
        contrasts.append({'contrast': f'AUC{high+1}-AUC{low+1}', **stats, 'pairs': pairs})
        overall.append({'contrast': f'M{high+1}-M{low+1}', 'metric': METRICS[1], **stats})
    decision = []
    for setting, mi in product(('base', 'fraction_07'), range(2)):
        selected = [grid[STRESS, setting, s, MECHANISMS[mi]] for s in SEEDS]
        decision.append({'condition': STRESS, 'setting': setting, 'mechanism': f'AUC{mi+1}',
                         'source_jobs': [r['job_id'] for r in selected],
                         'means': {m: sum(r[m] for r in selected) / 3 for m in METRICS}})
    data.references = {
        'sensitivity-descriptive-summary.json': {'parameter_robustness': {
            'verified_job_count': 216, 'settings': list(SETTINGS), 'settings_definition': definitions,
            'row_level': rows, 'contrasts_overall': overall}},
        'case_summary_table_evidence.json': {'parameter': {'definitions': definitions, 'profit_contrasts': contrasts}},
        'sensitivity_table_evidence.json': {'parameter_decision': decision},
    }
    return data


def build(data, tmp_path):
    return build_sensitivity(data, tmp_path / 'new-output')


def test_complete_scope_is_read_once_and_inputs_unchanged(data, tmp_path):
    before = copy.deepcopy((data.jobs, data.results, data.references))
    result = build(data, tmp_path)
    assert result['status'] == 'pass'
    assert result['mismatches'] == []
    assert result['verified_jobs'] == 216
    assert result['comparison_count'] == 2428
    assert len(data.reads) == 216 and set(data.reads.values()) == {1}
    assert (data.jobs, data.results, data.references) == before
    assert not (tmp_path / 'new-output').exists()
    assert set(result['tables']) == {'tab:parameter_settings', 'tab:parameter_decision'}


def test_all_positive_negative_and_zero_pairs_are_retained(data, tmp_path):
    result = build(data, tmp_path)
    for row in result['evidence']['profit_contrasts']:
        assert row['n'] == len(row['pairs']) == 54
        assert (row['positive_n'], row['negative_n'], row['zero_n']) == (18, 18, 18)
        assert len({tuple(p['source_jobs']) for p in row['pairs']}) == 54
        assert {p['seed'] for p in row['pairs']} == {0, 1, 2}
        assert {p['setting'] for p in row['pairs']} == set(SETTINGS)
    all_sources = [j for row in result['evidence']['profit_contrasts'] for p in row['pairs'] for j in p['source_jobs']]
    assert len(all_sources) == len(set(all_sources)) == 216


def test_display_uses_seed_means_then_converts_rates_to_percent(data, tmp_path):
    rows = build(data, tmp_path)['tables']['tab:parameter_decision']['rows']
    assert [(r['setting'], r['mechanism']) for r in rows] == [
        ('base', 'AUC1'), ('base', 'AUC2'), ('fraction_07', 'AUC1'), ('fraction_07', 'AUC2')]
    assert rows[0]['means'][METRICS[0]] == pytest.approx(21.1)
    assert rows[0]['means'][METRICS[1]] == pytest.approx(121)
    assert rows[0]['means'][METRICS[2]] == pytest.approx(.501)
    assert rows[0]['display_values'][METRICS[2]] == pytest.approx(50.1)
    assert rows[0]['display_values'][METRICS[3]] == pytest.approx(79.9)
    assert all(len(r['source_jobs']) == 3 for r in rows)


def test_table_labels_short_captions_panels_and_formatting(data, tmp_path):
    tables = build(data, tmp_path)['tables']
    scenario = tables['tab:parameter_settings']['latex']
    decision = tables['tab:parameter_decision']['latex']
    assert scenario.count(r'\begin{tabular*}') == 2
    assert r'\caption{Parameter scenarios and profit comparisons.}' in scenario
    assert r'\label{tab:parameter_settings}' in scenario
    assert 'Joint 7 & 1.50 & 1.50 & 0.85 & 0.80' in scenario
    assert 'AUC2 vs AUC1 & 18 & 18' in scenario
    assert r'\caption{Payment-rule outcomes under combined pressure.}' in decision
    assert r'\label{tab:parameter_decision}' in decision
    assert 'Base & AUC1 & 21.10 & 121.00 & 50.10 & 79.90' in decision


def test_references_validate_but_never_supply_display_numbers(data, tmp_path):
    original = build(data, tmp_path)
    data.references['sensitivity_table_evidence.json']['parameter_decision'][0]['means'][METRICS[0]] += 50
    data.references['sensitivity-descriptive-summary.json']['parameter_robustness']['row_level'][0][METRICS[1]] -= 10
    result = build(data, tmp_path)
    assert result['status'] == 'mismatch'
    assert len(result['mismatches']) == 2
    assert result['tables'] == original['tables']


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), True, '5'])
def test_nonfinite_or_nonnumeric_metric_rejected(data, tmp_path, value):
    r = next(iter(data.results.values()))
    r['outcome_evaluation']['distributions'][SELECTOR]['metrics'][METRICS[0]]['estimate'] = value
    with pytest.raises(DataError, match='nonnumeric parameter metric'):
        build(data, tmp_path)


@pytest.mark.parametrize('value', [-0.001, 1.001])
def test_rates_remain_fractions_before_display(data, tmp_path, value):
    r = next(iter(data.results.values()))
    r['outcome_evaluation']['distributions'][SELECTOR]['metrics'][METRICS[2]]['estimate'] = value
    with pytest.raises(DataError, match='outside'):
        build(data, tmp_path)


@pytest.mark.parametrize('stage,field', [('distributions', 'q_hash'),
    ('formal_audit', 'q_hash_before'), ('formal_audit', 'q_hash_after'), ('outcome_evaluation', 'q_hash')])
def test_every_stage_must_preserve_the_same_q(data, tmp_path, stage, field):
    r = next(iter(data.results.values()))
    container = r[stage] if stage == 'distributions' else r[stage]['distributions']
    container[SELECTOR][field] = 'bad-hash'
    with pytest.raises(DataError, match='distribution changed'):
        build(data, tmp_path)


@pytest.mark.parametrize('field,value', [('N', 5), ('J', 8), ('train_rollouts', 100),
                                       ('audit_rollouts', 500), ('outcome_rollouts', 200)])
def test_dimensions_and_sample_counts_are_frozen(data, tmp_path, field, value):
    next(iter(data.results.values()))[field] = value
    with pytest.raises(DataError, match='dimensions|rollout counts'):
        build(data, tmp_path)


def test_missing_job_rejected(data, tmp_path):
    data.jobs.pop(next(iter(data.jobs)))
    with pytest.raises(DataError, match='216'):
        build(data, tmp_path)


def test_duplicate_grid_cell_rejected(data, tmp_path):
    first, second = list(data.jobs)[:2]
    data.jobs[second] = copy.deepcopy(data.jobs[first])
    data.results[second] = copy.deepcopy(data.results[first])
    data.results[second]['job_id'] = second
    with pytest.raises(DataError, match='Duplicate parameter grid cell'):
        build(data, tmp_path)


def test_out_of_scope_seed_rejected_even_when_manifest_agrees(data, tmp_path):
    jid = next(iter(data.jobs))
    data.jobs[jid]['metadata']['seed'] = 3
    data.results[jid]['seed'] = 3
    with pytest.raises(DataError, match='missing or contains unexpected cells'):
        build(data, tmp_path)


def test_unexpected_selector_rejected(data, tmp_path):
    r = next(iter(data.results.values()))
    r['distributions']['different_objective'] = copy.deepcopy(r['distributions'][SELECTOR])
    with pytest.raises(DataError, match='Unexpected parameter selector'):
        build(data, tmp_path)


def test_invalid_distribution_support_rejected(data, tmp_path):
    q = next(iter(data.results.values()))['distributions'][SELECTOR]
    q['support'] *= 2
    q['probabilities'] = [0.5, 0.5]
    with pytest.raises(DataError, match='distribution support'):
        build(data, tmp_path)


def test_reference_scope_cannot_silently_drop_a_pair(data, tmp_path):
    data.references['case_summary_table_evidence.json']['parameter']['profit_contrasts'][0]['pairs'].pop()
    with pytest.raises(DataError, match='pair scope mismatch'):
        build(data, tmp_path)


def test_output_overlap_rejected_before_any_result_is_read(data):
    with pytest.raises(DataError, match='must not overlap'):
        build_sensitivity(data, data.root / 'output')
    assert not data.reads
