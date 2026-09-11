"""Rebuild the two parameter-sensitivity tables from archived formal results.

This module never runs the simulator, selects a distribution, or changes an
input. References check the calculations; they do not supply table values.
"""
from __future__ import annotations

from dataclasses import asdict
from itertools import product
import math
from pathlib import Path
from statistics import mean

import yaml

from cmfg_cce.evaluation.independent_audit import distribution_hash
from cmfg_cce.experiments import revision_full_v1_spec as frozen_spec
from .offline import DataError, file_sha256, require


FAMILY = 'parameter_robustness'
SELECTOR = 'platform_operating_score'
NORMAL = 'nominal__balanced__normal'
STRESS = 'high__m5_edm_intensive__high_outside_m5_g_edm'
CONDITIONS = (NORMAL, STRESS)
SEEDS = (0, 1, 2)
MECHANISMS = ('M1_price_first', 'M2_price_critical',
              'M3_delivery_first', 'M4_delivery_critical')
FACTORS = ('cost_dispersion_scale', 'alpha_scale',
           'effective_rate_scale', 'policy_coefficient_scale')
METRICS = ('payment_per_assignment', 'manufacturer_discounted_profit',
           'assignment_rate',
           'at_least_three_route_capacity_feasible_manufacturers_rate')
SETTINGS = ('base',) + tuple(f'fraction_{i:02d}' for i in range(1, 9))


def _auc(mechanism):
    return 'AUC' + str(MECHANISMS.index(mechanism) + 1)


def _setting_label(setting):
    return 'Base' if setting == 'base' else f'Joint {int(setting[-2:])}'


def _definitions():
    """Use the frozen design function and cross-check the packaged config."""
    definitions = [asdict(s) for s in frozen_spec.fractional_robustness_settings()]
    require(tuple(r['setting_id'] for r in definitions) == SETTINGS,
            'Parameter definition settings do not match the publication scope')
    source = Path(frozen_spec.__file__)
    config = source.parents[1] / 'configs' / 'revision_full_v1.yaml'
    block = yaml.safe_load(config.read_text())['parameter_robustness']
    require(block['design'] == 'resolution_iv_half_fraction_d_equals_abc_plus_base',
            'Unexpected parameter design')
    for factor in FACTORS:
        require(sorted({r[factor] for r in definitions[1:]}) == block[factor],
                'Parameter levels disagree with frozen config: ' + factor)
    require(tuple(block['operating_conditions']) == CONDITIONS and
            tuple(block['mechanisms']) == MECHANISMS and set(SEEDS) <= set(block['seeds']),
            'Parameter dimensions disagree with frozen config')
    return definitions, {
        'source_spec_sha256': file_sha256(source),
        'source_config_sha256': file_sha256(config),
        'scope_note': 'The frozen config lists five seeds; the publication scope uses seeds 0, 1, 2. No config or job is changed.',
    }


def _read_records(dataset):
    ids = sorted(j for j, spec in dataset.jobs.items() if spec['family'] == FAMILY)
    require(len(ids) == 216, 'Expected all 216 parameter results')
    records, lookup = [], {}
    for jid in ids:
        try:
            r = dataset.result(jid)
            spec = dataset.jobs[jid]['metadata']
            require(r['family'] == FAMILY and r['job_id'] == jid, 'Parameter job identity mismatch')
            require((r['N'], r['J']) == (4, 6), 'Parameter game dimensions changed')
            require((r['train_rollouts'], r['audit_rollouts'], r['outcome_rollouts']) == (200, 2000, 500),
                    'Parameter rollout counts changed')
            require(r['mechanism'] == spec['mechanism'] and r['seed'] == spec['seed'] and
                    r['variant'] == spec['variant'] == r['condition'] + '__' + r['robustness_setting'],
                    'Parameter result disagrees with its frozen job configuration')
            require(type(r['seed']) is int, 'Parameter seed must be an integer')
            key = (r['condition'], r['robustness_setting'], r['seed'], r['mechanism'])
            require(key not in lookup, 'Duplicate parameter grid cell: ' + repr(key))
            require(set(r['distributions']) == {SELECTOR}, 'Unexpected parameter selector')
            q = r['distributions'][SELECTOR]
            support, probabilities = q['support'], q['probabilities']
            require(len(support) == len(probabilities) > 0 and
                    len({tuple(s) for s in support}) == len(support),
                    'Invalid parameter distribution support')
            require(all(len(s) == 4 and set(s) <= set(r['policy_ids']) for s in support),
                    'Invalid parameter distribution policies')
            require(all(isinstance(p, (float, int)) and not isinstance(p, bool) and
                        math.isfinite(p) and p > 0 for p in probabilities) and
                    math.isclose(sum(probabilities), 1.0, rel_tol=0, abs_tol=1e-12),
                    'Invalid parameter distribution probabilities')
            q_hash = distribution_hash(support, probabilities)
            audit = r['formal_audit']['distributions'][SELECTOR]
            outcome = r['outcome_evaluation']['distributions'][SELECTOR]
            require(q_hash == q['q_hash'] == audit['q_hash_before'] == audit['q_hash_after'] == outcome['q_hash'],
                    'Parameter distribution changed between training and evaluation')
            metrics = {}
            for metric in METRICS:
                value = outcome['metrics'][metric]['estimate']
                require(type(value) in (int, float) and math.isfinite(value),
                        'Non-finite or nonnumeric parameter metric: ' + metric)
                if metric in METRICS[2:]:
                    require(0 <= value <= 1, 'Parameter rate outside [0, 1]: ' + metric)
                metrics[metric] = value
            record = {'job_id': jid, 'condition': r['condition'],
                      'setting': r['robustness_setting'], 'seed': r['seed'],
                      'mechanism': r['mechanism'], 'selector': SELECTOR,
                      'q_hash': q_hash, **metrics}
            records.append(record)
            lookup[key] = record
        except DataError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise DataError('Malformed parameter result: ' + jid) from exc
    require(set(lookup) == set(product(CONDITIONS, SETTINGS, SEEDS, MECHANISMS)),
            'Parameter grid is missing or contains unexpected cells')
    return records, lookup


def _contrasts(lookup):
    rows = []
    for left, right in ((MECHANISMS[0], MECHANISMS[1]), (MECHANISMS[2], MECHANISMS[3])):
        pairs = []
        for condition, setting, seed in product(CONDITIONS, SETTINGS, SEEDS):
            a, b = (lookup[(condition, setting, seed, m)] for m in (left, right))
            pairs.append({'condition': condition, 'setting': setting, 'seed': seed,
                          'difference': b[METRICS[1]] - a[METRICS[1]],
                          'source_jobs': [a['job_id'], b['job_id']]})
        deltas = [p['difference'] for p in pairs]
        rows.append({'contrast': f'{_auc(right)}-{_auc(left)}', 'n': len(pairs),
                     'positive_n': sum(d > 0 for d in deltas),
                     'negative_n': sum(d < 0 for d in deltas),
                     'zero_n': sum(d == 0 for d in deltas),
                     'mean': mean(deltas), 'min': min(deltas), 'max': max(deltas),
                     'pairs': pairs})
    return rows


def _decision(lookup):
    rows = []
    # Fixed display cells from the manuscript design, never outcome-based selection.
    for setting, mechanism in product(('base', 'fraction_07'), MECHANISMS[:2]):
        records = [lookup[(STRESS, setting, seed, mechanism)] for seed in SEEDS]
        means = {m: mean(r[m] for r in records) for m in METRICS}
        display = {m: means[m] * (100 if m in METRICS[2:] else 1) for m in METRICS}
        rows.append({'condition': STRESS, 'setting': setting, 'mechanism': _auc(mechanism),
                     'means': means, 'display_values': display,
                     'source_jobs': [r['job_id'] for r in records]})
    return rows


def _check_references(dataset, records, definitions, contrasts, decision):
    summary = dataset.reference('sensitivity-descriptive-summary.json')['parameter_robustness']
    case = dataset.reference('case_summary_table_evidence.json')['parameter']
    saved_decision = dataset.reference('sensitivity_table_evidence.json')['parameter_decision']
    mismatches, comparisons = [], 0

    def check(path, actual, expected):
        nonlocal comparisons
        comparisons += 1
        numeric = type(actual) in (int, float) and type(expected) in (int, float)
        equal = math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10) if numeric else actual == expected
        if not equal:
            mismatches.append({'path': path, 'actual': actual, 'expected': expected})

    check('summary.verified_job_count', len(records), summary['verified_job_count'])
    check('summary.settings', list(SETTINGS), summary['settings'])
    check('summary.settings_definition', definitions, summary['settings_definition'])
    check('case.definitions', definitions, case['definitions'])
    by_id = {r['job_id']: r for r in summary['row_level']}
    require(len(by_id) == len(summary['row_level']) == 216 and set(by_id) == {r['job_id'] for r in records},
            'Reference parameter job scope mismatch')
    for row in records:
        for field in ('condition', 'setting', 'seed', 'mechanism', 'selector', 'q_hash', *METRICS):
            check(f"summary.{row['job_id']}.{field}", row[field], by_id[row['job_id']][field])
    for contrast in contrasts:
        label = contrast['contrast']
        baseline = [r for r in summary['contrasts_overall'] if
                    r['metric'] == METRICS[1] and r['contrast'] == label.replace('AUC', 'M')]
        saved = [r for r in case['profit_contrasts'] if r['contrast'] == label]
        require(len(baseline) == len(saved) == 1, 'Missing or duplicate parameter contrast reference')
        for field in ('n', 'positive_n', 'negative_n', 'zero_n', 'mean', 'min', 'max'):
            check(f'summary.{label}.{field}', contrast[field], baseline[0][field])
            check(f'case.{label}.{field}', contrast[field], saved[0][field])
        saved_pairs = {(r['condition'], r['setting'], r['seed']): r for r in saved[0]['pairs']}
        require(len(saved_pairs) == len(saved[0]['pairs']) == 54 and
                set(saved_pairs) == set(product(CONDITIONS, SETTINGS, SEEDS)),
                'Reference parameter contrast pair scope mismatch')
        for pair in contrast['pairs']:
            key = pair['condition'], pair['setting'], pair['seed']
            for field in ('source_jobs', 'difference'):
                check(f'case.{label}.{key}.{field}', pair[field], saved_pairs[key][field])
    saved_cells = {(r['condition'], r['setting'], r['mechanism']): r for r in saved_decision}
    require(len(saved_cells) == len(saved_decision) == len(decision) and
            set(saved_cells) == {(r['condition'], r['setting'], r['mechanism']) for r in decision},
            'Reference decision cell scope mismatch')
    for row in decision:
        key = row['condition'], row['setting'], row['mechanism']
        saved = saved_cells[key]
        check(f'decision.{key}.source_jobs', row['source_jobs'], saved['source_jobs'])
        for metric in METRICS:
            check(f'decision.{key}.{metric}', row['means'][metric], saved['means'][metric])
    return comparisons, mismatches


def _tables(definitions, contrasts, decision):
    from .paper_common import table

    scenario_body = r'Setting & \shortstack{Cost\\dispersion} & Expediting & \shortstack{Processing\\rate} & \shortstack{Policy\\response} \\' + '\n' + r'\midrule' + '\n'
    scenario_body += '\n'.join(_setting_label(r['setting_id']) + ' & ' +
                              ' & '.join(f'{r[f]:.2f}' for f in FACTORS) + r' \\' for r in definitions)
    contrast_body = r'Profit comparison & Higher & Lower\\' + '\n' + r'\midrule' + '\n'
    contrast_body += '\n'.join(r['contrast'].replace('-', ' vs ') +
                              f" & {r['positive_n']} & {r['negative_n']}" + r'\\' for r in contrasts)
    notes = ('Multipliers apply to normalized inputs for the CNC case. Cost dispersion changes while mean manufacturer cost stays fixed. '
             'The lower panel counts matched cases with higher or lower manufacturer profit under threshold payment. '
             'Each comparison covers all parameter settings, both operating conditions, and the matched replications. '
             'These counts describe the direction of the observed differences.')
    # Two tabular panels form one numbered float, just as in the current manuscript.
    scenario = (r'\begin{table}[!htbp]' + '\n' + r'\centering\footnotesize' + '\n' +
                r'\caption{Parameter scenarios and profit comparisons.}' + '\n' +
                r'\label{tab:parameter_settings}' + '\n' +
                r'\setlength{\tabcolsep}{1.8pt}\renewcommand{\arraystretch}{1.08}' + '\n' +
                r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrr@{}}\toprule' + '\n' +
                scenario_body + '\n' + r'\bottomrule\end{tabular*}' + '\n' + r'\par\vspace{7pt}' + '\n' +
                r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrr@{}}\toprule' + '\n' +
                contrast_body + '\n' + r'\bottomrule\end{tabular*}' + '\n' +
                r'\par\smallskip\begin{minipage}{\linewidth}\footnotesize ' + notes +
                '\n' + r'\end{minipage}\end{table}' + '\n')
    decision_body = r'Setting & Mechanism & \shortstack{Payment/\\order} & Profit & \shortstack{Assigned\\(\%)} & \shortstack{Availability\\(\%)}\\' + '\n' + r'\midrule' + '\n'
    for index, row in enumerate(decision):
        if index == 2:
            decision_body += r'\addlinespace[4pt]' + '\n'
        label = _setting_label(row['setting']) if index % 2 == 0 else ''
        decision_body += label + ' & ' + row['mechanism'] + ' & ' + ' & '.join(
            f"{row['display_values'][m]:.2f}" for m in METRICS) + r'\\' + '\n'
    decision_latex = table('Payment-rule outcomes under combined pressure.', 'tab:parameter_decision',
                           'llrrrr', decision_body,
                           notes=('Entries are means over the matched replications. Payment per assigned order and total discounted '
                                  'manufacturer profit use the normalized units of the CNC mechanism comparison. '
                                  'Availability is the share of orders with at least three feasible suppliers.'))
    return {'tab:parameter_settings': {'latex': scenario, 'rows': definitions,
                                       'profit_contrasts': contrasts},
            'tab:parameter_decision': {'latex': decision_latex, 'rows': decision}}


def build_sensitivity(dataset, output_dir) -> dict:
    """Return labelled table LaTeX and full-precision evidence; write no files."""
    dataset.validate_output_dir(output_dir)
    definitions, source_evidence = _definitions()
    records, lookup = _read_records(dataset)
    contrasts, decision = _contrasts(lookup), _decision(lookup)
    comparisons, mismatches = _check_references(dataset, records, definitions, contrasts, decision)
    return {'status': 'pass' if not mismatches else 'mismatch',
            'tables': _tables(definitions, contrasts, decision),
            'verified_jobs': len(records), 'comparison_count': comparisons, 'mismatches': mismatches,
            'evidence': {**source_evidence, 'family': FAMILY, 'selector': SELECTOR,
                         'verified_jobs': len(records), 'records': records,
                         'definitions': definitions, 'profit_contrasts': contrasts,
                         'parameter_decision': decision,
                         'sign_count_interpretation': 'Descriptive signs of matched profit differences; not significance tests.',
                         'display_selection': {'condition': STRESS, 'settings': ['base', 'fraction_07'],
                                               'mechanisms': list(MECHANISMS[:2]), 'seeds': list(SEEDS)},
                         'reference_role': 'Validation only; all displayed numbers come from frozen definitions and archived results.'}}
