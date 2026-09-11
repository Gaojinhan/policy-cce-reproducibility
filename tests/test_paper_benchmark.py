"""Protect the identity joins, aggregate definitions and printed units."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cmfg_cce.evaluation.independent_audit import freeze_distribution
from cmfg_cce.evaluation.revision_statistics import support_diversity
from policy_cce_repro import paper_benchmark as paper
from policy_cce_repro.offline import DataError, Dataset
from policy_cce_repro.paper_common import numeric_rows


MECHANISMS = ("M1_price_first", "M2_price_critical", "M3_delivery_first", "M4_delivery_critical")


class PairDataset:
    def __init__(self):
        self.jobs = {"audit": {"metadata": {"n_agents": 4, "policies_per_agent": 6}}, "runtime": {"metadata": {"n_agents": 4, "policies_per_agent": 6}}}
        self.results = {jid: {"provenance_path": jid + ".json"} for jid in self.jobs}
        self.payloads = {}

    def result(self, job_id):
        return deepcopy(self.payloads[job_id])

    def read_json(self, key):
        return {"result_sha256": "a" * 64 if key == "runtime.json" else "b" * 64}


def pair_fixture():
    dataset = PairDataset()
    q = freeze_distribution("test", (("A1",) * 4,), (1.0,), 4, tuple(f"A{i}" for i in range(1, 7)))
    payload = {"solver": q.solver, "support": [list(profile) for profile in q.support], "probabilities": list(q.probabilities), "q_hash": q.q_hash}
    stats = dict.fromkeys((*paper.UNCERTAINTY_FIELDS, "max_t_relative_gap_lcb95_percent", "payoff_denominator"), 0.25)
    result = {
        "job_id": "audit", "family": "solver_benchmark", "N": 4, "J": 6,
        "mechanism": MECHANISMS[0], "seed": 0, "train_rollouts": 200, "audit_rollouts": 2000,
        "policy_ids": [f"A{i}" for i in range(1, 7)],
        "runtime_q_provenance": {"runtime_job_id": "runtime", "runtime_result_sha256": "a" * 64, "q_hashes": dict.fromkeys(paper.SOLVERS, q.q_hash)},
        "solver_results": {},
    }
    runtime = {
        "job_id": "runtime", "family": "solver_benchmark_runtime", "reported_N": 4, "reported_J": 6,
        "mechanism": MECHANISMS[0], "seed": 0,
        "runtime_contract": {"empty_cache_per_solver": True, "verification_runtime_included": False},
        "runtime_environment": {"workers": 32}, "solver_results": {},
    }
    report = {"job_id": "audit", "status": "pass", "mismatches": [], "recomputed": {"solver_audits": {}}}
    for index, solver in enumerate(paper.SOLVERS):
        result["solver_results"][solver] = {
            "distribution": deepcopy(payload), "formal_audit": {**stats, "q_hash_before": q.q_hash, "q_hash_after": q.q_hash},
            "runtime_q_source_job_id": "runtime", "runtime_q_source_sha256": "a" * 64, "q_restored_without_resolving": True,
            "support_diversity": asdict(support_diversity(q.probabilities)), "runtime_seconds": 9999.0, "training_profile_count": 888,
        }
        runtime["solver_results"][solver] = {"distribution": deepcopy(payload), "runtime_seconds": 60.0 + index, "training_profile_count": 100 + index}
        report["recomputed"]["solver_audits"][solver] = {"q_hash": q.q_hash, "statistics": deepcopy(stats)}
    dataset.payloads = {"audit": result, "runtime": runtime}
    return dataset, result, report


def synthetic_rows():
    rows = []
    for n, j in (*paper.SCALES, *paper.LARGE):
        mechanisms = MECHANISMS if (n, j) in paper.SCALES else (MECHANISMS[0], MECHANISMS[3])
        solvers = paper.SOLVERS if (n, j) in paper.SCALES else paper.SPARSE_SOLVERS
        for mi, mechanism in enumerate(mechanisms):
            for seed in range(3):
                for si, solver in enumerate(solvers):
                    elapsed = 60.0 if seed % 2 == 0 else 600.0
                    if solver != "REPAIR-SAD-CCE":
                        elapsed *= 2.0 if seed % 2 == 0 else 3.0
                    rows.append({"job_id": f"n{n}j{j}-{mi}-{seed}", "family": "solver_benchmark" if (n, j) in paper.SCALES else "scalability_sparse", "N": n, "J": j, "mechanism": mechanism, "seed": seed, "solver": solver, "runtime_seconds": elapsed, "relative_nominal_gap_percent": 0.125 + seed / 10, "training_profile_count": 100, "profile_saving_percent": 100 * (1 - 100 / j ** n), "support_size": 1, "effective_support_size": 1.0, "max_payoff_standard_error": 1.234, "max_replacement_gain_standard_error": 0.567, "max_t_relative_gap_ucb95_percent": 2.345})
    return rows


def formatting_evidence():
    rows = synthetic_rows()
    pure = [{"job_id": jid, "pure_nash_count": 1} for jid in sorted({r["job_id"] for r in rows if r["family"] == "solver_benchmark"})]
    uncertainty = []
    for n, j in paper.SCALES:
        group = [row for row in rows if (row["N"], row["J"]) == (n, j) and row["solver"] == "REPAIR-SAD-CCE"]
        uncertainty.append({"N": n, "J": j, "game_count": 12, **{field: paper._ms(row[field] for row in group) for field in paper.UNCERTAINTY_FIELDS}})
    return {
        "benchmark_cells": [paper._cell(rows, scale, solver) for scale in paper.SCALES for solver in paper.SOLVERS],
        "benchmark_summary": paper._aggregate(rows, pure), "uncertainty_cells": uncertainty,
        "nonpure_cases": [next(row for row in rows if row["seed"] == 1 and row["solver"] == "REPAIR-SAD-CCE")],
        "runtime_only_extension": [{"solver": solver, "runtime_seconds": {"mean": 120.0, "sd": 30.0}, "mean_training_profile_count": 1234.5, "mean_profile_saving_percent": 98.76} for solver in paper.SOLVERS],
    }


class BenchmarkPresentationTests(unittest.TestCase):
    def test_runtime_metrics_use_paired_empty_cache_run_not_audit_run(self):
        dataset, result, report = pair_fixture()
        rows, mismatches = paper._joined_job(dataset, result, report)
        self.assertEqual(mismatches, [])
        self.assertEqual(rows[0]["runtime_seconds"], 60.0)
        self.assertEqual(rows[0]["training_profile_count"], 100)
        self.assertAlmostEqual(rows[0]["profile_saving_percent"], 100 * (1 - 100 / 1296))

    def test_replayed_values_drive_output_and_mismatches_are_not_repaired(self):
        dataset, result, report = pair_fixture()
        report["recomputed"]["solver_audits"][paper.SOLVERS[0]]["statistics"]["relative_nominal_gap_percent"] = 0.5
        rows, mismatches = paper._joined_job(dataset, result, report)
        self.assertEqual(rows[0]["relative_nominal_gap_percent"], 0.5)
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0]["saved"], 0.25)

    def test_changed_timed_distribution_rejected_even_with_valid_new_hash(self):
        dataset, result, report = pair_fixture()
        q = freeze_distribution("test", (("A2",) * 4,), (1.0,), 4, tuple(f"A{i}" for i in range(1, 7)))
        dataset.payloads["runtime"]["solver_results"][paper.SOLVERS[0]]["distribution"] = {"solver": q.solver, "support": [list(profile) for profile in q.support], "probabilities": list(q.probabilities), "q_hash": q.q_hash}
        with self.assertRaisesRegex(DataError, "q identity mismatch"):
            paper._joined_job(dataset, result, report)

    def test_runtime_case_sha_contract_and_worker_count_are_validated(self):
        mutations = [
            (lambda d, r: d.payloads["runtime"].update(reported_N=5), "case identity"),
            (lambda d, r: r["runtime_q_provenance"].update(runtime_result_sha256="0" * 64), "SHA mismatch"),
            (lambda d, r: d.payloads["runtime"]["runtime_contract"].update(verification_runtime_included=True), "Verification time"),
            (lambda d, r: d.payloads["runtime"]["runtime_environment"].update(workers=8), "worker count"),
        ]
        for mutation, message in mutations:
            with self.subTest(message=message):
                dataset, result, report = pair_fixture()
                mutation(dataset, result)
                with self.assertRaisesRegex(DataError, message):
                    paper._joined_job(dataset, result, report)

    def test_cell_requires_each_mechanism_and_replicate_once(self):
        rows = synthetic_rows()
        selected = [r for r in rows if (r["N"], r["J"]) == (4, 6) and r["solver"] == paper.SOLVERS[0]]
        selected[1] = deepcopy(selected[0])
        with self.assertRaisesRegex(DataError, "Duplicate or missing"):
            paper._cell(selected, (4, 6), paper.SOLVERS[0])

    def test_time_ratio_is_mean_of_paired_ratios(self):
        evidence = formatting_evidence()
        aggregate = evidence["benchmark_summary"]
        ratio = aggregate["solver_rows"][0]["within_game_time_ratio_to_dss"]["mean"]
        self.assertAlmostEqual(ratio, (2 + 3 + 2) / 3)
        self.assertNotAlmostEqual(ratio, (120 + 1800 + 120) / (60 + 600 + 60))
        self.assertEqual(aggregate["solver_rows"][3]["within_game_time_ratio_to_dss"]["mean"], 1.0)

    def test_tables_preserve_names_minutes_precision_and_one_based_replicate(self):
        tables = paper.format_tables(formatting_evidence())
        self.assertEqual(set(tables), {"tab:solver_results", "tab:solver_benchmark_summary", "tab:solver_uncertainty", "tab:nonpure_cases", "tab:scalability_exact_runtime"})
        solver = tables["tab:solver_results"]
        self.assertIn("Time (min)", solver["headers"])
        self.assertEqual(solver["display_rows"][0][1], "FullSpace-CCE-LP")
        self.assertEqual(solver["display_rows"][1][1], "ColumnGen-CCE")
        self.assertEqual(solver["display_rows"][0][2], "$0.23 \\pm 0.09$")
        self.assertEqual(tables["tab:nonpure_cases"]["display_rows"][0][2], "2")
        n7 = tables["tab:scalability_exact_runtime"]
        self.assertEqual(n7["display_rows"][0], ["FullSpace-CCE-LP", "$2.0 \\pm 0.5$", "1,234", "98.8"])
        self.assertNotIn("gap", " ".join(n7["headers"]).lower())
        self.assertIn("relative_gap", solver["latex"])
        self.assertEqual(len(numeric_rows(solver["latex"])), 24)

    def test_frontier_axes_and_error_bars_match_manuscript_style(self):
        fig = paper._frontier(synthetic_rows())
        try:
            self.assertEqual(len(fig.axes), 3)
            self.assertEqual(fig.axes[0].get_yscale(), "symlog")
            self.assertEqual(fig.axes[1].get_yscale(), "log")
            self.assertEqual(fig.axes[1].get_ylabel(), "Wall-clock time (s)")
            self.assertEqual(len(fig.axes[0].containers), 4)
            self.assertEqual([label.get_text() for label in fig.axes[0].get_xticklabels()], [f"{n},{j}" for n, j in paper.SCALES])
        finally:
            plt.close(fig)

    def test_scalability_displays_only_n8_n10_and_converts_seconds_to_minutes(self):
        rows = synthetic_rows()
        fig = paper._scalability(rows)
        try:
            self.assertEqual([label.get_text() for label in fig.axes[0].get_xticklabels()], ["8", "10"])
            first_point = fig.axes[0].collections[0].get_offsets()[0]
            self.assertAlmostEqual(float(first_point[1]), 2.0)
            self.assertEqual(fig.axes[0].get_ylabel(), "Wall-clock time (min)")
            self.assertEqual(len(fig.axes[0].containers), 4)
        finally:
            plt.close(fig)


@unittest.skipUnless(os.environ.get("POLICY_CCE_TEST_DATA") and os.environ.get("POLICY_CCE_TEST_AUDIT_REPORT"), "Set archived-data and audit-report paths for full-data integration")
class RealBenchmarkIntegrationTests(unittest.TestCase):
    def test_raw_replayed_data_match_historical_evidence_and_current_display_contract(self):
        dataset = Dataset(os.environ["POLICY_CCE_TEST_DATA"])
        audit = json.loads(Path(os.environ["POLICY_CCE_TEST_AUDIT_REPORT"]).read_text())
        evidence = paper.collect_benchmark(dataset, audit)
        self.assertEqual(evidence["status"], "pass", evidence["mismatches"])
        self.assertEqual(len(evidence["nonpure_cases"]), 6)
        contract = json.loads((Path(__file__).parents[1] / "metadata" / "paper-layout-v1.json").read_text())
        for label, table in paper.format_tables(evidence).items():
            with self.subTest(label=label):
                self.assertEqual(table["caption"], contract["tables"][label]["caption"])
                self.assertEqual(numeric_rows(table["latex"]), contract["tables"][label]["expected_numeric_rows"])


if __name__ == "__main__":
    unittest.main()
