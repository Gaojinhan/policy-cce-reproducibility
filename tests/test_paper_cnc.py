"""Deterministic CNC presentation checks; no PDFs or scientific runs are made."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from policy_cce_repro.offline import DataError
from policy_cce_repro.paper_cnc import (
    AVAIL, FACTOR_METRICS, MECHANISM_PANELS, MECHANISMS, RELATIVE_METRICS,
    _audit_rows, _capacity_table, _endpoint_table, _family_table, _interval,
    _mechanism_figure_data, _policy_figure_data, _relative_table,
    _transplant_table, build_cnc,
)


def interval(estimate=2., lower=1., upper=3., se=.25):
    return {"estimate": estimate, "ci_lower": lower, "ci_upper": upper, "standard_error": se}


def outcomes_fixture():
    report = {"status": "pass", "case_count": 96, "mismatch_count": 0, "mismatches": []}
    metrics = [metric for metric, _ in RELATIVE_METRICS] + ["submitted_lead_time_multiplier"]
    report["relative_mechanism_effects"] = [
        {"contrast": contrast, "metrics": {metric: interval() for metric in metrics}}
        for contrast in ("AUC2-AUC1", "AUC4-AUC3", "AUC3-AUC1", "AUC4-AUC2")]
    report["operating_factor_effects"] = [
        {"factor": factor, "metrics": {metric: interval(-.02, -.03, -.01, .002) for metric, *_ in FACTOR_METRICS}}
        for factor in ("platform_load", "route_mix", "outside_load")]
    report["family_mismatch"] = [
        {"family": "F" + str(i), "unit": "percentage_points", **interval(-1.8, -1.9, -1.7, .05)}
        for i in range(1, 5)]
    report["endpoint_means"] = [
        {"label": label, "condition": label.lower().replace(" ", "_"), "n_cases": 12,
         "metrics": {metric: .9 for metric in ("assignment_rate", AVAIL, *(AVAIL + "_F" + str(i) for i in range(1, 5)))}}
        for label in ("Reference", "Combined pressure")]
    report["mechanism_levels"] = [
        {"mechanism": mechanism, "n_cases": 24,
         "metrics": {metric: {"mean": 1.012 if metric == "submitted_lead_time_multiplier" else .9, "sample_sd": .01}
                     for metric, *_ in MECHANISM_PANELS}}
        for mechanism in MECHANISMS]
    report["policy_composition"] = [
        {"mechanism": mechanism, "composition": [.2, .3, .5], "winner_shares": [.6, .3, .1]}
        for mechanism in MECHANISMS]
    return report


class FixtureDataset:
    def __init__(self):
        self.jobs, self.results, self.checked_outputs = {}, {}, []
        entries = []
        for index, mechanism in enumerate(MECHANISMS):
            target = MECHANISMS[index + 1 if index % 2 == 0 else index - 1]
            for case in range(24):
                jid = f"cnc-{index}-{case}"
                self.jobs[jid] = {"family": "cnc_main"}
                # Fake saved values ensure paper aggregation uses only recomputed fields.
                self.results[jid] = {"mechanism": mechanism, "formal_audit": {"relative_nominal_gap_percent": 999}}
                entries.append({"job_id": jid, "family": "cnc_main", "status": "pass", "mismatches": [],
                                "recomputed": {"distribution_audits": {"platform_operating_score": {"statistics": {
                                    "relative_nominal_gap_percent": .1 + case / 100,
                                    "max_replacement_gain_standard_error": .5,
                                    "max_t_relative_gap_ucb95_percent": .8 - case / 100}}}}})
                jid = f"transplant-{index}-{case}"
                self.jobs[jid] = {"family": "policy_transplant"}
                self.results[jid] = {"source_mechanism": mechanism, "target_mechanism": target}
                entries.append({"job_id": jid, "family": "policy_transplant", "status": "pass", "mismatches": [],
                    "recomputed": {"three_arm_summary": {
                        "arm_relative_nominal_gap_percent": {"same_control": .2, "cross_transplant": 12., "target_recomputed": .3},
                        "effects": {"cross_minus_same_control_percent": 11.8},
                        "effect_intervals": {"cross_minus_same_control_percent": {"lower": 10., "upper": 13., "standard_error": .2}}}}})
        self.audit = {"status": "pass", "mismatch_field_count": 0, "failures": [], "jobs": entries}

    def result(self, jid):
        return self.results[jid]

    def reference(self, name):
        raise AssertionError("Reference values must not be used for reconstruction")

    def validate_output_dir(self, output):
        self.checked_outputs.append(str(output))


class CNCUnitsTests(unittest.TestCase):
    def test_negative_scale_reverses_interval_and_keeps_standard_error_positive(self):
        result = _interval(interval(-.02, -.03, -.01, .002), -100)
        self.assertEqual(result, {"estimate": 2., "ci_lower": 1., "ci_upper": 3., "standard_error": .2})

    def test_bad_interval_and_nonfinite_values_fail(self):
        for value in (interval(1., 3., 2.), interval(float("nan")), interval(se=-1.)):
            with self.assertRaises(DataError):
                _interval(value)

    def test_capacity_rates_and_counts_have_different_scales(self):
        table = _capacity_table(outcomes_fixture())
        loss = next(row for row in table["rows"] if row["metric"] == "assignment_rate")
        count = next(row for row in table["rows"] if row["metric"] == "route_capacity_feasible_manufacturers_per_order")
        mismatch = next(row for row in table["rows"] if row["metric"] == "orders_with_scalar_route_mismatch_rate")
        self.assertEqual((loss["estimate"], loss["ci_lower"], loss["ci_upper"]), (2., 1., 3.))
        self.assertEqual((count["estimate"], count["unit"]), (.02, "count"))
        self.assertEqual(mismatch["estimate"], -2.)
        self.assertIn("$+0.020$", table["latex"])

    def test_family_mismatch_is_already_percentage_points(self):
        table = _family_table(outcomes_fixture())
        self.assertEqual(table["rows"][0]["estimate"], -1.8)
        self.assertIn("$-1.800$", table["latex"])
        self.assertNotIn("-180.000", table["latex"])

    def test_family_units_must_be_explicit(self):
        report = outcomes_fixture()
        report["family_mismatch"][0]["unit"] = "fraction"
        with self.assertRaisesRegex(DataError, "percentage points"):
            _family_table(report)

    def test_endpoint_rates_are_converted_to_percent_once(self):
        table = _endpoint_table(outcomes_fixture())
        self.assertEqual(table["rows"][0]["metrics"]["assignment_rate"], 90.)
        self.assertIn("90.00", table["latex"])

    def test_relative_change_is_already_percent_and_lead_time_has_one_comparison(self):
        table = _relative_table(outcomes_fixture())
        self.assertEqual(len(table["rows"]), 13)
        self.assertEqual(table["rows"][0]["contrasts"]["AUC2-AUC1"]["estimate"], 2.)
        self.assertIn("$+2.00$", table["latex"])
        lead = [row for row in table["rows"] if row["metric"] == "submitted_lead_time_multiplier"]
        self.assertEqual(len(lead), 1)
        self.assertIsNone(lead[0]["contrasts"]["AUC2-AUC1"])

    def test_lead_time_figure_shows_due_time_deviation_not_relative_mechanism_effect(self):
        report = outcomes_fixture()
        panel = _mechanism_figure_data(report)[-1]
        self.assertEqual(len(panel["points"]), 2)
        self.assertAlmostEqual(panel["points"][0]["mean"], 1.2)
        self.assertEqual(panel["points"][0]["sample_sd"], 1.)
        relative_lead = _relative_table(report)["rows"][6]["contrasts"]["AUC4-AUC3"]["estimate"]
        self.assertEqual(relative_lead, 2.)
        self.assertNotAlmostEqual(panel["points"][0]["mean"], relative_lead)

    def test_policy_composition_conserves_shares_and_rejects_incomplete_total(self):
        report = outcomes_fixture()
        rows = _policy_figure_data(report)
        self.assertEqual(rows[0]["composition"], [20., 30., 50.])
        self.assertEqual(sum(rows[0]["winner_shares"]), 100.)
        report["policy_composition"][0]["composition"] = [.2, .2, .2]
        with self.assertRaisesRegex(DataError, "shares"):
            _policy_figure_data(report)

    def test_duplicate_or_missing_summary_groups_fail(self):
        report = outcomes_fixture()
        report["relative_mechanism_effects"].append(report["relative_mechanism_effects"][0])
        with self.assertRaisesRegex(DataError, "Duplicate"):
            _relative_table(report)
        report = outcomes_fixture()
        report["mechanism_levels"].pop()
        with self.assertRaisesRegex(DataError, "Incomplete"):
            _mechanism_figure_data(report)


class CNCAssemblyTests(unittest.TestCase):
    def test_upper_bound_belongs_to_largest_gap_case_not_largest_bound(self):
        records = []
        for mechanism in MECHANISMS:
            for index, (gap, bound) in enumerate(((.8, 1.2), (.2, 2.1))):
                records.append({"job_id": mechanism + str(index), "mechanism": mechanism,
                    "statistics": {"relative_nominal_gap_percent": gap, "max_replacement_gain_standard_error": .3,
                                   "max_t_relative_gap_ucb95_percent": bound}})
        row = _audit_rows(records)[-1]
        self.assertEqual(row["largest_gap_upper_bound_percent"], 1.2)
        self.assertEqual(row["upper_bound_max_percent"], 2.1)
        self.assertEqual(row["mean_gap_percent"], .5)

    def test_transfer_uses_each_individual_interval_without_inventing_aggregate_ci(self):
        dataset = FixtureDataset()
        records = [{"job_id": r["job_id"], **dataset.results[r["job_id"]], "statistics": r["recomputed"]["three_arm_summary"]}
                   for r in dataset.audit["jobs"] if r["family"] == "policy_transplant"]
        records[0]["statistics"]["effect_intervals"]["cross_minus_same_control_percent"]["lower"] = -.1
        table = _transplant_table(records)
        self.assertEqual(table["rows"][0]["increase_pp"], 11.8)
        self.assertEqual(table["rows"][0]["individual_ci_positive_count"], 23)
        self.assertNotIn("ci_lower", table["rows"][0])

    def test_all_artifacts_use_recomputed_metrics_and_check_output_path_first(self):
        dataset = FixtureDataset()
        original_report = copy.deepcopy(dataset.audit)
        with tempfile.TemporaryDirectory() as directory, patch("policy_cce_repro.paper_cnc._plot_mechanisms") as mechanisms, patch("policy_cce_repro.paper_cnc._plot_policies") as policies:
            result = build_cnc(dataset, dataset.audit, outcomes_fixture(), directory)
            self.assertEqual(dataset.checked_outputs, [directory])
            self.assertEqual(len(result["tables"]), 6)
            self.assertEqual(len(result["figures"]), 2)
            self.assertAlmostEqual(result["tables"]["tab:cnc_audit"]["rows"][-1]["mean_gap_percent"], .215)
            self.assertEqual(list(Path(directory).rglob("*.pdf")), [])
            mechanisms.assert_called_once()
            policies.assert_called_once()
        self.assertEqual(dataset.audit, original_report)

    def test_failed_replay_stops_before_plotting(self):
        dataset = FixtureDataset()
        dataset.audit["status"] = "fail"
        with tempfile.TemporaryDirectory() as directory, patch("policy_cce_repro.paper_cnc._plot_mechanisms") as plot:
            with self.assertRaisesRegex(DataError, "did not pass"):
                build_cnc(dataset, dataset.audit, outcomes_fixture(), directory)
            plot.assert_not_called()
            self.assertFalse((Path(directory) / "figures").exists())

    def test_any_existing_figure_blocks_all_drawing_and_preserves_bytes(self):
        dataset = FixtureDataset()
        for filename in ("cnc_mechanism_levels.pdf", "cnc_mechanism_levels.png",
                         "cnc_policy_composition_original_style.pdf", "cnc_policy_composition_original_style.png"):
            with tempfile.TemporaryDirectory() as directory, patch("policy_cce_repro.paper_cnc._plot_mechanisms") as plot:
                target = Path(directory) / "figures" / filename
                target.parent.mkdir()
                target.write_bytes(b"existing published result")
                with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                    build_cnc(dataset, dataset.audit, outcomes_fixture(), directory)
                self.assertEqual(target.read_bytes(), b"existing published result")
                self.assertEqual(len(list(target.parent.iterdir())), 1)
                plot.assert_not_called()

    def test_missing_and_duplicate_replayed_jobs_fail_before_render(self):
        for duplicate in (False, True):
            dataset = FixtureDataset()
            if duplicate:
                dataset.audit["jobs"].append(dataset.audit["jobs"][0])
            else:
                dataset.audit["jobs"].pop()
            with tempfile.TemporaryDirectory() as directory, patch("policy_cce_repro.paper_cnc._plot_mechanisms") as plot:
                with self.assertRaises(DataError):
                    build_cnc(dataset, dataset.audit, outcomes_fixture(), directory)
                plot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
