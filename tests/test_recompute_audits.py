"""Behavior checks for the saved-sample adapter, using deterministic tiny games.

The game dimensions are small; the original 2,000 samples and 5,000 bootstrap
draws remain unchanged. These synthetic fixtures are not publication results.
"""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cmfg_cce.evaluation.independent_audit import (
    AuditSampleMatrix, build_joint_audit_samples, freeze_distribution,
)
from cmfg_cce.evaluation.revision_statistics import summarize_three_arm_transplant
from cmfg_cce.experiments.revision_pipeline import stable_solver_seed
from cmfg_cce.experiments.run_revision_cnc import _audit_sample_hash, _profile_vector_hash
from cmfg_cce.orchestration.chunks import ChunkPlan, deterministic_npz
from cmfg_cce.orchestration.manifest import canonical_json
from policy_cce_repro import recompute_audits as replay


class SavedDataset:
    def __init__(self):
        self.jobs = {}
        self.results = {}
        self.objects = {}

    def result(self, job_id):
        return deepcopy(self.results[job_id])

    def object_bytes(self, key):
        return self.objects[key]

    def object_json(self, key):
        return json.loads(self.objects[key])

    def validate_output_dir(self, path):
        # This fixture is in-memory: there is no input directory to overlap.
        return Path(path).resolve()


def _q_payload(q):
    return {"solver": q.solver, "q_hash": q.q_hash, "support": [list(p) for p in q.support], "probabilities": list(q.probabilities)}


def add_benchmark(dataset, job_id="benchmark-one"):
    q = freeze_distribution("saved-solver", (("A1",),), (1.0,), 1, ("A1", "A2"))
    t = np.linspace(-1, 1, replay.AUDIT_ROLLOUTS)
    gains = np.column_stack((np.zeros_like(t), 0.25 + t))
    returns = (20.0 + 2.0 * t)[:, None]
    samples = AuditSampleMatrix(((0, "A1"), (0, "A2")), gains, returns)
    statistics = replay._summarize(samples, job_id, "formal-bootstrap")
    group = f"q_{q.q_hash[:16]}"
    arrays = {
        group + "_gain_samples": gains,
        group + "_q_return_samples": returns,
        group + "_label_agent": np.array([0, 0], dtype=np.int16),
        group + "_label_policy_index": np.array([0, 1], dtype=np.int16),
    }
    artifact = deterministic_npz(arrays)
    key = f"jobs/{job_id}/artifacts/formal_audit_samples.npz"
    dataset.objects[key] = artifact
    stored = {**statistics, "q_hash_before": q.q_hash, "q_hash_after": q.q_hash, "audit_sample_group": group}
    dataset.jobs[job_id] = {"job_id": job_id, "family": "solver_benchmark"}
    dataset.results[job_id] = {
        "job_id": job_id, "family": "solver_benchmark", "status": "complete", "N": 1,
        "policy_ids": ["A1", "A2"], "audit_rollouts": 2000, "bootstrap_samples": 5000,
        "formal_audit_checkpoint": {"sample_artifact": {"path": "formal_audit_samples.npz", "sha256": sha256(artifact).hexdigest(), "q_count": 1}},
        "solver_results": {name: {"distribution": _q_payload(q), "formal_audit": deepcopy(stored)} for name in ("FullTensor-CCE-LP", "ExhaustiveCG-CCE")},
    }
    return arrays, key


def add_cnc(dataset, job_id="cnc-one"):
    policies = ("A1", "A2")
    q = freeze_distribution("saved-solver", (("A1",),), (1.0,), 1, policies)
    profiles = (("A1",), ("A2",))
    t = np.linspace(-1, 1, 2000)
    returns = {profiles[0]: (20 + t)[:, None], profiles[1]: (21 + 1.5 * t)[:, None]}
    samples = build_joint_audit_samples(returns, q, policies, n_agents=1, sample_count=2000)
    statistics = replay._summarize(samples, job_id, "formal-audit-shared-bootstrap")
    campaign, matrix = "a" * 64, "b" * 64
    plan = ChunkPlan.create(campaign_sha256=campaign, matrix_sha256=matrix, job_id=job_id + "::formal_audit", kind="audit", chunk_size=8, item_ids=["A1", "A2"]).to_payload()
    prefix = f"jobs/{job_id}/stages/formal_audit/chunks"
    arrays = {"returns": np.stack([returns[p] for p in profiles]), "profile_indices": np.array([0, 1], dtype=np.int64), "rollout_count": np.array([2000], dtype=np.int64)}
    payload = deterministic_npz(arrays)
    descriptor = {
        "chunk_id": "audit-000000000-000000002", "start": 0, "stop": 2,
        "campaign_sha256": campaign, "matrix_sha256": matrix, "job_id": job_id + "::formal_audit", "kind": "audit",
        "payload_sha256": sha256(payload).hexdigest(),
        "arrays": {name: {"dtype": str(array.dtype), "shape": list(array.shape)} for name, array in arrays.items()},
    }
    dataset.objects[prefix + "/chunk_plan.json"] = json.dumps(plan).encode()
    dataset.objects[prefix + "/audit-000000000-000000002.json"] = json.dumps(descriptor).encode()
    dataset.objects[prefix + "/audit-000000000-000000002.npz"] = payload
    dataset.jobs[job_id] = {"job_id": job_id, "family": "cnc_main"}
    dataset.results[job_id] = {
        "job_id": job_id, "family": "cnc_main", "status": "complete", "N": 1,
        "campaign_sha256": campaign, "matrix_sha256": matrix, "policy_ids": list(policies),
        "audit_rollouts": 2000, "bootstrap_samples": 5000,
        "distributions": {"platform_operating_score": _q_payload(q)},
        "formal_audit": {
            "rollouts": 2000, "bootstrap_samples": 5000, "profile_count": 2,
            "q_union_sha256": sha256(canonical_json({"platform_operating_score": q.q_hash})).hexdigest(),
            "raw_profile_return_vectors_sha256": _profile_vector_hash(profiles, policies, returns),
            "checkpoint": {"item_order_sha256": plan["item_order_sha256"], "chunks": [descriptor]},
            "distributions": {"platform_operating_score": {**statistics, "q_hash_before": q.q_hash, "q_hash_after": q.q_hash, "joint_replication_samples_sha256": _audit_sample_hash(samples)}},
        },
    }


def add_transplant(dataset, job_id="transplant-one"):
    policies = tuple(f"A{i}" for i in range(1, 7))
    q = freeze_distribution("saved-solver", (("A1",) * 4,), (1.0,), 4, policies)
    arms = ("same_mechanism_control", "cross_mechanism", "target_recomputed")
    labels = tuple((agent, policy) for agent in range(4) for policy in policies)
    arrays = {
        "label_agent": np.repeat(np.arange(4, dtype=np.int16), 6),
        "label_policy_index": np.tile(np.arange(6, dtype=np.int16), 4),
    }
    samples_by_arm = {}
    hashes = {}
    t = np.linspace(-1, 1, 2000)
    for arm, gain in zip(arms, (0.1, 2.0, 0.2), strict=True):
        gains = np.tile((gain + t)[:, None], (1, 24))
        gains[:, ::6] = 0.0
        returns = np.tile((20 + 2 * t)[:, None], (1, 4))
        samples_by_arm[arm] = AuditSampleMatrix(labels, gains, returns)
        arrays[f"{arm}__gain_samples"] = gains
        arrays[f"{arm}__q_return_samples"] = returns
        hashes[arm] = sha256(deterministic_npz({"gain_samples": gains, "q_return_samples": returns})).hexdigest()
    artifact = deterministic_npz(arrays)
    dataset.objects[f"jobs/{job_id}/artifacts/transplant_audit_samples.npz"] = artifact
    dataset.jobs[job_id] = {"job_id": job_id, "family": "policy_transplant", "metadata": {"n_agents": 4, "policies_per_agent": 6}}
    summary = summarize_three_arm_transplant(
        *(samples_by_arm[arm] for arm in arms), alpha=0.05, bootstrap_samples=5000,
        bootstrap_seed=stable_solver_seed("revision-full-v1", job_id, "three-arm-joint-bootstrap"),
    )
    dataset.results[job_id] = {
        "job_id": job_id, "family": "policy_transplant", "status": "complete",
        "audit_rollouts": 2000, "bootstrap_samples": 5000,
        "source_distribution": _q_payload(q), "target_distribution": _q_payload(q),
        "joint_sample_artifact": {"path": "transplant_audit_samples.npz", "sha256": sha256(artifact).hexdigest(), "arms": sorted(arms), "common_replication_index_across_arms": True},
        "joint_sample_hashes": hashes, "three_arm_summary": summary,
    }


class AuditReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = SavedDataset()
        add_benchmark(cls.dataset)
        add_cnc(cls.dataset)

    def test_benchmark_replays_original_statistics_and_deduplicates_q(self):
        with patch.object(replay, "_summarize", wraps=replay._summarize) as summarize:
            report = replay._recompute_job(self.dataset, "benchmark-one")
        self.assertEqual(report["status"], "pass", report)
        self.assertEqual(summarize.call_count, 1)
        self.assertEqual(report["unique_distribution_count"], 1)
        summary = report["recomputed"]["solver_audits"]["FullTensor-CCE-LP"]["statistics"]
        self.assertAlmostEqual(summary["nominal_gap"], 0.25, places=13)
        self.assertAlmostEqual(summary["relative_nominal_gap_percent"], 1.25, places=12)
        self.assertEqual(summary["audit_rollouts"], 2000)
        self.assertAlmostEqual(summary["max_replacement_gain_standard_error"], np.std(np.linspace(-1, 1, 2000), ddof=1) / np.sqrt(2000), places=13)

    def test_cnc_rebuilds_samples_from_complete_original_closure(self):
        report = replay._recompute_job(self.dataset, "cnc-one")
        self.assertEqual(report["status"], "pass", report)
        statistics = report["recomputed"]["distribution_audits"]["platform_operating_score"]["statistics"]
        self.assertAlmostEqual(statistics["nominal_gap"], 1.0, places=13)
        self.assertIn("joint_replication_samples_sha256", statistics)

    def test_transplant_replays_three_paired_arms_and_joint_effect_intervals(self):
        dataset = SavedDataset()
        add_transplant(dataset)
        report = replay._recompute_job(dataset, "transplant-one")
        self.assertEqual(report["status"], "pass", report)
        summary = report["recomputed"]["three_arm_summary"]
        self.assertEqual(summary["replications"], 2000)
        self.assertAlmostEqual(summary["effects"]["cross_minus_same_control_percent"], 9.5, places=11)
        self.assertAlmostEqual(summary["effects"]["cross_minus_target_recomputed_percent"], 9.0, places=11)
        self.assertEqual(set(summary["arm_audits"]), {"same_control", "cross_transplant", "target_recomputed"})
        self.assertLess(summary["effect_intervals"]["cross_minus_same_control_percent"]["lower"], 9.5)
        self.assertGreater(summary["effect_intervals"]["cross_minus_same_control_percent"]["upper"], 9.5)

    def test_transplant_joint_sample_digest_mismatch_is_preserved(self):
        dataset = SavedDataset()
        add_transplant(dataset)
        dataset.results["transplant-one"]["joint_sample_hashes"]["cross_mechanism"] = "0" * 64
        report = replay._recompute_job(dataset, "transplant-one")
        self.assertEqual(report["status"], "mismatch")
        self.assertEqual(report["mismatches"][0]["field"], "joint_sample_hashes.cross_mechanism")
        self.assertIn("three_arm_summary", report["recomputed"])

    def test_cnc_rejects_missing_closure_chunk(self):
        dataset = deepcopy(self.dataset)
        dataset.results["cnc-one"]["formal_audit"]["checkpoint"]["chunks"] = []
        report = replay._recompute_job(dataset, "cnc-one")
        self.assertEqual(report["status"], "error")
        self.assertIn("Incomplete CNC audit closure", report["error"]["message"])

    def test_mismatch_keeps_new_statistics_and_original_saved_values(self):
        dataset = deepcopy(self.dataset)
        saved = dataset.results["benchmark-one"]["solver_results"]["FullTensor-CCE-LP"]["formal_audit"]
        saved["nominal_gap"] = 10.0
        original = deepcopy(dataset.results)
        report = replay._recompute_job(dataset, "benchmark-one")
        self.assertEqual(report["status"], "mismatch")
        self.assertEqual(len(report["mismatches"]), 1)
        self.assertEqual(report["mismatches"][0]["saved"], 10.0)
        self.assertAlmostEqual(report["mismatches"][0]["recomputed"], 0.25)
        self.assertEqual(dataset.results, original)

    def test_tolerances_are_strict_and_discrete_fields_are_exact(self):
        self.assertEqual(replay._differences({"value": 1.0 + 1e-10}, {"value": 1.0}), [])
        self.assertEqual(len(replay._differences({"value": 1.0 + 1e-7}, {"value": 1.0})), 1)
        self.assertEqual(len(replay._differences({"agent": 1}, {"agent": 1.0})), 1)
        self.assertEqual(len(replay._differences({"flag": True}, {"flag": 1})), 1)
        self.assertEqual(len(replay._differences({"missing": 1.0}, {})), 1)

    def test_unexpected_nonfinite_output_is_mismatch_and_serializable_evidence(self):
        with patch.object(replay, "_summarize", return_value={"nominal_gap": float("nan")}):
            report = replay._recompute_job(self.dataset, "benchmark-one")
        self.assertEqual(report["status"], "mismatch")
        self.assertEqual(report["mismatches"][0]["recomputed"], {"nonfinite_float": "nan"})
        json.dumps(report, allow_nan=False)

    def test_rejects_corrupt_npz_before_computing(self):
        dataset = deepcopy(self.dataset)
        key = "jobs/benchmark-one/artifacts/formal_audit_samples.npz"
        dataset.objects[key] = b"not the saved bytes"
        with patch.object(replay, "_summarize", wraps=replay._summarize) as summarize:
            report = replay._recompute_job(dataset, "benchmark-one")
        self.assertEqual(report["status"], "error")
        self.assertIn("SHA256 mismatch", report["error"]["message"])
        self.assertEqual(summarize.call_count, 0)

    def test_sample_schema_rejects_reordered_labels_and_nonfinite_values(self):
        arrays = {"g": np.zeros((2000, 2)), "r": np.ones((2000, 1)), "a": np.array([0, 0], dtype=np.int16), "p": np.array([1, 0], dtype=np.int16)}
        kwargs = dict(gain_key="g", return_key="r", agent_key="a", policy_key="p", n_agents=1, policy_ids=("A1", "A2"))
        with self.assertRaisesRegex(ValueError, "Policy label order"):
            replay._samples(arrays, **kwargs)
        arrays["p"] = np.array([0, 1], dtype=np.int16)
        arrays["g"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            replay._samples(arrays, **kwargs)

    def test_frozen_sample_and_bootstrap_settings_cannot_be_reduced(self):
        dataset = deepcopy(self.dataset)
        dataset.results["benchmark-one"]["bootstrap_samples"] = 100
        report = replay._recompute_job(dataset, "benchmark-one")
        self.assertEqual(report["status"], "error")
        self.assertIn("2,000/5,000", report["error"]["message"])

    def test_job_filter_rejects_duplicates_unknown_and_empty_before_writes(self):
        for ids in (["benchmark-one", "benchmark-one"], ["absent"], []):
            with self.subTest(ids=ids), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "new"
                with self.assertRaises(ValueError):
                    replay.recompute_audits(self.dataset, destination, job_ids=ids)
                self.assertFalse(destination.exists())

    def test_direct_api_validates_output_location_before_any_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "forbidden"
            with patch.object(self.dataset, "validate_output_dir", side_effect=ValueError("Input/output overlap")) as validate:
                with self.assertRaisesRegex(ValueError, "Input/output overlap"):
                    replay.recompute_audits(self.dataset, destination)
            validate.assert_called_once_with(destination)
            self.assertFalse(destination.exists())

    def test_per_job_failure_preserves_other_job_and_reports_are_never_overwritten(self):
        dataset = deepcopy(self.dataset)
        dataset.results["cnc-one"]["bootstrap_samples"] = 100
        with tempfile.TemporaryDirectory() as directory:
            summary = replay.recompute_audits(dataset, directory, workers=2)
            self.assertEqual(summary["status"], "fail")
            self.assertEqual(summary["matched_job_count"], 1)
            self.assertEqual(summary["failed_job_count"], 1)
            self.assertEqual(summary["failures"][0]["job_id"], "cnc-one")
            saved = (Path(directory) / "audit-summary.json").read_bytes()
            json.loads(saved)
            self.assertEqual(len(list((Path(directory) / "audit-jobs").glob("*.json"))), 2)
            with self.assertRaisesRegex(ValueError, "already exists"):
                replay.recompute_audits(dataset, directory)
            self.assertEqual((Path(directory) / "audit-summary.json").read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
