"""A small end-to-end demo plus focused isolation and invariant checks."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from policy_cce_repro import demo


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name) / "run"
        # A fresh run must not need network access even if a workstation has
        # cloud packages or credentials installed.
        with patch("socket.socket.connect", side_effect=AssertionError("network forbidden")), patch("socket.getaddrinfo", side_effect=AssertionError("DNS forbidden")):
            cls.report = demo.run_demo(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_fixed_small_workload_and_output_schema(self):
        report = self.report
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["schema"], "policy_cce_synthetic_demo_result_v1")
        self.assertEqual(report["config"]["profile_count"], 27)
        self.assertEqual(report["config"]["environment"]["horizon"], 8)
        self.assertEqual(report["config"]["workers"], 1)
        self.assertLessEqual(report["simulated_episodes"], 432)
        self.assertEqual(set(report["solvers"]), {"FullTensor-CCE-LP", "DSS-CCE"})
        self.assertNotIn("campaign_sha256", report)
        self.assertNotIn("job_id", report)

    def test_frozen_q_and_closure_agree_before_and_after_evaluation(self):
        for solver in self.report["solvers"].values():
            q = solver["distribution"]
            audit = solver["independent_evaluation"]
            self.assertEqual(q["q_hash"], audit["q_hash_before"])
            self.assertEqual(q["q_hash"], audit["q_hash_after"])
            self.assertAlmostEqual(sum(q["probabilities"]), 1.0, places=12)
            self.assertTrue(all(value > 0 for value in q["probabilities"]))
            for check in (solver["training_verification"], audit):
                self.assertTrue(check["checks_agree"])
                self.assertAlmostEqual(check["full_tensor_gap"], check["closure_gap"], places=8)
            self.assertAlmostEqual(audit["statistics"]["nominal_gap"], audit["full_tensor_gap"], places=8)
            self.assertEqual(audit["statistics"]["audit_rollouts"], 8)

    def test_solvers_use_separate_empty_caches_and_honest_counts(self):
        full = self.report["solvers"]["FullTensor-CCE-LP"]
        self.assertEqual(full["training_profiles_evaluated"], 27)
        self.assertEqual(full["training_episodes"], 108)
        for solver in self.report["solvers"].values():
            self.assertTrue(solver["empty_cache_at_start"])
            self.assertEqual(solver["training_episodes"], solver["training_profiles_evaluated"] * 4)
            self.assertGreater(solver["solver_wall_seconds"], 0)
        self.assertGreater(self.report["independent_evaluation_seconds"], 0)
        self.assertGreater(self.report["training_verification_seconds"], 0)

    def test_samples_saved_with_fixed_shapes_and_checksum(self):
        payload = (self.output / "demo-samples.npz").read_bytes()
        self.assertEqual(sha256(payload).hexdigest(), self.report["artifacts"]["demo-samples.npz"])
        with np.load(self.output / "demo-samples.npz", allow_pickle=False) as arrays:
            self.assertEqual(arrays["profiles"].shape, (27, 3))
            self.assertEqual(arrays["training_mean_returns"].shape, (27, 3))
            self.assertEqual(arrays["training_ci_radius"].shape, (27, 3))
            self.assertEqual(arrays["training_objectives"].shape, (27,))
            self.assertEqual(arrays["independent_profile_returns"].shape, (27, 8, 3))
            self.assertTrue(np.isfinite(arrays["independent_profile_returns"]).all())
        self.assertEqual(json.loads((self.output / "demo-result.json").read_text()), self.report)

    def test_new_streams_keep_population_but_separate_actual_random_generators(self):
        training = demo._streams("training")
        evaluation = demo._streams("evaluation")
        self.assertEqual(training.type_seed, evaluation.type_seed)
        actual_seeds = []
        for stage, count in ((training, demo.TRAIN_ROLLOUTS), (evaluation, demo.EVALUATION_ROLLOUTS)):
            for replication in range(count):
                offset = stage.rollout_replication_seed * 100_000 + 10_000 * replication
                actual_seeds.extend(getattr(stage, name) + offset for name in ("order_seed", "tie_break_seed", "outside_seed", "availability_seed"))
        self.assertEqual(len(actual_seeds), len(set(actual_seeds)))
        self.assertEqual(training, demo._streams("training"))
        self.assertNotEqual(demo._seed("dss-search"), demo._seed("evaluation:bootstrap"))
        with self.assertRaises(ValueError):
            demo._streams("formal")

    def test_existing_output_is_refused_without_modification(self):
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        with self.assertRaises(FileExistsError):
            demo.run_demo(self.output)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir()})

    def test_symlink_and_source_destinations_are_refused(self):
        alias = Path(self.temp.name) / "alias"
        alias.symlink_to(self.output, target_is_directory=True)
        with self.assertRaises(FileExistsError):
            demo.run_demo(alias)
        source_destination = Path(demo.__file__).resolve().parent / "test-must-not-create"
        with self.assertRaises(ValueError):
            demo.run_demo(source_destination)
        self.assertFalse(source_destination.exists())

    def test_failure_is_preserved_in_new_output(self):
        failed = Path(self.temp.name) / "failed"
        with patch.object(demo, "solve_full_cce_lp", side_effect=RuntimeError("test LP failure")):
            with self.assertRaisesRegex(RuntimeError, "test LP failure"):
                demo.run_demo(failed)
        failure = json.loads((failed / "demo-error.json").read_text())
        self.assertEqual(failure["status"], "error")
        self.assertEqual(failure["message"], "test LP failure")
        self.assertTrue((failed / "demo-config.json").is_file())
        self.assertFalse((failed / "demo-result.json").exists())

    def test_gap_disagreement_is_not_hidden(self):
        game = demo.EmpiricalGame(
            profiles=(("A1",), ("A2",)), policy_ids=("A1", "A2"),
            payoffs=np.array([[0.0], [1.0]]), ci_radius=np.zeros((2, 1)),
            objectives=np.zeros(2), metrics=({}, {}),
        )
        q = demo.freeze_distribution("test", (("A1",),), (1.0,), 1, ("A1", "A2"))
        with patch.object(demo, "compute_cce_gap_from_deviation_closure", return_value=0.0):
            with self.assertRaisesRegex(ValueError, "disagree"):
                demo._verify(game, q)

    def test_second_directory_reproduces_science_not_elapsed_time(self):
        again = demo.run_demo(Path(self.temp.name) / "second-run")
        expected = deepcopy(self.report)
        for report in (again, expected):
            for key in ("workflow_wall_seconds", "training_verification_seconds", "independent_evaluation_seconds"):
                report.pop(key)
            for solver in report["solvers"].values():
                solver.pop("solver_wall_seconds")
        self.assertEqual(again, expected)


if __name__ == "__main__":
    unittest.main()
