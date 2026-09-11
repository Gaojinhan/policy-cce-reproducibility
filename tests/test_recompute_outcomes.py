"""Risk checks for fixed-q CNC outcome recomputation (no simulation)."""
import hashlib
import io
import json
import unittest

import numpy as np

from cmfg_cce.evaluation.independent_audit import distribution_hash
from policy_cce_repro.recompute_outcomes import (
    _array_hash, _bootstrap_weights, _bootstrap_metrics, _relative_interval,
    _factor_pairs, _Checks, _load_case, MECHANISMS,
    recompute_outcomes,
)


def raw_counts():
    return {
        "orders_offered_count": np.array([1., 9.]),
        "assignment_count": np.array([1., 3.]),
        "invitation_count": np.array([2., 10.]),
        "valid_bid_count": np.array([1., 5.]),
        "platform_total_payment": np.array([10., 90.]),
        "manufacturer_discounted_profit_sum": np.array([2., 6.]),
        **{f"wins_manufacturer_{i}_count": np.array([float(i == 0), float(i < 3)]) for i in range(4)},
    }


class PairedOutcomeTests(unittest.TestCase):
    def test_common_draws_cancel_shared_noise_but_independent_draws_do_not(self):
        weights = _bootstrap_weights(replications=6, draws=200)
        signal = np.array([0., 1., 2., 4., 8., 16.])
        delta = weights[0] @ (signal + 7) / 6 - weights[0] @ signal / 6
        np.testing.assert_allclose(delta, 7, atol=1e-12)
        independent_delta = weights[1] @ (signal + 7) / 6 - weights[0] @ signal / 6
        self.assertGreater(float(np.std(independent_delta)), 0)
        self.assertFalse(np.array_equal(weights[0], weights[1]))

    def test_raw_total_ratios_and_hhi_are_reconstructed_inside_each_draw(self):
        values = _bootstrap_metrics(raw_counts(), np.array([[1., 1.], [2., 0.]]))
        self.assertAlmostEqual(values["assignment_rate"][0], .4)
        self.assertAlmostEqual(values["payment_per_assignment"][0], 25)
        self.assertAlmostEqual(values["manufacturer_discounted_profit"][0], 4)
        self.assertAlmostEqual(values["normalized_winner_hhi"][0], 1 / 6)
        self.assertAlmostEqual(values["normalized_winner_hhi"][1], 1)
        self.assertNotAlmostEqual(values["assignment_rate"][0], np.mean([1., 1 / 3]))

    def test_relative_effect_is_ratio_of_case_means_with_resampled_denominator(self):
        baseline, target = np.array([10., 100.]), np.array([20., 110.])
        bd, td = np.array([50., 55., 60.]), np.array([60., 65., 70.])
        result = _relative_interval(baseline.mean(), target.mean(), bd, td)
        self.assertAlmostEqual(result["estimate"], 100 * (65 / 55 - 1))
        self.assertNotAlmostEqual(result["estimate"], np.mean(100 * (target / baseline - 1)))
        self.assertAlmostEqual(result["ci_lower"], np.quantile(100 * (td / bd - 1), .025))
        self.assertNotAlmostEqual(result["ci_lower"], np.quantile(100 * (td / 55 - 1), .025))

    def test_nonpositive_bootstrap_denominator_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            _relative_interval(1., 2., np.array([1., 0.]), np.array([2., 2.]))

    def test_bootstrap_weights_are_reproducible_and_have_exact_replication_mass(self):
        a, b = _bootstrap_weights(replications=6, draws=100), _bootstrap_weights(replications=6, draws=100)
        for seed in range(3):
            np.testing.assert_array_equal(a[seed], b[seed])
            np.testing.assert_array_equal(a[seed].sum(axis=1), 6)

    def test_factor_pairs_hold_other_conditions_and_mechanism_fixed(self):
        conditions = ["__".join((load, mix, outside)) for load in ("nominal", "high")
                      for mix in ("balanced", "m5_edm_intensive") for outside in ("normal", "high_outside_m5_g_edm")]
        points = {(s, c, m): {} for s in range(3) for c in conditions for m in MECHANISMS}
        pairs = _factor_pairs(points, 1, "balanced", "m5_edm_intensive")
        self.assertEqual(len(pairs), 48)
        for left, right in pairs:
            self.assertEqual(left[0], right[0])
            self.assertEqual(left[2], right[2])
            self.assertEqual(left[1].split("__")[::2], right[1].split("__")[::2])
        del points[pairs[0][1]]
        with self.assertRaisesRegex(ValueError, "48 complete"):
            _factor_pairs(points, 1, "balanced", "m5_edm_intensive")

    def test_comparison_records_mismatch_instead_of_modifying_values(self):
        checks = _Checks()
        actual = {"estimate": 1.2}
        checks.compare(actual, {"estimate": 1., "ci_lower": .8}, "test")
        self.assertEqual(len(checks.mismatches), 2)
        self.assertEqual(actual, {"estimate": 1.2})
        checks.compare(1 + 1e-12, 1., "tolerated")
        self.assertEqual(len(checks.mismatches), 2)


class SmallDataset:
    """Real NPZ/descriptor seam with one deterministic 500-record profile."""
    def __init__(self):
        self.jid = "training__cnc_main__fixture"
        profile = ("A1",) * 4
        q = {"support": [list(profile)], "probabilities": [1.], "q_hash": distribution_hash([profile], [1.])}
        records = {k: np.tile(v, 250) for k, v in raw_counts().items()}
        arrays = {"profile_indices": np.array([0], dtype=np.int64), "rollout_count": np.array([500], dtype=np.int64),
                  "returns": np.zeros((1, 500, 4)), **{"metric__" + k: v[None, :] for k, v in records.items()}}
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        raw = buf.getvalue()
        chunk = {"chunk_id": "audit-000000000-000000001", "start": 0, "stop": 1,
                 "payload_size": len(raw), "payload_sha256": hashlib.sha256(raw).hexdigest(),
                 "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()}}
        ids = ["|".join(profile)]
        order = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        raw_vectors = {"profile_indices": np.zeros((1, 4), dtype=np.int16), "returns": arrays["returns"],
                       **{"metric__" + k: v[None, :] for k, v in records.items()}}
        self.r = {"status": "complete", "smoke": False, "distribution": q,
                  "distributions": {"platform_operating_score": q}, "outcome_rollouts": 500, "policy_ids": ["A1"],
                  "outcome_evaluation": {"rollouts": 500, "raw_profile_vectors_sha256": _array_hash(raw_vectors),
                    "distributions": {"platform_operating_score": {"weighted_raw_vectors_sha256": _array_hash(records)}},
                    "checkpoint": {"chunks": [chunk], "item_order_sha256": order, "common_replication_index_across_profiles": True}}}
        stage = f"jobs/{self.jid}/stages/outcome_evaluation/chunks"
        self.objects = {stage + "/chunk_plan.json": {"job_id": self.jid + "::outcome_evaluation", "item_ids": ids, "item_order_sha256": order},
                        stage + "/" + chunk["chunk_id"] + ".json": chunk}
        self.raw, self.arrays = raw, arrays

    def result(self, jid):
        return self.r

    def object_json(self, key):
        return self.objects[key]

    def object_bytes(self, key):
        return self.raw

    def object_arrays(self, key):
        return self.arrays


class OutcomeDatasetBoundaryTests(unittest.TestCase):
    def test_direct_api_validates_output_before_reading_data(self):
        class RejectingDataset:
            def validate_output_dir(self, path):
                raise ValueError("Output overlaps immutable input")
        with self.assertRaisesRegex(ValueError, "overlaps immutable input"):
            recompute_outcomes(RejectingDataset(), "forbidden-output")

    def test_case_loader_crosses_descriptor_raw_hash_weighted_hash_and_metric_boundary(self):
        dataset = SmallDataset()
        _, weighted, point = _load_case(dataset, dataset.jid)
        self.assertEqual(len(weighted["assignment_count"]), 500)
        self.assertAlmostEqual(point["assignment_rate"], .4)

    def test_wrong_stage_cannot_be_used_even_when_chunk_basename_matches(self):
        dataset = SmallDataset()
        next(v for k, v in dataset.objects.items() if k.endswith("chunk_plan.json"))["job_id"] += "::formal_audit"
        with self.assertRaisesRegex(ValueError, "stage plan"):
            _load_case(dataset, dataset.jid)

    def test_raw_byte_corruption_fails_before_metric_reconstruction(self):
        dataset = SmallDataset()
        dataset.raw = dataset.raw[:-1] + bytes([dataset.raw[-1] ^ 1])
        with self.assertRaisesRegex(ValueError, "byte/hash"):
            _load_case(dataset, dataset.jid)


if __name__ == "__main__":
    unittest.main()
