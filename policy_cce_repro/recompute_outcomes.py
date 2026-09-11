"""Recompute CNC outcome contrasts from saved paired rollout records, offline.

No fitting, simulation, cloud access, author-machine paths or manuscript builders
are used. The frozen evaluation module supplies raw-count metric definitions.
Ported reporting-method lineage and exact historical source hashes are recorded
in docs/OUTCOME_METHOD_PROVENANCE.md; this module is a portability refactor, not
a byte-identical copy of the historical writing scripts.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from cmfg_cce.evaluation.independent_audit import distribution_hash
from cmfg_cce.evaluation.revision_statistics import (
    weighted_outcome_replications,
    aggregate_outcome_replications,
    _reconstruct_outcome_metric_samples,
)

ROLLOUTS, BOOTSTRAPS, BOOTSTRAP_SEED = 500, 5000, 2026090902
ATOL, RTOL = 1e-9, 1e-10
MECHANISMS = ("M1_price_first", "M2_price_critical", "M3_delivery_first", "M4_delivery_critical")
CONTRASTS = ((0, 1), (2, 3), (0, 2), (1, 3))
AVAIL = "at_least_three_route_capacity_feasible_manufacturers_rate"
PRIMARY = ("payment_per_assignment", "manufacturer_discounted_profit", "conditional_bid_rate",
           "assignment_rate", "normalized_winner_hhi", "average_markup")
RELATIVE_METRICS = PRIMARY + ("submitted_lead_time_multiplier",)
ROUTE = ("orders_with_scalar_route_mismatch_rate", "capability_feasible_manufacturers_per_order",
         "scalar_capacity_feasible_manufacturers_per_order", "route_capacity_feasible_manufacturers_per_order",
         AVAIL, "conditional_route_false_positive_rate",
         *(f"machine_group_remaining_capacity_{g}" for g in ("M5", "G", "EDM", "T", "M3")),
         *(f"conditional_route_false_positive_rate_F{i}" for i in range(1, 5)), AVAIL + "_F4")
METRICS = tuple(sorted(set(PRIMARY + ROUTE + ("platform_total_payment", "submitted_lead_time_multiplier"))))
FACTORS = (("platform_load", 0, "nominal", "high"),
           ("route_mix", 1, "balanced", "m5_edm_intensive"),
           ("outside_load", 2, "normal", "high_outside_m5_g_edm"))
ENDPOINT_METRICS = ("assignment_rate", AVAIL, *(AVAIL + f"_F{i}" for i in range(1, 5)))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array_hash(values):
    h = hashlib.sha256()
    for name in sorted(values):
        a = np.ascontiguousarray(np.asarray(values[name]))
        for piece in (name.encode(), str(a.dtype).encode(), json.dumps(a.shape, separators=(",", ":")).encode()):
            h.update(piece)
            h.update(b"\0")
        h.update(a.tobytes(order="C"))
    return h.hexdigest()


class _Checks:
    def __init__(self):
        self.count, self.max_absolute_error, self.mismatches = 0, 0.0, []

    def compare(self, actual, expected, path):
        """Compare predeclared reference fields; never change computed values."""
        if isinstance(expected, dict):
            for key, value in expected.items():
                if key not in actual:
                    self.mismatches.append({"field": path + "/" + key, "reason": "missing field"})
                else:
                    self.compare(actual[key], value, path + "/" + key)
        elif isinstance(expected, (list, tuple)):
            if len(actual) != len(expected):
                self.mismatches.append({"field": path, "reason": "length mismatch"})
            else:
                for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
                    self.compare(a, b, path + f"/{i}")
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            self.count += 1
            error = abs(float(actual) - expected)
            if np.isfinite(error):
                self.max_absolute_error = max(self.max_absolute_error, error)
            if not np.isfinite(actual) or not np.isclose(actual, expected, atol=ATOL, rtol=RTOL):
                self.mismatches.append({"field": path, "actual": float(actual), "expected": expected,
                                        "absolute_error": float(error)})
        elif actual != expected:
            self.mismatches.append({"field": path, "actual": actual, "expected": expected})


def _bootstrap_weights(*, replications=ROLLOUTS, draws=BOOTSTRAPS, seed=BOOTSTRAP_SEED):
    children = np.random.SeedSequence(seed).spawn(3)
    return {s: np.random.default_rng(children[s]).multinomial(
        replications, np.full(replications, 1 / replications), size=draws).astype(float) for s in range(3)}


def _bootstrap_metrics(records, weights):
    names = tuple(sorted(records))
    matrix = np.column_stack([records[n] for n in names])
    _require(matrix.shape[0] == weights.shape[1], "Bootstrap replication count mismatch")
    totals = weights @ matrix
    return _reconstruct_outcome_metric_samples(
        {name: totals[:, i] for i, name in enumerate(names)}, n_manufacturers=4,
        sample_sizes=np.full(len(weights), weights.shape[1], dtype=float))


def _interval(point, draws):
    _require(np.isfinite(point) and np.all(np.isfinite(draws)), "Nonfinite estimate or bootstrap draw")
    return {"estimate": float(point), "ci_lower": float(np.quantile(draws, .025)),
            "ci_upper": float(np.quantile(draws, .975)), "standard_error": float(np.std(draws, ddof=1))}


def _relative_interval(base, target, base_draws, target_draws):
    _require(base > 0 and np.all(base_draws > 0), "Relative-effect reference means must be positive")
    return _interval(100 * (target / base - 1), 100 * (target_draws / base_draws - 1))


def _load_case(dataset, jid):
    result = dataset.result(jid)
    _require(result["status"] == "complete" and not result["smoke"], f"Invalid result: {jid}")
    outcome = result["outcome_evaluation"]
    _require(result["outcome_rollouts"] == outcome["rollouts"] == ROLLOUTS, "Expected 500 outcome rollouts")
    q = result["distribution"]
    _require(q == result["distributions"]["platform_operating_score"], "Selected q mismatch")
    support = tuple(tuple(p) for p in q["support"])
    _require(distribution_hash(support, q["probabilities"]) == q["q_hash"], "q hash mismatch")
    profiles = tuple(sorted({tuple(p) for d in result["distributions"].values() for p in d["support"]}))
    item_ids = ["|".join(p) for p in profiles]
    order_hash = hashlib.sha256(json.dumps(item_ids, separators=(",", ":")).encode()).hexdigest()
    stage = f"jobs/{jid}/stages/outcome_evaluation/chunks"
    plan = dataset.object_json(f"{stage}/chunk_plan.json")
    _require(plan["job_id"] == jid + "::outcome_evaluation" and plan["item_ids"] == item_ids, "Outcome stage plan mismatch")
    _require(plan["item_order_sha256"] == outcome["checkpoint"]["item_order_sha256"] == order_hash, "Outcome profile order mismatch")
    _require(outcome["checkpoint"]["common_replication_index_across_profiles"], "Outcome pairing absent")
    records, returns, seen = {}, {}, set()
    for chunk in sorted(outcome["checkpoint"]["chunks"], key=lambda c: c["start"]):
        stem = f"{stage}/{chunk['chunk_id']}"
        _require(dataset.object_json(stem + ".json") == chunk, "Chunk descriptor mismatch")
        raw = dataset.object_bytes(stem + ".npz")
        _require(len(raw) == chunk["payload_size"] and hashlib.sha256(raw).hexdigest() == chunk["payload_sha256"], "Chunk byte/hash mismatch")
        payload = dataset.object_arrays(stem + ".npz")
        _require(set(payload) == set(chunk["arrays"]), "Chunk member names mismatch")
        for name, spec in chunk["arrays"].items():
            a = payload[name]
            _require(str(a.dtype) == spec["dtype"] and list(a.shape) == spec["shape"], "Chunk dtype/shape mismatch")
            _require(np.all(np.isfinite(a)), "Nonfinite outcome payload")
        _require(int(payload["rollout_count"][0]) == ROLLOUTS, "Wrong outcome rollout count")
        _require(np.array_equal(payload["profile_indices"], np.arange(chunk["start"], chunk["stop"])), "Chunk profile-index mismatch")
        for offset, index in enumerate(payload["profile_indices"]):
            _require(int(index) not in seen, "Duplicate outcome profile")
            seen.add(int(index))
            profile = profiles[int(index)]
            returns[profile] = payload["returns"][offset].copy()
            records[profile] = {name[len("metric__"):]: payload[name][offset].copy()
                                for name in payload if name.startswith("metric__")}
    _require(seen == set(range(len(profiles))), "Missing outcome profile")
    policy_index = {p: i for i, p in enumerate(result["policy_ids"])}
    arrays = {"profile_indices": np.asarray([[policy_index[p] for p in profile] for profile in profiles], dtype=np.int16),
              "returns": np.stack([returns[p] for p in profiles])}
    for name in sorted(records[profiles[0]]):
        arrays["metric__" + name] = np.stack([records[p][name] for p in profiles])
    _require(_array_hash(arrays) == outcome["raw_profile_vectors_sha256"], "Raw outcome vector hash mismatch")
    weighted = weighted_outcome_replications(records, support, q["probabilities"])
    expected = outcome["distributions"]["platform_operating_score"]
    _require(_array_hash(weighted) == expected["weighted_raw_vectors_sha256"], "Weighted raw outcome hash mismatch")
    point = aggregate_outcome_replications(weighted, n_manufacturers=4)
    return result, weighted, point


def _factor_pairs(points, dimension, low, high):
    conditions = sorted({key[1] for key in points})
    pairs = []
    for seed in range(3):
        for condition in conditions:
            parts = condition.split("__")
            if parts[dimension] == low:
                parts[dimension] = high
                pairs.extend(((seed, condition, m), (seed, "__".join(parts), m)) for m in MECHANISMS)
    _require(len(pairs) == 48 and all(a in points and b in points for a, b in pairs), "Expected 48 complete factor pairs")
    return pairs


def _grouped_policy_shares(result, point):
    composition = {p: 0.0 for p in result["policy_ids"]}
    for profile, probability in zip(result["distribution"]["support"], result["distribution"]["probabilities"], strict=True):
        for policy in profile:
            composition[policy] += float(probability) / 4
    winners = {p: point[f"winner_policy_{p}_share"] for p in result["policy_ids"]}
    _require(set(composition) == {f"A{i}" for i in range(1, 7)}, "CNC core policy library mismatch")
    _require(np.isclose(sum(composition.values()), 1) and np.isclose(sum(winners.values()), 1), "Policy shares do not sum to one")
    return {"composition": [composition["A1"], composition["A2"], sum(composition[f"A{i}"] for i in range(3, 7))],
            "winner_shares": [winners["A1"], winners["A2"], sum(winners[f"A{i}"] for i in range(3, 7))]}


def _descriptive(cases):
    levels, policy_rows, endpoints, endpoints_by_mechanism = [], [], [], []
    for mechanism in MECHANISMS:
        selected = [c for c in cases if c[0]["mechanism"] == mechanism]
        _require(len(selected) == 24, "Expected 24 cases per mechanism")
        levels.append({"mechanism": mechanism, "n_cases": 24, "metrics": {
            m: {"mean": float(np.mean([c[2][m] for c in selected])),
                "sample_sd": float(np.std([c[2][m] for c in selected], ddof=1))} for m in RELATIVE_METRICS}})
        grouped = [_grouped_policy_shares(c[0], c[2]) for c in selected]
        policy_rows.append({"mechanism": mechanism, **{name: np.mean([g[name] for g in grouped], axis=0).tolist()
                                                     for name in ("composition", "winner_shares")}})
    for condition, label in (("nominal__balanced__normal", "Reference"),
                             ("high__m5_edm_intensive__high_outside_m5_g_edm", "Combined pressure")):
        selected = [c for c in cases if c[0]["condition"] == condition]
        _require(len(selected) == 12, "Expected 12 endpoint cases")
        endpoints.append({"label": label, "condition": condition, "n_cases": 12,
                          "metrics": {m: float(np.mean([c[2][m] for c in selected])) for m in ENDPOINT_METRICS}})
        for mechanism in MECHANISMS:
            group = [c for c in selected if c[0]["mechanism"] == mechanism]
            _require(len(group) == 3, "Expected three endpoint strata")
            endpoints_by_mechanism.append({"condition": condition, "mechanism": mechanism,
                "metrics": {m: float(np.mean([c[2][m] for c in group])) for m in ENDPOINT_METRICS}})
    return levels, policy_rows, endpoints, endpoints_by_mechanism


def recompute_outcomes(dataset, output_dir, *, workers=1):
    """Recalculate all CNC paired contrasts and return a full-precision report."""
    _require(isinstance(workers, int) and workers >= 1, "workers must be a positive integer")
    output_dir = dataset.validate_output_dir(output_dir)
    jobs = sorted(jid for jid, spec in dataset.jobs.items() if spec["family"] == "cnc_main")
    _require(len(jobs) == 96, "The selected dataset must contain all 96 CNC cases")
    checks = _Checks()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        cases = list(pool.map(lambda jid: _load_case(dataset, jid), jobs))
    weights = _bootstrap_weights()
    points, draws, stream_ids = {}, {}, {}
    for result, records, point in cases:
        seed = int(result["seed"])
        _require(seed in weights, "Unexpected fitted-q stratum")
        key = (seed, result["condition"], result["mechanism"])
        _require(key not in points, "Duplicate paired case")
        identity = json.dumps(result["outcome_evaluation"]["seeds"], sort_keys=True)
        _require(seed not in stream_ids or stream_ids[seed] == identity, "Pairing streams differ within stratum")
        stream_ids[seed] = identity
        points[key] = point
        for name, metric in result["outcome_evaluation"]["distributions"]["platform_operating_score"]["metrics"].items():
            if metric["estimate"] is not None:
                checks.compare(point[name], metric["estimate"], f"case/{result['job_id']}/{name}")
    _require(len(set(stream_ids.values())) == 3, "Fitted-q strata must have distinct streams")
    conditions = sorted({k[1] for k in points})
    _require(len(conditions) == 8, "Expected eight operating conditions")
    def one_bootstrap(case):
        r, records, _ = case
        return (int(r["seed"]), r["condition"], r["mechanism"]), _bootstrap_metrics(records, weights[int(r["seed"])])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for key, value in pool.map(one_bootstrap, cases):
            draws[key] = value
    mechanism, relative, factors = [], [], []
    for li, ri in CONTRASTS:
        left, right = MECHANISMS[li], MECHANISMS[ri]
        contrast = f"AUC{ri + 1}-AUC{li + 1}"
        pairs = [((s, c, left), (s, c, right)) for s in range(3) for c in conditions]
        _require(len(pairs) == 24 and all(a in points and b in points for a, b in pairs), "Incomplete mechanism pairs")
        metrics = {}
        for metric in METRICS:
            if metric == "submitted_lead_time_multiplier" and (li, ri) != (2, 3):
                continue
            metrics[metric] = _interval(np.mean([points[b][metric] - points[a][metric] for a, b in pairs]),
                np.mean([draws[b][metric] - draws[a][metric] for a, b in pairs], axis=0))
        mechanism.append({"contrast": contrast, "left": left, "right": right, "metrics": metrics})
        rel_metrics = {}
        for metric in RELATIVE_METRICS:
            if metric == "submitted_lead_time_multiplier" and (li, ri) != (2, 3):
                continue
            base = np.mean([points[a][metric] for a, b in pairs])
            target = np.mean([points[b][metric] for a, b in pairs])
            bd = np.mean([draws[a][metric] for a, b in pairs], axis=0)
            td = np.mean([draws[b][metric] for a, b in pairs], axis=0)
            rel_metrics[metric] = {**_relative_interval(base, target, bd, td), "reference_mean": float(base),
                                   "target_mean": float(target), "baseline_bootstrap_min": float(bd.min())}
        relative.append({"contrast": contrast, "reference": left, "target": right, "metrics": rel_metrics})
    for name, dimension, low, high in FACTORS:
        pairs = _factor_pairs(points, dimension, low, high)
        factors.append({"factor": name, "low": low, "high": high, "metrics": {
            m: _interval(np.mean([points[b][m] - points[a][m] for a, b in pairs]),
                         np.mean([draws[b][m] - draws[a][m] for a, b in pairs], axis=0))
            for m in METRICS if m != "submitted_lead_time_multiplier"}})
    levels, policies, endpoints, endpoints_by_mechanism = _descriptive(cases)
    route_mix = next(r for r in factors if r["factor"] == "route_mix")
    family_rows = [{"family": f"F{i}", "metric": f"conditional_route_false_positive_rate_F{i}",
        **{k: 100 * v for k, v in route_mix["metrics"][f"conditional_route_false_positive_rate_F{i}"].items()},
        "unit": "percentage_points", "point_reconstructed_from_canonical_cache": route_mix["metrics"][f"conditional_route_false_positive_rate_F{i}"]["estimate"]}
        for i in range(1, 5)]
    for filename in ("cnc-paired-mc-summary.json", "cnc-original-style-evidence.json"):
        expected = dataset.reference(filename)
        for section, rows in (("mechanism_effects", mechanism), ("operating_factor_effects", factors)):
            checks.compare(rows, expected[section], f"{filename}/{section}")
    reference = dataset.reference("percent-effects.json")
    checks.compare(relative, reference["effects"], "percent-effects.json/effects")
    reference = dataset.reference("cnc-original-style-evidence.json")
    checks.compare(levels, reference["mechanism_levels"], "cnc-original-style-evidence.json/mechanism_levels")
    checks.compare(policies, reference["policy_composition"], "cnc-original-style-evidence.json/policy_composition")
    reference = dataset.reference("cnc-restoration-evidence.json")
    checks.compare(endpoints, reference["endpoint_means"], "cnc-restoration-evidence.json/endpoint_means")
    checks.compare(endpoints_by_mechanism, reference["endpoint_means_by_mechanism"], "cnc-restoration-evidence.json/endpoint_means_by_mechanism")
    checks.compare(family_rows, reference["family_mismatch"]["family_rows"], "cnc-restoration-evidence.json/family_mismatch")
    summary = {"schema": "offline_cnc_outcome_recomputation_v1", "status": "pass" if not checks.mismatches else "fail",
        "case_count": 96, "raw_vector_hashes_verified": 96, "weighted_vector_hashes_verified": 96,
        "comparison_count": checks.count, "mismatch_count": len(checks.mismatches), "mismatches": checks.mismatches,
        "max_absolute_error": checks.max_absolute_error, "tolerance": {"atol": ATOL, "rtol": RTOL},
        "method": {"rollouts": ROLLOUTS, "bootstrap_draws": BOOTSTRAPS, "bootstrap_seed": BOOTSTRAP_SEED,
            "fixed_q_strata": [0, 1, 2], "resample_fitted_q_strata": False,
            "pairing": "Shared replication weights within each stratum across all mechanisms and conditions",
            "aggregation": "Reconstruct case ratios/HHI from resampled raw totals, then average paired cells equally",
            "relative_formula": "100 * (equal-case target mean / equal-case reference mean - 1)",
            "interval": "pointwise two-sided 95% percentile bootstrap; conditional on fixed cases and q"},
        "mechanism_effects": mechanism, "relative_mechanism_effects": relative,
        "operating_factor_effects": factors, "mechanism_levels": levels, "policy_composition": policies,
        "endpoint_means": endpoints, "endpoint_means_by_mechanism": endpoints_by_mechanism,
        "family_mismatch": family_rows}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "outcomes-recomputed.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return summary
