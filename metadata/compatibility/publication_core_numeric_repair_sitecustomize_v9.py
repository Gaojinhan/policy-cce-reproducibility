"""Auditable numerical repair shim for the terminal publication-core jobs.

The frozen formal source remains unchanged.  When explicitly enabled by the
signed terminal-recovery launcher, this module is loaded as ``sitecustomize``
inside the job subprocess.  It only intervenes when HiGHS returns a stage-2
distribution that exceeds the existing post-solve feasibility allowance.

The repair performs a deterministic minimum-L1 projection of the stage-2
candidate onto the unchanged CCE face.  A convex step toward the unchanged
stage-1 minimax solution is retained only as a fail-safe.  No payoff, seed,
profile, selector objective, job identity, or canonical output path is changed.
Each activation is appended to a job-local diagnostic that the routed wrapper
includes in executor provenance.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linprog


REPAIR_ID = "publication-core-stage2-convex-feasibility-v9"
_MIN_FEASIBILITY_ALLOWANCE = 1.0e-8
_CANONICALIZATION_ALLOWANCE = 1.0e-9


def _normalized(values: np.ndarray) -> np.ndarray:
    q = np.maximum(np.asarray(values, dtype=float), 0.0)
    total = float(np.sum(q))
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("Numerical repair received an invalid distribution.")
    return q / total


def _gains(a: np.ndarray, q: np.ndarray) -> np.ndarray:
    if a.shape[0] == 0:
        return np.empty(0, dtype=float)
    return np.asarray(a @ q, dtype=float)


def _realized_epsilon(a: np.ndarray, q: np.ndarray) -> float:
    gains = _gains(a, q)
    return 0.0 if gains.size == 0 else max(0.0, float(np.max(gains)))


def _stage1_distribution(a: np.ndarray) -> tuple[np.ndarray, float]:
    n_profiles = int(a.shape[1])
    objective = np.zeros(n_profiles + 1, dtype=float)
    objective[-1] = 1.0
    a_ub = np.hstack([a, -np.ones((a.shape[0], 1), dtype=float)])
    a_eq = np.zeros((1, n_profiles + 1), dtype=float)
    a_eq[0, :n_profiles] = 1.0
    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=np.zeros(a.shape[0], dtype=float),
        A_eq=a_eq,
        b_eq=np.array([1.0], dtype=float),
        bounds=[(0.0, None)] * n_profiles + [(0.0, None)],
        method="highs",
        options={
            "primal_feasibility_tolerance": 1.0e-9,
            "dual_feasibility_tolerance": 1.0e-9,
        },
    )
    if not result.success:
        raise RuntimeError(f"Numerical repair stage 1 failed: {result.message}")
    q = _normalized(result.x[:n_profiles])
    epsilon_star = max(0.0, float(result.x[-1]), _realized_epsilon(a, q))
    return q, epsilon_star


def _minimum_feasible_blend(
    a: np.ndarray,
    candidate: np.ndarray,
    stage1_q: np.ndarray,
    epsilon_bound: float,
) -> tuple[np.ndarray, float]:
    """Return the closest point on the candidate/stage-1 line to candidate."""

    q = _normalized(candidate)
    base = _normalized(stage1_q)
    candidate_gains = _gains(a, q)
    base_gains = _gains(a, base)
    if candidate_gains.size == 0 or float(np.max(candidate_gains)) <= epsilon_bound:
        return q, 0.0
    if float(np.max(base_gains)) > epsilon_bound + _CANONICALIZATION_ALLOWANCE:
        raise RuntimeError("Numerical repair stage-1 anchor is outside the frozen epsilon bound.")

    alpha = 0.0
    for candidate_gain, base_gain in zip(candidate_gains, base_gains, strict=True):
        if candidate_gain <= epsilon_bound:
            continue
        denominator = float(candidate_gain - base_gain)
        if denominator <= 0.0:
            raise RuntimeError("Numerical repair cannot reach the frozen epsilon face.")
        alpha = max(alpha, float((candidate_gain - epsilon_bound) / denominator))
    alpha = min(1.0, max(0.0, float(np.nextafter(alpha, 1.0))))
    repaired = _normalized((1.0 - alpha) * q + alpha * base)

    # Floating-point rounding can leave the exact boundary a few ulps high.
    # Move monotonically toward the valid stage-1 anchor until the same
    # allowance used by canonicalization is met.
    if _realized_epsilon(a, repaired) > epsilon_bound + _CANONICALIZATION_ALLOWANCE:
        low = alpha
        high = 1.0
        for _ in range(80):
            mid = 0.5 * (low + high)
            trial = _normalized((1.0 - mid) * q + mid * base)
            if _realized_epsilon(a, trial) <= epsilon_bound:
                high = mid
                repaired = trial
            else:
                low = mid
        alpha = high
    return repaired, alpha


def _minimum_l1_projection(
    a: np.ndarray,
    candidate: np.ndarray,
    epsilon_bound: float,
) -> tuple[np.ndarray, float]:
    """Project onto the frozen CCE face with minimum total probability move."""

    q = _normalized(candidate)
    n_profiles = int(q.size)
    n_rows = int(a.shape[0])
    # Variables are (x, d), with d >= |x - q|.  Scaling only the deviation
    # rows makes the explicit 1e-9 HiGHS primal tolerance much smaller in the
    # original payoff units while leaving the feasible set mathematically
    # identical.
    deviation_scale = 1.0e6
    objective = np.concatenate(
        [np.zeros(n_profiles, dtype=float), np.ones(n_profiles, dtype=float)]
    )
    a_ub = np.vstack(
        [
            np.hstack(
                [
                    deviation_scale * a,
                    np.zeros((n_rows, n_profiles), dtype=float),
                ]
            ),
            np.hstack([np.eye(n_profiles), -np.eye(n_profiles)]),
            np.hstack([-np.eye(n_profiles), -np.eye(n_profiles)]),
        ]
    )
    b_ub = np.concatenate(
        [
            np.full(n_rows, deviation_scale * epsilon_bound, dtype=float),
            q,
            -q,
        ]
    )
    a_eq = np.zeros((1, 2 * n_profiles), dtype=float)
    a_eq[0, :n_profiles] = 1.0
    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=np.array([1.0], dtype=float),
        bounds=[(0.0, None)] * (2 * n_profiles),
        method="highs",
        options={
            "primal_feasibility_tolerance": 1.0e-9,
            "dual_feasibility_tolerance": 1.0e-9,
        },
    )
    if not result.success:
        raise RuntimeError(f"Numerical repair L1 projection failed: {result.message}")
    projected = _normalized(result.x[:n_profiles])
    return projected, float(np.sum(np.abs(projected - q)))


def _append_activation(record: dict[str, Any]) -> None:
    state_dir = os.environ.get("CMFG_STATE_DIR")
    if not state_dir:
        return
    path = Path(state_dir) / "numeric_repair_v9.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def install() -> None:
    from cmfg_cce.solvers import cce_lp

    if getattr(cce_lp, "_publication_core_numeric_repair_v9", False):
        return
    original = cce_lp.canonicalize_cce_distribution

    def repaired_canonicalize(
        values: np.ndarray,
        deviation_matrix: np.ndarray,
        epsilon_bound: float,
        *,
        probability_tolerance: float = 1.0e-9,
    ) -> np.ndarray:
        a = np.asarray(deviation_matrix, dtype=float)
        q = _normalized(values)
        before = _realized_epsilon(a, q)
        allowed = float(epsilon_bound) + _MIN_FEASIBILITY_ALLOWANCE
        if before <= allowed:
            return original(
                q,
                a,
                epsilon_bound,
                probability_tolerance=probability_tolerance,
            )

        stage1_q, epsilon_star = _stage1_distribution(a)
        repaired, l1_distance = _minimum_l1_projection(
            a, q, float(epsilon_bound)
        )
        fallback_alpha = 0.0
        if (
            _realized_epsilon(a, repaired)
            > float(epsilon_bound) + _CANONICALIZATION_ALLOWANCE
        ):
            repaired, fallback_alpha = _minimum_feasible_blend(
                a, repaired, stage1_q, float(epsilon_bound)
            )
        canonical = original(
            repaired,
            a,
            epsilon_bound,
            probability_tolerance=probability_tolerance,
        )
        after = _realized_epsilon(a, canonical)
        if after > allowed:
            raise RuntimeError("Numerical repair did not restore the frozen epsilon bound.")
        record = {
            "repair_id": REPAIR_ID,
            "rows": int(a.shape[0]),
            "columns": int(a.shape[1]),
            "epsilon_star": float(epsilon_star),
            "epsilon_bound": float(epsilon_bound),
            "realized_epsilon_before": float(before),
            "realized_epsilon_after": float(after),
            "l1_probability_distance": float(l1_distance),
            "fallback_convex_blend_alpha": float(fallback_alpha),
        }
        _append_activation(record)
        print(
            "CMFG_NUMERIC_REPAIR_ACTIVATED "
            + json.dumps(record, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
        return canonical

    cce_lp.canonicalize_cce_distribution = repaired_canonicalize
    cce_lp._publication_core_numeric_repair_v9 = True


if os.environ.get("CMFG_NUMERIC_REPAIR_V9") == "1":
    install()
