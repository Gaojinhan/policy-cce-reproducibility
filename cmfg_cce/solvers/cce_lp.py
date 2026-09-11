from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.optimize import linprog, minimize

from cmfg_cce.evaluation.empirical_game import EmpiricalGame
from cmfg_cce.evaluation.rollout import Profile


SelectorMode = Literal[
    "platform_operating_score",
    "total_manufacturer_return",
    "manufacturer_profit",
    "max_entropy_posthoc",
    "feasibility_only",
]

DEFAULT_SELECTOR: SelectorMode = "platform_operating_score"
DEFAULT_LEXICOGRAPHIC_EPSILON_TOLERANCE = 1.0e-9
_LP_OPTIONS = {
    "primal_feasibility_tolerance": 1.0e-9,
    "dual_feasibility_tolerance": 1.0e-9,
}


@dataclass(frozen=True)
class LexicographicSelection:
    """Result of the common two-stage CCE selector.

    Stage 1 minimizes the largest replacement gain. Stage 2 keeps that
    optimum within ``epsilon_tolerance`` and applies the requested selector.
    """

    q: np.ndarray
    epsilon_star: float
    epsilon_bound: float
    selector: str
    objective_value: float
    stage1_status: str
    stage2_status: str


@dataclass
class CceSolution:
    solver: str
    q: np.ndarray
    objective_value: float
    cce_gap_nominal: float
    cce_gap_ucb: float
    support_profiles: list[Profile]
    support_probabilities: list[float]
    status: str
    full_optimality_certified: bool = False
    pricing_mode: str | None = None
    max_reduced_cost: float | None = None
    pricing_iterations: int | None = None
    max_deviation: dict | None = None
    lower_bound: float | None = None
    certificate_gap: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def support_size(self) -> int:
        return len(self.support_profiles)

    def to_jsonable(self) -> dict:
        return {
            "solver": self.solver,
            "objective_value": self.objective_value,
            "cce_gap_nominal": self.cce_gap_nominal,
            "cce_gap_ucb": self.cce_gap_ucb,
            "support_profiles": [list(profile) for profile in self.support_profiles],
            "support_probabilities": self.support_probabilities,
            "support_size": self.support_size,
            "status": self.status,
            "full_optimality_certified": self.full_optimality_certified,
            "pricing_mode": self.pricing_mode,
            "max_reduced_cost": self.max_reduced_cost,
            "pricing_iterations": self.pricing_iterations,
            "max_deviation": self.max_deviation,
            "lower_bound": self.lower_bound,
            "certificate_gap": self.certificate_gap,
            "diagnostics": self.diagnostics,
        }


def canonical_selector(selector: str) -> str:
    """Return the public canonical name for a supported CCE selector."""

    if selector == "manufacturer_profit":
        return "total_manufacturer_return"
    supported = {
        "platform_operating_score",
        "total_manufacturer_return",
        "max_entropy_posthoc",
        "feasibility_only",
    }
    if selector not in supported:
        choices = ", ".join(sorted(supported | {"manufacturer_profit"}))
        raise ValueError(f"Unknown CCE selector {selector!r}; expected one of: {choices}.")
    return selector


def selector_values_from_game(game: EmpiricalGame, selector: str) -> np.ndarray:
    """Build the linear stage-2 profile objective for an empirical game."""

    canonical = canonical_selector(selector)
    if canonical == "platform_operating_score":
        values = np.asarray(game.objectives, dtype=float)
    elif canonical == "total_manufacturer_return":
        values = np.sum(np.asarray(game.payoffs, dtype=float), axis=1)
    else:
        values = np.zeros(game.n_profiles, dtype=float)
    if values.shape != (game.n_profiles,) or not np.all(np.isfinite(values)):
        raise ValueError(f"Selector {canonical!r} produced invalid profile objective values.")
    return values


def _normalized_distribution(values: np.ndarray, *, stage: str) -> np.ndarray:
    q = np.maximum(np.asarray(values, dtype=float), 0.0)
    total = float(np.sum(q))
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError(f"{stage} returned an invalid probability distribution.")
    return q / total


def canonicalize_cce_distribution(
    values: np.ndarray,
    deviation_matrix: np.ndarray,
    epsilon_bound: float,
    *,
    probability_tolerance: float = 1.0e-9,
) -> np.ndarray:
    """Return one normalized distribution for reporting, hashing, and audit.

    LP solvers can leave numerically tiny positive masses.  Reporting a
    thresholded support while retaining those masses in ``q`` makes a later
    support-based audit evaluate a different distribution.  This routine
    removes tiny masses only when the renormalized distribution still obeys
    the stage-2 replacement-gain bound.  Any tiny mass that is needed for
    feasibility is retained and must therefore remain in the public support.
    """

    q = _normalized_distribution(values, stage="CCE distribution canonicalization")
    a = np.asarray(deviation_matrix, dtype=float)
    if a.ndim != 2 or a.shape[1] != q.size:
        raise ValueError("deviation_matrix must have one column per probability.")
    probability_tolerance = float(probability_tolerance)
    if not np.isfinite(probability_tolerance) or probability_tolerance < 0.0:
        raise ValueError("probability_tolerance must be finite and nonnegative.")
    epsilon_bound = float(epsilon_bound)
    if not np.isfinite(epsilon_bound) or epsilon_bound < 0.0:
        raise ValueError("epsilon_bound must be finite and nonnegative.")
    if probability_tolerance == 0.0:
        return q

    feasibility_tolerance = max(1.0e-9, 10.0 * np.finfo(float).eps)
    tiny_indices = np.flatnonzero((q > 0.0) & (q <= probability_tolerance))
    for idx in tiny_indices[np.argsort(q[tiny_indices])]:
        candidate = q.copy()
        candidate[int(idx)] = 0.0
        if float(np.sum(candidate)) <= 0.0:
            continue
        candidate /= np.sum(candidate)
        if _realized_epsilon(a, candidate) <= epsilon_bound + feasibility_tolerance:
            q = candidate
    return q


def _realized_epsilon(a: np.ndarray, q: np.ndarray) -> float:
    if a.shape[0] == 0:
        return 0.0
    return max(0.0, float(np.max(a @ q)))


def _entropy(q: np.ndarray) -> float:
    positive = q[q > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def _solve_max_entropy_posthoc(
    a: np.ndarray,
    epsilon_bound: float,
    start: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Deterministically maximize entropy on a fixed epsilon-feasible face.

    SLSQP is used only for this optional nonlinear post-processing selector.
    Failure is explicit: callers receive ``RuntimeError`` rather than a
    silently substituted linear or stage-1 distribution.
    """

    n_profiles = int(start.size)
    if n_profiles == 1:
        return np.ones(1, dtype=float), "maximum entropy is trivial on one profile"

    floor = 1.0e-15

    def negative_entropy(q: np.ndarray) -> float:
        safe = np.maximum(q, floor)
        return float(np.sum(q * np.log(safe)))

    def negative_entropy_gradient(q: np.ndarray) -> np.ndarray:
        return np.log(np.maximum(q, floor)) + 1.0

    constraints: list[dict[str, Any]] = [
        {"type": "eq", "fun": lambda q: float(np.sum(q) - 1.0), "jac": lambda q: np.ones_like(q)},
    ]
    if a.shape[0] > 0:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda q: np.full(a.shape[0], epsilon_bound, dtype=float) - a @ q,
                "jac": lambda q: -a,
            }
        )
    result = minimize(
        negative_entropy,
        np.asarray(start, dtype=float),
        jac=negative_entropy_gradient,
        bounds=[(0.0, 1.0)] * n_profiles,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 2000, "ftol": 1.0e-12, "disp": False},
    )
    # Near a zero-gap face, SLSQP can report a positive directional
    # derivative even though its iterate is already feasible to the LP's
    # numerical tolerance.  Retry only the nonlinear secondary selector with
    # a standard optimizer tolerance; the explicit feasibility check below
    # still rejects any distribution outside the frozen epsilon bound.
    if not result.success:
        result = minimize(
            negative_entropy,
            np.asarray(start, dtype=float),
            jac=negative_entropy_gradient,
            bounds=[(0.0, 1.0)] * n_profiles,
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": 4000, "ftol": 1.0e-9, "disp": False},
        )
    if not result.success:
        raise RuntimeError(f"Maximum-entropy CCE selector failed: {result.message}")
    q = _normalized_distribution(result.x, stage="Maximum-entropy CCE selector")
    feasibility_tolerance = max(1.0e-8, 10.0 * np.finfo(float).eps)
    if abs(float(np.sum(q)) - 1.0) > feasibility_tolerance or np.min(q) < -feasibility_tolerance:
        raise RuntimeError("Maximum-entropy CCE selector returned an invalid distribution.")
    if _realized_epsilon(a, q) > float(epsilon_bound) + feasibility_tolerance:
        raise RuntimeError("Maximum-entropy CCE selector violated the stage-1 epsilon bound.")
    return q, str(result.message)


def solve_lexicographic_distribution(
    deviation_matrix: np.ndarray,
    selector_values: np.ndarray,
    *,
    selector: str = DEFAULT_SELECTOR,
    epsilon_tolerance: float = DEFAULT_LEXICOGRAPHIC_EPSILON_TOLERANCE,
    probability_tolerance: float = 1.0e-9,
) -> LexicographicSelection:
    """Solve the shared two-stage CCE selection problem on fixed columns.

    The first LP minimizes the maximum replacement gain ``epsilon``. The
    second optimization constrains every replacement gain to at most
    ``epsilon_star + epsilon_tolerance`` and applies the requested selector.
    """

    canonical = canonical_selector(selector)
    a = np.asarray(deviation_matrix, dtype=float)
    values = np.asarray(selector_values, dtype=float)
    if a.ndim != 2:
        raise ValueError("deviation_matrix must be two-dimensional.")
    n_profiles = int(a.shape[1])
    if n_profiles == 0:
        raise ValueError("Cannot solve a CCE selector on an empty profile set.")
    if values.shape != (n_profiles,):
        raise ValueError("selector_values must contain one value per profile.")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(values)):
        raise ValueError("CCE selector inputs must be finite.")
    epsilon_tolerance = float(epsilon_tolerance)
    if not np.isfinite(epsilon_tolerance) or epsilon_tolerance < 0.0:
        raise ValueError("epsilon_tolerance must be finite and nonnegative.")

    c = np.zeros(n_profiles + 1, dtype=float)
    c[-1] = 1.0
    a_ub = np.hstack([a, -np.ones((a.shape[0], 1), dtype=float)])
    a_eq = np.zeros((1, n_profiles + 1), dtype=float)
    a_eq[0, :n_profiles] = 1.0
    stage1 = linprog(
        c,
        A_ub=a_ub,
        b_ub=np.zeros(a.shape[0], dtype=float),
        A_eq=a_eq,
        b_eq=np.array([1.0], dtype=float),
        bounds=[(0.0, None)] * n_profiles + [(0.0, None)],
        method="highs",
        options=_LP_OPTIONS,
    )
    if not stage1.success:
        raise RuntimeError(f"CCE selector stage 1 failed: {stage1.message}")
    stage1_q = _normalized_distribution(stage1.x[:n_profiles], stage="CCE selector stage 1")
    epsilon_star = max(
        0.0,
        float(stage1.x[-1]),
        _realized_epsilon(a, stage1_q),
    )
    epsilon_bound = epsilon_star + epsilon_tolerance

    if canonical == "feasibility_only":
        q = stage1_q
        objective_value = 0.0
        stage2_status = "stage 2 skipped for feasibility_only selector"
    elif canonical == "max_entropy_posthoc":
        q, stage2_status = _solve_max_entropy_posthoc(a, epsilon_bound, stage1_q)
        objective_value = _entropy(q)
    else:
        stage2 = linprog(
            -values,
            A_ub=a,
            b_ub=np.full(a.shape[0], epsilon_bound, dtype=float),
            A_eq=np.ones((1, n_profiles), dtype=float),
            b_eq=np.array([1.0], dtype=float),
            bounds=[(0.0, None)] * n_profiles,
            method="highs",
            options=_LP_OPTIONS,
        )
        if not stage2.success:
            raise RuntimeError(f"CCE selector stage 2 failed: {stage2.message}")
        q = _normalized_distribution(stage2.x, stage="CCE selector stage 2")
        objective_value = float(np.dot(values, q))
        stage2_status = str(stage2.message)

    q = canonicalize_cce_distribution(
        q,
        a,
        epsilon_bound,
        probability_tolerance=probability_tolerance,
    )
    if canonical == "max_entropy_posthoc":
        objective_value = _entropy(q)
    elif canonical in {"platform_operating_score", "total_manufacturer_return"}:
        objective_value = float(np.dot(values, q))

    feasibility_tolerance = max(1.0e-8, epsilon_tolerance)
    if _realized_epsilon(a, q) > epsilon_bound + feasibility_tolerance:
        raise RuntimeError("CCE selector stage 2 violated the stage-1 epsilon bound.")
    return LexicographicSelection(
        q=q,
        epsilon_star=epsilon_star,
        epsilon_bound=epsilon_bound,
        selector=canonical,
        objective_value=objective_value,
        stage1_status=str(stage1.message),
        stage2_status=stage2_status,
    )


def deviation_rows(
    game: EmpiricalGame,
    variable_profiles: list[Profile] | tuple[Profile, ...],
    ucb: bool = False,
) -> np.ndarray:
    profile_to_index = game.profile_to_index
    rows: list[list[float]] = []
    for agent in range(game.n_agents):
        for dev_policy in game.policy_ids:
            row: list[float] = []
            for profile in variable_profiles:
                dev_profile_list = list(profile)
                dev_profile_list[agent] = dev_policy
                dev_profile = tuple(dev_profile_list)
                base_idx = profile_to_index[profile]
                dev_idx = profile_to_index[dev_profile]
                if ucb:
                    row.append(
                        float(
                            game.payoffs[dev_idx, agent]
                            + game.ci_radius[dev_idx, agent]
                            - game.payoffs[base_idx, agent]
                            + game.ci_radius[base_idx, agent]
                        )
                    )
                else:
                    row.append(float(game.payoffs[dev_idx, agent] - game.payoffs[base_idx, agent]))
            rows.append(row)
    return np.array(rows, dtype=float)


def distribution_support(game: EmpiricalGame, q: np.ndarray, tol: float = 1e-9) -> tuple[list[Profile], list[float]]:
    support_profiles: list[Profile] = []
    support_probabilities: list[float] = []
    for idx, prob in enumerate(q):
        if prob > tol:
            support_profiles.append(game.profiles[idx])
            support_probabilities.append(float(prob))
    return support_profiles, support_probabilities


def max_deviation(game: EmpiricalGame, q: np.ndarray) -> dict:
    profile_to_index = game.profile_to_index
    best = {
        "agent": None,
        "policy": None,
        "gain_nominal": 0.0,
    }
    for agent in range(game.n_agents):
        for dev_policy in game.policy_ids:
            gains = []
            for profile in game.profiles:
                dev_profile = list(profile)
                dev_profile[agent] = dev_policy
                dev_idx = profile_to_index[tuple(dev_profile)]
                base_idx = profile_to_index[profile]
                gains.append(game.payoffs[dev_idx, agent] - game.payoffs[base_idx, agent])
            gain = float(np.dot(q, np.array(gains, dtype=float)))
            if gain > best["gain_nominal"]:
                best = {"agent": agent, "policy": dev_policy, "gain_nominal": gain}
    return best


def max_deviation_ucb(game: EmpiricalGame, q: np.ndarray) -> dict:
    profile_to_index = game.profile_to_index
    best = {
        "agent": None,
        "policy": None,
        "gain_ucb": 0.0,
    }
    for agent in range(game.n_agents):
        for dev_policy in game.policy_ids:
            gains = []
            for profile in game.profiles:
                dev_profile = list(profile)
                dev_profile[agent] = dev_policy
                dev_idx = profile_to_index[tuple(dev_profile)]
                base_idx = profile_to_index[profile]
                gains.append(
                    game.payoffs[dev_idx, agent]
                    + game.ci_radius[dev_idx, agent]
                    - game.payoffs[base_idx, agent]
                    + game.ci_radius[base_idx, agent]
                )
            gain = float(np.dot(q, np.array(gains, dtype=float)))
            if gain > best["gain_ucb"]:
                best = {"agent": agent, "policy": dev_policy, "gain_ucb": gain}
    return best


def compute_cce_gap(game: EmpiricalGame, q: np.ndarray) -> float:
    return max(0.0, float(max_deviation(game, q)["gain_nominal"]))


def compute_cce_gap_ucb(game: EmpiricalGame, q: np.ndarray) -> float:
    return max(0.0, float(max_deviation_ucb(game, q)["gain_ucb"]))


def compute_cce_gap_from_deviation_closure(
    game: EmpiricalGame,
    support_profiles: list[Profile],
    support_probabilities: list[float],
) -> float:
    profile_to_index = game.profile_to_index
    q_by_profile = dict(zip(support_profiles, support_probabilities, strict=True))
    best_gain = 0.0
    for agent in range(game.n_agents):
        for dev_policy in game.policy_ids:
            gain = 0.0
            for profile, prob in q_by_profile.items():
                dev_profile = list(profile)
                dev_profile[agent] = dev_policy
                dev_idx = profile_to_index[tuple(dev_profile)]
                base_idx = profile_to_index[profile]
                gain += prob * float(game.payoffs[dev_idx, agent] - game.payoffs[base_idx, agent])
            best_gain = max(best_gain, gain)
    return max(0.0, best_gain)


def compute_cce_gap_ucb_from_deviation_closure(
    game: EmpiricalGame,
    support_profiles: list[Profile],
    support_probabilities: list[float],
) -> float:
    profile_to_index = game.profile_to_index
    q_by_profile = dict(zip(support_profiles, support_probabilities, strict=True))
    best_gain = 0.0
    for agent in range(game.n_agents):
        for dev_policy in game.policy_ids:
            gain = 0.0
            for profile, prob in q_by_profile.items():
                dev_profile = list(profile)
                dev_profile[agent] = dev_policy
                dev_idx = profile_to_index[tuple(dev_profile)]
                base_idx = profile_to_index[profile]
                gain += prob * float(
                    game.payoffs[dev_idx, agent]
                    + game.ci_radius[dev_idx, agent]
                    - game.payoffs[base_idx, agent]
                    + game.ci_radius[base_idx, agent]
                )
            best_gain = max(best_gain, gain)
    return max(0.0, best_gain)


def selection_diagnostics(
    game: EmpiricalGame,
    selection: LexicographicSelection,
) -> dict[str, Any]:
    q = selection.q
    return {
        "selector": selection.selector,
        "stage1_epsilon": selection.epsilon_star,
        "stage2_epsilon_bound": selection.epsilon_bound,
        "epsilon_tolerance": selection.epsilon_bound - selection.epsilon_star,
        "secondary_objective_value": selection.objective_value,
        "platform_operating_score": float(np.dot(game.objectives, q)),
        "total_manufacturer_return": float(np.dot(np.sum(game.payoffs, axis=1), q)),
        "distribution_entropy": _entropy(q),
        "stage1_status": selection.stage1_status,
        "stage2_status": selection.stage2_status,
    }


def solve_full_cce_lp(
    game: EmpiricalGame,
    tolerance: float = 1e-9,
    selector: str = DEFAULT_SELECTOR,
    epsilon_tolerance: float = DEFAULT_LEXICOGRAPHIC_EPSILON_TOLERANCE,
) -> CceSolution:
    a_ub = deviation_rows(game, game.profiles)
    selection = solve_lexicographic_distribution(
        a_ub,
        selector_values_from_game(game, selector),
        selector=selector,
        epsilon_tolerance=epsilon_tolerance,
    )
    q = selection.q
    # ``q`` has already been canonicalized.  Retain every remaining positive
    # mass so that the public support and subsequent audits identify exactly
    # the same distribution.
    support_profiles, support_probabilities = distribution_support(game, q, 0.0)
    gap = compute_cce_gap(game, q)
    ucb_dev = max_deviation_ucb(game, q)
    nominal_dev = max_deviation(game, q)
    nominal_dev["gain_ucb"] = ucb_dev["gain_ucb"]
    nominal_dev["ucb_agent"] = ucb_dev["agent"]
    nominal_dev["ucb_policy"] = ucb_dev["policy"]
    return CceSolution(
        solver="FullTensor-CCE-LP",
        q=q,
        objective_value=selection.objective_value,
        cce_gap_nominal=gap,
        cce_gap_ucb=max(0.0, float(ucb_dev["gain_ucb"])),
        support_profiles=support_profiles,
        support_probabilities=support_probabilities,
        status=f"stage 1: {selection.stage1_status}; stage 2: {selection.stage2_status}",
        full_optimality_certified=True,
        pricing_mode="full_tensor_lexicographic",
        max_deviation=nominal_dev,
        diagnostics=selection_diagnostics(game, selection),
    )


def solve_support_min_epsilon(
    game: EmpiricalGame,
    support_profiles: list[Profile],
    ucb: bool = False,
    solver_name: str = "Support-MinEpsilon-CCE",
    selector: str = DEFAULT_SELECTOR,
    epsilon_tolerance: float = DEFAULT_LEXICOGRAPHIC_EPSILON_TOLERANCE,
) -> CceSolution:
    a = deviation_rows(game, support_profiles, ucb=ucb)
    support_indices = [game.profile_to_index[profile] for profile in support_profiles]
    full_selector_values = selector_values_from_game(game, selector)
    selection = solve_lexicographic_distribution(
        a,
        full_selector_values[support_indices],
        selector=selector,
        epsilon_tolerance=epsilon_tolerance,
    )
    q_support = selection.q
    full_q = np.zeros(game.n_profiles, dtype=float)
    profile_to_index = game.profile_to_index
    for profile, prob in zip(support_profiles, q_support, strict=True):
        full_q[profile_to_index[profile]] = prob
    support_out, probabilities_out = distribution_support(game, full_q, 0.0)
    ucb_dev = max_deviation_ucb(game, full_q)
    nominal_dev = max_deviation(game, full_q)
    nominal_dev["gain_ucb"] = ucb_dev["gain_ucb"]
    nominal_dev["ucb_agent"] = ucb_dev["agent"]
    nominal_dev["ucb_policy"] = ucb_dev["policy"]
    return CceSolution(
        solver=solver_name,
        q=full_q,
        objective_value=selection.objective_value,
        cce_gap_nominal=compute_cce_gap(game, full_q),
        cce_gap_ucb=max(0.0, float(ucb_dev["gain_ucb"])),
        support_profiles=support_out,
        support_probabilities=probabilities_out,
        status=f"stage 1: {selection.stage1_status}; stage 2: {selection.stage2_status}",
        max_deviation=nominal_dev,
        diagnostics=selection_diagnostics(
            game,
            LexicographicSelection(
                q=full_q,
                epsilon_star=selection.epsilon_star,
                epsilon_bound=selection.epsilon_bound,
                selector=selection.selector,
                objective_value=selection.objective_value,
                stage1_status=selection.stage1_status,
                stage2_status=selection.stage2_status,
            ),
        ),
    )


def solve_full_tensor_min_epsilon(
    game: EmpiricalGame,
    ucb: bool = False,
    tolerance: float = 1e-9,
    selector: str = "feasibility_only",
    epsilon_tolerance: float = DEFAULT_LEXICOGRAPHIC_EPSILON_TOLERANCE,
) -> CceSolution:
    solver_name = "FullTensor-MinUCB-CCE-LP" if ucb else "FullTensor-MinGap-CCE-LP"
    solution = solve_support_min_epsilon(
        game,
        list(game.profiles),
        ucb=ucb,
        solver_name=solver_name,
        selector=selector,
        epsilon_tolerance=epsilon_tolerance,
    )
    solution.support_profiles, solution.support_probabilities = distribution_support(game, solution.q, 0.0)
    solution.full_optimality_certified = True
    solution.pricing_mode = "full_tensor"
    return solution
