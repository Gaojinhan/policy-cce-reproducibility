from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from cmfg_cce.evaluation.empirical_game import EmpiricalGame
from cmfg_cce.evaluation.rollout import Profile
from cmfg_cce.solvers.cce_lp import (
    DEFAULT_LEXICOGRAPHIC_EPSILON_TOLERANCE,
    DEFAULT_SELECTOR,
    CceSolution,
    LexicographicSelection,
    canonicalize_cce_distribution,
    canonical_selector,
    compute_cce_gap,
    deviation_rows,
    distribution_support,
    max_deviation,
    max_deviation_ucb,
    selection_diagnostics,
    selector_values_from_game,
    solve_lexicographic_distribution,
)


@dataclass
class _MasterResult:
    support: list[Profile]
    q_support: np.ndarray
    objective_value: float
    result: object
    epsilon: float = 0.0


def _profile_indices(game: EmpiricalGame, support: list[Profile]) -> list[int]:
    profile_to_index = game.profile_to_index
    return [profile_to_index[profile] for profile in support]


def _solve_relaxed_master(game: EmpiricalGame, support: list[Profile]) -> _MasterResult:
    a = deviation_rows(game, support)
    n_support = len(support)
    c = np.zeros(n_support + 1, dtype=float)
    c[-1] = 1.0
    a_ub = np.hstack([a, -np.ones((a.shape[0], 1), dtype=float)])
    a_eq = np.zeros((1, n_support + 1), dtype=float)
    a_eq[0, :n_support] = 1.0
    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=np.zeros(a.shape[0], dtype=float),
        A_eq=a_eq,
        b_eq=np.array([1.0], dtype=float),
        bounds=[(0.0, None)] * n_support + [(0.0, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"CG relaxed master failed: {result.message}")
    q_support = np.maximum(result.x[:n_support], 0.0)
    q_support = q_support / np.sum(q_support)
    indices = _profile_indices(game, support)
    objective = float(np.dot(game.objectives[indices], q_support))
    realized_epsilon = max(0.0, float(np.max(a @ q_support))) if a.shape[0] else 0.0
    return _MasterResult(
        support=support,
        q_support=q_support,
        objective_value=objective,
        result=result,
        epsilon=max(0.0, float(result.x[-1]), realized_epsilon),
    )


def _solve_objective_master(
    game: EmpiricalGame,
    support: list[Profile],
    selector_values: np.ndarray,
    epsilon_bound: float,
) -> _MasterResult:
    a = deviation_rows(game, support)
    indices = _profile_indices(game, support)
    c = -selector_values[indices]
    n_support = len(support)
    result = linprog(
        c,
        A_ub=a,
        b_ub=np.full(a.shape[0], float(epsilon_bound), dtype=float),
        A_eq=np.ones((1, n_support), dtype=float),
        b_eq=np.array([1.0], dtype=float),
        bounds=[(0.0, None)] * n_support,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"CG objective master failed: {result.message}")
    q_support = np.maximum(result.x, 0.0)
    q_support = q_support / np.sum(q_support)
    return _MasterResult(
        support=support,
        q_support=q_support,
        objective_value=float(np.dot(selector_values[indices], q_support)),
        result=result,
        epsilon=max(0.0, float(np.max(a @ q_support))) if a.shape[0] else 0.0,
    )


def _reduced_costs(
    game: EmpiricalGame,
    candidates: list[Profile],
    ineq_marginals: np.ndarray,
    eq_marginal: float,
    selector_values: np.ndarray | None,
) -> np.ndarray:
    columns = deviation_rows(game, candidates)
    costs = np.zeros(len(candidates), dtype=float)
    if selector_values is not None:
        profile_to_index = game.profile_to_index
        costs = np.array(
            [-selector_values[profile_to_index[profile]] for profile in candidates],
            dtype=float,
        )
    return costs - columns.T @ ineq_marginals - float(eq_marginal)


def _price_exhaustively(
    game: EmpiricalGame,
    support: list[Profile],
    master: _MasterResult,
    selector_values: np.ndarray | None,
    tolerance: float,
    batch_size: int,
) -> tuple[list[Profile], float]:
    support_set = set(support)
    candidates = [profile for profile in game.profiles if profile not in support_set]
    if not candidates:
        return [], 0.0
    reduced_costs = _reduced_costs(
        game,
        candidates,
        np.array(master.result.ineqlin.marginals, dtype=float),
        float(master.result.eqlin.marginals[0]),
        selector_values=selector_values,
    )
    order = np.argsort(reduced_costs)
    entering = [
        candidates[int(idx)]
        for idx in order
        if float(reduced_costs[int(idx)]) < -tolerance
    ][: max(1, int(batch_size))]
    return entering, float(reduced_costs[int(order[0])])


def _full_q(game: EmpiricalGame, support: list[Profile], q_support: np.ndarray) -> np.ndarray:
    q = np.zeros(game.n_profiles, dtype=float)
    profile_to_index = game.profile_to_index
    for profile, prob in zip(support, q_support, strict=True):
        q[profile_to_index[profile]] += float(prob)
    return q / np.sum(q)


def solve_cg_cce_exhaustive(
    game: EmpiricalGame,
    tolerance: float = 1.0e-8,
    pricing_batch_size: int = 16,
    max_iterations: int | None = None,
    selector: str = DEFAULT_SELECTOR,
    epsilon_tolerance: float = DEFAULT_LEXICOGRAPHIC_EPSILON_TOLERANCE,
) -> CceSolution:
    """Solve the shared lexicographic CCE problem by exhaustive pricing."""
    if game.n_profiles == 0:
        raise ValueError("Cannot solve an empty empirical game.")
    canonical = canonical_selector(selector)
    selector_values = selector_values_from_game(game, canonical)
    max_iterations = max_iterations or (2 * game.n_profiles + 10)
    start_idx = int(np.argmax(selector_values)) if canonical not in {"max_entropy_posthoc", "feasibility_only"} else 0
    support = [game.profiles[start_idx]]
    pricing_iterations = 0
    min_reduced_cost = float("-inf")

    relaxed = _solve_relaxed_master(game, support)
    for _ in range(max_iterations):
        entering, min_reduced_cost = _price_exhaustively(
            game,
            support,
            relaxed,
            selector_values=None,
            tolerance=tolerance,
            batch_size=pricing_batch_size,
        )
        pricing_iterations += 1
        if not entering:
            break
        support.extend(entering)
        relaxed = _solve_relaxed_master(game, support)
    if relaxed.epsilon > tolerance:
        remaining = [profile for profile in game.profiles if profile not in set(support)]
        support.extend(remaining)
        relaxed = _solve_relaxed_master(game, support)
        pricing_iterations += 1
    if relaxed.epsilon > max(1.0e-6, tolerance):
        raise RuntimeError(f"CG feasibility phase ended with epsilon={relaxed.epsilon:.6g}")

    epsilon_star = relaxed.epsilon
    epsilon_bound = epsilon_star + float(epsilon_tolerance)
    if canonical == "max_entropy_posthoc":
        # Entropy is nonlinear and has no linear reduced-cost pricing rule.
        # Exhaustive CG already receives the complete empirical game, so the
        # deterministic post-processing step is solved on all profiles.
        selection = solve_lexicographic_distribution(
            deviation_rows(game, game.profiles),
            selector_values,
            selector=canonical,
            epsilon_tolerance=epsilon_tolerance,
        )
        q = selection.q
        master_status = selection.stage2_status
        objective_value = selection.objective_value
    elif canonical == "feasibility_only":
        q = _full_q(game, support, relaxed.q_support)
        selection = LexicographicSelection(
            q=q,
            epsilon_star=epsilon_star,
            epsilon_bound=epsilon_bound,
            selector=canonical,
            objective_value=0.0,
            stage1_status=str(relaxed.result.message),
            stage2_status="stage 2 skipped for feasibility_only selector",
        )
        master_status = selection.stage2_status
        objective_value = 0.0
    else:
        master = _solve_objective_master(game, support, selector_values, epsilon_bound)
        for _ in range(max_iterations):
            entering, min_reduced_cost = _price_exhaustively(
                game,
                support,
                master,
                selector_values=selector_values,
                tolerance=tolerance,
                batch_size=pricing_batch_size,
            )
            pricing_iterations += 1
            if not entering:
                break
            support.extend(entering)
            master = _solve_objective_master(game, support, selector_values, epsilon_bound)
        q = _full_q(game, support, master.q_support)
        objective_value = float(np.dot(selector_values, q))
        selection = LexicographicSelection(
            q=q,
            epsilon_star=epsilon_star,
            epsilon_bound=epsilon_bound,
            selector=canonical,
            objective_value=objective_value,
            stage1_status=str(relaxed.result.message),
            stage2_status=str(master.result.message),
        )
        master_status = str(master.result.message)

    q = canonicalize_cce_distribution(
        q,
        deviation_rows(game, game.profiles),
        selection.epsilon_bound,
        probability_tolerance=1.0e-9,
    )
    if canonical == "max_entropy_posthoc":
        objective_value = float(-np.sum(q[q > 0.0] * np.log(q[q > 0.0])))
    elif canonical in {"platform_operating_score", "total_manufacturer_return"}:
        objective_value = float(np.dot(selector_values, q))
    else:
        objective_value = 0.0
    selection = LexicographicSelection(
        q=q,
        epsilon_star=selection.epsilon_star,
        epsilon_bound=selection.epsilon_bound,
        selector=selection.selector,
        objective_value=objective_value,
        stage1_status=selection.stage1_status,
        stage2_status=selection.stage2_status,
    )
    support_profiles, support_probabilities = distribution_support(game, q, 0.0)
    ucb_dev = max_deviation_ucb(game, q)
    nominal_dev = max_deviation(game, q)
    nominal_dev["gain_ucb"] = ucb_dev["gain_ucb"]
    nominal_dev["ucb_agent"] = ucb_dev["agent"]
    nominal_dev["ucb_policy"] = ucb_dev["policy"]
    return CceSolution(
        solver="ExhaustiveCG-CCE",
        q=q,
        objective_value=objective_value,
        cce_gap_nominal=compute_cce_gap(game, q),
        cce_gap_ucb=max(0.0, float(ucb_dev["gain_ucb"])),
        support_profiles=support_profiles,
        support_probabilities=support_probabilities,
        status=f"stage 1: {relaxed.result.message}; stage 2: {master_status}",
        full_optimality_certified=True,
        pricing_mode="exhaustive_column_generation",
        max_reduced_cost=max(0.0, -float(min_reduced_cost)),
        pricing_iterations=pricing_iterations,
        max_deviation=nominal_dev,
        diagnostics=selection_diagnostics(game, selection),
    )
