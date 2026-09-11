from __future__ import annotations

import numpy as np

from cmfg_cce.evaluation.empirical_game import EmpiricalGame
from cmfg_cce.evaluation.rollout import Profile
from cmfg_cce.solvers.cce_lp import CceSolution, compute_cce_gap, distribution_support, max_deviation, max_deviation_ucb
from cmfg_cce.solvers.sparse_cce import SparseCceResult


def q_from_support(
    game: EmpiricalGame,
    support_profiles: list[Profile] | tuple[Profile, ...],
    support_probabilities: list[float] | tuple[float, ...],
) -> np.ndarray:
    probs = np.array(support_probabilities, dtype=float)
    if probs.size == 0 or float(np.sum(probs)) <= 0.0:
        raise ValueError("Support probabilities must contain positive mass.")
    probs = probs / np.sum(probs)
    q = np.zeros(game.n_profiles, dtype=float)
    profile_to_index = game.profile_to_index
    for profile, prob in zip(support_profiles, probs, strict=True):
        if profile not in profile_to_index:
            raise KeyError(f"Profile {profile} is not in the full empirical game tensor.")
        q[profile_to_index[profile]] += float(prob)
    return q


def audit_distribution_on_full_game(
    game: EmpiricalGame,
    support_profiles: list[Profile] | tuple[Profile, ...],
    support_probabilities: list[float] | tuple[float, ...],
    solver_name: str,
    status: str = "full tensor audit of reported distribution",
) -> CceSolution:
    q = q_from_support(game, support_profiles, support_probabilities)
    nominal_dev = max_deviation(game, q)
    ucb_dev = max_deviation_ucb(game, q)
    nominal_dev["gain_ucb"] = ucb_dev["gain_ucb"]
    nominal_dev["ucb_agent"] = ucb_dev["agent"]
    nominal_dev["ucb_policy"] = ucb_dev["policy"]
    support_out, probabilities_out = distribution_support(game, q)
    return CceSolution(
        solver=solver_name,
        q=q,
        objective_value=float(np.dot(game.objectives, q)),
        cce_gap_nominal=compute_cce_gap(game, q),
        cce_gap_ucb=max(0.0, float(ucb_dev["gain_ucb"])),
        support_profiles=support_out,
        support_probabilities=probabilities_out,
        status=status,
        full_optimality_certified=False,
        pricing_mode="full_tensor_audit",
        max_deviation=nominal_dev,
    )


def audit_sparse_result_on_full_game(game: EmpiricalGame, result: SparseCceResult) -> CceSolution:
    return audit_distribution_on_full_game(
        game,
        result.support_profiles,
        result.support_probabilities,
        solver_name=f"{result.solver}-FullTensorAudit",
    )
