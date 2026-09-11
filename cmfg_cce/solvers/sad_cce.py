from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cmfg_cce.evaluation.empirical_game import EmpiricalGame
from cmfg_cce.evaluation.rollout import Profile
from cmfg_cce.solvers.cce_lp import CceSolution, max_deviation, solve_support_min_epsilon


@dataclass(frozen=True)
class SadConfig:
    initial_support_size: int = 8
    max_support_size: int = 48
    support_add_batch_size: int = 8
    max_rounds: int = 8
    target_gap: float = 1e-8


def _initial_support(game: EmpiricalGame, size: int) -> list[Profile]:
    ranked_indices = list(np.argsort(-game.objectives))
    support: list[Profile] = []
    for idx in ranked_indices:
        profile = game.profiles[int(idx)]
        if profile not in support:
            support.append(profile)
        if len(support) >= min(size, game.n_profiles):
            break
    return support


def _expand_support(
    game: EmpiricalGame,
    support: list[Profile],
    solution: CceSolution,
    batch_size: int,
) -> list[Profile]:
    deviation = max_deviation(game, solution.q)
    agent = deviation["agent"]
    dev_policy = deviation["policy"]
    if agent is None or dev_policy is None:
        return support
    candidates: list[tuple[float, Profile]] = []
    for profile, prob in zip(solution.support_profiles, solution.support_probabilities, strict=True):
        next_profile = list(profile)
        next_profile[int(agent)] = str(dev_policy)
        candidate = tuple(next_profile)
        if candidate not in support:
            candidates.append((prob, candidate))
    candidates.sort(reverse=True, key=lambda item: item[0])
    expanded = list(support)
    for _, profile in candidates[:batch_size]:
        expanded.append(profile)
    if len(expanded) == len(support):
        ranked_indices = list(np.argsort(-game.objectives))
        for idx in ranked_indices:
            profile = game.profiles[int(idx)]
            if profile not in expanded:
                expanded.append(profile)
                if len(expanded) - len(support) >= batch_size:
                    break
    return expanded


def solve_sad_cce(game: EmpiricalGame, config: SadConfig) -> CceSolution:
    support = _initial_support(game, config.initial_support_size)
    best_solution: CceSolution | None = None
    rounds = 0
    while rounds < config.max_rounds:
        rounds += 1
        solution = solve_support_min_epsilon(game, support)
        solution.solver = "SAD-CCE"
        solution.full_optimality_certified = False
        solution.pricing_iterations = rounds
        best_solution = solution
        if solution.cce_gap_nominal <= config.target_gap or len(support) >= min(config.max_support_size, game.n_profiles):
            break
        support = _expand_support(game, support, solution, config.support_add_batch_size)
        support = support[: min(config.max_support_size, game.n_profiles)]
    if best_solution is None:
        raise RuntimeError("SAD-CCE did not run any support LP rounds.")
    best_solution.max_deviation = max_deviation(game, best_solution.q)
    return best_solution

