from __future__ import annotations

from cmfg_cce.evaluation.empirical_game import EmpiricalGame
from cmfg_cce.solvers.cce_lp import CceSolution, solve_full_cce_lp


def solve_full_tensor_cce(game: EmpiricalGame) -> CceSolution:
    return solve_full_cce_lp(game)

