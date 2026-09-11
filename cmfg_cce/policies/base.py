from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cmfg_cce.envs.toy import Observation


@dataclass(frozen=True)
class PolicyContext:
    markup_grid: tuple[float, ...]
    lead_time_grid: tuple[float, ...] = (0.55, 0.70, 0.85, 1.00)
    delivery_score_weight: float = 0.4


@dataclass(frozen=True)
class PolicyAction:
    skip: bool
    markup: float = 0.0
    lead_time_multiplier: float = 1.0


class BiddingPolicy:
    name: str
    mechanism_family: Literal["price", "price_delivery"] = "price"

    def act(self, obs: Observation, feasible: bool, context: PolicyContext) -> PolicyAction:
        raise NotImplementedError


def grid_value(grid: tuple[float, ...], index: int) -> float:
    bounded_index = max(0, min(len(grid) - 1, index))
    return float(grid[bounded_index])
