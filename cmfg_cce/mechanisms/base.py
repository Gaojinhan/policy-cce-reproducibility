from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Bid:
    manufacturer_id: int
    price: float
    markup: float
    cost: float
    lead_time_multiplier: float = 1.0
    score: float | None = None
    p_ref: float = 1.0
    skip: bool = False


@dataclass(frozen=True)
class MechanismOutcome:
    mechanism_id: str
    winner: int | None
    payment: float
    valid_bids: tuple[Bid, ...]
    reserve_price: float

    @property
    def participation_count(self) -> int:
        return len(self.valid_bids)


class PriceMechanism:
    mechanism_id: str

    def allocate_and_pay(
        self,
        bids: list[Bid],
        reserve_price: float,
        rng: np.random.Generator,
    ) -> MechanismOutcome:
        raise NotImplementedError


def choose_lowest_bid(valid_bids: list[Bid], rng: np.random.Generator) -> Bid | None:
    if not valid_bids:
        return None
    min_price = min(bid.price for bid in valid_bids)
    tied = [bid for bid in valid_bids if abs(bid.price - min_price) <= 1e-10]
    return tied[int(rng.integers(0, len(tied)))]


def choose_lowest_score(valid_bids: list[Bid], rng: np.random.Generator) -> Bid | None:
    if not valid_bids:
        return None
    min_score = min(float(bid.score) for bid in valid_bids)
    tied = [bid for bid in valid_bids if abs(float(bid.score) - min_score) <= 1e-10]
    return tied[int(rng.integers(0, len(tied)))]


def sorted_valid_bids(bids: list[Bid]) -> list[Bid]:
    return sorted([bid for bid in bids if not bid.skip], key=lambda bid: (bid.price, bid.manufacturer_id))


def sorted_valid_bids_by_score(bids: list[Bid]) -> list[Bid]:
    return sorted(
        [bid for bid in bids if not bid.skip],
        key=lambda bid: (float(bid.score), bid.price, bid.manufacturer_id),
    )
