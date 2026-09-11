from __future__ import annotations

import numpy as np

from cmfg_cce.mechanisms.base import MechanismOutcome, PriceMechanism, choose_lowest_score, sorted_valid_bids_by_score


class DeliveryCriticalMechanism(PriceMechanism):
    mechanism_id = "M4_delivery_critical"

    def __init__(self, beta_p: float = 0.6, beta_l: float = 0.4, single_bidder_markup_cap: float = 0.20):
        self.beta_p = beta_p
        self.beta_l = beta_l
        self.single_bidder_markup_cap = single_bidder_markup_cap

    def allocate_and_pay(
        self,
        bids,
        reserve_price: float,
        rng: np.random.Generator,
    ) -> MechanismOutcome:
        valid_bids = sorted_valid_bids_by_score(bids)
        winner_bid = choose_lowest_score(valid_bids, rng)
        if winner_bid is None:
            return MechanismOutcome(self.mechanism_id, None, 0.0, tuple(), reserve_price)
        if reserve_price + 1e-10 < winner_bid.price:
            raise ValueError(
                f"Invalid reserve price {reserve_price:.4f}; below winner bid {winner_bid.price:.4f}."
            )
        if len(valid_bids) >= 2:
            other_scores = [float(bid.score) for bid in valid_bids if bid.manufacturer_id != winner_bid.manufacturer_id]
            c_second = min(other_scores)
            p_ref = max(1e-9, getattr(winner_bid, "p_ref", reserve_price / 1.75))
            critical_price = (p_ref / self.beta_p) * (c_second - self.beta_l * winner_bid.lead_time_multiplier)
            payment = min(reserve_price, critical_price)
        else:
            payment = min(reserve_price, winner_bid.price * (1.0 + self.single_bidder_markup_cap))
        payment = max(payment, winner_bid.price)
        return MechanismOutcome(
            mechanism_id=self.mechanism_id,
            winner=winner_bid.manufacturer_id,
            payment=payment,
            valid_bids=tuple(valid_bids),
            reserve_price=reserve_price,
        )
