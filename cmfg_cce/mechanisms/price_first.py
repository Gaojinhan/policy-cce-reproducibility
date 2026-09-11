from __future__ import annotations

import numpy as np

from cmfg_cce.mechanisms.base import MechanismOutcome, PriceMechanism, choose_lowest_bid, sorted_valid_bids


class PriceFirstMechanism(PriceMechanism):
    mechanism_id = "M1_price_first"

    def allocate_and_pay(
        self,
        bids,
        reserve_price: float,
        rng: np.random.Generator,
    ) -> MechanismOutcome:
        valid_bids = sorted_valid_bids(bids)
        winner_bid = choose_lowest_bid(valid_bids, rng)
        if winner_bid is None:
            return MechanismOutcome(self.mechanism_id, None, 0.0, tuple(), reserve_price)
        return MechanismOutcome(
            mechanism_id=self.mechanism_id,
            winner=winner_bid.manufacturer_id,
            payment=winner_bid.price,
            valid_bids=tuple(valid_bids),
            reserve_price=reserve_price,
        )

