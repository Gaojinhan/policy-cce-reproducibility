from __future__ import annotations

from cmfg_cce.policies.base import BiddingPolicy
from cmfg_cce.policies.delivery_policies import build_delivery_archetype_policies
from cmfg_cce.policies.price_policies import build_price_archetype_policies


def build_price_policy_library(
    k: int,
    *,
    state_coefficient_scale: float = 1.0,
) -> dict[str, BiddingPolicy]:
    policies = build_price_archetype_policies(
        k,
        state_coefficient_scale=state_coefficient_scale,
    )
    return {policy.name: policy for policy in policies}


def build_delivery_policy_library(
    k: int,
    *,
    state_coefficient_scale: float = 1.0,
) -> dict[str, BiddingPolicy]:
    policies = build_delivery_archetype_policies(
        k,
        state_coefficient_scale=state_coefficient_scale,
    )
    return {policy.name: policy for policy in policies}


def build_policy_library_for_mechanism(
    mechanism_id: str,
    k: int,
    *,
    state_coefficient_scale: float = 1.0,
) -> dict[str, BiddingPolicy]:
    if "delivery" in mechanism_id:
        return build_delivery_policy_library(
            k,
            state_coefficient_scale=state_coefficient_scale,
        )
    return build_price_policy_library(
        k,
        state_coefficient_scale=state_coefficient_scale,
    )


def build_policy_library_variant(
    mechanism_id: str,
    policy_ids: tuple[str, ...],
    *,
    state_coefficient_scale: float = 1.0,
) -> dict[str, BiddingPolicy]:
    """Build an explicitly identified sensitivity library.

    IDs must be a unique subset of A1--A8.  Keeping the original IDs avoids
    relabeling a leave-one-out library as if its fifth entry were A5.
    """

    normalized = tuple(str(value) for value in policy_ids)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("policy_ids must be a nonempty sequence of unique archetype IDs.")
    unknown = set(normalized).difference({f"A{index}" for index in range(1, 9)})
    if unknown:
        raise ValueError(f"Unknown policy archetypes: {sorted(unknown)}")
    full = build_policy_library_for_mechanism(
        mechanism_id,
        8,
        state_coefficient_scale=state_coefficient_scale,
    )
    return {policy_id: full[policy_id] for policy_id in normalized}
