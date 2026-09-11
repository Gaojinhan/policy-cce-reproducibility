from __future__ import annotations

from cmfg_cce.policies.archetype_policies import ArchetypePolicy


def build_price_archetype_policies(
    k: int,
    *,
    state_coefficient_scale: float = 1.0,
) -> list[ArchetypePolicy]:
    if k < 1 or k > 8:
        raise ValueError(f"Interpretable price policy library supports 1..8 archetypes, got {k}.")
    return [
        ArchetypePolicy(
            name=f"A{idx}",
            mechanism_family="price",
            state_coefficient_scale=state_coefficient_scale,
        )
        for idx in range(1, k + 1)
    ]
