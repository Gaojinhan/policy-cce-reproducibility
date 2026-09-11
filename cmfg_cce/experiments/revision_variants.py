from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from cmfg_cce.envs.route_capacity import (
    RouteCapacityConfig,
    RouteManufacturer,
    build_cnc_fleet,
)
from cmfg_cce.policies.base import BiddingPolicy
from cmfg_cce.policies.policy_library import build_policy_library_variant


CORE_POLICY_IDS = tuple(f"A{index}" for index in range(1, 7))
EXPANDED_POLICY_IDS = tuple(f"A{index}" for index in range(1, 9))


@dataclass(frozen=True)
class LibraryVariant:
    variant_id: str
    policy_ids: tuple[str, ...]
    state_coefficient_scale: float = 1.0


def revision_library_variants() -> tuple[LibraryVariant, ...]:
    rows = [
        LibraryVariant("base_a1_a6", CORE_POLICY_IDS),
        LibraryVariant("expanded_a1_a8", EXPANDED_POLICY_IDS),
    ]
    rows.extend(
        LibraryVariant(
            f"leave_out_a{removed}",
            tuple(policy_id for policy_id in CORE_POLICY_IDS if policy_id != f"A{removed}"),
        )
        for removed in range(1, 7)
    )
    rows.extend(
        (
            LibraryVariant("coefficient_scale_0p8", CORE_POLICY_IDS, 0.8),
            LibraryVariant("coefficient_scale_1p2", CORE_POLICY_IDS, 1.2),
        )
    )
    return tuple(rows)


def build_revision_library(
    mechanism_id: str,
    variant: LibraryVariant,
) -> dict[str, BiddingPolicy]:
    return build_policy_library_variant(
        mechanism_id,
        variant.policy_ids,
        state_coefficient_scale=variant.state_coefficient_scale,
    )


def transform_cnc_fleet(
    fleet: Sequence[RouteManufacturer] | None = None,
    *,
    cost_dispersion_scale: float = 1.0,
    alpha_scale: float = 1.0,
) -> tuple[RouteManufacturer, ...]:
    """Apply the frozen mean-preserving manufacturer sensitivity transforms."""

    source = tuple(fleet or build_cnc_fleet())
    if not source:
        raise ValueError("A CNC fleet sensitivity transform requires manufacturers.")
    cost_dispersion_scale = float(cost_dispersion_scale)
    alpha_scale = float(alpha_scale)
    if not np.isfinite(cost_dispersion_scale) or cost_dispersion_scale <= 0.0:
        raise ValueError("cost_dispersion_scale must be finite and positive.")
    if not np.isfinite(alpha_scale) or alpha_scale <= 0.0:
        raise ValueError("alpha_scale must be finite and positive.")
    mean_cost = float(np.mean([manufacturer.cost_multiplier for manufacturer in source]))
    transformed = tuple(
        replace(
            manufacturer,
            cost_multiplier=(
                mean_cost
                + cost_dispersion_scale * (manufacturer.cost_multiplier - mean_cost)
            ),
            alpha=alpha_scale * manufacturer.alpha,
        )
        for manufacturer in source
    )
    if any(manufacturer.cost_multiplier <= 0.0 for manufacturer in transformed):
        raise ValueError("The requested cost-dispersion transform produced a nonpositive cost.")
    return transformed


def transform_route_config(
    config: RouteCapacityConfig,
    *,
    effective_rate_scale: float,
    variant_tag: str,
) -> RouteCapacityConfig:
    effective_rate_scale = float(effective_rate_scale)
    if not np.isfinite(effective_rate_scale) or effective_rate_scale <= 0.0:
        raise ValueError("effective_rate_scale must be finite and positive.")
    tag = str(variant_tag).strip()
    if not tag or any(character in tag for character in ("/", "\\")):
        raise ValueError("variant_tag must be a nonempty path-safe identifier.")
    return replace(
        config,
        effective_rate_scale=effective_rate_scale,
        env_version=f"revision_full_v1__{tag}",
    )
