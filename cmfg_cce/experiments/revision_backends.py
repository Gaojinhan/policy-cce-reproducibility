from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping

import yaml

from cmfg_cce.envs.route_capacity import RouteCapacityConfig, build_cnc_fleet
from cmfg_cce.envs.toy import ToyEnvConfig, config_from_mapping
from cmfg_cce.evaluation.route_rollout import RouteCncBackend
from cmfg_cce.evaluation.rollout import RolloutSeeds
from cmfg_cce.evaluation.toy_backend import ToyPolicyGameBackend
from cmfg_cce.experiments.common import LEAD_TIME_GRID, MARKUP_GRID, solver_seed
from cmfg_cce.experiments.revision_full_v1_spec import (
    FormalSeedNamespaces,
    RobustnessSetting,
)
from cmfg_cce.experiments.revision_variants import (
    LibraryVariant,
    build_revision_library,
    transform_cnc_fleet,
    transform_route_config,
)
from cmfg_cce.policies.policy_library import build_policy_library_for_mechanism


ROOT = Path(__file__).resolve().parents[2]
BASE_BENCHMARK_CONFIG = ROOT / "cmfg_cce/configs/oracle_baselines.yaml"
BASE_CNC_CONFIG = ROOT / "cmfg_cce/configs/cnc_route_case.yaml"


def _yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration {path} must contain a mapping.")
    return payload


def base_benchmark_mapping() -> dict[str, object]:
    return _yaml(BASE_BENCHMARK_CONFIG)


def base_benchmark_config() -> ToyEnvConfig:
    return config_from_mapping(base_benchmark_mapping())


def mixed_challenge_config(variant: str) -> ToyEnvConfig:
    """Return one predeclared CMfg game used to challenge pure support.

    Each setting changes one interpretable source of strategic pressure.  The
    five settings are always reported together; they are not searched after
    inspecting which games happen to produce mixed support.
    """

    base = base_benchmark_config()
    variants = {
        "tight_capacity": replace(base, capacity_range=(45.0, 90.0)),
        "tight_due_dates": replace(base, due_date_range=(3, 14)),
        "high_core_pressure": replace(
            base,
            pressure_decay=0.96,
            shock_size_range=(0.15, 0.35),
            max_pressure=0.80,
        ),
        # The midpoint remains 0.8, matching the base interval [0.1, 1.5].
        "high_alpha_heterogeneity": replace(base, alpha_range=(0.0, 1.6)),
        "narrow_capacity": replace(base, capacity_range=(95.0, 105.0)),
    }
    if variant not in variants:
        raise KeyError(f"Unknown mixed-support challenge setting: {variant}")
    return variants[variant]


def _namespace_seeds(
    base: RolloutSeeds,
    namespace: str,
    namespaces: FormalSeedNamespaces | None = None,
) -> RolloutSeeds:
    return (namespaces or FormalSeedNamespaces()).apply(base, namespace)


def build_benchmark_backend(
    *,
    mechanism: str,
    n_agents: int,
    policies_per_agent: int,
    seed: int,
    namespace: str,
    variant: str = "base",
    stream_label: str | None = None,
) -> ToyPolicyGameBackend:
    config = (
        base_benchmark_config()
        if variant in {"base", "exact_full_tensor", "sparse"}
        else mixed_challenge_config(variant)
    )
    policies = build_policy_library_for_mechanism(mechanism, policies_per_agent)
    raw = base_benchmark_mapping()
    policy_cfg = dict(raw.get("policies", {}))
    return ToyPolicyGameBackend(
        mechanism_id=mechanism,
        policies=policies,
        config=config,
        n_agents=n_agents,
        seeds=_namespace_seeds(solver_seed(seed), namespace),
        markup_grid=tuple(float(value) for value in policy_cfg.get("markup_grid", MARKUP_GRID)),
        lead_time_grid=tuple(
            float(value) for value in policy_cfg.get("lead_time_grid", LEAD_TIME_GRID)
        ),
        stream_label=stream_label or namespace,
    )


def base_cnc_config() -> RouteCapacityConfig:
    raw = _yaml(BASE_CNC_CONFIG)
    route = dict(raw.get("route_environment", {}))
    route["due_slack_range"] = tuple(float(value) for value in route["due_slack_range"])
    route["rate_multiplier_bounds"] = tuple(
        float(value) for value in route["rate_multiplier_bounds"]
    )
    return RouteCapacityConfig(**route)


def cnc_seed(seed: int) -> RolloutSeeds:
    return RolloutSeeds(
        type_seed=int(seed),
        order_seed=110_000 + int(seed),
        tie_break_seed=220_000 + int(seed),
        rollout_replication_seed=330_000 + int(seed),
        outside_seed=440_000 + int(seed),
        availability_seed=550_000 + int(seed),
    )


def parse_cnc_condition(condition: str) -> tuple[str, str, str]:
    parts = tuple(str(condition).split("__"))
    if len(parts) != 3:
        raise ValueError(f"Invalid CNC operating condition: {condition!r}")
    load, mix, outside = parts
    mix_internal = {
        "balanced": "balanced",
        "m5_edm_intensive": "bottleneck_heavy",
        "bottleneck_heavy": "bottleneck_heavy",
    }.get(mix)
    outside_internal = {
        "normal": "normal",
        "high_outside_m5_g_edm": "critical_machine",
        "critical_machine": "critical_machine",
    }.get(outside)
    if load not in {"nominal", "high"} or mix_internal is None or outside_internal is None:
        raise ValueError(f"Invalid CNC operating condition: {condition!r}")
    return load, mix_internal, outside_internal


def build_cnc_backend(
    *,
    mechanism: str,
    seed: int,
    condition: str,
    namespace: str,
    library_variant: LibraryVariant | None = None,
    robustness: RobustnessSetting | None = None,
    stream_label: str | None = None,
) -> RouteCncBackend:
    load, mix, outside = parse_cnc_condition(condition)
    config = replace(
        base_cnc_config(),
        load_level=load,
        route_mix=mix,
        outside_regime=outside,
        env_version="revision_full_v1_route_capacity_v1",
    )
    fleet = build_cnc_fleet()
    if robustness is not None:
        config = transform_route_config(
            config,
            effective_rate_scale=robustness.effective_rate_scale,
            variant_tag=robustness.setting_id,
        )
        fleet = transform_cnc_fleet(
            fleet,
            cost_dispersion_scale=robustness.cost_dispersion_scale,
            alpha_scale=robustness.alpha_scale,
        )
        if library_variant is not None:
            raise ValueError("Library and parameter variants must be separate formal experiments.")
        library_variant = LibraryVariant(
            robustness.setting_id,
            tuple(f"A{index}" for index in range(1, 7)),
            robustness.policy_coefficient_scale,
        )
    if library_variant is None:
        policies = build_policy_library_for_mechanism(mechanism, 6)
    else:
        policies = build_revision_library(mechanism, library_variant)
    raw = _yaml(BASE_CNC_CONFIG)
    policy_cfg = dict(raw.get("policies", {}))
    return RouteCncBackend(
        mechanism_id=mechanism,
        policies=policies,
        config=config,
        seeds=_namespace_seeds(cnc_seed(seed), namespace),
        markup_grid=tuple(float(value) for value in policy_cfg["markup_grid"]),
        lead_time_grid=tuple(float(value) for value in policy_cfg["lead_time_grid"]),
        fleet=fleet,
        stream_label=stream_label or namespace,
    )
