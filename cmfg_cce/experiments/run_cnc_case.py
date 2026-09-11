from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.stats import t as student_t

from cmfg_cce.envs.route_capacity import (
    ORDER_FAMILY_SPECS,
    RouteCapacityConfig,
    build_cnc_fleet,
    expected_standard_machine_hours,
)
from cmfg_cce.evaluation.backend_cache import BackendPayoffCache
from cmfg_cce.evaluation.cache_io import load_backend_cache, save_backend_cache
from cmfg_cce.evaluation.empirical_game import EmpiricalGame, build_empirical_game
from cmfg_cce.evaluation.full_audit import audit_distribution_on_full_game
from cmfg_cce.evaluation.payoff_cache import profile_space, support_deviation_closure
from cmfg_cce.evaluation.rollout import Profile, RolloutSeeds
from cmfg_cce.evaluation.route_rollout import RouteCncBackend
from cmfg_cce.policies.policy_library import build_policy_library_for_mechanism
from cmfg_cce.solvers.cce_lp import CceSolution, solve_full_cce_lp
from cmfg_cce.solvers.sparse_cce import (
    SparseCceResult,
    audit_sparse_distribution,
    solve_repair_sparse,
)


DEFAULT_CONFIG = Path("cmfg_cce/configs/cnc_route_case.yaml")
DEFAULT_OUTPUT = Path("outputs/cnc_case")
MAIN_GAME_KIND = "main"
SENSITIVITY_GAME_KIND = "sensitivity"
RUN_SIGNATURE_VERSION = "cnc_case_run_v3"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame, *, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_case_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("The CNC case configuration must be a YAML mapping.")
    return raw


def config_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _completed_result_artifacts(result: Mapping[str, Any]) -> tuple[Path, ...]:
    artifacts = result.get("artifacts", {})
    required = ["solver_cache", "complete_cache", "complete_payoff_table"]
    if int(result.get("holdout_rollouts", 0)) > 0:
        required.append("holdout_cache")
    return tuple(Path(str(artifacts[key])) for key in required if artifacts.get(key))


def _validate_completed_result(
    result: Mapping[str, Any],
    *,
    result_path: Path,
    config_hash: str,
    run_signature: str,
) -> None:
    if result.get("config_sha256") != config_hash:
        raise ValueError(f"Existing result at {result_path} uses a different configuration hash.")
    if result.get("run_signature_sha256") != run_signature:
        raise ValueError(
            f"Existing result at {result_path} uses a different effective run signature; "
            "use --force only if an intentional rerun is required."
        )
    if result.get("status") != "complete":
        raise RuntimeError(
            f"Refusing to skip non-complete game result at {result_path}; use --force to rerun."
        )
    artifacts = result.get("artifacts", {})
    required_keys = {"solver_cache", "complete_cache", "complete_payoff_table"}
    if int(result.get("holdout_rollouts", 0)) > 0:
        required_keys.add("holdout_cache")
    missing_metadata = sorted(key for key in required_keys if not artifacts.get(key))
    missing_files = [path for path in _completed_result_artifacts(result) if not path.exists()]
    if missing_metadata or missing_files:
        raise RuntimeError(
            f"Completed result {result_path} has missing artifacts: "
            f"metadata={missing_metadata}, files={[str(path) for path in missing_files]}; "
            "use --force to rebuild this game."
        )


def _tuple_fields(route_values: dict[str, Any]) -> dict[str, Any]:
    converted = dict(route_values)
    for key in ("due_slack_range", "rate_multiplier_bounds"):
        if key in converted:
            converted[key] = tuple(float(item) for item in converted[key])
    return converted


def route_config_from_raw(
    raw: Mapping[str, Any],
    *,
    cell: str | None = None,
    sensitivity_name: str | None = None,
) -> RouteCapacityConfig:
    allowed = {item.name for item in fields(RouteCapacityConfig)}
    values = {
        key: value
        for key, value in _tuple_fields(dict(raw.get("route_environment", {}))).items()
        if key in allowed
    }
    if cell is not None:
        load_level, route_mix, outside_regime = parse_pressure_cell(cell)
        values.update(
            load_level=load_level,
            route_mix=route_mix,
            outside_regime=outside_regime,
        )
    if sensitivity_name is not None:
        variants = raw.get("sensitivity", {}).get("variants", {})
        if sensitivity_name not in variants:
            raise KeyError(f"Unknown CNC sensitivity variant: {sensitivity_name}")
        variant = variants[sensitivity_name]
        values["effective_rate_scale"] = float(variant.get("effective_rate_scale", 1.0))
        values["cost_scale"] = float(variant.get("cost_scale", 1.0))
        # Cache identity explicitly separates every sensitivity table.
        base_version = str(values.get("env_version", RouteCapacityConfig.env_version))
        values["env_version"] = f"{base_version}__{sensitivity_name}"
    return RouteCapacityConfig(**values)


def parse_pressure_cell(cell: str) -> tuple[str, str, str]:
    parts = tuple(str(cell).split("__"))
    if len(parts) != 3:
        raise ValueError(
            "A pressure cell must be '<nominal|high>__<balanced|bottleneck_heavy>__"
            "<normal|critical_machine>'."
        )
    # Let RouteCapacityConfig perform the authoritative level validation.
    RouteCapacityConfig(load_level=parts[0], route_mix=parts[1], outside_regime=parts[2])
    return parts  # type: ignore[return-value]


def factorial_cells(raw: Mapping[str, Any]) -> tuple[str, ...]:
    factorial = raw["factorial"]
    return tuple(
        f"{load}__{mix}__{outside}"
        for load, mix, outside in product(
            factorial["load_levels"],
            factorial["route_mixes"],
            factorial["outside_regimes"],
        )
    )


def treatment_map(raw: Mapping[str, Any]) -> dict[str, str]:
    mapping = {str(key): str(value) for key, value in raw["auction_treatments"].items()}
    if tuple(mapping) != ("AUC1", "AUC2", "AUC3", "AUC4"):
        raise ValueError("The published case requires ordered treatments AUC1--AUC4.")
    return mapping


def normalize_treatment(raw: Mapping[str, Any], treatment: str) -> tuple[str, str]:
    mapping = treatment_map(raw)
    if treatment in mapping:
        return treatment, mapping[treatment]
    reverse = {mechanism: auc for auc, mechanism in mapping.items()}
    if treatment in reverse:
        return reverse[treatment], treatment
    raise KeyError(f"Unknown auction treatment or mechanism ID: {treatment}")


def case_seeds(seed: int) -> RolloutSeeds:
    """Return CRN streams shared across AUC1--AUC4 for one environment seed."""

    return RolloutSeeds(
        type_seed=int(seed),
        order_seed=110_000 + int(seed),
        tie_break_seed=220_000 + int(seed),
        rollout_replication_seed=330_000 + int(seed),
        outside_seed=440_000 + int(seed),
        availability_seed=550_000 + int(seed),
    )


def build_backend(
    raw: Mapping[str, Any],
    route_config: RouteCapacityConfig,
    mechanism: str,
    seed: int,
    *,
    stream_label: str = "main",
    k: int | None = None,
) -> tuple[RouteCncBackend, tuple[str, ...]]:
    policies_cfg = raw["policies"]
    k_value = int(k if k is not None else policies_cfg["policies_per_agent"])
    policies = build_policy_library_for_mechanism(mechanism, k_value)
    backend = RouteCncBackend(
        mechanism_id=mechanism,
        policies=policies,
        config=route_config,
        seeds=case_seeds(seed),
        markup_grid=tuple(float(value) for value in policies_cfg["markup_grid"]),
        lead_time_grid=tuple(float(value) for value in policies_cfg["lead_time_grid"]),
        stream_label=stream_label,
    )
    return backend, tuple(policies)


def _sparse_to_dict(result: SparseCceResult) -> dict[str, Any]:
    return _jsonable(asdict(result))


def _sparse_from_dict(payload: Mapping[str, Any]) -> SparseCceResult:
    known = {item.name for item in fields(SparseCceResult)}
    values = {key: value for key, value in payload.items() if key in known}
    values["support_profiles"] = [tuple(str(item) for item in profile) for profile in values["support_profiles"]]
    values["support_probabilities"] = [float(value) for value in values["support_probabilities"]]
    return SparseCceResult(**values)


def _cce_to_dict(result: CceSolution, *, include_q: bool = False) -> dict[str, Any]:
    payload = result.to_jsonable()
    if include_q:
        payload["q"] = result.q
    return _jsonable(payload)


def _finite_weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    """Average a diagnostic over its observed, positive-probability entries.

    Some per-episode ratios are intentionally missing when their denominator is
    zero.  A missing diagnostic must not poison unrelated equilibrium metrics,
    and a zero-probability profile must not contribute through ``0 * NaN``.
    """

    value_array = np.asarray(values, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    if value_array.shape != weight_array.shape:
        raise ValueError("Metric values and equilibrium weights must have matching shapes.")
    mask = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array > 0.0)
    observed_weight = float(np.sum(weight_array[mask]))
    if not np.any(mask) or observed_weight <= 0.0:
        return float("nan")
    return float(np.dot(value_array[mask], weight_array[mask]) / observed_weight)


def _weighted_cache_metrics(
    cache: BackendPayoffCache,
    support: Sequence[Profile],
    probabilities: Sequence[float],
) -> dict[str, float]:
    keys: set[str] = set()
    for profile in support:
        keys.update(cache.get(profile).mean_metrics)
    probs = np.asarray(probabilities, dtype=float)
    probs /= np.sum(probs)
    return {
        key: _finite_weighted_mean(
            [cache.get(profile).mean_metrics.get(key, float("nan")) for profile in support],
            probs,
        )
        for key in sorted(keys)
    }


def _weighted_game_metrics(game: EmpiricalGame, q: np.ndarray) -> dict[str, float]:
    keys = set().union(*(metrics.keys() for metrics in game.metrics))
    return {
        key: _finite_weighted_mean(
            [metrics.get(key, float("nan")) for metrics in game.metrics],
            q,
        )
        for key in sorted(keys)
    }


def _weighted_cache_returns(
    cache: BackendPayoffCache,
    support: Sequence[Profile],
    probabilities: Sequence[float],
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if probs.size == 0 or not np.all(np.isfinite(probs)) or float(np.sum(probs)) <= 0.0:
        raise ValueError("Equilibrium support probabilities must contain finite positive mass.")
    probs /= np.sum(probs)
    returns = np.vstack([cache.get(profile).mean_returns for profile in support])
    return np.asarray(probs @ returns, dtype=float)


def _weighted_game_returns(game: EmpiricalGame, q: np.ndarray) -> np.ndarray:
    probs = np.asarray(q, dtype=float)
    if probs.shape != (game.n_profiles,) or not np.all(np.isfinite(probs)):
        raise ValueError("The empirical-game distribution has an invalid shape or value.")
    if float(np.sum(probs)) <= 0.0:
        raise ValueError("The empirical-game distribution must contain positive mass.")
    probs /= np.sum(probs)
    return np.asarray(probs @ game.payoffs, dtype=float)


def _relative_gap_percent(gap: float, expected_returns: Sequence[float]) -> float:
    values = np.asarray(expected_returns, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return float("nan")
    scale = max(1.0, float(np.mean(np.abs(values))))
    return float(100.0 * max(0.0, float(gap)) / scale)


def _support_policy_shares(
    support: Sequence[Profile], probabilities: Sequence[float], policy_ids: Sequence[str]
) -> dict[str, float]:
    probs = np.asarray(probabilities, dtype=float)
    probs /= np.sum(probs)
    n_agents = len(support[0])
    return {
        policy: float(
            sum(
                float(prob) * profile.count(policy) / n_agents
                for profile, prob in zip(support, probs, strict=True)
            )
        )
        for policy in policy_ids
    }


def _game_directory(
    output_dir: Path,
    game_kind: str,
    cell: str,
    auc: str,
    seed: int,
    *,
    sensitivity_name: str | None = None,
) -> Path:
    if game_kind == MAIN_GAME_KIND:
        return output_dir / "games" / MAIN_GAME_KIND / cell / auc / f"seed_{seed}"
    if game_kind == SENSITIVITY_GAME_KIND and sensitivity_name:
        return (
            output_dir
            / "games"
            / SENSITIVITY_GAME_KIND
            / sensitivity_name
            / cell
            / auc
            / f"seed_{seed}"
        )
    raise ValueError(f"Unsupported game kind or missing sensitivity name: {game_kind}")


def _solver_kwargs(raw: Mapping[str, Any], cache: BackendPayoffCache, seed: int) -> dict[str, Any]:
    dss = raw["dss"]
    if dss.get("solver") != "REPAIR-SAD-CCE":
        raise ValueError("The CNC case is frozen to the DSS implementation REPAIR-SAD-CCE.")
    repair = dss["repair"]
    return {
        "initial_support_size": int(dss["initial_support_size"]),
        "max_support_size": int(dss["max_support_size"]),
        "support_add_batch_size": int(dss["support_add_batch_size"]),
        "max_rounds": int(dss["max_rounds"]),
        "target_gap": float(dss["target_gap"]),
        "seed": 990_000 + int(seed),
        "rollouts_max": int(dss.get("rollouts_max", cache.n_rollouts)),
        "active_sampling": bool(dss.get("active_sampling", False)),
        "repair_rounds": int(repair["repair_rounds"]),
        "top_constraints": int(repair["top_constraints"]),
        "top_profiles_per_constraint": int(repair["top_profiles_per_constraint"]),
        "q_min": float(repair["q_min"]),
        "mean_threshold": float(repair["mean_threshold"]),
        "contribution_min": float(repair["contribution_min"]),
        "enable_multi_agent_repair": bool(repair["enable_multi_agent_repair"]),
        "max_agents_repaired_per_profile": int(repair["max_agents_repaired_per_profile"]),
        "beam_width": int(repair["beam_width"]),
        "profile_budget_multiplier": float(repair["profile_budget_multiplier"]),
        "targeted_deviation_policies": repair.get("targeted_deviation_policies", {}),
        "solver_name": "DSS-CCE",
    }


def _ensure_profiles_checkpointed(
    cache: BackendPayoffCache,
    profiles: Sequence[Profile],
    cache_path: Path,
    progress_path: Path,
    *,
    batch_size: int,
) -> float:
    progress = _read_json(progress_path) if progress_path.exists() else {}
    elapsed_total = float(progress.get("evaluation_seconds", 0.0))
    missing = [profile for profile in profiles if profile not in cache.estimates]
    for offset in range(0, len(missing), max(1, batch_size)):
        batch = missing[offset : offset + max(1, batch_size)]
        started = time.perf_counter()
        cache.ensure(batch)
        elapsed_total += time.perf_counter() - started
        save_backend_cache(cache, cache_path)
        _atomic_json(
            progress_path,
            {
                "status": "complete" if len(cache.estimates) >= len(profiles) else "in_progress",
                "evaluated_profile_count": len(cache.estimates),
                "target_profile_count": len(profiles),
                "evaluation_seconds": elapsed_total,
            },
        )
    if not missing and not progress_path.exists():
        _atomic_json(
            progress_path,
            {
                "status": "complete",
                "evaluated_profile_count": len(cache.estimates),
                "target_profile_count": len(profiles),
                "evaluation_seconds": elapsed_total,
            },
        )
    return elapsed_total


def run_one_game(
    raw: Mapping[str, Any],
    *,
    config_hash: str,
    output_dir: str | Path,
    cell: str,
    treatment: str,
    seed: int,
    game_kind: str = MAIN_GAME_KIND,
    sensitivity_name: str | None = None,
    force: bool = False,
    rollouts_per_profile: int | None = None,
    holdout_rollouts: int | None = None,
    workers: int | None = None,
    k: int | None = None,
    dss_override: Mapping[str, Any] | None = None,
    run_holdout: bool = True,
) -> dict[str, Any]:
    """Run DSS, complete-table audit, and independent holdout for one game.

    This function is intentionally public enough for the smoke integration
    test and for targeted reruns.  The command-line matrix runner calls it with
    the frozen YAML values.
    """

    output_root = Path(output_dir)
    auc, mechanism = normalize_treatment(raw, treatment)
    route_config = route_config_from_raw(
        raw,
        cell=cell,
        sensitivity_name=sensitivity_name if game_kind == SENSITIVITY_GAME_KIND else None,
    )
    backend, policy_ids = build_backend(raw, route_config, mechanism, int(seed), k=k)
    factorial = raw["factorial"]
    n_rollouts = int(
        rollouts_per_profile
        if rollouts_per_profile is not None
        else factorial["rollouts_per_profile"]
    )
    n_holdout = int(
        holdout_rollouts if holdout_rollouts is not None else factorial["holdout_rollouts"]
    )
    worker_count = int(workers if workers is not None else factorial.get("workers", 1))
    if n_rollouts <= 0 or (run_holdout and n_holdout <= 0) or worker_count <= 0:
        raise ValueError("Rollout counts and worker count must be positive.")
    effective_dss = dss_override if dss_override is not None else raw["dss"]
    signature_payload = {
        "signature_version": RUN_SIGNATURE_VERSION,
        "config_sha256": config_hash,
        "game_kind": game_kind,
        "sensitivity_name": sensitivity_name,
        "pressure_cell": cell,
        "auction_treatment": auc,
        "mechanism": mechanism,
        "seed": int(seed),
        "route_config": asdict(route_config),
        "cache_identity": backend.cache_identity,
        "policy_ids": policy_ids,
        "markup_grid": backend.markup_grid,
        "lead_time_grid": backend.lead_time_grid,
        "rollouts_per_profile": n_rollouts,
        "holdout_rollouts": n_holdout if run_holdout else 0,
        "holdout_cache_identity": (
            backend.holdout_backend().cache_identity if run_holdout else None
        ),
        "workers": worker_count,
        "dss": effective_dss,
    }
    run_signature = _payload_sha256(signature_payload)
    game_dir = _game_directory(
        output_root,
        game_kind,
        cell,
        auc,
        int(seed),
        sensitivity_name=sensitivity_name,
    )
    game_dir.mkdir(parents=True, exist_ok=True)
    result_path = game_dir / "game_result.json"
    if result_path.exists() and not force:
        result = _read_json(result_path)
        _validate_completed_result(
            result,
            result_path=result_path,
            config_hash=config_hash,
            run_signature=run_signature,
        )
        return result

    if force:
        # ``--force`` is intentionally scoped to this exact game directory.
        # Removing stale timing/checkpoint metadata prevents a formal rerun
        # from inheriting seconds or estimates from the earlier execution.
        for stale_name in (
            "game_result.json",
            "dss_checkpoint.json",
            "solver_cache.json",
            "complete_payoff_cache.json",
            "complete_audit_progress.json",
            "holdout_payoff_cache.json",
            "holdout_progress.json",
            "complete_payoff_table.csv.gz",
            "failure.json",
        ):
            stale_path = game_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()

    cache = BackendPayoffCache(backend=backend, n_rollouts=n_rollouts, workers=worker_count)
    solver_cache_path = game_dir / "solver_cache.json"
    complete_cache_path = game_dir / "complete_payoff_cache.json"
    dss_path = game_dir / "dss_checkpoint.json"
    audit_progress_path = game_dir / "complete_audit_progress.json"

    if dss_path.exists() and not force:
        resumed_dss_checkpoint = True
        dss_checkpoint = _read_json(dss_path)
        if dss_checkpoint.get("config_sha256") != config_hash:
            raise ValueError(f"DSS checkpoint at {dss_path} uses a different configuration hash.")
        if dss_checkpoint.get("run_signature_sha256") != run_signature:
            raise ValueError(
                f"DSS checkpoint at {dss_path} uses a different effective run signature; "
                "use --force to restart this game."
            )
        dss_result = _sparse_from_dict(dss_checkpoint["result"])
        solver_runtime = float(dss_checkpoint["solver_runtime_seconds"])
        solver_eval_time = float(dss_checkpoint["solver_payoff_eval_seconds"])
        solver_stats = dict(dss_checkpoint["solver_access_stats"])
        load_path = complete_cache_path if complete_cache_path.exists() else solver_cache_path
        load_backend_cache(cache, load_path)
    else:
        resumed_dss_checkpoint = False
        cache.reset_access_log()
        if cache.evaluated_profile_count != 0:
            raise RuntimeError("DSS timing must begin with a genuinely empty payoff cache.")
        solver_started = time.perf_counter()
        if dss_override is None:
            kwargs = _solver_kwargs(raw, cache, int(seed))
        else:
            # Test/calibration callers may supply a tiny, explicitly marked
            # budget without altering the published YAML.
            local_raw = {**raw, "dss": dss_override}
            kwargs = _solver_kwargs(local_raw, cache, int(seed))
        dss_result = solve_repair_sparse(cache, policy_ids, **kwargs)
        solver_runtime = time.perf_counter() - solver_started
        solver_eval_time = float(cache.eval_time_seconds)
        solver_stats = cache.access_stats()
        save_backend_cache(cache, solver_cache_path)
        _atomic_json(
            dss_path,
            {
                "status": "dss_complete_from_empty_cache",
                "config_sha256": config_hash,
                "run_signature_sha256": run_signature,
                "result": _sparse_to_dict(dss_result),
                "solver_runtime_seconds": solver_runtime,
                "solver_payoff_eval_seconds": solver_eval_time,
                "solver_compute_seconds": max(0.0, solver_runtime - solver_eval_time),
                "solver_access_stats": solver_stats,
            },
        )

    profiles = tuple(profile_space(policy_ids, backend.n_agents))
    checkpoint_batch = int(raw.get("checkpoint_every_profiles", 64))
    audit_eval_time = _ensure_profiles_checkpointed(
        cache,
        profiles,
        complete_cache_path,
        audit_progress_path,
        batch_size=checkpoint_batch,
    )
    if len(cache.estimates) != len(profiles):
        raise RuntimeError(
            f"Complete-table audit has {len(cache.estimates)} profiles; expected {len(profiles)}."
        )

    audit_compute_started = time.perf_counter()
    game = build_empirical_game([cache.estimates[profile] for profile in profiles], policy_ids)
    full_audit = audit_distribution_on_full_game(
        game,
        dss_result.support_profiles,
        dss_result.support_probabilities,
        solver_name="DSS-CCE-CompleteTableAudit",
        status="complete 1296-profile payoff-table audit"
        if len(profiles) == 1296
        else f"complete {len(profiles)}-profile payoff-table audit",
    )
    equilibrium_metrics = _weighted_game_metrics(game, full_audit.q)
    equilibrium_returns = _weighted_game_returns(game, full_audit.q)
    audit_compute_time = time.perf_counter() - audit_compute_started

    holdout_cache_path = game_dir / "holdout_payoff_cache.json"
    holdout_progress_path = game_dir / "holdout_progress.json"
    if run_holdout:
        holdout_backend = backend.holdout_backend()
        holdout_cache = BackendPayoffCache(
            backend=holdout_backend,
            n_rollouts=n_holdout,
            workers=worker_count,
        )
        if holdout_cache_path.exists() and not force:
            load_backend_cache(holdout_cache, holdout_cache_path)
        closure = tuple(
            sorted(support_deviation_closure(dss_result.support_profiles, policy_ids))
        )
        holdout_eval_time = _ensure_profiles_checkpointed(
            holdout_cache,
            closure,
            holdout_cache_path,
            holdout_progress_path,
            batch_size=checkpoint_batch,
        )
        holdout_compute_started = time.perf_counter()
        holdout_audit = audit_sparse_distribution(
            holdout_cache,
            dss_result.support_profiles,
            dss_result.support_probabilities,
            policy_ids,
            solver_name="DSS-CCE-IndependentHoldout",
            status=f"independent {n_holdout}-rollout support-deviation audit",
        )
        holdout_metrics = _weighted_cache_metrics(
            holdout_cache,
            dss_result.support_profiles,
            dss_result.support_probabilities,
        )
        holdout_returns = _weighted_cache_returns(
            holdout_cache,
            dss_result.support_profiles,
            dss_result.support_probabilities,
        )
        holdout_compute_time = time.perf_counter() - holdout_compute_started
        holdout_record: dict[str, Any] = {
            "status": "complete",
            "gap": holdout_audit.cce_gap_nominal,
            "ucb_gap": holdout_audit.cce_gap_ucb,
            "relative_gap_percent": _relative_gap_percent(
                holdout_audit.cce_gap_nominal,
                holdout_returns,
            ),
            "expected_manufacturer_returns": holdout_returns,
            "support_size": holdout_audit.support_size,
            "closure_profile_count": len(closure),
            "common_random_numbers_across_profiles": True,
            "paired_difference_estimator": False,
            "evaluation_seconds": holdout_eval_time,
            "compute_seconds": holdout_compute_time,
            "runtime_seconds": holdout_eval_time + holdout_compute_time,
            "max_deviation": holdout_audit.max_deviation,
            "stream": holdout_backend.cache_identity,
        }
    else:
        holdout_metrics = {}
        holdout_returns = np.full(backend.n_agents, np.nan, dtype=float)
        holdout_record = {
            "status": "not_run_for_sensitivity",
            "reason": "Sensitivity runs use DSS plus complete-table gap audit only.",
            "gap": None,
            "ucb_gap": None,
            "relative_gap_percent": None,
            "expected_manufacturer_returns": None,
            "support_size": None,
            "closure_profile_count": None,
            "common_random_numbers_across_profiles": None,
            "paired_difference_estimator": None,
            "evaluation_seconds": 0.0,
            "compute_seconds": 0.0,
            "runtime_seconds": 0.0,
            "max_deviation": None,
            "stream": None,
        }

    payoff_rows = cache.records(
        {
            "game_kind": game_kind,
            "sensitivity_name": sensitivity_name or "",
            "auction_treatment": auc,
            "config_sha256": config_hash,
        }
    )
    _atomic_csv(
        game_dir / "complete_payoff_table.csv.gz",
        pd.DataFrame(payoff_rows),
        compression="gzip",
    )

    support_shares = _support_policy_shares(
        dss_result.support_profiles,
        dss_result.support_probabilities,
        policy_ids,
    )
    dss_expected_returns = _weighted_cache_returns(
        cache,
        dss_result.support_profiles,
        dss_result.support_probabilities,
    )
    result: dict[str, Any] = {
        "status": "complete",
        "experiment": str(raw["experiment"]),
        "run_budget_label": str(raw.get("run_budget_label", "")),
        "config_sha256": config_hash,
        "run_signature_version": RUN_SIGNATURE_VERSION,
        "run_signature_sha256": run_signature,
        "run_signature": _jsonable(signature_payload),
        "game_kind": game_kind,
        "sensitivity_name": sensitivity_name,
        "pressure_cell": cell,
        "load_level": route_config.load_level,
        "route_mix": route_config.route_mix,
        "outside_regime": route_config.outside_regime,
        "effective_rate_scale": route_config.effective_rate_scale,
        "cost_scale": route_config.cost_scale,
        "auction_treatment": auc,
        "mechanism": mechanism,
        "seed": int(seed),
        "N": backend.n_agents,
        "K": len(policy_ids),
        "horizon": backend.horizon,
        "rollouts_per_profile": n_rollouts,
        "holdout_rollouts": n_holdout if run_holdout else 0,
        "complete_profile_count": len(profiles),
        "cache_identity": backend.cache_identity,
        "dss": {
            "solver": dss_result.solver,
            "started_from_empty_cache": True,
            "resumed_from_recorded_empty_start_checkpoint": resumed_dss_checkpoint,
            "provisional_gap": dss_result.cce_gap_nominal,
            "provisional_ucb_gap": dss_result.cce_gap_ucb,
            "provisional_relative_gap_percent": _relative_gap_percent(
                dss_result.cce_gap_nominal,
                dss_expected_returns,
            ),
            "expected_manufacturer_returns": dss_expected_returns,
            "objective_value": dss_result.objective_value,
            "support_size": dss_result.support_size,
            "support": [
                {"profile": list(profile), "probability": probability}
                for profile, probability in zip(
                    dss_result.support_profiles,
                    dss_result.support_probabilities,
                    strict=True,
                )
            ],
            "support_policy_shares": support_shares,
            "candidate_profile_count": int(solver_stats["solver_required_profile_count"]),
            "profile_coverage": float(
                int(solver_stats["solver_required_profile_count"]) / max(1, len(profiles))
            ),
            "solver_runtime_seconds": solver_runtime,
            "solver_payoff_eval_seconds": solver_eval_time,
            "solver_compute_seconds": max(0.0, solver_runtime - solver_eval_time),
            "solver_access_stats": solver_stats,
            "repair_rounds": dss_result.repair_rounds,
            "repair_profiles_added": dss_result.repair_profiles_added,
            "max_deviation": dss_result.max_deviation,
        },
        "complete_table_audit": {
            **_cce_to_dict(full_audit),
            "relative_gap_percent": _relative_gap_percent(
                full_audit.cce_gap_nominal,
                equilibrium_returns,
            ),
            "expected_manufacturer_returns": equilibrium_returns,
            "gap_difference_vs_provisional": float(
                full_audit.cce_gap_nominal - dss_result.cce_gap_nominal
            ),
            "gap_consistent_with_deviation_closure": bool(
                math.isclose(
                    full_audit.cce_gap_nominal,
                    dss_result.cce_gap_nominal,
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-8,
                )
            ),
            "payoff_table_profiles": len(profiles),
            "posthoc_evaluation_seconds": audit_eval_time,
            "audit_compute_seconds": audit_compute_time,
            "posthoc_audit_runtime_seconds": audit_eval_time + audit_compute_time,
            "solver_runtime_excluded": True,
        },
        "holdout_audit": holdout_record,
        "mechanism_metrics": equilibrium_metrics,
        "holdout_mechanism_metrics": holdout_metrics,
        "artifacts": {
            "game_directory": str(game_dir),
            "solver_cache": str(solver_cache_path),
            "complete_cache": str(complete_cache_path),
            "holdout_cache": str(holdout_cache_path) if run_holdout else None,
            "complete_payoff_table": str(game_dir / "complete_payoff_table.csv.gz"),
        },
    }
    result = _jsonable(result)
    _atomic_json(result_path, result)
    return result


def _calibration_capability_counts() -> dict[str, int]:
    fleet = build_cnc_fleet()
    counts: dict[str, int] = {}
    for family, spec in ORDER_FAMILY_SPECS.items():
        required = {group for group, _, _ in spec.route}
        counts[family.value] = sum(required.issubset(manufacturer.capabilities) for manufacturer in fleet)
    return counts


def _write_empty_result_templates(output_dir: Path) -> None:
    """Create declared table schemas without inserting a result direction."""

    template_dir = output_dir / "templates"
    schemas = {
        "auction_outcomes_across_pressure_cells.csv": [
            "pressure_cell",
            "auction_treatment",
            "seed",
            "assignment_rate",
            "conditional_bid_rate",
            "platform_total_payment",
            "payment_per_assignment",
            "manufacturer_total_profit",
            "normalized_winner_hhi",
        ],
        "machine_group_bottleneck_diagnostics.csv": [
            "pressure_cell",
            "auction_treatment",
            "seed",
            "machine_group",
            "utilization",
            "slack",
            "capacity_shortfall_rate",
            "aggregate_feasible_but_route_infeasible_rate",
        ],
        "dss_bridge_and_transplant_results.csv": [
            "record_type",
            "pressure_cell",
            "source_treatment",
            "target_treatment",
            "seed",
            "dss_complete_table_gap",
            "fullspace_gap",
            "transplanted_gap",
            "runtime_seconds",
        ],
        "matched_contrasts.csv": [
            "contrast_axis",
            "pressure_cell",
            "left_treatment",
            "right_treatment",
            "metric",
            "paired_seed_count",
            "mean_left_minus_right",
            "sample_sd",
        ],
    }
    for filename, columns in schemas.items():
        path = template_dir / filename
        if not path.exists():
            _atomic_csv(path, pd.DataFrame(columns=columns))


def run_pilot(
    raw: Mapping[str, Any],
    *,
    config_hash: str,
    output_dir: str | Path,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate frozen inputs without using a reporting seed or ranking AUCs."""

    output_root = Path(output_dir)
    pilot_dir = output_root / "calibration"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    seed = int(raw["calibration_seed"])
    pilot_cfg = raw["pilot"]
    episode_rollouts = int(pilot_cfg["episode_rollouts"])
    if episode_rollouts < 2:
        raise ValueError("The capacity-coverage pilot requires at least two rollout clusters.")
    confidence_level = float(pilot_cfg["confidence_level"])
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("Pilot confidence_level must lie strictly between 0.5 and 1.")
    coverage_requirements = {
        str(cell): float(threshold)
        for cell, threshold in pilot_cfg["coverage_cells"].items()
    }
    expected_cells = set(factorial_cells(raw))
    if set(coverage_requirements) != expected_cells:
        raise ValueError(
            "pilot.coverage_cells must specify each factorial pressure cell exactly once."
        )
    if not all(0.0 <= threshold <= 1.0 for threshold in coverage_requirements.values()):
        raise ValueError("Pilot coverage thresholds must lie in [0, 1].")
    calibration_treatment = str(pilot_cfg["calibration_treatment"])
    calibration_policy = str(pilot_cfg["calibration_profile"])
    coverage_metric = "at_least_three_route_capacity_feasible_provider_rate"
    capability_counts = _calibration_capability_counts()
    capability_check = all(count >= 3 for count in capability_counts.values())

    balanced = route_config_from_raw(raw, cell="nominal__balanced__normal")
    bottleneck = route_config_from_raw(raw, cell="nominal__bottleneck_heavy__normal")
    expected_balanced = expected_standard_machine_hours(balanced)
    expected_bottleneck = expected_standard_machine_hours(bottleneck)
    workload_equalization_check = math.isclose(
        expected_balanced,
        expected_bottleneck,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    )

    episode_rows: list[dict[str, Any]] = []
    for cell, threshold in coverage_requirements.items():
        route_config = route_config_from_raw(raw, cell=str(cell))
        _, mechanism = normalize_treatment(raw, calibration_treatment)
        backend, policy_ids = build_backend(raw, route_config, mechanism, seed)
        if calibration_policy not in policy_ids:
            raise ValueError(
                f"Calibration policy {calibration_policy!r} is absent from the active policy library."
            )
        profile = tuple(calibration_policy for _ in range(backend.n_agents))
        started = time.perf_counter()
        estimate = backend.estimate(profile, episode_rollouts)
        elapsed = time.perf_counter() - started
        coverage_mean = float(estimate.mean_metrics[coverage_metric])
        coverage_sd = math.sqrt(max(0.0, float(estimate.var_metrics[coverage_metric])))
        coverage_se = coverage_sd / math.sqrt(episode_rollouts)
        critical_value = float(
            student_t.ppf(confidence_level, df=episode_rollouts - 1)
        )
        coverage_lcb = coverage_mean - critical_value * coverage_se
        episode_rows.append(
            {
                "pressure_cell": cell,
                "auction_treatment": calibration_treatment,
                "mechanism": mechanism,
                "profile": list(profile),
                "rollouts": episode_rollouts,
                "runtime_seconds": elapsed,
                "expected_standard_machine_hours": expected_standard_machine_hours(route_config),
                "three_provider_coverage_threshold": threshold,
                "three_provider_coverage_confidence_level": confidence_level,
                "three_provider_coverage_rollout_sd": coverage_sd,
                "three_provider_coverage_standard_error": coverage_se,
                "three_provider_coverage_one_sided_lcb": coverage_lcb,
                "three_provider_coverage_gate_passed": coverage_lcb >= threshold,
                **estimate.mean_metrics,
            }
        )

    # Exercise the one frozen DSS budget in the nominal and combined-stress
    # cells.  AUC1 is used as a single numerical check; no treatment ranking is
    # computed and the outputs never enter the formal result aggregates.
    dss_pilot_rows: list[dict[str, Any]] = []
    dss_pilot_rollouts = int(pilot_cfg["dss_rollouts_per_profile"])
    for cell in pilot_cfg["dss_cells"]:
        route_config = route_config_from_raw(raw, cell=str(cell))
        _, mechanism = normalize_treatment(raw, calibration_treatment)
        dss_backend, dss_policy_ids = build_backend(raw, route_config, mechanism, seed)
        dss_cache = BackendPayoffCache(
            backend=dss_backend,
            n_rollouts=dss_pilot_rollouts,
            workers=int(raw["factorial"].get("workers", 1)),
        )
        if dss_cache.evaluated_profile_count != 0:
            raise RuntimeError("Pilot DSS budget check did not start from an empty cache.")
        dss_cache.reset_access_log()
        dss_started = time.perf_counter()
        dss_result = solve_repair_sparse(
            dss_cache,
            dss_policy_ids,
            **_solver_kwargs(raw, dss_cache, seed),
        )
        dss_elapsed = time.perf_counter() - dss_started
        dss_stats = dss_cache.access_stats()
        finite_output = all(
            math.isfinite(float(value))
            for value in (
                dss_result.cce_gap_nominal,
                dss_result.cce_gap_ucb,
                dss_result.objective_value,
            )
        )
        dss_pilot_rows.append(
            {
                "pressure_cell": cell,
                "auction_treatment": calibration_treatment,
                "mechanism": mechanism,
                "calibration_only": True,
                "started_from_empty_cache": True,
                "rollouts_per_profile": dss_pilot_rollouts,
                "provisional_gap": dss_result.cce_gap_nominal,
                "provisional_ucb_gap": dss_result.cce_gap_ucb,
                "objective_value": dss_result.objective_value,
                "support_size": dss_result.support_size,
                "candidate_profile_count": int(
                    dss_stats["solver_required_profile_count"]
                ),
                "complete_profile_count": len(dss_policy_ids) ** dss_backend.n_agents,
                "runtime_seconds": dss_elapsed,
                "payoff_evaluation_seconds": dss_cache.eval_time_seconds,
                "finite_output": finite_output,
                "nonempty_support": dss_result.support_size > 0,
                "candidate_count_within_complete_space": int(
                    dss_stats["solver_required_profile_count"]
                )
                <= len(dss_policy_ids) ** dss_backend.n_agents,
            }
        )

    # One complete 6^4 game measures backend scaling; it is not a mechanism result.
    complete_k = int(pilot_cfg["complete_game_policies_per_agent"])
    complete_rollouts = int(pilot_cfg["rollouts_per_profile"])
    route_config = route_config_from_raw(raw, cell=next(iter(coverage_requirements)))
    _, mechanism = normalize_treatment(raw, calibration_treatment)
    backend, policy_ids = build_backend(raw, route_config, mechanism, seed, k=complete_k)
    profiles = tuple(profile_space(policy_ids, backend.n_agents))
    cache = BackendPayoffCache(
        backend=backend,
        n_rollouts=complete_rollouts,
        workers=int(raw["factorial"].get("workers", 1)),
    )
    complete_started = time.perf_counter()
    cache.ensure(profiles)
    complete_runtime = time.perf_counter() - complete_started

    assignment_values = [float(row["assignment_rate"]) for row in episode_rows]
    bid_values = [float(row["conditional_bid_rate"]) for row in episode_rows]
    utilization_values = [float(row["mean_platform_queue_utilization"]) for row in episode_rows]
    positive_assignment = all(value > 0.0 for value in assignment_values)
    nonzero_bidding = all(value > 0.0 for value in bid_values)
    nonidle_resources = all(value > 0.0 for value in utilization_values)
    base_cell = "nominal__balanced__normal"
    combined_cell = "high__bottleneck_heavy__critical_machine"
    coverage_by_cell = {
        str(row["pressure_cell"]): float(row[coverage_metric])
        for row in episode_rows
    }
    coverage_gates_by_cell = {
        str(row["pressure_cell"]): bool(row["three_provider_coverage_gate_passed"])
        for row in episode_rows
    }
    checks = {
        "at_least_three_capability_feasible_providers_per_family": capability_check,
        "route_mix_expected_hours_equalized": workload_equalization_check,
        "every_pilot_has_positive_assignment": positive_assignment,
        "every_pilot_has_nonzero_bidding": nonzero_bidding,
        "every_pilot_has_nonidle_resources": nonidle_resources,
        "base_nominal_three_provider_coverage_lcb_at_least_80_percent": (
            coverage_gates_by_cell[base_cell]
            and math.isclose(coverage_requirements[base_cell], 0.80)
        ),
        "combined_stress_three_provider_coverage_lcb_at_least_70_percent": (
            coverage_gates_by_cell[combined_cell]
            and math.isclose(coverage_requirements[combined_cell], 0.70)
        ),
        "every_factorial_cell_passes_frozen_three_provider_coverage_gate": all(
            coverage_gates_by_cell.values()
        ),
        "combined_stress_reduces_three_provider_coverage": (
            coverage_by_cell[combined_cell] < coverage_by_cell[base_cell]
        ),
        "published_main_design_has_96_games": (
            len(factorial_cells(raw))
            * len(treatment_map(raw))
            * len(raw["main_seeds"])
            == 96
        ),
        "bridge_design_has_24_fullspace_lps": (
            len(raw["bridge"]["cells"])
            * len(treatment_map(raw))
            * len(raw["main_seeds"])
            == 24
        ),
        "sensitivity_is_nominal_balanced_normal": (
            str(raw["sensitivity"]["base_cell"]) == "nominal__balanced__normal"
        ),
        "pilot_dss_outputs_are_finite": all(
            bool(row["finite_output"]) for row in dss_pilot_rows
        ),
        "pilot_dss_supports_are_nonempty": all(
            bool(row["nonempty_support"]) for row in dss_pilot_rows
        ),
        "pilot_dss_candidate_counts_within_complete_space": all(
            bool(row["candidate_count_within_complete_space"])
            for row in dss_pilot_rows
        ),
    }
    passed = all(checks.values())
    report = {
        "status": "passed" if passed else "needs_calibration_review",
        "config_sha256": config_hash,
        "calibration_seed": seed,
        "reporting_seed": False,
        "mechanism_rankings_inspected": False,
        "capability_provider_counts": capability_counts,
        "three_provider_capacity_coverage_design": {
            "metric": coverage_metric,
            "calibration_treatment": calibration_treatment,
            "calibration_policy": calibration_policy,
            "rollouts": episode_rollouts,
            "independent_cluster": "rollout",
            "one_sided_confidence_level": confidence_level,
            "thresholds": coverage_requirements,
        },
        "expected_standard_hours": {
            "balanced_nominal": expected_balanced,
            "bottleneck_heavy_nominal": expected_bottleneck,
            "high_multiplier": route_config_from_raw(
                raw, cell="high__balanced__normal"
            ).offered_load_multiplier,
        },
        "checks": checks,
        "pilot_profiles": episode_rows,
        "pilot_dss_budget_checks": dss_pilot_rows,
        "complete_game_timing": {
            "K": complete_k,
            "N": backend.n_agents,
            "profile_count": len(profiles),
            "rollouts_per_profile": complete_rollouts,
            "runtime_seconds": complete_runtime,
            "estimated_main_game_seconds_linear": complete_runtime
            * (int(raw["policies"]["policies_per_agent"]) ** backend.n_agents)
            / max(1, len(profiles))
            * int(raw["factorial"]["rollouts_per_profile"])
            / max(1, complete_rollouts),
        },
        "formal_design_counts": {
            "pressure_cells": len(factorial_cells(raw)),
            "auction_treatments": len(treatment_map(raw)),
            "reporting_seeds": len(raw["main_seeds"]),
            "main_empirical_games": len(factorial_cells(raw))
            * len(treatment_map(raw))
            * len(raw["main_seeds"]),
            "profiles_per_main_game": int(raw["policies"]["policies_per_agent"]) ** 4,
            "bridge_fullspace_lps": len(raw["bridge"]["cells"])
            * len(treatment_map(raw))
            * len(raw["main_seeds"]),
            "sensitivity_games": len(raw["sensitivity"]["variants"])
            * len(treatment_map(raw))
            * len(raw["main_seeds"]),
        },
        "frozen_dss_budget": raw["dss"],
    }
    _atomic_json(pilot_dir / "pilot_report.json", report)
    manifest = {
        "experiment": raw["experiment"],
        "config_sha256": config_hash,
        "runner_signature_version": RUN_SIGNATURE_VERSION,
        "calibration_seed": seed,
        "frozen": passed,
        "formal_results_may_start": passed,
        "calibration_report": str(pilot_dir / "pilot_report.json"),
    }
    _atomic_json(pilot_dir / "config_manifest.json", manifest)
    _atomic_csv(pilot_dir / "pilot_metrics.csv", pd.DataFrame(episode_rows))
    _atomic_csv(pilot_dir / "pilot_dss_budget_checks.csv", pd.DataFrame(dss_pilot_rows))
    _write_empty_result_templates(output_root)
    if strict and not passed:
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"CNC pilot did not pass frozen-input checks: {failed}")
    return report


def ensure_frozen_config(output_dir: Path, config_hash: str, *, allow_unfrozen: bool) -> None:
    if allow_unfrozen:
        return
    manifest_path = output_dir / "calibration" / "config_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Run '--stage pilot' before formal CNC experiments.")
    manifest = _read_json(manifest_path)
    if manifest.get("config_sha256") != config_hash:
        raise RuntimeError("The YAML changed after calibration; rerun the pilot before formal experiments.")
    if not manifest.get("frozen", False):
        raise RuntimeError("Calibration checks have not frozen this CNC configuration.")


def run_main_matrix(
    raw: Mapping[str, Any],
    *,
    config_hash: str,
    output_dir: str | Path,
    force: bool = False,
    max_games: int | None = None,
) -> list[dict[str, Any]]:
    expected_games = len(factorial_cells(raw)) * len(treatment_map(raw)) * len(raw["main_seeds"])
    if expected_games != 96:
        raise ValueError(f"The frozen published main design must contain 96 games, not {expected_games}.")
    results: list[dict[str, Any]] = []
    executed_count = 0
    for cell in factorial_cells(raw):
        for auc in treatment_map(raw):
            for seed in raw["main_seeds"]:
                result_path = _game_directory(
                    Path(output_dir), MAIN_GAME_KIND, cell, auc, int(seed)
                ) / "game_result.json"
                needs_execution = bool(force or not result_path.exists())
                if max_games is not None and needs_execution and executed_count >= max_games:
                    refresh_summaries(Path(output_dir), raw=raw)
                    return results
                results.append(
                    _run_one_game_recording_failure(
                        raw,
                        config_hash=config_hash,
                        output_dir=output_dir,
                        cell=cell,
                        treatment=auc,
                        seed=int(seed),
                        force=force,
                    )
                )
                executed_count += int(needs_execution)
                refresh_summaries(Path(output_dir), raw=raw)
    return results


def run_sensitivity_matrix(
    raw: Mapping[str, Any],
    *,
    config_hash: str,
    output_dir: str | Path,
    force: bool = False,
    max_games: int | None = None,
) -> list[dict[str, Any]]:
    sensitivity = raw["sensitivity"]
    if str(sensitivity["base_cell"]) != "nominal__balanced__normal":
        raise ValueError(
            "Published rate/cost sensitivities are restricted to the nominal, balanced, normal cell."
        )
    expected_games = (
        len(sensitivity["variants"]) * len(treatment_map(raw)) * len(raw["main_seeds"])
    )
    if expected_games != 48:
        raise ValueError(
            f"The frozen sensitivity design must contain 48 games, not {expected_games}."
        )
    results: list[dict[str, Any]] = []
    executed_count = 0
    for variant in sensitivity["variants"]:
        for auc in treatment_map(raw):
            for seed in raw["main_seeds"]:
                result_path = _game_directory(
                    Path(output_dir),
                    SENSITIVITY_GAME_KIND,
                    str(sensitivity["base_cell"]),
                    auc,
                    int(seed),
                    sensitivity_name=str(variant),
                ) / "game_result.json"
                needs_execution = bool(force or not result_path.exists())
                if max_games is not None and needs_execution and executed_count >= max_games:
                    refresh_summaries(Path(output_dir), raw=raw)
                    return results
                results.append(
                    _run_one_game_recording_failure(
                        raw,
                        config_hash=config_hash,
                        output_dir=output_dir,
                        cell=str(sensitivity["base_cell"]),
                        treatment=auc,
                        seed=int(seed),
                        game_kind=SENSITIVITY_GAME_KIND,
                        sensitivity_name=str(variant),
                        run_holdout=False,
                        force=force,
                    )
                )
                executed_count += int(needs_execution)
                refresh_summaries(Path(output_dir), raw=raw)
    return results


def _run_one_game_recording_failure(
    raw: Mapping[str, Any],
    *,
    config_hash: str,
    output_dir: str | Path,
    cell: str,
    treatment: str,
    seed: int,
    game_kind: str = MAIN_GAME_KIND,
    sensitivity_name: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Persist an exact failed-game record before propagating the error."""

    auc, mechanism = normalize_treatment(raw, treatment)
    game_dir = _game_directory(
        Path(output_dir),
        game_kind,
        cell,
        auc,
        int(seed),
        sensitivity_name=sensitivity_name,
    )
    failure_path = game_dir / "failure.json"
    try:
        result = run_one_game(
            raw,
            config_hash=config_hash,
            output_dir=output_dir,
            cell=cell,
            treatment=treatment,
            seed=seed,
            game_kind=game_kind,
            sensitivity_name=sensitivity_name,
            **kwargs,
        )
    except Exception as error:
        _atomic_json(
            failure_path,
            {
                "status": "failed",
                "experiment": raw.get("experiment"),
                "config_sha256": config_hash,
                "game_kind": game_kind,
                "sensitivity_name": sensitivity_name,
                "pressure_cell": cell,
                "auction_treatment": auc,
                "mechanism": mechanism,
                "seed": int(seed),
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        refresh_summaries(Path(output_dir), raw=raw)
        raise
    if failure_path.exists():
        failure_path.unlink()
    return result


def _load_complete_game(
    raw: Mapping[str, Any], result: Mapping[str, Any]
) -> tuple[BackendPayoffCache, EmpiricalGame, tuple[str, ...]]:
    route_config = route_config_from_raw(
        raw,
        cell=str(result["pressure_cell"]),
        sensitivity_name=(
            str(result["sensitivity_name"])
            if result.get("game_kind") == SENSITIVITY_GAME_KIND
            else None
        ),
    )
    backend, policy_ids = build_backend(
        raw,
        route_config,
        str(result["mechanism"]),
        int(result["seed"]),
        k=int(result["K"]),
    )
    cache = BackendPayoffCache(
        backend=backend,
        n_rollouts=int(result["rollouts_per_profile"]),
        workers=1,
    )
    cache_path = Path(str(result["artifacts"]["complete_cache"]))
    load_backend_cache(cache, cache_path)
    profiles = tuple(profile_space(policy_ids, backend.n_agents))
    missing = [profile for profile in profiles if profile not in cache.estimates]
    if missing:
        raise RuntimeError(f"Complete table {cache_path} is missing {len(missing)} profiles.")
    game = build_empirical_game([cache.estimates[profile] for profile in profiles], policy_ids)
    return cache, game, policy_ids


def run_bridge_validation(
    raw: Mapping[str, Any],
    *,
    config_hash: str,
    output_dir: str | Path,
    force: bool = False,
) -> list[dict[str, Any]]:
    output_root = Path(output_dir)
    expected_lps = (
        len(raw["bridge"]["cells"]) * len(treatment_map(raw)) * len(raw["main_seeds"])
    )
    if expected_lps != 24:
        raise ValueError(f"The frozen bridge design must contain 24 LPs, not {expected_lps}.")
    records: list[dict[str, Any]] = []
    for cell in raw["bridge"]["cells"]:
        for auc in treatment_map(raw):
            for seed in raw["main_seeds"]:
                game_dir = _game_directory(output_root, MAIN_GAME_KIND, str(cell), auc, int(seed))
                game_result_path = game_dir / "game_result.json"
                if not game_result_path.exists():
                    raise FileNotFoundError(f"Bridge requires completed main game: {game_result_path}")
                bridge_path = game_dir / "bridge_result.json"
                if bridge_path.exists() and not force:
                    existing = _read_json(bridge_path)
                    if existing.get("config_sha256") != config_hash or existing.get("status") != "complete":
                        raise ValueError(
                            f"Bridge checkpoint {bridge_path} is stale or incomplete; use --force."
                        )
                    records.append(existing)
                    continue
                game_result = _read_json(game_result_path)
                if game_result.get("config_sha256") != config_hash:
                    raise ValueError(f"Main game {game_result_path} uses a different config hash.")
                _, game, _ = _load_complete_game(raw, game_result)
                started = time.perf_counter()
                solution = solve_full_cce_lp(game)
                runtime = time.perf_counter() - started
                fullspace_returns = _weighted_game_returns(game, solution.q)
                record = {
                    "status": "complete",
                    "config_sha256": config_hash,
                    "pressure_cell": cell,
                    "auction_treatment": auc,
                    "mechanism": game_result["mechanism"],
                    "seed": int(seed),
                    "fullspace_lp_runtime_seconds": runtime,
                    "fullspace_solution": _cce_to_dict(solution),
                    "fullspace_expected_manufacturer_returns": fullspace_returns,
                    "fullspace_relative_gap_percent": _relative_gap_percent(
                        solution.cce_gap_nominal,
                        fullspace_returns,
                    ),
                    "dss_complete_table_gap": game_result["complete_table_audit"][
                        "cce_gap_nominal"
                    ],
                    "dss_complete_table_relative_gap_percent": game_result[
                        "complete_table_audit"
                    ]["relative_gap_percent"],
                    "dss_complete_table_objective": game_result["complete_table_audit"][
                        "objective_value"
                    ],
                    "objective_difference_dss_minus_fullspace": float(
                        game_result["complete_table_audit"]["objective_value"]
                        - solution.objective_value
                    ),
                }
                _atomic_json(bridge_path, record)
                records.append(record)
    _write_bridge_summary(output_root, records)
    return records


def _source_support(result: Mapping[str, Any]) -> tuple[list[Profile], list[float]]:
    profiles = [
        tuple(str(value) for value in item["profile"])
        for item in result["dss"]["support"]
    ]
    probabilities = [float(item["probability"]) for item in result["dss"]["support"]]
    return profiles, probabilities


def run_policy_transplants(
    raw: Mapping[str, Any],
    *,
    config_hash: str,
    output_dir: str | Path,
    force: bool = False,
) -> list[dict[str, Any]]:
    output_root = Path(output_dir)
    expected_transplants = (
        len(factorial_cells(raw))
        * len(raw["policy_transplant"]["pairs"])
        * 2
        * len(raw["main_seeds"])
    )
    if expected_transplants != 96:
        raise ValueError(
            f"The frozen policy-transplant design must contain 96 directed audits, not {expected_transplants}."
        )
    transplant_dir = output_root / "transplants"
    transplant_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for cell in factorial_cells(raw):
        for pair in raw["policy_transplant"]["pairs"]:
            for source_auc, target_auc in (tuple(pair), tuple(reversed(pair))):
                for seed in raw["main_seeds"]:
                    record_path = (
                        transplant_dir
                        / cell
                        / f"{source_auc}_to_{target_auc}"
                        / f"seed_{seed}.json"
                    )
                    if record_path.exists() and not force:
                        existing = _read_json(record_path)
                        if existing.get("config_sha256") != config_hash or existing.get("status") != "complete":
                            raise ValueError(
                                f"Transplant checkpoint {record_path} is stale or incomplete; use --force."
                            )
                        records.append(existing)
                        continue
                    source_path = _game_directory(
                        output_root, MAIN_GAME_KIND, cell, source_auc, int(seed)
                    ) / "game_result.json"
                    target_path = _game_directory(
                        output_root, MAIN_GAME_KIND, cell, target_auc, int(seed)
                    ) / "game_result.json"
                    if not source_path.exists() or not target_path.exists():
                        raise FileNotFoundError(
                            f"Policy transplant requires {source_path} and {target_path}."
                        )
                    source_result = _read_json(source_path)
                    target_result = _read_json(target_path)
                    if {
                        source_result.get("config_sha256"),
                        target_result.get("config_sha256"),
                    } != {config_hash}:
                        raise ValueError("A transplant source or target uses a different config hash.")
                    _, target_game, _ = _load_complete_game(raw, target_result)
                    support, probabilities = _source_support(source_result)
                    transplanted = audit_distribution_on_full_game(
                        target_game,
                        support,
                        probabilities,
                        solver_name=f"{source_auc}-distribution-in-{target_auc}",
                        status="cross-mechanism fixed-policy-distribution audit",
                    )
                    fixed_target_metrics = _weighted_game_metrics(target_game, transplanted.q)
                    transplanted_returns = _weighted_game_returns(target_game, transplanted.q)
                    record = {
                        "status": "complete",
                        "config_sha256": config_hash,
                        "pressure_cell": cell,
                        "seed": int(seed),
                        "source_treatment": source_auc,
                        "source_mechanism": source_result["mechanism"],
                        "target_treatment": target_auc,
                        "target_mechanism": target_result["mechanism"],
                        "transplanted_cce_gap": transplanted.cce_gap_nominal,
                        "transplanted_ucb_gap": transplanted.cce_gap_ucb,
                        "transplanted_relative_gap_percent": _relative_gap_percent(
                            transplanted.cce_gap_nominal,
                            transplanted_returns,
                        ),
                        "transplanted_expected_manufacturer_returns": transplanted_returns,
                        "transplanted_max_deviation": transplanted.max_deviation,
                        "source_equilibrium_metrics": source_result["mechanism_metrics"],
                        "fixed_distribution_target_metrics": fixed_target_metrics,
                        "target_reselected_equilibrium_metrics": target_result["mechanism_metrics"],
                        "target_reselected_cce_gap": target_result["complete_table_audit"][
                            "cce_gap_nominal"
                        ],
                        "target_reselected_relative_gap_percent": target_result[
                            "complete_table_audit"
                        ]["relative_gap_percent"],
                    }
                    _atomic_json(record_path, record)
                    records.append(record)
    _write_transplant_summary(output_root, records)
    return records


def _identity_row(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "game_kind": result.get("game_kind"),
        "sensitivity_name": result.get("sensitivity_name"),
        "pressure_cell": result.get("pressure_cell"),
        "load_level": result.get("load_level"),
        "route_mix": result.get("route_mix"),
        "outside_regime": result.get("outside_regime"),
        "effective_rate_scale": result.get("effective_rate_scale"),
        "cost_scale": result.get("cost_scale"),
        "auction_treatment": result.get("auction_treatment"),
        "mechanism": result.get("mechanism"),
        "seed": result.get("seed"),
        "N": result.get("N"),
        "K": result.get("K"),
        "horizon": result.get("horizon"),
        "config_sha256": result.get("config_sha256"),
    }


def refresh_summaries(output_dir: Path, *, raw: Mapping[str, Any] | None = None) -> None:
    result_paths = sorted((output_dir / "games").glob("**/game_result.json"))
    failure_paths = sorted((output_dir / "games").glob("**/failure.json"))
    results = [_read_json(path) for path in result_paths]
    failures = [_read_json(path) for path in failure_paths]
    tables = output_dir / "tables"
    if raw is not None:
        expected_main_keys = {
            (cell, auc, int(seed))
            for cell in factorial_cells(raw)
            for auc in treatment_map(raw)
            for seed in raw["main_seeds"]
        }
        sensitivity = raw["sensitivity"]
        expected_sensitivity_keys = {
            (str(variant), str(sensitivity["base_cell"]), auc, int(seed))
            for variant in sensitivity["variants"]
            for auc in treatment_map(raw)
            for seed in raw["main_seeds"]
        }
        observed_main_keys = {
            (
                str(item.get("pressure_cell")),
                str(item.get("auction_treatment")),
                int(item.get("seed")),
            )
            for item in results
            if item.get("game_kind") == MAIN_GAME_KIND
            and item.get("status") == "complete"
        }
        observed_sensitivity_keys = {
            (
                str(item.get("sensitivity_name")),
                str(item.get("pressure_cell")),
                str(item.get("auction_treatment")),
                int(item.get("seed")),
            )
            for item in results
            if item.get("game_kind") == SENSITIVITY_GAME_KIND
            and item.get("status") == "complete"
        }
        completed_main = len(observed_main_keys & expected_main_keys)
        completed_sensitivity = len(
            observed_sensitivity_keys & expected_sensitivity_keys
        )
        unexpected_result_count = len(observed_main_keys - expected_main_keys) + len(
            observed_sensitivity_keys - expected_sensitivity_keys
        )
        expected_main_count = len(expected_main_keys)
        expected_sensitivity_count = len(expected_sensitivity_keys)
    else:
        completed_main = sum(
            item.get("game_kind") == MAIN_GAME_KIND and item.get("status") == "complete"
            for item in results
        )
        completed_sensitivity = sum(
            item.get("game_kind") == SENSITIVITY_GAME_KIND
            and item.get("status") == "complete"
            for item in results
        )
        unexpected_result_count = 0
        expected_main_count = 96
        expected_sensitivity_count = 48
    _atomic_json(
        tables / "cnc_execution_status.json",
        {
            "completed_main_games": completed_main,
            "expected_main_games": expected_main_count,
            "main_complete": completed_main == expected_main_count,
            "completed_sensitivity_games": completed_sensitivity,
            "expected_sensitivity_games": expected_sensitivity_count,
            "sensitivity_complete": completed_sensitivity
            == expected_sensitivity_count,
            "unexpected_completed_result_count": unexpected_result_count,
            "recorded_failure_count": len(failures),
            "all_recorded_games_successful": len(failures) == 0,
        },
    )
    if failures:
        _atomic_csv(tables / "cnc_failures.csv", pd.DataFrame(failures))
        _atomic_json(tables / "cnc_failures.json", failures)
    elif (tables / "cnc_failures.csv").exists():
        _atomic_csv(tables / "cnc_failures.csv", pd.DataFrame())
        _atomic_json(tables / "cnc_failures.json", [])
    if not results:
        return
    summary_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    chart_metrics = {
        "assignment_rate",
        "conditional_bid_rate",
        "platform_total_payment",
        "payment_per_assignment",
        "manufacturer_total_profit",
        "normalized_winner_hhi",
        "bottleneck_machine_utilization",
        "bottleneck_machine_slack",
        "route_capacity_feasible_providers_per_order",
        "at_least_three_route_capacity_feasible_provider_rate",
        "aggregate_feasible_but_route_infeasible_rate",
    }
    chart_rows: list[dict[str, Any]] = []
    for result in results:
        identity = _identity_row(result)
        audit = result["complete_table_audit"]
        dss = result["dss"]
        holdout = result["holdout_audit"]
        summary_rows.append(
            {
                **identity,
                "dss_support_size": dss["support_size"],
                "dss_candidate_profile_count": dss["candidate_profile_count"],
                "dss_profile_coverage": dss["profile_coverage"],
                "dss_provisional_gap": dss["provisional_gap"],
                "dss_provisional_relative_gap_percent": dss[
                    "provisional_relative_gap_percent"
                ],
                "complete_table_gap": audit["cce_gap_nominal"],
                "complete_table_relative_gap_percent": audit[
                    "relative_gap_percent"
                ],
                "complete_table_ucb_gap": audit["cce_gap_ucb"],
                "holdout_gap": holdout["gap"],
                "holdout_relative_gap_percent": holdout["relative_gap_percent"],
                "holdout_ucb_gap": holdout["ucb_gap"],
                "solver_runtime_seconds": dss["solver_runtime_seconds"],
                "posthoc_audit_runtime_seconds": audit["posthoc_audit_runtime_seconds"],
                **result["mechanism_metrics"],
            }
        )
        for metric, value in result["mechanism_metrics"].items():
            metric_row = {**identity, "metric": metric, "value": value}
            metric_rows.append(metric_row)
            if metric in chart_metrics:
                chart_rows.append(metric_row)
            if (
                metric.startswith("machine_group_")
                or metric.startswith("capacity_shortfall_rate_")
                or metric.startswith("outside_pressure_")
                or metric.startswith("assignment_rate_F")
                or "feasible_provider" in metric
                or "route_infeasible" in metric
                or metric.startswith("single_")
            ):
                resource_rows.append(metric_row)
        for support_item in dss["support"]:
            support_rows.append(
                {
                    **identity,
                    "profile": "|".join(support_item["profile"]),
                    "probability": support_item["probability"],
                }
            )
    _atomic_csv(tables / "cnc_game_summary.csv", pd.DataFrame(summary_rows))
    _atomic_json(tables / "cnc_game_summary.json", summary_rows)
    _atomic_csv(tables / "cnc_metrics_long.csv", pd.DataFrame(metric_rows))
    _atomic_json(tables / "cnc_metrics_long.json", metric_rows)
    _atomic_csv(tables / "cnc_resource_metrics_long.csv", pd.DataFrame(resource_rows))
    _atomic_csv(tables / "cnc_policy_support.csv", pd.DataFrame(support_rows))
    _atomic_csv(tables / "cnc_chart_data.csv", pd.DataFrame(chart_rows))


def _write_bridge_summary(output_dir: Path, records: Sequence[Mapping[str, Any]]) -> None:
    rows = [
        {
            "pressure_cell": item["pressure_cell"],
            "auction_treatment": item["auction_treatment"],
            "mechanism": item["mechanism"],
            "seed": item["seed"],
            "dss_complete_table_gap": item["dss_complete_table_gap"],
            "dss_complete_table_relative_gap_percent": item[
                "dss_complete_table_relative_gap_percent"
            ],
            "fullspace_gap": item["fullspace_solution"]["cce_gap_nominal"],
            "fullspace_relative_gap_percent": item[
                "fullspace_relative_gap_percent"
            ],
            "dss_objective": item["dss_complete_table_objective"],
            "fullspace_objective": item["fullspace_solution"]["objective_value"],
            "objective_difference_dss_minus_fullspace": item[
                "objective_difference_dss_minus_fullspace"
            ],
            "fullspace_lp_runtime_seconds": item["fullspace_lp_runtime_seconds"],
        }
        for item in records
    ]
    _atomic_csv(output_dir / "tables" / "cnc_bridge_validation.csv", pd.DataFrame(rows))
    _atomic_json(output_dir / "tables" / "cnc_bridge_validation.json", records)


def _write_transplant_summary(output_dir: Path, records: Sequence[Mapping[str, Any]]) -> None:
    rows = [
        {
            "pressure_cell": item["pressure_cell"],
            "seed": item["seed"],
            "source_treatment": item["source_treatment"],
            "target_treatment": item["target_treatment"],
            "transplanted_cce_gap": item["transplanted_cce_gap"],
            "transplanted_relative_gap_percent": item[
                "transplanted_relative_gap_percent"
            ],
            "transplanted_ucb_gap": item["transplanted_ucb_gap"],
            "target_reselected_cce_gap": item["target_reselected_cce_gap"],
            "target_reselected_relative_gap_percent": item[
                "target_reselected_relative_gap_percent"
            ],
        }
        for item in records
    ]
    _atomic_csv(output_dir / "tables" / "cnc_policy_transplants.csv", pd.DataFrame(rows))
    _atomic_json(output_dir / "tables" / "cnc_policy_transplants.json", records)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen route-aware CNC policy-game case study."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--stage",
        choices=("pilot", "game", "main", "bridge", "transplant", "sensitivity", "all", "summaries"),
        default="pilot",
    )
    parser.add_argument("--cell")
    parser.add_argument("--treatment")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict-calibration", action="store_true")
    parser.add_argument("--allow-unfrozen-config", action="store_true")
    parser.add_argument("--max-games", type=int)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    raw = load_case_config(config_path)
    digest = config_sha256(config_path)

    if args.max_games is not None and args.max_games <= 0:
        raise SystemExit("--max-games must be a positive integer.")
    if args.max_games is not None and args.stage not in {"main", "sensitivity"}:
        raise SystemExit("--max-games is supported only for --stage main or sensitivity.")

    if args.stage == "pilot":
        report = run_pilot(
            raw,
            config_hash=digest,
            output_dir=output_dir,
            strict=args.strict_calibration,
        )
        print(f"CNC pilot status: {report['status']}; config SHA256={digest}")
        return
    if args.stage == "summaries":
        refresh_summaries(output_dir, raw=raw)
        print("CNC summary tables refreshed.")
        return

    ensure_frozen_config(output_dir, digest, allow_unfrozen=args.allow_unfrozen_config)
    if args.stage == "game":
        if args.cell is None or args.treatment is None or args.seed is None:
            raise SystemExit("--stage game requires --cell, --treatment, and --seed.")
        auc, _ = normalize_treatment(raw, args.treatment)
        if args.cell not in factorial_cells(raw):
            raise SystemExit(f"--stage game cell is outside the frozen factorial: {args.cell}")
        if auc not in treatment_map(raw):
            raise SystemExit(f"--stage game treatment is outside AUC1--AUC4: {args.treatment}")
        if args.seed not in {int(value) for value in raw["main_seeds"]}:
            raise SystemExit(f"--stage game seed is not a reporting seed: {args.seed}")
        result = _run_one_game_recording_failure(
            raw,
            config_hash=digest,
            output_dir=output_dir,
            cell=args.cell,
            treatment=args.treatment,
            seed=args.seed,
            force=args.force,
        )
        refresh_summaries(output_dir, raw=raw)
        print(
            f"Completed {result['pressure_cell']} {result['auction_treatment']} seed "
            f"{result['seed']}; audited gap={result['complete_table_audit']['cce_gap_nominal']:.6g}."
        )
        return
    if args.stage == "main":
        records = run_main_matrix(
            raw,
            config_hash=digest,
            output_dir=output_dir,
            force=args.force,
            max_games=args.max_games,
        )
        print(f"Completed or resumed {len(records)} CNC main games.")
        return
    if args.stage == "bridge":
        records = run_bridge_validation(
            raw, config_hash=digest, output_dir=output_dir, force=args.force
        )
        print(f"Completed or resumed {len(records)} FullSpace bridge LPs.")
        return
    if args.stage == "transplant":
        records = run_policy_transplants(
            raw, config_hash=digest, output_dir=output_dir, force=args.force
        )
        print(f"Completed or resumed {len(records)} policy transplants.")
        return
    if args.stage == "sensitivity":
        records = run_sensitivity_matrix(
            raw,
            config_hash=digest,
            output_dir=output_dir,
            force=args.force,
            max_games=args.max_games,
        )
        print(f"Completed or resumed {len(records)} CNC sensitivity games.")
        return
    if args.stage == "all":
        # Pilot precedes all formal games and freezes the exact YAML digest.
        run_pilot(raw, config_hash=digest, output_dir=output_dir, strict=True)
        run_main_matrix(raw, config_hash=digest, output_dir=output_dir, force=args.force)
        run_bridge_validation(raw, config_hash=digest, output_dir=output_dir, force=args.force)
        run_policy_transplants(raw, config_hash=digest, output_dir=output_dir, force=args.force)
        run_sensitivity_matrix(raw, config_hash=digest, output_dir=output_dir, force=args.force)
        refresh_summaries(output_dir, raw=raw)
        print("Completed the full frozen CNC case-study workflow.")
        return
    raise AssertionError(f"Unhandled stage: {args.stage}")


if __name__ == "__main__":
    main()
