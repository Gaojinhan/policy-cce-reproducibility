from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from cmfg_cce.envs.toy import ToyEnvConfig, config_from_mapping
from cmfg_cce.evaluation.independent_audit import (
    FrozenDistribution,
    build_joint_audit_samples,
    freeze_distribution,
    summarize_audit_samples,
)
from cmfg_cce.evaluation.payoff_cache import support_deviation_closure
from cmfg_cce.evaluation.rollout import Profile, RolloutSeeds, mechanism_from_id, run_episode
from cmfg_cce.experiments.common import solver_seed
from cmfg_cce.policies.base import BiddingPolicy
from cmfg_cce.policies.policy_library import build_policy_library_for_mechanism


ROOT = Path(__file__).resolve().parents[2]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _stable_seed(*values: object) -> int:
    payload = "|".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _seed_dict(seeds: RolloutSeeds) -> dict[str, int]:
    return {
        "type_seed": int(seeds.type_seed),
        "order_seed": int(seeds.order_seed),
        "outside_seed": int(seeds.outside_seed),
        "availability_seed": int(seeds.availability_seed),
        "tie_break_seed": int(seeds.tie_break_seed),
        "rollout_replication_seed": int(seeds.rollout_replication_seed),
    }


def audit_seeds_for_game(seed: int, offset: int) -> tuple[RolloutSeeds, RolloutSeeds]:
    training = solver_seed(int(seed))
    audit = RolloutSeeds(
        # Manufacturer types define the empirical-game instance and stay fixed.
        type_seed=training.type_seed,
        order_seed=training.order_seed + int(offset),
        outside_seed=training.outside_seed + int(offset),
        availability_seed=training.availability_seed + int(offset),
        tie_break_seed=training.tie_break_seed + int(offset),
        rollout_replication_seed=training.rollout_replication_seed + int(offset),
    )
    validate_seed_isolation(training, audit)
    return training, audit


def validate_seed_isolation(training: RolloutSeeds, audit: RolloutSeeds) -> None:
    if int(training.type_seed) != int(audit.type_seed):
        raise ValueError("type_seed must remain fixed so the audit uses the same manufacturer population.")
    dynamic_fields = (
        "order_seed",
        "outside_seed",
        "availability_seed",
        "tie_break_seed",
        "rollout_replication_seed",
    )
    training_values = {int(getattr(training, field)) for field in dynamic_fields}
    audit_values = {int(getattr(audit, field)) for field in dynamic_fields}
    if training_values.intersection(audit_values):
        raise ValueError("Training and audit dynamic random streams overlap.")
    if any(int(getattr(training, field)) == int(getattr(audit, field)) for field in dynamic_fields):
        raise ValueError("Every dynamic audit stream must differ from its training stream.")


def _profile_returns_worker(args: tuple[Any, ...]) -> tuple[Profile, np.ndarray]:
    (
        profile,
        mechanism_id,
        policies,
        env_config,
        n_agents,
        seeds,
        replication_start,
        replication_stop,
        markup_grid,
        lead_time_grid,
    ) = args
    mechanism = mechanism_from_id(mechanism_id, env_config)
    values = np.empty(
        (int(replication_stop) - int(replication_start), int(n_agents)), dtype=float
    )
    for row, replication in enumerate(range(int(replication_start), int(replication_stop))):
        returns, _ = run_episode(
            profile=profile,
            mechanism=mechanism,
            policies=policies,
            config=env_config,
            n_agents=int(n_agents),
            seeds=seeds,
            replication=replication,
            markup_grid=markup_grid,
            lead_time_grid=lead_time_grid,
        )
        values[row] = returns
    return profile, values


def evaluate_profile_returns(
    profiles: Iterable[Profile],
    mechanism_id: str,
    policies: dict[str, BiddingPolicy],
    env_config: ToyEnvConfig,
    n_agents: int,
    seeds: RolloutSeeds,
    rollouts: int,
    markup_grid: tuple[float, ...],
    lead_time_grid: tuple[float, ...],
    workers: int,
    replication_start: int = 0,
) -> dict[Profile, np.ndarray]:
    replication_stop = int(replication_start) + int(rollouts)
    if int(replication_start) < 0 or int(rollouts) <= 0:
        raise ValueError("replication_start must be nonnegative and rollouts must be positive.")
    ordered = sorted(set(tuple(profile) for profile in profiles))
    args = [
        (
            profile,
            mechanism_id,
            policies,
            env_config,
            int(n_agents),
            seeds,
            int(replication_start),
            replication_stop,
            markup_grid,
            lead_time_grid,
        )
        for profile in ordered
    ]
    worker_count = max(1, min(int(workers), len(args)))
    if worker_count == 1:
        return dict(_profile_returns_worker(item) for item in args)
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=mp.get_context("fork"),
    ) as executor:
        return dict(executor.map(_profile_returns_worker, args, chunksize=1))


def _profile_returns_prefix_hash(
    profile_returns: dict[Profile, np.ndarray],
    prefix_rollouts: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(int(prefix_rollouts)).encode("utf-8"))
    for profile in sorted(profile_returns):
        values = np.asarray(profile_returns[profile], dtype=np.float64)
        if values.shape[0] < int(prefix_rollouts):
            raise ValueError(f"Profile {profile} lacks the requested prefix.")
        digest.update("|".join(profile).encode("utf-8"))
        digest.update(np.ascontiguousarray(values[: int(prefix_rollouts)]).tobytes())
    return digest.hexdigest()


def _distribution_from_record(
    record: dict[str, Any],
    n_agents: int,
    policy_ids: tuple[str, ...],
) -> FrozenDistribution:
    support: list[Profile] = []
    probabilities: list[float] = []
    for item in record.get("support", []):
        probability = float(item["prob"])
        if probability <= 1.0e-12:
            continue
        support.append(tuple(str(value) for value in item["profile"]))
        probabilities.append(probability)
    distribution = freeze_distribution(
        solver=str(record["solver"]),
        support=support,
        probabilities=probabilities,
        n_agents=int(n_agents),
        policy_ids=policy_ids,
    )
    if int(record.get("support_size", len(support))) != len(distribution.support):
        raise ValueError(f"Recorded support size does not match positive support for {record['solver']}.")
    return distribution


def _load_case_sources(
    source_root: Path,
    case: dict[str, Any],
    solver_names: list[str],
) -> tuple[dict[str, Any], Path, Path, dict[str, dict[str, Any]]]:
    n, j, seed = int(case["N"]), int(case["J"]), int(case["seed"])
    mechanism = str(case["mechanism"])
    config_path = source_root / "configs" / f"n{n}k{j}.yaml"
    results_path = source_root / f"n{n}k{j}" / "raw" / "cce_results_oracle_baselines.json"
    if not config_path.exists() or not results_path.exists():
        raise FileNotFoundError(f"Missing frozen source files for N={n}, J={j}: {config_path}, {results_path}")
    source_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    matrix = source_config.get("oracle_baselines", {})
    if list(matrix.get("N_values", [])) != [n] or list(matrix.get("policies_per_agent", [])) != [j]:
        raise ValueError(f"Frozen config does not match requested N={n}, J={j}.")
    if seed not in [int(value) for value in matrix.get("seeds", [])]:
        raise ValueError(f"Seed {seed} is absent from frozen config {config_path}.")
    if mechanism not in source_config.get("mechanisms", []):
        raise ValueError(f"Mechanism {mechanism} is absent from frozen config {config_path}.")
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    selected: dict[str, dict[str, Any]] = {}
    for solver_name in solver_names:
        matches = [
            row
            for row in rows
            if row.get("solver") == solver_name
            and int(row.get("N", -1)) == n
            and int(row.get("K", -1)) == j
            and int(row.get("seed", -1)) == seed
            and row.get("mechanism") == mechanism
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one frozen record for {(n, j, seed, mechanism, solver_name)}, found {len(matches)}."
            )
        selected[solver_name] = matches[0]
    return source_config, config_path, results_path, selected


def _profile_returns_frame(
    profile_returns: dict[Profile, np.ndarray],
    n_agents: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for profile in sorted(profile_returns):
        values = profile_returns[profile]
        for replication in range(values.shape[0]):
            row: dict[str, Any] = {
                "profile": "|".join(profile),
                "replication": int(replication),
            }
            for agent in range(int(n_agents)):
                row[f"return_agent_{agent}"] = float(values[replication, agent])
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def _profile_returns_from_frame(frame: pd.DataFrame, n_agents: int) -> dict[Profile, np.ndarray]:
    returns: dict[Profile, np.ndarray] = {}
    columns = [f"return_agent_{agent}" for agent in range(int(n_agents))]
    for profile_text, group in frame.groupby("profile", sort=True):
        ordered = group.sort_values("replication")
        expected = np.arange(len(ordered), dtype=int)
        if not np.array_equal(ordered["replication"].to_numpy(dtype=int), expected):
            raise ValueError(f"Non-contiguous replication indices for {profile_text}.")
        returns[tuple(str(profile_text).split("|"))] = ordered[columns].to_numpy(dtype=float)
    return returns


def _case_key(case: dict[str, Any]) -> str:
    return f"n{int(case['N'])}j{int(case['J'])}_s{int(case['seed'])}_{case['mechanism']}"


def _training_relative_gap(record: dict[str, Any], n_agents: int) -> float:
    metrics = record.get("mechanism_metrics")
    if not isinstance(metrics, dict):
        metrics = record
    # Reproduce the denominator used by the submitted Table 4 generator.  This
    # field is the sum of the agents' discounted episode returns; the similarly
    # named profit_manufacturer_i fields are undiscounted reporting metrics and
    # must not be substituted here.
    total_profit = metrics.get("manufacturer_total_profit")
    if total_profit is None:
        raise ValueError(
            "Frozen solver record lacks the manufacturer profits required by Eq. (11)."
        )
    denominator = max(1.0, abs(float(total_profit)) / int(n_agents))
    if not np.isfinite(denominator):
        raise ValueError("Frozen solver record has a non-finite Eq. (11) denominator.")
    return 100.0 * max(0.0, float(record["cce_gap_full_audit"])) / denominator


def run_case(
    source_root: Path,
    output_root: Path,
    case: dict[str, Any],
    solver_names: list[str],
    audit_cfg: dict[str, Any],
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    n, j, seed = int(case["N"]), int(case["J"]), int(case["seed"])
    mechanism = str(case["mechanism"])
    source_config, config_path, results_path, records = _load_case_sources(
        source_root, case, solver_names
    )
    env_config = config_from_mapping(source_config)
    if int(env_config.horizon) != int(source_config["oracle_baselines"]["horizon"]):
        raise ValueError("Environment horizon does not match the frozen oracle config.")
    policies = build_policy_library_for_mechanism(mechanism, j)
    policy_ids = tuple(policies.keys())
    distributions = {
        solver: _distribution_from_record(record, n, policy_ids)
        for solver, record in records.items()
    }
    frozen_hashes_before = {solver: value.q_hash for solver, value in distributions.items()}
    union_closure: set[Profile] = set()
    for distribution in distributions.values():
        union_closure.update(support_deviation_closure(distribution.support, policy_ids))

    max_rollouts = max(int(value) for value in audit_cfg["nested_rollouts"])
    training_seeds, audit_seeds = audit_seeds_for_game(seed, int(audit_cfg["seed_offset"]))
    source_hashes = {
        "config_sha256": _sha256_file(config_path),
        "results_sha256": _sha256_file(results_path),
    }
    source_identity = {
        "case": case,
        "source_hashes": source_hashes,
        "q_hashes": frozen_hashes_before,
        "audit_seeds": _seed_dict(audit_seeds),
        "policy_ids": list(policy_ids),
        "union_closure": [list(profile) for profile in sorted(union_closure)],
    }
    source_bundle_hash = _sha256_bytes(
        json.dumps(
            source_identity,
            sort_keys=True,
        ).encode("utf-8")
    )
    case_dir = output_root / "games" / _case_key(case)
    returns_path = case_dir / "profile_returns.parquet"
    metadata_path = case_dir / "metadata.json"
    profile_stage_start = time.perf_counter()
    resumed = False
    checkpoint_load_runtime = 0.0
    incremental_profile_evaluation_runtime = 0.0
    prefix_rollouts_before = 0
    prefix_hash_before: str | None = None
    if returns_path.exists() and metadata_path.exists() and not force:
        old_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        compatible = bool(
            old_metadata.get("source_bundle_hash") == source_bundle_hash
            or (
                old_metadata.get("case") == case
                and old_metadata.get("source_hashes") == source_hashes
                and old_metadata.get("audit_seeds") == _seed_dict(audit_seeds)
                and old_metadata.get("q_hashes_before") == frozen_hashes_before
                and old_metadata.get("policy_ids") == list(policy_ids)
                and old_metadata.get("union_closure")
                == [list(profile) for profile in sorted(union_closure)]
            )
        )
        if not compatible:
            raise ValueError(
                f"Existing checkpoint for {_case_key(case)} does not match frozen sources or audit settings."
            )
        profile_returns = _profile_returns_from_frame(pd.read_parquet(returns_path), n)
        if set(profile_returns) != union_closure:
            raise ValueError(f"Existing checkpoint closure does not match {_case_key(case)}.")
        rollout_counts = {values.shape[0] for values in profile_returns.values()}
        if len(rollout_counts) != 1:
            raise ValueError(f"Checkpoint profiles have inconsistent replication counts for {_case_key(case)}.")
        prefix_rollouts_before = int(next(iter(rollout_counts)))
        prefix_hash_before = _profile_returns_prefix_hash(
            profile_returns, prefix_rollouts_before
        )
        resumed = True
        checkpoint_load_runtime = time.perf_counter() - profile_stage_start
        profile_evaluation_runtime = float(
            old_metadata.get(
                "profile_evaluation_runtime_seconds",
                old_metadata.get("evaluation_runtime_seconds", 0.0),
            )
        )
        if prefix_rollouts_before < max_rollouts:
            extension_start = time.perf_counter()
            additional = evaluate_profile_returns(
                profiles=union_closure,
                mechanism_id=mechanism,
                policies=policies,
                env_config=env_config,
                n_agents=n,
                seeds=audit_seeds,
                rollouts=max_rollouts - prefix_rollouts_before,
                markup_grid=tuple(float(value) for value in source_config["policies"]["markup_grid"]),
                lead_time_grid=tuple(
                    float(value) for value in source_config["policies"]["lead_time_grid"]
                ),
                workers=int(audit_cfg.get("workers", 1)),
                replication_start=prefix_rollouts_before,
            )
            for profile in sorted(union_closure):
                profile_returns[profile] = np.vstack(
                    [profile_returns[profile], additional[profile]]
                )
            incremental_profile_evaluation_runtime = time.perf_counter() - extension_start
            profile_evaluation_runtime += incremental_profile_evaluation_runtime
            _profile_returns_frame(profile_returns, n).to_parquet(returns_path, index=False)
    else:
        markup_grid = tuple(float(value) for value in source_config["policies"]["markup_grid"])
        lead_time_grid = tuple(float(value) for value in source_config["policies"]["lead_time_grid"])
        profile_returns = evaluate_profile_returns(
            profiles=union_closure,
            mechanism_id=mechanism,
            policies=policies,
            env_config=env_config,
            n_agents=n,
            seeds=audit_seeds,
            rollouts=max_rollouts,
            markup_grid=markup_grid,
            lead_time_grid=lead_time_grid,
            workers=int(audit_cfg.get("workers", 1)),
        )
        case_dir.mkdir(parents=True, exist_ok=True)
        _profile_returns_frame(profile_returns, n).to_parquet(returns_path, index=False)
        profile_evaluation_runtime = time.perf_counter() - profile_stage_start

    available_rollouts = min(values.shape[0] for values in profile_returns.values())
    if available_rollouts < max_rollouts:
        raise RuntimeError(
            f"Checkpoint contains {available_rollouts} replications but {max_rollouts} are required."
        )
    prefix_hash_after = (
        _profile_returns_prefix_hash(profile_returns, prefix_rollouts_before)
        if prefix_rollouts_before > 0
        else None
    )
    if prefix_hash_before != prefix_hash_after:
        raise RuntimeError("An existing audit replication prefix changed during checkpoint extension.")

    statistics_start = time.perf_counter()
    summary_rows: list[dict[str, Any]] = []
    gain_rows: list[dict[str, Any]] = []
    for solver_name, distribution in distributions.items():
        max_samples = build_joint_audit_samples(
            profile_returns,
            distribution,
            policy_ids,
            n_agents=n,
            sample_count=max_rollouts,
        )
        for col_idx, (agent, dev_policy) in enumerate(max_samples.labels):
            for replication in range(max_rollouts):
                gain_rows.append(
                    {
                        "case_key": _case_key(case),
                        "N": n,
                        "J": j,
                        "seed": seed,
                        "mechanism": mechanism,
                        "solver": solver_name,
                        "q_hash": distribution.q_hash,
                        "replication": replication,
                        "agent": int(agent),
                        "dev_policy": str(dev_policy),
                        "gain": float(max_samples.gain_samples[replication, col_idx]),
                    }
                )
        for sample_count in [int(value) for value in audit_cfg["nested_rollouts"]]:
            samples = build_joint_audit_samples(
                profile_returns,
                distribution,
                policy_ids,
                n_agents=n,
                sample_count=sample_count,
            )
            summary = summarize_audit_samples(
                samples,
                alpha=float(audit_cfg["alpha"]),
                bootstrap_samples=int(audit_cfg["bootstrap_samples"]),
                bootstrap_seed=_stable_seed(
                    _case_key(case), sample_count, "joint_bootstrap"
                ),
            )
            record = records[solver_name]
            summary_rows.append(
                {
                    "case_key": _case_key(case),
                    "stratum": str(case["stratum"]),
                    "N": n,
                    "J": j,
                    "seed": seed,
                    "mechanism": mechanism,
                    "solver": solver_name,
                    "q_hash": distribution.q_hash,
                    "support_size": len(distribution.support),
                    "union_closure_size": len(union_closure),
                    "solver_closure_size": len(
                        support_deviation_closure(distribution.support, policy_ids)
                    ),
                    "training_complete_table_gap": float(record["cce_gap_full_audit"]),
                    "training_relative_gap_percent": _training_relative_gap(record, n),
                    "evaluation_runtime_seconds": profile_evaluation_runtime,
                    "resumed_from_checkpoint": resumed,
                    **summary,
                }
            )

    frozen_hashes_after = {
        solver: distribution.q_hash for solver, distribution in distributions.items()
    }
    if frozen_hashes_before != frozen_hashes_after:
        raise RuntimeError("A frozen distribution changed during the audit.")
    statistics_runtime = time.perf_counter() - statistics_start
    metadata = {
        "bundle_hash": source_bundle_hash,
        "source_bundle_hash": source_bundle_hash,
        "case": case,
        "source_config": str(config_path.relative_to(ROOT)),
        "source_results": str(results_path.relative_to(ROOT)),
        "source_hashes": source_hashes,
        "training_seeds": _seed_dict(training_seeds),
        "audit_seeds": _seed_dict(audit_seeds),
        "type_seed_fixed_for_same_population": True,
        "dynamic_streams_are_independent": True,
        "q_hashes_before": frozen_hashes_before,
        "q_hashes_after": frozen_hashes_after,
        "policy_ids": list(policy_ids),
        "union_closure_size": len(union_closure),
        "union_closure": [list(profile) for profile in sorted(union_closure)],
        "max_audit_rollouts": int(available_rollouts),
        "requested_audit_rollouts": max_rollouts,
        "nested_rollouts": [int(value) for value in audit_cfg["nested_rollouts"]],
        "evaluation_runtime_seconds": profile_evaluation_runtime,
        "profile_evaluation_runtime_seconds": profile_evaluation_runtime,
        "incremental_profile_evaluation_runtime_seconds": incremental_profile_evaluation_runtime,
        "checkpoint_load_runtime_seconds": checkpoint_load_runtime,
        "statistics_runtime_seconds": statistics_runtime,
        "prefix_rollouts_before": prefix_rollouts_before,
        "prefix_hash_before": prefix_hash_before,
        "prefix_hash_after": prefix_hash_after,
        "resumed_from_checkpoint": resumed,
    }
    _json_dump(metadata_path, metadata)
    summary_frame = pd.DataFrame.from_records(summary_rows)
    gain_frame = pd.DataFrame.from_records(gain_rows)
    summary_frame.to_csv(case_dir / "summary.csv", index=False)
    gain_frame.to_parquet(case_dir / "gain_samples.parquet", index=False)
    return summary_frame, gain_frame, metadata


def _write_diagnostics(
    summary: pd.DataFrame,
    output_root: Path,
    audit_cfg: dict[str, Any],
) -> None:
    tables = output_root / "tables"
    figures = output_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables / "pilot_summary.csv", index=False)
    max_r = int(summary["audit_rollouts"].max())
    final = summary[summary["audit_rollouts"].eq(max_r)].copy()
    final.to_csv(tables / "training_vs_fresh_audit.csv", index=False)

    comparison = final.pivot_table(
        index=["case_key", "N", "J", "seed", "mechanism", "stratum"],
        columns="solver",
        values=[
            "relative_nominal_gap_percent",
            "max_t_relative_gap_ucb95_percent",
        ],
    )
    comparison.columns = [f"{metric}__{solver}" for metric, solver in comparison.columns]
    comparison = comparison.reset_index()
    dss_col = "relative_nominal_gap_percent__REPAIR-SAD-CCE"
    full_col = "relative_nominal_gap_percent__FullTensor-CCE-LP"
    comparison["dss_minus_fullspace_nominal_gap_percent"] = comparison[dss_col] - comparison[full_col]
    comparison.to_csv(tables / "dss_vs_fullspace.csv", index=False)

    convergence = summary.pivot_table(
        index=["case_key", "solver"],
        columns="audit_rollouts",
        values="relative_nominal_gap_percent",
    ).reset_index()
    if 100 in convergence.columns and 200 in convergence.columns:
        convergence["absolute_change_100_to_200_percent"] = (
            convergence[200] - convergence[100]
        ).abs()
    convergence.to_csv(tables / "rollout_convergence.csv", index=False)

    low = float(audit_cfg["low_gap_percent"])
    width = float(audit_cfg["interval_width_percent"])
    excess = float(audit_cfg["dss_excess_percent"])
    change = float(audit_cfg["convergence_change_percent"])
    flags = {
        "pilot_only_not_for_publication": True,
        "max_audit_rollouts": max_r,
        "all_dss_nominal_at_or_below_low_gap": bool(
            final[final["solver"].eq("REPAIR-SAD-CCE")]["relative_nominal_gap_percent"].le(low).all()
        ),
        "all_fullspace_nominal_at_or_below_low_gap": bool(
            final[final["solver"].eq("FullTensor-CCE-LP")]["relative_nominal_gap_percent"].le(low).all()
        ),
        "any_dss_max_t_interval_wider_than_threshold": bool(
            (
                final[final["solver"].eq("REPAIR-SAD-CCE")]["max_t_relative_gap_ucb95_percent"]
                - final[final["solver"].eq("REPAIR-SAD-CCE")]["max_t_relative_gap_lcb95_percent"]
            ).gt(width).any()
        ),
        "any_dss_excess_over_fullspace": bool(
            comparison["dss_minus_fullspace_nominal_gap_percent"].gt(excess).any()
        ),
        "any_large_100_to_200_change": bool(
            convergence.get("absolute_change_100_to_200_percent", pd.Series(dtype=float)).gt(change).any()
        ),
        "thresholds": {
            "low_gap_percent": low,
            "interval_width_percent": width,
            "dss_excess_percent": excess,
            "convergence_change_percent": change,
        },
    }
    _json_dump(tables / "diagnostic_flags.json", flags)

    solver_colors = {
        "REPAIR-SAD-CCE": "#1b9e77",
        "FullTensor-CCE-LP": "#333333",
    }
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for solver, group in final.groupby("solver"):
        ax.scatter(
            group["training_relative_gap_percent"],
            group["relative_nominal_gap_percent"],
            label=solver,
            color=solver_colors.get(solver),
            alpha=0.85,
        )
    upper = max(
        1.0,
        float(final["training_relative_gap_percent"].max()),
        float(final["relative_nominal_gap_percent"].max()),
    )
    ax.plot([0, upper], [0, upper], color="0.6", linestyle="--", linewidth=1)
    ax.set_xlabel("Training complete-table relative gap (%)")
    ax.set_ylabel("Fresh-audit relative nominal gap (%)")
    ax.legend(frameon=False)
    ax.grid(color="0.9", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(figures / "training_vs_fresh_gap.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for solver, group in summary.groupby("solver"):
        aggregated = group.groupby("audit_rollouts")[
            ["relative_nominal_gap_percent", "max_t_relative_gap_ucb95_percent"]
        ].mean()
        color = solver_colors.get(solver)
        ax.plot(
            aggregated.index,
            aggregated["relative_nominal_gap_percent"],
            marker="o",
            color=color,
            label=f"{solver}: nominal",
        )
        ax.plot(
            aggregated.index,
            aggregated["max_t_relative_gap_ucb95_percent"],
            marker="s",
            linestyle="--",
            color=color,
            label=f"{solver}: max-t UCB",
        )
    ax.set_xlabel("Independent audit rollouts")
    ax.set_ylabel("Relative gap (%)")
    ax.set_xticks(sorted(summary["audit_rollouts"].unique()))
    ax.grid(color="0.9", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "rollout_convergence.pdf")
    plt.close(fig)


def _adaptive_extension_reasons(
    summary: pd.DataFrame,
    audit_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    adaptive = dict(audit_cfg.get("adaptive_extension", {}))
    if not adaptive.get("enabled", False):
        return []
    from_r = int(adaptive.get("convergence_from", 800))
    to_r = int(adaptive.get("convergence_to", 1000))
    change_threshold = float(adaptive.get("change_threshold_percent", 0.5))
    width_threshold = float(adaptive.get("interval_width_threshold_percent", 2.0))
    available = set(int(value) for value in summary["audit_rollouts"].unique())
    if not {from_r, to_r}.issubset(available):
        raise ValueError(
            f"Adaptive extension requires nested samples {from_r} and {to_r}."
        )
    reasons: list[dict[str, Any]] = []
    for (case_key, solver), group in summary.groupby(["case_key", "solver"]):
        indexed = group.set_index("audit_rollouts")
        change = abs(
            float(indexed.loc[to_r, "relative_nominal_gap_percent"])
            - float(indexed.loc[from_r, "relative_nominal_gap_percent"])
        )
        width = float(indexed.loc[to_r, "max_t_relative_gap_ucb95_percent"]) - float(
            indexed.loc[to_r, "max_t_relative_gap_lcb95_percent"]
        )
        if change > change_threshold or width > width_threshold:
            reasons.append(
                {
                    "case_key": str(case_key),
                    "solver": str(solver),
                    "absolute_nominal_change_percent": change,
                    "max_t_interval_width_percent": width,
                    "change_triggered": bool(change > change_threshold),
                    "width_triggered": bool(width > width_threshold),
                }
            )
    return reasons


def run_pilot(config_path: Path, force: bool = False, case_indices: set[int] | None = None) -> pd.DataFrame:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_root = ROOT / str(config["source_root"])
    output_root = ROOT / str(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    solver_names = [str(value) for value in config["solvers"]]
    if solver_names != ["REPAIR-SAD-CCE", "FullTensor-CCE-LP"]:
        raise ValueError("Pilot solver contract changed unexpectedly.")
    cases = list(config["cases"])
    if len(cases) != 12:
        raise ValueError(f"Reviewer 1 pilot must contain exactly 12 cases, found {len(cases)}.")
    if len({_case_key(case) for case in cases}) != len(cases):
        raise ValueError("Reviewer 1 pilot contains duplicate case keys.")
    all_summary: list[pd.DataFrame] = []
    for index, case in enumerate(cases):
        if case_indices is not None and index not in case_indices:
            continue
        print(f"[{index + 1}/{len(cases)}] auditing {_case_key(case)}", flush=True)
        summary, _, _ = run_case(
            source_root=source_root,
            output_root=output_root,
            case=case,
            solver_names=solver_names,
            audit_cfg=config["audit"],
            force=force,
        )
        all_summary.append(summary)
    if not all_summary:
        raise ValueError("No pilot cases were selected.")
    combined = pd.concat(all_summary, ignore_index=True)
    selected_all = case_indices is None or case_indices == set(range(len(cases)))
    if selected_all:
        extension_reasons = _adaptive_extension_reasons(combined, config["audit"])
        adaptive = dict(config["audit"].get("adaptive_extension", {}))
        extension_triggered = bool(extension_reasons)
        if extension_triggered:
            extended_cfg = dict(config["audit"])
            extended_cfg["nested_rollouts"] = list(
                dict.fromkeys(
                    [int(value) for value in config["audit"]["nested_rollouts"]]
                    + [int(value) for value in adaptive.get("extension_rollouts", [1500, 2000])]
                )
            )
            print(
                f"Adaptive gate triggered by {len(extension_reasons)} game/solver rows; "
                f"extending all cases to {max(extended_cfg['nested_rollouts'])} rollouts.",
                flush=True,
            )
            extended_summaries: list[pd.DataFrame] = []
            for index, case in enumerate(cases):
                print(
                    f"[extension {index + 1}/{len(cases)}] auditing {_case_key(case)}",
                    flush=True,
                )
                summary, _, _ = run_case(
                    source_root=source_root,
                    output_root=output_root,
                    case=case,
                    solver_names=solver_names,
                    audit_cfg=extended_cfg,
                    force=False,
                )
                extended_summaries.append(summary)
            combined = pd.concat(extended_summaries, ignore_index=True)
        else:
            extended_cfg = config["audit"]
        expected = len(cases) * len(solver_names) * len(extended_cfg["nested_rollouts"])
        if len(combined) != expected:
            raise RuntimeError(f"Expected {expected} pilot summaries, found {len(combined)}.")
        _write_diagnostics(combined, output_root, config["audit"])
        _json_dump(
            output_root / "tables" / "adaptive_extension.json",
            {
                "triggered": extension_triggered,
                "initial_max_rollouts": max(config["audit"]["nested_rollouts"]),
                "final_max_rollouts": int(combined["audit_rollouts"].max()),
                "reasons": extension_reasons,
                "thresholds": adaptive,
            },
        )
        _json_dump(
            output_root / "pilot_manifest.json",
            {
                "experiment": config["experiment"],
                "config": str(config_path.relative_to(ROOT)),
                "config_sha256": _sha256_file(config_path),
                "case_count": len(cases),
                "solver_count": len(solver_names),
                "nested_rollouts": config["audit"]["nested_rollouts"],
                "final_nested_rollouts": sorted(
                    int(value) for value in combined["audit_rollouts"].unique()
                ),
                "adaptive_extension_triggered": extension_triggered,
                "pilot_only_not_for_publication": True,
                "manuscript_modified": False,
            },
        )
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Reviewer 1 independent CCE audit pilot.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "cmfg_cce/configs/reviewer1_cce_audit_pilot.yaml",
    )
    parser.add_argument(
        "--case-indices",
        type=str,
        default=None,
        help="Optional comma-separated zero-based case indices for smoke or chunked execution.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute existing per-game checkpoints.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = None
    if args.case_indices:
        selected = {int(value.strip()) for value in args.case_indices.split(",") if value.strip()}
    run_pilot(args.config.resolve(), force=bool(args.force), case_indices=selected)


if __name__ == "__main__":
    main()
