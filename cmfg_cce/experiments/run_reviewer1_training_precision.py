from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

from cmfg_cce.envs.toy import config_from_mapping
from cmfg_cce.evaluation.empirical_game import EmpiricalGame
from cmfg_cce.evaluation.full_audit import audit_distribution_on_full_game
from cmfg_cce.evaluation.independent_audit import (
    FrozenDistribution,
    build_joint_audit_samples,
    freeze_distribution,
    summarize_audit_samples,
)
from cmfg_cce.evaluation.payoff_cache import PayoffCache, profile_space, support_deviation_closure
from cmfg_cce.evaluation.rollout import (
    Profile,
    RolloutEstimate,
    mechanism_from_id,
    run_episode,
)
from cmfg_cce.experiments.common import run_solver, solver_seed
from cmfg_cce.experiments.run_reviewer1_cce_audit import (
    _profile_returns_frame,
    _profile_returns_from_frame,
    _stable_seed,
    audit_seeds_for_game,
    evaluate_profile_returns,
)
from cmfg_cce.policies.policy_library import build_policy_library_for_mechanism
from cmfg_cce.solvers.cce_lp import solve_full_cce_lp


ROOT = Path(__file__).resolve().parents[2]
FULL_SOLVER = "FullTensor-CCE-LP"
DSS_SOLVER = "REPAIR-SAD-CCE"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _case_key(case: dict[str, Any]) -> str:
    return f"n{int(case['N'])}j{int(case['J'])}_s{int(case['seed'])}_{case['mechanism']}"


def _seed_dict(seeds: Any) -> dict[str, int]:
    return {
        "type_seed": int(seeds.type_seed),
        "order_seed": int(seeds.order_seed),
        "outside_seed": int(seeds.outside_seed),
        "availability_seed": int(seeds.availability_seed),
        "tie_break_seed": int(seeds.tie_break_seed),
        "rollout_replication_seed": int(seeds.rollout_replication_seed),
    }


def _training_worker(args: tuple[Any, ...]) -> tuple[int, np.ndarray, np.ndarray]:
    (
        profile_index,
        profile,
        mechanism_id,
        policies,
        env_config,
        n_agents,
        seeds,
        max_rollouts,
        markup_grid,
        lead_time_grid,
    ) = args
    mechanism = mechanism_from_id(mechanism_id, env_config)
    returns = np.empty((int(max_rollouts), int(n_agents)), dtype=np.float64)
    objectives = np.empty(int(max_rollouts), dtype=np.float64)
    for replication in range(int(max_rollouts)):
        episode_returns, metrics = run_episode(
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
        returns[replication] = episode_returns
        objectives[replication] = float(metrics["platform_operating_score"])
    return int(profile_index), returns, objectives


def _load_sources(
    source_root: Path,
    case: dict[str, Any],
) -> tuple[dict[str, Any], Path, Path, dict[str, dict[str, Any]]]:
    n, j, seed = int(case["N"]), int(case["J"]), int(case["seed"])
    mechanism = str(case["mechanism"])
    config_path = source_root / "configs" / f"n{n}k{j}.yaml"
    results_path = source_root / f"n{n}k{j}" / "raw" / "cce_results_oracle_baselines.json"
    if not config_path.exists() or not results_path.exists():
        raise FileNotFoundError(f"Missing source files for {_case_key(case)}.")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    matrix = raw["oracle_baselines"]
    if n not in matrix["N_values"] or j not in matrix["policies_per_agent"]:
        raise ValueError("Source configuration does not contain the requested scale.")
    if seed not in matrix["seeds"] or mechanism not in raw["mechanisms"]:
        raise ValueError("Source configuration does not contain the requested game.")
    records = json.loads(results_path.read_text(encoding="utf-8"))
    selected: dict[str, dict[str, Any]] = {}
    for solver in (FULL_SOLVER, DSS_SOLVER):
        matches = [
            row
            for row in records
            if row.get("solver") == solver
            and int(row.get("N", -1)) == n
            and int(row.get("K", -1)) == j
            and int(row.get("seed", -1)) == seed
            and row.get("mechanism") == mechanism
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one source record for {solver}; found {len(matches)}.")
        selected[solver] = matches[0]
    return raw, config_path, results_path, selected


def _source_dss_seed(raw: dict[str, Any], case: dict[str, Any]) -> int:
    matrix = raw["oracle_baselines"]
    skipped = {
        "FullTensor-CCE-LP",
        "FullTensor-MinGap-CCE-LP",
        "FullTensor-MinUCB-CCE-LP",
    }
    effective_solvers = [str(value) for value in matrix["solvers"] if value not in skipped]
    if DSS_SOLVER not in effective_solvers:
        raise ValueError("DSS solver is missing from the source configuration.")
    target = (
        int(case["N"]),
        int(case["J"]),
        int(case["seed"]),
        str(case["mechanism"]),
    )
    games: list[tuple[int, int, int, str]] = []
    for n in matrix["N_values"]:
        for j in matrix["policies_per_agent"]:
            for seed in matrix["seeds"]:
                for mechanism in raw["mechanisms"]:
                    games.append((int(n), int(j), int(seed), str(mechanism)))
    if target not in games:
        raise ValueError("Cannot locate the requested game in the source execution order.")
    game_index = games.index(target)
    records_per_game = 1 + len(effective_solvers)
    records_before_dss = (
        game_index * records_per_game + 1 + effective_solvers.index(DSS_SOLVER)
    )
    return int(case["seed"]) + 7919 * records_before_dss


def _tensor_identity(
    case: dict[str, Any],
    config_path: Path,
    results_path: Path,
    profiles: tuple[Profile, ...],
    max_rollouts: int,
) -> dict[str, Any]:
    return {
        "case": case,
        "config_sha256": _sha256_file(config_path),
        "results_sha256": _sha256_file(results_path),
        "profile_count": len(profiles),
        "profile_order_sha256": hashlib.sha256(
            "\n".join("|".join(profile) for profile in profiles).encode("utf-8")
        ).hexdigest(),
        "max_train_rollouts": int(max_rollouts),
    }


def evaluate_training_tensor(
    case_dir: Path,
    case: dict[str, Any],
    raw: dict[str, Any],
    config_path: Path,
    results_path: Path,
    workers: int,
    max_rollouts: int,
) -> tuple[tuple[Profile, ...], np.ndarray, np.ndarray, dict[str, Any]]:
    n, j, seed = int(case["N"]), int(case["J"]), int(case["seed"])
    mechanism = str(case["mechanism"])
    policies = build_policy_library_for_mechanism(mechanism, j)
    profiles = tuple(profile_space(tuple(policies.keys()), n))
    identity = _tensor_identity(case, config_path, results_path, profiles, max_rollouts)
    tensor_dir = case_dir / "training_tensor"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = tensor_dir / "metadata.json"
    returns_path = tensor_dir / "returns.npy"
    objectives_path = tensor_dir / "platform_objectives.npy"
    completed_path = tensor_dir / "completed_profiles.npy"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("identity") != identity:
            raise ValueError(f"Training tensor checkpoint mismatch for {_case_key(case)}.")
        returns = np.load(returns_path, mmap_mode="r+")
        objectives = np.load(objectives_path, mmap_mode="r+")
        completed = np.load(completed_path, mmap_mode="r+")
    else:
        returns = np.lib.format.open_memmap(
            returns_path,
            mode="w+",
            dtype=np.float64,
            shape=(len(profiles), int(max_rollouts), n),
        )
        objectives = np.lib.format.open_memmap(
            objectives_path,
            mode="w+",
            dtype=np.float64,
            shape=(len(profiles), int(max_rollouts)),
        )
        completed = np.lib.format.open_memmap(
            completed_path,
            mode="w+",
            dtype=np.uint8,
            shape=(len(profiles),),
        )
        completed[:] = 0
        returns.flush()
        objectives.flush()
        completed.flush()
        metadata = {
            "identity": identity,
            "complete": False,
            "profile_count": len(profiles),
            "completed_profile_count": 0,
            "runtime_seconds": 0.0,
        }
        _json_dump(metadata_path, metadata)
    expected_shapes = ((len(profiles), max_rollouts, n), (len(profiles), max_rollouts), (len(profiles),))
    if (returns.shape, objectives.shape, completed.shape) != expected_shapes:
        raise ValueError("Training tensor checkpoint arrays have unexpected shapes.")
    missing = [idx for idx in range(len(profiles)) if not bool(completed[idx])]
    if missing:
        env_config = config_from_mapping(raw)
        markup_grid = tuple(float(value) for value in raw["policies"]["markup_grid"])
        lead_time_grid = tuple(float(value) for value in raw["policies"]["lead_time_grid"])
        seeds = solver_seed(seed)
        args = [
            (
                idx,
                profiles[idx],
                mechanism,
                policies,
                env_config,
                n,
                seeds,
                max_rollouts,
                markup_grid,
                lead_time_grid,
            )
            for idx in missing
        ]
        start = time.perf_counter()
        with ProcessPoolExecutor(
            max_workers=max(1, min(int(workers), len(args))),
            mp_context=mp.get_context("fork"),
        ) as executor:
            for completed_count, (idx, profile_returns, profile_objectives) in enumerate(
                executor.map(_training_worker, args, chunksize=1), start=1
            ):
                returns[idx] = profile_returns
                objectives[idx] = profile_objectives
                completed[idx] = 1
                if completed_count % 1024 == 0 or completed_count == len(args):
                    returns.flush()
                    objectives.flush()
                    completed.flush()
                    metadata["completed_profile_count"] = int(np.sum(completed))
                    metadata["runtime_seconds"] = float(metadata.get("runtime_seconds", 0.0)) + (
                        time.perf_counter() - start
                    )
                    metadata["complete"] = bool(np.all(completed))
                    _json_dump(metadata_path, metadata)
                    print(
                        f"{_case_key(case)} training tensor: "
                        f"{metadata['completed_profile_count']}/{len(profiles)} profiles",
                        flush=True,
                    )
                    start = time.perf_counter()
    if not bool(np.all(completed)):
        raise RuntimeError("Training tensor is incomplete after evaluation.")
    metadata["complete"] = True
    metadata["completed_profile_count"] = len(profiles)
    metadata["training_seeds"] = _seed_dict(solver_seed(seed))
    _json_dump(metadata_path, metadata)
    return profiles, returns, objectives, metadata


def _prefix_game_and_estimates(
    profiles: tuple[Profile, ...],
    policy_ids: tuple[str, ...],
    returns: np.ndarray,
    objectives: np.ndarray,
    train_rollouts: int,
) -> tuple[EmpiricalGame, dict[Profile, RolloutEstimate]]:
    count = int(train_rollouts)
    returns_prefix = np.asarray(returns[:, :count, :])
    objectives_prefix = np.asarray(objectives[:, :count])
    means = np.mean(returns_prefix, axis=1)
    variances = np.var(returns_prefix, axis=1, ddof=1)
    ci = 1.96 * np.sqrt(variances / count)
    objective_means = np.mean(objectives_prefix, axis=1)
    objective_vars = np.var(objectives_prefix, axis=1, ddof=1)
    metrics = tuple(
        {"platform_operating_score": float(value)} for value in objective_means
    )
    game = EmpiricalGame(
        profiles=profiles,
        policy_ids=policy_ids,
        payoffs=means,
        ci_radius=ci,
        objectives=objective_means,
        metrics=metrics,
    )
    estimates = {
        profile: RolloutEstimate(
            profile=profile,
            n_rollouts=count,
            mean_returns=means[idx],
            var_returns=variances[idx],
            ci_radius=ci[idx],
            mean_metrics=metrics[idx],
            var_metrics={"platform_operating_score": float(objective_vars[idx])},
        )
        for idx, profile in enumerate(profiles)
    }
    return game, estimates


def _frozen_from_source(
    record: dict[str, Any], n_agents: int, policy_ids: tuple[str, ...]
) -> FrozenDistribution:
    support = []
    probabilities = []
    for item in record["support"]:
        if float(item["prob"]) > 1.0e-12:
            support.append(tuple(str(value) for value in item["profile"]))
            probabilities.append(float(item["prob"]))
    return freeze_distribution(record["solver"], support, probabilities, n_agents, policy_ids)


def _frozen_from_solution(
    solver: str,
    support: list[Profile],
    probabilities: list[float],
    n_agents: int,
    policy_ids: tuple[str, ...],
) -> FrozenDistribution:
    return freeze_distribution(solver, support, probabilities, n_agents, policy_ids)


def _q_vector(distribution: FrozenDistribution, profiles: tuple[Profile, ...]) -> np.ndarray:
    index = {profile: idx for idx, profile in enumerate(profiles)}
    q = np.zeros(len(profiles), dtype=float)
    for profile, probability in zip(
        distribution.support, distribution.probabilities, strict=True
    ):
        q[index[profile]] = probability
    return q


def _relative_gap(gap: float, q: np.ndarray, payoffs: np.ndarray) -> tuple[float, float]:
    expected = np.asarray(q, dtype=float) @ np.asarray(payoffs, dtype=float)
    denominator = max(1.0, float(np.mean(np.abs(expected))))
    return 100.0 * max(0.0, float(gap)) / denominator, denominator


def solve_training_prefixes(
    case_dir: Path,
    case: dict[str, Any],
    raw: dict[str, Any],
    source_records: dict[str, dict[str, Any]],
    profiles: tuple[Profile, ...],
    returns: np.ndarray,
    objectives: np.ndarray,
    train_rollouts: list[int],
    reproduction_tolerance: float,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], FrozenDistribution]]:
    n, j, seed = int(case["N"]), int(case["J"]), int(case["seed"])
    mechanism = str(case["mechanism"])
    policies = build_policy_library_for_mechanism(mechanism, j)
    policy_ids = tuple(policies.keys())
    source_distributions = {
        solver: _frozen_from_source(record, n, policy_ids)
        for solver, record in source_records.items()
    }
    dss_seed = _source_dss_seed(raw, case)
    rows: list[dict[str, Any]] = []
    distributions: dict[tuple[str, int], FrozenDistribution] = {}
    previous_q: dict[str, np.ndarray] = {}
    for count in train_rollouts:
        print(f"{_case_key(case)} solving training prefix R={count}", flush=True)
        game, estimates = _prefix_game_and_estimates(
            profiles, policy_ids, returns, objectives, count
        )
        start = time.perf_counter()
        full_solution = solve_full_cce_lp(game)
        full_runtime = time.perf_counter() - start
        full_distribution = _frozen_from_solution(
            FULL_SOLVER,
            full_solution.support_profiles,
            full_solution.support_probabilities,
            n,
            policy_ids,
        )
        solver_cfg = dict(raw.get("solver", {}))
        solver_cfg["rollouts_max"] = int(count)
        cache = PayoffCache(
            mechanism_id=mechanism,
            policies=policies,
            config=config_from_mapping(raw),
            n_agents=n,
            seeds=solver_seed(seed),
            n_rollouts=int(count),
            markup_grid=tuple(float(value) for value in raw["policies"]["markup_grid"]),
            lead_time_grid=tuple(float(value) for value in raw["policies"]["lead_time_grid"]),
            workers=int(solver_cfg.get("workers", 1)),
            estimates=estimates,
        )
        cache.reset_access_log()
        start = time.perf_counter()
        dss_solution = run_solver(
            DSS_SOLVER,
            cache,
            policy_ids,
            solver_cfg,
            seed=dss_seed,
            metadata={
                "experiment": "reviewer1_training_precision_calibration",
                "N": n,
                "K": j,
                "seed": seed,
            },
        )
        dss_runtime = time.perf_counter() - start
        dss_distribution = _frozen_from_solution(
            DSS_SOLVER,
            dss_solution.support_profiles,
            dss_solution.support_probabilities,
            n,
            policy_ids,
        )
        dss_audit = audit_distribution_on_full_game(
            game,
            list(dss_distribution.support),
            list(dss_distribution.probabilities),
            solver_name="DSS-TrainingTensorAudit",
        )
        for solver, distribution, gap, runtime, profile_count in (
            (FULL_SOLVER, full_distribution, full_solution.cce_gap_nominal, full_runtime, len(profiles)),
            (
                DSS_SOLVER,
                dss_distribution,
                dss_audit.cce_gap_nominal,
                dss_runtime,
                int(cache.access_stats()["solver_required_profile_count"]),
            ),
        ):
            q = _q_vector(distribution, profiles)
            relative, denominator = _relative_gap(gap, q, game.payoffs)
            source_q = _q_vector(source_distributions[solver], profiles)
            source_max_diff = float(np.max(np.abs(q - source_q))) if count == 20 else None
            source_support_match = (
                set(distribution.support) == set(source_distributions[solver].support)
                if count == 20
                else None
            )
            if count == 20 and (
                not source_support_match or source_max_diff > float(reproduction_tolerance)
            ):
                raise RuntimeError(
                    f"R=20 failed to reproduce submitted {solver} q for {_case_key(case)}: "
                    f"support_match={source_support_match}, max_diff={source_max_diff:.3g}."
                )
            tv = (
                0.5 * float(np.sum(np.abs(q - previous_q[solver])))
                if solver in previous_q
                else 0.0
            )
            previous_q[solver] = q
            distributions[(solver, int(count))] = distribution
            rows.append(
                {
                    "case_key": _case_key(case),
                    "role": str(case["role"]),
                    "N": n,
                    "J": j,
                    "seed": seed,
                    "mechanism": mechanism,
                    "solver": solver,
                    "train_rollouts": int(count),
                    "q_hash": distribution.q_hash,
                    "support_size": len(distribution.support),
                    "training_absolute_gap": float(gap),
                    "training_relative_gap_percent": relative,
                    "training_payoff_denominator": denominator,
                    "q_total_variation_from_previous_R": tv,
                    "source_q_hash": source_distributions[solver].q_hash,
                    "source_support_match_at_R20": source_support_match,
                    "source_max_probability_difference_at_R20": source_max_diff,
                    "solver_compute_runtime_seconds": runtime,
                    "solver_required_profile_count": int(profile_count),
                    "source_dss_solver_seed": dss_seed,
                }
            )
    q_records = []
    for (solver, count), distribution in distributions.items():
        q_records.append(
            {
                "solver": solver,
                "train_rollouts": count,
                "q_hash": distribution.q_hash,
                "support": [
                    {"profile": list(profile), "prob": probability}
                    for profile, probability in zip(
                        distribution.support, distribution.probabilities, strict=True
                    )
                ],
            }
        )
    _json_dump(case_dir / "training_distributions.json", q_records)
    return rows, distributions


def audit_training_distributions(
    case_dir: Path,
    case: dict[str, Any],
    raw: dict[str, Any],
    distributions: dict[tuple[str, int], FrozenDistribution],
    audit_rollouts: int,
    audit_seed_offset: int,
    bootstrap_samples: int,
    alpha: float,
    workers: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    n, j, seed = int(case["N"]), int(case["J"]), int(case["seed"])
    mechanism = str(case["mechanism"])
    policies = build_policy_library_for_mechanism(mechanism, j)
    policy_ids = tuple(policies.keys())
    closure: set[Profile] = set()
    for distribution in distributions.values():
        closure.update(support_deviation_closure(distribution.support, policy_ids))
    identity = {
        "case": case,
        "audit_rollouts": int(audit_rollouts),
        "audit_seed_offset": int(audit_seed_offset),
        "q_hashes": sorted(value.q_hash for value in distributions.values()),
        "closure": [list(profile) for profile in sorted(closure)],
    }
    identity_hash = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    training_seeds, audit_seeds = audit_seeds_for_game(seed, audit_seed_offset)
    returns_path = case_dir / "audit_profile_returns.parquet"
    metadata_path = case_dir / "audit_metadata.json"
    if returns_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("identity_hash") != identity_hash:
            raise ValueError(f"Audit checkpoint mismatch for {_case_key(case)}.")
        profile_returns = _profile_returns_from_frame(pd.read_parquet(returns_path), n)
        if set(profile_returns) != closure:
            raise ValueError("Audit checkpoint closure mismatch.")
    else:
        start = time.perf_counter()
        profile_returns = evaluate_profile_returns(
            profiles=closure,
            mechanism_id=mechanism,
            policies=policies,
            env_config=config_from_mapping(raw),
            n_agents=n,
            seeds=audit_seeds,
            rollouts=int(audit_rollouts),
            markup_grid=tuple(float(value) for value in raw["policies"]["markup_grid"]),
            lead_time_grid=tuple(float(value) for value in raw["policies"]["lead_time_grid"]),
            workers=int(workers),
        )
        _profile_returns_frame(profile_returns, n).to_parquet(returns_path, index=False)
        metadata = {
            "identity_hash": identity_hash,
            "identity": identity,
            "evaluation_runtime_seconds": time.perf_counter() - start,
            "training_audit_streams_overlap": False,
        }
    metadata["training_seeds"] = _seed_dict(training_seeds)
    metadata["audit_seeds"] = _seed_dict(audit_seeds)
    metadata["training_audit_streams_overlap"] = False
    _json_dump(metadata_path, metadata)
    rows: list[dict[str, Any]] = []
    gain_rows: list[dict[str, Any]] = []
    shared_bootstrap_seed = _stable_seed(
        _case_key(case), int(audit_rollouts), "training_precision_bootstrap"
    )
    for (solver, train_count), distribution in sorted(distributions.items()):
        samples = build_joint_audit_samples(
            profile_returns,
            distribution,
            policy_ids,
            n_agents=n,
            sample_count=int(audit_rollouts),
        )
        summary = summarize_audit_samples(
            samples,
            alpha=float(alpha),
            bootstrap_samples=int(bootstrap_samples),
            bootstrap_seed=shared_bootstrap_seed,
        )
        rows.append(
            {
                "case_key": _case_key(case),
                "solver": solver,
                "train_rollouts": int(train_count),
                "audit_rollouts": int(audit_rollouts),
                "q_hash": distribution.q_hash,
                "audit_union_closure_size": len(closure),
                "solver_closure_size": len(
                    support_deviation_closure(distribution.support, policy_ids)
                ),
                **summary,
            }
        )
        for column, (agent, policy) in enumerate(samples.labels):
            for replication in range(int(audit_rollouts)):
                gain_rows.append(
                    {
                        "case_key": _case_key(case),
                        "solver": solver,
                        "train_rollouts": int(train_count),
                        "q_hash": distribution.q_hash,
                        "replication": replication,
                        "agent": int(agent),
                        "dev_policy": str(policy),
                        "gain": float(samples.gain_samples[replication, column]),
                    }
                )
    gains = pd.DataFrame.from_records(gain_rows)
    gains.to_parquet(case_dir / "audit_gain_samples.parquet", index=False)
    return rows, gains


def run_calibration(config_path: Path) -> pd.DataFrame:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_root = ROOT / str(config["source_root"])
    audit_root = ROOT / str(config["frozen_audit_root"])
    output_root = ROOT / str(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = audit_root / "pilot_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Frozen-q audit manifest is missing.")
    audit_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit_rollouts = int(max(audit_manifest["final_nested_rollouts"]))
    train_rollouts = [int(value) for value in config["train_rollouts"]]
    if train_rollouts != sorted(train_rollouts) or train_rollouts[0] != 20:
        raise ValueError("Training rollout prefixes must be sorted and begin at 20.")
    all_rows: list[dict[str, Any]] = []
    for case in config["cases"]:
        case_dir = output_root / "games" / _case_key(case)
        case_dir.mkdir(parents=True, exist_ok=True)
        raw, config_source, results_source, source_records = _load_sources(source_root, case)
        profiles, returns, objectives, tensor_metadata = evaluate_training_tensor(
            case_dir=case_dir,
            case=case,
            raw=raw,
            config_path=config_source,
            results_path=results_source,
            workers=int(config["workers"]),
            max_rollouts=max(train_rollouts),
        )
        training_rows, distributions = solve_training_prefixes(
            case_dir=case_dir,
            case=case,
            raw=raw,
            source_records=source_records,
            profiles=profiles,
            returns=returns,
            objectives=objectives,
            train_rollouts=train_rollouts,
            reproduction_tolerance=float(config["q_reproduction_tolerance"]),
        )
        audit_rows, _ = audit_training_distributions(
            case_dir=case_dir,
            case=case,
            raw=raw,
            distributions=distributions,
            audit_rollouts=audit_rollouts,
            audit_seed_offset=int(config["audit_seed_offset"]),
            bootstrap_samples=int(config["bootstrap_samples"]),
            alpha=float(config["alpha"]),
            workers=int(config["workers"]),
        )
        training_frame = pd.DataFrame.from_records(training_rows)
        audit_frame = pd.DataFrame.from_records(audit_rows)
        merged = training_frame.merge(
            audit_frame,
            on=["case_key", "solver", "train_rollouts", "q_hash"],
            validate="one_to_one",
        )
        merged.to_csv(case_dir / "training_precision_summary.csv", index=False)
        all_rows.extend(merged.to_dict(orient="records"))
        _json_dump(
            case_dir / "case_manifest.json",
            {
                "case": case,
                "audit_rollouts": audit_rollouts,
                "train_rollouts": train_rollouts,
                "training_tensor_metadata": tensor_metadata,
                "calibration_only_not_for_publication": True,
            },
        )
    combined = pd.DataFrame.from_records(all_rows)
    expected = len(config["cases"]) * len(train_rollouts) * 2
    if len(combined) != expected:
        raise RuntimeError(f"Expected {expected} calibration rows, found {len(combined)}.")
    if combined.duplicated(["case_key", "solver", "train_rollouts"]).any():
        raise RuntimeError("Training-precision output contains duplicate keys.")
    if combined.isna().drop(
        columns=[
            "source_support_match_at_R20",
            "source_max_probability_difference_at_R20",
        ],
        errors="ignore",
    ).any().any():
        raise RuntimeError("Training-precision output contains unexpected missing values.")
    tables = output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    combined.to_csv(tables / "training_precision_summary.csv", index=False)
    _json_dump(
        output_root / "calibration_manifest.json",
        {
            "experiment": config["experiment"],
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": _sha256_file(config_path),
            "case_count": len(config["cases"]),
            "row_count": len(combined),
            "train_rollouts": train_rollouts,
            "audit_rollouts": audit_rollouts,
            "calibration_only_not_for_publication": True,
            "manuscript_modified": False,
        },
    )
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Reviewer 1 training-payoff precision calibration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "cmfg_cce/configs/reviewer1_training_precision_calibration.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_calibration(args.config.resolve())


if __name__ == "__main__":
    main()
