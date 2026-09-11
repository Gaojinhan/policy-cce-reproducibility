from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from cmfg_cce.envs.toy import ToyEnvConfig, config_from_mapping
from cmfg_cce.evaluation.payoff_cache import PayoffCache, random_profiles, support_deviation_closure
from cmfg_cce.evaluation.rollout import Profile, RolloutSeeds
from cmfg_cce.experiments.common import (
    LEAD_TIME_GRID,
    MARKUP_GRID,
    load_yaml,
    result_record,
    run_solver,
    solver_seed,
    write_experiment_outputs,
)
from cmfg_cce.policies.policy_library import build_policy_library_for_mechanism
from cmfg_cce.solvers.sparse_cce import (
    SparseCceResult,
    audit_delta_distribution,
    audit_sparse_distribution,
    paired_deviation_arrays,
    solve_sad_sparse,
    solve_sparse_support,
)


def shifted_solver_seed(seed: int, offset: int) -> RolloutSeeds:
    base = solver_seed(seed)
    return RolloutSeeds(
        type_seed=base.type_seed + offset,
        order_seed=base.order_seed + offset,
        tie_break_seed=base.tie_break_seed + offset,
        rollout_replication_seed=base.rollout_replication_seed + offset,
        outside_seed=base.outside_seed + offset,
        availability_seed=base.availability_seed + offset,
    )


def make_cache(
    raw: dict[str, Any],
    env_config: ToyEnvConfig,
    mechanism: str,
    n: int,
    k: int,
    seed: int,
    n_rollouts: int,
    workers: int,
    seed_offset: int = 0,
) -> tuple[PayoffCache, tuple[str, ...]]:
    policies = build_policy_library_for_mechanism(mechanism, int(k))
    policy_ids = tuple(policies.keys())
    seeds = shifted_solver_seed(int(seed), int(seed_offset)) if seed_offset else solver_seed(int(seed))
    cache = PayoffCache(
        mechanism_id=mechanism,
        policies=policies,
        config=env_config,
        n_agents=int(n),
        seeds=seeds,
        n_rollouts=int(n_rollouts),
        markup_grid=tuple(raw.get("policies", {}).get("markup_grid", MARKUP_GRID)),
        lead_time_grid=tuple(raw.get("policies", {}).get("lead_time_grid", LEAD_TIME_GRID)),
        workers=int(workers),
    )
    return cache, policy_ids


def support_from_record(record: dict[str, Any]) -> tuple[list[Profile], list[float]]:
    support = [tuple(item["profile"]) for item in record["support"]]
    probs = [float(item["prob"]) for item in record["support"]]
    total = sum(probs)
    return support, [prob / total for prob in probs]


def train_record_key(record: dict[str, Any]) -> tuple[int, int, int, str]:
    return int(record["N"]), int(record["K"]), int(record["seed"]), str(record["mechanism"])


def load_train_records(path: str | Path, solver: str) -> dict[tuple[int, int, int, str], dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))
    return {train_record_key(record): record for record in records if record.get("solver") == solver}


def bootstrap_max_t_beta(
    cache: PayoffCache,
    support: list[Profile],
    probs: list[float],
    labels: list[tuple[int, str]],
    delta_mean: np.ndarray,
    delta_var: np.ndarray,
    delta_n: np.ndarray,
    bootstrap_samples: int,
    seed: int,
    quantile: float = 0.95,
) -> float:
    probs_arr = np.array(probs, dtype=float)
    se = np.sqrt(np.sum((probs_arr[None, :] ** 2) * delta_var / np.maximum(delta_n, 1.0), axis=1))
    valid = se > 1.0e-12
    if not np.any(valid) or bootstrap_samples <= 0:
        return 1.96
    rng = np.random.default_rng(seed)
    stats: list[float] = []
    for _ in range(int(bootstrap_samples)):
        centered = np.zeros(len(labels), dtype=float)
        for row_idx, (agent, dev_policy) in enumerate(labels):
            total = 0.0
            for col_idx, profile in enumerate(support):
                if profile[agent] == dev_policy:
                    continue
                samples = np.array(cache.paired_delta_samples.get((profile, agent, dev_policy), []), dtype=float)
                if samples.size == 0:
                    continue
                draw = rng.choice(samples, size=samples.size, replace=True)
                total += probs_arr[col_idx] * (float(np.mean(draw)) - float(delta_mean[row_idx, col_idx]))
            centered[row_idx] = total
        stats.append(float(np.max(centered[valid] / se[valid])))
    return max(0.0, float(np.quantile(stats, quantile)))


def max_entropy_distribution(delta_mean: np.ndarray, epsilon: float, tau: float) -> tuple[list[float], float]:
    n_support = int(delta_mean.shape[1])
    if n_support <= 1:
        return [1.0], 0.0

    def objective(q: np.ndarray) -> float:
        q = np.clip(q, 1.0e-12, 1.0)
        return float(np.sum(q * np.log(q)))

    constraints = [
        {"type": "eq", "fun": lambda q: np.sum(q) - 1.0},
        {"type": "ineq", "fun": lambda q: (float(epsilon) + float(tau)) - delta_mean @ q},
    ]
    start = np.ones(n_support, dtype=float) / n_support
    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_support,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1.0e-9},
    )
    if not result.success:
        return start.tolist(), float(-objective(start))
    q = np.maximum(result.x, 0.0)
    q = q / np.sum(q)
    return q.tolist(), float(-objective(q))


def paired_holdout_results(
    cache: PayoffCache,
    support: list[Profile],
    probs: list[float],
    policy_ids: tuple[str, ...],
    cfg: dict[str, Any],
    metadata: dict[str, Any],
    base_solver_name: str,
) -> tuple[list[SparseCceResult], dict[str, float]]:
    samples = int(cfg.get("pair_samples", 100))
    beta = float(cfg.get("pointwise_beta", 1.96))
    alpha = float(cfg.get("simultaneous_alpha", 0.05))
    bootstrap_samples = int(cfg.get("bootstrap_samples", 500))
    delta_mean, delta_var, delta_n, labels = paired_deviation_arrays(
        cache,
        support,
        policy_ids,
        target_samples=samples,
        solver_name=base_solver_name,
        metadata=metadata,
    )
    constraint_count = max(1, len(labels))
    bonf_beta = float(NormalDist().inv_cdf(1.0 - alpha / constraint_count))
    max_t_beta = bootstrap_max_t_beta(
        cache,
        support,
        probs,
        labels,
        delta_mean,
        delta_var,
        delta_n,
        bootstrap_samples=bootstrap_samples,
        seed=int(metadata["seed"]) + 991,
        quantile=1.0 - alpha,
    )
    variants = [
        ("REPAIR-PAIR-HoldoutPointwise", beta, "holdout_pointwise"),
        ("REPAIR-PAIR-HoldoutBonferroni", bonf_beta, "holdout_bonferroni"),
        ("REPAIR-PAIR-HoldoutMaxT", max_t_beta, "holdout_max_t_bootstrap"),
    ]
    results = [
        audit_delta_distribution(
            cache,
            support,
            probs,
            delta_mean,
            delta_var,
            delta_n,
            labels,
            solver_name=name,
            beta=variant_beta,
            pricing_mode="frozen_train_support",
            audit_mode=audit_mode,
            certificate_status="independent_holdout_paired_crn_audit",
        )
        for name, variant_beta, audit_mode in variants
    ]
    entropy_probs, entropy = max_entropy_distribution(
        delta_mean,
        epsilon=results[0].cce_gap_nominal,
        tau=float(cfg.get("entropy_tau", 1.0)),
    )
    info = {
        "pointwise_beta": beta,
        "bonferroni_beta": bonf_beta,
        "max_t_beta": max_t_beta,
        "constraint_count": float(constraint_count),
        "holdout_pair_samples": float(samples),
        "bootstrap_samples": float(bootstrap_samples),
        "train_support_entropy": float(-sum(prob * math.log(max(prob, 1.0e-12)) for prob in probs)),
        "train_effective_support_size": float(math.exp(-sum(prob * math.log(max(prob, 1.0e-12)) for prob in probs))),
        "entropy_tie_break_entropy": entropy,
        "entropy_tie_break_effective_support_size": float(math.exp(entropy)),
        "entropy_tie_break_support_size": float(sum(1 for prob in entropy_probs if prob > 1.0e-6)),
    }
    return results, info


def run_random_extra_rollouts(
    cache: PayoffCache,
    policy_ids: tuple[str, ...],
    solver_cfg: dict[str, Any],
    seed: int,
    target_rollout_steps: int,
    max_rollouts_per_profile: int,
) -> SparseCceResult:
    sad = solve_sad_sparse(
        cache,
        policy_ids,
        initial_support_size=int(solver_cfg.get("initial_support_size", 12)),
        max_support_size=int(solver_cfg.get("max_support_size", 40)),
        support_add_batch_size=int(solver_cfg.get("support_add_batch_size", 8)),
        max_rounds=int(solver_cfg.get("max_rounds", 4)),
        target_gap=float(solver_cfg.get("target_gap", 1.0)),
        seed=seed,
        rollouts_max=int(solver_cfg.get("rollouts_max", cache.n_rollouts)),
        active_sampling=bool(solver_cfg.get("active_sampling", True)),
    )
    closure = list(support_deviation_closure(sad.support_profiles, policy_ids))
    cache.ensure(closure)
    rng = np.random.default_rng(seed + 4441)
    rng.shuffle(closure)
    target_episodes = int(math.ceil(target_rollout_steps / max(1, cache.config.horizon)))
    current_requests = dict(cache.requested_rollouts_by_profile)
    current_total = int(sum(current_requests.values()))
    for profile in closure:
        if current_total >= target_episodes:
            break
        current = int(current_requests.get(profile, cache.n_rollouts))
        additional = min(max(0, target_episodes - current_total), max(0, int(max_rollouts_per_profile) - current))
        if additional <= 0:
            continue
        target = current + additional
        cache.resample(profile, target)
        current_requests[profile] = target
        current_total += additional
    audited = audit_sparse_distribution(
        cache,
        sad.support_profiles,
        sad.support_probabilities,
        policy_ids,
        "SAD-CCE+RandomExtraRollouts",
        "budget matched random extra rollouts",
    )
    audited.support_expansion_rounds = sad.support_expansion_rounds
    audited.active_sampling_rounds = sad.active_sampling_rounds
    return audited


def run_random_extra_profiles(
    cache: PayoffCache,
    policy_ids: tuple[str, ...],
    solver_cfg: dict[str, Any],
    seed: int,
    target_profile_count: int,
) -> SparseCceResult:
    sad = solve_sad_sparse(
        cache,
        policy_ids,
        initial_support_size=int(solver_cfg.get("initial_support_size", 12)),
        max_support_size=int(solver_cfg.get("max_support_size", 40)),
        support_add_batch_size=int(solver_cfg.get("support_add_batch_size", 8)),
        max_rounds=int(solver_cfg.get("max_rounds", 4)),
        target_gap=float(solver_cfg.get("target_gap", 1.0)),
        seed=seed,
        rollouts_max=int(solver_cfg.get("rollouts_max", cache.n_rollouts)),
        active_sampling=bool(solver_cfg.get("active_sampling", True)),
    )
    support = list(dict.fromkeys(sad.support_profiles))
    rng = np.random.default_rng(seed + 7711)
    closure_per_profile = max(1, cache.n_agents * (len(policy_ids) - 1) + 1)
    needed = max(0, int(target_profile_count) - cache.evaluated_profile_count)
    extra_support = int(math.ceil(needed / closure_per_profile))
    candidates = random_profiles(policy_ids, cache.n_agents, extra_support + len(support), rng)
    for candidate in candidates:
        if candidate not in support:
            support.append(candidate)
        if len(support) >= len(sad.support_profiles) + extra_support:
            break
    result = solve_sparse_support(cache, support, policy_ids, "SAD-CCE+RandomExtraProfiles")
    result.pricing_mode = "budget_matched_random_profiles"
    return result


def run_best_pure_profile(
    cache: PayoffCache,
    policy_ids: tuple[str, ...],
    train_support: list[Profile],
    seed: int,
    target_profile_count: int,
    max_candidates: int,
) -> SparseCceResult:
    rng = np.random.default_rng(seed + 9921)
    closure_per_profile = max(1, cache.n_agents * (len(policy_ids) - 1) + 1)
    candidate_count = max(len(train_support), int(math.ceil(target_profile_count / closure_per_profile)))
    candidate_count = min(max(1, int(max_candidates)), candidate_count)
    candidates = list(dict.fromkeys(train_support + random_profiles(policy_ids, cache.n_agents, candidate_count, rng)))
    best: SparseCceResult | None = None
    for profile in candidates:
        result = audit_sparse_distribution(
            cache,
            [profile],
            [1.0],
            policy_ids,
            "BestPureProfile",
            "best pure by audited UCB over budget matched candidates",
        )
        if best is None or result.cce_gap_ucb < best.cce_gap_ucb:
            best = result
    if best is None:
        raise RuntimeError("BestPureProfile did not evaluate any candidate.")
    best.pricing_mode = "budget_matched_best_pure"
    best.diagnostics["candidate_count"] = len(candidates)
    return best


def _override_list(values: list[Any], override: str | None) -> list[Any]:
    if not override:
        return values
    return [int(value.strip()) for value in override.split(",") if value.strip()]


def run_reviewer_audit_matrix(
    config_path: str | Path,
    n_values_override: str | None = None,
    k_values_override: str | None = None,
    seeds_override: str | None = None,
) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    raw = load_yaml(config_path)
    env_config = config_from_mapping(raw)
    matrix = raw["reviewer_audits"]
    solver_cfg = raw.get("solver", {})
    audit_cfg = raw.get("reviewer_audit", {})
    train_records = load_train_records(
        audit_cfg.get("train_results_path", "outputs/pair_repair_paper/raw/cce_results_pair_repair.json"),
        audit_cfg.get("train_solver", "REPAIR-PAIR-SAD-CCE"),
    )
    records: list[dict] = []
    payoff_records: list[dict] = []
    paired_delta_records: list[dict] = []
    workers = int(solver_cfg.get("workers", raw.get("workers", 1)))
    n_values = _override_list(list(matrix["N_values"]), n_values_override)
    k_values = _override_list(list(matrix["policies_per_agent"]), k_values_override)
    seeds = _override_list(list(matrix["seeds"]), seeds_override)
    for n in n_values:
        for k in k_values:
            for seed in seeds:
                for mechanism in raw["mechanisms"]:
                    key = (int(n), int(k), int(seed), str(mechanism))
                    train_record = train_records.get(key)
                    train_cache: PayoffCache | None = None
                    if train_record is None:
                        if not bool(audit_cfg.get("train_if_missing", False)):
                            raise KeyError(f"Missing train record for {key}")
                        train_cache, policy_ids = make_cache(
                            raw,
                            env_config,
                            mechanism,
                            int(n),
                            int(k),
                            int(seed),
                            int(matrix.get("rollouts_initial", 10)),
                            workers,
                        )
                        metadata = {
                            "experiment": raw["experiment"],
                            "env_version": raw.get("env_version", ""),
                            "N": int(n),
                            "K": int(k),
                            "seed": int(seed),
                            "run_budget_label": str(raw.get("run_budget_label", "paper_candidate")),
                        }
                        train_result = run_solver(
                            "REPAIR-PAIR-SAD-CCE",
                            train_cache,
                            policy_ids,
                            solver_cfg,
                            seed=int(seed) + 3100,
                            metadata=metadata,
                        )
                        stats = train_cache.access_stats()
                        train_record = result_record(
                            raw["experiment"],
                            mechanism,
                            train_result,
                            train_cache,
                            int(n),
                            int(k),
                            env_config.horizon,
                            0.0,
                            train_cache.eval_time_seconds,
                            0.0,
                            str(raw.get("run_budget_label", "paper_candidate")),
                            int(seed),
                            stats,
                            train_cache.evaluated_profile_count,
                            train_cache.evaluated_profile_count,
                            train_cache.eval_rollout_episode_count,
                        )
                    support, probs = support_from_record(train_record)

                    # Experiment 1 and 2: independent holdout and simultaneous CI.
                    if bool(audit_cfg.get("run_holdout", True)):
                        cache, policy_ids = make_cache(
                            raw,
                            env_config,
                            mechanism,
                            int(n),
                            int(k),
                            int(seed),
                            int(matrix.get("rollouts_initial", 10)),
                            workers,
                            seed_offset=int(audit_cfg.get("holdout_seed_offset", 100000)),
                        )
                        cache.reset_access_log()
                        profiles_before = cache.evaluated_profile_count
                        eval_before = cache.eval_time_seconds
                        episodes_before = cache.eval_rollout_episode_count
                        start = time.perf_counter()
                        metadata = {
                            "experiment": raw["experiment"],
                            "env_version": raw.get("env_version", ""),
                            "N": int(n),
                            "K": int(k),
                            "seed": int(seed),
                            "run_budget_label": str(raw.get("run_budget_label", "paper_candidate")),
                        }
                        holdout_results, holdout_info = paired_holdout_results(
                            cache,
                            support,
                            probs,
                            policy_ids,
                            audit_cfg,
                            metadata,
                            base_solver_name="REPAIR-PAIR-HoldoutAudit",
                        )
                        runtime = time.perf_counter() - start
                        stats = cache.access_stats()
                        profiles_after = cache.evaluated_profile_count
                        eval_delta = cache.eval_time_seconds - eval_before
                        episodes_after = cache.eval_rollout_episode_count
                        for result in holdout_results:
                            record = result_record(
                                raw["experiment"],
                                mechanism,
                                result,
                                cache,
                                int(n),
                                int(k),
                                env_config.horizon,
                                runtime,
                                eval_delta,
                                max(0.0, runtime - eval_delta),
                                str(raw.get("run_budget_label", "paper_candidate")),
                                int(seed),
                                stats,
                                profiles_after,
                                profiles_after - profiles_before,
                                episodes_after - episodes_before,
                            )
                            record.update(
                                {
                                    "experiment_family": "independent_holdout_simultaneous_ci",
                                    "train_solver": train_record["solver"],
                                    "train_cce_gap_ucb": train_record["cce_gap_ucb"],
                                    "train_gap_rel_profit": train_record["gap_rel_profit"],
                                    **holdout_info,
                                }
                            )
                            records.append(record)
                        payoff_records.extend(
                            cache.records(
                                {
                                    "experiment": raw["experiment"],
                                    "solver": "reviewer_holdout_payoff_cache",
                                    "N": int(n),
                                    "K": int(k),
                                    "seed": int(seed),
                                    "run_budget_label": str(raw.get("run_budget_label", "paper_candidate")),
                                }
                            )
                        )
                        paired_delta_records.extend(cache.paired_delta_rows)

                    # Experiment 3 and 4: budget-matched diagnostics and pure support check.
                    run_budget_diagnostics = audit_cfg.get(
                        "run_budget_diagnostics",
                        audit_cfg.get("run_budget_baselines", True),
                    )
                    if bool(run_budget_diagnostics):
                        diagnostic_specs = [
                            "SAD-CCE+RandomExtraRollouts",
                            "SAD-CCE+RandomExtraProfiles",
                            "BestPureProfile",
                        ]
                        for solver_name in diagnostic_specs:
                            cache, policy_ids = make_cache(
                                raw,
                                env_config,
                                mechanism,
                                int(n),
                                int(k),
                                int(seed),
                                int(matrix.get("rollouts_initial", 10)),
                                workers,
                            )
                            cache.reset_access_log()
                            profiles_before = cache.evaluated_profile_count
                            eval_before = cache.eval_time_seconds
                            episodes_before = cache.eval_rollout_episode_count
                            start = time.perf_counter()
                            if solver_name == "SAD-CCE+RandomExtraRollouts":
                                result = run_random_extra_rollouts(
                                    cache,
                                    policy_ids,
                                    solver_cfg,
                                    seed=int(seed) + 4100,
                                    target_rollout_steps=int(train_record["rollout_steps_total"]),
                                    max_rollouts_per_profile=int(audit_cfg.get("random_extra_rollouts_max", 200)),
                                )
                            elif solver_name == "SAD-CCE+RandomExtraProfiles":
                                result = run_random_extra_profiles(
                                    cache,
                                    policy_ids,
                                    solver_cfg,
                                    seed=int(seed) + 5100,
                                    target_profile_count=int(train_record["evaluated_profile_count"]),
                                )
                            else:
                                result = run_best_pure_profile(
                                    cache,
                                    policy_ids,
                                    support,
                                    seed=int(seed) + 6100,
                                    target_profile_count=int(train_record["evaluated_profile_count"]),
                                    max_candidates=int(audit_cfg.get("best_pure_max_candidates", 64)),
                                )
                            runtime = time.perf_counter() - start
                            stats = cache.access_stats()
                            profiles_after = cache.evaluated_profile_count
                            eval_delta = cache.eval_time_seconds - eval_before
                            episodes_after = cache.eval_rollout_episode_count
                            record = result_record(
                                raw["experiment"],
                                mechanism,
                                result,
                                cache,
                                int(n),
                                int(k),
                                env_config.horizon,
                                runtime,
                                eval_delta,
                                max(0.0, runtime - eval_delta),
                                str(raw.get("run_budget_label", "paper_candidate")),
                                int(seed),
                                stats,
                                profiles_after,
                                profiles_after - profiles_before,
                                episodes_after - episodes_before,
                            )
                            record.update(
                                {
                                    "experiment_family": "budget_matched_diagnostic",
                                    "budget_reference_solver": train_record["solver"],
                                    "budget_reference_cce_gap_ucb": train_record["cce_gap_ucb"],
                                    "budget_reference_gap_rel_profit": train_record["gap_rel_profit"],
                                    "budget_reference_profiles": train_record["evaluated_profile_count"],
                                    "budget_reference_rollout_steps": train_record["rollout_steps_total"],
                                    "budget_reference_runtime": train_record["runtime_seconds"],
                                }
                            )
                            records.append(record)
                            payoff_records.extend(
                                cache.records(
                                    {
                                        "experiment": raw["experiment"],
                                        "solver": solver_name,
                                        "N": int(n),
                                        "K": int(k),
                                        "seed": int(seed),
                                        "run_budget_label": str(raw.get("run_budget_label", "paper_candidate")),
                                    }
                                )
                            )
                    print(f"{raw['experiment']} N={n} K={k} seed={seed} {mechanism}: reviewer audit records={len(records)}")
    return records, pd.DataFrame(payoff_records), pd.DataFrame(paired_delta_records)


def write_reviewer_tables(flat: pd.DataFrame, output_dir: str | Path) -> None:
    table_dir = Path(output_dir) / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    if flat.empty:
        return
    metric_cols = [
        "cce_gap_nominal",
        "cce_gap_ci",
        "cce_gap_ucb",
        "gap_rel_profit",
        "gap_rel_range",
        "support_size",
        "evaluated_profile_count",
        "rollout_steps_total",
        "runtime_seconds",
    ]
    available = [col for col in metric_cols if col in flat.columns]
    holdout = flat[flat["experiment_family"] == "independent_holdout_simultaneous_ci"]
    if not holdout.empty:
        holdout.groupby("solver", as_index=False)[
            ["train_cce_gap_ucb", "train_gap_rel_profit"] + available
        ].mean().sort_values("cce_gap_ucb").to_csv(table_dir / "paper_holdout_audit.csv", index=False)
        holdout.groupby(["mechanism", "solver"], as_index=False)[available].mean().sort_values(
            ["mechanism", "cce_gap_ucb"]
        ).to_csv(table_dir / "paper_simultaneous_ci.csv", index=False)
    budget = flat[
        flat["experiment_family"].isin(["budget_matched_diagnostic", "budget_matched_baseline"])
    ]
    if not budget.empty:
        budget.groupby("solver", as_index=False)[
            [
                "budget_reference_gap_rel_profit",
                "budget_reference_profiles",
                "budget_reference_rollout_steps",
                "budget_reference_runtime",
            ]
            + available
        ].mean().sort_values("gap_rel_profit").to_csv(table_dir / "paper_budget_matched.csv", index=False)
        budget[budget["solver"].isin(["BestPureProfile"])].groupby("solver", as_index=False)[available].mean().to_csv(
            table_dir / "paper_pure_support_check.csv",
            index=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cmfg_cce/configs/reviewer_audits.yaml")
    parser.add_argument("--output-dir", default="outputs/reviewer_audits")
    parser.add_argument("--n-values", default=None, help="Comma-separated override for reviewer_audits.N_values.")
    parser.add_argument("--k-values", default=None, help="Comma-separated override for reviewer_audits.policies_per_agent.")
    parser.add_argument("--seeds", default=None, help="Comma-separated override for reviewer_audits.seeds.")
    args = parser.parse_args()
    records, payoff_table, paired_delta_table = run_reviewer_audit_matrix(
        args.config,
        n_values_override=args.n_values,
        k_values_override=args.k_values,
        seeds_override=args.seeds,
    )
    flat = write_experiment_outputs(
        records,
        payoff_table,
        args.output_dir,
        "reviewer_audits",
        "reviewer_audits.csv",
        paired_delta_table=paired_delta_table,
    )
    write_reviewer_tables(flat, args.output_dir)
    print(f"Reviewer audits wrote {len(records)} result records.")


if __name__ == "__main__":
    main()
