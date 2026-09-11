from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cmfg_cce.baselines import mwu_policy_trace_time_budget, regret_matching_policy_trace_time_budget
from cmfg_cce.envs.toy import config_from_mapping
from cmfg_cce.evaluation.empirical_game import build_empirical_game
from cmfg_cce.evaluation.full_audit import audit_sparse_result_on_full_game
from cmfg_cce.evaluation.payoff_cache import PayoffCache, full_tensor_size
from cmfg_cce.evaluation.rollout import RolloutEstimate
from cmfg_cce.experiments.common import LEAD_TIME_GRID, MARKUP_GRID
from cmfg_cce.experiments.run_oracle_baselines import (
    load_yaml,
    solver_seed,
    sparse_solution_record,
    weighted_metrics,
)
from cmfg_cce.policies.policy_library import build_policy_library_for_mechanism
from cmfg_cce.solvers.cce_lp import CceSolution


SOLVERS = ("MWU-PolicyTrace", "RegretMatching-PolicyTrace")
REFERENCE_SOLVER = "REPAIR-SAD-CCE"
FULL_TENSOR_SOLVER = "FullTensor-CCE-LP"


def _round_budget(runtime_seconds: float, increment_seconds: int) -> int:
    increment = max(1, int(increment_seconds))
    return max(increment, int(math.ceil(float(runtime_seconds) / increment) * increment))


def _load_budget_table(source_rows_path: Path, increment_seconds: int) -> pd.DataFrame:
    rows = pd.read_csv(source_rows_path)
    required = {"N", "K", "seed", "mechanism", "solver", "runtime_seconds"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Missing columns in source solver rows: {sorted(missing)}")
    dss = rows[rows["solver"] == REFERENCE_SOLVER].copy()
    if dss.empty:
        raise ValueError(f"No {REFERENCE_SOLVER} rows found in {source_rows_path}")
    dss["time_budget_seconds"] = dss["runtime_seconds"].map(
        lambda value: _round_budget(float(value), increment_seconds)
    )
    dss = dss.rename(columns={"runtime_seconds": "dss_reference_runtime_seconds"})
    key_cols = ["N", "K", "seed", "mechanism"]
    if dss.duplicated(key_cols).any():
        duplicates = dss[dss.duplicated(key_cols, keep=False)][key_cols]
        raise ValueError(f"Duplicate DSS scenario rows:\n{duplicates}")
    return dss[
        key_cols
        + [
            "dss_reference_runtime_seconds",
            "time_budget_seconds",
            "full_tensor_size",
            "cce_gap_full_audit",
        ]
    ].sort_values(key_cols)


def _metric_columns(frame: pd.DataFrame, n_agents: int) -> list[str]:
    excluded = {"mechanism", "profile", "n_rollouts"}
    for agent in range(n_agents):
        excluded.update({f"return_agent_{agent}", f"var_agent_{agent}", f"ci_agent_{agent}"})
    return [column for column in frame.columns if column not in excluded]


def _estimates_from_payoff_block(frame: pd.DataFrame, n_agents: int) -> list[RolloutEstimate]:
    metric_cols = _metric_columns(frame, n_agents)
    estimates: list[RolloutEstimate] = []
    for _, row in frame.iterrows():
        profile = tuple(str(row["profile"]).split("|"))
        mean_returns = np.array([float(row[f"return_agent_{i}"]) for i in range(n_agents)], dtype=float)
        var_returns = np.array([float(row[f"var_agent_{i}"]) for i in range(n_agents)], dtype=float)
        ci_radius = np.array([float(row[f"ci_agent_{i}"]) for i in range(n_agents)], dtype=float)
        mean_metrics = {
            column: float(row[column])
            for column in metric_cols
            if column in row and pd.notna(row[column])
        }
        estimates.append(
            RolloutEstimate(
                profile=profile,
                n_rollouts=int(row["n_rollouts"]),
                mean_returns=mean_returns,
                var_returns=var_returns,
                ci_radius=ci_radius,
                mean_metrics=mean_metrics,
                var_metrics={},
            )
        )
    return estimates


def _load_source_game(
    source_root: Path,
    n_agents: int,
    k: int,
    seed: int,
    mechanism: str,
    policy_ids: tuple[str, ...],
) -> tuple[Any, float]:
    size_name = f"n{n_agents}k{k}"
    config_path = source_root / "configs" / f"{size_name}.yaml"
    payoff_path = source_root / size_name / "raw" / "payoff_cache_oracle_baselines.csv"
    rows_path = source_root / size_name / "tables" / "oracle_baselines.csv"
    raw = load_yaml(config_path)
    seeds = [int(item) for item in raw["oracle_baselines"]["seeds"]]
    if int(seed) not in seeds:
        raise ValueError(f"Seed {seed} not found in {config_path}: {seeds}")
    seed_pos = seeds.index(int(seed))
    tensor_size = full_tensor_size(n_agents, k)
    payoff_rows = pd.read_csv(payoff_path)
    mechanism_rows = payoff_rows[payoff_rows["mechanism"] == mechanism].reset_index(drop=True)
    expected_rows = len(seeds) * tensor_size
    if len(mechanism_rows) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} payoff rows for {size_name} {mechanism}; found {len(mechanism_rows)}."
        )
    start = seed_pos * tensor_size
    block = mechanism_rows.iloc[start : start + tensor_size].reset_index(drop=True)
    game = build_empirical_game(_estimates_from_payoff_block(block, n_agents), policy_ids)
    if game.n_profiles != tensor_size:
        raise ValueError(f"Source game has {game.n_profiles} profiles; expected {tensor_size}.")
    source_solver_rows = pd.read_csv(rows_path)
    full_rows = source_solver_rows[
        (source_solver_rows["N"] == n_agents)
        & (source_solver_rows["K"] == k)
        & (source_solver_rows["seed"] == seed)
        & (source_solver_rows["mechanism"] == mechanism)
        & (source_solver_rows["solver"] == FULL_TENSOR_SOLVER)
    ]
    if len(full_rows) != 1:
        raise ValueError(f"Expected one full tensor row for {size_name} seed={seed} {mechanism}; found {len(full_rows)}.")
    full_eval_time = float(full_rows.iloc[0].get("full_tensor_eval_time", 0.0))
    return game, full_eval_time


def _full_oracle_from_source(
    source_rows: pd.DataFrame,
    n_agents: int,
    k: int,
    seed: int,
    mechanism: str,
) -> CceSolution:
    row = source_rows[
        (source_rows["N"] == n_agents)
        & (source_rows["K"] == k)
        & (source_rows["seed"] == seed)
        & (source_rows["mechanism"] == mechanism)
        & (source_rows["solver"] == FULL_TENSOR_SOLVER)
    ]
    if len(row) != 1:
        raise ValueError(f"Expected one {FULL_TENSOR_SOLVER} source row; found {len(row)}.")
    record = row.iloc[0]
    return CceSolution(
        solver=FULL_TENSOR_SOLVER,
        q=np.array([], dtype=float),
        objective_value=float(record.get("objective_value", 0.0)),
        cce_gap_nominal=float(record.get("cce_gap_full_audit", record.get("cce_gap_nominal", 0.0))),
        cce_gap_ucb=float(record.get("cce_gap_ucb_full_audit", record.get("cce_gap_ucb", 0.0))),
        support_profiles=[],
        support_probabilities=[],
        status="loaded_full_tensor_oracle_from_source_queue",
        full_optimality_certified=True,
        pricing_mode="full_tensor_cce_lp",
        max_deviation={},
    )


def _run_time_budget_solver(
    solver_name: str,
    cache: PayoffCache,
    policy_ids: tuple[str, ...],
    budget_seconds: float,
    seed: int,
    solver_cfg: dict[str, Any] | None = None,
):
    solver_cfg = solver_cfg or {}
    if solver_name == "MWU-PolicyTrace":
        mwu_cfg = solver_cfg.get("mwu", {})
        return mwu_policy_trace_time_budget(
            cache,
            policy_ids,
            budget_seconds,
            seed,
            eta=float(mwu_cfg.get("eta", 0.15)),
            schedule=str(mwu_cfg.get("schedule", "constant")),
            exploration_floor=float(mwu_cfg.get("exploration_floor", 0.0)),
            burn_in_rounds=int(mwu_cfg.get("burn_in_rounds", 0)),
        )
    if solver_name == "RegretMatching-PolicyTrace":
        return regret_matching_policy_trace_time_budget(cache, policy_ids, budget_seconds, seed)
    raise ValueError(f"Unsupported runtime-matched solver: {solver_name}")


def _flatten_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = {key: value for key, value in record.items() if key not in {"support", "mechanism_metrics"}}
        for key, value in record.get("mechanism_metrics", {}).items():
            row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(records: list[dict[str, Any]], budget_table: pd.DataFrame, output_dir: Path) -> None:
    raw_dir = output_dir / "raw"
    table_dir = output_dir / "tables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "runtime_matched_no_regret_results.json").write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )
    flat = _flatten_records(records)
    flat.to_csv(table_dir / "runtime_matched_no_regret_solver_rows.csv", index=False)
    budget_table.to_csv(table_dir / "runtime_matched_dss_budgets.csv", index=False)
    summary_cols = [
        "cce_gap_full_audit",
        "cce_gap_ucb_full_audit",
        "gap_to_full_tensor_optimum",
        "runtime_seconds",
        "time_budget_seconds",
        "dss_reference_runtime_seconds",
        "solver_required_profile_count",
        "profile_saving_ratio",
        "support_size",
        "rounds",
    ]
    available = [column for column in summary_cols if column in flat.columns]
    if not flat.empty:
        summary = flat.groupby(["N", "K", "solver"], as_index=False)[available].agg(["mean", "min", "max"])
        summary.columns = [
            "_".join(str(part) for part in column).strip("_")
            for column in summary.columns.to_flat_index()
        ]
        summary.to_csv(table_dir / "runtime_matched_summary_by_size_solver.csv", index=False)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def run_experiment(
    source_root: Path,
    source_rows_path: Path,
    output_dir: Path,
    rounding_seconds: int,
    limit_scenarios: int | None = None,
    resume: bool = True,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    budget_table = _load_budget_table(source_rows_path, rounding_seconds)
    if limit_scenarios is not None:
        budget_table = budget_table.head(max(1, int(limit_scenarios))).copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = raw_dir / "runtime_matched_no_regret_results.jsonl"
    records = _read_jsonl(jsonl_path) if resume else []
    completed = {
        (int(record["N"]), int(record["K"]), int(record["seed"]), str(record["mechanism"]), str(record["solver"]))
        for record in records
    }
    source_rows = pd.read_csv(source_rows_path)
    config_cache: dict[tuple[int, int], dict[str, Any]] = {}

    for _, budget_row in budget_table.iterrows():
        n_agents = int(budget_row["N"])
        k = int(budget_row["K"])
        seed = int(budget_row["seed"])
        mechanism = str(budget_row["mechanism"])
        budget_seconds = float(budget_row["time_budget_seconds"])
        size_key = (n_agents, k)
        if size_key not in config_cache:
            config_cache[size_key] = load_yaml(source_root / "configs" / f"n{n_agents}k{k}.yaml")
        raw = config_cache[size_key]
        env_config = config_from_mapping(raw)
        solver_cfg = raw.get("solver", {})
        workers = int(solver_cfg.get("workers", raw.get("workers", 1)))
        markup_grid = tuple(raw.get("policies", {}).get("markup_grid", MARKUP_GRID))
        lead_time_grid = tuple(raw.get("policies", {}).get("lead_time_grid", LEAD_TIME_GRID))
        policies = build_policy_library_for_mechanism(mechanism, k)
        policy_ids = tuple(policies.keys())
        game, full_eval_time = _load_source_game(source_root, n_agents, k, seed, mechanism, policy_ids)
        full_oracle = _full_oracle_from_source(source_rows, n_agents, k, seed, mechanism)
        seeds = solver_seed(seed)

        for solver_name in SOLVERS:
            key = (n_agents, k, seed, mechanism, solver_name)
            if resume and key in completed:
                continue
            solver_cache = PayoffCache(
                mechanism_id=mechanism,
                policies=policies,
                config=env_config,
                n_agents=n_agents,
                seeds=seeds,
                n_rollouts=int(raw["oracle_baselines"]["rollouts_per_profile"]),
                markup_grid=markup_grid,
                lead_time_grid=lead_time_grid,
                workers=workers,
            )
            solver_cache.reset_access_log()
            start = time.perf_counter()
            result = _run_time_budget_solver(
                solver_name,
                solver_cache,
                policy_ids,
                budget_seconds,
                seed=seed + 7919 * (len(records) + 1),
                solver_cfg=solver_cfg,
            )
            solver_runtime = time.perf_counter() - start
            solver_stats = solver_cache.access_stats()
            solver_payoff_eval_time = float(solver_cache.eval_time_seconds)
            solver_stats["solver_payoff_eval_time"] = solver_payoff_eval_time
            solver_stats["solver_compute_time"] = max(0.0, solver_runtime - solver_payoff_eval_time)
            audit_start = time.perf_counter()
            full_audit = audit_sparse_result_on_full_game(game, result)
            full_audit_compute_time = time.perf_counter() - audit_start

            result.objective_value = full_audit.objective_value
            result.cce_gap_nominal = full_audit.cce_gap_nominal
            result.cce_gap_ucb = full_audit.cce_gap_ucb
            result.max_deviation = dict(full_audit.max_deviation or {})

            record = sparse_solution_record(
                raw["experiment"],
                mechanism,
                result,
                full_audit,
                full_oracle,
                full_oracle,
                solver_cache,
                game,
                n_agents,
                k,
                seed,
                env_config.horizon,
                solver_runtime,
                solver_stats,
                None,
                full_eval_time,
                full_audit_compute_time,
                f"runtime_matched_to_dss_{int(budget_seconds)}s",
            )
            record.update(
                {
                    "experiment_family": "runtime_matched_no_regret",
                    "budget_reference_solver": REFERENCE_SOLVER,
                    "dss_reference_runtime_seconds": float(budget_row["dss_reference_runtime_seconds"]),
                    "time_budget_seconds": budget_seconds,
                    "time_budget_rounding_seconds": int(rounding_seconds),
                    "solver_runtime_over_budget_seconds": solver_runtime - budget_seconds,
                    "solver_runtime_ratio_to_budget": solver_runtime / max(1.0, budget_seconds),
                    "rounds": int(result.diagnostics.get("rounds", 0)),
                    "full_verification_payoff_source": str(source_root),
                    "full_verification_payoff_source_mode": "loaded_existing_full_tensor_payoffs",
                }
            )
            records.append(record)
            completed.add(key)
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"runtime-matched N={n_agents} K={k} seed={seed} {mechanism} {solver_name}: "
                f"budget={budget_seconds:.0f}s runtime={solver_runtime:.2f}s "
                f"rounds={record['rounds']} verified_gap={full_audit.cce_gap_nominal:.4g} "
                f"profiles={int(solver_stats['solver_required_profile_count'])}/{game.n_profiles}",
                flush=True,
            )
    write_outputs(records, budget_table, output_dir)
    return records, budget_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        default="outputs/policy_library_full_queue_20260613_213147_policy_library_full",
    )
    parser.add_argument("--source-rows", default=None)
    parser.add_argument("--output-dir", default="outputs/runtime_matched_no_regret")
    parser.add_argument("--rounding-seconds", type=int, default=5)
    parser.add_argument("--limit-scenarios", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    source_root = Path(args.source_root)
    source_rows = (
        Path(args.source_rows)
        if args.source_rows
        else source_root / "tables_combined" / "policy_library_full_queue_solver_rows.csv"
    )
    run_experiment(
        source_root=source_root,
        source_rows_path=source_rows,
        output_dir=Path(args.output_dir),
        rounding_seconds=int(args.rounding_seconds),
        limit_scenarios=args.limit_scenarios,
        resume=not bool(args.no_resume),
    )


if __name__ == "__main__":
    main()
