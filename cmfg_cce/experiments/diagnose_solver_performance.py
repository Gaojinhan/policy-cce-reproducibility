from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cmfg_cce.policies.policy_library import build_policy_library_for_mechanism


RESULT_SPECS = [
    ("core_mechanisms", "payoff_cache_core_mechanisms.parquet"),
    ("scalability", "payoff_cache_scalability.parquet"),
    ("full_robustness", "payoff_cache_full_robustness.parquet"),
    ("ablations", "payoff_cache_ablations.parquet"),
]


def profile_key(profile: list[str] | tuple[str, ...]) -> str:
    return "|".join(profile)


def mean_abs_episode_profit(record: dict[str, Any]) -> float:
    metrics = record.get("mechanism_metrics", {})
    profits = [
        abs(float(value))
        for key, value in metrics.items()
        if key.startswith("profit_manufacturer_") and value is not None
    ]
    if profits:
        return float(np.mean(profits))
    total = abs(float(metrics.get("manufacturer_total_profit", 0.0)))
    return total / max(1, int(record["N"]))


def flatten_result(record: dict[str, Any]) -> dict[str, Any]:
    row = {key: value for key, value in record.items() if key not in {"support", "mechanism_metrics"}}
    row["mean_abs_episode_profit"] = mean_abs_episode_profit(record)
    row["manufacturer_total_profit"] = float(record.get("mechanism_metrics", {}).get("manufacturer_total_profit", 0.0))
    return row


def build_payoff_lookup(group: pd.DataFrame, n_agents: int) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], float]:
    return_cols = [f"return_agent_{i}" for i in range(n_agents)]
    ci_cols = [f"ci_agent_{i}" for i in range(n_agents)]
    lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for _, row in group.iterrows():
        lookup[str(row["profile"])] = (
            row[return_cols].to_numpy(dtype=float),
            row[ci_cols].to_numpy(dtype=float),
        )
    returns = group[return_cols].to_numpy(dtype=float)
    payoff_range = float(np.nanmax(returns) - np.nanmin(returns)) if returns.size else 0.0
    return lookup, payoff_range


def exact_gap_diagnostics(
    record: dict[str, Any],
    lookup: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    n_agents = int(record["N"])
    k = int(record["K"])
    policy_ids = tuple(build_policy_library_for_mechanism(str(record["mechanism"]), k).keys())
    support = [(tuple(item["profile"]), float(item["prob"])) for item in record["support"]]

    best_nominal = {
        "gap_mean": -np.inf,
        "agent": None,
        "policy": None,
    }
    best_ucb = {
        "gap_ucb": -np.inf,
        "gap_mean_at_ucb": np.nan,
        "gap_ci_at_ucb": np.nan,
        "agent": None,
        "policy": None,
        "support_profile": None,
        "support_profile_prob": np.nan,
        "support_profile_ucb_contribution": np.nan,
    }
    missing_profiles = 0

    for agent in range(n_agents):
        for dev_policy in policy_ids:
            nominal_gain = 0.0
            ucb_gain = 0.0
            largest_contribution = {
                "profile": None,
                "prob": np.nan,
                "value": -np.inf,
            }
            for profile, prob in support:
                dev_profile = list(profile)
                dev_profile[agent] = dev_policy
                base_key = profile_key(profile)
                dev_key = profile_key(tuple(dev_profile))
                if base_key not in lookup or dev_key not in lookup:
                    missing_profiles += 1
                    continue
                base_return, base_ci = lookup[base_key]
                dev_return, dev_ci = lookup[dev_key]
                profile_nominal = float(dev_return[agent] - base_return[agent])
                profile_ucb = float(dev_return[agent] + dev_ci[agent] - (base_return[agent] - base_ci[agent]))
                nominal_gain += prob * profile_nominal
                ucb_gain += prob * profile_ucb
                contribution = prob * profile_ucb
                if contribution > largest_contribution["value"]:
                    largest_contribution = {
                        "profile": base_key,
                        "prob": prob,
                        "value": contribution,
                    }

            if nominal_gain > float(best_nominal["gap_mean"]):
                best_nominal = {
                    "gap_mean": nominal_gain,
                    "agent": agent,
                    "policy": dev_policy,
                }
            if ucb_gain > float(best_ucb["gap_ucb"]):
                best_ucb = {
                    "gap_ucb": ucb_gain,
                    "gap_mean_at_ucb": nominal_gain,
                    "gap_ci_at_ucb": max(0.0, ucb_gain - nominal_gain),
                    "agent": agent,
                    "policy": dev_policy,
                    "support_profile": largest_contribution["profile"],
                    "support_profile_prob": largest_contribution["prob"],
                    "support_profile_ucb_contribution": largest_contribution["value"],
                }

    gap_mean = max(0.0, float(best_nominal["gap_mean"]))
    gap_ucb = max(0.0, float(best_ucb["gap_ucb"]))
    gap_mean_at_ucb = max(0.0, float(best_ucb["gap_mean_at_ucb"]))
    return {
        "gap_mean": gap_mean,
        "gap_ucb_exact": gap_ucb,
        "gap_ci": max(0.0, gap_ucb - gap_mean_at_ucb),
        "gap_mean_at_ucb": gap_mean_at_ucb,
        "worst_mean_agent": best_nominal["agent"],
        "worst_mean_policy": best_nominal["policy"],
        "worst_ucb_agent": best_ucb["agent"],
        "worst_ucb_policy": best_ucb["policy"],
        "worst_support_profile": best_ucb["support_profile"],
        "worst_support_profile_prob": best_ucb["support_profile_prob"],
        "worst_support_profile_ucb_contribution": best_ucb["support_profile_ucb_contribution"],
        "missing_profile_lookups": missing_profiles,
    }


def diagnose_result_file(raw_dir: Path, raw_name: str, payoff_name: str) -> pd.DataFrame:
    records = json.loads((raw_dir / f"cce_results_{raw_name}.json").read_text(encoding="utf-8"))
    payoff = pd.read_parquet(raw_dir / payoff_name, engine="pyarrow")
    grouped = payoff.groupby(["experiment", "N", "K", "seed", "mechanism"], sort=False)
    lookup_cache: dict[tuple[Any, ...], tuple[dict[str, tuple[np.ndarray, np.ndarray]], float]] = {}
    rows: list[dict[str, Any]] = []

    for record in records:
        row = flatten_result(record)
        key = (
            record["experiment"],
            int(record["N"]),
            int(record["K"]),
            int(record["seed"]),
            record["mechanism"],
        )
        if key not in lookup_cache:
            group = grouped.get_group(key)
            lookup_cache[key] = build_payoff_lookup(group, int(record["N"]))
        lookup, payoff_range = lookup_cache[key]
        row.update(exact_gap_diagnostics(record, lookup))
        row["payoff_range"] = payoff_range
        rows.append(row)

    return pd.DataFrame(rows)


def diagnose_toy(raw_dir: Path) -> pd.DataFrame:
    records = json.loads((raw_dir / "cce_results_toy.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for record in records:
        row = flatten_result(record)
        row["gap_mean"] = float(record["cce_gap_nominal"])
        row["gap_ucb_exact"] = float(record["cce_gap_ucb"])
        row["gap_mean_at_ucb"] = float(record["cce_gap_nominal"])
        row["gap_ci"] = max(0.0, float(record["cce_gap_ucb"]) - float(record["cce_gap_nominal"]))
        row["worst_mean_agent"] = record.get("max_deviation_agent")
        row["worst_mean_policy"] = record.get("max_deviation_policy")
        row["worst_ucb_agent"] = record.get("max_deviation_agent")
        row["worst_ucb_policy"] = record.get("max_deviation_policy")
        row["worst_support_profile"] = ""
        row["worst_support_profile_prob"] = np.nan
        row["worst_support_profile_ucb_contribution"] = np.nan
        row["missing_profile_lookups"] = 0
        row["payoff_range"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def add_normalized_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gap_per_order"] = df["gap_ucb_exact"] / df["horizon"].clip(lower=1)
    df["gap_mean_per_order"] = df["gap_mean"] / df["horizon"].clip(lower=1)
    df["gap_ci_per_order"] = df["gap_ci"] / df["horizon"].clip(lower=1)
    df["gap_rel_profit"] = df["gap_ucb_exact"] / df["mean_abs_episode_profit"].replace(0.0, np.nan)
    df["gap_mean_rel_profit"] = df["gap_mean"] / df["mean_abs_episode_profit"].replace(0.0, np.nan)
    df["gap_rel_range"] = df["gap_ucb_exact"] / df["payoff_range"].replace(0.0, np.nan)
    df["gap_mean_rel_range"] = df["gap_mean"] / df["payoff_range"].replace(0.0, np.nan)
    return df


def write_summary_tables(df: pd.DataFrame, table_dir: Path) -> None:
    summary = (
        df.groupby(["experiment", "solver"], as_index=False)
        .agg(
            rows=("solver", "size"),
            gap_mean=("gap_mean", "mean"),
            gap_ci=("gap_ci", "mean"),
            gap_ucb=("gap_ucb_exact", "mean"),
            gap_per_order=("gap_per_order", "mean"),
            gap_rel_profit=("gap_rel_profit", "mean"),
            gap_rel_range=("gap_rel_range", "mean"),
            runtime_seconds=("runtime_seconds", "mean"),
            evaluated_profile_count=("evaluated_profile_count", "mean"),
            rollout_steps_total=("rollout_steps_total", "mean"),
            support_size=("support_size", "mean"),
            good_rel_profit_share=("gap_rel_profit", lambda x: float(np.mean(x < 0.05))),
            acceptable_rel_profit_share=("gap_rel_profit", lambda x: float(np.mean(x < 0.10))),
            weak_rel_profit_share=("gap_rel_profit", lambda x: float(np.mean(x > 0.20))),
        )
        .sort_values(["experiment", "gap_ucb"])
    )
    summary.to_csv(table_dir / "paper_solver_diagnostics.csv", index=False)

    core = df[df["experiment"] == "core_mechanism_comparison"]
    if not core.empty:
        core.groupby(["mechanism", "solver"], as_index=False).agg(
            rows=("solver", "size"),
            gap_mean=("gap_mean", "mean"),
            gap_ci=("gap_ci", "mean"),
            gap_ucb=("gap_ucb_exact", "mean"),
            gap_per_order=("gap_per_order", "mean"),
            gap_rel_profit=("gap_rel_profit", "mean"),
            gap_rel_range=("gap_rel_range", "mean"),
            runtime_seconds=("runtime_seconds", "mean"),
            evaluated_profile_count=("evaluated_profile_count", "mean"),
        ).sort_values(["mechanism", "gap_ucb"]).to_csv(table_dir / "paper_solver_diagnostics_core_by_mechanism.csv", index=False)

    policy_counts = (
        df.groupby(["experiment", "solver", "mechanism", "worst_ucb_policy"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["experiment", "solver", "mechanism", "count"], ascending=[True, True, True, False])
    )
    policy_counts.to_csv(table_dir / "solver_worst_deviation_policy_counts.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    table_dir = output_dir / "tables"
    frames = [diagnose_result_file(raw_dir, raw_name, payoff_name) for raw_name, payoff_name in RESULT_SPECS]
    frames.append(diagnose_toy(raw_dir))
    diagnostics = add_normalized_metrics(pd.concat(frames, ignore_index=True))
    diagnostics.to_csv(table_dir / "solver_diagnostics.csv", index=False)
    write_summary_tables(diagnostics, table_dir)
    print(f"Wrote {table_dir / 'solver_diagnostics.csv'}")
    print(f"Wrote {table_dir / 'paper_solver_diagnostics.csv'}")
    print(f"Wrote {table_dir / 'paper_solver_diagnostics_core_by_mechanism.csv'}")
    print(f"Wrote {table_dir / 'solver_worst_deviation_policy_counts.csv'}")


if __name__ == "__main__":
    main()
