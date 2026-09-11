"""Baseline solvers and traces."""

from cmfg_cce.baselines.simple import (
    mwu_policy_trace,
    mwu_policy_trace_time_budget,
    random_support_cce,
    regret_matching_policy_trace,
    regret_matching_policy_trace_time_budget,
    uniform_mixture,
    vanilla_sampled_cce,
    welfare_best_pure,
)
from cmfg_cce.baselines.mwu_tuning import (
    MwuTuningConfig,
    revision_full_v1_mwu_grid,
    select_global_mwu_config,
)

__all__ = [
    "mwu_policy_trace",
    "mwu_policy_trace_time_budget",
    "random_support_cce",
    "regret_matching_policy_trace",
    "regret_matching_policy_trace_time_budget",
    "uniform_mixture",
    "vanilla_sampled_cce",
    "welfare_best_pure",
    "MwuTuningConfig",
    "revision_full_v1_mwu_grid",
    "select_global_mwu_config",
]
