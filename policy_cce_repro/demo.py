"""A bounded synthetic workflow, separate from every paper experiment.

The frozen simulator and solvers are called directly. This module owns only
the small demo configuration, new random streams, and output serialization.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
import time
import traceback
from typing import Any

import numpy as np

import cmfg_cce
from cmfg_cce.envs.toy import ToyEnvConfig
from cmfg_cce.evaluation.empirical_game import EmpiricalGame, build_empirical_game, enumerate_profiles
from cmfg_cce.evaluation.independent_audit import (
    build_joint_audit_samples, distribution_hash, freeze_distribution, summarize_audit_samples,
)
from cmfg_cce.evaluation.payoff_cache import PayoffCache, support_deviation_closure
from cmfg_cce.evaluation.rollout import RolloutSeeds
from cmfg_cce.evaluation.toy_backend import ToyPolicyGameBackend
from cmfg_cce.orchestration.chunks import deterministic_npz
from cmfg_cce.policies.policy_library import build_policy_library_for_mechanism
from cmfg_cce.solvers.cce_lp import compute_cce_gap, compute_cce_gap_from_deviation_closure, solve_full_cce_lp
from cmfg_cce.solvers.sparse_cce import solve_repair_sparse


DEMO_NAMESPACE = "policy-cce-repro-synthetic-demo-v1"
N_AGENTS = 3
N_POLICIES = 3
HORIZON = 8
TRAIN_ROLLOUTS = 4
EVALUATION_ROLLOUTS = 8
BOOTSTRAP_SAMPLES = 100
MECHANISM = "M1_price_first"
MARKUP_GRID = (0.05, 0.10, 0.20, 0.35, 0.50)
LEAD_TIME_GRID = (0.55, 0.70, 0.85, 1.00)
GAP_CHECK_TOLERANCE = 1.0e-8


def _seed(label: str) -> int:
    # Demo-only integers; no campaign seed helper or manifest is consulted.
    payload = f"{DEMO_NAMESPACE}:{label}".encode("utf-8")
    return 2**40 + int.from_bytes(sha256(payload).digest()[:4], "big")


def _streams(stage: str) -> RolloutSeeds:
    if stage not in {"training", "evaluation"}:
        raise ValueError("Unknown demo stream stage")
    return RolloutSeeds(
        type_seed=_seed("fixed-population"),
        order_seed=_seed(f"{stage}:orders"),
        tie_break_seed=_seed(f"{stage}:ties"),
        rollout_replication_seed=_seed(f"{stage}:replications"),
        outside_seed=_seed(f"{stage}:outside"),
        availability_seed=_seed(f"{stage}:availability"),
    )


def _write_json(path: Path, value: Any) -> None:
    # Exclusive creation: even interrupted runs must never overwrite evidence.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _new_output(path: str | Path) -> Path:
    requested = Path(path).expanduser()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f"Demo output must be a new directory: {requested}")
    output = requested.resolve()
    protected = (
        Path(cmfg_cce.__file__).resolve().parent,
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parents[1] / "metadata",
    )
    if any(output == root or root in output.parents for root in protected):
        raise ValueError("Demo output cannot be inside source or package metadata")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _cache(config: ToyEnvConfig, policies: dict) -> PayoffCache:
    return PayoffCache(
        mechanism_id=MECHANISM, policies=policies, config=config,
        n_agents=N_AGENTS, seeds=_streams("training"), n_rollouts=TRAIN_ROLLOUTS,
        markup_grid=MARKUP_GRID, lead_time_grid=LEAD_TIME_GRID, workers=1,
    )


def _verify(game: EmpiricalGame, frozen: Any) -> dict[str, Any]:
    """Compare all-tensor and closure-only checks for exactly the same q."""
    q = np.zeros(game.n_profiles)
    index = game.profile_to_index
    for profile, probability in zip(frozen.support, frozen.probabilities, strict=True):
        q[index[profile]] = probability
    closure = tuple(sorted(support_deviation_closure(frozen.support, game.policy_ids)))
    indices = np.array([index[profile] for profile in closure], dtype=int)
    closure_game = EmpiricalGame(
        profiles=closure, policy_ids=game.policy_ids, payoffs=game.payoffs[indices],
        ci_radius=game.ci_radius[indices], objectives=game.objectives[indices],
        metrics=tuple(game.metrics[i] for i in indices),
    )
    full_gap = float(compute_cce_gap(game, q))
    closure_gap = float(compute_cce_gap_from_deviation_closure(
        closure_game, list(frozen.support), list(frozen.probabilities),
    ))
    if not np.isfinite([full_gap, closure_gap]).all():
        raise ValueError("Demo verification produced nonfinite gaps")
    if abs(full_gap - closure_gap) > GAP_CHECK_TOLERANCE:
        raise ValueError("Demo closure and full-tensor gaps disagree")
    return {
        "full_tensor_gap": full_gap, "closure_gap": closure_gap,
        "closure_profiles": len(closure), "checks_agree": True,
    }


def run_demo(output_dir: str | Path) -> dict[str, Any]:
    """Run one fixed, small synthetic game into a previously absent directory.

    No data directory, cloud credential, campaign ID, or configurable workload
    is accepted. On an error, retain the new output and write demo-error.json.
    """
    started = time.perf_counter()
    output = _new_output(output_dir)
    try:
        config = ToyEnvConfig(horizon=HORIZON)
        policies = build_policy_library_for_mechanism(MECHANISM, N_POLICIES)
        policy_ids = tuple(policies)
        profiles = enumerate_profiles(policy_ids, N_AGENTS)
        settings = {
            "schema": "policy_cce_synthetic_demo_config_v1",
            "purpose": "Small software demonstration; not a paper result or runtime benchmark",
            "seed_namespace": DEMO_NAMESPACE, "mechanism": MECHANISM,
            "n_agents": N_AGENTS, "policy_ids": list(policy_ids),
            "profile_count": len(profiles), "environment": asdict(config),
            "training_rollouts_per_profile": TRAIN_ROLLOUTS,
            "evaluation_rollouts_per_profile": EVALUATION_ROLLOUTS,
            "bootstrap_samples": BOOTSTRAP_SAMPLES, "workers": 1,
            "markup_grid": list(MARKUP_GRID), "lead_time_grid": list(LEAD_TIME_GRID),
            "training_seeds": asdict(_streams("training")),
            "evaluation_seeds": asdict(_streams("evaluation")),
            "dss_seed": _seed("dss-search"), "bootstrap_seed": _seed("evaluation:bootstrap"),
            "full_lp_settings": {
                "tolerance": 1.0e-9, "selector": "platform_operating_score", "epsilon_tolerance": 1.0e-9,
            },
            "dss_settings": {
                "initial_support_size": 2, "max_support_size": 8,
                "support_add_batch_size": 2, "max_rounds": 3, "target_gap": 1.0e-8,
                "seed": _seed("dss-search"), "rollouts_max": TRAIN_ROLLOUTS,
                "active_sampling": False, "repair_rounds": 1, "top_constraints": 2,
                "top_profiles_per_constraint": 2, "q_min": 1.0e-4,
                "mean_threshold": 1.0, "contribution_min": 0.0,
                "enable_multi_agent_repair": True, "max_agents_repaired_per_profile": 2,
                "beam_width": 2, "profile_budget_multiplier": 1.0,
                "targeted_deviation_policies": None, "solver_name": "REPAIR-SAD-CCE",
                "selector": "platform_operating_score", "epsilon_tolerance": 1.0e-9,
            },
            "max_simulated_episodes": len(profiles) * (2 * TRAIN_ROLLOUTS + EVALUATION_ROLLOUTS),
        }
        # Give the public API the same list-valued structure as its JSON file.
        settings = json.loads(json.dumps(settings))
        _write_json(output / "demo-config.json", settings)
        records: dict[str, Any] = {}
        distributions: dict[str, Any] = {}
        cache_by_solver: dict[str, PayoffCache] = {}

        for name in ("FullTensor-CCE-LP", "DSS-CCE"):
            solve_started = time.perf_counter()
            cache = _cache(config, policies)
            if cache.estimates:
                raise ValueError("Demo solver did not start with an empty payoff cache")
            if name == "FullTensor-CCE-LP":
                cache.ensure(profiles)
                game = build_empirical_game([cache.estimates[p] for p in profiles], policy_ids)
                result = solve_full_cce_lp(game, **settings["full_lp_settings"])
            else:
                result = solve_repair_sparse(cache, policy_ids, **settings["dss_settings"])
            frozen = freeze_distribution(
                name, result.support_profiles, result.support_probabilities, N_AGENTS, policy_ids,
            )
            elapsed = time.perf_counter() - solve_started
            distributions[name] = frozen
            cache_by_solver[name] = cache
            records[name] = {
                "implementation_solver": result.solver,
                "status": result.status,
                "solver_wall_seconds": elapsed,
                "training_profiles_evaluated": len(cache.estimates),
                "training_episodes": cache.eval_rollout_episode_count,
                "empty_cache_at_start": True,
                "distribution": {
                    "support": [list(p) for p in frozen.support],
                    "probabilities": list(frozen.probabilities), "q_hash": frozen.q_hash,
                },
            }
            _write_json(output / f"demo-{name}-solver.json", records[name])

        # Read-only verification after both q values have been frozen. DSS had
        # no access to the complete LP cache during its search.
        verify_started = time.perf_counter()
        full_cache = cache_by_solver["FullTensor-CCE-LP"]
        sparse_cache = cache_by_solver["DSS-CCE"]
        for profile, estimate in sparse_cache.estimates.items():
            if not np.array_equal(estimate.mean_returns, full_cache.estimates[profile].mean_returns):
                raise ValueError("Demo solvers did not use identical training streams")
        for name, frozen in distributions.items():
            records[name]["training_verification"] = _verify(game, frozen)
        training_verification_seconds = time.perf_counter() - verify_started

        evaluate_started = time.perf_counter()
        backend = ToyPolicyGameBackend(
            mechanism_id=MECHANISM, policies=policies, config=config, n_agents=N_AGENTS,
            seeds=_streams("evaluation"), markup_grid=MARKUP_GRID,
            lead_time_grid=LEAD_TIME_GRID, stream_label=DEMO_NAMESPACE + ":evaluation",
        )
        returns = np.stack([
            np.stack([backend.run_episode(p, r)[0] for r in range(EVALUATION_ROLLOUTS)])
            for p in profiles
        ])
        if returns.shape != (len(profiles), EVALUATION_ROLLOUTS, N_AGENTS) or not np.isfinite(returns).all():
            raise ValueError("Unexpected demo evaluation return array")
        evaluation_game = EmpiricalGame(
            profiles=profiles, policy_ids=policy_ids, payoffs=returns.mean(axis=1),
            ci_radius=np.zeros((len(profiles), N_AGENTS)), objectives=np.zeros(len(profiles)),
            metrics=tuple({} for _ in profiles),
        )
        samples_by_profile = dict(zip(profiles, returns, strict=True))
        for name, frozen in distributions.items():
            samples = build_joint_audit_samples(samples_by_profile, frozen, policy_ids, N_AGENTS)
            summary = summarize_audit_samples(
                samples, bootstrap_samples=BOOTSTRAP_SAMPLES, bootstrap_seed=_seed("evaluation:bootstrap"),
            )
            checks = _verify(evaluation_game, frozen)
            if abs(summary["nominal_gap"] - checks["full_tensor_gap"]) > GAP_CHECK_TOLERANCE:
                raise ValueError("Demo sample-based and tensor-based independent gaps disagree")
            after_hash = distribution_hash(frozen.support, frozen.probabilities)
            if after_hash != frozen.q_hash:
                raise ValueError("Demo evaluation changed the frozen distribution")
            records[name]["independent_evaluation"] = {
                **checks, "q_hash_before": frozen.q_hash, "q_hash_after": after_hash,
                "statistics": summary,
            }
        independent_evaluation_seconds = time.perf_counter() - evaluate_started
        arrays = {
            "profiles": np.array(profiles, dtype="U2"),
            "training_mean_returns": game.payoffs,
            "training_ci_radius": game.ci_radius,
            "training_objectives": game.objectives,
            "dss_training_profile_indices": np.array([profiles.index(p) for p in sorted(sparse_cache.estimates)], dtype=np.int64),
            "independent_profile_returns": returns,
        }
        payload = deterministic_npz(arrays)
        with (output / "demo-samples.npz").open("xb") as handle:
            handle.write(payload)
        episode_count = sum(cache.eval_rollout_episode_count for cache in cache_by_solver.values()) + len(profiles) * EVALUATION_ROLLOUTS
        if episode_count > settings["max_simulated_episodes"]:
            raise ValueError("Demo exceeded its fixed episode budget")
        report = {
            "schema": "policy_cce_synthetic_demo_result_v1", "status": "pass",
            "purpose": settings["purpose"], "seed_namespace": DEMO_NAMESPACE,
            "config": settings, "solvers": records,
            "training_verification_seconds": training_verification_seconds,
            "independent_evaluation_seconds": independent_evaluation_seconds,
            "simulated_episodes": episode_count,
            "runtime_environment": {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version()},
            "artifacts": {"demo-samples.npz": sha256(payload).hexdigest()},
            "workflow_wall_seconds": time.perf_counter() - started,
        }
        _write_json(output / "demo-result.json", report)
        return report
    except Exception as error:
        _write_json(output / "demo-error.json", {
            "schema": "policy_cce_synthetic_demo_error_v1", "status": "error",
            "error_type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(), "seed_namespace": DEMO_NAMESPACE,
        })
        raise
