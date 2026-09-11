from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from cmfg_cce.evaluation.backend_cache import BackendPayoffCache
from cmfg_cce.evaluation.rollout import RolloutEstimate


SCHEMA_VERSION = "backend_payoff_cache_v1"


def _normalized(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _payload(cache: BackendPayoffCache) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "cache_identity": dict(cache.backend.cache_identity),
        "mechanism_id": cache.mechanism_id,
        "n_agents": cache.n_agents,
        "horizon": int(cache.backend.horizon),
        "env_version": str(cache.backend.env_version),
        "default_rollouts": int(cache.n_rollouts),
        "estimates": [
            cache.estimates[profile].to_jsonable()
            for profile in sorted(cache.estimates)
        ],
    }


def save_backend_cache(cache: BackendPayoffCache, path: str | Path) -> Path:
    """Atomically checkpoint a pluggable-backend payoff cache.

    The identity block is deliberately stored alongside the estimates.  A
    route-aware case can therefore never resume from a legacy, different
    fleet, order generator, pressure cell, mechanism, or seed by accident.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(_payload(cache), indent=2), encoding="utf-8")
    temporary.replace(target)
    return target


def load_backend_cache(
    cache: BackendPayoffCache,
    path: str | Path,
    *,
    strict_identity: bool = True,
) -> int:
    """Restore estimates into ``cache`` and return the number loaded.

    Loading does not count as solver access or simulation time.  The runner
    must still start its reported DSS timing from an empty cache; this helper
    is intended for crash-safe completion/audit checkpoints and explicit
    resume workflows, not for hiding precomputed profiles from accounting.
    """

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported payoff-cache schema: {payload.get('schema_version')!r}"
        )

    expected = {
        "cache_identity": dict(cache.backend.cache_identity),
        "mechanism_id": cache.mechanism_id,
        "n_agents": cache.n_agents,
        "horizon": int(cache.backend.horizon),
        "env_version": str(cache.backend.env_version),
        "default_rollouts": int(cache.n_rollouts),
    }
    observed = {key: payload.get(key) for key in expected}
    if strict_identity and _normalized(observed) != _normalized(expected):
        raise ValueError(
            "Payoff-cache identity mismatch; refusing to mix incompatible "
            f"simulation data. expected={expected!r}, observed={observed!r}"
        )

    loaded = 0
    for record in payload.get("estimates", []):
        estimate = RolloutEstimate(
            profile=tuple(str(value) for value in record["profile"]),
            n_rollouts=int(record["n_rollouts"]),
            mean_returns=np.asarray(record["mean_returns"], dtype=float),
            var_returns=np.asarray(record["var_returns"], dtype=float),
            ci_radius=np.asarray(record["ci_radius"], dtype=float),
            mean_metrics={
                str(key): float(value)
                for key, value in record.get("mean_metrics", {}).items()
            },
            var_metrics={
                str(key): float(value)
                for key, value in record.get("var_metrics", {}).items()
            },
        )
        if len(estimate.profile) != cache.n_agents:
            raise ValueError(
                f"Profile {estimate.profile!r} has {len(estimate.profile)} agents; "
                f"expected {cache.n_agents}."
            )
        if estimate.mean_returns.shape != (cache.n_agents,):
            raise ValueError(
                f"Estimate for {estimate.profile!r} has return shape "
                f"{estimate.mean_returns.shape}; expected {(cache.n_agents,)}."
            )
        cache.estimates[estimate.profile] = estimate
        loaded += 1
    return loaded
