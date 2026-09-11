from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True, order=True)
class MwuTuningConfig:
    """One globally applicable MWU configuration.

    ``exploration_floor`` is the total probability mass mixed with the
    uniform policy distribution.  ``burn_in_rounds`` is explicit even though
    revision-full-v1 freezes it at zero rather than tuning it per game.
    """

    eta: float
    schedule: str
    exploration_floor: float
    burn_in_rounds: int = 0

    @property
    def config_id(self) -> str:
        eta = format(float(self.eta), ".12g").replace(".", "p")
        exploration = format(float(self.exploration_floor), ".12g").replace(".", "p")
        schedule = str(self.schedule).replace("/", "_").replace("(", "").replace(")", "")
        return f"eta-{eta}__{schedule}__explore-{exploration}__burn-{int(self.burn_in_rounds)}"

    def to_solver_mapping(self) -> dict[str, Any]:
        return asdict(self)


REVISION_FULL_V1_MWU_ETAS = (0.03, 0.1, 0.3, 1.0)
REVISION_FULL_V1_MWU_SCHEDULES = ("constant", "inverse_sqrt")
REVISION_FULL_V1_MWU_EXPLORATION = (0.0, 0.05)


def revision_full_v1_mwu_grid() -> tuple[MwuTuningConfig, ...]:
    """Return the frozen 4 x 2 x 2 tuning grid in deterministic order."""

    return tuple(
        MwuTuningConfig(
            eta=float(eta),
            schedule=str(schedule),
            exploration_floor=float(exploration),
            burn_in_rounds=0,
        )
        for eta, schedule, exploration in product(
            REVISION_FULL_V1_MWU_ETAS,
            REVISION_FULL_V1_MWU_SCHEDULES,
            REVISION_FULL_V1_MWU_EXPLORATION,
        )
    )


def _finite_float(row: Mapping[str, Any], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"MWU tuning row has no valid {field!r}: {row!r}.") from exc
    if not np.isfinite(value):
        raise ValueError(f"MWU tuning field {field!r} must be finite; got {value!r}.")
    return value


def select_global_mwu_config(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_game_keys: Sequence[str],
    expected_grid: Sequence[MwuTuningConfig] | None = None,
) -> tuple[MwuTuningConfig, list[dict[str, Any]]]:
    """Select one MWU configuration without per-game cherry-picking.

    The predeclared rule minimizes the mean fresh max-t relative-gap UCB over
    every calibration game.  Deterministic tie-breakers are, in order: the
    worst-game UCB, the mean fresh nominal relative gap, and ``config_id``.
    Every configuration must have exactly one result for every expected game;
    failed or selectively missing configurations are rejected instead of
    silently dropped.
    """

    expected_games = tuple(str(value) for value in expected_game_keys)
    if not expected_games or len(set(expected_games)) != len(expected_games):
        raise ValueError("expected_game_keys must be nonempty and unique.")
    grid = tuple(expected_grid or revision_full_v1_mwu_grid())
    if not grid or len({item.config_id for item in grid}) != len(grid):
        raise ValueError("expected_grid must contain unique configurations.")

    keyed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        config_id = str(row.get("mwu_config_id", ""))
        game_key = str(row.get("game_key", ""))
        key = (config_id, game_key)
        if key in keyed:
            raise ValueError(f"Duplicate MWU tuning result for {key}.")
        keyed[key] = row

    summaries: list[dict[str, Any]] = []
    for config in grid:
        config_rows: list[Mapping[str, Any]] = []
        for game_key in expected_games:
            key = (config.config_id, game_key)
            if key not in keyed:
                raise ValueError(f"Missing MWU tuning result for {key}.")
            row = keyed[key]
            if str(row.get("status", "complete")) != "complete":
                raise ValueError(f"MWU tuning result is not complete for {key}: {row.get('status')!r}.")
            config_rows.append(row)
        ucb = np.array(
            [_finite_float(row, "fresh_max_t_relative_gap_ucb_percent") for row in config_rows],
            dtype=float,
        )
        nominal = np.array(
            [_finite_float(row, "fresh_nominal_relative_gap_percent") for row in config_rows],
            dtype=float,
        )
        summaries.append(
            {
                "mwu_config_id": config.config_id,
                **config.to_solver_mapping(),
                "calibration_game_count": len(config_rows),
                "mean_fresh_max_t_relative_gap_ucb_percent": float(np.mean(ucb)),
                "worst_fresh_max_t_relative_gap_ucb_percent": float(np.max(ucb)),
                "mean_fresh_nominal_relative_gap_percent": float(np.mean(nominal)),
            }
        )

    summaries.sort(
        key=lambda row: (
            row["mean_fresh_max_t_relative_gap_ucb_percent"],
            row["worst_fresh_max_t_relative_gap_ucb_percent"],
            row["mean_fresh_nominal_relative_gap_percent"],
            row["mwu_config_id"],
        )
    )
    selected_id = str(summaries[0]["mwu_config_id"])
    selected = next(config for config in grid if config.config_id == selected_id)
    for rank, summary in enumerate(summaries, start=1):
        summary["selection_rank"] = rank
        summary["selected"] = summary["mwu_config_id"] == selected_id
        summary["selection_rule"] = (
            "minimum mean fresh max-t relative-gap UCB; then worst UCB; "
            "then mean nominal gap; then config ID"
        )
    return selected, summaries

