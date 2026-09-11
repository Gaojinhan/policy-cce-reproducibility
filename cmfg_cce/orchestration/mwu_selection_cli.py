from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from cmfg_cce.experiments.run_revision_mwu_tuning import reduce_mwu_tuning_results
from cmfg_cce.orchestration.manifest import atomic_json


def discover_tuning_results(root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for path in sorted(Path(root).rglob("result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if payload.get("family") == "mwu_tuning_pipeline":
            paths.append(path)
    return tuple(paths)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reduce all 24 calibration games to one frozen global MWU selection."
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = discover_tuning_results(args.results_root)
    payload = reduce_mwu_tuning_results(paths)
    atomic_json(args.output, payload)
    print(
        f"selected={payload['selected_config']['mwu_config_id']} "
        f"selection_sha256={payload['selection_sha256']}"
    )


if __name__ == "__main__":
    main()
