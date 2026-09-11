from __future__ import annotations

"""Executable campaign reducers for downloaded ``revision-full-v1`` outputs."""

import argparse
import csv
import io
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid

from cmfg_cce.evaluation.revision_campaign_reducers import (
    DEFAULT_CNC_REDUCER_METRICS,
    DEFAULT_CNC_SENSITIVITY_METRICS,
    reduce_cnc_mechanism_results,
    reduce_cce_selection_sensitivity,
    reduce_parameter_robustness,
    reduce_policy_library_sensitivity,
    reduce_policy_transplant_results,
    reduce_pure_mixed_analysis,
    reduce_scalability_results,
    reduce_solver_benchmark_audits,
)
from cmfg_cce.orchestration.manifest import sha256_bytes, sha256_file


FAMILY_TO_RESULT_FAMILIES = {
    "benchmark": ("solver_benchmark", "solver_benchmark_runtime"),
    "cnc": ("cnc_main",),
    "transplant": ("policy_transplant",),
    "library": ("policy_library_sensitivity",),
    "parameter": ("parameter_robustness",),
    "selection": ("selection_sensitivity",),
    "pure-mixed": ("solver_benchmark", "mixed_challenge"),
    "scalability": (
        "scalability_exact",
        "scalability_sparse",
        "scalability_exact_runtime",
        "scalability_sparse_runtime",
    ),
}
REDUCER_MANIFEST_SCHEMA = "revision_full_v1_reducer_output_manifest_v1"


def discover_revision_results(root: str | Path) -> tuple[Path, ...]:
    """Discover worker result JSONs without silently deduplicating job IDs."""

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Campaign result root does not exist: {root}")
    paths: list[Path] = []
    recognized = {
        result_family
        for result_families in FAMILY_TO_RESULT_FAMILIES.values()
        for result_family in result_families
    }
    for path in sorted(root.rglob("result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unreadable result JSON under campaign root: {path}") from exc
        if payload.get("family") in recognized:
            paths.append(path)
    return tuple(paths)


def _resolve_explicit_result(value: str | Path) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    if path.is_dir():
        candidates = (path / "result.json", path / "artifacts" / "result.json")
        existing = [candidate for candidate in candidates if candidate.is_file()]
        if len(existing) == 1:
            return existing[0]
    raise FileNotFoundError(f"Explicit result path is missing or ambiguous: {path}")


def _read_result_list(path: Path) -> tuple[Path, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Result-list file does not exist: {path}")
    rows: list[Path] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line)
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        rows.append(_resolve_explicit_result(candidate))
    return tuple(rows)


def _classify(paths: Sequence[Path]) -> dict[str, list[Path]]:
    raw_families = {
        result_family
        for result_families in FAMILY_TO_RESULT_FAMILIES.values()
        for result_family in result_families
    }
    by_family = {name: [] for name in raw_families}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        result_family = payload.get("family")
        if result_family in by_family:
            by_family[str(result_family)].append(path)
    return by_family


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (tuple, list, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if value is None:
        return ""
    return value


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("A formal CSV summary cannot be empty.")
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(str(name))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: _csv_cell(row.get(name)) for name in fieldnames})
    return stream.getvalue().encode("utf-8")


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(
                f"Immutable reducer output already exists with different content: {path}"
            )
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _benchmark_csv(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in payload["joined_solver_rows"]]


def _transplant_csv(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for effect_type, source in (
        ("cross_minus_same_control", payload["direction_rows"]),
        ("cross_minus_target_recomputed", payload["secondary_direction_rows"]),
    ):
        for row in source:
            rows.append({"effect_type": effect_type, **row})
    return rows


def run_reducers(
    result_paths: Sequence[str | Path],
    *,
    output_dir: str | Path,
    families: Sequence[str] = tuple(FAMILY_TO_RESULT_FAMILIES),
    cnc_metrics: Sequence[str] = DEFAULT_CNC_REDUCER_METRICS,
    sensitivity_metrics: Sequence[str] = DEFAULT_CNC_SENSITIVITY_METRICS,
    campaign_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Run selected strict reducers and commit deterministic JSON/CSV outputs."""

    requested = tuple(str(value) for value in families)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Reducer families must be nonempty and unique.")
    unknown = set(requested).difference(FAMILY_TO_RESULT_FAMILIES)
    if unknown:
        raise ValueError(f"Unknown reducer families: {sorted(unknown)}")
    paths = tuple(_resolve_explicit_result(path) for path in result_paths)
    classified = _classify(paths)
    output_dir = Path(output_dir)
    summaries: dict[str, Mapping[str, Any]] = {}
    csv_outputs: dict[str, list[tuple[str, Sequence[Mapping[str, Any]]]]] = {}
    for family in requested:
        raw_families = FAMILY_TO_RESULT_FAMILIES[family]
        family_paths = classified[raw_families[0]]
        if family == "benchmark":
            if campaign_manifest is None:
                raise ValueError(
                    "The benchmark reducer requires the frozen formal campaign manifest."
                )
            summary = reduce_solver_benchmark_audits(
                family_paths,
                classified["solver_benchmark_runtime"],
                campaign_manifest=campaign_manifest,
            )
            csv_outputs[family] = [
                ("solver_benchmark_games.csv", _benchmark_csv(summary)),
                (
                    "solver_benchmark_by_size.csv",
                    summary["size_solver_summaries"],
                ),
                (
                    "solver_benchmark_dss_mwu_runtime.csv",
                    summary["matched_dss_mwu_runtime_rows"],
                ),
                (
                    "solver_benchmark_dss_mwu_runtime_by_size.csv",
                    summary["matched_dss_mwu_runtime_size_summaries"],
                ),
            ]
        elif family == "cnc":
            summary = reduce_cnc_mechanism_results(
                family_paths, metrics=tuple(cnc_metrics)
            )
            csv_outputs[family] = [
                ("cnc_mechanism_primary_inference.csv", summary["inference_rows"]),
                ("cnc_mechanism_diagnostics.csv", summary["diagnostic_rows"]),
                (
                    "cnc_operating_factor_primary_inference.csv",
                    summary["operating_factor_analysis"]["primary_inference_rows"],
                ),
                (
                    "cnc_operating_factor_diagnostics.csv",
                    summary["operating_factor_analysis"]["diagnostic_rows"],
                ),
            ]
        elif family == "transplant":
            summary = reduce_policy_transplant_results(family_paths)
            csv_outputs[family] = [
                ("policy_transplant_direction_inference.csv", _transplant_csv(summary))
            ]
        elif family == "library":
            summary = reduce_policy_library_sensitivity(
                family_paths, metrics=tuple(sensitivity_metrics)
            )
            csv_outputs[family] = [
                ("policy_library_setting_summaries.csv", summary["setting_summaries"]),
                (
                    "policy_library_core_expanded_diagnostics.csv",
                    summary["core_to_expanded_stability"]["mechanism_contrast_rows"],
                ),
            ]
        elif family == "parameter":
            summary = reduce_parameter_robustness(
                family_paths, metrics=tuple(sensitivity_metrics)
            )
            csv_outputs[family] = [
                (
                    "parameter_robustness_factor_contrasts.csv",
                    summary["resolution_iv_factor_contrasts"],
                ),
                ("parameter_robustness_setting_summaries.csv", summary["setting_summaries"]),
            ]
        elif family == "selection":
            summary = reduce_cce_selection_sensitivity(
                family_paths, metrics=tuple(sensitivity_metrics)
            )
            csv_outputs[family] = [
                (
                    "cce_selection_paired_differences.csv",
                    summary["paired_difference_summaries"],
                ),
                ("cce_selection_outputs.csv", summary["selector_rows"]),
            ]
        elif family == "pure-mixed":
            summary = reduce_pure_mixed_analysis(
                classified["solver_benchmark"], classified["mixed_challenge"]
            )
            csv_outputs[family] = [
                ("pure_mixed_solver_rows.csv", summary["solver_rows"]),
                ("pure_mixed_game_rows.csv", summary["game_rows"]),
            ]
        elif family == "scalability":
            summary = reduce_scalability_results(
                classified["scalability_exact"],
                classified["scalability_sparse"],
                classified["scalability_exact_runtime"],
                classified["scalability_sparse_runtime"],
            )
            csv_outputs[family] = [
                ("scalability_solver_rows.csv", summary["rows"]),
            ]
        else:  # pragma: no cover - guarded above
            raise AssertionError(family)
        summaries[family] = summary

    campaign_hashes = {str(summary["campaign_sha256"]) for summary in summaries.values()}
    matrix_hashes = {str(summary["matrix_sha256"]) for summary in summaries.values()}
    if len(campaign_hashes) != 1 or len(matrix_hashes) != 1:
        raise ValueError("Selected reducer families do not share one campaign/matrix hash.")

    written: list[Path] = []
    for family in requested:
        summary_path = output_dir / f"{family}_summary.json"
        _write_immutable(summary_path, _json_bytes(summaries[family]))
        written.append(summary_path)
        for filename, rows in csv_outputs[family]:
            # A diagnostic family may be structurally empty (for example when
            # the CLI is intentionally run on primary CNC metrics only).  Its
            # absence is recorded in JSON rather than represented by a fake row.
            if not rows:
                continue
            csv_path = output_dir / filename
            _write_immutable(csv_path, _csv_bytes(rows))
            written.append(csv_path)

    manifest = {
        "schema_version": REDUCER_MANIFEST_SCHEMA,
        "matrix_sha256": matrix_hashes.pop(),
        "campaign_sha256": campaign_hashes.pop(),
        "families": list(requested),
        "files": [
            {
                "path": path.relative_to(output_dir).as_posix(),
                "size": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for path in sorted(written)
        ],
    }
    manifest["output_set_sha256"] = sha256_bytes(
        json.dumps(manifest["files"], sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    manifest_path = output_dir / "reducer_manifest.json"
    _write_immutable(manifest_path, _json_bytes(manifest))
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly reduce a complete downloaded revision-full-v1 campaign to "
            "immutable JSON/CSV summaries."
        )
    )
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--result", action="append", default=[])
    parser.add_argument("--result-list", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--campaign-manifest",
        type=Path,
        help=(
            "Frozen formal CampaignManifest required for benchmark source/config/"
            "dependency verification."
        ),
    )
    parser.add_argument(
        "--family",
        action="append",
        choices=tuple(FAMILY_TO_RESULT_FAMILIES),
        help="Reducer family; repeat as needed. Defaults to all frozen reducers.",
    )
    parser.add_argument(
        "--cnc-metrics",
        help="Comma-separated override for CNC mechanism metrics (testing/reanalysis only).",
    )
    parser.add_argument(
        "--sensitivity-metrics",
        help="Comma-separated override for library/parameter descriptive metrics.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths: list[Path] = []
    if args.results_root is not None:
        paths.extend(discover_revision_results(args.results_root))
    paths.extend(_resolve_explicit_result(value) for value in args.result)
    if args.result_list is not None:
        paths.extend(_read_result_list(args.result_list))
    if not paths:
        raise ValueError(
            "Provide --results-root, at least one --result, or --result-list."
        )
    cnc_metrics = (
        tuple(value.strip() for value in args.cnc_metrics.split(",") if value.strip())
        if args.cnc_metrics
        else DEFAULT_CNC_REDUCER_METRICS
    )
    sensitivity_metrics = (
        tuple(
            value.strip()
            for value in args.sensitivity_metrics.split(",")
            if value.strip()
        )
        if args.sensitivity_metrics
        else DEFAULT_CNC_SENSITIVITY_METRICS
    )
    manifest = run_reducers(
        paths,
        output_dir=args.output_dir,
        families=tuple(args.family or FAMILY_TO_RESULT_FAMILIES),
        cnc_metrics=cnc_metrics,
        sensitivity_metrics=sensitivity_metrics,
        campaign_manifest=args.campaign_manifest,
    )
    print(
        f"families={','.join(manifest['families'])} "
        f"output_set_sha256={manifest['output_set_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
