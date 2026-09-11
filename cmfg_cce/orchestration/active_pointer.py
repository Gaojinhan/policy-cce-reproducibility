from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cmfg_cce.orchestration.manifest import (
    CampaignManifest,
    ManifestError,
    SourceManifest,
    atomic_json,
    canonical_json,
    sha256_bytes,
    sha256_file,
)


SCHEMA_VERSION = "revision_full_v1_active_campaign_v1"


def _selection_contract(
    campaign: CampaignManifest,
    selection_path: Path,
) -> dict[str, Any]:
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "revision_full_v1_global_mwu_selection_v2":
        raise ManifestError("Formal active pointer requires the hyperparameter-only MWU selection.")
    file_sha = sha256_file(selection_path)
    unsigned = dict(payload)
    selection_sha = str(unsigned.pop("selection_sha256", ""))
    if selection_sha != sha256_bytes(canonical_json(unsigned)):
        raise ManifestError("MWU selection has an invalid signed payload hash.")
    selected = dict(payload.get("selected_config", {}))
    if "formal_rounds" in selected or "formal_rounds_rule" in selected:
        raise ManifestError("Formal active pointer cannot freeze a global MWU round budget.")
    contract = {
        "selection_sha256": selection_sha,
        "selection_file_sha256": file_sha,
        "mwu_config_id": str(selected.get("mwu_config_id", "")),
    }
    frozen_contracts = {
        canonical_json(dict(job.metadata.get("mwu_selection", {})))
        for job in campaign.jobs
    }
    if frozen_contracts != {canonical_json(contract)}:
        raise ManifestError("Formal campaign jobs do not freeze this MWU selection.")
    return {
        **contract,
        "object_key": f"bootstrap/selections/{file_sha}.json",
    }


def build_active_pointer(
    *,
    phase: str,
    campaign_path: Path,
    source_manifest_path: Path,
    selection_path: Path | None = None,
) -> dict[str, Any]:
    if phase not in {"calibration", "formal"}:
        raise ValueError("Active phase must be calibration or formal.")
    campaign = CampaignManifest.load(campaign_path)
    source_payload = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source = SourceManifest.from_payload(source_payload)
    if source.source_sha256 != campaign.source_sha256:
        raise ManifestError("Active campaign and source manifest do not match.")
    if campaign.campaign_id != f"revision-full-v1-{phase}":
        raise ManifestError(
            f"Campaign {campaign.campaign_id!r} is not the requested {phase!r} phase."
        )
    selection: dict[str, Any] | None = None
    if phase == "formal":
        if selection_path is None:
            raise ManifestError("Formal active pointer requires an MWU selection.")
        selection = _selection_contract(campaign, Path(selection_path))
    elif selection_path is not None:
        raise ManifestError("Calibration active pointer must not include an MWU selection.")

    campaign_file_sha = sha256_file(campaign_path)
    source_file_sha = sha256_file(source_manifest_path)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "campaign_id": campaign.campaign_id,
        "campaign_sha256": campaign.campaign_sha256,
        "campaign_file_sha256": campaign_file_sha,
        "campaign_object_key": (
            f"bootstrap/manifests/{phase}/{campaign.campaign_sha256}.json"
        ),
        "source_sha256": source.source_sha256,
        "source_file_sha256": source_file_sha,
        "source_object_key": f"bootstrap/sources/{source.source_sha256}.json",
        "image_by_platform": dict(campaign.image_by_platform),
        "selection": selection,
    }
    payload["pointer_sha256"] = sha256_bytes(canonical_json(payload))
    return payload


def validate_active_pointer(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    observed_hash = str(normalized.pop("pointer_sha256", ""))
    if normalized.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError("Unsupported active-campaign pointer schema.")
    if observed_hash != sha256_bytes(canonical_json(normalized)):
        raise ManifestError("Active-campaign pointer hash mismatch.")
    for name in (
        "campaign_sha256",
        "campaign_file_sha256",
        "source_sha256",
        "source_file_sha256",
    ):
        value = str(normalized.get(name, ""))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ManifestError(f"Active-campaign pointer has invalid {name}.")
    if normalized.get("phase") not in {"calibration", "formal"}:
        raise ManifestError("Active-campaign pointer has an invalid phase.")
    if normalized.get("phase") == "formal" and not isinstance(
        normalized.get("selection"), Mapping
    ):
        raise ManifestError("Formal active-campaign pointer omits the MWU selection.")
    normalized["pointer_sha256"] = observed_hash
    return normalized


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a hash-verified active campaign pointer.")
    parser.add_argument("--phase", required=True, choices=("calibration", "formal"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--mwu-selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = build_active_pointer(
        phase=args.phase,
        campaign_path=args.campaign,
        source_manifest_path=args.source_manifest,
        selection_path=args.mwu_selection,
    )
    validate_active_pointer(payload)
    atomic_json(args.output, payload)
    print(
        f"active_phase={payload['phase']} campaign_sha256={payload['campaign_sha256']} "
        f"pointer_sha256={payload['pointer_sha256']}"
    )


if __name__ == "__main__":
    main()
