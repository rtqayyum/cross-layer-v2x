#!/usr/bin/env python3
"""Freeze a moderate, capture-group-disjoint dataset from real source indexes."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise SystemExit(f"required receipt is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, action="append", required=True)
    parser.add_argument("--license-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--target-frames", type=int, default=6500)
    parser.add_argument("--frames-min", type=int, default=5000)
    parser.add_argument("--frames-max", type=int, default=8000)
    parser.add_argument("--sequences-min", type=int, default=36)
    parser.add_argument("--sequences-max", type=int, default=48)
    parser.add_argument("--capture-groups-min", type=int, default=12)
    args = parser.parse_args()

    license_receipt = read_json(args.license_receipt.resolve())
    license_status = str(license_receipt.get("license_status"))
    if (
        license_status not in {"verified", "user_attested"}
        or license_receipt.get("research_use_permitted") is not True
    ):
        raise SystemExit("dataset license receipt does not permit research use")

    pair_frames: list[pd.DataFrame] = []
    object_frames: list[pd.DataFrame] = []
    source_manifests: list[dict[str, object]] = []
    for source_dir in [path.resolve() for path in args.source_index]:
        manifest_path = source_dir / "source_manifest.json"
        pairs_path = source_dir / "cooperative_pairs.parquet"
        objects_path = source_dir / "ground_truth_objects.parquet"
        manifest = read_json(manifest_path)
        if manifest.get("synthetic_rows") != 0:
            raise SystemExit(f"source index does not assert zero synthetic rows: {source_dir}")
        if manifest.get("dataset_role") != "full_candidate":
            raise SystemExit(f"source index is not a full candidate: {source_dir}")
        source_license_status = str(manifest.get("license_status"))
        allowed_source_statuses = (
            {"verified"}
            if license_status == "verified"
            else {"verified", "pending_verification"}
        )
        if source_license_status not in allowed_source_statuses:
            raise SystemExit(f"source index license status is incompatible: {source_dir}")
        for path in (pairs_path, objects_path):
            expected = manifest["files"][path.name]["sha256"]
            observed = sha256_file(path)
            if observed != expected:
                raise SystemExit(f"source-index hash mismatch: {path}")
        pair_frames.append(pd.read_parquet(pairs_path))
        object_frames.append(pd.read_parquet(objects_path))
        source_manifests.append(manifest)

    pairs = pd.concat(pair_frames, ignore_index=True)
    objects = pd.concat(object_frames, ignore_index=True)
    if pairs["pair_id"].duplicated().any() or objects["ground_truth_id"].duplicated().any():
        raise SystemExit("duplicate source keys exist across indexes")
    if set(pairs["provenance_class"].unique()) != {"REAL_ROAD_CAPTURE"}:
        raise SystemExit("pair source contains non-real provenance")
    if set(objects["provenance_class"].unique()) != {"REAL_ROAD_CAPTURE"}:
        raise SystemExit("ground-truth source contains non-real provenance")

    summary = (
        pairs.groupby(["source_sequence_id", "capture_group_id"], as_index=False)
        .agg(frame_count=("pair_id", "count"))
    )
    summary["selection_key"] = summary.apply(
        lambda row: hashlib.sha256(
            f"{args.seed}:{row.source_sequence_id}:{row.capture_group_id}".encode()
        ).hexdigest(),
        axis=1,
    )
    summary = summary.sort_values("selection_key")
    selected_sequences: list[str] = []
    selected_frames = 0
    for row in summary.itertuples(index=False):
        if len(selected_sequences) >= args.sequences_max:
            break
        if (
            len(selected_sequences) >= args.sequences_min
            and selected_frames >= args.target_frames
        ):
            break
        selected_sequences.append(str(row.source_sequence_id))
        selected_frames += int(row.frame_count)
    if not args.sequences_min <= len(selected_sequences) <= args.sequences_max:
        raise SystemExit(f"selected sequence count is invalid: {len(selected_sequences)}")
    if not args.frames_min <= selected_frames <= args.frames_max:
        raise SystemExit(f"selected frame count is invalid: {selected_frames}")

    selected_pairs = pairs[
        pairs["source_sequence_id"].astype(str).isin(selected_sequences)
    ].copy()
    groups = sorted(selected_pairs["capture_group_id"].astype(str).unique())
    if len(groups) < args.capture_groups_min:
        raise SystemExit(f"only {len(groups)} capture groups survived selection")
    rng = random.Random(args.seed)
    rng.shuffle(groups)
    train_count = max(1, round(len(groups) * 0.60))
    validation_count = max(1, round(len(groups) * 0.20))
    if train_count + validation_count >= len(groups):
        validation_count = 1
        train_count = len(groups) - 2
    group_split = {
        group: (
            "train"
            if index < train_count
            else "validation"
            if index < train_count + validation_count
            else "test"
        )
        for index, group in enumerate(groups)
    }
    selected_pairs["split"] = selected_pairs["capture_group_id"].map(group_split)
    if selected_pairs.groupby("capture_group_id")["split"].nunique().max() != 1:
        raise SystemExit("capture-group split leakage detected")
    selected_pair_ids = set(selected_pairs["pair_id"])
    selected_objects = objects[objects["pair_id"].isin(selected_pair_ids)].copy()

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty frozen dataset: {output}")
    output.mkdir(parents=True, exist_ok=True)
    pairs_path = output / "cooperative_pairs.parquet"
    objects_path = output / "ground_truth_objects.parquet"
    selected_pairs.to_parquet(pairs_path, index=False)
    selected_objects.to_parquet(objects_path, index=False)
    manifest = {
        "capture_group_count": len(groups),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            pairs_path.name: {"rows": len(selected_pairs), "sha256": sha256_file(pairs_path)},
            objects_path.name: {"rows": len(selected_objects), "sha256": sha256_file(objects_path)},
        },
        "license_receipt_path": str(args.license_receipt.resolve()),
        "license_receipt_sha256": sha256_file(args.license_receipt.resolve()),
        "license_status": license_status,
        "paper_ranking_eligible": license_status == "verified",
        "publication_gate": (
            "PASS" if license_status == "verified" else "BLOCKED_PENDING_DOCUMENTARY_LICENSE"
        ),
        "selection": {
            "capture_groups_min": args.capture_groups_min,
            "frames_max": args.frames_max,
            "frames_min": args.frames_min,
            "outcome_blind": True,
            "seed": args.seed,
            "sequences_max": args.sequences_max,
            "sequences_min": args.sequences_min,
            "target_frames": args.target_frames,
        },
        "sequence_count": len(selected_sequences),
        "source_manifest_sha256": [
            hashlib.sha256(
                json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            for value in source_manifests
        ],
        "split_capture_group_counts": selected_pairs.groupby("split")[
            "capture_group_id"
        ].nunique().to_dict(),
        "synthetic_rows": 0,
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
