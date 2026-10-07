#!/usr/bin/env python3
"""Build compact Parquet indexes from a real V2X-Seq SPD archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


def load_json(zipped: ZipFile, member: str) -> object:
    with zipped.open(member) as handle:
        return json.load(handle)


def only_member(names: list[str], suffix: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise SystemExit(f"expected one {suffix!r}, found {len(matches)}")
    return matches[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value(mapping: object, key: str) -> object:
    return mapping.get(key) if isinstance(mapping, dict) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-id", default="v2x_seq_spd_example_real")
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument(
        "--dataset-role",
        choices=("ingestion_smoke", "full_candidate"),
        default="ingestion_smoke",
    )
    parser.add_argument(
        "--license-status",
        choices=("verified", "pending_verification"),
        default="pending_verification",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    archive = args.archive.resolve()
    if not archive.is_file():
        raise SystemExit(f"archive does not exist: {archive}")
    observed_sha256 = sha256_file(archive)
    if observed_sha256 != args.source_sha256:
        raise SystemExit(
            f"source SHA-256 mismatch: expected {args.source_sha256}, "
            f"observed {observed_sha256}"
        )

    root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or root / "data" / "interim" / args.source_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(archive) as zipped:
        names = zipped.namelist()
        cooperative_member = only_member(names, "/cooperative/data_info.json")
        infrastructure_member = only_member(
            names, "/infrastructure-side/data_info.json"
        )
        vehicle_member = only_member(names, "/vehicle-side/data_info.json")
        cooperative_rows = load_json(zipped, cooperative_member)
        infrastructure_rows = load_json(zipped, infrastructure_member)
        vehicle_rows = load_json(zipped, vehicle_member)
        if not all(
            isinstance(rows, list)
            for rows in (cooperative_rows, infrastructure_rows, vehicle_rows)
        ):
            raise SystemExit("source data_info files must contain JSON arrays")

        vehicle_by_frame = {
            str(row["frame_id"]): row
            for row in vehicle_rows
            if isinstance(row, dict) and row.get("frame_id") is not None
        }
        infrastructure_by_frame = {
            str(row["frame_id"]): row
            for row in infrastructure_rows
            if isinstance(row, dict) and row.get("frame_id") is not None
        }

        cooperative_labels = {
            Path(name).stem: name
            for name in names
            if "/cooperative/label/" in name and name.endswith(".json")
        }
        pair_records: list[dict[str, object]] = []
        object_records: list[dict[str, object]] = []
        for pair in cooperative_rows:
            if not isinstance(pair, dict):
                raise SystemExit("cooperative data_info contains a non-object row")
            vehicle_frame = str(pair["vehicle_frame"])
            infrastructure_frame = str(pair["infrastructure_frame"])
            vehicle = vehicle_by_frame.get(vehicle_frame)
            infrastructure = infrastructure_by_frame.get(infrastructure_frame)
            if vehicle is None or infrastructure is None:
                raise SystemExit(
                    f"pair references missing frame: {vehicle_frame}/{infrastructure_frame}"
                )
            sequence_id = str(pair["vehicle_sequence"])
            if sequence_id != str(pair["infrastructure_sequence"]):
                raise SystemExit("vehicle/infrastructure sequence mismatch")
            vehicle_intersection = str(vehicle.get("intersection_loc") or "").strip()
            infrastructure_intersection = str(
                infrastructure.get("intersection_loc") or ""
            ).strip()
            if (
                vehicle_intersection
                and infrastructure_intersection
                and vehicle_intersection != infrastructure_intersection
            ):
                raise SystemExit("vehicle/infrastructure intersection mismatch")
            intersection_id = vehicle_intersection or infrastructure_intersection
            if not intersection_id:
                raise SystemExit("pair has no source-native intersection identifier")

            vehicle_timestamp_us = int(vehicle["pointcloud_timestamp"])
            infrastructure_timestamp_us = int(infrastructure["pointcloud_timestamp"])
            pair_id = (
                f"{args.source_id}:{sequence_id}:{vehicle_frame}:{infrastructure_frame}"
            )
            label_member = cooperative_labels.get(vehicle_frame)
            if label_member is None:
                raise SystemExit(f"missing cooperative label for {vehicle_frame}")
            objects = load_json(zipped, label_member)
            if not isinstance(objects, list):
                raise SystemExit(f"cooperative label is not a list: {label_member}")

            pair_records.append(
                {
                    "pair_id": pair_id,
                    "source_id": args.source_id,
                    "source_sequence_id": sequence_id,
                    "intersection_id": intersection_id,
                    "capture_group_id": f"{intersection_id}:{sequence_id}",
                    "vehicle_frame_id": vehicle_frame,
                    "infrastructure_frame_id": infrastructure_frame,
                    "vehicle_pointcloud_timestamp_us": vehicle_timestamp_us,
                    "infrastructure_pointcloud_timestamp_us": infrastructure_timestamp_us,
                    "absolute_sync_delta_ms": abs(
                        vehicle_timestamp_us - infrastructure_timestamp_us
                    )
                    / 1_000.0,
                    "system_error_delta_x_m": value(
                        pair.get("system_error_offset"), "delta_x"
                    ),
                    "system_error_delta_y_m": value(
                        pair.get("system_error_offset"), "delta_y"
                    ),
                    "vehicle_image_ref": vehicle.get("image_path"),
                    "vehicle_pointcloud_ref": vehicle.get("pointcloud_path"),
                    "infrastructure_image_ref": infrastructure.get("image_path"),
                    "infrastructure_pointcloud_ref": infrastructure.get(
                        "pointcloud_path"
                    ),
                    "cooperative_annotation_ref": label_member,
                    "source_sha256": observed_sha256,
                    "provenance_class": "REAL_ROAD_CAPTURE",
                }
            )

            for index, obj in enumerate(objects):
                if not isinstance(obj, dict):
                    raise SystemExit(f"non-object annotation in {label_member}")
                dimensions = obj.get("3d_dimensions")
                location = obj.get("3d_location")
                object_records.append(
                    {
                        "ground_truth_id": f"{pair_id}:{index:03d}",
                        "pair_id": pair_id,
                        "token": obj.get("token"),
                        "track_id": obj.get("track_id"),
                        "object_class": obj.get("type"),
                        "occluded_state": obj.get("occluded_state"),
                        "truncated_state": obj.get("truncated_state"),
                        "length_m": value(dimensions, "l"),
                        "width_m": value(dimensions, "w"),
                        "height_m": value(dimensions, "h"),
                        "x_m": value(location, "x"),
                        "y_m": value(location, "y"),
                        "z_m": value(location, "z"),
                        "rotation_rad": obj.get("rotation"),
                        "from_side": obj.get("from_side"),
                        "source_annotation_ref": label_member,
                        "usage_boundary": "OFFLINE_GROUND_TRUTH_ONLY",
                        "provenance_class": "REAL_ROAD_CAPTURE",
                    }
                )

    capture_group_count = len(
        {record["capture_group_id"] for record in pair_records}
    )
    if args.dataset_role == "full_candidate" and capture_group_count < 12:
        raise SystemExit(
            "refusing to label this archive as a full candidate: "
            f"only {capture_group_count} capture group(s) were found; at least 12 are required"
        )
    pairs_path = output_dir / "cooperative_pairs.parquet"
    objects_path = output_dir / "ground_truth_objects.parquet"
    pd.DataFrame(pair_records).to_parquet(pairs_path, index=False)
    pd.DataFrame(object_records).to_parquet(objects_path, index=False)
    ranking_reasons: list[str] = []
    if args.dataset_role != "full_candidate":
        ranking_reasons.append("ingestion smoke source only")
    if args.license_status != "verified":
        ranking_reasons.append("official dataset license receipt pending")
    if capture_group_count < 12:
        ranking_reasons.append("fewer than 12 capture groups")
    manifest = {
        "archive_path": str(archive),
        "build_time_utc": datetime.now(timezone.utc).isoformat(),
        "capture_group_count": capture_group_count,
        "dataset_role": args.dataset_role,
        "files": {
            pairs_path.name: {
                "rows": len(pair_records),
                "sha256": sha256_file(pairs_path),
            },
            objects_path.name: {
                "rows": len(object_records),
                "sha256": sha256_file(objects_path),
            },
        },
        "license_status": args.license_status,
        "paper_ranking_eligible": not ranking_reasons,
        "paper_ranking_ineligibility_reasons": ranking_reasons,
        "provenance_class": "DERIVED_FROM_REAL_CAPTURE",
        "source_id": args.source_id,
        "source_sha256": observed_sha256,
        "synthetic_rows": 0,
        "usage_boundary": (
            "Ground-truth objects are offline evaluation labels and must never be "
            "available to a controller at decision time."
        ),
    }
    manifest_path = output_dir / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
