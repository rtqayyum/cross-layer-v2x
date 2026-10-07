#!/usr/bin/env python3
"""Run a causal geometric detector on real V2X-Seq LiDAR frames in a ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cross_layer_v2x.pcd import read_binary_compressed_pcd  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def member_by_basename(zipped: ZipFile) -> dict[str, str]:
    return {
        Path(name).name: name
        for name in zipped.namelist()
        if name.lower().endswith(".pcd")
    }


def detect(payload: bytes, voxel_m: float, eps_m: float) -> list[dict[str, float]]:
    fields = read_binary_compressed_pcd(payload)
    xyz = np.column_stack((fields["x"], fields["y"], fields["z"])).astype(np.float32)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    roi = (
        (xyz[:, 0] >= -20.0)
        & (xyz[:, 0] <= 100.0)
        & (np.abs(xyz[:, 1]) <= 50.0)
        & (xyz[:, 2] >= -5.0)
        & (xyz[:, 2] <= 5.0)
    )
    xyz = xyz[roi]
    if len(xyz) == 0:
        return []
    voxels = np.floor(xyz / voxel_m).astype(np.int32)
    _, indices = np.unique(voxels, axis=0, return_index=True)
    xyz = xyz[np.sort(indices)]
    ground = float(np.percentile(xyz[:, 2], 12.0))
    xyz = xyz[xyz[:, 2] > ground + 0.25]
    if len(xyz) < 12:
        return []
    scaled = xyz.copy()
    scaled[:, 2] *= 0.6
    labels = DBSCAN(eps=eps_m, min_samples=6, n_jobs=1).fit_predict(scaled)
    detections: list[dict[str, float]] = []
    for label in sorted(set(labels) - {-1}):
        points = xyz[labels == label]
        if not 8 <= len(points) <= 2500:
            continue
        # Robust PCA-oriented bounding box (OBB): frame-local, causal and fit
        # only to the measured cluster points. This is a standard classical
        # LiDAR baseline and does not use labels or future frames.
        xy=points[:,:2];xy_center=np.median(xy,axis=0);cov=np.cov(xy-xy_center,rowvar=False);values,vectors=np.linalg.eigh(cov);axis=vectors[:,int(np.argmax(values))]
        if axis[0]<0:axis=-axis
        lateral=np.array([-axis[1],axis[0]]);projection=np.column_stack(((xy-xy_center)@axis,(xy-xy_center)@lateral));plo=np.percentile(projection,2,axis=0);phi=np.percentile(projection,98,axis=0);zlo,zhi=np.percentile(points[:,2],[2,98]);dimensions=np.array([phi[0]-plo[0],phi[1]-plo[1],zhi-zlo]);local_center=(plo+phi)/2;center_xy=xy_center+local_center[0]*axis+local_center[1]*lateral
        if not (
            0.25 <= dimensions[0] <= 15.0
            and 0.25 <= dimensions[1] <= 8.0
            and 0.25 <= dimensions[2] <= 5.0
        ):
            continue
        center = np.array([center_xy[0],center_xy[1],(zlo+zhi)/2]);yaw=float(math.atan2(axis[1],axis[0]))
        length,width,height=map(float,dimensions)
        if width>length:length,width=width,length;yaw=((yaw+math.pi/2+math.pi)%(2*math.pi))-math.pi
        if 1.0<=height<=2.5 and length<=1.5 and width<=1.2: object_class='Pedestrian'
        elif .8<=height<=2.5 and 1.0<=length<=3.2 and width<=1.5: object_class='Cyclist'
        elif .8<=height<=3.2 and 2.5<=length<=7.5 and 1.1<=width<=3.2: object_class='Car'
        else: object_class='unclassified_obstacle'
        distance = float(np.linalg.norm(center[:2]))
        corridor = math.exp(-abs(float(center[1])) / 4.0)
        proximity = max(0.0, 1.0 - distance / 80.0)
        confidence = min(0.99, 0.35 + math.log1p(len(points)) / 10.0)
        occlusion = min(1.0, distance / 100.0 + 1.0 / math.sqrt(len(points)))
        risk = min(1.0, 0.65 * proximity * corridor + 0.35 * confidence)
        detections.append(
            {
                "x_m": float(center[0]),
                "y_m": float(center[1]),
                "z_m": float(center[2]),
                "length_m": length,
                "width_m": width,
                "height_m": height,
                "rotation_rad": yaw,
                "object_class": object_class,
                "confidence": confidence,
                "occlusion_probability": occlusion,
                "risk_score": risk,
            }
        )
    return detections


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--role", choices=("vehicle", "infrastructure"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--voxel-m", type=float, default=0.35)
    parser.add_argument("--eps-m", type=float, default=0.85)
    args = parser.parse_args()

    pairs = pd.read_parquet(args.pairs).sort_values(
        ["source_sequence_id", "vehicle_pointcloud_timestamp_us", "pair_id"]
    )
    if args.max_pairs is not None:
        pairs = pairs.head(args.max_pairs)
    reference_column = f"{args.role}_pointcloud_ref"
    timestamp_column = f"{args.role}_pointcloud_timestamp_us"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = Path(str(args.output) + ".shards")
    if shard_dir.exists() and not args.resume:
        raise SystemExit(f"checkpoint directory exists; rerun with --resume: {shard_dir}")
    shard_dir.mkdir(parents=True, exist_ok=True)
    completed_sequences = 0
    with ZipFile(args.archive) as zipped:
        members = member_by_basename(zipped)
        for sequence_id, sequence_pairs in pairs.groupby("source_sequence_id", sort=True):
            shard = shard_dir / f"{sequence_id}.jsonl"
            if shard.is_file() and args.resume:
                completed_sequences += 1
                print(f"RESUME sequence={sequence_id} status=already_complete", flush=True)
                continue
            temporary_shard = shard.with_suffix(".jsonl.incomplete")
            if temporary_shard.exists():
                temporary_shard.unlink()
            sequence_rows = 0
            with temporary_shard.open("w", encoding="utf-8") as handle:
                for pair in sequence_pairs.itertuples(index=False):
                    source_ref = str(getattr(pair, reference_column))
                    member = members.get(Path(source_ref).name)
                    if member is None:
                        raise SystemExit(f"point cloud is absent from archive: {source_ref}")
                    detections = detect(zipped.read(member), args.voxel_m, args.eps_m)
                    timestamp_us = int(getattr(pair, timestamp_column))
                    for index, detection in enumerate(detections):
                        row = {
                            "prediction_id": f"geom:{args.role}:{pair.pair_id}:{index:04d}",
                            "pair_id": str(pair.pair_id),
                            "track_id": f"frame-local-{index:04d}",
                            "object_class": detection.pop("object_class"),
                            "information_age_ms": 0.0,
                            "deadline_ms": 50.0 + 50.0 * (1.0 - detection["risk_score"]),
                            **detection,
                        }
                        handle.write(
                            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                        )
                        sequence_rows += 1
                handle.flush()
            temporary_shard.replace(shard)
            completed_sequences += 1
            print(
                f"PASS sequence={sequence_id} pairs={len(sequence_pairs)} "
                f"detections={sequence_rows}",
                flush=True,
            )

    ordered_shards = [
        shard_dir / f"{sequence_id}.jsonl"
        for sequence_id in sorted(pairs["source_sequence_id"].astype(str).unique())
    ]
    missing_shards = [str(path) for path in ordered_shards if not path.is_file()]
    if missing_shards:
        raise SystemExit(f"missing completed sequence shards: {missing_shards[:3]}")
    temporary_output = args.output.with_suffix(args.output.suffix + ".incomplete")
    row_count = 0
    with temporary_output.open("w", encoding="utf-8") as combined:
        for shard in ordered_shards:
            with shard.open(encoding="utf-8") as source:
                for line in source:
                    combined.write(line)
                    row_count += 1
    temporary_output.replace(args.output)

    code_hash = hashlib.sha256(
        Path(__file__).read_bytes() + (ROOT / "src/cross_layer_v2x/pcd.py").read_bytes()
    ).hexdigest()
    receipt = {
        "code_commit": "WORKTREE_GEOMETRIC_DETECTOR",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inference_input_sha256": sha256_file(args.pairs),
        "model_id": "causal-pca-obb-lidar-v2",
        "model_sha256": code_hash,
        "provenance_class": "MEASURED_PERCEPTION_OUTPUT",
        "pseudo_labels": False,
        "uses_future_frames": False,
        "uses_ground_truth_at_inference": False,
        "role": args.role,
        "archive_sha256": sha256_file(args.archive),
        "limitations": "Deterministic PCA-OBB geometric detector; frame-local IDs; dimension-rule semantic classes are not learned detector classes.",
    }
    args.receipt_output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "pairs": len(pairs),
                "sequences": completed_sequences,
                "detections": row_count,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
