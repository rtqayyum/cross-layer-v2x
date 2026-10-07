#!/usr/bin/env python3
"""Transform infrastructure-LiDAR predictions into V2X-Seq vehicle coordinates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


def matrix(value: dict, nested: bool = False) -> np.ndarray:
    if nested:
        value = value["transform"]
    result = np.eye(4)
    result[:3, :3] = np.asarray(value["rotation"], dtype=float)
    result[:3, 3] = np.asarray(value["translation"], dtype=float).reshape(3)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = pd.read_parquet(args.pairs)[["pair_id", "vehicle_frame_id", "infrastructure_frame_id"]]
    predictions = pd.read_parquet(args.predictions).merge(pairs, on="pair_id", validate="many_to_one")
    chunks: list[pd.DataFrame] = []
    prefix = "V2X-Seq-SPD"
    with ZipFile(args.archive) as zipped:
        def read(name: str) -> dict:
            return json.loads(zipped.read(name))
        for (vehicle, infrastructure), frame in predictions.groupby(
            ["vehicle_frame_id", "infrastructure_frame_id"], sort=False
        ):
            inf_to_world = matrix(read(f"{prefix}/infrastructure-side/calib/virtuallidar_to_world/{infrastructure}.json"))
            lidar_to_novatel = matrix(read(f"{prefix}/vehicle-side/calib/lidar_to_novatel/{vehicle}.json"), nested=True)
            novatel_to_world = matrix(read(f"{prefix}/vehicle-side/calib/novatel_to_world/{vehicle}.json"))
            inf_to_vehicle = np.linalg.inv(novatel_to_world @ lidar_to_novatel) @ inf_to_world
            xyz = frame[["x_m", "y_m", "z_m"]].to_numpy(float)
            transformed = (inf_to_vehicle @ np.column_stack((xyz, np.ones(len(xyz)))).T).T[:, :3]
            aligned = frame.copy()
            aligned[["x_m", "y_m", "z_m"]] = transformed
            if "rotation_rad" in aligned:
                headings=np.column_stack((np.cos(aligned.rotation_rad),np.sin(aligned.rotation_rad),np.zeros(len(aligned))))
                rotated=headings@inf_to_vehicle[:3,:3].T
                aligned["rotation_rad"]=np.arctan2(rotated[:,1],rotated[:,0])
            aligned["coordinate_frame"] = "V2X_SEQ_VEHICLE_LIDAR"
            aligned["coordinate_transform"] = "INF_LIDAR_TO_WORLD_TO_VEHICLE_LIDAR"
            chunks.append(aligned.drop(columns=["vehicle_frame_id", "infrastructure_frame_id"]))
    result = pd.concat(chunks, ignore_index=True).sort_values(["pair_id", "prediction_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    receipt = {"status": "PASS", "rows": len(result), "coordinate_frame": "V2X_SEQ_VEHICLE_LIDAR",
               "source_prediction_sha256": sha256(args.predictions), "calibration_archive_sha256": sha256(args.archive),
               "output_sha256": sha256(args.output), "output": str(args.output.resolve())}
    args.output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
