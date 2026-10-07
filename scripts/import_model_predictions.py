#!/usr/bin/env python3
"""Validate and import causal perception-model JSONL outputs into Parquet."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


MODEL_FIELDS = {
    "prediction_id",
    "pair_id",
    "track_id",
    "object_class",
    "confidence",
    "occlusion_probability",
    "risk_score",
    "information_age_ms",
    "deadline_ms",
    "x_m",
    "y_m",
    "z_m",
    "length_m",
    "width_m",
    "height_m",
    "rotation_rad",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--frozen-dataset-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.receipt.is_file():
        raise SystemExit(f"required perception receipt is missing: {args.receipt}")
    if not args.jsonl.is_file():
        raise SystemExit(f"official model prediction JSONL is missing: {args.jsonl}")
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    if (
        receipt.get("uses_ground_truth_at_inference") is not False
        or receipt.get("uses_future_frames") is not False
        or receipt.get("pseudo_labels") is not False
        or not receipt.get("model_sha256")
        or not receipt.get("model_id")
    ):
        raise SystemExit("model receipt does not establish causal measured inference")
    rows: list[dict[str, object]] = []
    with args.jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise SystemExit(f"line {line_number} is not a JSON object")
            missing = MODEL_FIELDS - set(value)
            forbidden = [name for name in value if "ground_truth" in name.lower()]
            if missing or forbidden:
                raise SystemExit(
                    f"line {line_number}: missing={sorted(missing)}, forbidden={forbidden}"
                )
            rows.append(value)
    if not rows:
        raise SystemExit("prediction JSONL is empty")
    predictions = pd.DataFrame(rows)
    if predictions["prediction_id"].duplicated().any():
        raise SystemExit("prediction_id values are not unique")

    pairs = pd.read_parquet(args.frozen_dataset_dir / "cooperative_pairs.parquet")
    metadata = pairs[
        [
            "pair_id",
            "source_sequence_id",
            "capture_group_id",
            "vehicle_pointcloud_timestamp_us",
        ]
    ].rename(columns={"vehicle_pointcloud_timestamp_us": "capture_timestamp_us"})
    predictions = predictions.merge(metadata, on="pair_id", how="left", validate="many_to_one")
    if predictions["source_sequence_id"].isna().any():
        raise SystemExit("predictions reference pair IDs outside the frozen dataset")
    for field in ("confidence", "occlusion_probability", "risk_score"):
        values = pd.to_numeric(predictions[field], errors="raise")
        if not values.map(math.isfinite).all():
            raise SystemExit(f"{field} must be finite")
        if ((values < 0) | (values > 1)).any():
            raise SystemExit(f"{field} must be in [0, 1]")
    if (pd.to_numeric(predictions["information_age_ms"], errors="raise") < 0).any():
        raise SystemExit("information_age_ms must be nonnegative")
    if (pd.to_numeric(predictions["deadline_ms"], errors="raise") <= 0).any():
        raise SystemExit("deadline_ms must be positive")
    for field in ("x_m", "y_m", "z_m", "length_m", "width_m", "height_m"):
        values = pd.to_numeric(predictions[field], errors="raise")
        if not values.map(math.isfinite).all():
            raise SystemExit(f"{field} must be finite")
    predictions["model_id"] = str(receipt["model_id"])
    predictions["model_sha256"] = str(receipt["model_sha256"])
    predictions["provenance_class"] = "MEASURED_PERCEPTION_OUTPUT"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.output, index=False)
    print(
        json.dumps(
            {
                "status": "PASS",
                "rows": len(predictions),
                "pairs": int(predictions["pair_id"].nunique()),
                "model_id": receipt["model_id"],
                "output": str(args.output),
                "synthetic_rows": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
