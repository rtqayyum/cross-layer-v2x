#!/usr/bin/env python3
"""Stage 1: build the V2X-Seq-SPD scene tables and SEE-V2X service windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from certisched import seev2x, v2xseq


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v2x-seq", type=Path, required=True)
    ap.add_argument("--see-v2x", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/processed/v2"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tables = v2xseq.load(a.v2x_seq)
    for name, frame in tables.items():
        frame.to_parquet(a.out / f"{name}.parquet", index=False)
    links = seev2x.load(a.see_v2x)
    links.to_parquet(a.out / "links.parquet", index=False)
    gt = tables["gt"]
    summary = {
        "sequences": int(tables["rsu_objects"].sequence.nunique()),
        "infrastructure_frames": int(tables["rsu_objects"].infrastructure_frame.nunique()),
        "vehicle_frames": int(gt.vehicle_frame.nunique()),
        "rsu_objects_per_frame": float(tables["rsu_objects"].groupby("infrastructure_frame").size().mean()),
        "gt_objects": int(len(gt)), "gt_hidden": int(gt.hidden.sum()), "gt_critical": int(gt.critical.sum()),
        "from_side_values": gt.from_side.value_counts().to_dict(),
        "links": int(links.link_id.nunique()), "scenarios": int(links.scenario_id.nunique()),
        "windows": int(len(links)),
        "rate_bpms_quantiles": links.groupby("setting").rate_bpms.quantile([0.1, 0.5, 0.9]).unstack().to_dict(),
        "loss_mean": links.groupby("setting").loss.mean().to_dict(),
    }
    (a.out / "prepare_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
