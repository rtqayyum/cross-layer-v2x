#!/usr/bin/env python3
"""Report the schema of the local dataset archives without exporting raw records.

Run this first on the machine that holds the data.  The output is a JSON file
with directory patterns, file counts, JSON keys, CSV column names and numeric
summaries (min/median/max).  It contains no label rows, point clouds or packet
records, so it can be committed and used to adapt the loaders.

Usage:
    python scripts/inspect_data.py \
        --v2x-seq /path/V2X-Seq-SPD.zip \
        --v2x-infra /path/V2X-Seq-SPD-infrastructure-side-velodyne.zip \
        --see-v2x /path/emulating_applications.tar.gz \
        --output results/inspection/data_schema.json
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import re
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DIGITS = re.compile(r"\d+")


def pattern(name: str) -> str:
    """Collapse numeric path components so per-frame files group together."""
    return DIGITS.sub("#", name)


def keys_of(value, depth: int = 0) -> object:
    if depth > 3:
        return type(value).__name__
    if isinstance(value, dict):
        return {k: keys_of(v, depth + 1) for k, v in list(value.items())[:40]}
    if isinstance(value, list):
        return [keys_of(value[0], depth + 1), f"len={len(value)}"] if value else []
    return type(value).__name__


def summarize_numeric(series: pd.Series) -> dict:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"non_numeric": True, "distinct": int(series.nunique())}
    return {
        "n": int(len(values)),
        "min": float(values.min()),
        "p10": float(values.quantile(0.1)),
        "median": float(values.median()),
        "p90": float(values.quantile(0.9)),
        "max": float(values.max()),
        "distinct": int(values.nunique()),
    }


def inspect_v2x_seq(path: Path) -> dict:
    report: dict = {"archive": path.name}
    with zipfile.ZipFile(path) as zipped:
        names = zipped.namelist()
        counts = collections.Counter(pattern(n) for n in names if not n.endswith("/"))
        report["member_patterns"] = dict(counts.most_common(80))
        jsons = [n for n in names if n.endswith(".json")]
        by_dir = collections.defaultdict(list)
        for name in jsons:
            by_dir[str(Path(name).parent)].append(name)
        samples = {}
        for directory, members in sorted(by_dir.items()):
            first = sorted(members)[0]
            try:
                samples[pattern(directory)] = {
                    "files": len(members),
                    "schema": keys_of(json.loads(zipped.read(first))),
                }
            except Exception as error:  # noqa: BLE001 - report and continue
                samples[pattern(directory)] = {"files": len(members), "error": str(error)}
        report["json_directories"] = samples

        # Value distributions that decide the method design (no raw rows).
        info = {n: json.loads(zipped.read(n)) for n in jsons if n.endswith("data_info.json")}
        report["data_info"] = {}
        for name, rows in info.items():
            frame = pd.DataFrame(rows)
            report["data_info"][name] = {
                "rows": len(frame),
                "columns": list(frame.columns),
                "numeric": {c: summarize_numeric(frame[c]) for c in frame.columns
                            if "timestamp" in c or c.endswith("_id") or "frame" in c},
                "categorical": {c: frame[c].astype(str).value_counts().head(15).to_dict()
                                for c in frame.columns
                                if frame[c].astype(str).nunique() <= 60},
            }
        label_stats: dict = {}
        for side in ("cooperative", "infrastructure-side", "vehicle-side"):
            members = [n for n in jsons if f"/{side}/label/" in n]
            if not members:
                continue
            fields = collections.Counter()
            types = collections.Counter()
            from_side = collections.Counter()
            occluded = collections.Counter()
            per_frame = []
            subdirs = collections.Counter(pattern(str(Path(n).parent)) for n in members)
            for name in members[:4000]:
                objects = json.loads(zipped.read(name))
                if isinstance(objects, dict):
                    objects = objects.get("objects", [objects])
                per_frame.append(len(objects))
                for obj in objects:
                    fields.update(obj.keys())
                    types[str(obj.get("type"))] += 1
                    from_side[str(obj.get("from_side"))] += 1
                    occluded[str(obj.get("occluded_state"))] += 1
            label_stats[side] = {
                "files": len(members),
                "subdirectories": dict(subdirs),
                "sampled_files": min(len(members), 4000),
                "fields": dict(fields),
                "types": dict(types.most_common(20)),
                "from_side": dict(from_side),
                "occluded_state": dict(occluded),
                "objects_per_frame": summarize_numeric(pd.Series(per_frame)),
            }
        report["labels"] = label_stats
    return report


def inspect_infra_lidar(path: Path) -> dict:
    with zipfile.ZipFile(path) as zipped:
        names = zipped.namelist()
    return {"archive": path.name, "members": len(names),
            "patterns": dict(collections.Counter(pattern(n) for n in names).most_common(20))}


def inspect_see_v2x(path: Path, max_csv: int = 400) -> dict:
    report: dict = {"archive": path.name}
    patterns = collections.Counter()
    csv_reports: dict = {}
    parameters: dict = {}
    seen = 0
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            patterns[pattern(member.name)] += 1
            if not member.name.endswith(".csv") or seen >= max_csv:
                continue
            key = pattern(member.name)
            if key in csv_reports and not member.name.endswith("parameters.csv"):
                csv_reports[key]["files"] += 1
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            raw = handle.read()
            seen += 1
            try:
                frame = pd.read_csv(io.BytesIO(raw), low_memory=False)
            except Exception as error:  # noqa: BLE001
                csv_reports[key] = {"files": 1, "error": str(error)}
                continue
            if member.name.endswith("parameters.csv"):
                # Configuration of the emulated application (rates, sizes, MCS).
                parameters[member.name] = {c: summarize_numeric(frame[c]) for c in frame.columns}
                parameters[member.name]["columns"] = list(frame.columns)
                continue
            time_cols = [c for c in frame.columns if "time" in c.lower()]
            duration = None
            if time_cols:
                t = pd.to_numeric(frame[time_cols[0]], errors="coerce").dropna()
                if len(t) > 1:
                    duration = float(t.max() - t.min())
            csv_reports[key] = {
                "files": 1,
                "rows_first_file": len(frame),
                "columns": list(frame.columns),
                "first_time_column_span": duration,
                "summary": {c: summarize_numeric(frame[c]) for c in frame.columns[:40]},
            }
    report["member_patterns"] = dict(patterns.most_common(120))
    report["csv_schemas"] = csv_reports
    report["parameters_files"] = parameters
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v2x-seq", type=Path)
    parser.add_argument("--v2x-infra", type=Path)
    parser.add_argument("--v2x-vehicle", type=Path, help="optional vehicle-side Velodyne zip")
    parser.add_argument("--see-v2x", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report: dict = {"numpy": np.__version__, "pandas": pd.__version__}
    if args.v2x_seq:
        report["v2x_seq"] = inspect_v2x_seq(args.v2x_seq)
    if args.v2x_infra:
        report["v2x_infra_lidar"] = inspect_infra_lidar(args.v2x_infra)
    if args.v2x_vehicle:
        report["v2x_vehicle_lidar"] = inspect_infra_lidar(args.v2x_vehicle)
    if args.see_v2x:
        report["see_v2x"] = inspect_see_v2x(args.see_v2x)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
