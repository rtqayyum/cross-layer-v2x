#!/usr/bin/env python3
"""Normalize measured SEE-V2X receiver CSVs into an immutable replay table."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def clean(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def find_column(columns: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    for normalized, original in columns.items():
        if any(candidate in normalized for candidate in candidates):
            return original
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input_dir.resolve()
    if not source.exists():
        raise SystemExit(
            f"MEASURED TRACE GATE FAILED: {source} does not exist. Download the official SEE-V2X "
            "dataset and place its extracted contents there. Generated telemetry is prohibited."
        )
    files = sorted(source.rglob("*.csv"))
    rows: list[pd.DataFrame] = []
    receipts: list[dict[str, object]] = []
    for path in files:
        try:
            frame = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        columns = {clean(value): str(value) for value in frame.columns}
        latency = find_column(columns, ("latency_ms", "latency"))
        throughput = find_column(columns, ("avg_throughput_100ms", "throughput_100ms", "avg_throughput"))
        loss = find_column(columns, ("avg_packet_loss_100ms", "per_ue_loss_pct", "packet_loss", "loss_pct"))
        busy = find_column(columns, ("channel_busy_percentage", "channel_busy", "cbr"))
        packet_size = find_column(columns, ("packet_size", "size_bytes"))
        timestamp = find_column(columns, ("rx_timestamp_us", "timestamp_us", "tx_timestamp_us"))
        if latency is None or throughput is None:
            continue
        latency_values = pd.to_numeric(frame[latency], errors="coerce")
        # SEE-V2X interleaves packet-arrival rows (latency populated) with rolling
        # telemetry rows (throughput populated). Carry the nearest measured
        # packet latency onto the telemetry row; no generated latency is used.
        latency_values = latency_values.ffill().bfill()
        busy_values = (
            pd.to_numeric(frame[busy].astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce")
            if busy else pd.Series(np.nan, index=frame.index)
        )
        normalized = pd.DataFrame({
            "latency_ms": latency_values,
            "throughput_bps": pd.to_numeric(frame[throughput], errors="coerce"),
            "packet_loss": pd.to_numeric(frame[loss], errors="coerce") if loss else np.nan,
            "channel_busy_pct": busy_values,
            "packet_size_bytes": pd.to_numeric(frame[packet_size], errors="coerce") if packet_size else np.nan,
            "timestamp_us": pd.to_numeric(frame[timestamp], errors="coerce") if timestamp else np.nan,
        })
        normalized = normalized.replace([np.inf, -np.inf], np.nan).dropna(subset=["latency_ms", "throughput_bps"])
        if normalized.empty:
            continue
        if normalized["packet_loss"].dropna().median() > 1.0:
            normalized["packet_loss"] /= 100.0
        if normalized["channel_busy_pct"].dropna().median() <= 1.0:
            normalized["channel_busy_pct"] *= 100.0
        normalized["packet_loss"] = normalized["packet_loss"].fillna(0.0).clip(0.0, 1.0)
        normalized["channel_busy_pct"] = normalized["channel_busy_pct"].fillna(0.0).clip(0.0, 100.0)
        normalized["latency_ms"] = normalized["latency_ms"].clip(lower=0.0)
        normalized["throughput_bps"] = normalized["throughput_bps"].clip(lower=1.0)
        normalized["packet_size_bytes"] = normalized["packet_size_bytes"].fillna(1200).clip(lower=1)
        # Preserve measured time where available. The sample index remains the
        # deterministic fallback and is explicitly marked by a null timestamp.
        trace_id = hashlib.sha256(str(path.relative_to(source)).encode()).hexdigest()[:16]
        normalized.insert(0, "sample_index", np.arange(len(normalized), dtype=int))
        normalized.insert(0, "trace_id", trace_id)
        normalized["source_file"] = str(path.relative_to(source))
        normalized["source_sha256"] = sha256(path)
        normalized["provenance"] = "MEASURED_SEE_V2X_APPLICATION_TRACE"
        rows.append(normalized)
        receipts.append({"trace_id": trace_id, "path": str(path), "sha256": sha256(path), "rows": len(normalized)})
    if not rows:
        raise SystemExit(
            "MEASURED TRACE GATE FAILED: no SEE-V2X receiver CSV with latency and throughput columns was found."
        )
    result = pd.concat(rows, ignore_index=True)
    if len(result) < 1000 or result.trace_id.nunique() < 2:
        raise SystemExit("MEASURED TRACE GATE FAILED: require >=1000 samples across >=2 independent trace files")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    manifest = {
        "status": "PASS_MEASURED_TRACE_GATE", "rows": len(result),
        "traces": int(result.trace_id.nunique()), "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output), "sources": receipts,
        "network_provenance": "MEASURED_SEE_V2X_TRACE_REPLAY",
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
