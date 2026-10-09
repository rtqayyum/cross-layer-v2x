"""SEE-V2X loader: per-link 100-ms service windows from measured C-V2X traces.

For each receiver file rx_<UE1>_tx_<UE2>_<app>.csv of the Emulating
Applications release we build one *link* with, per window w:

  delivered_bytes   bytes received in the window (from the 100-ms rolling
                    throughput the receiver logs, or packet sizes if present)
  loss              measured packet-loss fraction in the window
  rate_bpms         gross served rate = delivered / (1 - loss) / 100 ms
  lat_*_ms          quantiles of measured one-way packet latency in the window
  cbr               channel busy ratio (%) when logged
  sci_*             SCI aggregates of the scenario (all transmitters), in the window

Windows inside the capture with no received packet are outages (rate 0).
The counterfactual RSU may use at most the service the measured flow
demonstrably obtained ("service-capped replay"); the paper sweeps a scale
factor on it to cover scarcer and richer channels.
"""
from __future__ import annotations

import io
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_US = 100_000
SCENARIO = re.compile(r"emulating_applications/([^/]+)/([^/]+)/(intersection_[^/]+)/")
RX = re.compile(r"rx_([^_/]+)_tx_([^_/]+)(?:_([^/]+))?\.csv$")


@dataclass(frozen=True)
class SeeV2XConfig:
    bsm_bytes: float = 300.0          # SAE J2735 BSM size used for the RSU background
    background_streams: int = 2       # e.g. BSM + SPaT/MAP relay, 10 Hz each
    total_prbs: int = 100             # LTE-V2X 20 MHz carrier


def _clean(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _column(frame: pd.DataFrame, *candidates: str) -> str | None:
    cleaned = {_clean(c): c for c in frame.columns}
    for cand in candidates:
        if cand in cleaned:
            return cleaned[cand]
    for cand in candidates:
        for key, original in cleaned.items():
            if cand in key:
                return original
    return None


def _numeric(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column].astype(str).str.replace("%", "", regex=False), errors="coerce")


def _windows_from_rx(frame: pd.DataFrame) -> pd.DataFrame:
    ts = _column(frame, "rx_timestamp_us", "rx_timestamp", "timestamp_us", "timestamp")
    if ts is None:
        raise ValueError("receiver file has no timestamp column")
    t = _numeric(frame, ts)
    if t.dropna().median() < 1e14:          # seconds or ms -> us heuristics
        t = t * (1e6 if t.dropna().median() < 1e11 else 1e3)
    lat = _numeric(frame, _column(frame, "latency_ms", "latency"))
    thr = _numeric(frame, _column(frame, "avg_throughput_100ms", "throughput_100ms", "avg_throughput"))
    loss = _numeric(frame, _column(frame, "avg_packet_loss_100ms", "packet_loss", "per_ue_loss_pct"))
    if loss.dropna().median() > 1.0:
        loss = loss / 100.0
    cbr = _numeric(frame, _column(frame, "channel_busy_percentage", "channel_busy", "cbr"))
    size = _numeric(frame, _column(frame, "packet_size", "size_bytes"))
    data = pd.DataFrame({"w": (t // WINDOW_US), "lat": lat, "thr": thr, "loss": loss, "cbr": cbr,
                         "size": size}).dropna(subset=["w"])
    data["w"] = data["w"].astype(np.int64)
    g = data.groupby("w")
    out = pd.DataFrame({
        "packets": g["lat"].count(),
        "thr_last": g["thr"].last(),
        "size_sum": g["size"].sum(min_count=1),
        "loss": g["loss"].last(),
        "cbr": g["cbr"].median(),
        "lat_min_ms": g["lat"].min(), "lat_p10_ms": g["lat"].quantile(0.1),
        "lat_p50_ms": g["lat"].median(), "lat_p90_ms": g["lat"].quantile(0.9),
        "lat_max_ms": g["lat"].max(),
    })
    full = pd.RangeIndex(out.index.min(), out.index.max() + 1, name="w")
    out = out.reindex(full)
    delivered = out["size_sum"].where(out["size_sum"].notna() & (out["packets"] > 0),
                                      out["thr_last"] * 0.1 / 8.0)
    out["delivered_bytes"] = delivered.where(out["packets"].fillna(0) > 0, 0.0).fillna(0.0)
    out["loss"] = out["loss"].fillna(1.0).clip(0.0, 1.0)
    out.loc[out["packets"].fillna(0) == 0, "loss"] = 1.0
    for col in ("lat_min_ms", "lat_p10_ms", "lat_p50_ms", "lat_p90_ms", "lat_max_ms"):
        out[col] = out[col].ffill().bfill()
    gross = out["delivered_bytes"] / np.maximum(1e-6, 1.0 - out["loss"].clip(upper=0.999))
    out["rate_bpms"] = np.where(out["delivered_bytes"] > 0, gross / 100.0, 0.0)
    out["cbr"] = out["cbr"].ffill()
    return out.reset_index().rename(columns={"w": "window"})


def _sci_windows(frame: pd.DataFrame, cfg: SeeV2XConfig) -> pd.DataFrame:
    ts = _column(frame, "rx_timestamp_us", "timestamp")
    t = _numeric(frame, ts)
    if t.dropna().median() < 1e14:
        t = t * (1e6 if t.dropna().median() < 1e11 else 1e3)
    data = pd.DataFrame({"w": t // WINDOW_US, "mcs": _numeric(frame, _column(frame, "mcs_idx")),
                         "prb": _numeric(frame, _column(frame, "nof_prb_pssch")),
                         "retx": _numeric(frame, _column(frame, "retransmission"))}).dropna(subset=["w"])
    data["w"] = data["w"].astype(np.int64)
    g = data.groupby("w")
    return pd.DataFrame({"sci_count": g["mcs"].count(), "sci_mean_mcs": g["mcs"].mean(),
                         "sci_mean_prb": g["prb"].mean(), "sci_retx": g["retx"].mean(),
                         # PRB-subframes used / available in 100 subframes: occupancy proxy
                         "sci_occupancy": g["prb"].sum() / (cfg.total_prbs * 100.0)}
                        ).reset_index().rename(columns={"w": "window"})


def load(archive: Path, cfg: SeeV2XConfig = SeeV2XConfig(), apps=("perception",)) -> pd.DataFrame:
    """All links of the Emulating Applications release as one window table."""
    rx_frames, sci = [], {}
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".csv"):
                continue
            scen = SCENARIO.search(member.name)
            if not scen:
                continue
            scenario = "/".join(scen.groups())
            base = Path(member.name).name
            if base == "sci.csv":
                sci[scenario] = _sci_windows(pd.read_csv(io.BytesIO(tar.extractfile(member).read()),
                                                         low_memory=False), cfg)
                continue
            m = RX.search(base)
            if not m or not base.startswith("rx_"):
                continue
            app = (m.group(3) or "").lower()
            if apps and not any(a in app for a in apps):
                continue
            frame = pd.read_csv(io.BytesIO(tar.extractfile(member).read()), low_memory=False)
            try:
                win = _windows_from_rx(frame)
            except ValueError:
                continue
            win["link_id"] = f"{scenario}/{base[:-4]}"
            win["scenario_id"] = scenario
            win["traffic"], win["setting"], win["intersection"] = scen.groups()
            win["receiver"], win["transmitter"], win["app"] = m.group(1), m.group(2), app
            rx_frames.append(win)
    if not rx_frames:
        raise SystemExit("no SEE-V2X receiver files found; run scripts/inspect_data.py")
    links = pd.concat(rx_frames, ignore_index=True)
    sci_all = pd.concat([v.assign(scenario_id=k) for k, v in sci.items()], ignore_index=True) if sci else None
    if sci_all is not None:
        links = links.merge(sci_all, on=["scenario_id", "window"], how="left")
    for col in ("sci_count", "sci_mean_mcs", "sci_mean_prb", "sci_retx", "sci_occupancy"):
        if col not in links:
            links[col] = np.nan
    links["sci_count"] = links["sci_count"].fillna(0)
    # RSU background: periodic BSM-sized messages from `background_streams` services.
    links["bg_bytes"] = cfg.bsm_bytes * cfg.background_streams
    links = links.sort_values(["link_id", "window"]).reset_index(drop=True)
    links["loss_hat"] = links.groupby("link_id")["loss"].shift(1).fillna(0.0)
    return links
