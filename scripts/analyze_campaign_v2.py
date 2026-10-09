#!/usr/bin/env python3
"""Stage 4: statistics, figure data and LaTeX macros for the manuscript.

Inference unit: the road sequence.  Run-level outcomes are averaged within
sequence before bootstrapping (5,000 resamples).  The confirmatory family is
declared here, before any result is seen: CertiSched versus each
contract-compliant baseline on critical awareness and hidden awareness
(Wilcoxon signed-rank on sequence means, Holm-adjusted).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROPOSED = "CertiSched"
BASELINES = ["ETSI CPM rules", "VoI redundancy mitigation", "Max-weight AoI", "Confidence density", "Send all"]
PRIMARY = ["critical_awareness", "hidden_awareness"]
METRICS = ["critical_awareness", "hidden_awareness", "critical_aoi_ms", "hidden_error_m", "ap_bev_50",
           "ap_bev_70", "ap_3d_50", "ap_3d_70", "background_delivery", "kbytes_per_s", "decision_ms_mean"]
MAIN_ORDER = [PROPOSED, "CertiSched-greedy", "Oracle service", *BASELINES,
              "Max-weight AoI (ahead of bg)", "VoI redundancy mitigation (ahead of bg)"]
ABLATIONS = [PROPOSED, "w/o conformal (point)", "w/o adaptation (split)", "fixed-level rolling",
             "last-value budget", "w/o ego relevance", "w/o tiers", "w/o opportunistic tier", "w/o DPP",
             "box mode only"]
RNG = np.random.default_rng(20261009)


def seq_means(d: pd.DataFrame, metric: str) -> pd.Series:
    return d.groupby("sequence")[metric].mean().dropna()


def boot_ci(values: np.ndarray, n: int = 5000) -> tuple[float, float]:
    if len(values) < 2:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(values), (n, len(values)))
    means = values[idx].mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def pooled_cert(d: pd.DataFrame) -> dict:
    e, em = d.certified_epochs.sum(), d.certified_epoch_misses.sum()
    o, om = d.certified_tx.sum(), d.certified_misses.sum()
    n = d.decisions.sum()
    # Theorem 1 bounds coverage failures over all decision windows; the
    # conditional rate (given certified traffic) and object rate are empirical.
    return {"cert_window_miss": em / n if n and e else np.nan,
            "cert_window_miss_given_traffic": em / e if e else np.nan,
            "cert_object_miss": om / o if o else np.nan, "cert_epochs": int(e)}


def summary_table(d: pd.DataFrame, policies: list[str]) -> pd.DataFrame:
    rows = []
    for p in policies:
        g = d[d.policy == p]
        if g.empty:
            continue
        row = {"policy": p, "runs": len(g), "sequences": g.sequence.nunique()}
        for m in METRICS:
            v = seq_means(g, m).to_numpy()
            row[m] = float(np.mean(v)) if len(v) else np.nan
            row[f"{m}_lo"], row[f"{m}_hi"] = boot_ci(v)
        row.update(pooled_cert(g))
        rows.append(row)
    return pd.DataFrame(rows)


def paired(d: pd.DataFrame, a: str, b: str, metric: str) -> dict:
    x = d[d.policy == a].set_index("block")[metric]
    y = d[d.policy == b].set_index("block")[metric]
    blocks = x.index.intersection(y.index)
    diff = (x[blocks] - y[blocks]).to_frame("diff")
    diff["sequence"] = d.drop_duplicates("block").set_index("block").loc[blocks, "sequence"]
    s = diff.dropna().groupby("sequence")["diff"].mean()
    lo, hi = boot_ci(s.to_numpy())
    p = 1.0 if len(s) < 2 or np.allclose(s, 0) else float(wilcoxon(s, zero_method="pratt").pvalue)
    return {"proposed": a, "baseline": b, "metric": metric, "diff": float(s.mean()), "lo": lo, "hi": hi,
            "p": p, "favorable": float((s > 0).mean()), "sequences": len(s)}


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adjusted, running = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, ps[i] * (len(ps) - rank)))
        adjusted[i] = running
    return adjusted.tolist()


DIGITS = "Zero One Two Three Four Five Six Seven Eight Nine".split()


def macro_name(*parts: str) -> str:
    text = re.sub(r"\d", lambda m: f" {DIGITS[int(m.group())]} ", " ".join(parts))
    words = re.sub(r"[^A-Za-z ]+", " ", text).title().replace(" ", "")
    return "\\res" + words


def fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    return f"{value:.{digits}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=Path("results/v2/runs.parquet"))
    ap.add_argument("--results", type=Path, default=Path("results/v2"))
    a = ap.parse_args()
    d = pd.read_parquet(a.runs)
    out = a.results
    out.mkdir(parents=True, exist_ok=True)
    macros: dict[str, str] = {}
    prim, sec = d[d.tier == "primary"], d[d.tier == "secondary"]

    main = summary_table(prim, MAIN_ORDER)
    main.to_csv(out / "table_main.csv", index=False)
    summary_table(prim, ABLATIONS).to_csv(out / "table_ablation.csv", index=False)
    summary_table(sec, MAIN_ORDER).to_csv(out / "table_secondary.csv", index=False)

    tests = [paired(prim, PROPOSED, b, m) for b in BASELINES for m in PRIMARY]
    for t, h in zip(tests, holm([t["p"] for t in tests])):
        t["holm_p"] = h
    pd.DataFrame(tests).to_csv(out / "confirmatory_tests.csv", index=False)
    ablation_tests = [paired(prim, PROPOSED, b, m) for b in ABLATIONS[1:] for m in PRIMARY + ["background_delivery"]]
    pd.DataFrame(ablation_tests).to_csv(out / "ablation_tests.csv", index=False)

    # Sweeps -------------------------------------------------------------
    def sweep(prefix: str, key: str) -> pd.DataFrame:
        g = prim[prim.policy.str.startswith(prefix)]
        rows = []
        for value, h in g.groupby(key):
            row = {key: value}
            for m in ("critical_awareness", "hidden_awareness", "background_delivery"):
                row[m] = float(seq_means(h, m).mean())
            row.update(pooled_cert(h))
            rows.append(row)
        return pd.DataFrame(rows)

    sweep("CertiSched V=", "p_v_dpp").to_csv(out / "fig_v_sweep.csv", index=False)
    sweep("CertiSched alpha=", "p_alpha_crit").to_csv(out / "fig_alpha_sweep.csv", index=False)
    loads = []
    for name in [PROPOSED, "ETSI CPM rules", "VoI redundancy mitigation", "Max-weight AoI",
                 "w/o conformal (point)", "Oracle service"]:
        for lf, suffix in ((1.0, ""), (2.0, " x2"), (4.0, " x4")):
            g = prim[prim.policy == name + suffix]
            if len(g):
                loads.append({"policy": name, "load_factor": lf,
                              "critical_awareness": float(seq_means(g, "critical_awareness").mean()),
                              "hidden_awareness": float(seq_means(g, "hidden_awareness").mean()),
                              **pooled_cert(g)})
    pd.DataFrame(loads).to_csv(out / "fig_load_sweep.csv", index=False)

    # Macros --------------------------------------------------------------
    for r in main.itertuples():
        for m in ("critical_awareness", "hidden_awareness", "ap_bev_50", "ap_3d_70", "background_delivery"):
            macros[macro_name(r.policy, m)] = fmt(getattr(r, m))
        macros[macro_name(r.policy, "aoi")] = fmt(r.critical_aoi_ms, 0)
        macros[macro_name(r.policy, "kbps")] = fmt(r.kbytes_per_s * 8, 1)
        macros[macro_name(r.policy, "cert window miss")] = fmt(r.cert_window_miss, 4)
        macros[macro_name(r.policy, "cert object miss")] = fmt(r.cert_object_miss, 4)
        macros[macro_name(r.policy, "cert cond miss")] = fmt(r.cert_window_miss_given_traffic, 4)
        macros[macro_name(r.policy, "decision ms")] = fmt(r.decision_ms_mean, 2)
    for t in tests:
        key = macro_name("vs", t["baseline"], t["metric"])
        macros[key + "Diff"] = fmt(t["diff"])
        macros[key + "Lo"] = fmt(t["lo"])
        macros[key + "Hi"] = fmt(t["hi"])
        macros[key + "Holm"] = f"{t['holm_p']:.2g}"
    macros["\\resRuns"] = f"{len(d):,}".replace(",", "{,}")
    macros["\\resSequences"] = str(d.sequence.nunique())
    macros["\\resPrimaryBlocks"] = str(prim.block.nunique())
    macros["\\resLinks"] = str(d.link_id.nunique())
    calib = out / "envelope_calibration.csv"
    if calib.exists():
        c = pd.read_csv(calib)
        for r in c.itertuples():
            if pd.notna(r.alpha):
                tag = f"{r.method} a{int(round(r.alpha * 100)):02d}"
                macros[macro_name("env", tag, "miss")] = fmt(r.miss, 4)
                macros[macro_name("env", tag, "eff")] = fmt(r.efficiency, 3)
            else:
                macros[macro_name("env", r.method, "miss")] = fmt(r.miss, 3)
    prep = Path("data/processed/v2/prepare_summary.json")
    if prep.exists():
        p = json.loads(prep.read_text())
        for k in ("sequences", "infrastructure_frames", "vehicle_frames", "gt_hidden", "gt_critical", "links",
                  "scenarios", "windows"):
            macros[macro_name("data", k)] = f"{p[k]:,}".replace(",", "{,}")
    lines = ["% Auto-generated by scripts/analyze_campaign_v2.py -- do not edit."]
    lines += [f"\\newcommand{{{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    (out / "macros.tex").write_text("\n".join(lines) + "\n")
    print(main[["policy", "critical_awareness", "hidden_awareness", "background_delivery", "cert_window_miss"]]
          .to_string(index=False))
    print(pd.DataFrame(tests)[["baseline", "metric", "diff", "lo", "hi", "holm_p"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
