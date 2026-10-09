#!/usr/bin/env python3
"""Stage 2: leave-one-scenario-out base models and causal envelopes for every link.

Also writes the window-level calibration study (empirical miss frequency of each
envelope variant versus the target alpha), which needs no controller at all.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from certisched import envelope as E


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("config/v2.yaml"))
    ap.add_argument("--data", type=Path, default=Path("data/processed/v2"))
    ap.add_argument("--results", type=Path, default=Path("results/v2"))
    ap.add_argument("--max-train", type=int, default=150_000)
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    ec, rc = cfg["envelope"], cfg["replay"]
    tau = rc["deadline_ms"] - rc["processing_ms"]
    taus = {"crit": tau, "norm": tau, "tot": rc["interval_ms"]}
    alphas = {"crit": ec["alpha_crit"], "norm": ec["alpha_norm"], "tot": ec["alpha_norm"]}
    env_cfg = E.EnvelopeConfig(alpha=ec["alpha_crit"], gamma=ec["gamma"],
                               residual_window=ec["residual_window"],
                               calibration_fraction=ec["calibration_fraction"],
                               base_quantile=ec["base_quantile"])
    links = pd.read_parquet(a.data / "links.parquet")
    feats = {t: pd.concat([E.lagged_features(g, v).assign(link_id=k, scenario_id=g.scenario_id.iloc[0])
                           for k, g in links.groupby("link_id")], ignore_index=True)
             for t, v in {"tau": tau, "tot": rc["interval_ms"]}.items()}
    rng = np.random.default_rng(20261009)
    envs, calib = [], []
    for scenario in sorted(links.scenario_id.unique()):
        models = {}
        for key, frame in feats.items():
            train = frame[frame.scenario_id != scenario].drop(columns=["link_id"])
            if len(train) > a.max_train:
                train = train.iloc[rng.choice(len(train), a.max_train, replace=False)]
            models[key] = E.fit_base_model(train, ec["base_quantile"])
        tier_models = {"crit": models["tau"], "norm": models["tau"], "tot": models["tot"]}
        for link_id, g in links[links.scenario_id == scenario].groupby("link_id"):
            env = E.build_trace_envelopes(g, tier_models, taus, alphas, env_cfg)
            env["link_id"] = link_id
            # Calibration sweep on the deadline tier (window level, controller-free).
            f = E.lagged_features(g, tau).reset_index(drop=True)
            target, pred = f["target"].to_numpy(float), models["tau"](f)
            start = int(np.argmax(env["evaluable"].to_numpy()))
            for alpha in ec["alpha_sweep"]:
                c = E.EnvelopeConfig(**{**env_cfg.__dict__, "alpha": alpha})
                bounds = {"aci": E.aci_lower_bounds(pred, target, c, start)[0],
                          "split": E.split_lower_bounds(pred, target, c, start),
                          "rolling": E.rolling_lower_bounds(pred, target, c, start)}
                env[f"aci_crit@{alpha}"] = bounds["aci"]   # controller-level alpha sweep
                for method, b in bounds.items():
                    ok = slice(start, None)
                    calib.append({"link_id": link_id, "scenario_id": scenario, "alpha": alpha,
                                  "method": method, "windows": len(target) - start,
                                  "miss": float(np.mean(target[ok] < b[ok])),
                                  "mean_bound_bytes": float(np.mean(np.minimum(b[ok], 1e7))),
                                  "mean_service_bytes": float(np.mean(target[ok]))})
            for method in ("point", "last"):
                b = env[f"{method}_crit"].to_numpy()
                ok = env["evaluable"].to_numpy()
                calib.append({"link_id": link_id, "scenario_id": scenario, "alpha": np.nan,
                              "method": method, "windows": int(ok.sum()),
                              "miss": float(np.nanmean(target[ok] < b[ok])),
                              "mean_bound_bytes": float(np.nanmean(b[ok])),
                              "mean_service_bytes": float(np.mean(target[ok]))})
            envs.append(env)
        print(f"envelopes done: {scenario}", flush=True)
    a.results.mkdir(parents=True, exist_ok=True)
    pd.concat(envs, ignore_index=True).to_parquet(a.data / "envelopes.parquet", index=False)
    calib = pd.DataFrame(calib)
    calib.to_csv(a.results / "envelope_calibration_links.csv", index=False)
    summary = (calib.assign(wm=calib.miss * calib.windows)
               .groupby(["method", "alpha"], dropna=False)
               .agg(windows=("windows", "sum"), wm=("wm", "sum"),
                    bound=("mean_bound_bytes", "mean"), service=("mean_service_bytes", "mean"))
               .reset_index())
    summary["miss"] = summary.wm / summary.windows
    summary["efficiency"] = summary.bound / summary.service
    summary.drop(columns="wm").to_csv(a.results / "envelope_calibration.csv", index=False)
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
