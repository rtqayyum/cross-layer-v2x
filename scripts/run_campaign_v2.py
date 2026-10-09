#!/usr/bin/env python3
"""Stage 3: paired replay campaign over sequences x link segments x policies.

Every policy in a (sequence, link segment) block sees identical objects, ego
poses, service, latency draws and erasure draws; the road sequence is the
statistical unit.  Results are written incrementally and resumable.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from certisched import v2xseq
from certisched.controllers import Policy, default_policies
from certisched.replay import ReplayConfig, run_episode
from certisched.scene import SceneConfig

_STATE: dict = {}


def _h(*parts) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "big")


def _init(data_dir: str, cfg: dict):
    d = Path(data_dir)
    tables = {k: pd.read_parquet(d / f"{k}.parquet") for k in ("rsu_objects", "ego", "gt", "ego_objects")}
    _STATE["sequences"] = v2xseq.to_sequences(tables)
    links = pd.read_parquet(d / "links.parquet")
    env = pd.read_parquet(d / "envelopes.parquet")
    _STATE["links"] = {k: g.sort_values("window").reset_index(drop=True) for k, g in links.groupby("link_id")}
    _STATE["env"] = {k: g.sort_values("window").reset_index(drop=True) for k, g in env.groupby("link_id")}
    _STATE["replay"] = ReplayConfig(**cfg["replay"])
    _STATE["scene"] = SceneConfig(**cfg["scene"])


def _run(job: dict) -> dict:
    seq = _STATE["sequences"][job["sequence"]]
    win, env = _STATE["links"][job["link_id"]], _STATE["env"][job["link_id"]]
    n = len(seq["epochs"]) + 5
    sl = slice(job["start"], job["start"] + n)
    windows = win.iloc[sl].to_dict("records")
    envs = env.iloc[sl].to_dict("records")
    policy = Policy(**{**job["policy"], "mode_names": tuple(job["policy"]["mode_names"])})
    out = run_episode(seq, {"windows": windows, "env": envs}, policy, _STATE["replay"], _STATE["scene"],
                      job["block"])
    return {**{k: v for k, v in job.items() if k != "policy"}, **out,
            **{f"p_{k}": v for k, v in job["policy"].items() if k in ("v_dpp", "alpha_crit", "load_factor")}}


def jobs(cfg: dict, data_dir: Path) -> list[dict]:
    camp = cfg["campaign"]
    links = pd.read_parquet(data_dir / "links.parquet", columns=["link_id", "setting", "window"])
    env = pd.read_parquet(data_dir / "envelopes.parquet", columns=["link_id", "window", "evaluable"])
    first_eval = env[env.evaluable].groupby("link_id").window.min()
    lengths = links.groupby("link_id").window.agg(["min", "max", "count"])
    settings = links.drop_duplicates("link_id").set_index("link_id").setting
    rsu = pd.read_parquet(data_dir / "rsu_objects.parquet", columns=["sequence", "infrastructure_frame"])
    epochs = rsu.groupby("sequence").infrastructure_frame.nunique()
    pols = [dataclasses.asdict(p) for p in default_policies()]
    sweep = camp["sweeps"]
    sweep_pols = ([dataclasses.asdict(Policy(f"CertiSched V={v:g}", "certisched", v_dpp=v)) for v in map(float, sweep["v_dpp"])]
                  + [dataclasses.asdict(Policy(f"CertiSched alpha={a:g}", "certisched", alpha_crit=a))
                     for a in map(float, sweep["alpha_crit"])])
    load_pols = []
    for lf in sweep["load_factor"]:
        if lf == 1:
            continue
        for name in ("CertiSched", "ETSI CPM rules", "VoI redundancy mitigation", "Max-weight AoI",
                     "w/o conformal (point)", "Oracle service"):
            base = next(p for p in pols if p["name"] == name)
            load_pols.append({**base, "name": f"{name} x{lf}", "load_factor": float(lf)})
    out = []
    for seq, n_ep in epochs.items():
        for tier, count in (("primary", camp["assignments_primary"]), ("secondary", camp["assignments_secondary"])):
            pool = [l for l in settings.index if (settings[l] == camp["primary_setting"]) == (tier == "primary")]
            if not pool:
                continue
            for a in range(count):
                link = sorted(pool)[_h(camp["seed"], seq, tier, a) % len(pool)]
                lo = int(first_eval.get(link, lengths.loc[link, "min"])) - int(lengths.loc[link, "min"])
                hi = int(lengths.loc[link, "count"]) - int(n_ep) - 6
                if hi <= lo:
                    continue
                start = lo + _h(camp["seed"], seq, tier, a, "offset") % (hi - lo)
                block = f"{seq}|{tier}|{a}"
                run_pols = pols + (sweep_pols + load_pols if tier == "primary" else [])
                for p in run_pols:
                    out.append({"sequence": seq, "tier": tier, "assignment": a, "link_id": link,
                                "start": int(start), "block": block, "policy": p})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("config/v2.yaml"))
    ap.add_argument("--data", type=Path, default=Path("data/processed/v2"))
    ap.add_argument("--out", type=Path, default=Path("results/v2/runs.parquet"))
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N jobs")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    todo = jobs(cfg, a.data)
    if a.limit:
        todo = todo[: a.limit]
    done, rows = set(), []
    if a.out.exists():
        old = pd.read_parquet(a.out)
        rows = old.to_dict("records")
        done = set(zip(old.block, old.policy))
    todo = [j for j in todo if (j["block"], j["policy"]["name"]) not in done]
    print(f"jobs to run: {len(todo)} (already done: {len(done)})", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    workers = cfg["campaign"]["workers"] or os.cpu_count()
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(str(a.data), cfg)) as ex:
        futures = [ex.submit(_run, j) for j in todo]
        for i, f in enumerate(as_completed(futures), 1):
            rows.append(f.result())
            if i % 500 == 0 or i == len(futures):
                pd.DataFrame(rows).to_parquet(a.out, index=False)
                print(f"progress {i}/{len(futures)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
