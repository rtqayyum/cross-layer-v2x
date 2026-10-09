"""TEST-ONLY synthetic fixture.  Never used for any reported result.

Builds a small intersection episode (ego driving east, crossing traffic
driving north behind parked trucks) and a Markov-modulated link so the
whole pipeline can be exercised without the licensed datasets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geometry import segment_hits_box


def make_sequence(seed: int = 0, frames: int = 80) -> dict:
    rng = np.random.default_rng(seed)
    ego0, ego_v = np.array([-60.0, -2.0]), np.array([8.0, 0.0])
    parked = [(-12.0 + 8 * i, 3.5, 9.0, 2.6, 3.2) for i in range(4)]        # tall occluders
    movers = []
    for _ in range(6):
        # Crossing traffic timed to meet the ego near x (a true conflict).
        x, y0 = rng.uniform(-30, 0), rng.uniform(12, 25)
        t_meet = (x - ego0[0]) / ego_v[0] + rng.uniform(-0.3, 0.3)
        movers.append((x, y0, 0.0, -(y0 - ego0[1]) / t_meet))
    epochs, frames_out, ego_rows = [], [], []
    for k in range(frames):
        t = 100.0 * k
        ego = ego0 + ego_v * t / 1000.0
        ego_rows.append((t, ego[0], ego[1], ego_v[0], ego_v[1], 0.0))
        objs = []
        for i, (x, y, l, w, h) in enumerate(parked):
            objs.append(dict(track_id=f"p{i}", object_class="truck", x=x, y=y, z=1.6, length=l,
                             width=w, height=h, yaw=0.0, vx=0.0, vy=0.0, confidence=1.0, capture_ms=t))
        for i, (x, y, vx, vy) in enumerate(movers):
            px, py = x + vx * t / 1000.0, y + vy * t / 1000.0
            objs.append(dict(track_id=f"m{i}", object_class="car", x=px, y=py, z=0.8, length=4.5,
                             width=1.9, height=1.6, yaw=float(np.arctan2(vy, vx)), vx=vx, vy=vy,
                             confidence=1.0, capture_ms=t))
        epochs.append({"t_ms": t, "objects": objs})
        gt = []
        for o in objs:
            hidden = any(segment_hits_box(ego, (o["x"], o["y"]), (q["x"], q["y"], q["length"], q["width"], 0.0))
                         for q in objs if q is not o and q["height"] > 2.5)
            future = [(o["x"] + o["vx"] * s, o["y"] + o["vy"] * s) for s in np.arange(0, 5, 0.5)]
            ego_future = [(ego[0] + ego_v[0] * s, ego[1] + ego_v[1] * s) for s in np.arange(0, 5, 0.5)]
            dmin = min(np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(future, ego_future))
            gt.append(dict(o, hidden=hidden, critical=hidden and dmin < 5.0,
                           in_range=np.hypot(o["x"] - ego[0], o["y"] - ego[1]) < 80))
        visible = [dict(g, confidence=1.0) for g in gt if not g["hidden"]]
        frames_out.append({"t_ms": t + 50.0, "gt": gt, "ego_objects": visible})
    return {"epochs": epochs, "frames": frames_out, "ego": np.array(ego_rows)}


def make_link(seed: int = 0, windows: int = 3000, mean_rate_bpms: float = 25.0) -> pd.DataFrame:
    """Two-state Markov rate with measured-style latency and loss columns."""
    rng = np.random.default_rng(seed)
    state, rows = 0, []
    for w in range(windows):
        if rng.random() < 0.02:
            state = 1 - state
        rate = max(0.0, rng.normal(mean_rate_bpms * (1.0 if state == 0 else 0.25), 0.6))
        lat = np.sort(rng.gamma(2.0, 6.0 if state == 0 else 12.0, size=8))
        rows.append(dict(window=w, rate_bpms=rate, lat_min_ms=lat[0], lat_p10_ms=lat[1],
                         lat_p50_ms=np.median(lat), lat_p90_ms=lat[-2], lat_max_ms=lat[-1],
                         loss=0.02 if state == 0 else 0.15, cbr=30 + 40 * state + rng.normal(0, 3),
                         bg_bytes=float(rng.integers(150, 450))))
    frame = pd.DataFrame(rows)
    frame["loss_hat"] = frame["loss"].shift(1).fillna(frame["loss"].iloc[0])
    return frame
