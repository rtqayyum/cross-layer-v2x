"""Trace-driven replay of one (sequence, link segment, policy) episode.

Service model (stated in the paper):
* Window k of the measured link has gross served rate r_k (bytes/ms), a
  measured per-packet access-latency distribution and an erasure rate p_k.
* At each decision the RSU enqueues CPM segments.  The radio serves strict
  priority classes 1-4 (class 3 = the RSU's background traffic) as a
  fluid at rate r_k from the window start; a packet completes when its last
  byte is served, then arrives after an access latency drawn from the
  window's measured distribution.  Packets unfinished at the window end are
  superseded by the next CPM (dropped), so over-admission costs deliveries.
* Erasures are independent per packet with probability p_k.
Nothing in a window's service depends on any controller, so all policies see
identical, exogenous service (paired comparison).
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np

from .controllers import Context, Policy, decide
from .messages import CPM_HEADER_BYTES, Mode, delivery_probability, mode_bytes, transmitted_state
from .metrics import average_precision, frame_ap_records, fuse
from .scene import SceneConfig, annotate


@dataclass(frozen=True)
class ReplayConfig:
    interval_ms: float = 100.0
    processing_ms: float = 20.0
    deadline_ms: float = 100.0          # freshness requirement from capture
    segment_objects: int = 10           # max objects per CPM segment
    background_target: float = 0.95     # rho: required background delivery ratio
    match_distance_m: float = 2.0
    eval_radius_m: float = 80.0


def unit(*parts) -> float:
    digest = hashlib.blake2b(":".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def latency_sample(win: dict, u: float) -> float:
    """Inverse CDF through the window's measured latency quantiles."""
    qs = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    vals = np.array([win.get("lat_min_ms", win["lat_p10_ms"]), win["lat_p10_ms"], win["lat_p50_ms"],
                     win["lat_p90_ms"], win["lat_max_ms"]], float)
    return float(np.interp(u, qs, np.maximum.accumulate(vals)))


def taus(cfg: ReplayConfig) -> dict:
    tau = cfg.deadline_ms - cfg.processing_ms
    return {"crit": tau, "norm": tau, "tot": cfg.interval_ms}


def _packets(decision, segment_objects: int):
    """Group a decision into CPM segments: (priority, copy, [(obj, mode)]).

    Objects keep the policy's order inside a priority class and are cut into
    segments of at most `segment_objects` (ETSI CPM segmentation).  Repetition
    copies form their own segments after the primary ones of that class.
    """
    groups: dict[tuple[int, int], list] = {}
    for obj, mode, prio in decision:
        for copy in range(mode.copies):
            groups.setdefault((prio, copy), []).append((obj, mode))
    out = []
    for (prio, copy), items in sorted(groups.items()):
        for i in range(0, len(items), segment_objects):
            out.append((prio, copy, items[i:i + segment_objects]))
    return out


def run_episode(seq: dict, link: dict, policy: Policy, cfg: ReplayConfig,
                scene_cfg: SceneConfig, run_key: str) -> dict:
    """Simulate one episode and return per-run metrics plus per-frame records.

    seq:  {"epochs": [{"t_ms", "objects": [...]}], "ego": array rows (t_ms,x,y,vx,vy,yaw),
           "frames": [{"t_ms", "gt": [...], "ego_objects": [...]}]}
    link: {"windows": list of window dicts (rate_bpms, lat_*, loss, loss_hat, bg_bytes),
           "env": list of envelope dicts aligned with windows}
    """
    tau = taus(cfg)
    beliefs: dict = {}
    last_tx: dict = {}
    deliveries: dict[str, list] = {}
    z = 0.0
    t0 = seq["epochs"][0]["t_ms"] if seq["epochs"] else 0.0
    ego = seq["ego"]
    stats = {"cert_tx": 0, "cert_service_miss": 0, "crit_tx": 0, "crit_ontime": 0, "crit_service_miss": 0, "crit_erased": 0,
             "norm_tx": 0, "norm_ontime": 0, "bytes": 0.0, "bg_offered": 0.0, "bg_served": 0.0,
             "z_sum": 0.0, "decisions": 0, "decision_ms": [], "over_admitted": 0}
    for k, epoch in enumerate(seq["epochs"]):
        widx = int(round((epoch["t_ms"] - t0) / cfg.interval_ms))
        if widx >= len(link["windows"]):
            break
        win, env = link["windows"][widx], link["env"][widx]
        decision_ms = epoch["t_ms"] + cfg.processing_ms
        past = ego[ego[:, 0] <= epoch["t_ms"]] if len(ego) else ego
        ego_now = None if len(past) == 0 else dict(zip(("t_ms", "x", "y", "vx", "vy", "yaw"), past[-1]))
        objects = [dict(o) for o in epoch["objects"]]
        annotate(objects, ego_now, beliefs, decision_ms, tau, scene_cfg)
        ctx = Context(decision_ms, objects, env, float(win.get("loss_hat", 0.0)),
                      float(win.get("bg_bytes", 0.0)), z, beliefs, last_tx, scene_cfg)
        started = time.perf_counter()
        decision = decide(ctx, policy, tau)
        stats["decision_ms"].append((time.perf_counter() - started) * 1e3)
        stats["decisions"] += 1

        rate = float(win["rate_bpms"])
        horizon = cfg.interval_ms
        served = 0.0                       # bytes served so far in this window
        bg = float(win.get("bg_bytes", 0.0))
        packets = _packets(decision, cfg.segment_objects)
        ordered = sorted(packets + [(3, 0, None)], key=lambda p: (p[0], p[1]))
        bg_served = 0.0
        outcome: dict[str, dict] = {}      # per object: any copy on time / delivered
        for prio, copy, items in ordered:
            if items is None:              # RSU background (e.g. BSM/SPaT relay)
                bg_served = min(bg, max(0.0, rate * horizon - served))
                served += bg
                continue
            size = CPM_HEADER_BYTES + sum(m.bytes_per_object for _, m in items)
            served += size
            done = served / rate if rate > 0 else np.inf
            completed = done <= horizon
            stats["bytes"] += size if completed else 0.0
            latency = latency_sample(win, unit(run_key, k, prio, copy, "lat"))
            erased = unit(run_key, k, prio, copy, "erase") < float(win["loss"])
            in_time = completed and (done + latency) <= tau["crit"]
            for obj, mode in items:
                rec = outcome.setdefault(obj["track_id"], {
                    "tier": obj["tier"], "served_in_time": False, "on_time": False,
                    "certified": prio == 1 and policy.kind == "certisched" and policy.tiers})
                rec["served_in_time"] |= in_time
                rec["on_time"] |= in_time and not erased
                if completed and not erased:
                    deliveries.setdefault(obj["track_id"], []).append(
                        (decision_ms + done + latency, transmitted_state(obj, mode)))
        for rec in outcome.values():
            if rec["certified"]:
                stats["cert_tx"] += 1
                stats["cert_service_miss"] += int(not rec["served_in_time"])
            if rec["tier"] == "crit":
                stats["crit_tx"] += 1
                stats["crit_service_miss"] += int(not rec["served_in_time"])
                stats["crit_erased"] += int(rec["served_in_time"] and not rec["on_time"])
                stats["crit_ontime"] += int(rec["on_time"])
            else:
                stats["norm_tx"] += 1
                stats["norm_ontime"] += int(rec["on_time"])
        stats["over_admitted"] += int(served > rate * horizon + 1e-9)
        # RSU-side belief: last transmitted state (no acknowledgement on broadcast).
        for obj, mode, _ in decision:
            belief = transmitted_state(obj, mode)
            belief["delivery_prob"] = delivery_probability(mode, float(win.get("loss_hat", 0.0)))
            beliefs[obj["track_id"]] = belief
            last_tx[obj["track_id"]] = dict(obj)
        stats["bg_offered"] += bg
        stats["bg_served"] += bg_served
        z = max(0.0, z + cfg.background_target * bg - bg_served)
        stats["z_sum"] += z

    frames, ap = evaluate_frames(seq, deliveries, cfg)
    return {**summarize(stats, frames, policy), **ap}


def receiver_view(deliveries: dict, t_ms: float) -> list[dict]:
    """Latest delivered state per track that has arrived by t_ms, extrapolated."""
    view = []
    for track, items in deliveries.items():
        arrived = [it for it in items if it[0] <= t_ms]
        if not arrived:
            continue
        _, state = max(arrived, key=lambda it: it[1]["capture_ms"])
        dt = (t_ms - state["capture_ms"]) / 1000.0
        if dt > 1.0:      # receivers discard object states older than 1 s
            continue
        s = dict(state)
        s["x"] += s["vx"] * dt
        s["y"] += s["vy"] * dt
        s["age_ms"] = t_ms - state["capture_ms"]
        s["track_id"] = track
        view.append(s)
    return view


AP_SETTINGS = (("bev", 0.5), ("bev", 0.7), ("3d", 0.5), ("3d", 0.7))


def evaluate_frames(seq: dict, deliveries: dict, cfg: ReplayConfig) -> tuple[list[dict], dict]:
    records = []
    ap_records = {s: [] for s in AP_SETTINGS}
    ap_gt = {s: 0 for s in AP_SETTINGS}
    for frame in seq["frames"]:
        view = receiver_view(deliveries, frame["t_ms"])
        pts = np.array([[v["x"], v["y"]] for v in view]) if view else np.zeros((0, 2))
        used = set()
        for gt in frame["gt"]:
            if not gt.get("in_range", True):
                continue
            match, err = None, np.inf
            if len(pts):
                d = np.hypot(pts[:, 0] - gt["x"], pts[:, 1] - gt["y"])
                for i in np.argsort(d):
                    if d[i] > cfg.match_distance_m:
                        break
                    if i not in used:
                        match, err = int(i), float(d[i])
                        break
            if match is not None:
                used.add(match)
            records.append({"t_ms": frame["t_ms"], "hidden": bool(gt["hidden"]),
                            "critical": bool(gt["critical"]), "aware": match is not None,
                            "error_m": err if match is not None else np.nan,
                            "aoi_ms": view[match]["age_ms"] if match is not None else np.nan})
        fused = fuse(frame.get("ego_objects", []), view)
        for setting in AP_SETTINGS:
            recs, n = frame_ap_records(fused, frame["gt"], setting[1], setting[0])
            ap_records[setting].extend(recs)
            ap_gt[setting] += n
    ap = {f"ap_{v}_{int(t * 100)}": average_precision(ap_records[(v, t)], ap_gt[(v, t)])
          for (v, t) in AP_SETTINGS}
    return records, ap


def summarize(stats: dict, records: list[dict], policy: Policy) -> dict:
    def rate(mask_key):
        sel = [r for r in records if r[mask_key]]
        return (np.mean([r["aware"] for r in sel]) if sel else np.nan), len(sel)

    crit_rate, n_crit = rate("critical")
    hid_rate, n_hid = rate("hidden")
    aware_crit = [r for r in records if r["critical"] and r["aware"]]
    aware_hid = [r for r in records if r["hidden"] and r["aware"]]
    dts = np.array(stats["decision_ms"]) if stats["decision_ms"] else np.array([np.nan])
    return {
        "policy": policy.name,
        "critical_awareness": crit_rate, "critical_object_frames": n_crit,
        "hidden_awareness": hid_rate, "hidden_object_frames": n_hid,
        "critical_aoi_ms": float(np.mean([r["aoi_ms"] for r in aware_crit])) if aware_crit else np.nan,
        "hidden_aoi_ms": float(np.mean([r["aoi_ms"] for r in aware_hid])) if aware_hid else np.nan,
        "hidden_error_m": float(np.mean([r["error_m"] for r in aware_hid])) if aware_hid else np.nan,
        "crit_tx": stats["crit_tx"],
        "certified_tx": stats["cert_tx"],
        "certified_miss_rate": stats["cert_service_miss"] / stats["cert_tx"] if stats["cert_tx"] else np.nan,
        "crit_service_miss_rate": stats["crit_service_miss"] / stats["crit_tx"] if stats["crit_tx"] else np.nan,
        "crit_ontime_rate": stats["crit_ontime"] / stats["crit_tx"] if stats["crit_tx"] else np.nan,
        "norm_ontime_rate": stats["norm_ontime"] / stats["norm_tx"] if stats["norm_tx"] else np.nan,
        "kbytes_per_s": stats["bytes"] / max(stats["decisions"], 1) * 10 / 1000,
        "background_delivery": stats["bg_served"] / stats["bg_offered"] if stats["bg_offered"] else 1.0,
        "mean_virtual_queue": stats["z_sum"] / max(stats["decisions"], 1),
        "over_admission_rate": stats["over_admitted"] / max(stats["decisions"], 1),
        "decision_ms_mean": float(np.nanmean(dts)), "decision_ms_max": float(np.nanmax(dts)),
        "decisions": stats["decisions"],
    }
