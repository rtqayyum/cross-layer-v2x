"""Ego-centric relevance and value of an object update.

Everything in this module that feeds a decision uses only information the
roadside unit (RSU) has when it decides: its own tracked objects and the
connected vehicle's latest broadcast pose (CAM/BSM).  Ground-truth labels
are used only by `ground_truth_criticality`, which the evaluator calls.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import bev_iou, segment_hits_box
from .messages import Mode, delivery_probability, prior_dimensions


@dataclass(frozen=True)
class SceneConfig:
    ego_sensor_range_m: float = 60.0
    occluder_min_height_m: float = 1.2
    horizon_s: float = 5.0
    conflict_scale_m: float = 5.0
    critical_occlusion: float = 0.5
    critical_distance_m: float = 8.0
    base_relevance: float = 0.05
    error_scale_m: float = 1.0
    position_weight: float = 0.5   # eta: positional awareness vs box fidelity
    undelivered_error_m: float = 1e6
    state_expiry_ms: float = 1000.0  # receivers discard object states older than this
    hidden_floor: float = 0.3        # kappa: value of a hidden object without conflict
    accel_uncertainty: float = 1.5   # a_max (m/s^2): unmodelled acceleration growth


def closest_approach(p_ego, v_ego, p_obj, v_obj, horizon_s: float) -> tuple[float, float]:
    """Minimum distance and its time for two constant-velocity points."""
    dp = np.asarray(p_obj, float) - np.asarray(p_ego, float)
    dv = np.asarray(v_obj, float) - np.asarray(v_ego, float)
    denom = float(dv @ dv)
    t = 0.0 if denom < 1e-9 else float(np.clip(-(dp @ dv) / denom, 0.0, horizon_s))
    return float(np.linalg.norm(dp + dv * t)), t


def occlusion_from_ego(objects: list[dict], ego_xy, cfg: SceneConfig) -> np.ndarray:
    """Fraction of sight lines (centre + 4 shrunk corners) blocked by other objects."""
    ego_xy = np.asarray(ego_xy, float)
    occl = np.zeros(len(objects))
    boxes = [(o["x"], o["y"], o["length"], o["width"], o["yaw"]) for o in objects]
    tall = [o["height"] >= cfg.occluder_min_height_m for o in objects]
    dists = [float(np.hypot(o["x"] - ego_xy[0], o["y"] - ego_xy[1])) for o in objects]
    for j, obj in enumerate(objects):
        if dists[j] > cfg.ego_sensor_range_m:
            occl[j] = 1.0
            continue
        c, s = np.cos(obj["yaw"]), np.sin(obj["yaw"])
        hl, hw = 0.35 * obj["length"], 0.35 * obj["width"]
        targets = [(obj["x"], obj["y"])] + [
            (obj["x"] + c * a - s * b, obj["y"] + s * a + c * b)
            for a, b in ((hl, hw), (hl, -hw), (-hl, hw), (-hl, -hw))]
        blocked = 0
        for target in targets:
            for k, box in enumerate(boxes):
                if k == j or not tall[k] or dists[k] >= dists[j]:
                    continue
                if segment_hits_box(ego_xy, target, box, margin=0.1):
                    blocked += 1
                    break
        occl[j] = blocked / len(targets)
    return occl


def kinematic_mode_quality(obj: dict) -> float:
    """BEV IoU between the RSU box and what a kinematic-only message reconstructs."""
    length, width, _ = prior_dimensions(obj["object_class"])
    speed = float(np.hypot(obj.get("vx", 0.0), obj.get("vy", 0.0)))
    yaw = float(np.arctan2(obj["vy"], obj["vx"])) if speed > 0.5 else 0.0
    return bev_iou((obj["x"], obj["y"], obj["length"], obj["width"], obj["yaw"]),
                   (obj["x"], obj["y"], length, width, yaw))


def receiver_error(obj: dict, belief: dict | None, at_ms: float, cfg: SceneConfig) -> float:
    """Position error of the receiver's extrapolated estimate at `at_ms` if not updated.

    A state that will have expired at the receiver by `at_ms` counts as absent.
    """
    if belief is None or at_ms - belief["capture_ms"] >= cfg.state_expiry_ms:
        return cfg.undelivered_error_m
    dt = (at_ms - belief["capture_ms"]) / 1000.0
    est = np.array([belief["x"] + belief["vx"] * dt, belief["y"] + belief["vy"] * dt])
    dt_true = (at_ms - obj["capture_ms"]) / 1000.0
    true = np.array([obj["x"] + obj.get("vx", 0.0) * dt_true, obj["y"] + obj.get("vy", 0.0) * dt_true])
    # Known extrapolation error plus the growth an unmodelled acceleration can
    # cause over the belief's age (0.5 a t^2); the two add in quadrature.
    growth = 0.5 * cfg.accel_uncertainty * dt ** 2
    return float(np.hypot(np.linalg.norm(est - true), growth))


def annotate(objects: list[dict], ego: dict | None, beliefs: dict, decision_ms: float,
             deadline_ms: dict, cfg: SceneConfig) -> list[dict]:
    """Add relevance, tier, estimation error and mode quality to each object."""
    if not objects:
        return objects
    if ego is None:
        occl = np.ones(len(objects))
        ego_xy, ego_v = np.zeros(2), np.zeros(2)
    else:
        ego_xy = np.array([ego["x"], ego["y"]])
        ego_v = np.array([ego.get("vx", 0.0), ego.get("vy", 0.0)])
        occl = occlusion_from_ego(objects, ego_xy, cfg)
    for obj, o in zip(objects, occl):
        d_min, t_min = closest_approach(ego_xy, ego_v, (obj["x"], obj["y"]),
                                        (obj.get("vx", 0.0), obj.get("vy", 0.0)), cfg.horizon_s)
        conflict = float(np.exp(-d_min / cfg.conflict_scale_m))
        critical = bool(o >= cfg.critical_occlusion and d_min <= cfg.critical_distance_m)
        tier = "crit" if critical else "norm"
        obj.update(
            occlusion=float(o), d_min=d_min, t_min=t_min, conflict=conflict, tier=tier,
            relevance=cfg.base_relevance + (1 - cfg.base_relevance) * float(o)
            * (cfg.hidden_floor + (1 - cfg.hidden_floor) * conflict),
            ego_distance=float(np.hypot(obj["x"] - ego_xy[0], obj["y"] - ego_xy[1])),
            receiver_error=receiver_error(obj, beliefs.get(obj["track_id"]),
                                          decision_ms + deadline_ms[tier], cfg),
            belief_confidence=float(beliefs.get(obj["track_id"], {}).get("delivery_prob", 0.0)),
            kin_quality=kinematic_mode_quality(obj),
        )
    return objects


def update_value(obj: dict, mode: Mode, loss_hat: float, cfg: SceneConfig) -> float:
    """Expected reduction in receiver awareness loss from sending `obj` in `mode`.

    value = w_j * h(e_j) * P_del(m) * (eta + (1 - eta) * q_{j,m}),
    with h(e) = 1 - exp(-e / eps), q = 1 for box modes and the object-specific
    BEV IoU of the class-prior reconstruction for the kinematic mode.
    """
    # Broadcast has no acknowledgement: the last transmission reached the
    # receiver only with probability belief_confidence.
    has = obj.get("belief_confidence", 0.0)
    h_known = 1.0 - np.exp(-obj["receiver_error"] / cfg.error_scale_m)
    h = has * h_known + (1.0 - has)
    quality = 1.0 if mode.sends_box else obj["kin_quality"]
    eta = cfg.position_weight
    return float(obj["relevance"] * h * delivery_probability(mode, loss_hat)
                 * (eta + (1 - eta) * quality))


def ground_truth_criticality(gt_now, gt_future, ego_future, hidden, cfg: SceneConfig,
                             distance_m: float = 5.0) -> np.ndarray:
    """Offline evaluation label: hidden from the vehicle and on a conflicting path.

    gt_future[k] / ego_future are (T, 2) arrays sampled at common timestamps
    over the horizon; an object is conflicting if their minimum distance is
    below `distance_m`.
    """
    critical = np.zeros(len(gt_now), dtype=bool)
    for k in range(len(gt_now)):
        traj = gt_future[k]
        n = min(len(traj), len(ego_future))
        if n == 0:
            continue
        d = np.linalg.norm(traj[:n] - ego_future[:n], axis=1)
        critical[k] = bool(hidden[k] and d.min() <= distance_m)
    return critical
