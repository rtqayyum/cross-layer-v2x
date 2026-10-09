"""V2X-Seq-SPD loader: world-frame RSU objects, ego poses and evaluation labels.

Outputs (all positions in the dataset's world frame, times in ms):
  rsu_objects   one row per infrastructure object per infrastructure frame
                (track id, class, box, causal finite-difference velocity)
  ego           connected-vehicle LiDAR pose per vehicle frame (+ causal velocity)
  gt            cooperative labels per vehicle frame with hidden / critical flags
  ego_objects   vehicle-side labels (the vehicle's own perception, for late fusion)

`rsu_objects` and `ego` are the only tables a policy can see, and only the rows
at or before its decision time.  `gt` and `ego_objects` are evaluation-only.
"""
from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import transform_points, transform_yaw


@dataclass(frozen=True)
class V2XSeqConfig:
    horizon_s: float = 5.0
    conflict_distance_m: float = 5.0
    eval_radius_m: float = 80.0
    hidden_sides: tuple = ("inf",)          # from_side values meaning "infrastructure only"
    max_velocity_gap_ms: float = 400.0


def _matrix(value: dict) -> np.ndarray:
    if "transform" in value:
        value = value["transform"]
    m = np.eye(4)
    m[:3, :3] = np.asarray(value["rotation"], float).reshape(3, 3)
    m[:3, 3] = np.asarray(value["translation"], float).reshape(3)
    return m


def _yaw_of(matrix: np.ndarray) -> float:
    return float(np.arctan2(matrix[1, 0], matrix[0, 0]))


class Archive:
    """Lazy reader over the official V2X-Seq-SPD zip."""

    def __init__(self, path: Path):
        self.zip = zipfile.ZipFile(path)
        self.names = self.zip.namelist()
        self.index: dict[str, dict[str, str]] = {}
        for name in self.names:
            if name.endswith(".json"):
                self.index.setdefault(str(Path(name).parent), {})[Path(name).stem] = name

    def json(self, name: str):
        return json.loads(self.zip.read(name))

    def find_dir(self, *parts: str) -> dict[str, str]:
        """Member map of the first directory whose path ends with the given parts."""
        suffix = "/".join(parts)
        for directory, members in self.index.items():
            if directory.endswith(suffix):
                return members
        return {}

    def data_info(self, side: str) -> list[dict]:
        members = [n for n in self.names if n.endswith(f"{side}/data_info.json")]
        if len(members) != 1:
            raise SystemExit(f"expected one {side}/data_info.json, found {len(members)}")
        return self.json(members[0])


def _boxes(objects: list[dict]) -> pd.DataFrame:
    rows = []
    for o in objects:
        d, loc = o.get("3d_dimensions", {}), o.get("3d_location", {})
        rows.append({"track_id": str(o.get("track_id", "")), "object_class": str(o.get("type", "")).lower(),
                     "length": float(d.get("l", np.nan)), "width": float(d.get("w", np.nan)),
                     "height": float(d.get("h", np.nan)), "x": float(loc.get("x", np.nan)),
                     "y": float(loc.get("y", np.nan)), "z": float(loc.get("z", np.nan)),
                     "yaw": float(o.get("rotation", 0.0)), "from_side": str(o.get("from_side", "")),
                     "occluded_state": o.get("occluded_state")})
    return pd.DataFrame(rows)


def _to_world(frame: pd.DataFrame, matrix: np.ndarray, shift=None) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    xyz = transform_points(matrix, out[["x", "y", "z"]].to_numpy(float))
    if shift is not None:
        xyz[:, :2] += shift
    out[["x", "y", "z"]] = xyz
    out["yaw"] = transform_yaw(matrix, out["yaw"].to_numpy(float))
    return out


def _causal_velocity(frame: pd.DataFrame, key: str, max_gap_ms: float) -> pd.DataFrame:
    """Velocity from the current and previous observation of the same key only."""
    frame = frame.sort_values([key, "t_ms"]).copy()
    prev = frame.groupby(key)[["x", "y", "t_ms"]].shift(1)
    dt = (frame["t_ms"] - prev["t_ms"]) / 1000.0
    ok = (dt > 0) & (dt * 1000.0 <= max_gap_ms)
    frame["vx"] = np.where(ok, (frame["x"] - prev["x"]) / dt.where(ok, 1.0), 0.0)
    frame["vy"] = np.where(ok, (frame["y"] - prev["y"]) / dt.where(ok, 1.0), 0.0)
    return frame


def load(path: Path, cfg: V2XSeqConfig = V2XSeqConfig(), sequences: list[str] | None = None) -> dict:
    ar = Archive(path)
    coop = ar.data_info("cooperative")
    inf_info = {str(r["frame_id"]): r for r in ar.data_info("infrastructure-side")}
    veh_info = {str(r["frame_id"]): r for r in ar.data_info("vehicle-side")}
    coop_labels = ar.find_dir("cooperative", "label")
    inf_labels = ar.find_dir("infrastructure-side", "label", "virtuallidar") or ar.find_dir(
        "infrastructure-side", "label", "lidar")
    veh_labels = ar.find_dir("vehicle-side", "label", "lidar")
    inf_calib = ar.find_dir("infrastructure-side", "calib", "virtuallidar_to_world")
    veh_l2n = ar.find_dir("vehicle-side", "calib", "lidar_to_novatel")
    veh_n2w = ar.find_dir("vehicle-side", "calib", "novatel_to_world")
    for name, table in (("cooperative/label", coop_labels), ("infrastructure label", inf_labels),
                        ("vehicle label", veh_labels), ("virtuallidar_to_world", inf_calib),
                        ("lidar_to_novatel", veh_l2n), ("novatel_to_world", veh_n2w)):
        if not table:
            raise SystemExit(f"V2X-Seq-SPD archive has no {name} directory; run scripts/inspect_data.py")

    pairs = pd.DataFrame([{
        "sequence": str(p["vehicle_sequence"]), "vehicle_frame": str(p["vehicle_frame"]),
        "infrastructure_frame": str(p["infrastructure_frame"]),
        "dx": float((p.get("system_error_offset") or {}).get("delta_x", 0.0) or 0.0),
        "dy": float((p.get("system_error_offset") or {}).get("delta_y", 0.0) or 0.0)}
        for p in coop])
    if sequences is not None:
        pairs = pairs[pairs.sequence.isin(sequences)]

    veh_pose: dict[str, np.ndarray] = {}
    ego_rows, gt_frames, ego_obj_frames = [], [], []
    for p in pairs.itertuples(index=False):
        v = p.vehicle_frame
        if v not in veh_pose:
            veh_pose[v] = _matrix(ar.json(veh_n2w[v])) @ _matrix(ar.json(veh_l2n[v]))
        pose = veh_pose[v]
        t_ms = float(veh_info[v]["pointcloud_timestamp"]) / 1000.0
        ego_rows.append({"sequence": p.sequence, "vehicle_frame": v, "t_ms": t_ms,
                         "x": pose[0, 3], "y": pose[1, 3], "yaw": _yaw_of(pose)})
        gt = _to_world(_boxes(ar.json(coop_labels[v])), pose)
        gt["sequence"], gt["vehicle_frame"], gt["t_ms"] = p.sequence, v, t_ms
        gt_frames.append(gt)
        if v in veh_labels:
            own = _to_world(_boxes(ar.json(veh_labels[v])), pose)
            own["sequence"], own["vehicle_frame"], own["t_ms"] = p.sequence, v, t_ms
            ego_obj_frames.append(own)
    ego = _causal_velocity(pd.DataFrame(ego_rows).drop_duplicates(["sequence", "vehicle_frame"]),
                           "sequence", cfg.max_velocity_gap_ms)

    # Infrastructure objects in the world frame.  The dataset's per-pair system
    # error offset (vehicle LiDAR frame) is applied as in the official
    # cooperative evaluation, rotated into the world frame.
    inf_rows = []
    offsets = pairs.drop_duplicates("infrastructure_frame").set_index("infrastructure_frame")
    for f, p in offsets.iterrows():
        if f not in inf_labels:
            continue
        shift = veh_pose[p.vehicle_frame][:2, :2] @ np.array([p.dx, p.dy])
        boxes = _to_world(_boxes(ar.json(inf_labels[f])), _matrix(ar.json(inf_calib[f])), shift)
        boxes["sequence"], boxes["infrastructure_frame"] = p.sequence, f
        boxes["t_ms"] = float(inf_info[f]["pointcloud_timestamp"]) / 1000.0
        inf_rows.append(boxes)
    rsu = pd.concat(inf_rows, ignore_index=True)
    rsu["track_key"] = rsu["sequence"] + ":" + rsu["track_id"]
    rsu = _causal_velocity(rsu, "track_key", cfg.max_velocity_gap_ms)
    rsu["confidence"] = 1.0

    gt = pd.concat(gt_frames, ignore_index=True)
    gt = _label_gt(gt, ego, cfg)
    ego_objects = pd.concat(ego_obj_frames, ignore_index=True) if ego_obj_frames else pd.DataFrame()
    return {"rsu_objects": rsu, "ego": ego, "gt": gt, "ego_objects": ego_objects, "pairs": pairs}


def _label_gt(gt: pd.DataFrame, ego: pd.DataFrame, cfg: V2XSeqConfig) -> pd.DataFrame:
    """Hidden = infrastructure-only label; critical = hidden and future path conflict.

    Future paths come from the same track in later cooperative labels and the
    ego's later poses (offline ground truth; never visible to a policy).
    """
    side = gt["from_side"].str.lower()
    hidden_pattern = "|".join(re.escape(s) for s in cfg.hidden_sides)
    gt["hidden"] = side.str.fullmatch(hidden_pattern).fillna(False)
    gt["critical"] = False
    gt["in_range"] = False
    steps = np.arange(0.0, cfg.horizon_s + 1e-9, 0.5) * 1000.0
    for seq, frame in gt.groupby("sequence"):
        e = ego[ego.sequence == seq].sort_values("t_ms")
        et, ex, ey = e.t_ms.to_numpy(), e.x.to_numpy(), e.y.to_numpy()
        tracks = {k: g.sort_values("t_ms") for k, g in frame.groupby("track_id")}
        for idx, row in frame.iterrows():
            ego_x, ego_y = np.interp(row.t_ms, et, ex), np.interp(row.t_ms, et, ey)
            gt.at[idx, "in_range"] = bool(np.hypot(row.x - ego_x, row.y - ego_y) <= cfg.eval_radius_m)
            if not gt.at[idx, "hidden"]:
                continue
            tr = tracks[row.track_id]
            times = row.t_ms + steps
            times = times[(times <= tr.t_ms.max()) & (times <= et.max())]
            if len(times) == 0:
                continue
            ox, oy = np.interp(times, tr.t_ms, tr.x), np.interp(times, tr.t_ms, tr.y)
            px, py = np.interp(times, et, ex), np.interp(times, et, ey)
            gt.at[idx, "critical"] = bool(np.min(np.hypot(ox - px, oy - py)) <= cfg.conflict_distance_m)
    return gt


def to_sequences(tables: dict) -> dict[str, dict]:
    """Group tables into the per-sequence structure consumed by replay.run_episode."""
    out = {}
    rsu, ego, gt, own = tables["rsu_objects"], tables["ego"], tables["gt"], tables["ego_objects"]
    cols = ["track_id", "object_class", "x", "y", "z", "length", "width", "height", "yaw", "vx", "vy",
            "confidence"]
    for seq, objs in rsu.groupby("sequence"):
        epochs = []
        for (f, t), g in objs.groupby(["infrastructure_frame", "t_ms"]):
            records = g[cols].assign(capture_ms=t, track_id=g["track_key"]).to_dict("records")
            epochs.append({"t_ms": float(t), "objects": records})
        epochs.sort(key=lambda e: e["t_ms"])
        e = ego[ego.sequence == seq].sort_values("t_ms")
        frames = []
        own_seq = own[own.sequence == seq] if len(own) else own
        for (v, t), g in gt[gt.sequence == seq].groupby(["vehicle_frame", "t_ms"]):
            ego_objs = own_seq[own_seq.vehicle_frame == v] if len(own_seq) else own_seq
            frames.append({"t_ms": float(t), "gt": g.to_dict("records"),
                           "ego_objects": ego_objs.assign(confidence=1.0).to_dict("records")
                           if len(ego_objs) else []})
        frames.sort(key=lambda f: f["t_ms"])
        out[seq] = {"epochs": epochs, "frames": frames,
                    "ego": e[["t_ms", "x", "y", "vx", "vy", "yaw"]].to_numpy(float)}
    return out
