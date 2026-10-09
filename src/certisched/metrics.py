"""Receiver-side detection metrics for the fused (ego + received) object list."""
from __future__ import annotations

import numpy as np

from .geometry import bev_iou, iou_3d

CAR_LIKE = {"car", "van", "truck", "bus"}


def is_car(cls: str) -> bool:
    return str(cls).lower() in CAR_LIKE


def fuse(ego_objects: list[dict], received: list[dict], dedup_m: float = 2.0) -> list[dict]:
    """Late fusion: keep ego detections, add received objects not already seen."""
    fused = [dict(o, score=o.get("confidence", 1.0), source="ego") for o in ego_objects]
    pts = np.array([[o["x"], o["y"]] for o in ego_objects]) if ego_objects else np.zeros((0, 2))
    for o in received:
        if len(pts) and np.min(np.hypot(pts[:, 0] - o["x"], pts[:, 1] - o["y"])) < dedup_m:
            continue
        fused.append(dict(o, score=o.get("confidence", 1.0), source="rsu"))
    return fused


def frame_ap_records(detections: list[dict], gts: list[dict], threshold: float, view: str):
    """Greedy score-ordered matching (DAIR/KITTI convention) for one frame."""
    dets = sorted([d for d in detections if is_car(d["object_class"])], key=lambda d: -d["score"])
    gts = [g for g in gts if is_car(g["object_class"]) and g.get("in_range", True)]
    taken = [False] * len(gts)
    records = []
    for d in dets:
        best, best_iou = -1, threshold
        for i, g in enumerate(gts):
            if taken[i] or abs(d["x"] - g["x"]) > 8 or abs(d["y"] - g["y"]) > 8:
                continue
            if view == "bev":
                value = bev_iou((d["x"], d["y"], d["length"], d["width"], d["yaw"]),
                                (g["x"], g["y"], g["length"], g["width"], g["yaw"]))
            else:
                value = iou_3d((d["x"], d["y"], d["z"], d["length"], d["width"], d["height"], d["yaw"]),
                               (g["x"], g["y"], g["z"], g["length"], g["width"], g["height"], g["yaw"]))
            if value >= best_iou:
                best, best_iou = i, value
        if best >= 0:
            taken[best] = True
        records.append((d["score"], best >= 0))
    return records, len(gts)


def average_precision(records: list[tuple[float, bool]], num_gt: int) -> float:
    """All-point interpolated AP (VOC 2010+), the convention of the DAIR-V2X evaluator core."""
    if num_gt == 0 or not records:
        return float("nan") if num_gt == 0 else 0.0
    ordered = sorted(records, key=lambda r: -r[0])
    tp = np.cumsum([int(r[1]) for r in ordered])
    precision = tp / np.arange(1, len(tp) + 1)
    recall = tp / num_gt
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall = np.r_[0.0, recall]
    return float(np.sum((recall[1:] - recall[:-1]) * precision))
