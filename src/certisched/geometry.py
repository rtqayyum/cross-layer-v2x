"""Planar and 3-D box geometry used by the scheduler and the evaluator."""
from __future__ import annotations

import numpy as np


def rotation_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def transform_points(matrix: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to an (N, 3) array."""
    xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
    return (matrix[:3, :3] @ xyz.T).T + matrix[:3, 3]


def transform_yaw(matrix: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Rotate headings (about z) by the rotation part of a transform."""
    yaw = np.asarray(yaw, dtype=float)
    heading = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], axis=-1)
    rotated = heading @ matrix[:3, :3].T
    return np.arctan2(rotated[..., 1], rotated[..., 0])


def bev_corners(x: float, y: float, length: float, width: float, yaw: float) -> np.ndarray:
    """Counter-clockwise BEV rectangle corners, shape (4, 2)."""
    c, s = np.cos(yaw), np.sin(yaw)
    dx = np.array([length, length, -length, -length]) / 2.0
    dy = np.array([width, -width, -width, width]) / 2.0
    corners = np.stack([x + c * dx - s * dy, y + s * dx + c * dy], axis=1)
    return corners[::-1]  # (+,+),(+,-),(-,-),(-,+) is clockwise; reverse it


def _polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clipping of a convex polygon by a convex CCW polygon."""
    output = list(subject)
    for i in range(len(clipper)):
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        edge = b - a
        inputs, output = output, []
        if not inputs:
            break

        def inside(p: np.ndarray) -> bool:
            return edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]) >= -1e-12

        def intersect(p: np.ndarray, q: np.ndarray) -> np.ndarray:
            d = q - p
            denom = edge[0] * d[1] - edge[1] * d[0]
            if abs(denom) < 1e-15:
                return q
            t = (edge[0] * (a[1] - p[1]) - edge[1] * (a[0] - p[0])) / -denom
            return p + t * d

        prev = inputs[-1]
        for cur in inputs:
            if inside(cur):
                if not inside(prev):
                    output.append(intersect(prev, cur))
                output.append(cur)
            elif inside(prev):
                output.append(intersect(prev, cur))
            prev = cur
    return np.array(output) if output else np.zeros((0, 2))


def bev_iou(box_a: tuple, box_b: tuple) -> float:
    """BEV IoU of two (x, y, length, width, yaw) rotated rectangles."""
    pa, pb = bev_corners(*box_a), bev_corners(*box_b)
    if np.linalg.norm(np.subtract(box_a[:2], box_b[:2])) > 0.5 * (
        np.hypot(box_a[2], box_a[3]) + np.hypot(box_b[2], box_b[3])
    ):
        return 0.0
    inter = _polygon_area(_clip(pa, pb))
    union = box_a[2] * box_a[3] + box_b[2] * box_b[3] - inter
    return float(inter / union) if union > 0 else 0.0


def iou_3d(box_a: tuple, box_b: tuple) -> float:
    """3-D IoU of (x, y, z_center, length, width, height, yaw) boxes."""
    xa, ya, za, la, wa, ha, ra = box_a
    xb, yb, zb, lb, wb, hb, rb = box_b
    inter_bev = bev_iou((xa, ya, la, wa, ra), (xb, yb, lb, wb, rb)) 
    if inter_bev <= 0:
        return 0.0
    # Recover intersection area from the IoU definition.
    area_a, area_b = la * wa, lb * wb
    inter_area = inter_bev * (area_a + area_b) / (1.0 + inter_bev)
    overlap_h = max(0.0, min(za + ha / 2, zb + hb / 2) - max(za - ha / 2, zb - hb / 2))
    inter = inter_area * overlap_h
    union = area_a * ha + area_b * hb - inter
    return float(inter / union) if union > 0 else 0.0


def segment_hits_box(p0: np.ndarray, p1: np.ndarray, box: tuple, margin: float = 0.0) -> bool:
    """True if the segment p0->p1 crosses a BEV box (x, y, l, w, yaw)."""
    x, y, length, width, yaw = box
    c, s = np.cos(-yaw), np.sin(-yaw)
    rot = np.array([[c, -s], [s, c]])
    a = rot @ (np.asarray(p0, float) - (x, y))
    b = rot @ (np.asarray(p1, float) - (x, y))
    hx, hy = length / 2 + margin, width / 2 + margin
    # Liang-Barsky clipping against the axis-aligned box in object coordinates.
    t0, t1 = 0.0, 1.0
    d = b - a
    for p, q in ((-d[0], a[0] + hx), (d[0], hx - a[0]), (-d[1], a[1] + hy), (d[1], hy - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return False
            continue
        r = q / p
        if p < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return False
    return True
