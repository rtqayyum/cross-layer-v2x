"""ETSI Collective Perception Message (CPM) payload model.

Each perceived object can be sent in one of several representation modes.
A mode changes both the number of bytes and the content the receiver gets,
so the evaluation recomputes IoU from what was actually transmitted.

Sizes follow the CPM structure of ETSI TS 103 324 encoded with ASN.1 UPER,
using the per-message and per-object sizes commonly used in the CPM
literature (121-byte header including security, 35 bytes for a basic object).
They are parameters, and the paper reports a sensitivity analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Class-prior dimensions (l, w, h) used when the box is not transmitted.
CLASS_PRIOR = {
    "car": (4.5, 1.9, 1.6),
    "van": (5.0, 2.0, 2.1),
    "truck": (9.0, 2.6, 3.2),
    "bus": (11.0, 2.7, 3.2),
    "cyclist": (1.8, 0.7, 1.6),
    "motorcyclist": (2.0, 0.8, 1.6),
    "tricyclist": (2.5, 1.2, 1.7),
    "pedestrian": (0.6, 0.6, 1.7),
}
DEFAULT_PRIOR = (4.5, 1.9, 1.6)


@dataclass(frozen=True)
class Mode:
    name: str
    bytes_per_object: int
    sends_box: bool        # dimensions + heading included
    copies: int            # 2 = one blind repetition (sidelink retransmission)
    position_step_m: float  # quantization of the transmitted position


MODES: tuple[Mode, ...] = (
    Mode("kinematic", 35, False, 1, 0.01),
    Mode("box", 60, True, 1, 0.01),
    Mode("box_rep", 60, True, 2, 0.01),
)
CPM_HEADER_BYTES = 121


def mode_bytes(mode: Mode) -> int:
    return mode.bytes_per_object * mode.copies


def prior_dimensions(object_class: str) -> tuple[float, float, float]:
    return CLASS_PRIOR.get(str(object_class).lower(), DEFAULT_PRIOR)


def transmitted_state(obj: dict, mode: Mode) -> dict:
    """State the receiver reconstructs from one transmitted object."""
    step = mode.position_step_m
    out = {
        "object_class": obj["object_class"],
        "x": round(obj["x"] / step) * step,
        "y": round(obj["y"] / step) * step,
        "z": obj["z"],
        "vx": obj.get("vx", 0.0),
        "vy": obj.get("vy", 0.0),
        "capture_ms": obj["capture_ms"],
        "confidence": obj.get("confidence", 1.0),
        "track_id": obj["track_id"],
    }
    if mode.sends_box:
        out.update(length=obj["length"], width=obj["width"], height=obj["height"], yaw=obj["yaw"])
    else:
        length, width, height = prior_dimensions(obj["object_class"])
        speed = float(np.hypot(out["vx"], out["vy"]))
        # Heading is inferred from motion; a parked object keeps the axis default.
        yaw = float(np.arctan2(out["vy"], out["vx"])) if speed > 0.5 else 0.0
        out.update(length=length, width=width, height=height, yaw=yaw)
    return out


def delivery_probability(mode: Mode, loss: float) -> float:
    """Erasure-only success probability, independent copies (stated assumption)."""
    loss = float(np.clip(loss, 0.0, 1.0))
    return 1.0 - loss ** mode.copies
