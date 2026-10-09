"""TEST-ONLY: write tiny archives in the official V2X-Seq-SPD and SEE-V2X layouts.

The content is synthetic (certisched.synthetic); it exists only to exercise the
loaders and the full pipeline end to end.  Never used for reported results.
"""
from __future__ import annotations

import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from certisched.synthetic import make_link, make_sequence  # noqa: E402

T0_US = 1_626_155_000_000_000


def _label(o, side=None):
    d = {"type": o["object_class"].capitalize(), "track_id": str(o["track_id"]),
         "3d_dimensions": {"l": o["length"], "w": o["width"], "h": o["height"]},
         "3d_location": {"x": o["x"], "y": o["y"], "z": o["z"]}, "rotation": o["yaw"]}
    if side:
        d["from_side"] = side
    return d


def write_v2x_seq(path: Path, sequences: int = 3) -> None:
    coop, inf_info, veh_info = [], [], []
    files: dict[str, object] = {}
    root = "V2X-Seq-SPD"
    eye = {"rotation": np.eye(3).tolist(), "translation": [[0.0], [0.0], [0.0]]}
    for s in range(sequences):
        seq = make_sequence(seed=s, frames=60)
        sid = f"{s:04d}"
        for k, (ep, fr) in enumerate(zip(seq["epochs"], seq["frames"])):
            inf_id, veh_id = f"{s}{k:05d}", f"{s}{k:05d}"
            t_inf = T0_US + s * 10**9 + int(ep["t_ms"] * 1000)
            t_veh = T0_US + s * 10**9 + int(fr["t_ms"] * 1000)
            inf_info.append({"frame_id": inf_id, "pointcloud_timestamp": str(t_inf), "sequence_id": sid,
                             "intersection_loc": "fake"})
            ex, ey = seq["ego"][k, 1], seq["ego"][k, 2]
            veh_info.append({"frame_id": veh_id, "pointcloud_timestamp": str(t_veh), "sequence_id": sid,
                             "intersection_loc": "fake"})
            coop.append({"vehicle_frame": veh_id, "infrastructure_frame": inf_id, "vehicle_sequence": sid,
                         "infrastructure_sequence": sid, "system_error_offset": {"delta_x": 0, "delta_y": 0}})
            # Infrastructure "virtual LiDAR" frame == world frame; vehicle LiDAR at the ego.
            files[f"{root}/infrastructure-side/label/virtuallidar/{inf_id}.json"] = [
                _label(o) for o in ep["objects"]]
            files[f"{root}/infrastructure-side/calib/virtuallidar_to_world/{inf_id}.json"] = eye
            pose = {"rotation": np.eye(3).tolist(), "translation": [[ex], [ey], [0.0]]}
            files[f"{root}/vehicle-side/calib/novatel_to_world/{veh_id}.json"] = pose
            files[f"{root}/vehicle-side/calib/lidar_to_novatel/{veh_id}.json"] = {"transform": eye}
            local = lambda o: {**o, "x": o["x"] - ex, "y": o["y"] - ey}  # noqa: E731
            files[f"{root}/cooperative/label/{veh_id}.json"] = [
                _label(local(g), "inf" if g["hidden"] else "veh_inf") for g in fr["gt"]]
            files[f"{root}/vehicle-side/label/lidar/{veh_id}.json"] = [
                _label(local(g)) for g in fr["gt"] if not g["hidden"]]
    files[f"{root}/cooperative/data_info.json"] = coop
    files[f"{root}/infrastructure-side/data_info.json"] = inf_info
    files[f"{root}/vehicle-side/data_info.json"] = veh_info
    with zipfile.ZipFile(path, "w") as z:
        for name, value in files.items():
            z.writestr(name, json.dumps(value))


def write_see_v2x(path: Path) -> None:
    scenarios = [("bsm_perception", "outdoor_intersection", "intersection_6"),
                 ("perception_only", "outdoor_intersection", "intersection_3"),
                 ("perception_only", "indoor_timeExtend", "intersection_4"),
                 ("bsm_perception", "indoor_timeExtend", "intersection_7")]
    with tarfile.open(path, "w:gz") as tar:
        def add(name, text):
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for i, (traffic, setting, inter) in enumerate(scenarios):
            base = f"emulating_applications/{traffic}/{setting}/{inter}"
            sci_rows = ["timestamp,mcs_idx,nof_prb_pssch,retransmission"]
            for ue in ("ue1", "ue2"):
                link = make_link(seed=10 * i + int(ue[-1]), windows=1500,
                                 mean_rate_bpms=12.0 if "outdoor" in setting else 25.0)
                rows = ["timestamp,latency (ms),avg_throughput(100ms),avg_packet_loss(100ms),channel_busy_percentage"]
                rng = np.random.default_rng(i)
                for w in link.itertuples():
                    n = max(1, int(round(w.rate_bpms * 100 / 200)))
                    delivered = w.rate_bpms * 100 * (1 - w.loss)
                    for j in range(n):
                        t = T0_US + w.window * 100_000 + int(j * 100_000 / n)
                        lat = float(np.interp(rng.random(), [0, .1, .5, .9, 1],
                                              [w.lat_min_ms, w.lat_p10_ms, w.lat_p50_ms, w.lat_p90_ms, w.lat_max_ms]))
                        rows.append(f"{t},{lat:.3f},{delivered * 8 / 0.1:.1f},{w.loss * 100:.2f},{w.cbr:.1f}%")
                    sci_rows.append(f"{T0_US + w.window * 100_000},{rng.integers(5, 20)},{rng.integers(10, 50)},"
                                    f"{int(rng.random() < 0.3)}")
                add(f"{base}/rx_rsu_tx_{ue}_perception.csv", "\n".join(rows) + "\n")
            add(f"{base}/sci.csv", "\n".join(sci_rows) + "\n")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    write_v2x_seq(out / "V2X-Seq-SPD.zip")
    write_see_v2x(out / "emulating_applications.tar.gz")
    print("fake archives written to", out)
