import importlib.util
import sys
from pathlib import Path

import pytest


def official_repo(root: Path) -> Path:
    repo = root / "external" / "DAIR-V2X"
    evaluator = repo / "v2x" / "v2x_utils" / "eval_utils.py"
    if not evaluator.is_file():
        pytest.skip("pinned DAIR-V2X checkout is created by run_all.sh")
    return repo


def test_dair_label_prediction_corner_permutations_agree():
    root = Path(__file__).resolve().parents[1]
    repo = official_repo(root)
    sys.path.insert(0, str(repo / "v2x"))
    spec = importlib.util.spec_from_file_location("metric_adapter", root / "scripts" / "evaluate_v2x_official_metric.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    official = adapter.load_official(repo)

    class Box:
        length_m, width_m, height_m = 4.2, 1.9, 1.6
        x_m, y_m, z_m = 12.0, -3.0, 0.2

    label = adapter.label_corners(Box(), 0.31)
    prediction = adapter.prediction_corners(Box(), 0.31, official)
    assert (label[official.perm_label] == prediction[official.perm_pred]).all()


def test_fast_official_match_finds_overlapping_box():
    root = Path(__file__).resolve().parents[1]
    repo = official_repo(root)
    sys.path.insert(0, str(repo / "v2x"))
    spec = importlib.util.spec_from_file_location("metric_adapter_match", root / "scripts" / "evaluate_v2x_official_metric.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    official = adapter.load_official(repo)

    class Ground:
        length_m, width_m, height_m = 4.2, 1.9, 1.6
        x_m, y_m, z_m = 12.0, -3.0, 0.2
    class Prediction:
        length_m, width_m, height_m = 4.1, 1.8, 1.55
        x_m, y_m, z_m = 12.1, -2.9, 0.2

    gt = adapter.label_corners(Ground(), 0.31)[None, ...]
    pred = adapter.prediction_corners(Prediction(), 0.28, official)[None, ...]
    records, tp = adapter.official_frame_records(pred, [0.9], gt, 0.3, "3d", official)
    assert tp == 1
    assert records[0]["type"] == "tp"


def test_controller_metric_matches_pinned_dair_conventions():
    root = Path(__file__).resolve().parents[1]
    repo = official_repo(root)
    sys.path.insert(0, str(repo / "v2x"))
    adapter_spec = importlib.util.spec_from_file_location("metric_adapter_controller", root / "scripts" / "evaluate_v2x_official_metric.py")
    adapter = importlib.util.module_from_spec(adapter_spec)
    adapter_spec.loader.exec_module(adapter)
    official = adapter.load_official(repo)
    spec = importlib.util.spec_from_file_location("controller_metric", root / "scripts" / "run_experiment.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    received = [
        ({"prediction_id": "high", "object_class": "car", "confidence": 0.9}, 1.0),
        ({"prediction_id": "low", "object_class": "car", "confidence": 0.8}, 1.0),
    ]
    candidates = {"g1": [("high", 0.4), ("low", 0.8)], "g2": [("high", 0.7)]}
    assert runner.official_match(received, candidates, ["g1", "g2"], 0.3) == {"high", "low"}
    records = [(0.9, True), (0.8, False), (0.7, True)]
    upstream = official.compute_ap([{"score": score, "type": "tp" if tp else "fp"} for score, tp in records], 2)
    assert runner.official_ap(records, 2) == upstream
