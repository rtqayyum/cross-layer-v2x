from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "final" / "run_summaries.parquet"


def test_frozen_campaign_shape_and_provenance() -> None:
    frame = pd.read_parquet(RESULTS)
    assert len(frame) == 4212
    assert frame["sequence_id"].nunique() == 39
    assert frame["trace_id"].nunique() == 48
    assert frame["controller_id"].nunique() == 9
    assert set(frame["network_provenance"]) == {
        "MEASURED_SEE_V2X_CROSSFIT_CAPACITY_REPLAY"
    }
    assert set(frame["application_provenance"]) == {
        "MEASURED_ALIGNED_PERCEPTION_OUTPUT"
    }
    assert not frame["physical_radio_claim"].any()


def test_proposed_controller_frozen_means() -> None:
    frame = pd.read_parquet(RESULTS)
    proposed = frame[frame["controller_id"] == "proposed"]
    expected = {
        "critical_on_time_ratio": 0.6739677417068823,
        "official_car_3d_recall_03": 0.10877876054498538,
        "official_car_3d_ap_03": 0.05278008425113857,
        "background_sla_ratio": 1.0,
    }
    for column, value in expected.items():
        assert abs(float(proposed[column].mean()) - value) < 1e-12

