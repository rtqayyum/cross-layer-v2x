"""End-to-end pipeline on fake archives in the official layouts (no real data)."""
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run(*args, cwd):
    env = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=cwd, env=env,
                   capture_output=True, text=True)


def test_pipeline_end_to_end(tmp_path):
    run(ROOT / "tests/make_fake_archives.py", tmp_path / "raw", cwd=ROOT)
    data, res = tmp_path / "data", tmp_path / "results"
    run(ROOT / "scripts/prepare_v2.py", "--v2x-seq", tmp_path / "raw/V2X-Seq-SPD.zip",
        "--see-v2x", tmp_path / "raw/emulating_applications.tar.gz", "--out", data, cwd=ROOT)
    run(ROOT / "scripts/build_envelopes_v2.py", "--data", data, "--results", res, "--max-train", "20000", cwd=ROOT)
    run(ROOT / "scripts/run_campaign_v2.py", "--data", data, "--out", res / "runs.parquet", "--limit", "60", cwd=ROOT)
    run(ROOT / "scripts/analyze_campaign_v2.py", "--runs", res / "runs.parquet", "--results", res, cwd=ROOT)
    runs = pd.read_parquet(res / "runs.parquet")
    assert len(runs) == 60
    assert (res / "macros.tex").read_text().count("\\newcommand") > 20
    calib = pd.read_csv(res / "envelope_calibration.csv")
    aci = calib[calib.method == "aci"].set_index("alpha")["miss"]
    assert all(abs(aci[a] - a) < 0.02 for a in aci.index)
