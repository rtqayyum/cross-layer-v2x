# Instructions for the local Claude session (machine with the datasets)

The cloud session that writes the code cannot see the datasets. This file tells a
Claude Code session running on the data machine what to run and what to send back.
Never commit raw dataset content (point clouds, label rows, packet records).
Commit only the files listed under "Send back".

## Setup (once)

```bash
git fetch origin claude/charming-wozniak-jr6k5b
git checkout claude/charming-wozniak-jr6k5b
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Set the archive paths (adjust to where the files live):

```bash
export V2X_SEQ_ARCHIVE=/abs/path/V2X-Seq-SPD.zip
export V2X_INFRA_ARCHIVE=/abs/path/V2X-Seq-SPD-infrastructure-side-velodyne.zip   # .z01 beside it
export SEE_V2X_ARCHIVE=/abs/path/emulating_applications.tar.gz
# optional, if present:
export V2X_VEHICLE_ARCHIVE=/abs/path/V2X-Seq-SPD-vehicle-side-velodyne.zip
```

## Step 0 - schema inspection (minutes)

```bash
.venv/bin/python scripts/inspect_data.py \
  --v2x-seq "$V2X_SEQ_ARCHIVE" --v2x-infra "$V2X_INFRA_ARCHIVE" \
  --see-v2x "$SEE_V2X_ARCHIVE" ${V2X_VEHICLE_ARCHIVE:+--v2x-vehicle "$V2X_VEHICLE_ARCHIVE"} \
  --output results/inspection/data_schema.json
```

Send back: `results/inspection/data_schema.json`
(commit it, push to `claude/charming-wozniak-jr6k5b`).

For a split infrastructure archive (.zip + .z01), the inspector only needs the
`.zip`; if it fails, combine first with `zip -s 0 in.zip --out combined.zip`.

## Step 1 - full CertiSched pipeline (about 1-3 hours on a laptop CPU)

```bash
git pull
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests -q            # must pass before running on data
./run_v2.sh                                      # uses V2X_SEQ_ARCHIVE and SEE_V2X_ARCHIVE
```

The run is resumable: if it stops, run `./run_v2.sh` again.  For a quick
check first, `PYTHONPATH=src .venv/bin/python scripts/run_campaign_v2.py --limit 200`
after stages 1-2.

If stage 1 or 2 fails, do not edit the method code.  Commit the full error
text to `results/v2/ERROR.txt` together with `results/inspection/data_schema.json`
and push, so the loader can be fixed against the real schema.

### Send back (commit and push to `claude/charming-wozniak-jr6k5b`)

Everything in `results/v2/` (CSV, JSON, `macros.tex`, `runs.parquet`).  These
contain aggregate metrics only.  Do **not** commit `data/processed/`.

Always `git pull` first and follow the newest version of this file.
