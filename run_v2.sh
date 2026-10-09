#!/usr/bin/env bash
# CertiSched full pipeline on the official datasets (run on the data machine).
#   V2X_SEQ_ARCHIVE=/abs/V2X-Seq-SPD.zip SEE_V2X_ARCHIVE=/abs/emulating_applications.tar.gz ./run_v2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
: "${V2X_SEQ_ARCHIVE:?set V2X_SEQ_ARCHIVE}"
: "${SEE_V2X_ARCHIVE:?set SEE_V2X_ARCHIVE}"
DATA="$ROOT/data/processed/v2"
OUT="$ROOT/results/v2"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$DATA" "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*"; }

if [[ ! -f "$DATA/links.parquet" ]]; then
  log "stage 1/4: prepare scene tables and service windows"
  "$PY" "$ROOT/scripts/prepare_v2.py" --v2x-seq "$V2X_SEQ_ARCHIVE" --see-v2x "$SEE_V2X_ARCHIVE" --out "$DATA"
fi
cp "$DATA/prepare_summary.json" "$OUT/prepare_summary.json"
if [[ ! -f "$DATA/envelopes.parquet" ]]; then
  log "stage 2/4: leave-one-scenario-out models and conformal envelopes"
  "$PY" "$ROOT/scripts/build_envelopes_v2.py" --data "$DATA" --results "$OUT"
fi
log "stage 3/4: paired replay campaign (resumable)"
"$PY" "$ROOT/scripts/run_campaign_v2.py" --data "$DATA" --out "$OUT/runs.parquet"
log "stage 4/4: statistics, figure data and LaTeX macros"
"$PY" "$ROOT/scripts/analyze_campaign_v2.py" --runs "$OUT/runs.parquet" --results "$OUT"
log "done: $OUT"
