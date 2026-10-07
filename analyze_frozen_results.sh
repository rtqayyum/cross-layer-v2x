#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
MPL_DIR="${MPLCONFIGDIR:-/tmp/cross-layer-v2x-matplotlib}"

[[ -x "$PYTHON" ]] || {
  echo "Missing Python environment. Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 2
}

mkdir -p "$MPL_DIR" "$ROOT/results/reproduction/analysis"
MPLCONFIGDIR="$MPL_DIR" "$PYTHON" "$ROOT/scripts/analyze_results.py" \
  --results "$ROOT/results/final/run_summaries.parquet" \
  --output-dir "$ROOT/results/reproduction/analysis"
"$PYTHON" "$ROOT/scripts/analyze_timing_assurance.py" \
  --results "$ROOT/results/final/run_summaries.parquet" \
  --output-dir "$ROOT/results/reproduction/analysis" \
  --deadline-ms 100

echo "Analysis complete: $ROOT/results/reproduction/analysis"

