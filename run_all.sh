#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
MPL_DIR="${MPLCONFIGDIR:-/tmp/cross-layer-v2x-matplotlib}"
DATA="$ROOT/data/processed/study"
OUT="$ROOT/results/reproduction"

: "${V2X_SEQ_ARCHIVE:?Set V2X_SEQ_ARCHIVE to the official V2X-Seq-SPD metadata ZIP}"
: "${V2X_INFRA_ARCHIVE:?Set V2X_INFRA_ARCHIVE to the official infrastructure-side Velodyne .zip; keep its .z01 beside it}"
: "${SEE_V2X_ARCHIVE:?Set SEE_V2X_ARCHIVE to the official SEE-V2X emulating_applications.tar.gz}"
: "${DATASET_ACCESS_CONFIRMED:?Set DATASET_ACCESS_CONFIRMED=yes only after confirming research access}"
[[ "$DATASET_ACCESS_CONFIRMED" == "yes" ]] || { echo "DATASET_ACCESS_CONFIRMED must equal yes" >&2; exit 2; }

if [[ ! -x "$PYTHON" ]]; then
  python3 -m venv "$ROOT/.venv"
  "$ROOT/.venv/bin/pip" install -r "$ROOT/requirements.txt"
  PYTHON="$ROOT/.venv/bin/python"
fi

clone_pinned() {
  local url="$1" directory="$2" commit="$3"
  if [[ ! -d "$directory/.git" ]]; then
    git clone "$url" "$directory"
  fi
  git -C "$directory" fetch --all --tags
  git -C "$directory" checkout --detach "$commit"
}

clone_pinned https://github.com/AIR-THU/DAIR-V2X.git "$ROOT/external/DAIR-V2X" c885c54af0c34bc515fa9ca8b5e8fda76a15462c
clone_pinned https://github.com/AIR-THU/DAIR-V2X-Seq.git "$ROOT/external/DAIR-V2X-Seq" 8a5b4f82da5ab5c6eeacbf961aafbf2c883f437c
clone_pinned https://github.com/UCR-CISL/SEE-V2X.git "$ROOT/external/SEE-V2X" daba452da8c0683d68a63a858e1c861494c59efd

mkdir -p "$ROOT/data/interim/v2x_seq" "$DATA" "$ROOT/data/raw/see_v2x" "$OUT/analysis" "$OUT/capacity_validation" "$MPL_DIR"

if [[ ! -f "$ROOT/data/interim/v2x_seq/source_manifest.json" ]]; then
  V2X_SHA="$(shasum -a 256 "$V2X_SEQ_ARCHIVE" | awk '{print $1}')"
  "$PYTHON" "$ROOT/scripts/build_v2x_seq_source_index.py" \
    --archive "$V2X_SEQ_ARCHIVE" \
    --source-id v2x_seq_spd_full_real \
    --source-sha256 "$V2X_SHA" \
    --dataset-role full_candidate \
    --license-status verified \
    --output-dir "$ROOT/data/interim/v2x_seq"
fi

if [[ ! -f "$DATA/dataset_manifest.json" ]]; then
  cp "$ROOT/config/dataset_access.example.json" "$ROOT/data/interim/dataset_access.json"
  "$PYTHON" "$ROOT/scripts/freeze_full_dataset.py" \
    --source-index "$ROOT/data/interim/v2x_seq" \
    --license-receipt "$ROOT/data/interim/dataset_access.json" \
    --output-dir "$DATA"
fi

if [[ ! -f "$DATA/model_predictions.parquet" ]]; then
  COMBINED="${TMPDIR:-/tmp}/cross_layer_v2x_infrastructure.zip"
  rm -f "$COMBINED"
  trap 'rm -f "$COMBINED"' EXIT INT TERM
  zip -b "${TMPDIR:-/tmp}" -s 0 "$V2X_INFRA_ARCHIVE" --out "$COMBINED"
  "$PYTHON" "$ROOT/scripts/run_causal_lidar_detector.py" \
    --archive "$COMBINED" \
    --pairs "$DATA/cooperative_pairs.parquet" \
    --role infrastructure \
    --output "$DATA/infrastructure_predictions.jsonl" \
    --receipt-output "$DATA/perception_model_receipt.json" \
    --resume
  "$PYTHON" "$ROOT/scripts/import_model_predictions.py" \
    --jsonl "$DATA/infrastructure_predictions.jsonl" \
    --frozen-dataset-dir "$DATA" \
    --receipt "$DATA/perception_model_receipt.json" \
    --output "$DATA/model_predictions.parquet"
fi

if [[ ! -f "$ROOT/data/raw/see_v2x/.complete" ]]; then
  "$ROOT/scripts/extract_see_v2x_subset.sh" "$SEE_V2X_ARCHIVE" "$ROOT/data/raw/see_v2x"
  touch "$ROOT/data/raw/see_v2x/.complete"
fi

"$PYTHON" "$ROOT/scripts/normalize_see_v2x_traces.py" \
  --input-dir "$ROOT/data/raw/see_v2x" \
  --output "$ROOT/data/processed/see_v2x_network.parquet"
"$PYTHON" "$ROOT/scripts/normalize_see_v2x_sci.py" \
  --input-dir "$ROOT/data/raw/see_v2x" \
  --output "$ROOT/data/processed/see_v2x_sci.parquet"
MPLCONFIGDIR="$MPL_DIR" "$PYTHON" "$ROOT/scripts/build_sci_calibrated_capacity.py" \
  --application "$ROOT/data/processed/see_v2x_network.parquet" \
  --sci "$ROOT/data/processed/see_v2x_sci.parquet" \
  --output "$ROOT/data/processed/see_v2x_capacity_lower.parquet" \
  --validation-dir "$OUT/capacity_validation"

"$PYTHON" "$ROOT/scripts/refine_causal_predictions.py" \
  --input "$DATA/model_predictions.parquet" \
  --output "$DATA/model_predictions_refined.parquet" \
  --receipt "$DATA/perception_model_refined_receipt.json"
"$PYTHON" "$ROOT/scripts/align_v2x_predictions.py" \
  --archive "$V2X_SEQ_ARCHIVE" \
  --pairs "$DATA/cooperative_pairs.parquet" \
  --predictions "$DATA/model_predictions_refined.parquet" \
  --output "$DATA/model_predictions_vehicle_frame.parquet"

MPLCONFIGDIR="$MPL_DIR" "$PYTHON" "$ROOT/scripts/evaluate_v2x_official_metric.py" \
  --predictions "$DATA/model_predictions_vehicle_frame.parquet" \
  --ground-truth "$DATA/ground_truth_objects.parquet" \
  --official-repo "$ROOT/external/DAIR-V2X" \
  --output "$OUT/official_dair_metrics.json"
MPLCONFIGDIR="$MPL_DIR" "$PYTHON" "$ROOT/scripts/build_official_iou_candidates.py" \
  --predictions "$DATA/model_predictions_vehicle_frame.parquet" \
  --ground-truth "$DATA/ground_truth_objects.parquet" \
  --official-repo "$ROOT/external/DAIR-V2X" \
  --output "$DATA/official_car_iou_candidates.parquet"

"$PYTHON" "$ROOT/scripts/run_experiment.py" --config "$ROOT/config/final.yaml" --resume
MPLCONFIGDIR="$MPL_DIR" "$PYTHON" "$ROOT/scripts/analyze_results.py" \
  --results "$OUT/run_summaries.parquet" --output-dir "$OUT/analysis"
"$PYTHON" "$ROOT/scripts/analyze_timing_assurance.py" \
  --results "$OUT/run_summaries.parquet" --output-dir "$OUT/analysis" --deadline-ms 100
"$PYTHON" "$ROOT/scripts/audit_publication_gates.py" \
  --experiment-root "$ROOT" --output "$OUT/publication_gate_audit.json"

echo "Full reproduction complete: $OUT"
