#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then echo "usage: $0 ARCHIVE OUTPUT_DIR" >&2; exit 2; fi
archive="$1"; output="$2"; mkdir -p "$output"
# Balanced real subset: three UE->RSU perception links for every one of the
# 16 perception scenarios, plus all available SCI files (20 scenarios).
tar -xzf "$archive" -C "$output" \
  'emulating_applications/bsm_perception/indoor_timeExtend/intersection_*/rx_rsu_tx_*_perception.csv' \
  'emulating_applications/bsm_perception/outdoor_intersection/intersection_*/rx_rsu_tx_*_perception.csv' \
  'emulating_applications/perception_only/indoor_timeExtend/intersection_*/rx_rsu_tx_*_perception.csv' \
  'emulating_applications/perception_only/outdoor_intersection/intersection_*/rx_rsu_tx_*_perception.csv' \
  'emulating_applications/*/*/intersection_*/sci.csv'
echo "SEE-V2X study subset extracted to $output"
