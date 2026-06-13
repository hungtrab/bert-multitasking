#!/usr/bin/env bash
set -euo pipefail

cd /home/hungtq/bert-multitasking

WAIT_PID="${1:-}"
if [[ -n "$WAIT_PID" ]]; then
  echo "[$(date)] Waiting for existing training PID $WAIT_PID"
  while ps -p "$WAIT_PID" >/dev/null 2>&1; do
    sleep 60
  done
  echo "[$(date)] Existing training PID $WAIT_PID finished"
fi

run_config() {
  local cfg="$1"
  local exp="$2"
  local log="train_${exp}.log"

  echo "[$(date)] START $exp ($cfg)"
  rm -rf "runs/${exp}"
  /home/hungtq/snc_env/bin/python -m scripts.train \
    --config "configs/${cfg}" \
    --device cuda 2>&1 | tee "${log}"
  echo "[$(date)] DONE $exp"

  /home/hungtq/snc_env/bin/python - "$exp" <<'PY'
import json
import pathlib
import sys

exp = sys.argv[1]
p = pathlib.Path("runs") / exp / "history.json"
print(f"SUMMARY {exp}")
if not p.exists():
    print("  history missing")
    raise SystemExit(0)
h = json.loads(p.read_text())
best = max(h, key=lambda r: r["combined"])
last = h[-1]
print("  best", best)
print("  last ", last)
PY
}

run_config \
  "report_best_self_impl_adamw_20e_plus10_lr5e6.yaml" \
  "report_best_self_impl_adamw_20e_plus10_lr5e6"

run_config \
  "report_best_self_impl_adamw_20e_plus10_stspara.yaml" \
  "report_best_self_impl_adamw_20e_plus10_stspara"

run_config \
  "report_best_self_impl_muon20_to_adamw_stspara.yaml" \
  "report_best_self_impl_muon20_to_adamw_stspara"

run_config \
  "report_best_self_impl_adamw_20e_plus10_stsheavy.yaml" \
  "report_best_self_impl_adamw_20e_plus10_stsheavy"

run_config \
  "report_best_self_impl_muon20_to_adamw_balanced.yaml" \
  "report_best_self_impl_muon20_to_adamw_balanced"

run_config \
  "report_best_self_impl_muon20_plus10_lr5e5.yaml" \
  "report_best_self_impl_muon20_plus10_lr5e5"

run_config \
  "report_best_hf_10e.yaml" \
  "report_best_hf_10e"

echo "[$(date)] Queue finished"
