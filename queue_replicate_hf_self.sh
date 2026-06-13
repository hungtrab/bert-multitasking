#!/usr/bin/env bash
set -euo pipefail

cd /home/hungtq/bert-multitasking

WAIT_PID="${1:-}"
if [[ -n "$WAIT_PID" ]]; then
  echo "[$(date)] Waiting for previous queue PID $WAIT_PID"
  while ps -p "$WAIT_PID" >/dev/null 2>&1; do
    sleep 60
  done
  echo "[$(date)] Previous queue PID $WAIT_PID finished"
fi

EXP="report_best_self_impl_bs32_10e"
CFG="${EXP}.yaml"
LOG="train_${EXP}.log"

echo "[$(date)] START $EXP ($CFG)"
rm -rf "runs/${EXP}"
/home/hungtq/snc_env/bin/python -m scripts.train \
  --config "configs/${CFG}" \
  --device cuda 2>&1 | tee "${LOG}"
echo "[$(date)] DONE $EXP"

/home/hungtq/snc_env/bin/python - "$EXP" <<'PY'
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
