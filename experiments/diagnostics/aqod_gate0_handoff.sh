#!/usr/bin/env bash
# One-time outcome-blind handoff after the frozen Gate 0 collection completes.
set -euo pipefail

cd /root/trust_mvp
mkdir -p logs
if ! mkdir logs/aqod-fallback-gate0-handoff-run1.lock 2>/dev/null; then
  echo "handoff already registered; refusing duplicate" >&2
  exit 1
fi

collect=reports/aqod-fallback-gate0-valid-unseen-collect-run1
branch=reports/aqod-fallback-gate0-valid-unseen-branch-run1
teacher=reports/aqod-fallback-gate0-teacher-fullgame-run1
panel=analysis/aqod_fallback_gate0_panels_valid_unseen.json
collect_pid=39052

echo "$(date -u '+%F %T UTC') waiting for collection PID ${collect_pid}"
while :; do
  status=$(.venv-qwen35/bin/python -B - "$collect/state.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    print("missing")
else:
    s = json.loads(p.read_text())
    print("complete" if s.get("complete") is True and
          s.get("stage") == "complete" else s.get("stage", "unknown"))
PY
)
  if [ "$status" = complete ]; then
    process_state=$(ps -o stat= -p "$collect_pid" 2>/dev/null || true)
    if [ -z "$process_state" ] || [[ "$process_state" == Z* ]]; then
      break
    fi
  elif [ "$status" = failed ]; then
    echo "$(date -u '+%F %T UTC') collection failed; handoff stopped" >&2
    exit 2
  elif ! kill -0 "$collect_pid" 2>/dev/null; then
    echo "$(date -u '+%F %T UTC') collection PID missing before completion; handoff stopped" >&2
    exit 3
  fi
  process_args=$(ps -o args= -p "$collect_pid" 2>/dev/null || true)
  if [[ "$process_args" != *"trustlab.aqod_fallback_gate0 --phase collect"* ]] ||
     [[ "$process_args" != *"aqod-fallback-gate0-valid-unseen-collect-run1"* ]]; then
    echo "$(date -u '+%F %T UTC') collection PID has changed identity; handoff stopped" >&2
    exit 5
  fi
  sleep 120
done

echo "$(date -u '+%F %T UTC') collection complete; validating immutable inputs"
printf '%s  %s\n' \
  89f4637e2583914fb509e2b83faefd3d36f346e7dc9ef7e8e3ea0a1bc79994e5 trustlab/aqod_fallback_gate0.py \
  e2ce5b6877b0af8a17d2cc795533b4fb18edf5f6a0551b4edee4b8a472533990 "$panel" \
  f8ad044bc75996cf4674cc460f2f812a0bd017b84951ff4851c7c39d2e7e5889 trustlab/eval_fallback_gate0_teacher_full_game.py \
  ac3dee4aa1eee7fe7a6774c1f42ed86f8c3515dad063978422a24907b9061c0c analysis/analyze_fallback_gate0.py \
  | sha256sum -c -
.venv-qwen35/bin/python -B - "$collect" "$panel" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "/root/trust_mvp/analysis")
from analyze_fallback_gate0 import load_collection
load_collection(Path(sys.argv[1]), Path(sys.argv[2]))
print("30-game collection, panel hashes, counts, and frozen selection verified")
PY

if [ -e "$branch" ] || [ -e "$teacher" ]; then
  echo "A downstream report already exists; refusing duplicate launch" >&2
  exit 4
fi

nohup .venv-qwen35/bin/python -u -m trustlab.aqod_fallback_gate0 \
  --phase branch \
  --collect-report /root/trust_mvp/reports/aqod-fallback-gate0-valid-unseen-collect-run1 \
  --student-model /root/trust_mvp/models/Qwen3.5-2B \
  --environment-python /root/trust_mvp/.venv-alfworld/bin/python \
  --train-root /root/trust_mvp/assets/alfworld/json_2.1.1/valid_unseen \
  --output /root/trust_mvp/reports/aqod-fallback-gate0-valid-unseen-branch-run1 \
  --student-device cuda:0 \
  > logs/aqod-fallback-gate0-valid-unseen-branch-run1.log 2>&1 < /dev/null &
branch_pid=$!
printf '%s\n' "$branch_pid" > logs/aqod-fallback-gate0-valid-unseen-branch-run1.pid

nohup .venv-qwen35/bin/python -u -m trustlab.eval_fallback_gate0_teacher_full_game \
  --selection-manifest /root/trust_mvp/analysis/aqod_fallback_gate0_panels_valid_unseen.json \
  --teacher-model /root/shared-nvme/models/ATOD_ckpt/alfworld_grpo_qwen3_4b/actor_hf \
  --environment-python /root/trust_mvp/.venv-alfworld/bin/python \
  --train-root /root/trust_mvp/assets/alfworld/json_2.1.1/valid_unseen \
  --output /root/trust_mvp/reports/aqod-fallback-gate0-teacher-fullgame-run1 \
  --teacher-device cuda:1 \
  > logs/aqod-fallback-gate0-teacher-fullgame-run1.log 2>&1 < /dev/null &
teacher_pid=$!
printf '%s\n' "$teacher_pid" > logs/aqod-fallback-gate0-teacher-fullgame-run1.pid
echo "$(date -u '+%F %T UTC') launched branch PID ${branch_pid} and teacher-full-game PID ${teacher_pid}"
