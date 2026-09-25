#!/usr/bin/env bash
set -euo pipefail
cd /root/trust_mvp
python=/root/trust_mvp/.venv-qwen35/bin/python
panel=analysis/aqod_teacher_v4_budget_selection.json
training=reports/aqod-teacher-v4-budget-run2
student=reports/aqod-gate-t-v4-student-run2
raw=reports/aqod-gate-t-v4-raw-run2
rl=reports/aqod-gate-t-v4-rl-run2
analysis_output=reports/aqod-gate-t-v4-analysis-run2
linkage_output=reports/aqod-gate-t-v4-checkpoint-audit-run2

sha256sum -c <<'HASHES'
8970df8a9692a57d2f48206efb8d83ae2c992a9d4382f6127d90f9a39e71cea5  analysis/aqod_teacher_v4_budget_selection.json
f1339d9c10596224e169ea5bf0dbaba5d9bd82d9215d3d755b50cb859bb37b01  analysis/analyze_gate_t_v4.py
8f4787854ad49d166f956d015cdc4e6356634487f7842e4ee835c7be146efc7e  analysis/audit_teacher_v4_checkpoint.py
HASHES
test ! -e "$analysis_output"
test ! -e "$linkage_output"

complete() {
  "$python" - "$training" "$student" "$raw" "$rl" <<'PY'
import json,sys
from pathlib import Path
for root in map(Path,sys.argv[1:]):
    path=root/'state.json'
    if not path.is_file():
        raise SystemExit(1)
    state=json.loads(path.read_text())
    if state.get('stage')=='failed':
        raise SystemExit(f'Failed run: {root}: {state.get("error")}')
    if not state.get('complete'):
        raise SystemExit(1)
PY
}

for attempt in $(seq 1 720); do
  if complete; then
    break
  fi
  if ! ps -p 21597 -o args= 2>/dev/null | grep -q 'run_teacher_v4_jobs_run2.sh train'; then
    echo 'V4 trainer wrapper exited before all reports completed' >&2
    exit 1
  fi
  sleep 120
done
complete
"$python" analysis/audit_teacher_v4_checkpoint.py \
  --training "$training" --rl "$rl" --panel "$panel" --output "$linkage_output"
"$python" analysis/analyze_gate_t_v4.py \
  --student "$student" --raw "$raw" --rl "$rl" --panel "$panel" \
  --output "$analysis_output"
sha256sum "$linkage_output/audit.json" "$analysis_output/metrics.json"
echo 'V4 postrun checks complete. Human scientific review remains required before Gate0.'
