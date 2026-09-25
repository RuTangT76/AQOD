#!/usr/bin/env bash
set -euo pipefail
cd /root/trust_mvp
export LD_LIBRARY_PATH="/root/trust_mvp/.driver-550.54.14${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4
python=/root/trust_mvp/.venv-qwen35/bin/python
model9=/root/trust_mvp/models/Qwen3.5-9B
env_python=/root/trust_mvp/.venv-alfworld/bin/python
train_root=/root/trust_mvp/assets/alfworld/json_2.1.1/train
selection=/root/trust_mvp/analysis/aqod_gate_t_v2_selection.json
teacher_root=/root/trust_mvp/reports/aqod-teacher-alfworld-grpo-v2-full-rerun1
student=reports/aqod-gate-t-v2-student-base-rerun1
raw=reports/aqod-gate-t-v2-teacher-raw-rerun1
rl=reports/aqod-gate-t-v2-teacher-full-rerun1
pid=$(cat logs/aqod-teacher-alfworld-grpo-v2-full-rerun1.python.pid)

while ps -p "$pid" -o args= 2>/dev/null | grep -q 'trustlab.teacher_alfworld_grpo'; do
  sleep 30
done
checkpoint=$($python - <<'PY'
import json
from pathlib import Path
root=Path('/root/trust_mvp/reports/aqod-teacher-alfworld-grpo-v2-full-rerun1')
state=json.loads((root/'state.json').read_text())
if not state.get('complete') or state.get('groups')!=30 or state.get('updates',0)<1:
    raise SystemExit('Full frozen-protocol rerun incomplete; formal Gate T evaluation skipped')
checkpoint=root/'snapshots'/f"update-{state['updates']:03d}"
if not (checkpoint/'adapter_model.safetensors').is_file():
    raise SystemExit('Selected full-rerun checkpoint missing')
print(checkpoint)
PY
)
echo "Formal frozen-protocol checkpoint: $checkpoint"
CUDA_VISIBLE_DEVICES=1 "$python" -u -m trustlab.eval_aqod_teacher \
  --role teacher_rl --model "$model9" --adapter "$checkpoint" \
  --environment-python "$env_python" --train-root "$train_root" \
  --selection-manifest "$selection" --train-manifest "$selection" \
  --output "$rl" --device cuda:0 --prompt-format current_commands \
  --max-steps 40 --max-new-tokens 32 --context-length 8192 --seed 2026092317 \
  > logs/aqod-gate-t-v2-teacher-full-rerun1.log 2>&1

# The controls run on GPU 0 after the weights-only run exits. Wait for both
# complete states, then calculate the primary three-way paired comparison.
for attempt in $(seq 1 720); do
  if "$python" - "$student" "$raw" <<'PY'
import json,sys
from pathlib import Path
for p in map(Path,sys.argv[1:]):
    state=p/'state.json'
    if not state.exists() or not json.loads(state.read_text()).get('complete'):
        raise SystemExit(1)
PY
  then
    break
  fi
  sleep 30
done
"$python" -m trustlab.analyze_gate_t \
  --student "$student" --raw "$raw" --rl "$rl" \
  --output reports/aqod-gate-t-v2-full-rerun1-analysis \
  > logs/aqod-gate-t-v2-full-rerun1-analysis.log 2>&1
"$python" analysis/analyze_gate_t_v2_detailed.py \
  --student "$student" --raw "$raw" --rl "$rl" \
  --output reports/aqod-gate-t-v2-full-rerun1-detailed.json \
  > logs/aqod-gate-t-v2-full-rerun1-detailed.log 2>&1
echo 'Frozen-protocol Gate T raw metrics complete; scientific review required.'
