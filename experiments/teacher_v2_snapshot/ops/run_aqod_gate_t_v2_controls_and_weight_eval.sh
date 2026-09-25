#!/usr/bin/env bash
set -euo pipefail
cd /root/trust_mvp
export LD_LIBRARY_PATH="/root/trust_mvp/.driver-550.54.14${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4
python=/root/trust_mvp/.venv-qwen35/bin/python
model2=/root/trust_mvp/models/Qwen3.5-2B
model9=/root/trust_mvp/models/Qwen3.5-9B
env_python=/root/trust_mvp/.venv-alfworld/bin/python
train_root=/root/trust_mvp/assets/alfworld/json_2.1.1/train
selection=/root/trust_mvp/analysis/aqod_gate_t_v2_selection.json
resume_root=/root/trust_mvp/reports/aqod-teacher-alfworld-grpo-v2-weight-resume1
student=reports/aqod-gate-t-v2-student-base-rerun1
raw=reports/aqod-gate-t-v2-teacher-raw-rerun1
weight=reports/aqod-gate-t-v2-teacher-weight-resume1
pid=$(cat logs/aqod-teacher-alfworld-grpo-v2-weight-resume1.python.pid)

while ps -p "$pid" -o args= 2>/dev/null | grep -q 'trustlab.teacher_alfworld_grpo_weight_resume'; do
  sleep 30
done
echo "Weights-only continuation exited at $(date -u +'%F %T UTC')."

# These frozen controls are useful for both the exploratory continuation and
# the independent full rerun, even if the continuation itself failed.
CUDA_VISIBLE_DEVICES=0 "$python" -u -m trustlab.eval_aqod_teacher \
  --role student_base --model "$model2" --environment-python "$env_python" \
  --train-root "$train_root" --selection-manifest "$selection" --train-manifest "$selection" \
  --output "$student" --device cuda:0 --prompt-format current_commands \
  --max-steps 40 --max-new-tokens 32 --context-length 8192 --seed 2026092317 \
  > logs/aqod-gate-t-v2-student-base-rerun1.log 2>&1
CUDA_VISIBLE_DEVICES=0 "$python" -u -m trustlab.eval_aqod_teacher \
  --role teacher_raw --model "$model9" --environment-python "$env_python" \
  --train-root "$train_root" --selection-manifest "$selection" --train-manifest "$selection" \
  --output "$raw" --device cuda:0 --prompt-format current_commands \
  --max-steps 40 --max-new-tokens 32 --context-length 8192 --seed 2026092317 \
  > logs/aqod-gate-t-v2-teacher-raw-rerun1.log 2>&1

checkpoint=$($python - <<'PY'
import json
from pathlib import Path
root=Path('/root/trust_mvp/reports/aqod-teacher-alfworld-grpo-v2-weight-resume1')
state=json.loads((root/'state.json').read_text())
if not state.get('complete') or state.get('total_completed_groups')!=30:
    raise SystemExit('Weights-only continuation incomplete; controls finished, exploratory teacher evaluation skipped')
if state['continuation_updates']:
    checkpoint=root/'snapshots'/f"update-{state['total_updates']:03d}"
else:
    checkpoint=Path('/root/trust_mvp/reports/aqod-teacher-alfworld-grpo-v2/snapshots/update-004')
if not (checkpoint/'adapter_model.safetensors').is_file():
    raise SystemExit('Selected continuation checkpoint missing')
print(checkpoint)
PY
)
echo "Exploratory weights-only checkpoint: $checkpoint"
CUDA_VISIBLE_DEVICES=0 "$python" -u -m trustlab.eval_aqod_teacher \
  --role teacher_rl --model "$model9" --adapter "$checkpoint" \
  --environment-python "$env_python" --train-root "$train_root" \
  --selection-manifest "$selection" --train-manifest "$selection" \
  --output "$weight" --device cuda:0 --prompt-format current_commands \
  --max-steps 40 --max-new-tokens 32 --context-length 8192 --seed 2026092317 \
  > logs/aqod-gate-t-v2-teacher-weight-resume1.log 2>&1
"$python" -m trustlab.analyze_gate_t \
  --student "$student" --raw "$raw" --rl "$weight" \
  --output reports/aqod-gate-t-v2-weight-resume1-analysis \
  > logs/aqod-gate-t-v2-weight-resume1-analysis.log 2>&1
"$python" analysis/analyze_gate_t_v2_detailed.py \
  --student "$student" --raw "$raw" --rl "$weight" \
  --output reports/aqod-gate-t-v2-weight-resume1-detailed.json \
  > logs/aqod-gate-t-v2-weight-resume1-detailed.log 2>&1
echo 'Exploratory weights-only Gate T comparison complete; scientific review required.'
