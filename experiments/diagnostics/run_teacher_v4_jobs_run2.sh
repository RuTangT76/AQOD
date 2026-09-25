#!/usr/bin/env bash
set -euo pipefail
cd /root/trust_mvp
export LD_LIBRARY_PATH="/root/trust_mvp/.driver-550.54.14${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4
py=/root/trust_mvp/.venv-qwen35/bin/python
envpy=/root/trust_mvp/.venv-alfworld/bin/python
root=/root/trust_mvp/assets/alfworld/json_2.1.1/train
panel=/root/trust_mvp/analysis/aqod_teacher_v4_budget_selection.json
training=/root/trust_mvp/reports/aqod-teacher-v4-budget-run2
sha256sum -c <<'HASHES'
b0eb95aef682e311cf258034ca5e4a738b89cacce9862584a3b51e03ac49d8f4  trustlab/teacher_alfworld_grpo.py
c49bd3e20908c2830d5147462ed9647583d71eb8f958af4cbf384c09b1e70884  trustlab/aqod_prompt.py
1116c0bcd53d1caf8e6ebe12e85cc38d5eb80fc0a381b5467bb826a7977083a7  trustlab/eval_aqod_teacher_v4.py
8970df8a9692a57d2f48206efb8d83ae2c992a9d4382f6127d90f9a39e71cea5  analysis/aqod_teacher_v4_budget_selection.json
02b89be2c985afacace26d7da5e02d2fe2996ae9b108667a230f21ba20931a48  analysis/aqod_gate_t_v2_selection.json
HASHES
evaluate() {
  local role=$1 model=$2 output=$3
  shift 3
  "$py" -u -m trustlab.eval_aqod_teacher_v4 --role "$role" --model "$model" \
    --environment-python "$envpy" --train-root "$root" --selection-manifest "$panel" \
    --train-manifest "$panel" --output "$output" --device cuda:0 \
    --prompt-format current_commands --max-steps 40 --max-new-tokens 32 \
    --context-length 8192 --seed 2026092501 "$@"
}
case "${1:-}" in
train)
  export CUDA_VISIBLE_DEVICES=0
  test ! -e "$training"
  echo "$(date -u +%FT%TZ) V4 training wrapper PID $$"
  "$py" -u -m trustlab.teacher_alfworld_grpo --model models/Qwen3.5-9B \
    --environment-python "$envpy" --train-root "$root" --selection-manifest "$panel" \
    --output "$training" --device cuda:0 --prompt-format current_commands \
    --context-length 8192 --max-new-tokens 32 --max-train-tokens 2048 --max-steps 40 \
    --group-size 4 --rank 8 --lr 1e-5 --beta 0.01 --temperature 0.8 --top-p 0.95 --seed 2026092501
  checkpoint=$("$py" - "$training" <<'PY'
import hashlib,json,sys
from pathlib import Path
import torch
from safetensors.torch import load_file
root=Path(sys.argv[1])
state=json.loads((root/'state.json').read_text())
if not state.get('complete') or state.get('groups')!=180 or state.get('updates',0)<1:
    raise SystemExit('Incomplete V4 training; Gate T teacher evaluation skipped')
checkpoint=root/'snapshots'/f"update-{state['updates']:03d}"
path=checkpoint/'adapter_model.safetensors'
tensors=load_file(str(path),device='cpu')
if not tensors or not all(torch.isfinite(t).all().item() for t in tensors.values()):
    raise SystemExit('Final adapter parameters not finite; no checkpoint substitution')
record={'checkpoint':str(checkpoint),'adapter_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'all_parameters_finite':True,'selection':'last saved update after complete 180 groups'}
with (root/'frozen_checkpoint.json').open('x') as f: json.dump(record,f,indent=2)
print(checkpoint)
PY
)
  echo "$(date -u +%FT%TZ) frozen checkpoint $checkpoint"
  evaluate teacher_rl models/Qwen3.5-9B reports/aqod-gate-t-v4-rl-run2 --adapter "$checkpoint"
  ;;
controls)
  export CUDA_VISIBLE_DEVICES=1
  test ! -e reports/aqod-gate-t-v4-raw-run2
  test ! -e reports/aqod-gate-t-v4-student-run2
  echo "$(date -u +%FT%TZ) V4 controls wrapper PID $$"
  evaluate teacher_raw models/Qwen3.5-9B reports/aqod-gate-t-v4-raw-run2
  evaluate student_base models/Qwen3.5-2B reports/aqod-gate-t-v4-student-run2
  ;;
*) echo 'Expected train or controls' >&2; exit 2 ;;
esac
echo "$(date -u +%FT%TZ) V4 $1 complete"
