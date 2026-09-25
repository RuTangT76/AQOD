#!/usr/bin/env bash
set -euo pipefail
cd /root/trust_mvp

echo 'c98a4e7bc803abf9979f3e648dde87716507924bf2fd98285983e2113c74e7b5  analysis/aqod_gpu1_valid_seen_controls_selection.json' | sha256sum -c -
echo '6f666f647c0963f001ebdb085629275177c12d6a6d35559a8995fe437c26fdda  trustlab/eval_gpu1_valid_seen_controls.py' | sha256sum -c -
echo 'c49bd3e20908c2830d5147462ed9647583d71eb8f958af4cbf384c09b1e70884  trustlab/aqod_prompt.py' | sha256sum -c -
echo '20e6bb7f6820c6d1006a171555f21302aa4287348042d2585ed920749c2531c8  trustlab/alfworld_bridge_seen.py' | sha256sum -c -

raw=reports/aqod-gpu1-valid-seen-raw9b-run1
student=reports/aqod-gpu1-valid-seen-student2b-run1
test ! -e "$raw" && test ! -e "$student"

.venv-qwen35/bin/python -u -m trustlab.eval_gpu1_valid_seen_controls \
  --role teacher_raw --model models/Qwen3.5-9B \
  --environment-python .venv-alfworld/bin/python \
  --train-root assets/alfworld/json_2.1.1/valid_seen \
  --selection-manifest analysis/aqod_gpu1_valid_seen_controls_selection.json \
  --output "$raw" --device cuda:1 --prompt-format current_commands \
  --max-steps 40 --max-new-tokens 32 --context-length 8192 --seed 2026092508

.venv-qwen35/bin/python -u -m trustlab.eval_gpu1_valid_seen_controls \
  --role student_base --model models/Qwen3.5-2B \
  --environment-python .venv-alfworld/bin/python \
  --train-root assets/alfworld/json_2.1.1/valid_seen \
  --selection-manifest analysis/aqod_gpu1_valid_seen_controls_selection.json \
  --output "$student" --device cuda:1 --prompt-format current_commands \
  --max-steps 40 --max-new-tokens 32 --context-length 8192 --seed 2026092508
