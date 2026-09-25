#!/usr/bin/env bash
set -euo pipefail
cd /root/trust_mvp

echo 'c98a4e7bc803abf9979f3e648dde87716507924bf2fd98285983e2113c74e7b5  analysis/aqod_gpu1_valid_seen_controls_selection.json' | sha256sum -c -
echo '6f666f647c0963f001ebdb085629275177c12d6a6d35559a8995fe437c26fdda  trustlab/eval_gpu1_valid_seen_controls.py' | sha256sum -c -
echo 'fe898985e4926ece30f57c4741e13256b3f543081afbb57e1b518d7eac205df3  analysis/analyze_gpu1_valid_seen_controls.py' | sha256sum -c -

for attempt in $(seq 1 720); do
  status=$(.venv-qwen35/bin/python - <<'PY'
import json
from pathlib import Path
names = ('aqod-gpu1-valid-seen-raw9b-run1', 'aqod-gpu1-valid-seen-student2b-run1')
states = []
for name in names:
    path = Path('reports') / name / 'state.json'
    states.append(json.loads(path.read_text()) if path.exists() else {})
if any(x.get('stage') == 'failed' for x in states):
    print('failed')
elif all(x.get('stage') == 'complete' and x.get('complete') is True and
         x.get('completed_games') == 48 for x in states):
    print('complete')
else:
    print('wait')
PY
)
  case "$status" in
    complete) break ;;
    failed) echo 'One GPU1 control failed; retaining raw reports' >&2; exit 1 ;;
    wait) sleep 120 ;;
    *) echo "Unexpected report state: $status" >&2; exit 1 ;;
  esac
done

test "$status" = complete || { echo 'Timed out waiting for GPU1 controls' >&2; exit 1; }
output=reports/aqod-gpu1-valid-seen-analysis-run1.json
test ! -e "$output"
.venv-qwen35/bin/python analysis/analyze_gpu1_valid_seen_controls.py \
  --raw reports/aqod-gpu1-valid-seen-raw9b-run1 \
  --student reports/aqod-gpu1-valid-seen-student2b-run1 \
  --panel analysis/aqod_gpu1_valid_seen_controls_selection.json \
  --output "$output"
sha256sum "$output"
