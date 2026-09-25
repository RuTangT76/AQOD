#!/usr/bin/env bash
# One-time, read-only Gate 0 quality checks and preregistered analysis.
set -euo pipefail

cd /root/trust_mvp
collect=reports/aqod-fallback-gate0-valid-unseen-collect-run1
branch=reports/aqod-fallback-gate0-valid-unseen-branch-run1
teacher=reports/aqod-fallback-gate0-teacher-fullgame-run1
panel=analysis/aqod_fallback_gate0_panels_valid_unseen.json
out=reports/aqod-fallback-gate0-analysis-run1
python=.venv-qwen35/bin/python

printf '%s  %s\n' \
  89f4637e2583914fb509e2b83faefd3d36f346e7dc9ef7e8e3ea0a1bc79994e5 trustlab/aqod_fallback_gate0.py \
  e2ce5b6877b0af8a17d2cc795533b4fb18edf5f6a0551b4edee4b8a472533990 "$panel" \
  ac3dee4aa1eee7fe7a6774c1f42ed86f8c3515dad063978422a24907b9061c0c analysis/analyze_fallback_gate0.py \
  8caea68598d7dad639d686e396b6d82425409805fc47f63773d61d8932df5767 analysis/audit_fallback_gate0_descriptives.py \
  13c6b2599ecfe5d1f902564170e8dd57ef74fad6a9d08a91f1e18f2c61d99ab5 analysis/audit_fallback_student_replay.py \
  | sha256sum -c -

"$python" -B - "$collect" "$branch" "$teacher" <<'PY'
import json, sys
from pathlib import Path
for path in map(Path, sys.argv[1:]):
    state = json.loads((path / "state.json").read_text())
    if state.get("stage") != "complete" or state.get("complete") is not True:
        raise SystemExit(f"Incomplete report: {path}: {state}")
print("all three raw reports are complete")
PY

mkdir "$out"
"$python" -B analysis/audit_fallback_student_replay.py \
  --collect "$collect" --branch "$branch" > "$out/student_replay_audit.json"
"$python" -B - "$out/student_replay_audit.json" <<'PY'
import json, sys
from pathlib import Path
audit = json.loads(Path(sys.argv[1]).read_text())
if not audit["all_nonerror_replays_match"]:
    raise SystemExit("Student replay mismatch: inspect raw pairs before inference")
print("student replay consistency passed; replay errors remain explicit")
PY

"$python" -B analysis/analyze_fallback_gate0.py \
  --collect "$collect" --branch "$branch" --teacher-full-game "$teacher" \
  --panel "$panel" --output "$out/primary_analysis.json"
"$python" -B analysis/audit_fallback_gate0_descriptives.py \
  --collect "$collect" --branch "$branch" --panel "$panel" \
  --output "$out/action_validity_audit.json"

sha256sum "$out"/*.json > "$out/analysis_sha256.txt"
echo "$(date -u '+%F %T UTC') Gate 0 analysis artifacts complete: $out"
