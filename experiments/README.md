# Experiment code snapshots

These AQOD-authored scripts are included for traceability. They are **not** the current method package and are not automatically run by `pip install -e .`.

- `teacher_v2_snapshot/` contains the earlier server-side GRPO teacher and evaluation source snapshot. Some `trustlab` bridge and model modules are supplied by the original server workspace, so this is not a standalone training distribution.
- `diagnostics/` contains later teacher, interface, panel-selection, and Gate0 diagnostic scripts. Their absolute paths and frozen hashes refer to the original experiment environment. The scripts represent distinct recorded protocols; combining their scores as one benchmark is invalid.
- `verl_restart/` contains the independent V5 panel selector and raw-teacher technical smoke evaluator. They do not establish a qualified RL teacher or an AQOD result.

No raw episode traces, held-out panel manifests, model weights, server credentials, or external repository source are included here. Inspect each script's configuration, panel hash, and environment requirements before running it on another machine. Superseded bundles and the abandoned TRL draft are intentionally omitted from this public code snapshot.
