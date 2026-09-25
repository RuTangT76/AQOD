# Experimental veRL adapter

This package contains AQOD's Gate1 on-policy distillation interface and the adaptive-trust state contract. The current GPU entry point follows infrastructure APIs in the [ATOD repository](https://github.com/TanQitai/ATOD), including its veRL fork and verl-agent trajectory collector. A generic `verl` installation is **not** a drop-in replacement for this adapter. No ATOD source is vendored here.

`main.py` checks the selected game panel, Gate T and Gate0 evidence, model-export hashes, required modules, and GPU count before a run. The example `configs/gate1_2x4090.json` has unset gate evidence and server-specific paths by design. It is not a preregistered benchmark command or a passing experiment.

Gate1 uses current-student trajectories and frozen-teacher token probabilities. The sampled-token OPD signal is separate from task reward. The teacher worker is configured without the student's LoRA, and tokenizer identity must hold on actual prompts and generated tokens. `trust.py` serializes the provisional student-relative posterior, pending queries, consumed budget, and interaction costs. It does not establish that the online AQOD phase has run.

After providing compatible upstream packages, the complete `trustlab` environment bridge, a qualified frozen teacher, and matching evidence, use:

```bash
python -m aqod_verl.main --config /path/to/recipe.json --upstream /path/to/ATOD --check
python -m aqod_verl.main --config /path/to/recipe.json --upstream /path/to/ATOD --run
```

`--check` does not launch training. `--run` records resolved configuration, versions, code hashes, metrics, and failures. The included CPU tests check method contracts only. GPU capacity, environment parity, and actual experimental outcomes require separate validation.
