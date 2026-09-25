# AQOD

AQOD is a research prototype for **adaptive teacher trust** in multi-turn agent learning. Its intended progression is **Trust → Question → Discriminate → Explore**. This repository publishes the AQOD-owned method contract, an experimental veRL adapter, experiment code snapshots, and CPU tests.

## Status

The trust controller and adapter are **provisional research code**. The adapter's Gate1 path is conditional on a qualified teacher and completed Gate0 evidence. The online AQOD training loop and full GPU experiment are not established by this release. The configuration under `aqod_verl/configs/` is an unexecuted engineering recipe with paths specific to the original experiment server; replace those paths and satisfy its gate checks before use. Do not cite the CPU tests or GPU smoke scripts as benchmark evidence.

## Layout

| Path | Purpose |
| --- | --- |
| `aqod_online_trust.py` | One student-relative value posterior, query budget, and trust decisions. |
| `aqod_verl/` | Gate1 adapter contracts, environment bridge, objective, and trainer entry point. |
| `tests/` | Synthetic CPU contract checks. |
| `examples/verl_preflight/` | Small server-specific GPU integration probes. |
| `experiments/` | Historical teacher implementation, diagnostics, and veRL restart scripts; see its status notes. |

The adapter imports veRL and other runtime components but does **not** vendor their source. For the current ATOD-based adapter, see [`aqod_verl/README.md`](aqod_verl/README.md) before attempting a GPU run.

## Local checks

Use Python 3.10 or newer. Install the package in a fresh environment and run the CPU contract tests:

```bash
python -m pip install -e .
python -m unittest discover -s tests -p 'test_*.py' -v
```

The preflight scripts need a Linux GPU runtime, model weights, and paths adjusted to the machine. They are not part of the CPU test suite.

## Research boundaries

AQOD estimates whether querying a teacher is valuable for the **current student** at the current public state. A valid paired label requires the same public history, student policy, and remaining action budget on both continuations. The controller does not use privileged environment state. Method changes and evaluation choices should be logged before examining outcome panels.

ATOD, veRL, and other public work informed infrastructure review. AQOD's adaptive teacher-trust rule is a separate research proposal; no ATOD or veRL repository is copied here. See the source links in the adapter documentation for upstream dependencies.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
