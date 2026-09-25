"""Analyze three complete, same-game paper-style interface reports.

This script was frozen before either 2B/9B control finished. It does not
select a teacher or modify the AQOD evaluation protocol.
"""

import argparse
import json
import random
from pathlib import Path

from analyze_atod_protocol_diagnostics import load_report, sha, trace_summary


def paired(left, right, rng_seed):
    a = {x["game"]: int(x["won"]) for x in left}
    b = {x["game"]: int(x["won"]) for x in right}
    if set(a) != set(b) or len(a) != 30:
        raise ValueError("Paired game sets differ")
    games = sorted(a)
    differences = [a[g] - b[g] for g in games]
    rng = random.Random(rng_seed)
    boot = sorted(sum(rng.choice(differences) for _ in games) / len(games)
                  for _ in range(20000))
    return {"left_only_success": sum(a[g] and not b[g] for g in games),
            "right_only_success": sum(b[g] and not a[g] for g in games),
            "both_success": sum(a[g] and b[g] for g in games),
            "both_failure": sum(not a[g] and not b[g] for g in games),
            "mean_difference": sum(differences) / len(games),
            "paired_game_bootstrap_95ci": [boot[int(0.025 * len(boot))],
                                             boot[int(0.975 * len(boot))]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--released-4b", type=Path, required=True)
    ap.add_argument("--student-2b", type=Path, required=True)
    ap.add_argument("--raw-9b", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = {"released_4b": args.released_4b,
             "student_2b": args.student_2b,
             "raw_9b": args.raw_9b}
    loaded = {name: load_report(path) for name, path in paths.items()}
    runs = {name: value[0] for name, value in loaded.items()}
    expected_roles = {"student_2b": "student_base", "raw_9b": "teacher_raw"}
    for name, role in expected_roles.items():
        if runs[name].get("role") != role:
            raise ValueError(f"Unexpected role for {name}")
    if runs["released_4b"].get("role") is not None:
        raise ValueError("Released 4B report has unexpected role")
    if not runs["released_4b"]["model"].endswith("/alfworld_grpo_qwen3_4b/actor_hf"):
        raise ValueError("Wrong released checkpoint")
    if not runs["student_2b"]["model"].endswith("/Qwen3.5-2B") or \
       not runs["raw_9b"]["model"].endswith("/Qwen3.5-9B"):
        raise ValueError("Wrong control model")
    reference = runs["released_4b"]
    settings = ("selection_manifest_sha256", "train_manifest_sha256",
                "selection", "sampling", "thinking_chat_template",
                "max_steps", "max_new_tokens", "context_length", "seed")
    for name in ("student_2b", "raw_9b"):
        for key in settings:
            if runs[name].get(key) != reference.get(key):
                raise ValueError(f"Shared setting {key} differs in {name}")
        if runs[name].get("prompt_source_sha256") != reference["source_sha256"]:
            raise ValueError(f"Frozen imported prompt/parser source differs in {name}")
    if runs["student_2b"]["source_sha256"] != runs["raw_9b"]["source_sha256"]:
        raise ValueError("Control evaluator source differs")
    report = {
        "scope": "Shared paper-style interface screening, not AQOD Gate T",
        "source_sha256": sha(Path(__file__)),
        "selection_manifest_sha256": reference["selection_manifest_sha256"],
        "prompt_parser_source_sha256": reference["source_sha256"],
        "control_evaluator_source_sha256": runs["student_2b"]["source_sha256"],
        "models": {
            name: {"run_sha256": sha(path / "run.json"),
                   "metrics": loaded[name][2],
                   "trace": trace_summary(path, loaded[name][3])}
            for name, path in paths.items()
        },
        "paired": {
            "released_4b_minus_student_2b": paired(loaded["released_4b"][3],
                                                     loaded["student_2b"][3], 2026092455),
            "released_4b_minus_raw_9b": paired(loaded["released_4b"][3],
                                                loaded["raw_9b"][3], 2026092456),
            "raw_9b_minus_student_2b": paired(loaded["raw_9b"][3],
                                               loaded["student_2b"][3], 2026092457),
        },
        "limits": [
            "The panel was previously used to diagnose the released teacher's interface; it is a screening set, not independent confirmation.",
            "The protocol approximates rather than reproduces ATOD's source evaluation.",
            "Model families may have different familiarity with this shared prompt.",
            "Full-game success does not establish teacher advice value on states visited by the compact-policy student.",
            "No result here passes AQOD Gate T or Gate 0 by itself.",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(args.output, sha(args.output))


if __name__ == "__main__":
    main()
