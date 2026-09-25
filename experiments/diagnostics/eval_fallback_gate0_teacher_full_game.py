"""Descriptive full-game evaluation of the frozen fallback Gate 0 teacher.

This uses the same paper-style prompt, one-action parser, stochastic decoding,
and public-state query seed as the Gate 0 collector. It is not the action-level
branch estimand and must not be used to tune or select the teacher.
"""

import argparse
import json
from pathlib import Path

import torch

from trustlab.alfworld_bridge_validation import EnvironmentClient
from trustlab.alfworld_rollout import parse_command
from trustlab.aqod_fallback_gate0 import (
    ENV_SEED, MAX_STEPS, check_panel, file_sha, generate_teacher,
    model_weight_hashes, public_sha, query_seed, read_json, write_json,
)
from trustlab.eval_atod_protocol_diag import paper_style_prompt
from trustlab.modeling import load_inference


def evaluate_game(env, game, model, tokenizer, device):
    result = env.call("reset", game=game["game"], seed=ENV_SEED)
    history = [{"public": result["public"]}]
    trace = []
    stop = "step_limit"
    for step in range(MAX_STEPS):
        if result["audit"]["done"]:
            stop = "environment_done"
            break
        public = result["public"]
        state_sha = public_sha(public)
        prompt = paper_style_prompt(history)
        seed = query_seed(game["sha256"], step, state_sha)
        output = generate_teacher(model, tokenizer, prompt, device, seed)
        action = None if output.get("context_limit") else parse_command(output["text"])
        row = {"step": step, "public_state_sha256": state_sha,
               "query_seed": seed, "output": output, "action": action,
               "listed": action in public["admissible_commands"]}
        trace.append(row)
        if action is None:
            stop = "context_limit" if output.get("context_limit") else "format_error"
            break
        result = env.call("step", action=action)
        row["next_public_state_sha256"] = public_sha(result["public"])
        row["audit"] = result["audit"]
        history[-1]["action"] = action
        history.append({"public": result["public"]})
        if result["audit"]["done"]:
            stop = "environment_done"
            break
    return {"game": game, "won": bool(result["audit"]["won"]),
            "stop_reason": stop, "steps": sum("audit" in row for row in trace),
            "trace": trace}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection-manifest", type=Path, required=True)
    ap.add_argument("--teacher-model", type=Path, required=True)
    ap.add_argument("--environment-python", type=Path, required=True)
    ap.add_argument("--train-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--teacher-device", default="cuda:1")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    games = check_panel(args.selection_manifest, args.train_root)
    run = {"role": "frozen_teacher_full_game_descriptive",
           "source_sha256": file_sha(__file__),
           "gate0_source_sha256": file_sha(Path(__file__).parent /
                                          "aqod_fallback_gate0.py"),
           "paper_prompt_source_sha256": file_sha(Path(__file__).parent /
                                                  "eval_atod_protocol_diag.py"),
           "selection_manifest_sha256": file_sha(args.selection_manifest),
           "teacher_model": str(args.teacher_model),
           "teacher_config_sha256": file_sha(args.teacher_model / "config.json"),
           "teacher_weight_sha256": model_weight_hashes(args.teacher_model),
           "train_root": str(args.train_root.resolve(strict=True)),
           "environment_python": str(args.environment_python.absolute()),
           "bridge_source_sha256": file_sha(Path(__file__).parent /
                                           "alfworld_bridge_validation.py"),
           "teacher_device": args.teacher_device,
           "env_seed": ENV_SEED, "max_steps": MAX_STEPS,
           "decoding": {"do_sample": True, "temperature": 0.4,
                        "top_p": 1.0, "max_new_tokens": 512,
                        "enable_thinking": False},
           "games": games}
    if args.resume:
        if not args.output.is_dir() or read_json(args.output / "run.json") != run:
            raise ValueError("Resume requested with absent or changed run")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "run.json", run)
    if args.resume and (args.output / "state.json").exists():
        state = read_json(args.output / "state.json")
        if state.get("stage") == "complete" and state.get("complete"):
            return
    write_json(args.output / "state.json", {"stage": "loading", "complete": False},
               overwrite=True)
    try:
        torch.manual_seed(ENV_SEED)
        model, tokenizer = load_inference(str(args.teacher_model), None,
                                          args.teacher_device)
        with EnvironmentClient(args.environment_python, args.train_root) as env:
            for index, game in enumerate(games):
                completed = args.output / f"game-{index:02d}.json"
                if completed.exists():
                    if read_json(completed)["game"] != game:
                        raise ValueError(f"Resume game mismatch at {index}")
                    continue
                write_json(args.output / "state.json",
                           {"stage": "evaluating", "complete": False,
                            "game_index": index}, overwrite=True)
                outcome = evaluate_game(env, game, model, tokenizer,
                                        args.teacher_device)
                write_json(completed, outcome)
                print(json.dumps({"game_index": index, "won": outcome["won"],
                                  "steps": outcome["steps"]}), flush=True)
        outcomes = [read_json(args.output / f"game-{i:02d}.json")
                    for i in range(len(games))]
        write_json(args.output / "summary.json",
                   {"games": len(outcomes),
                    "wins": sum(x["won"] for x in outcomes),
                    "format_errors": sum(x["stop_reason"] == "format_error"
                                         for x in outcomes),
                    "context_limits": sum(x["stop_reason"] == "context_limit"
                                          for x in outcomes)})
        write_json(args.output / "state.json",
                   {"stage": "complete", "complete": True,
                    "completed_games": len(games)}, overwrite=True)
    except BaseException as exc:
        write_json(args.output / "state.json",
                   {"stage": "failed", "complete": False,
                    "error": f"{type(exc).__name__}: {exc}"}, overwrite=True)
        raise


if __name__ == "__main__":
    main()
