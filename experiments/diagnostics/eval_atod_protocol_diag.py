"""Diagnostic of released ATOD teacher under a paper-style action interface.

This is not a reproduction of ATOD's validation score: the panel is AQOD's
frozen valid_unseen sample, and generation uses local Transformers.
"""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from trustlab.alfworld_bridge_validation import EnvironmentClient
from trustlab.alfworld_rollout import run_episode
from trustlab.aqod_mainline import assert_teacher_frozen
from trustlab.modeling import encode_prompt, load_inference
from trustlab.teacher_grpo_probe import save, sha


ACTION = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)
TASK = re.compile(r"Your task is to:\s*(.+?)(?:\n|$)", re.IGNORECASE)
HISTORY_LENGTH = 2  # ATOD repository ppo_trainer.yaml default.


def paper_style_prompt(history):
    initial = history[0]["public"]["observation"]
    task_match = TASK.search(initial)
    if task_match is None:
        raise ValueError("Task description missing from reset observation")
    task = task_match.group(1).strip()
    completed = history[:-1]
    recent = completed[-HISTORY_LENGTH:]
    action_history = "\n".join(
        f"Observation: {turn['public']['observation']}\nAction: {turn['action']}"
        for turn in recent
    ) or "None.\n"
    current = history[-1]["public"]
    actions = "\n ".join(repr(action) for action in current["admissible_commands"])
    step = len(completed)
    return (
        "You are an expert agent operating in the ALFRED Embodied Environment. "
        f"Your task is to: {task}. Prior to this step, you have already taken "
        f"{step} step(s). Below are the most recent {HISTORY_LENGTH} observations "
        f"and the corresponding actions you took: {action_history} "
        f"You are now at step {step + 1} and your current observation is: "
        f"{current['observation']} Your admissible actions of the current "
        f"situation are: [[{actions}]].\n"
        "Now it's your turn to take an action. You should first reason step-by-step "
        "about the current situation. This reasoning process MUST be enclosed "
        "within <think> </think> tags. Once you've finished your reasoning, "
        "you should choose an admissible action for current step and present it "
        "within <action> </action> tags."
    )


def run(args):
    import torch

    if args.output.exists():
        raise ValueError("Output exists; refusing to overwrite")
    source = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    if source.get("teacher_or_outcome_used_for_selection") is not False:
        raise ValueError("Panel must be selected before teacher outcomes")
    games = source.get("eval", source.get("selected"))
    if len(games) != 30 or len({x["game"] for x in games}) != 30:
        raise ValueError("Expected frozen 30-game panel")
    training = json.loads(args.train_manifest.read_text(encoding="utf-8"))
    train_games = {x["game"] for x in training.get("train", training.get("selection"))}
    if train_games.intersection(x["game"] for x in games):
        raise ValueError("Training and diagnostic game paths overlap")
    selection = []
    root = args.train_root.resolve(strict=True)
    for item in games:
        path = (root / item["game"]).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("Game outside validation root")
        digest = sha(path)
        if item.get("sha256") != digest:
            raise ValueError("Game hash mismatch")
        selection.append({"game": item["game"], "task_type": item["task_type"], "sha256": digest})
    args.output.mkdir(parents=True)
    run_doc = {
        "scope": "External-teacher interface diagnostic; not ATOD paper reproduction or AQOD Gate T",
        "model": str(args.model),
        "model_config_sha256": sha(args.model / "config.json"),
        "selection_manifest_sha256": sha(args.selection_manifest),
        "train_manifest_sha256": sha(args.train_manifest),
        "source_sha256": sha(__file__),
        "selection": selection,
        "panel": "official valid_unseen; same 30 games as AQOD short-action diagnostic",
        "prompt": "ATOD paper Figure 9 approximation; history_length=2 from released default config",
        "action_parser": "first complete <action>...</action> tag; no best-match correction",
        "sampling": {"do_sample": True, "temperature": 0.4, "top_p": 1.0},
        "thinking_chat_template": False,
        "max_steps": 50,
        "max_new_tokens": 512,
        "context_length": args.context_length,
        "seed": args.seed,
        "scientific_mvp_passed": False,
        "caveat": "ATOD's default evaluation is in-distribution; this panel is valid_unseen."
    }
    save(args.output / "run.json", run_doc)
    state = {"stage": "loading_model", "complete": False, "completed_games": 0}
    save(args.output / "state.json", state)
    started = time.time()
    try:
        torch.manual_seed(args.seed)
        model, tokenizer = load_inference(str(args.model), None, args.device)
        assert_teacher_frozen(model)
        outcomes = []

        def generate(prompt):
            try:
                ids = encode_prompt(tokenizer, prompt, args.context_length, 512,
                                    enable_thinking=False)
            except ValueError as exc:
                if "max_length=" not in str(exc):
                    raise
                return {"context_limit": True, "text": "", "raw_text": "", "error": str(exc)}
            tensor = torch.tensor([ids], device=args.device)
            with torch.inference_mode():
                output = model.generate(
                    input_ids=tensor, attention_mask=torch.ones_like(tensor),
                    do_sample=True, temperature=0.4, top_p=1.0,
                    max_new_tokens=512, pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
            tokens = output[0, len(ids):].tolist()
            raw = tokenizer.decode(tokens, skip_special_tokens=True)
            match = ACTION.search(raw)
            parsed = match.group(1).strip() if match else ""
            return {"text": parsed, "raw_text": raw, "action_tag_found": bool(match),
                    "input_tokens": len(ids), "output_tokens": len(tokens),
                    "generation_hit_limit": len(tokens) == 512}

        with EnvironmentClient(args.environment_python, root) as env:
            for index, item in enumerate(selection):
                state.update(stage="evaluating", game_index=index)
                save(args.output / "state.json", state)
                with (args.output / f"episode-{index:02d}.jsonl").open(
                    "x", encoding="utf-8", buffering=1
                ) as stream:
                    def emit(row):
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")

                    result = run_episode(
                        env, item["game"], generate, 50, args.seed, emit,
                        history_format="flat", prompt_builder=paper_style_prompt,
                    )
                result.update(game=item["game"], task_type=item["task_type"])
                outcomes.append(result)
                save(args.output / "episodes.json", outcomes)
                state.update(completed_games=len(outcomes), elapsed_seconds=time.time() - started)
                save(args.output / "state.json", state)
                print(json.dumps({"index": index, **result}), flush=True)
        by_type = {
            task: {"n": sum(x["task_type"] == task for x in outcomes),
                   "successes": sum(x["task_type"] == task and x["won"] for x in outcomes)}
            for task in sorted({x["task_type"] for x in outcomes})
        }
        metrics = {"n": len(outcomes), "successes": sum(x["won"] for x in outcomes),
                   "success_rate": sum(x["won"] for x in outcomes) / len(outcomes),
                   "invalid_actions": sum(x["unlisted_commands"] for x in outcomes),
                   "total_steps": sum(x["steps"] for x in outcomes),
                   "by_task_type": by_type, "elapsed_seconds": time.time() - started,
                   "format_errors": sum(x["stop_reason"] == "format_error" for x in outcomes)}
        save(args.output / "metrics.json", metrics)
        state.update(stage="complete", complete=True, **metrics)
        save(args.output / "state.json", state)
    except BaseException as exc:
        state.update(stage="failed", error=f"{type(exc).__name__}: {exc}",
                     elapsed_seconds=time.time() - started)
        save(args.output / "state.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--environment-python", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=2026092417)
    run(parser.parse_args())
