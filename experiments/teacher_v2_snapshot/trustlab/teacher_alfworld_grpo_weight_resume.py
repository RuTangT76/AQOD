"""Weights-only continuation after a paused AQOD teacher run.

This is explicitly *not* an exact optimizer/RNG resume. It leaves the paused
directory immutable, starts from its final complete LoRA adapter, and trains
the remaining frozen game groups with a fresh AdamW optimizer.
"""

import argparse
import json
from pathlib import Path
import time

from .modeling import load_base_model
from .teacher_alfworld_grpo import collect, episode_seeds, optimize_group
from .teacher_grpo_probe import save, sha


def read_source(args):
    pause_path = args.source_run / "pause.json"
    pause = json.loads(pause_path.read_text())
    source_manifest_path = args.source_run / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    selection = json.loads(args.selection_manifest.read_text())
    if pause.get("status") != "paused_by_user" or pause.get("exact_resume_supported") is not False:
        raise ValueError("Expected an explicitly paused, non-exact-resumable source")
    if pause.get("optimizer_state_saved") is not False:
        raise ValueError("Unexpected optimizer-state condition")
    if sha(source_manifest_path) != pause["manifest_sha256"]:
        raise ValueError("Source manifest hash mismatch")
    if source_manifest["selection_manifest_sha256"] != sha(args.selection_manifest):
        raise ValueError("Source used a different frozen game selection")
    if source_manifest["selection"] != selection["train"]:
        raise ValueError("Source training-game order differs")
    start_group = pause["completed_groups"]
    if not 0 < start_group < len(selection["train"]):
        raise ValueError("Invalid continuation group index")
    source_rows = (args.source_run / "groups.jsonl").read_text().splitlines()
    if len(source_rows) != start_group:
        raise ValueError("Source completed-group count mismatch")
    if json.loads(source_rows[-1])["group"] != start_group - 1:
        raise ValueError("Source last-complete group mismatch")
    checkpoint = Path(pause["checkpoint"])
    if not checkpoint.is_relative_to(args.source_run.resolve()):
        raise ValueError("Checkpoint outside source run")
    adapter = checkpoint / "adapter_model.safetensors"
    if not adapter.is_file() or sha(adapter) != pause["adapter_sha256"]:
        raise ValueError("Source adapter hash mismatch")
    if pause["completed_updates"] < 1:
        raise ValueError("Source has no completed update")
    return pause, source_manifest, selection["train"], checkpoint


def run(args):
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    if args.output.exists():
        raise ValueError("Output exists; refusing to overwrite")
    pause, source_manifest, games, checkpoint = read_source(args)
    if args.group_size != 4 or args.max_train_tokens != 2048 or args.context_length != 8192:
        raise ValueError("Continuation must retain frozen v2 training budgets")
    source_args = source_manifest["arguments"]
    required_equal = {
        "max_steps": args.max_steps, "max_new_tokens": args.max_new_tokens,
        "group_size": args.group_size, "max_train_tokens": args.max_train_tokens,
        "context_length": args.context_length, "rank": args.rank,
        "lr": args.lr, "beta": args.beta, "temperature": args.temperature,
        "top_p": args.top_p, "seed": args.seed, "prompt_format": args.prompt_format,
    }
    for key, value in required_equal.items():
        if source_args.get(key) != str(value):
            raise ValueError(f"Frozen source argument differs: {key}")
    if args.prompt_format != "current_commands":
        raise ValueError("Expected frozen v2 prompt")

    args.output.mkdir(parents=True)
    started = time.time()
    manifest = {
        "mode": "teacher_alfworld_lora_grpo_weights_only_continuation",
        "exact_resume": False, "optimizer_reset": True, "rng_reset": True,
        "source_run": str(args.source_run), "source_pause_sha256": sha(args.source_run / "pause.json"),
        "source_manifest_sha256": sha(args.source_run / "manifest.json"),
        "source_training_source_sha256": source_manifest["source_sha256"],
        "source_checkpoint": str(checkpoint),
        "source_adapter_sha256": pause["adapter_sha256"],
        "source_completed_groups": pause["completed_groups"],
        "source_completed_updates": pause["completed_updates"],
        "selection_manifest_sha256": sha(args.selection_manifest),
        "selection": games, "run_selection": games[pause["completed_groups"]:],
        "source_sha256": sha(__file__),
        "model_config_sha256": sha(args.model / "config.json"),
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "scientific_gate_t_preregistered": False,
        "teacher_frozen_after_training": True,
        "student_task_warmup": False,
        "start_time": started,
    }
    save(args.output / "manifest.json", manifest)
    state = {
        "stage": "loading", "complete": False,
        "source_completed_groups": pause["completed_groups"],
        "continuation_groups": 0, "total_completed_groups": pause["completed_groups"],
        "source_completed_updates": pause["completed_updates"],
        "continuation_updates": 0, "total_updates": pause["completed_updates"],
    }
    save(args.output / "state.json", state)
    try:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = load_base_model(str(args.model), args.device)
        base.config.use_cache = False
        if hasattr(base.config, "text_config"):
            base.config.text_config.use_cache = False
        model = PeftModel.from_pretrained(base, checkpoint, is_trainable=True)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=args.lr, weight_decay=0.0)
        torch.cuda.reset_peak_memory_stats(args.device)
        model.eval()
        with (args.output / "groups.jsonl").open("x", buffering=1) as stream:
            for i in range(pause["completed_groups"], len(games)):
                game = games[i]
                state.update(stage="collecting", group_index=i)
                save(args.output / "state.json", state)
                episodes = []
                for j in range(args.group_size):
                    episode = collect(model, tokenizer, args, game["game"],
                                      *episode_seeds(args.seed, i, j))
                    episodes.append(episode)
                    with (args.output / f"group-{i:02d}-episode-{j}.json").open("x") as f:
                        json.dump(episode, f, ensure_ascii=False)
                state["stage"] = "updating"
                save(args.output / "state.json", state)
                update = optimize_group(model, optimizer, episodes, args)
                state["continuation_groups"] += 1
                state["total_completed_groups"] += 1
                if update["updated"]:
                    state["continuation_updates"] += 1
                    state["total_updates"] += 1
                    snapshot = args.output / "snapshots" / f'update-{state["total_updates"]:03d}'
                    snapshot.mkdir(parents=True)
                    model.save_pretrained(snapshot, safe_serialization=True)
                    tokenizer.save_pretrained(snapshot)
                    update["checkpoint"] = str(snapshot)
                    update["adapter_sha256"] = sha(snapshot / "adapter_model.safetensors")
                row = {
                    "group": i, "game": game, "update": update,
                    "episode_summary": [
                        {k: episode[k] for k in ("won", "reward", "stop", "steps", "invalid")}
                        for episode in episodes
                    ],
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device),
                    "elapsed_seconds": time.time() - started,
                }
                stream.write(json.dumps(row) + "\n")
                state.update(stage="collecting", elapsed_seconds=row["elapsed_seconds"],
                             peak_allocated_bytes=row["peak_allocated_bytes"],
                             peak_reserved_bytes=row["peak_reserved_bytes"])
                save(args.output / "state.json", state)
                print(json.dumps(row), flush=True)
        state.update(stage="complete", complete=True, elapsed_seconds=time.time() - started)
        save(args.output / "state.json", state)
    except BaseException as exc:
        state.update(stage="failed", error=f"{type(exc).__name__}: {exc}",
                     elapsed_seconds=time.time() - started)
        save(args.output / "state.json", state)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--environment-python", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-format", default="current_commands")
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-train-tokens", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2026092317)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
