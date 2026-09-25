"""Run the frozen V5 six-game technical smoke with the raw 9B teacher."""

import hashlib
import json
from pathlib import Path
import time

import torch

from trustlab.alfworld_bridge import EnvironmentClient
from trustlab.alfworld_rollout import run_episode
from trustlab.aqod_prompt import build_current_commands_prompt
from trustlab.modeling import encode_prompt, load_inference
from trustlab.aqod_mainline import assert_teacher_frozen
from trustlab.teacher_grpo_probe import save


ROOT = Path("/root/trust_mvp")
PANEL = ROOT / "analysis/aqod_teacher_v5_verl_panel.json"
PANEL_SHA = "c15c5c7bc7091f588896fdbd771020a98e225de48e26e3b6f05dd22bcb1d1527"
TRAIN_ROOT = ROOT / "assets/alfworld/json_2.1.1/train"
MODEL = Path("/root/shared-nvme/models/Qwen3.5-9B")
OUTPUT = ROOT / "runs/aqod-v5-verl-raw9b-smoke6"
SEED = 2026092514


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(PANEL) != PANEL_SHA:
        raise ValueError("V5 selection hash mismatch")
    panel = json.loads(PANEL.read_text())
    games = panel["smoke"]
    assert len(games) == 6 and len({x["game"] for x in games}) == 6
    assert not {x["game"] for x in games} & {x["game"] for x in panel["train"] + panel["eval"]}
    for item in games:
        path = (TRAIN_ROOT / item["game"]).resolve(strict=True)
        assert path.is_relative_to(TRAIN_ROOT.resolve(strict=True))
        assert sha(path) == item["sha256"]
    OUTPUT.mkdir(parents=True)
    config = {
        "role": "raw_teacher_smoke_only", "model": str(MODEL),
        "model_config_sha256": sha(MODEL / "config.json"),
        "selection_manifest_sha256": PANEL_SHA, "source_sha256": sha(__file__),
        "prompt_source_sha256": sha(ROOT / "trustlab/aqod_prompt.py"),
        "games": games, "seed": SEED, "decoding": "greedy", "thinking": False,
        "prompt_format": "current_commands", "max_steps": 40,
        "max_new_tokens": 32, "context_length": 8192,
        "scientific_gate_result": False,
    }
    save(OUTPUT / "run.json", config)
    state = {"stage": "loading_model", "complete": False, "completed_games": 0}
    save(OUTPUT / "state.json", state)
    started = time.time()
    try:
        torch.manual_seed(SEED)
        model, tokenizer = load_inference(str(MODEL), None, "cuda:1")
        assert_teacher_frozen(model)

        def generate(prompt):
            try:
                ids = encode_prompt(tokenizer, prompt, 8192, 32, enable_thinking=False)
            except ValueError as exc:
                if "max_length=" not in str(exc):
                    raise
                return {"context_limit": True, "text": "", "error": str(exc)}
            tensor = torch.tensor([ids], device="cuda:1")
            with torch.inference_mode():
                out = model.generate(
                    input_ids=tensor, attention_mask=torch.ones_like(tensor),
                    do_sample=False, max_new_tokens=32,
                    pad_token_id=tokenizer.pad_token_id, use_cache=True,
                )
            tokens = out[0, len(ids):].tolist()
            return {"text": tokenizer.decode(tokens, skip_special_tokens=True),
                    "input_tokens": len(ids), "output_tokens": len(tokens)}

        outcomes = []
        with EnvironmentClient(ROOT / ".venv-alfworld/bin/python", TRAIN_ROOT) as env:
            for i, item in enumerate(games):
                state.update(stage="evaluating", game_index=i)
                save(OUTPUT / "state.json", state)
                with (OUTPUT / f"episode-{i:02d}.jsonl").open("x", buffering=1) as stream:
                    def emit(row):
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    result = run_episode(
                        env, item["game"], generate, 40, SEED, emit,
                        history_format="flat", prompt_builder=build_current_commands_prompt,
                    )
                result.update({"game": item["game"], "task_type": item["task_type"]})
                outcomes.append(result)
                save(OUTPUT / "episodes.json", outcomes)
                state.update(completed_games=len(outcomes), elapsed_seconds=time.time() - started)
                save(OUTPUT / "state.json", state)
                print(json.dumps({"index": i, **result}), flush=True)
        metrics = {"n": len(outcomes), "successes": sum(x["won"] for x in outcomes),
                   "success_rate": sum(x["won"] for x in outcomes) / len(outcomes),
                   "elapsed_seconds": time.time() - started,
                   "smoke_only": True}
        save(OUTPUT / "metrics.json", metrics)
        state.update(stage="complete", complete=True, **metrics)
        save(OUTPUT / "state.json", state)
    except BaseException as exc:
        state.update(stage="failed", error=f"{type(exc).__name__}: {exc}",
                     elapsed_seconds=time.time() - started)
        save(OUTPUT / "state.json", state)
        raise


if __name__ == "__main__":
    main()
