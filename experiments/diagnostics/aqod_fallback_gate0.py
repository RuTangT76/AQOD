"""Conditional, resumable student-relative one-action Gate 0 for ALFWorld.

This module is infrastructure only until the frozen fair controls qualify the
released action-level teacher. It never trains either policy.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

from trustlab.alfworld_bridge_validation import EnvironmentClient
from trustlab.alfworld_rollout import parse_command
from trustlab.aqod_prompt import build_current_commands_prompt
from trustlab.eval_atod_protocol_diag import ACTION, paper_style_prompt
from trustlab.modeling import encode_prompt, load_inference


KINDS = (
    "pick_and_place_simple", "look_at_obj_in_light",
    "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep", "pick_two_obj_and_place",
)
ENV_SEED = 2026092417
TEACHER_SEED_PREFIX = 2026092452
SELECTION_SEED_PREFIX = 2026092453
MAX_STEPS = 40


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def model_weight_hashes(path):
    shards = sorted(Path(path).glob("*.safetensors"))
    if not shards:
        raise ValueError(f"No safetensors model weights under {path}")
    return {shard.name: file_sha(shard) for shard in shards}


def public_sha(public):
    payload = {"observation": public["observation"],
               "admissible_commands": public["admissible_commands"]}
    return digest(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8"))


def query_seed(game_sha, step, state_sha):
    message = f"{TEACHER_SEED_PREFIX}:{game_sha}:{step}:{state_sha}"
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], "big")


def rank_candidate(candidate):
    message = (f"{SELECTION_SEED_PREFIX}:{candidate['game_sha256']}:"
               f"{candidate['step']}:{candidate['public_state_sha256']}")
    return digest(message.encode())


def write_json(path, value, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_panel(manifest, root):
    doc = read_json(manifest)
    if doc.get("teacher_or_outcome_used_for_selection") is not False:
        raise ValueError("Panel was not selected independently")
    games = doc["gate0"]
    if len(games) != 30 or len({x["game"] for x in games}) != 30:
        raise ValueError("Expected 30 distinct Gate 0 games")
    if sorted(x["task_type"] for x in games) != sorted(k for k in KINDS for _ in range(5)):
        raise ValueError("Gate 0 panel task balance changed")
    root = root.resolve(strict=True)
    if root.name != "valid_unseen":
        raise ValueError("Fallback Gate 0 uses official valid_unseen")
    for item in games:
        path = (root / item["game"]).resolve(strict=True)
        if not path.is_relative_to(root) or file_sha(path) != item["sha256"]:
            raise ValueError(f"Game hash mismatch: {item['game']}")
    return games


def generate_student(model, tokenizer, prompt, device):
    import torch
    try:
        ids = encode_prompt(tokenizer, prompt, 8192, 32, enable_thinking=False)
    except ValueError as exc:
        if "max_length=" not in str(exc):
            raise
        return {"context_limit": True, "text": "", "error": str(exc)}
    tensor = torch.tensor([ids], device=device)
    with torch.inference_mode():
        output = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                                do_sample=False, max_new_tokens=32,
                                pad_token_id=tokenizer.pad_token_id, use_cache=True)
    tokens = output[0, len(ids):].tolist()
    return {"text": tokenizer.decode(tokens, skip_special_tokens=True),
            "input_tokens": len(ids), "output_tokens": len(tokens)}


def generate_teacher(model, tokenizer, prompt, device, seed):
    import torch
    try:
        ids = encode_prompt(tokenizer, prompt, 8192, 512, enable_thinking=False)
    except ValueError as exc:
        if "max_length=" not in str(exc):
            raise
        return {"context_limit": True, "raw_text": "", "text": "", "error": str(exc)}
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tensor = torch.tensor([ids], device=device)
    with torch.inference_mode():
        output = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                                do_sample=True, temperature=0.4, top_p=1.0,
                                max_new_tokens=512,
                                pad_token_id=tokenizer.pad_token_id, use_cache=True)
    tokens = output[0, len(ids):].tolist()
    raw = tokenizer.decode(tokens, skip_special_tokens=True)
    match = ACTION.search(raw)
    return {"raw_text": raw, "text": match.group(1).strip() if match else "",
            "action_tag_found": bool(match), "input_tokens": len(ids),
            "output_tokens": len(tokens), "generation_hit_limit": len(tokens) == 512}


def collect_game(env, game, student, student_tok, teacher, teacher_tok,
                 student_device, teacher_device):
    reset = env.call("reset", game=game["game"], seed=ENV_SEED)
    history = [{"public": reset["public"]}]
    queries, candidates, transitions = [], [], []
    stop = "step_limit"
    result = reset
    for step in range(MAX_STEPS):
        if result["audit"]["done"]:
            stop = "environment_done"
            break
        public = result["public"]
        state_sha = public_sha(public)
        student_prompt = build_current_commands_prompt(history)
        teacher_prompt = paper_style_prompt(history)
        seed = query_seed(game["sha256"], step, state_sha)
        student_out = generate_student(student, student_tok, student_prompt,
                                       student_device)
        teacher_out = generate_teacher(teacher, teacher_tok, teacher_prompt,
                                       teacher_device, seed)
        student_action = (None if student_out.get("context_limit") else
                          parse_command(student_out["text"]))
        teacher_action = (None if teacher_out.get("context_limit") else
                          parse_command(teacher_out["text"]))
        query = {"step": step, "public_state_sha256": state_sha,
                 "student_prompt_sha256": digest(student_prompt.encode()),
                 "teacher_prompt_sha256": digest(teacher_prompt.encode()),
                 "teacher_seed": seed, "student": student_out,
                 "teacher": teacher_out, "student_action": student_action,
                 "teacher_action": teacher_action,
                 "student_listed": student_action in public["admissible_commands"],
                 "teacher_listed": teacher_action in public["admissible_commands"]}
        queries.append(query)
        if student_action is None:
            stop = "context_limit" if student_out.get("context_limit") else "format_error"
            break
        if student_action != teacher_action:
            candidates.append({"game": game["game"],
                               "game_sha256": game["sha256"],
                               "task_type": game["task_type"], "step": step,
                               "quartile": step // 10,
                               "public_state_sha256": state_sha,
                               "prefix_actions": [h["action"] for h in history[:-1]],
                               "prefix_public_sha256": [public_sha(h["public"])
                                                        for h in history],
                               "student_action": student_action,
                               "teacher_action": teacher_action,
                               "teacher_seed": seed,
                               "teacher_format_error": teacher_action is None,
                               "teacher_listed": query["teacher_listed"]})
        result = env.call("step", action=student_action)
        transitions.append({"step": step, "action": student_action,
                            "public": result["public"], "audit": result["audit"]})
        history[-1]["action"] = student_action
        history.append({"public": result["public"]})
        if result["audit"]["done"]:
            stop = "environment_done"
            break
    return {"game": game, "reset": reset, "queries": queries,
            "candidates": candidates, "transitions": transitions,
            "student_won": bool(result["audit"]["won"]), "stop_reason": stop,
            "steps": len(transitions)}


def select_candidates(all_candidates):
    groups = {}
    for candidate in all_candidates:
        key = (candidate["task_type"], candidate["quartile"])
        groups.setdefault(key, []).append(candidate)
    for values in groups.values():
        values.sort(key=rank_candidate)
    selected = []
    while len(selected) < 120 and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(selected) < 120:
                selected.append(groups[key].pop(0))
    return selected


def collect(args):
    import torch
    games = check_panel(args.selection_manifest, args.train_root)
    run = {"phase": "collect", "source_sha256": file_sha(__file__),
           "selection_manifest_sha256": file_sha(args.selection_manifest),
           "student_config_sha256": file_sha(args.student_model / "config.json"),
           "teacher_config_sha256": file_sha(args.teacher_model / "config.json"),
           "student_weight_sha256": model_weight_hashes(args.student_model),
           "teacher_weight_sha256": model_weight_hashes(args.teacher_model),
           "paper_prompt_source_sha256": file_sha(Path(__file__).parent /
                                                "eval_atod_protocol_diag.py"),
           "student_prompt_source_sha256": file_sha(Path(__file__).parent /
                                                  "aqod_prompt.py"),
           "student_model": str(args.student_model),
           "teacher_model": str(args.teacher_model), "env_seed": ENV_SEED,
           "train_root": str(args.train_root.resolve(strict=True)),
           "environment_python": str(args.environment_python.absolute()),
           "bridge_source_sha256": file_sha(Path(__file__).parent /
                                           "alfworld_bridge_validation.py"),
           "student_device": args.student_device,
           "teacher_device": args.teacher_device,
           "games": games}
    if args.resume:
        if not args.output.is_dir() or read_json(args.output / "run.json") != run:
            raise ValueError("Resume requested with absent or changed collection manifest")
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
        student, student_tok = load_inference(str(args.student_model), None,
                                              args.student_device)
        teacher, teacher_tok = load_inference(str(args.teacher_model), None,
                                              args.teacher_device)
        with EnvironmentClient(args.environment_python, args.train_root) as env:
            for index, game in enumerate(games):
                completed_file = args.output / f"game-{index:02d}.json"
                if completed_file.exists():
                    if read_json(completed_file)["game"] != game:
                        raise ValueError(f"Resume game mismatch at index {index}")
                    continue
                write_json(args.output / "state.json",
                           {"stage": "collecting", "complete": False,
                            "game_index": index}, overwrite=True)
                episode = collect_game(env, game, student, student_tok, teacher,
                                       teacher_tok, args.student_device,
                                       args.teacher_device)
                write_json(completed_file, episode)
                print(json.dumps({"game_index": index,
                                  "student_won": episode["student_won"],
                                  "queries": len(episode["queries"]),
                                  "disagreements": len(episode["candidates"])}),
                      flush=True)
        episodes = [read_json(args.output / f"game-{i:02d}.json") for i in range(30)]
        all_candidates = [c for e in episodes for c in e["candidates"]]
        selected = select_candidates(all_candidates)
        selection_doc = {"n_student_states": sum(len(e["queries"]) for e in episodes),
                         "n_disagreements": len(all_candidates),
                         "n_selected": len(selected),
                         "student_full_game_wins": sum(e["student_won"] for e in episodes),
                         "selected": selected}
        selection_file = args.output / "selection.json"
        if selection_file.exists():
            if read_json(selection_file) != selection_doc:
                raise ValueError("Resume selection mismatch")
        else:
            write_json(selection_file, selection_doc)
        write_json(args.output / "state.json",
                   {"stage": "complete", "complete": True,
                    "completed_games": 30}, overwrite=True)
    except BaseException as exc:
        write_json(args.output / "state.json",
                   {"stage": "failed", "complete": False,
                    "error": f"{type(exc).__name__}: {exc}"}, overwrite=True)
        raise


def replay(env, candidate):
    result = env.call("reset", game=candidate["game"], seed=ENV_SEED)
    history = [{"public": result["public"]}]
    expected = candidate["prefix_public_sha256"]
    if len(expected) != len(candidate["prefix_actions"]) + 1:
        raise ValueError("Stored replay prefix length mismatch")
    if public_sha(result["public"]) != expected[0]:
        raise ValueError("Public reset hash mismatch")
    for index, action in enumerate(candidate["prefix_actions"]):
        if result["audit"]["done"]:
            raise ValueError("Prefix reached terminal state")
        result = env.call("step", action=action)
        if public_sha(result["public"]) != expected[index + 1]:
            raise ValueError(f"Public replay transition mismatch at {index}")
        history[-1]["action"] = action
        history.append({"public": result["public"]})
    if public_sha(result["public"]) != candidate["public_state_sha256"]:
        raise ValueError("Public replay hash mismatch")
    if result["audit"]["done"]:
        raise ValueError("Branch point is already terminal")
    return result, history


def continue_branch(env, candidate, action, model, tokenizer, device):
    result, history = replay(env, candidate)
    if action is None:
        return {"won": False, "stop_reason": "intervention_format_error",
                "steps": candidate["step"], "continuation": []}
    result = env.call("step", action=action)
    history[-1]["action"] = action
    history.append({"public": result["public"]})
    continuation = []
    steps = candidate["step"] + 1
    stop = "environment_done" if result["audit"]["done"] else "step_limit"
    while not result["audit"]["done"] and steps < MAX_STEPS:
        prompt = build_current_commands_prompt(history)
        output = generate_student(model, tokenizer, prompt, device)
        command = None if output.get("context_limit") else parse_command(output["text"])
        if command is None:
            stop = "context_limit" if output.get("context_limit") else "format_error"
            continuation.append({"step": steps, "output": output, "command": None})
            break
        listed = command in result["public"]["admissible_commands"]
        result = env.call("step", action=command)
        continuation.append({"step": steps, "output": output,
                             "command": command, "listed": listed,
                             "public": result["public"], "audit": result["audit"]})
        history[-1]["action"] = command
        history.append({"public": result["public"]})
        steps += 1
        if result["audit"]["done"]:
            stop = "environment_done"
    return {"won": bool(result["audit"]["won"]), "stop_reason": stop,
            "steps": steps, "continuation": continuation}


def branch(args):
    report = args.collect_report
    collect_run = read_json(report / "run.json")
    collect_state = read_json(report / "state.json")
    if collect_state.get("stage") != "complete" or not collect_state.get("complete"):
        raise ValueError("Collection is incomplete")
    if collect_run["source_sha256"] != file_sha(__file__):
        raise ValueError("Collection source differs from branch source")
    if str(args.student_model) != collect_run["student_model"] or \
       str(args.train_root.resolve(strict=True)) != collect_run["train_root"] or \
       str(args.environment_python.absolute()) != collect_run["environment_python"]:
        raise ValueError("Branch policy or environment differs from collection")
    if model_weight_hashes(args.student_model) != collect_run["student_weight_sha256"]:
        raise ValueError("Student weights changed since collection")
    selection = read_json(report / "selection.json")
    candidates = selection["selected"]
    if len(candidates) != selection["n_selected"] or len(candidates) > 120:
        raise ValueError("Branch selection changed")
    run = {"phase": "branch", "source_sha256": file_sha(__file__),
           "collect_run_sha256": file_sha(report / "run.json"),
           "collect_selection_sha256": file_sha(report / "selection.json"),
           "student_config_sha256": file_sha(args.student_model / "config.json"),
           "student_weight_sha256": collect_run["student_weight_sha256"],
           "student_model": str(args.student_model),
           "train_root": str(args.train_root.resolve(strict=True)),
           "environment_python": str(args.environment_python.absolute()),
           "bridge_source_sha256": file_sha(Path(__file__).parent /
                                           "alfworld_bridge_validation.py"),
           "n_pairs": len(candidates)}
    if args.resume:
        if not args.output.is_dir() or read_json(args.output / "run.json") != run:
            raise ValueError("Resume requested with absent or changed branch manifest")
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
        model, tokenizer = load_inference(str(args.student_model), None,
                                          args.student_device)
        with EnvironmentClient(args.environment_python, args.train_root) as student_env, \
             EnvironmentClient(args.environment_python, args.train_root) as teacher_env:
            for index, candidate in enumerate(candidates):
                completed_file = args.output / f"pair-{index:03d}.json"
                if completed_file.exists():
                    row = read_json(completed_file)
                    if row["index"] != index or row["candidate"] != candidate:
                        raise ValueError(f"Resume pair mismatch at index {index}")
                    continue
                write_json(args.output / "state.json",
                           {"stage": "branching", "complete": False,
                            "pair_index": index}, overwrite=True)
                try:
                    own = continue_branch(student_env, candidate,
                                          candidate["student_action"], model,
                                          tokenizer, args.student_device)
                    advised = continue_branch(teacher_env, candidate,
                                              candidate["teacher_action"], model,
                                              tokenizer, args.student_device)
                    row = {"index": index, "candidate": candidate,
                           "student_branch": own, "teacher_branch": advised,
                           "delta": int(advised["won"]) - int(own["won"])}
                except ValueError as exc:
                    row = {"index": index, "candidate": candidate,
                           "replay_error": str(exc), "delta": None}
                write_json(completed_file, row)
                print(json.dumps({"pair_index": index, "delta": row["delta"],
                                  "replay_error": row.get("replay_error")}), flush=True)
        write_json(args.output / "state.json",
                   {"stage": "complete", "complete": True,
                    "completed_pairs": len(candidates)}, overwrite=True)
    except BaseException as exc:
        write_json(args.output / "state.json",
                   {"stage": "failed", "complete": False,
                    "error": f"{type(exc).__name__}: {exc}"}, overwrite=True)
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("collect", "branch"), required=True)
    ap.add_argument("--selection-manifest", type=Path)
    ap.add_argument("--collect-report", type=Path)
    ap.add_argument("--student-model", type=Path, required=True)
    ap.add_argument("--teacher-model", type=Path)
    ap.add_argument("--environment-python", type=Path, required=True)
    ap.add_argument("--train-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--student-device", default="cuda:0")
    ap.add_argument("--teacher-device", default="cuda:1")
    ap.add_argument("--resume", action="store_true",
                    help="Continue only from complete per-game or per-pair files with identical manifests")
    args = ap.parse_args()
    if args.phase == "collect":
        if args.selection_manifest is None or args.teacher_model is None:
            ap.error("collect requires --selection-manifest and --teacher-model")
        collect(args)
    else:
        if args.collect_report is None:
            ap.error("branch requires --collect-report")
        branch(args)


if __name__ == "__main__":
    main()
