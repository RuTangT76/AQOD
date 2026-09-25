"""Freeze a larger, outcome-blind teacher-preparation panel from ALFWorld train."""

import argparse
import hashlib
import json
from pathlib import Path


TASKS = (
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
)
SEED = 2026092501
TRAIN_PER_TYPE = 30
EVAL_PER_TYPE = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--prior-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output exists; refusing overwrite")
    prior = json.loads(args.prior_selection.read_text(encoding="utf-8"))
    excluded = {
        item if isinstance(item, str) else item.get("game", item.get("relative_path"))
        for key in ("train", "eval", "excluded")
        for item in prior.get(key, [])
    }
    if None in excluded:
        raise SystemExit("Prior selection contains a missing game path")
    by_type = {}
    for task in TASKS:
        paths = [
            path for path in args.train_root.glob(f"{task}-*/trial_*/game.tw-pddl")
            if path.relative_to(args.train_root).as_posix() not in excluded
        ]
        paths.sort(key=lambda path: hashlib.sha256(
            f"{SEED}:{path.relative_to(args.train_root).as_posix()}".encode()
        ).hexdigest())
        if len(paths) < TRAIN_PER_TYPE + EVAL_PER_TYPE:
            raise SystemExit(f"Insufficient independent games for {task}: {len(paths)}")
        by_type[task] = paths[:TRAIN_PER_TYPE + EVAL_PER_TYPE]

    def record(path: Path, task: str) -> dict:
        return {
            "game": path.relative_to(args.train_root).as_posix(),
            "task_type": task,
            "sha256": sha256(path),
        }

    train = [record(by_type[task][index], task)
             for index in range(TRAIN_PER_TYPE) for task in TASKS]
    evaluation = [record(by_type[task][TRAIN_PER_TYPE + index], task)
                  for index in range(EVAL_PER_TYPE) for task in TASKS]
    all_games = [item["game"] for item in train + evaluation]
    if len(all_games) != len(set(all_games)) or set(all_games) & excluded:
        raise SystemExit("Selection overlap detected")
    payload = {
        "purpose": "teacher-prerequisite V4 training-budget test; not AQOD method data",
        "split": "ALFWorld train",
        "seed": SEED,
        "selection_rule": "SHA256(seed:path) within task type; first 30 training, next 10 evaluation; round-robin task order",
        "prior_selection_sha256": sha256(args.prior_selection),
        "selection_source_sha256": sha256(Path(__file__)),
        "teacher_or_student_outcomes_used": False,
        "excluded_prior_games": len(excluded),
        "train": train,
        "eval": evaluation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output),
                      "train": len(train), "eval": len(evaluation),
                      "excluded_prior_games": len(excluded)}))


if __name__ == "__main__":
    main()
