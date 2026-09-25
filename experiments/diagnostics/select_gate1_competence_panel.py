"""Outcome-blind, conditional Gate1 OPD competence and snapshot-dev panel.

This selects game paths only. It does not start student training or open either
sealed Gate0 confirmation outcomes or any V4 success metrics.
"""

import argparse
import hashlib
import json
from pathlib import Path


TASKS = (
    "pick_and_place_simple", "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep", "pick_heat_then_place_in_recep",
    "pick_two_obj_and_place", "look_at_obj_in_light",
)
SEED = 2026092506
TRAIN_PER_TYPE = 30
DEV_PER_TYPE = 10
V2_SHA = "02b89be2c985afacace26d7da5e02d2fe2996ae9b108667a230f21ba20931a48"
V4_SHA = "8970df8a9692a57d2f48206efb8d83ae2c992a9d4382f6127d90f9a39e71cea5"
GATE0_SHA = "c32ff2cd60fb54e52523689d18bb9149671ec7cf2124f0295ab8ceab795eb22b"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path, expected_hash):
    if sha(path) != expected_hash:
        raise ValueError(f"Prior panel hash changed: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def select(root, v2_path, v4_path, gate0_path):
    root = root.resolve(strict=True)
    if root.name != "train":
        raise ValueError("Gate1 panel must use official ALFWorld train")
    v2 = load(v2_path, V2_SHA)
    v4 = load(v4_path, V4_SHA)
    gate0 = load(gate0_path, GATE0_SHA)
    excluded = set(v2["excluded"])
    excluded.update(x["game"] for key in ("train", "eval") for x in v2[key])
    excluded.update(x["game"] for key in ("train", "eval") for x in v4[key])
    excluded.update(x["game"] for key in ("gate0", "confirmation") for x in gate0[key])
    if len(excluded) != 402:
        raise ValueError(f"Expected 402 disjoint prior games; found {len(excluded)}")
    by_type = {}
    for task in TASKS:
        ranked = []
        for path in root.glob(f"{task}-*/trial_*/game.tw-pddl"):
            rel = path.relative_to(root).as_posix()
            if rel not in excluded:
                ranked.append((hashlib.sha256(f"{SEED}:{rel}".encode()).hexdigest(),
                               rel, path))
        ranked.sort()
        if len(ranked) < TRAIN_PER_TYPE + DEV_PER_TYPE:
            raise ValueError(f"Insufficient unused {task} games: {len(ranked)}")
        by_type[task] = [{"game": rel, "task_type": task, "sha256": sha(path)}
                         for _, rel, path in ranked[:TRAIN_PER_TYPE + DEV_PER_TYPE]]
    training = [by_type[task][i] for i in range(TRAIN_PER_TYPE) for task in TASKS]
    development = [by_type[task][TRAIN_PER_TYPE + i]
                   for i in range(DEV_PER_TYPE) for task in TASKS]
    paths = [x["game"] for x in training + development]
    if len(paths) != 240 or len(set(paths)) != 240 or set(paths) & excluded:
        raise ValueError("Gate1 panel overlap or count mismatch")
    return {
        "purpose": "Conditional Gate1 teacher-sourced student competence; pure OPD, not AQOD",
        "split": "ALFWorld train", "seed": SEED,
        "selection_rule": "SHA256(seed:relative_path) within task type; first 30 OPD train, next 10 snapshot development; round-robin task order",
        "v2_panel_sha256": V2_SHA, "v4_panel_sha256": V4_SHA,
        "gate0_panel_sha256": GATE0_SHA, "selector_sha256": sha(__file__),
        "teacher_or_student_outcomes_used_for_selection": False,
        "excluded_prior_games": len(excluded),
        "gate1_train": training, "snapshot_dev": development,
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("train-root", "v2-panel", "v4-panel", "gate0-panel", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = select(args.train_root, args.v2_panel, args.v4_panel, args.gate0_panel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output),
                      "gate1_train": len(result["gate1_train"]),
                      "snapshot_dev": len(result["snapshot_dev"]),
                      "excluded_prior_games": result["excluded_prior_games"]}))


if __name__ == "__main__":
    main()
