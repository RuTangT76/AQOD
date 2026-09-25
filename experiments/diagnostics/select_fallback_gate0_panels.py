"""Freeze fresh ALFWorld valid_unseen panels for conditional fallback Gate 0.

Selection uses only game paths and bytes. No model outcomes or teacher scores.
The output is never overwritten.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path("/root/trust_mvp/assets/alfworld/json_2.1.1/valid_unseen")
EXCLUDED = Path("analysis/aqod_atod_gap_valid_unseen_selection.json")
OUTPUT = Path("analysis/aqod_fallback_gate0_panels_valid_unseen.json")
SEED = 2026092451
KINDS = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    excluded_doc = json.loads(EXCLUDED.read_text(encoding="utf-8"))
    excluded_rows = excluded_doc.get("eval", excluded_doc.get("selected"))
    excluded_games = {row["game"] for row in excluded_rows}
    if len(excluded_games) != 30:
        raise ValueError("Expected frozen 30-game valid_unseen diagnostic exclusion")
    files = list(ROOT.rglob("game.tw-pddl"))
    gate0, confirmation = [], []
    for kind in KINDS:
        candidates = []
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            if rel in excluded_games or not rel.split("/", 1)[0].startswith(kind + "-"):
                continue
            rank = sha_bytes(f"{SEED}:{rel}".encode("utf-8"))
            candidates.append((rank, rel, path))
        if len(candidates) < 10:
            raise RuntimeError(f"Insufficient unused {kind}: {len(candidates)}")
        ranked = sorted(candidates)
        for target, entries in ((gate0, ranked[:5]), (confirmation, ranked[5:10])):
            for rank, rel, path in entries:
                target.append({"game": rel, "task_type": kind,
                               "sha256": sha_bytes(path.read_bytes()), "rank": rank})
    all_games = [row["game"] for row in gate0 + confirmation]
    if len(all_games) != 60 or len(set(all_games)) != 60 or excluded_games.intersection(all_games):
        raise AssertionError("Panel count or exclusion failure")
    doc = {
        "purpose": "Conditional action-level fallback Gate 0 and untouched confirmation panel",
        "split": "ALFWorld valid_unseen",
        "selection_seed": SEED,
        "selection_rule": "Per task type: ascending sha256(seed:relative_path), first 5 Gate 0, next 5 confirmation",
        "excluded_manifest_sha256": sha_bytes(EXCLUDED.read_bytes()),
        "teacher_or_outcome_used_for_selection": False,
        "gate0": gate0,
        "confirmation": confirmation,
    }
    OUTPUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT, "gate0", len(gate0), "confirmation", len(confirmation),
          "sha256", sha_bytes(OUTPUT.read_bytes()))


if __name__ == "__main__":
    main()
