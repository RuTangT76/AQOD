"""Freeze a balanced, outcome-independent ALFWorld valid_seen diagnostic panel."""

import hashlib
import json
from pathlib import Path


ROOT = Path("/root/trust_mvp/assets/alfworld/json_2.1.1/valid_seen")
SEED = 2026092423
KINDS = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)
OUTPUT = Path("analysis/aqod_atod_gap_valid_seen_selection.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = list(ROOT.rglob("game.tw-pddl"))
    selected = []
    for kind in KINDS:
        candidates = []
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            if rel.split("/", 1)[0].startswith(kind + "-"):
                rank = hashlib.sha256(f"{SEED}:{rel}".encode()).hexdigest()
                candidates.append((rank, rel, path))
        if len(candidates) < 5:
            raise RuntimeError(f"Insufficient {kind}: {len(candidates)}")
        for rank, rel, path in sorted(candidates)[:5]:
            selected.append({"game": rel, "task_type": kind,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "rank": rank})
    doc = {
        "purpose": "External teacher in-distribution interface diagnostic; not AQOD Gate T",
        "split": "ALFWorld valid_seen",
        "selection_seed": SEED,
        "selection_rule": "5 per task type, ascending sha256(seed:relative_path)",
        "teacher_or_outcome_used_for_selection": False,
        "selected": selected,
    }
    OUTPUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT, "games", len(selected), "sha256",
          hashlib.sha256(OUTPUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
