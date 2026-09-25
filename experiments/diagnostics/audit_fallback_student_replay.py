"""Outcome-blind consistency audit for the frozen Gate 0 student branch.

The student-action branch should exactly reproduce the original greedy-student
trajectory from the selected state. This script never reads teacher-branch
outcomes or changes the preregistered paired-effect analysis.
"""

import argparse
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def expected_continuation(episode, branch_step):
    queries = episode["queries"]
    transitions = episode["transitions"]
    expected = []
    for query in queries[branch_step + 1:]:
        step = query["step"]
        action = query["student_action"]
        row = {"step": step, "output": query["student"], "command": action}
        if action is not None:
            transition = transitions[step]
            row.update({"listed": query["student_listed"],
                        "public": transition["public"],
                        "audit": transition["audit"]})
        expected.append(row)
    return expected


def audit(collect_dir, branch_dir):
    collect_state = read_json(collect_dir / "state.json")
    branch_state = read_json(branch_dir / "state.json")
    if not (collect_state.get("complete") and branch_state.get("complete")):
        raise ValueError("Both reports must be complete before the replay audit")
    episodes = [read_json(collect_dir / f"game-{i:02d}.json") for i in range(30)]
    by_game = {episode["game"]["sha256"]: episode for episode in episodes}
    if len(by_game) != 30:
        raise ValueError("Collection does not contain 30 distinct games")
    selected = read_json(collect_dir / "selection.json")["selected"]
    if len(selected) != branch_state.get("completed_pairs"):
        raise ValueError("Selected and completed pair counts differ")
    mismatches = []
    replay_errors = 0
    matched = 0
    for index, candidate in enumerate(selected):
        pair = read_json(branch_dir / f"pair-{index:03d}.json")
        if pair["index"] != index or pair["candidate"] != candidate:
            raise ValueError(f"Pair {index} does not match frozen selection")
        if "replay_error" in pair:
            replay_errors += 1
            continue
        episode = by_game[candidate["game_sha256"]]
        step = candidate["step"]
        if candidate["student_action"] != episode["transitions"][step]["action"]:
            mismatches.append({"index": index, "field": "intervention_action"})
            continue
        own = pair["student_branch"]
        expected = {
            "won": episode["student_won"],
            "stop_reason": episode["stop_reason"],
            "steps": episode["steps"],
            "continuation": expected_continuation(episode, step),
        }
        for field, value in expected.items():
            if own.get(field) != value:
                mismatches.append({"index": index, "field": field})
        if not any(item["index"] == index for item in mismatches):
            matched += 1
    return {"selected_pairs": len(selected), "matched_student_replays": matched,
            "replay_errors": replay_errors,
            "mismatches": mismatches,
            "all_nonerror_replays_match": not mismatches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", type=Path, required=True)
    parser.add_argument("--branch", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.collect, args.branch), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
