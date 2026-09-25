"""Supplemental, outcome-blind validity audit for completed Gate 0 records.

This adds denominators and action-validity counts omitted from the primary
analysis summary. It does not estimate a new effect or decide the Gate 0 gate.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from analyze_fallback_gate0 import load_branches, load_collection, sha


def count_branch_actions(branch, first_action, first_listed):
    actions = []
    if first_action is not None:
        actions.append(bool(first_listed))
    continuation = branch.get("continuation", [])
    for item in continuation:
        if item.get("command") is not None:
            actions.append(bool(item["listed"]))
    return {"executed": len(actions), "unlisted": sum(not x for x in actions),
            "continuation_attempts": len(continuation),
            "continuation_parse_failures": sum(
                item.get("command") is None for item in continuation)}


def audit(episodes, pairs):
    queries = [q for ep in episodes for q in ep["queries"]]
    valid_student = [q for q in queries if q["student_action"] is not None]
    actionable_disagreements = sum(
        q["student_action"] != q["teacher_action"] for q in valid_student)
    candidates = [c for ep in episodes for c in ep["candidates"]]
    if actionable_disagreements != len(candidates):
        raise ValueError("Actionable disagreements differ from candidate records")
    result = {
        "n_queried_states": len(queries),
        "n_student_parseable_actions": len(valid_student),
        "n_student_unparseable_actions": len(queries) - len(valid_student),
        "n_actionable_disagreements": actionable_disagreements,
        "actionable_disagreement_rate":
            actionable_disagreements / len(valid_student) if valid_student else None,
        "n_teacher_unparseable_actions_all_states": sum(
            q["teacher_action"] is None for q in queries),
        "n_teacher_missing_action_tag_all_states": sum(
            not q["teacher"].get("action_tag_found", False) for q in queries),
        "n_teacher_unlisted_parseable_actions_all_states": sum(
            q["teacher_action"] is not None and not q["teacher_listed"]
            for q in queries),
        "n_student_unlisted_parseable_actions_all_states": sum(
            q["student_action"] is not None and not q["student_listed"]
            for q in queries),
        "selected_per_task_and_quartile": {},
        "branch_actions": {},
    }
    result["selected_per_task_and_quartile"] = {
        f"{task}|q{quartile}": count
        for (task, quartile), count in sorted(Counter(
            (pair["candidate"]["task_type"], pair["candidate"]["quartile"])
            for pair in pairs).items())
    }
    valid_pairs = [p for p in pairs if p.get("replay_error") is None]
    for role, field, first in (
        ("student", "student_branch", "student_action"),
        ("teacher", "teacher_branch", "teacher_action"),
    ):
        totals = Counter()
        stop_reasons = Counter()
        for pair in valid_pairs:
            candidate = pair["candidate"]
            branch = pair[field]
            listed = candidate[f"{role}_listed"] if role == "teacher" else None
            if role == "student":
                # The collection query stores student_listed, but selection
                # stores only the action. All student branch actions are
                # accounted for as executed; the exact initial validity is
                # looked up from the corresponding collection query below.
                listed = pair["student_initial_listed"]
            counts = count_branch_actions(branch, candidate[first], listed)
            totals.update(counts)
            stop_reasons[branch["stop_reason"]] += 1
        result["branch_actions"][role] = {
            **dict(totals),
            "unlisted_rate": totals["unlisted"] / totals["executed"]
            if totals["executed"] else None,
            "stop_reasons": dict(stop_reasons),
            "format_or_context_stop_pairs":
                stop_reasons["format_error"] + stop_reasons["context_limit"] +
                stop_reasons["intervention_format_error"],
        }
    result["n_replay_failed_pairs"] = len(pairs) - len(valid_pairs)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", type=Path, required=True)
    ap.add_argument("--branch", type=Path, required=True)
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    _, selection, episodes, queries = load_collection(args.collect, args.panel)
    _, pairs = load_branches(args.branch, args.collect, selection)
    augmented = []
    for pair in pairs:
        candidate = pair["candidate"]
        query = queries[(candidate["game"], candidate["step"],
                         candidate["public_state_sha256"])]
        augmented.append({**pair, "student_initial_listed":
                          query["student_listed"]})
    report = {"scope": "Descriptive action-validity supplement only",
              "source_sha256": sha(__file__),
              "panel_sha256": sha(args.panel),
              "collect_run_sha256": sha(args.collect / "run.json"),
              "branch_run_sha256": sha(args.branch / "run.json"),
              "counts": audit(episodes, augmented)}
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(args.output, sha(args.output))


if __name__ == "__main__":
    main()
