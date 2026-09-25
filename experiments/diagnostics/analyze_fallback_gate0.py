"""Frozen, outcome-blind analysis for the independent fallback Gate 0 panel.

Run only after both collection and paired branching report complete. The script
verifies the deterministic state selection and reports replay failures without
replacing them. It never chooses a policy, checkpoint, prompt, or gate outcome.
"""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


SELECTION_SEED = 2026092453
BOOTSTRAP_SEED = 2026092454
BOOTSTRAP_DRAWS = 20000
N_GAMES = 30
MAX_PAIRS = 120


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def selected_by_frozen_rule(candidates):
    groups = defaultdict(list)
    for candidate in candidates:
        groups[(candidate["task_type"], candidate["quartile"])].append(candidate)
    for group in groups.values():
        group.sort(key=lambda item: hashlib.sha256(
            f"{SELECTION_SEED}:{item['game_sha256']}:{item['step']}:"
            f"{item['public_state_sha256']}".encode()).hexdigest())
    chosen = []
    while len(chosen) < MAX_PAIRS and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(chosen) < MAX_PAIRS:
                chosen.append(groups[key].pop(0))
    return chosen


def load_collection(root, panel):
    run = read(root / "run.json")
    state = read(root / "state.json")
    selection = read(root / "selection.json")
    manifest = read(panel)
    require(run["phase"] == "collect", "Wrong collection phase")
    require(state.get("complete") is True and state.get("stage") == "complete",
            "Collection is incomplete")
    require(run["selection_manifest_sha256"] == sha(panel), "Panel hash mismatch")
    require(manifest.get("teacher_or_outcome_used_for_selection") is False,
            "Panel was not selected independently")
    games = manifest["gate0"]
    require(len(games) == N_GAMES and run["games"] == games,
            "Collection game list differs from frozen panel")
    require(len({g["game"] for g in games}) == N_GAMES,
            "Duplicate panel games")
    episodes = []
    queries = {}
    candidates = []
    for index, game in enumerate(games):
        episode = read(root / f"game-{index:02d}.json")
        require(episode["game"] == game, f"Wrong game at {index}")
        require(len(episode["queries"]) <= 40, f"Too many states at {index}")
        for query in episode["queries"]:
            key = (game["game"], query["step"],
                   query["public_state_sha256"])
            require(key not in queries, f"Duplicate queried state {key}")
            queries[key] = query
        for candidate in episode["candidates"]:
            key = (candidate["game"], candidate["step"],
                   candidate["public_state_sha256"])
            require(key in queries, f"Candidate without query {key}")
            query = queries[key]
            require(candidate["game"] == game["game"] and
                    candidate["game_sha256"] == game["sha256"] and
                    candidate["task_type"] == game["task_type"],
                    f"Candidate game metadata changed {key}")
            require(candidate["quartile"] == candidate["step"] // 10 and
                    candidate["student_action"] == query["student_action"] and
                    candidate["teacher_action"] == query["teacher_action"] and
                    candidate["teacher_seed"] == query["teacher_seed"],
                    f"Candidate action or stratum differs from query {key}")
            require(candidate["student_action"] != candidate["teacher_action"],
                    f"Agreement included as candidate {key}")
            candidates.append(candidate)
        episodes.append(episode)
    require(selection["n_student_states"] == len(queries),
            "Queried-state count changed")
    require(selection["n_disagreements"] == len(candidates),
            "Disagreement count changed")
    require(selection["n_selected"] == len(selection["selected"]) <= MAX_PAIRS,
            "Selected-pair count invalid")
    require(selection["selected"] == selected_by_frozen_rule(candidates),
            "Selected pairs differ from preregistered hash rule")
    require(selection["student_full_game_wins"] ==
            sum(bool(ep["student_won"]) for ep in episodes),
            "Full-game student wins changed")
    return run, selection, episodes, queries


def load_branches(root, collect_root, selection):
    run = read(root / "run.json")
    state = read(root / "state.json")
    require(run["phase"] == "branch", "Wrong branch phase")
    require(state.get("complete") is True and state.get("stage") == "complete",
            "Branching is incomplete")
    require(run["collect_run_sha256"] == sha(collect_root / "run.json") and
            run["collect_selection_sha256"] == sha(collect_root / "selection.json"),
            "Branching used a different collection or selection")
    require(run["n_pairs"] == selection["n_selected"],
            "Branch count differs from selection")
    pairs = []
    for index, candidate in enumerate(selection["selected"]):
        pair = read(root / f"pair-{index:03d}.json")
        require(pair["index"] == index and pair["candidate"] == candidate,
                f"Pair {index} does not match selected state")
        if pair.get("replay_error") is not None:
            require(pair.get("delta") is None, f"Replay error with delta {index}")
        else:
            own = pair["student_branch"]
            advised = pair["teacher_branch"]
            require(type(own["won"]) is bool and type(advised["won"]) is bool,
                    f"Nonbinary terminal outcome {index}")
            require(pair["delta"] == int(advised["won"]) - int(own["won"]),
                    f"Inconsistent delta {index}")
        pairs.append(pair)
    return run, pairs


def load_teacher_full_game(root, panel, collect_run):
    run = read(root / "run.json")
    state = read(root / "state.json")
    summary = read(root / "summary.json")
    games = read(panel)["gate0"]
    require(run["role"] == "frozen_teacher_full_game_descriptive",
            "Wrong teacher full-game role")
    require(state.get("complete") is True and state.get("stage") == "complete",
            "Teacher full-game evaluation incomplete")
    require(run["selection_manifest_sha256"] == sha(panel) and
            run["games"] == games, "Teacher full-game panel changed")
    require(run["gate0_source_sha256"] == collect_run["source_sha256"] and
            run["paper_prompt_source_sha256"] ==
            collect_run["paper_prompt_source_sha256"],
            "Teacher full-game policy source changed")
    require(run["teacher_config_sha256"] ==
            collect_run["teacher_config_sha256"] and
            run["teacher_weight_sha256"] ==
            collect_run["teacher_weight_sha256"] and
            run["teacher_model"] == collect_run["teacher_model"],
            "Teacher full-game checkpoint changed")
    require(run["train_root"] == collect_run["train_root"] and
            run["environment_python"] == collect_run["environment_python"] and
            run["bridge_source_sha256"] ==
            collect_run["bridge_source_sha256"],
            "Teacher full-game environment changed")
    require(run["env_seed"] == collect_run["env_seed"] and
            run["max_steps"] == 40 and
            run["decoding"] == {"do_sample": True, "temperature": 0.4,
                                "top_p": 1.0, "max_new_tokens": 512,
                                "enable_thinking": False},
            "Teacher full-game protocol changed")
    outcomes = [read(root / f"game-{i:02d}.json") for i in range(N_GAMES)]
    require(all(outcomes[i]["game"] == games[i] for i in range(N_GAMES)),
            "Teacher full-game outcomes are out of panel order")
    wins = sum(bool(x["won"]) for x in outcomes)
    require(summary["games"] == N_GAMES and summary["wins"] == wins,
            "Teacher full-game summary does not match raw outcomes")
    return wins, outcomes


def mean(values):
    return sum(values) / len(values) if values else None


def game_cluster_ci(rows, games):
    by_game = defaultdict(list)
    for row in rows:
        if row["delta"] is not None:
            by_game[row["game"]].append(row["delta"])
    if not any(by_game.values()):
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    draws = []
    zero_pair_resamples = 0
    while len(draws) < BOOTSTRAP_DRAWS:
        sampled = [rng.choice(games) for _ in games]
        numerator = sum(sum(by_game[g]) for g in sampled)
        denominator = sum(len(by_game[g]) for g in sampled)
        if denominator:
            draws.append(numerator / denominator)
        else:
            zero_pair_resamples += 1
    draws.sort()
    return {"interval_95": [draws[int(0.025 * BOOTSTRAP_DRAWS)],
                            draws[int(0.975 * BOOTSTRAP_DRAWS)]],
            "resamples": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "zero_pair_resamples_discarded": zero_pair_resamples,
            "cluster_count": len(games),
            "clusters_with_valid_pairs": sum(bool(by_game[g]) for g in games)}


def summarize(rows, games):
    valid = [r for r in rows if r["delta"] is not None]
    missing = [r for r in rows if r["delta"] is None]
    deltas = [r["delta"] for r in valid]
    n = len(rows)
    per_task = {}
    for task in sorted({r["task_type"] for r in rows}):
        subset = [r for r in rows if r["task_type"] == task]
        values = [r["delta"] for r in subset if r["delta"] is not None]
        per_task[task] = {"selected": len(subset), "valid": len(values),
                          "replay_failed": len(subset) - len(values),
                          "helped": values.count(1), "harmed": values.count(-1),
                          "tied": values.count(0), "mean_delta_valid": mean(values)}
    return {
        "n_selected": n,
        "n_valid_pairs": len(valid),
        "n_replay_failed_or_missing": len(missing),
        "helped": deltas.count(1), "harmed": deltas.count(-1),
        "tied": deltas.count(0),
        "mean_delta_valid_pairs": mean(deltas),
        "game_cluster_bootstrap_95ci_valid_pairs": game_cluster_ci(rows, games),
        "all_selected_state_mean_identification_bounds":
            [(sum(deltas) - len(missing)) / n,
             (sum(deltas) + len(missing)) / n] if n else None,
        "missing_pair_indices": [r["index"] for r in missing],
        "per_task": per_task,
        "per_step_quartile": {
            str(q): {"selected": sum(r["quartile"] == q for r in rows),
                     "valid": sum(r["quartile"] == q and r["delta"] is not None
                                  for r in rows),
                     "mean_delta_valid": mean([r["delta"] for r in rows
                                               if r["quartile"] == q and
                                               r["delta"] is not None])}
            for q in range(4)
        },
        "teacher_format_error_selected": sum(not r["teacher_format_valid"]
                                             for r in rows),
        "teacher_unlisted_selected": sum(r["teacher_format_valid"] and
                                         not r["teacher_listed"] for r in rows),
        "student_unlisted_selected": sum(not r["student_listed"] for r in rows),
        "student_branch_stop_reasons": dict(Counter(
            r["student_branch_stop_reason"] for r in valid)),
        "teacher_branch_stop_reasons": dict(Counter(
            r["teacher_branch_stop_reason"] for r in valid)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", type=Path, required=True)
    ap.add_argument("--branch", type=Path, required=True)
    ap.add_argument("--teacher-full-game", type=Path, required=True)
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    require(not args.output.exists(), "Output already exists")
    collect_run, selection, episodes, queries = load_collection(
        args.collect, args.panel)
    branch_run, pairs = load_branches(args.branch, args.collect, selection)
    require(branch_run["source_sha256"] == collect_run["source_sha256"],
            "Branch implementation changed from collection")
    teacher_wins, teacher_outcomes = load_teacher_full_game(
        args.teacher_full_game, args.panel, collect_run)
    games = [ep["game"]["game"] for ep in episodes]
    rows = []
    for pair in pairs:
        candidate = pair["candidate"]
        query = queries[(candidate["game"], candidate["step"],
                         candidate["public_state_sha256"])]
        own = pair.get("student_branch", {})
        advised = pair.get("teacher_branch", {})
        rows.append({
            "index": pair["index"], "game": candidate["game"],
            "task_type": candidate["task_type"], "step": candidate["step"],
            "quartile": candidate["quartile"],
            "public_state_sha256": candidate["public_state_sha256"],
            "student_action": candidate["student_action"],
            "student_raw_text": query["student"].get("text"),
            "student_listed": query["student_listed"],
            "teacher_action": candidate["teacher_action"],
            "teacher_raw_text": query["teacher"].get("raw_text"),
            "teacher_action_tag_found": query["teacher"].get("action_tag_found", False),
            "teacher_format_valid": candidate["teacher_action"] is not None,
            "teacher_listed": candidate["teacher_listed"],
            "teacher_seed": candidate["teacher_seed"],
            "replay_equal": pair.get("replay_error") is None,
            "replay_error": pair.get("replay_error"),
            "student_branch_won": own.get("won"),
            "teacher_branch_won": advised.get("won"),
            "student_branch_stop_reason": own.get("stop_reason"),
            "teacher_branch_stop_reason": advised.get("stop_reason"),
            "delta": pair["delta"],
        })
    summary = summarize(rows, games)
    summary.update({
        "scope": "Independent action-level fallback Gate 0; no tuning or automatic pass",
        "analysis_source_sha256": sha(__file__),
        "panel_sha256": sha(args.panel),
        "collect_run_sha256": sha(args.collect / "run.json"),
        "collect_selection_sha256": sha(args.collect / "selection.json"),
        "branch_run_sha256": sha(args.branch / "run.json"),
        "teacher_full_game_run_sha256": sha(args.teacher_full_game / "run.json"),
        "collection_source_sha256": collect_run["source_sha256"],
        "branch_source_sha256": branch_run["source_sha256"],
        "student_full_game_wins": selection["student_full_game_wins"],
        "teacher_full_game_wins": teacher_wins,
        "teacher_full_game_by_task": {
            task: {"wins": sum(x["won"] for x in teacher_outcomes
                                if x["game"]["task_type"] == task),
                   "games": sum(x["game"]["task_type"] == task
                                for x in teacher_outcomes)}
            for task in sorted({x["game"]["task_type"] for x in teacher_outcomes})
        },
        "n_student_visited_states": selection["n_student_states"],
        "n_disagreements": selection["n_disagreements"],
        "disagreement_rate": selection["n_disagreements"] /
                             selection["n_student_states"]
                             if selection["n_student_states"] else None,
        "n_excluded_disagreements": selection["n_disagreements"] -
                                    selection["n_selected"],
        "n_collection_student_format_errors": sum(
            ep["stop_reason"] == "format_error" for ep in episodes),
        "n_collection_student_context_limits": sum(
            ep["stop_reason"] == "context_limit" for ep in episodes),
        "caution": "The primary complete-pair bootstrap excludes explicit replay failures; inspect the all-selected-state identification bounds and missing-pair records before any inference. The teacher interface differs from the student interface and is an ATOD paper-style approximation. Gate T was not passed by this run.",
    })
    args.output.write_text(json.dumps({"summary": summary, "pairs": rows},
                                      indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(args.output, sha(args.output))


if __name__ == "__main__":
    main()
