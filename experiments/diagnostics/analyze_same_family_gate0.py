"""Frozen-protocol analysis for conditional V4 same-family paired Gate0.

Reads complete reports; never queries a model, selects a checkpoint or launches
student training. A failed or incomplete Gate0 cannot be rescued by filtering.
"""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


PANEL_SHA = "c32ff2cd60fb54e52523689d18bb9149671ec7cf2124f0295ab8ceab795eb22b"
TRAIN_SHA = "8970df8a9692a57d2f48206efb8d83ae2c992a9d4382f6127d90f9a39e71cea5"
COLLECTOR_SHA = "4d63f2a7c3119d3b29e3ee823b6ace9411f7b42a1fa2f2babb879f5674aa356c"
PROMPT_SHA = "c49bd3e20908c2830d5147462ed9647583d71eb8f958af4cbf384c09b1e70884"
SELECT_SEED = 2026092504
BOOT_SEED = 2026092505
BOOT_DRAWS = 20000
MANIP = {"pick_clean_then_place_in_recep", "pick_cool_then_place_in_recep",
         "pick_heat_then_place_in_recep", "pick_two_obj_and_place"}


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


def public_sha(public):
    payload = {"observation": public["observation"],
               "admissible_commands": public["admissible_commands"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def parsed(text):
    command = text.strip()
    return None if not command or "\n" in command or "\r" in command or len(command) > 4096 else command


def reconstruct_episode(ep):
    transitions, queries = ep["transitions"], ep["queries"]
    steps = len(transitions)
    require(ep["steps"] == steps and steps <= 40 and
            len(queries) in (steps, steps + 1),
            "Collected query/transition count is inconsistent")
    publics = [ep["reset"]["public"]] + [x["public"] for x in transitions]
    require(ep["reset"]["audit"]["done"] is False,
            "Collected game was terminal at reset")
    for i, transition in enumerate(transitions):
        require(transition["step"] == i and
                transition["action"] == queries[i]["student_action"],
                "Collected student transition changed")
    require(ep["student_won"] == bool((transitions[-1]["audit"] if transitions
                                      else ep["reset"]["audit"])["won"]),
            "Collected student win differs from environment audit")
    if len(queries) == steps + 1:
        require(steps < 40 and queries[-1]["student_action"] is None and
                ep["stop_reason"] in ("format_error", "context_limit"),
                "Extra query is not a recorded student format/context failure")
    else:
        require(ep["stop_reason"] == ("environment_done" if
                transitions and transitions[-1]["audit"]["done"] else "step_limit") and
                (steps == 40 or transitions[-1]["audit"]["done"]),
                "Collected stop reason or step budget changed")
    expected = []
    for i, query in enumerate(queries):
        state = publics[i]
        student = None if query["student"].get("context_limit") else parsed(query["student"]["text"])
        teacher = None if query["teacher"].get("context_limit") else parsed(query["teacher"]["text"])
        require(query["step"] == i and query["public_state_sha256"] == public_sha(state) and
                query["student_prompt_sha256"] == query["teacher_prompt_sha256"] and
                query["student_action"] == student and query["teacher_action"] == teacher and
                query["student_listed"] == (student in state["admissible_commands"]) and
                query["teacher_listed"] == (teacher in state["admissible_commands"]),
                "Collected query does not match its public state or raw model outputs")
        if student is not None and student != teacher:
            expected.append({"game": ep["game"]["game"],
                             "game_sha256": ep["game"]["sha256"],
                             "task_type": ep["game"]["task_type"],
                             "step": i, "quartile": i // 10,
                             "public_state_sha256": public_sha(state),
                             "prefix_actions": [x["action"] for x in transitions[:i]],
                             "prefix_public_sha256": [public_sha(x) for x in publics[:i + 1]],
                             "student_action": student, "teacher_action": teacher,
                             "teacher_format_error": teacher is None,
                             "teacher_listed": query["teacher_listed"]})
    require(ep["candidates"] == expected,
            "Disagreement candidates omit or alter queried states")
    return expected


def select(candidates):
    grouped = defaultdict(list)
    for item in candidates:
        grouped[(item["task_type"], item["quartile"])].append(item)
    for group in grouped.values():
        group.sort(key=lambda x: hashlib.sha256(
            f"{SELECT_SEED}:{x['game_sha256']}:{x['step']}:"
            f"{x['public_state_sha256']}".encode()).hexdigest())
    selected = []
    while len(selected) < 120 and any(grouped.values()):
        for key in sorted(grouped):
            if grouped[key] and len(selected) < 120:
                selected.append(grouped[key].pop(0))
    return selected


def own_suffix(episode, candidate, branch):
    if branch["steps"] > 40:
        return False
    replayed = [{"step": candidate["step"], "action": candidate["student_action"],
                 "public": branch["intervention_public"],
                 "audit": branch["intervention_audit"]}]
    replayed.extend({"step": x["step"], "action": x["command"],
                     "public": x["public"], "audit": x["audit"]}
                    for x in branch["continuation"] if x["command"] is not None)
    return (branch["intervention_action"] == candidate["student_action"] and
            replayed == episode["transitions"][candidate["step"]:] and
            branch["won"] == episode["student_won"] and
            branch["stop_reason"] == episode["stop_reason"])


def bootstrap(rows, games):
    by_game = defaultdict(list)
    for row in rows:
        if row["delta"] is not None:
            by_game[row["game"]].append(row["delta"])
    if not any(by_game.values()):
        return None
    rng = random.Random(BOOT_SEED)
    draws = []
    empty = 0
    while len(draws) < BOOT_DRAWS:
        sampled = [rng.choice(games) for _ in games]
        den = sum(len(by_game[g]) for g in sampled)
        if den:
            draws.append(sum(sum(by_game[g]) for g in sampled) / den)
        else:
            empty += 1
    draws.sort()
    return {"interval_95": [draws[int(.025 * BOOT_DRAWS)],
                            draws[int(.975 * BOOT_DRAWS)]],
            "draws": BOOT_DRAWS, "seed": BOOT_SEED,
            "empty_resamples_discarded": empty}


def analyze(collect, branch, teacher, panel_path, train_path):
    require(sha(panel_path) == PANEL_SHA, "Gate0 panel hash changed")
    require(sha(train_path) == TRAIN_SHA, "V4 train/Gate T panel hash changed")
    panel = read(panel_path)
    require(panel.get("teacher_or_outcome_used_for_selection") is False,
            "Gate0 panel is not outcome blind")
    games = panel["gate0"]
    require(len(games) == 30 and len({x["game"] for x in games}) == 30,
            "Gate0 panel is not 30 distinct games")
    train = read(train_path)
    used = {x["game"] for x in train["train"] + train["eval"]}
    require(not used.intersection(x["game"] for x in games),
            "Gate0 overlaps teacher train/Gate T")

    crun, cstate, selection = (read(collect / name) for name in
                               ("run.json", "state.json", "selection.json"))
    require(crun.get("phase") == "collect" and cstate.get("complete") is True and
            cstate.get("stage") == "complete" and cstate.get("completed_games") == 30,
            "Collection incomplete")
    require(crun.get("selection_manifest_sha256") == PANEL_SHA and
            crun.get("train_manifest_sha256") == TRAIN_SHA and
            crun.get("games") == games and crun.get("env_seed") == 2026092503 and
            crun.get("source_sha256") == COLLECTOR_SHA and
            crun.get("student_prompt_source_sha256") == PROMPT_SHA,
            "Collection protocol changed")
    require(crun.get("teacher_adapter_sha256") and
            crun.get("gate_t_analysis_sha256") and
            crun.get("checkpoint_audit_sha256"),
            "Teacher qualification provenance missing")
    episodes = [read(collect / f"game-{i:02d}.json") for i in range(30)]
    require(all(ep["game"] == games[i] for i, ep in enumerate(episodes)),
            "Collection game order changed")
    ep_by_game = {ep["game"]["game"]: ep for ep in episodes}
    queried = {}
    candidates = []
    for ep in episodes:
        candidates.extend(reconstruct_episode(ep))
        for query in ep["queries"]:
            key = (ep["game"]["game"], query["step"],
                   query["public_state_sha256"])
            require(key not in queried, "Duplicate teacher query")
            queried[key] = query
    require(selection["n_student_states"] == len(queried) and
            selection["n_disagreements"] == len(candidates) and
            selection["n_selected"] == len(selection["selected"]) and
            selection["selected"] == select(candidates) and
            selection["student_full_game_wins"] == sum(ep["student_won"] for ep in episodes),
            "Outcome-blind disagreement selection changed")

    brun, bstate = read(branch / "run.json"), read(branch / "state.json")
    require(brun.get("phase") == "branch" and bstate.get("complete") is True and
            bstate.get("stage") == "complete" and
            bstate.get("completed_pairs") == selection["n_selected"],
            "Paired branches incomplete")
    require(brun.get("collect_run_sha256") == sha(collect / "run.json") and
            brun.get("collect_selection_sha256") == sha(collect / "selection.json") and
            brun.get("source_sha256") == crun.get("source_sha256") and
            brun.get("n_pairs") == selection["n_selected"],
            "Paired branch source or selection changed")
    rows = []
    for index, candidate in enumerate(selection["selected"]):
        pair = read(branch / f"pair-{index:03d}.json")
        require(pair.get("index") == index and pair.get("candidate") == candidate,
                f"Pair {index} differs from frozen selection")
        query = queried[(candidate["game"], candidate["step"],
                         candidate["public_state_sha256"])]
        replay_error = pair.get("replay_error")
        if replay_error is None:
            own, advised = pair["student_branch"], pair["teacher_branch"]
            require(own_suffix(ep_by_game[candidate["game"]], candidate, own),
                    f"Student own-action suffix mismatch at pair {index}")
            require(advised["intervention_action"] == candidate["teacher_action"] and
                    advised["steps"] <= 40,
                    f"Teacher intervention changed at pair {index}")
            require(type(own["won"]) is bool and type(advised["won"]) is bool and
                    pair["delta"] == int(advised["won"]) - int(own["won"]),
                    f"Paired outcome inconsistent at {index}")
        else:
            require(pair.get("delta") is None, "Replay failure was assigned an outcome")
        rows.append({"index": index, "game": candidate["game"],
                     "task_type": candidate["task_type"],
                     "step": candidate["step"], "quartile": candidate["quartile"],
                     "public_state_sha256": candidate["public_state_sha256"],
                     "student_action": candidate["student_action"],
                     "teacher_action": candidate["teacher_action"],
                     "student_listed": query["student_listed"],
                     "teacher_listed": query["teacher_listed"],
                     "teacher_format_error": candidate["teacher_format_error"],
                     "replay_error": replay_error, "delta": pair["delta"]})

    trun, tstate, tmetrics = (read(teacher / name) for name in
                              ("run.json", "state.json", "metrics.json"))
    require(trun.get("role") == "teacher_rl" and tstate.get("complete") is True and
            tstate.get("completed_games") == 30 and
            trun.get("selection_manifest_sha256") == PANEL_SHA and
            trun.get("adapter_sha256") == crun["teacher_adapter_sha256"] and
            trun.get("gate_t_analysis_sha256") == crun["gate_t_analysis_sha256"] and
            trun.get("checkpoint_audit_sha256") == crun["checkpoint_audit_sha256"] and
            trun.get("seed") == 2026092503 and
            trun.get("prompt_format") == "current_commands" and
            trun.get("max_steps") == 40 and trun.get("max_new_tokens") == 32 and
            trun.get("context_length") == 8192 and trun.get("decoding") == "greedy",
            "Teacher full-game control protocol or checkpoint changed")
    toutcomes = read(teacher / "episodes.json")
    require(len(toutcomes) == 30 and
            [x["game"] for x in toutcomes] == [x["game"] for x in games] and
            tmetrics.get("n") == 30 and
            tmetrics.get("successes") == sum(bool(x["won"]) for x in toutcomes),
            "Teacher full-game outcomes incomplete or mismatched")

    valid = [x for x in rows if x["delta"] is not None]
    deltas = [x["delta"] for x in valid]
    by_type = {}
    for task in sorted({x["task_type"] for x in games}):
        subset = [x for x in rows if x["task_type"] == task]
        values = [x["delta"] for x in subset if x["delta"] is not None]
        by_type[task] = {"selected": len(subset), "valid": len(values),
                         "helped": values.count(1), "harmed": values.count(-1),
                         "tied": values.count(0),
                         "mean_valid": sum(values) / len(values) if values else None}
    ci = bootstrap(rows, [x["game"] for x in games])
    helped, harmed = deltas.count(1), deltas.count(-1)
    net_positive_types = [task for task, row in by_type.items()
                          if row["helped"] > row["harmed"]]
    complete = len(valid) == len(rows) and len(rows) > 0
    checks = {
        "complete_valid_replay": complete,
        "positive_effect_and_ci": bool(deltas) and sum(deltas) > 0 and
            ci is not None and ci["interval_95"][0] > 0,
        "help_to_harm_ratio": helped >= 3 and helped >= 2 * harmed,
        "task_breadth": len(net_positive_types) >= 3 and
            bool(MANIP.intersection(net_positive_types)),
    }
    n = len(rows)
    summary = {"scope": "Conditional V4 same-family early-student Gate0 only",
               "panel_sha256": PANEL_SHA, "train_manifest_sha256": TRAIN_SHA,
               "analysis_source_sha256": sha(__file__),
               "collect_run_sha256": sha(collect / "run.json"),
               "branch_run_sha256": sha(branch / "run.json"),
               "teacher_control_run_sha256": sha(teacher / "run.json"),
               "student_full_game_wins": selection["student_full_game_wins"],
               "teacher_full_game_wins": tmetrics["successes"],
               "n_student_states": selection["n_student_states"],
               "n_disagreements": selection["n_disagreements"],
               "n_selected": n, "n_excluded": selection["n_disagreements"] - n,
               "n_valid_pairs": len(valid), "replay_errors": [x["index"] for x in rows
                                                          if x["replay_error"] is not None],
               "helped": helped, "harmed": harmed, "tied": deltas.count(0),
               "mean_delta_valid": sum(deltas) / len(deltas) if deltas else None,
               "game_cluster_bootstrap": ci,
               "all_selected_mean_bounds":
                   [(sum(deltas) - (n - len(valid))) / n,
                    (sum(deltas) + (n - len(valid))) / n] if n else None,
               "teacher_format_errors_selected": sum(x["teacher_format_error"] for x in rows),
               "teacher_unlisted_selected": sum(not x["teacher_listed"] for x in rows),
               "student_unlisted_selected": sum(not x["student_listed"] for x in rows),
               "per_task": by_type,
               "per_step_quartile": {str(q): {"selected": sum(x["quartile"] == q for x in rows),
                                               "valid": sum(x["quartile"] == q and
                                                            x["delta"] is not None for x in rows)}
                                     for q in range(4)},
               "paired_delta_counts": dict(Counter(deltas)),
               "gate0_checks": checks, "gate0_passed": all(checks.values())}
    return {"summary": summary, "pairs": rows}


def main():
    parser = argparse.ArgumentParser()
    for name in ("collect", "branch", "teacher", "panel", "train", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing existing analysis output")
    result = analyze(args.collect, args.branch, args.teacher, args.panel, args.train)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
