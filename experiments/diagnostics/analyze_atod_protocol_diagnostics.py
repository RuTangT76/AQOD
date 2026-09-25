"""Analyze frozen external-teacher protocol diagnostics after both complete.

No model inference or selection is performed here. This analysis refuses partial
reports, mismatched panels, or a changed released checkpoint.
"""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(p * len(values)))]


def load_report(path):
    run = read(path / "run.json")
    state = read(path / "state.json")
    if state.get("stage") != "complete" or not state.get("complete"):
        raise ValueError(f"Incomplete report: {path}")
    metrics = read(path / "metrics.json")
    episodes = read(path / "episodes.json")
    if len(episodes) != 30 or metrics.get("n") != 30:
        raise ValueError(f"Expected 30 complete episodes: {path}")
    selection = run["selection"]
    if [x["game"] for x in selection] != [x["game"] for x in episodes]:
        raise ValueError(f"Episode/selection order mismatch: {path}")
    if len({x["game"] for x in episodes}) != 30:
        raise ValueError(f"Duplicate game: {path}")
    if sum(x["won"] for x in episodes) != metrics["successes"]:
        raise ValueError(f"Success mismatch: {path}")
    if sum(x["steps"] for x in episodes) != metrics["total_steps"]:
        raise ValueError(f"Step mismatch: {path}")
    return run, state, metrics, episodes


def trace_summary(path, episodes):
    counts = Counter()
    raw_examples = []
    per_type = defaultdict(Counter)
    for i, episode in enumerate(episodes):
        with (path / f"episode-{i:02d}.jsonl").open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        gens = [row for row in rows if row["event"] == "generation"]
        transitions = [row for row in rows if row["event"] == "transition"]
        if len(transitions) != episode["steps"]:
            raise ValueError(f"Transition count mismatch: {path} episode {i}")
        counts["generations"] += len(gens)
        counts["steps"] += episode["steps"]
        counts["unlisted_actions"] += episode["unlisted_commands"]
        counts[f"stop_{episode['stop_reason']}"] += 1
        per_type[episode["task_type"]]["games"] += 1
        per_type[episode["task_type"]]["successes"] += int(episode["won"])
        per_type[episode["task_type"]]["unlisted_actions"] += episode["unlisted_commands"]
        per_type[episode["task_type"]]["steps"] += episode["steps"]
        for gen in gens:
            counts["action_tag_found"] += int(gen.get("action_tag_found", False))
            counts["generation_hit_limit"] += int(gen.get("generation_hit_limit", False))
            raw = gen.get("raw_text", gen.get("text", ""))
            counts["open_think_tag"] += int("<think>" in raw)
            counts["close_think_tag"] += int("</think>" in raw)
            counts["both_think_tags"] += int("<think>" in raw and "</think>" in raw)
            counts["executed_generations"] += int("listed_as_admissible" in gen)
            counts["listed_generations"] += int(gen.get("listed_as_admissible", False))
            if gen.get("generation_hit_limit") and len(raw_examples) < 5:
                raw_examples.append({"episode": i, "step": gen["step"],
                                     "output_tokens": gen.get("output_tokens"),
                                     "raw_start": raw[:300]})
    if counts["steps"] != sum(x["steps"] for x in episodes):
        raise ValueError(f"Trace step mismatch: {path}")
    if counts["steps"] != counts["executed_generations"]:
        raise ValueError(f"Executed generation count mismatch: {path}")
    result = dict(counts)
    result["unlisted_per_step"] = counts["unlisted_actions"] / counts["steps"]
    result["action_tag_rate"] = counts["action_tag_found"] / counts["generations"]
    result["both_think_tags_rate"] = counts["both_think_tags"] / counts["generations"]
    result["token_limit_examples"] = raw_examples
    result["by_task_type"] = {kind: dict(values) for kind, values in sorted(per_type.items())}
    return result


def paired(short_episodes, paper_episodes):
    short = {x["game"]: int(x["won"]) for x in short_episodes}
    paper = {x["game"]: int(x["won"]) for x in paper_episodes}
    if set(short) != set(paper):
        raise ValueError("Unseen report panels differ")
    games = [x["game"] for x in paper_episodes]
    differences = [paper[g] - short[g] for g in games]
    rng = random.Random(2026092424)
    boot = [sum(rng.choice(differences) for _ in games) / len(games)
            for _ in range(20000)]
    return {
        "comparison": "paper-style minus AQOD short-action, same released 4B teacher and games",
        "paper_only_success": sum(paper[g] and not short[g] for g in games),
        "short_only_success": sum(short[g] and not paper[g] for g in games),
        "both_success": sum(short[g] and paper[g] for g in games),
        "both_failure": sum(not short[g] and not paper[g] for g in games),
        "mean_difference": sum(differences) / len(games),
        "paired_game_bootstrap_95ci": [quantile(boot, 0.025), quantile(boot, 0.975)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--short", type=Path, required=True)
    parser.add_argument("--paper-unseen", type=Path, required=True)
    parser.add_argument("--paper-seen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    short_run, _, short_metrics, short_episodes = load_report(args.short)
    unseen_run, _, unseen_metrics, unseen_episodes = load_report(args.paper_unseen)
    seen_run, _, seen_metrics, seen_episodes = load_report(args.paper_seen)
    if short_run["model_config_sha256"] != unseen_run["model_config_sha256"] or \
       short_run["model_config_sha256"] != seen_run["model_config_sha256"]:
        raise ValueError("Released teacher model config differs")
    if short_run["selection_manifest_sha256"] != unseen_run["selection_manifest_sha256"]:
        raise ValueError("Paired unseen manifest differs")
    if short_run["selection"] != unseen_run["selection"]:
        raise ValueError("Paired unseen selections differ")
    if seen_run["selection_manifest_sha256"] == unseen_run["selection_manifest_sha256"]:
        raise ValueError("Seen and unseen panels unexpectedly identical")
    report = {
        "scope": "external released-teacher interface and distribution diagnostics only",
        "source_sha256": sha(Path(__file__)),
        "models_match_on_config_sha256": short_run["model_config_sha256"],
        "panels": {
            "short_unseen_manifest_sha256": short_run["selection_manifest_sha256"],
            "paper_unseen_manifest_sha256": unseen_run["selection_manifest_sha256"],
            "paper_seen_manifest_sha256": seen_run["selection_manifest_sha256"],
        },
        "short_unseen": {"metrics": short_metrics,
                         "trace": trace_summary(args.short, short_episodes)},
        "paper_unseen": {"metrics": unseen_metrics,
                         "trace": trace_summary(args.paper_unseen, unseen_episodes)},
        "paper_seen": {"metrics": seen_metrics,
                       "trace": trace_summary(args.paper_seen, seen_episodes)},
        "paired_unseen": paired(short_episodes, unseen_episodes),
        "interpretation_limits": [
            "Paper-style runs approximate but do not reproduce the ATOD source pipeline.",
            "Paper-style vs short-action changes prompt, parser, decoding, and step/token limits together.",
            "Seen vs unseen uses different games and is descriptive, not a paired split effect.",
            "The external checkpoint may have been selected using validation performance.",
            "None of these results retroactively passes AQOD Gate T.",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(args.output, "sha256", sha(args.output))


if __name__ == "__main__":
    main()
