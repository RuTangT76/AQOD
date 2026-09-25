"""Post-run, trace-only audit of the released ATOD projection semantics.

This does not replay episodes or estimate counterfactual wins. It only applies
the released parser rules to already saved model generations and public states.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ACTION = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)
CJK = re.compile(r"[\u4e00-\u9fff]")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def released_projection(text):
    """Literal behavior of ATOD projection.py SHA e794d121... on one string."""
    original = text
    lower = text.lower()
    start = lower.find("<action>")
    end = lower.find("</action>")
    if start < 0 or end < 0:
        return lower[-30:], False, "missing_action_marker"
    action = lower[start + len("<action>"):end].strip().lower()
    valid = "<think>" in original and "</think>" in original and not CJK.search(original)
    return action, bool(valid), "tag_extracted"


def analyze(path):
    state = json.loads((path / "state.json").read_text(encoding="utf-8"))
    if state.get("stage") != "complete" or not state.get("complete"):
        raise ValueError(f"Incomplete diagnostic report: {path}")
    episodes = json.loads((path / "episodes.json").read_text(encoding="utf-8"))
    if len(episodes) != 30:
        raise ValueError("Expected 30 episodes")
    counts = Counter()
    examples = []
    for index, episode in enumerate(episodes):
        rows = [json.loads(line) for line in
                (path / f"episode-{index:02d}.jsonl").open(encoding="utf-8")]
        public = None
        for row in rows:
            if row["event"] in ("reset", "transition"):
                public = row["public"]
                continue
            if row["event"] != "generation":
                continue
            if public is None:
                raise ValueError(f"Missing public state: episode {index}")
            raw = row.get("raw_text")
            if raw is None:
                raise ValueError(f"Missing raw generation: episode {index}")
            projected, valid, reason = released_projection(raw)
            current = row.get("command")
            listed = {x.lower() for x in public["admissible_commands"]}
            counts["generations"] += 1
            counts["projection_flag_valid"] += valid
            counts["projected_command_listed"] += projected in listed
            counts["projected_command_empty"] += not bool(projected)
            counts["current_command_empty"] += not bool(current)
            counts["command_changed_after_projection"] += projected != (current or "").lower()
            counts[f"projection_{reason}"] += 1
            if not current:
                counts["diagnostic_format_error_generations"] += 1
                counts["format_error_fallback_listed"] += projected in listed
                if len(examples) < 10:
                    examples.append({"episode": index, "step": row["step"],
                                     "fallback": projected,
                                     "fallback_listed": projected in listed,
                                     "raw_tail": raw[-100:]})
        counts["format_error_episodes"] += episode["stop_reason"] == "format_error"
    return {"report": str(path), "source_sha256": sha(Path(__file__)),
            "atod_projection_source_sha256":
                "e794d1217d613ef4b550cfe4bfd0b39b4deb10d47493f31f78aef77b5ebf6dd98",
            "counts": dict(counts), "format_error_examples": examples,
            "limit": "Trace-only parser comparison; fallback episode success cannot be inferred."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.report)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(args.output, sha(args.output))


if __name__ == "__main__":
    main()
