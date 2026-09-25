"""Synthetic protocol checks; never loads models or ALFWorld outcomes."""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


HERE = Path(__file__).parent


def load_module(name, path):
    source = HERE / path
    if not source.exists() and path == "aqod_same_family_gate0.py":
        source = HERE.parent / "trustlab" / path
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class SameFamilyGate0Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name, attrs in {
            "trustlab": {},
            "trustlab.alfworld_bridge": {"EnvironmentClient": object},
            "trustlab.alfworld_rollout": {"parse_command": lambda x: x},
            "trustlab.aqod_prompt": {"build_current_commands_prompt": lambda x: ""},
            "trustlab.modeling": {"encode_prompt": object,
                                  "load_inference": object},
        }.items():
            module = types.ModuleType(name)
            module.__dict__.update(attrs)
            sys.modules[name] = module
        cls.collector = load_module("same_family_collector_test", "aqod_same_family_gate0.py")
        cls.analyzer = load_module("same_family_analyzer_test", "analyze_same_family_gate0.py")

    def test_gate_t_and_adapter_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"frozen weights")
            gate = root / "gate.json"
            audit = root / "audit.json"
            args = types.SimpleNamespace(gate_t_analysis=gate, checkpoint_audit=audit,
                                         teacher_adapter=adapter)
            run_hash = "same-rl-run"
            save(gate, {"gate_t_passed": True,
                        "panel_sha256": self.collector.GATE_T_PANEL_SHA,
                        "reports": {"rl": {"provenance": {"run.json": run_hash}}}})
            save(audit, {"linkage_passed": True, "rl_run_sha256": run_hash,
                         "adapter_sha256": self.collector.file_sha(
                             adapter / "adapter_model.safetensors")})
            self.assertEqual(self.collector.check_qualification(args)["rl_run_sha256"],
                             run_hash)
            save(gate, {"gate_t_passed": False,
                        "panel_sha256": self.collector.GATE_T_PANEL_SHA,
                        "reports": {"rl": {"provenance": {"run.json": run_hash}}}})
            with self.assertRaisesRegex(ValueError, "must both pass"):
                self.collector.check_qualification(args)

    def test_full_analysis_and_tamper_rejection(self):
        a = self.analyzer
        a.BOOT_DRAWS = 100
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            panel_path, train_path = root / "panel.json", root / "train.json"
            collect, branch, teacher = (root / x for x in ("collect", "branch", "teacher"))
            kinds = ("pick_and_place_simple", "look_at_obj_in_light",
                     "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
                     "pick_cool_then_place_in_recep", "pick_two_obj_and_place")
            games = [{"game": f"game-{i}", "task_type": kinds[i % 6],
                      "sha256": f"game-hash-{i}"} for i in range(30)]
            save(panel_path, {"gate0": games,
                              "teacher_or_outcome_used_for_selection": False})
            save(train_path, {"train": [{"game": "other-train"}],
                              "eval": [{"game": "other-eval"}]})
            a.PANEL_SHA, a.TRAIN_SHA = a.sha(panel_path), a.sha(train_path)
            a.COLLECTOR_SHA = "source-hash"
            a.PROMPT_SHA = "prompt-hash"
            crun = {"phase": "collect", "source_sha256": "source-hash",
                    "selection_manifest_sha256": a.PANEL_SHA,
                    "train_manifest_sha256": a.TRAIN_SHA, "games": games,
                    "env_seed": 2026092503,
                    "student_prompt_source_sha256": "prompt-hash",
                    "teacher_adapter_sha256": "adapter-hash",
                    "gate_t_analysis_sha256": "gate-hash",
                    "checkpoint_audit_sha256": "audit-hash"}
            save(collect / "run.json", crun)
            save(collect / "state.json", {"stage": "complete", "complete": True,
                                           "completed_games": 30})
            candidates = []
            for i, game in enumerate(games):
                start_public = {"observation": f"before-{i}",
                                "admissible_commands": ["student move", "teacher move"]}
                state_hash = a.public_sha(start_public)
                candidate = {"game": game["game"], "game_sha256": game["sha256"],
                             "task_type": game["task_type"], "step": 0, "quartile": 0,
                             "public_state_sha256": state_hash,
                             "prefix_actions": [], "prefix_public_sha256": [state_hash],
                             "student_action": "student move", "teacher_action": "teacher move",
                             "teacher_format_error": False, "teacher_listed": True}
                candidates.append(candidate)
                save(collect / f"game-{i:02d}.json",
                     {"game": game, "reset": {"public": start_public,
                                                  "audit": {"done": False, "won": False}},
                      "student_won": False, "stop_reason": "environment_done",
                      "steps": 1,
                      "queries": [{"step": 0, "public_state_sha256": state_hash,
                                   "student_prompt_sha256": "prompt-input-hash",
                                   "teacher_prompt_sha256": "prompt-input-hash",
                                   "student": {"text": "student move"},
                                   "teacher": {"text": "teacher move"},
                                   "student_action": "student move",
                                   "teacher_action": "teacher move",
                                   "student_listed": True, "teacher_listed": True}],
                      "candidates": [candidate],
                      "transitions": [{"step": 0, "action": "student move",
                                       "public": {"observation": "after",
                                                  "admissible_commands": []},
                                       "audit": {"done": True, "won": False}}]})
            chosen = a.select(candidates)
            save(collect / "selection.json",
                 {"n_student_states": 30, "n_disagreements": 30,
                  "n_selected": 30, "student_full_game_wins": 0,
                  "selected": chosen})
            save(branch / "run.json",
                 {"phase": "branch", "source_sha256": "source-hash",
                  "collect_run_sha256": a.sha(collect / "run.json"),
                  "collect_selection_sha256": a.sha(collect / "selection.json"),
                  "n_pairs": 30})
            save(branch / "state.json", {"stage": "complete", "complete": True,
                                          "completed_pairs": 30})
            for i, candidate in enumerate(chosen):
                save(branch / f"pair-{i:03d}.json",
                     {"index": i, "candidate": candidate, "delta": 1,
                      "student_branch": {"won": False, "steps": 1,
                                         "stop_reason": "environment_done",
                                         "intervention_action": "student move",
                                         "intervention_public": {"observation": "after",
                                                                 "admissible_commands": []},
                                         "intervention_audit": {"done": True, "won": False},
                                         "continuation": []},
                      "teacher_branch": {"won": True, "steps": 1,
                                         "stop_reason": "environment_done",
                                         "intervention_action": "teacher move",
                                         "continuation": []}})
            save(teacher / "run.json",
                 {"role": "teacher_rl", "selection_manifest_sha256": a.PANEL_SHA,
                  "adapter_sha256": "adapter-hash",
                  "gate_t_analysis_sha256": "gate-hash",
                  "checkpoint_audit_sha256": "audit-hash",
                  "seed": 2026092503, "prompt_format": "current_commands",
                  "max_steps": 40, "max_new_tokens": 32,
                  "context_length": 8192, "decoding": "greedy"})
            save(teacher / "state.json", {"complete": True, "completed_games": 30})
            save(teacher / "metrics.json", {"n": 30, "successes": 30})
            save(teacher / "episodes.json",
                 [{"game": g["game"], "won": True} for g in games])
            result = a.analyze(collect, branch, teacher, panel_path, train_path)
            self.assertTrue(result["summary"]["gate0_passed"])
            self.assertEqual(result["summary"]["helped"], 30)
            episode_path = collect / "game-00.json"
            episode = read_json(episode_path)
            save(episode_path, {**episode, "candidates": []})
            with self.assertRaisesRegex(ValueError, "Disagreement candidates omit"):
                a.analyze(collect, branch, teacher, panel_path, train_path)
            save(episode_path, episode)
            pair = read_json(branch / "pair-000.json")
            pair["student_branch"]["intervention_public"] = {"observation": "tampered"}
            save(branch / "pair-000.json", pair)
            with self.assertRaisesRegex(ValueError, "own-action suffix mismatch"):
                a.analyze(collect, branch, teacher, panel_path, train_path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
