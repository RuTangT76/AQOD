"""Meaningful CPU checks; these do not claim GPU or ALFWorld integration passed."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from aqod_online_trust import AQODTrustController, PairedLabel, StudentRelativeValuePosterior
from aqod_verl.contracts import (
    assert_same_tokenizer, teacher_worker_config, validate_recipe, sha256, verify_panel,
    verify_gates,
)
from aqod_verl.environment import AQODEnvironmentManager
from aqod_verl.objective import sampled_opd_signal
from aqod_verl.trust import TrustSession


RECIPE = Path(__file__).parents[1] / 'aqod_verl/configs/gate1_2x4090.json'


class AdapterTest(unittest.TestCase):
    def test_teacher_role_does_not_inherit_student_lora(self):
        original = json.loads(RECIPE.read_text())['actor_rollout_ref']
        actor = teacher_worker_config(original, 'actor_rollout')
        teacher = teacher_worker_config(original, 'ref')
        self.assertEqual(actor['model']['lora_rank'], 8)
        self.assertEqual(teacher['model']['lora_rank'], 0)
        self.assertEqual(teacher['model']['path'], original['ref']['model']['path'])
        self.assertEqual(original['model']['lora_rank'], 8)
        self.assertNotEqual(actor['model']['path'], teacher['model']['path'])

    def test_recipe_rejects_method_drift(self):
        recipe = json.loads(RECIPE.read_text())
        validate_recipe(recipe)
        for modify in (
            lambda c: c['algorithm'].update(atod={'enable_coef_anneal': True}),
            lambda c: c['actor_rollout_ref']['actor'].update(entropy_coeff=0.001),
            lambda c: c['aqod'].update(stage='online_aqod'),
            lambda c: c['data'].update(truncation='left'),
            lambda c: c['trainer'].update(resume_mode='auto'),
        ):
            changed = deepcopy(recipe)
            modify(changed)
            with self.assertRaises(ValueError):
                validate_recipe(changed)

    def test_sampled_opd_sign_padding_and_nonfinite_values(self):
        student = np.array([[-2., -1., np.nan], [-1., -2., np.inf]])
        teacher = np.array([[-1., -2., np.nan], [-1., -1., np.inf]])
        mask = np.array([[1, 1, 0], [1, 1, 0]])
        signal = sampled_opd_signal(student, teacher, mask)
        np.testing.assert_equal(signal, [[1, -1, 0], [0, 1, 0]])
        teacher[0, 0] = np.nan
        with self.assertRaises(ValueError):
            sampled_opd_signal(student, teacher, mask)
        with self.assertRaises(ValueError):
            sampled_opd_signal([[0]], [[0]], [[0]])

    def test_equal_vocab_size_does_not_prove_tokenizer_compatibility(self):
        class Tokenizer:
            bos_token_id, eos_token_id, pad_token_id = 1, 2, 0
            chat_template = '{{ messages }}'
            def __init__(self, vocab): self.vocab = vocab
            def get_vocab(self): return self.vocab
        student = Tokenizer({'a': 0, 'b': 1})
        assert_same_tokenizer(student, Tokenizer({'a': 0, 'b': 1}))
        with self.assertRaises(ValueError):
            assert_same_tokenizer(student, Tokenizer({'a': 1, 'b': 0}))

    def test_pending_query_and_budget_survive_json_checkpoint(self):
        posterior = StudentRelativeValuePosterior(
            ['bias', 'competence'], prior_precision=1, label_variance=0.25)
        session = TrustSession(AQODTrustController(posterior, question_budget=1,
                               confidence_multiplier=1, min_query_sd=0))
        decision = session.decide(decision_id='q', features={'bias': 1., 'competence': 0.4},
            student_policy_sha256='student', public_history_sha256='history',
            remaining_action_budget=4, student_action='a', teacher_action='b')
        self.assertEqual(decision.mode, 'question')
        resumed = TrustSession.from_state_dict(json.loads(json.dumps(session.state_dict())))
        self.assertEqual(resumed.controller.remaining, 0)
        label = PairedLabel('q', 'student', 'student', 'student',
                            'history', 'history', 'history', 4, 4, 4, 0., 1., True)
        with self.assertRaises(ValueError):
            resumed.observe(replace(label, teacher_branch_policy_sha256='different'), environment_steps=3)
        self.assertEqual(resumed.paired_environment_steps, 3)
        self.assertEqual(resumed.controller.posterior.labels_seen, 0)
        self.assertTrue(resumed.observe(label, environment_steps=4))
        self.assertEqual(resumed.controller.posterior.labels_seen, 1)
        self.assertEqual(resumed.paired_environment_steps, 7)
        with self.assertRaises(ValueError):
            resumed.controller.observe_question(label)
        self.assertEqual(resumed.controller.remaining, 0)

    def test_panel_disjointness_and_file_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'train'
            root.mkdir()
            a, b = root / 'a', root / 'b'
            a.write_text('a'); b.write_text('b')
            panel = {'teacher_or_student_outcomes_used_for_selection': False,
                     'gate1_train': [{'game': 'a', 'sha256': sha256(a)}],
                     'snapshot_dev': [{'game': 'b', 'sha256': sha256(b)}]}
            path = Path(tmp) / 'panel.json'
            path.write_text(json.dumps(panel))
            verify_panel(path, sha256(path), root)
            panel['snapshot_dev'] = panel['gate1_train']
            path.write_text(json.dumps(panel))
            with self.assertRaises(ValueError):
                verify_panel(path, sha256(path), root)

    def test_failed_gate_cannot_start_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'gate_t.json'
            path.write_text(json.dumps({'gate_t_passed': False}))
            with self.assertRaisesRegex(ValueError, 'has not passed'):
                verify_gates({'gate_t': {'path': str(path), 'sha256': sha256(path)}})

    def test_environment_preserves_public_history_and_stops_invalid_output(self):
        class Client:
            def __init__(self): self.calls = []; self.closed = False
            def __enter__(self): return self
            def __exit__(self, *args): self.closed = True
            def call(self, operation, **kwargs):
                self.calls.append((operation, kwargs))
                return {'public': {'observation': operation, 'admissible_commands': ['look']},
                        'audit': {'done': False, 'won': False, 'hidden_plan': 'SECRET'}}
        clients = []
        def factory():
            clients.append(Client())
            return clients[-1]
        prompts = []
        def prompt(history):
            prompts.append(deepcopy(history))
            return json.dumps(history)
        manager = AQODEnvironmentManager(factory, prompt,
                                        lambda text: None if '\n' in text else text)
        obs, _ = manager.reset([{'game': 'a', 'seed': 7}, {'game': 'b', 'seed': 7}])
        self.assertNotIn('SECRET', str(obs))
        manager.step(['look', 'bad\nformat'])
        self.assertEqual(manager.environment_steps, 1)
        self.assertEqual(len(clients[1].calls), 1)
        manager.step(['look', 'look'])
        self.assertEqual(manager.environment_steps, 2)
        self.assertEqual(len(clients[1].calls), 1)
        self.assertEqual(prompts[-2][0]['action'], 'look')
        manager.close()
        self.assertTrue(all(c.closed for c in clients))


if __name__ == '__main__':
    unittest.main()
