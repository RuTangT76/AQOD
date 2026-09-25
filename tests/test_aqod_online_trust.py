"""Small contract tests for the provisional AQOD trust controller."""

import unittest
from dataclasses import replace

from aqod_online_trust import (
    AQODTrustController,
    PairedLabel,
    StudentRelativeValuePosterior,
)


FEATURES = {'bias': 1.0, 'competence': 0.4}
STUDENT_SHA = 'a' * 64
HISTORY_SHA = 'h' * 64
REMAINING_STEPS = 19


def controller(question_budget=1):
    posterior = StudentRelativeValuePosterior(
        ('bias', 'competence'), prior_precision=1.0, label_variance=0.25)
    return AQODTrustController(
        posterior, question_budget=question_budget,
        confidence_multiplier=1.0, min_query_sd=0.0)


def label(decision_id, *, policy_sha=STUDENT_SHA, valid=True,
          student_return=0.0, teacher_return=1.0):
    return PairedLabel(
        decision_id, policy_sha, policy_sha, policy_sha,
        HISTORY_SHA, HISTORY_SHA, HISTORY_SHA,
        REMAINING_STEPS, REMAINING_STEPS, REMAINING_STEPS,
        student_return, teacher_return, valid)


class TrustContractTest(unittest.TestCase):
    def test_question_budget_and_single_posterior_update(self):
        agent = controller()
        first = agent.decide(
            decision_id='q1', features=FEATURES,
            student_policy_sha256=STUDENT_SHA,
            public_history_sha256=HISTORY_SHA,
            remaining_action_budget=REMAINING_STEPS,
            student_action='a', teacher_action='b')
        self.assertEqual(first.mode, 'question')
        self.assertEqual(first.question_budget_remaining, 0)
        self.assertEqual(agent.posterior.labels_seen, 0)
        second = agent.decide(
            decision_id='q2', features=FEATURES,
            student_policy_sha256=STUDENT_SHA,
            public_history_sha256=HISTORY_SHA,
            remaining_action_budget=REMAINING_STEPS,
            student_action='a', teacher_action='b')
        self.assertEqual(second.mode, 'student_default')
        self.assertTrue(agent.observe_question(label('q1')))
        self.assertEqual(agent.posterior.labels_seen, 1)
        self.assertGreater(agent.posterior.estimate(
            FEATURES, confidence_multiplier=1.0).mean_delta, 0)
        with self.assertRaises(ValueError):
            agent.observe_question(label('q1'))

    def test_mismatched_student_cannot_poison_or_consume_query(self):
        agent = controller()
        agent.decide(
            decision_id='q1', features=FEATURES,
            student_policy_sha256=STUDENT_SHA,
            public_history_sha256=HISTORY_SHA,
            remaining_action_budget=REMAINING_STEPS,
            student_action='a', teacher_action='b')
        with self.assertRaisesRegex(ValueError, 'different student'):
            agent.observe_question(label('q1', policy_sha='b' * 64))
        self.assertIn('q1', agent.pending)
        self.assertEqual(agent.posterior.labels_seen, 0)
        invalid_branch = PairedLabel(
            'q1', STUDENT_SHA, STUDENT_SHA, 'c' * 64,
            HISTORY_SHA, HISTORY_SHA, HISTORY_SHA,
            REMAINING_STEPS, REMAINING_STEPS, REMAINING_STEPS,
            0.0, 1.0, True)
        with self.assertRaisesRegex(ValueError, 'same queried student'):
            agent.observe_question(invalid_branch)
        self.assertIn('q1', agent.pending)
        self.assertTrue(agent.observe_question(label('q1')))

    def test_invalid_replay_has_no_value_label(self):
        agent = controller()
        agent.decide(
            decision_id='q1', features=FEATURES,
            student_policy_sha256=STUDENT_SHA,
            public_history_sha256=HISTORY_SHA,
            remaining_action_budget=REMAINING_STEPS,
            student_action='a', teacher_action='b')
        self.assertFalse(agent.observe_question(label('q1', valid=False)))
        self.assertEqual(agent.posterior.labels_seen, 0)
        self.assertNotIn('q1', agent.pending)
        self.assertEqual(agent.remaining, 0)

    def test_history_and_budget_mismatch_cannot_update_posterior(self):
        agent = controller()
        agent.decide(
            decision_id='q1', features=FEATURES,
            student_policy_sha256=STUDENT_SHA,
            public_history_sha256=HISTORY_SHA,
            remaining_action_budget=REMAINING_STEPS,
            student_action='a', teacher_action='b')
        for changed, message in (
                ({'queried_history_sha256': 'x' * 64}, 'different public history'),
                ({'teacher_branch_start_sha256': 'x' * 64}, 'same public history'),
                ({'queried_remaining_steps': 18}, 'different action budget'),
                ({'teacher_branch_remaining_steps': 18}, 'same action budget')):
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(ValueError, message):
                    agent.observe_question(replace(label('q1'), **changed))
                self.assertEqual(agent.posterior.labels_seen, 0)
                self.assertIn('q1', agent.pending)
        self.assertTrue(agent.observe_question(label('q1')))

    def test_rejected_feature_schema_does_not_consume_decision(self):
        agent = controller()
        with self.assertRaisesRegex(ValueError, 'schema'):
            agent.decide(
                decision_id='q1', features={'competence': 0.4},
                student_policy_sha256=STUDENT_SHA,
                public_history_sha256=HISTORY_SHA,
                remaining_action_budget=REMAINING_STEPS,
                student_action='a', teacher_action='b')
        self.assertNotIn('q1', agent.used_decisions)
        self.assertEqual(agent.remaining, 1)
        self.assertEqual(agent.decide(
            decision_id='q1', features=FEATURES,
            student_policy_sha256=STUDENT_SHA,
            public_history_sha256=HISTORY_SHA,
            remaining_action_budget=REMAINING_STEPS,
            student_action='a', teacher_action='b').mode, 'question')

    def test_confident_positive_and_negative_value_route_without_query(self):
        for delta, expected in ((1.0, 'trust'), (-1.0, 'explore')):
            with self.subTest(expected=expected):
                agent = controller(question_budget=3)
                for _ in range(10):
                    agent.posterior.update(FEATURES, delta)
                decision = agent.decide(
                    decision_id=expected, features=FEATURES,
                    student_policy_sha256=STUDENT_SHA,
                    public_history_sha256=HISTORY_SHA,
                    remaining_action_budget=REMAINING_STEPS,
                    student_action='a', teacher_action='b')
                self.assertEqual(decision.mode, expected)
                self.assertEqual(decision.question_budget_remaining, 3)
                self.assertNotIn(expected, agent.pending)

    def test_one_posterior_can_change_trust_with_competence(self):
        agent = controller(question_budget=3)
        novice = {'bias': 1.0, 'competence': 0.0}
        competent = {'bias': 1.0, 'competence': 1.0}
        for _ in range(20):
            agent.posterior.update(novice, 1.0)
            agent.posterior.update(competent, -1.0)
        novice_choice = agent.decide(
            decision_id='novice', features=novice,
            student_policy_sha256=STUDENT_SHA,
            public_history_sha256=HISTORY_SHA,
            remaining_action_budget=REMAINING_STEPS,
            student_action='a', teacher_action='b')
        competent_choice = agent.decide(
            decision_id='competent', features=competent,
            student_policy_sha256=STUDENT_SHA,
            public_history_sha256=HISTORY_SHA,
            remaining_action_budget=REMAINING_STEPS,
            student_action='a', teacher_action='b')
        self.assertEqual(novice_choice.mode, 'trust')
        self.assertEqual(competent_choice.mode, 'explore')
        self.assertEqual(agent.remaining, 3)


if __name__ == '__main__':
    unittest.main()
