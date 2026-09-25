"""Minimal AQOD trust posterior and decision contract (no model rollout code).

The learned target is paired terminal teacher-action value to the *current*
student. This module has no absolute teacher-correctness verifier, epoch switch,
or separate discrimination model. It is provisional until Gates 0-4 pass.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Estimate:
    mean_delta: float
    epistemic_sd: float
    lower: float
    upper: float


@dataclass(frozen=True)
class Decision:
    mode: str
    estimate: Estimate | None
    question_budget_remaining: int
    decision_id: str


@dataclass(frozen=True)
class PairedLabel:
    decision_id: str
    queried_student_policy_sha256: str
    student_branch_policy_sha256: str
    teacher_branch_policy_sha256: str
    queried_history_sha256: str
    student_branch_start_sha256: str
    teacher_branch_start_sha256: str
    queried_remaining_steps: int
    student_branch_remaining_steps: int
    teacher_branch_remaining_steps: int
    student_branch_return: float
    teacher_branch_return: float
    replay_valid: bool

    @property
    def delta(self):
        if not self.replay_valid:
            raise ValueError('Invalid replay has no paired value label')
        if not (self.queried_student_policy_sha256 and
                self.queried_student_policy_sha256 ==
                self.student_branch_policy_sha256 ==
                self.teacher_branch_policy_sha256):
            raise ValueError('Both branches must continue with the same queried student')
        if not (self.queried_history_sha256 and
                self.queried_history_sha256 ==
                self.student_branch_start_sha256 ==
                self.teacher_branch_start_sha256):
            raise ValueError('Both branches must start from the same public history')
        if not (isinstance(self.queried_remaining_steps, int) and
                self.queried_remaining_steps > 0 and
                self.queried_remaining_steps ==
                self.student_branch_remaining_steps ==
                self.teacher_branch_remaining_steps):
            raise ValueError('Both branches must have the same action budget')
        for value in (self.student_branch_return, self.teacher_branch_return):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Branch returns must be finite and within [0,1]')
        return self.teacher_branch_return - self.student_branch_return


class StudentRelativeValuePosterior:
    """Small online Bayesian linear estimator for E[Delta | current student].

    `feature_names` and their normalization must be frozen before collecting
    online labels. The schema must include current student competence. The
    posterior's uncertainty is epistemic uncertainty in the expected Delta;
    it is not a claim that the ternary terminal outcome is Gaussian.
    """

    def __init__(self, feature_names, *, prior_precision, label_variance):
        names = tuple(feature_names)
        if not names or len(names) != len(set(names)) or 'competence' not in names:
            raise ValueError('Unique frozen features must include competence')
        if not math.isfinite(prior_precision) or prior_precision <= 0 or \
                not math.isfinite(label_variance) or label_variance <= 0:
            raise ValueError('Positive finite prior precision and label variance required')
        self.feature_names = names
        self.prior_precision = float(prior_precision)
        self.label_variance = float(label_variance)
        self.precision = np.eye(len(names), dtype=np.float64) * prior_precision
        self.natural_mean = np.zeros(len(names), dtype=np.float64)
        self.labels_seen = 0

    def vector(self, features):
        if set(features) != set(self.feature_names):
            raise ValueError('Feature schema changed')
        x = np.asarray([features[name] for name in self.feature_names],
                       dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite feature')
        if not 0 <= features['competence'] <= 1:
            raise ValueError('Competence must be a pre-outcome rate in [0,1]')
        return x

    def estimate(self, features, *, confidence_multiplier):
        if not math.isfinite(confidence_multiplier) or confidence_multiplier <= 0:
            raise ValueError('Positive finite confidence multiplier required')
        x = self.vector(features)
        mean_weights = np.linalg.solve(self.precision, self.natural_mean)
        variance = max(0.0, float(x @ np.linalg.solve(self.precision, x)))
        mean = float(x @ mean_weights)
        sd = math.sqrt(variance)
        radius = confidence_multiplier * sd
        return Estimate(mean, sd, mean - radius, mean + radius)

    def update(self, features, delta):
        if not math.isfinite(delta) or not -1 <= delta <= 1:
            raise ValueError('Paired terminal Delta must lie in [-1,1]')
        x = self.vector(features)
        scale = 1 / self.label_variance
        self.precision += scale * np.outer(x, x)
        self.natural_mean += scale * x * delta
        self.labels_seen += 1


class AQODTrustController:
    """One posterior drives Trust, budgeted Question and Explore.

    Discriminate is the measured ability to make correct confident decisions
    without a paired query. The actual OPD and student exploration optimizers
    are external and must share a separately logged cost budget.
    """

    def __init__(self, posterior, *, question_budget,
                 confidence_multiplier, min_query_sd):
        if not isinstance(posterior, StudentRelativeValuePosterior):
            raise TypeError('AQOD requires the single student-relative posterior')
        if not isinstance(question_budget, int) or question_budget < 0:
            raise ValueError('Question budget must be a nonnegative integer')
        if not math.isfinite(min_query_sd) or min_query_sd < 0:
            raise ValueError('Minimum query uncertainty must be nonnegative')
        if not math.isfinite(confidence_multiplier) or confidence_multiplier <= 0:
            raise ValueError('Confidence multiplier must be positive')
        self.posterior = posterior
        self.remaining = question_budget
        self.confidence_multiplier = float(confidence_multiplier)
        self.min_query_sd = float(min_query_sd)
        self.pending = {}
        self.used_decisions = set()

    def decide(self, *, decision_id, features, student_policy_sha256,
               public_history_sha256, remaining_action_budget,
               student_action, teacher_action):
        if not decision_id or decision_id in self.used_decisions:
            raise ValueError('Decision id is absent or duplicated')
        if not student_policy_sha256:
            raise ValueError('Current student policy hash is required')
        if not public_history_sha256:
            raise ValueError('Current public history hash is required')
        if not isinstance(remaining_action_budget, int) or \
                remaining_action_budget <= 0:
            raise ValueError('Positive remaining action budget required')
        if student_action == teacher_action:
            self.used_decisions.add(decision_id)
            return Decision('agree', None, self.remaining, decision_id)
        estimate = self.posterior.estimate(
            features, confidence_multiplier=self.confidence_multiplier)
        self.used_decisions.add(decision_id)
        if estimate.lower > 0:
            mode = 'trust'
        elif estimate.upper < 0:
            mode = 'explore'
        elif self.remaining and estimate.epistemic_sd >= self.min_query_sd:
            mode = 'question'
            self.remaining -= 1
            self.pending[decision_id] = (
                dict(features), student_policy_sha256,
                public_history_sha256, remaining_action_budget)
        else:
            mode = 'student_default'
        return Decision(mode, estimate, self.remaining, decision_id)

    def observe_question(self, label):
        if not isinstance(label, PairedLabel):
            raise TypeError('Question updates require a paired label')
        if label.decision_id not in self.pending:
            raise ValueError('No pending budgeted query for this decision')
        features, policy_sha, history_sha, remaining_steps = \
            self.pending[label.decision_id]
        if label.queried_student_policy_sha256 != policy_sha:
            raise ValueError('Pair uses a different student snapshot')
        if label.queried_history_sha256 != history_sha:
            raise ValueError('Pair uses a different public history')
        if label.queried_remaining_steps != remaining_steps:
            raise ValueError('Pair uses a different action budget')
        if label.replay_valid:
            delta = label.delta
            self.posterior.update(features, delta)
            del self.pending[label.decision_id]
            return True
        del self.pending[label.decision_id]
        return False
