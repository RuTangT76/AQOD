"""Backend-independent access to the existing, single AQOD trust posterior."""

from copy import deepcopy
import numpy as np

from aqod_online_trust import (
    AQODTrustController, StudentRelativeValuePosterior,
)


class TrustSession:
    """One driver owns decisions/budget; Ray rollout workers never clone it.

    A query is consumed when issued. Invalid replay still consumes its cost.
    Checkpoints include pending questions and IDs to prevent budget resets or
    duplicate updates on restart. This is an interface, not an online trainer.
    """

    def __init__(self, controller):
        self.controller = controller
        self.teacher_proposals = 0
        self.paired_environment_steps = 0
        self.exploration_environment_steps = 0

    def decide(self, **kwargs):
        decision = self.controller.decide(**kwargs)
        self.teacher_proposals += 1
        return decision

    def observe(self, label, *, environment_steps):
        if type(environment_steps) is not int or environment_steps < 0:
            raise ValueError('Actual paired environment cost must be nonnegative')
        # Charge executed work even if replay integrity subsequently fails.
        self.paired_environment_steps += environment_steps
        return self.controller.observe_question(label)

    def record_exploration(self, environment_steps):
        if type(environment_steps) is not int or environment_steps < 0:
            raise ValueError('Actual exploration cost must be nonnegative')
        self.exploration_environment_steps += environment_steps

    def state_dict(self):
        c, p = self.controller, self.controller.posterior
        return {
            'schema': 1, 'feature_names': list(p.feature_names),
            'prior_precision': p.prior_precision, 'label_variance': p.label_variance,
            'precision': p.precision.tolist(), 'natural_mean': p.natural_mean.tolist(),
            'labels_seen': p.labels_seen, 'remaining': c.remaining,
            'confidence_multiplier': c.confidence_multiplier,
            'min_query_sd': c.min_query_sd, 'pending': deepcopy(c.pending),
            'used_decisions': sorted(c.used_decisions),
            'teacher_proposals': self.teacher_proposals,
            'paired_environment_steps': self.paired_environment_steps,
            'exploration_environment_steps': self.exploration_environment_steps,
        }

    @classmethod
    def from_state_dict(cls, state):
        if state['schema'] != 1:
            raise ValueError('Unsupported trust checkpoint schema')
        p = StudentRelativeValuePosterior(
            state['feature_names'], prior_precision=state['prior_precision'],
            label_variance=state['label_variance'])
        precision = np.asarray(state['precision'], dtype=np.float64)
        natural = np.asarray(state['natural_mean'], dtype=np.float64)
        if precision.shape != p.precision.shape or natural.shape != p.natural_mean.shape:
            raise ValueError('Trust checkpoint dimensions differ')
        if not np.isfinite(precision).all() or not np.isfinite(natural).all() or \
                not np.allclose(precision, precision.T):
            raise ValueError('Invalid posterior state')
        np.linalg.cholesky(precision)
        p.precision, p.natural_mean = precision.copy(), natural.copy()
        p.labels_seen = state['labels_seen']
        c = AQODTrustController(p, question_budget=state['remaining'],
                               confidence_multiplier=state['confidence_multiplier'],
                               min_query_sd=state['min_query_sd'])
        c.pending = {key: tuple(value) for key, value in state['pending'].items()}
        c.used_decisions = set(state['used_decisions'])
        if not set(c.pending).issubset(c.used_decisions):
            raise ValueError('Pending question lacks a recorded decision')
        for features, policy, history, budget in c.pending.values():
            p.vector(features)
            if not policy or not history or type(budget) is not int or budget <= 0:
                raise ValueError('Invalid pending query identity')
        result = cls(c)
        for name in ('teacher_proposals', 'paired_environment_steps',
                     'exploration_environment_steps'):
            value = state[name]
            if type(value) is not int or value < 0:
                raise ValueError('Invalid checkpoint cost')
            setattr(result, name, value)
        if type(p.labels_seen) is not int or p.labels_seen < 0:
            raise ValueError('Invalid label count')
        return result
