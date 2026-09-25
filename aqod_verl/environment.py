"""Adapt the existing AQOD public-state bridge to verl-agent's collector API."""

from contextlib import ExitStack
from copy import deepcopy
import numpy as np


class AQODEnvironmentManager:
    def __init__(self, client_factory, prompt_builder, parser, emit=None):
        self.client_factory = client_factory
        self.prompt_builder = prompt_builder
        self.parser = parser
        self.emit = emit or (lambda row: None)
        self.stack = ExitStack()
        self.clients = []
        self.environment_steps = 0
        self.reset_count = 0

    def reset(self, kwargs):
        if kwargs is None or not len(kwargs):
            raise ValueError('Explicit manifest game/seed required; no random game reset')
        self.close()
        self.stack = ExitStack()
        self.clients, self.history, self.results, self.done = [], [], [], []
        self.games = [dict(row) for row in kwargs]
        try:
            for row in self.games:
                client = self.stack.enter_context(self.client_factory())
                result = client.call('reset', game=row['game'], seed=int(row['seed']))
                self.clients.append(client)
                self.results.append(result)
                self.history.append([{'public': deepcopy(result['public'])}])
                self.done.append(bool(result['audit']['done']))
                self.emit({'event': 'reset', 'game': row['game'], 'seed': row['seed'],
                           'public': result['public']})
            self.reset_count += len(kwargs)
            return self._observations(), self._infos([True] * len(kwargs))
        except BaseException:
            self.close()
            raise

    def _observations(self):
        return {'text': [self.prompt_builder(h) for h in self.history],
                'image': None,
                'anchor': [h[-1]['public']['observation'] for h in self.history]}

    def _infos(self, valids):
        return [{'won': bool(r['audit']['won']), 'is_action_valid': np.asarray(valid),
                 'extra.gamefile': row['game']}
                for r, valid, row in zip(self.results, valids, self.games)]

    def step(self, text_actions):
        if len(text_actions) != len(self.clients):
            raise ValueError('Action/environment batch sizes differ')
        rewards, valids = np.zeros(len(self.clients), dtype=np.float32), []
        for i, text in enumerate(text_actions):
            if self.done[i]:
                valids.append(True)
                continue
            command = self.parser(text)
            listed = command in self.results[i]['public']['admissible_commands']
            valids.append(command is not None and listed)
            self.emit({'event': 'generation', 'game': self.games[i]['game'],
                       'text': text, 'command': command, 'listed': listed,
                       'public': self.results[i]['public']})
            if command is None:
                self.done[i] = True
                continue
            result = self.clients[i].call('step', action=command)
            self.environment_steps += 1
            self.history[i][-1]['action'] = command
            self.history[i].append({'public': deepcopy(result['public'])})
            self.results[i] = result
            self.done[i] = bool(result['audit']['done'])
            rewards[i] = float(result['audit']['won'])
        return self._observations(), rewards, np.asarray(self.done), self._infos(valids)

    def success_evaluator(self, **kwargs):
        return {'success_rate': np.asarray([float(r['audit']['won']) for r in self.results])}

    def close(self):
        self.stack.close()
        self.clients = []
