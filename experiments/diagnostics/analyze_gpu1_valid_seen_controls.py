"""Read-only paired analysis of independent valid_seen raw9B and vanilla2B controls."""
import argparse
import hashlib
import json
import random
from pathlib import Path

PANEL_SHA = 'c98a4e7bc803abf9979f3e648dde87716507924bf2fd98285983e2113c74e7b5'
EVAL_SHA = '6f666f647c0963f001ebdb085629275177c12d6a6d35559a8995fe437c26fdda'
PROMPT_SHA = 'c49bd3e20908c2830d5147462ed9647583d71eb8f958af4cbf384c09b1e70884'
BOOT_SEED = 2026092509
BOOT_DRAWS = 20000


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_report(root, role, games):
    config = read(root / 'run.json')
    state = read(root / 'state.json')
    metrics = read(root / 'metrics.json')
    episodes = read(root / 'episodes.json')
    require(config.get('role') == role and config.get('source_sha256') == EVAL_SHA and
            config.get('prompt_source_sha256') == PROMPT_SHA and
            config.get('selection_manifest_sha256') == PANEL_SHA and
            config.get('selection') == games and
            config.get('adapter') is None and config.get('adapter_sha256') is None and
            config.get('prompt_format') == 'current_commands' and
            config.get('max_steps') == 40 and config.get('max_new_tokens') == 32 and
            config.get('context_length') == 8192 and config.get('seed') == 2026092508 and
            config.get('decoding') == 'greedy' and config.get('thinking') is False,
            f'{role} protocol changed')
    expected_model = 'Qwen3.5-9B' if role == 'teacher_raw' else 'Qwen3.5-2B'
    require(Path(config['model']).name == expected_model,
            f'{role} model changed')
    require(state.get('complete') is True and state.get('stage') == 'complete' and
            state.get('completed_games') == 48 and len(episodes) == 48 and
            metrics.get('n') == 48, f'{role} report incomplete')
    require([x.get('game') for x in episodes] == [x['game'] for x in games] and
            all(type(x.get('won')) is bool and 0 <= x.get('steps', -1) <= 40
                for x in episodes), f'{role} episodes changed')
    wins = sum(x['won'] for x in episodes)
    require(metrics.get('successes') == wins and
            metrics.get('success_rate') == wins / 48,
            f'{role} aggregate wins mismatch')
    trace_hashes = {}
    for i, episode in enumerate(episodes):
        trace = root / f'episode-{i:02d}.jsonl'
        require(trace.is_file(), f'{role} trace missing at {i}')
        lines = [json.loads(x) for x in trace.read_text(encoding='utf-8').splitlines()]
        require(len(lines) >= episode['steps'] + 1,
                f'{role} trace too short at {i}')
        trace_hashes[f'episode-{i:02d}.jsonl'] = sha(trace)
    return {'config': config, 'state': state, 'metrics': metrics,
            'episodes': episodes,
            'provenance': {name: sha(root / name) for name in
                           ('run.json', 'state.json', 'metrics.json', 'episodes.json')},
            'trace_sha256': trace_hashes}


def paired_interval(deltas):
    rng = random.Random(BOOT_SEED)
    draws = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas)
                   for _ in range(BOOT_DRAWS))
    return [draws[int(.025 * BOOT_DRAWS)], draws[int(.975 * BOOT_DRAWS)]]


def analyze(raw_root, student_root, panel_path):
    require(sha(panel_path) == PANEL_SHA, 'Frozen valid_seen panel changed')
    panel = read(panel_path)
    require(panel.get('teacher_or_student_outcomes_used') is False,
            'Panel was not outcome blind')
    games = [{k: x[k] for k in ('game', 'task_type', 'sha256')}
             for x in panel['selected']]
    require(len(games) == 48 and len({x['game'] for x in games}) == 48,
            'Panel count or uniqueness changed')
    raw = load_report(raw_root, 'teacher_raw', games)
    student = load_report(student_root, 'student_base', games)
    deltas = [int(a['won']) - int(b['won'])
              for a, b in zip(raw['episodes'], student['episodes'])]
    by_type = {}
    for task in sorted({x['task_type'] for x in games}):
        indices = [i for i, x in enumerate(games) if x['task_type'] == task]
        require(len(indices) == 8, 'Unbalanced task panel')
        by_type[task] = {'n': 8,
                         'raw_wins': sum(raw['episodes'][i]['won'] for i in indices),
                         'student_wins': sum(student['episodes'][i]['won'] for i in indices),
                         'raw_only': sum(deltas[i] == 1 for i in indices),
                         'student_only': sum(deltas[i] == -1 for i in indices),
                         'tie': sum(deltas[i] == 0 for i in indices)}
    return {'scope': 'Independent descriptive valid_seen control; no AQOD gate or method selection',
            'panel_sha256': PANEL_SHA, 'analysis_source_sha256': sha(__file__),
            'raw': {'metrics': raw['metrics'], 'provenance': raw['provenance'],
                    'trace_sha256': raw['trace_sha256']},
            'student': {'metrics': student['metrics'], 'provenance': student['provenance'],
                        'trace_sha256': student['trace_sha256']},
            'paired': {'games': 48, 'raw_only': deltas.count(1),
                       'student_only': deltas.count(-1), 'tie': deltas.count(0),
                       'mean_success_difference': sum(deltas) / 48,
                       'game_bootstrap_95ci': paired_interval(deltas),
                       'bootstrap_draws': BOOT_DRAWS, 'bootstrap_seed': BOOT_SEED,
                       'by_task_type': by_type,
                       'outcomes': [{'game': games[i]['game'],
                                     'task_type': games[i]['task_type'],
                                     'raw_won': raw['episodes'][i]['won'],
                                     'student_won': student['episodes'][i]['won'],
                                     'delta': deltas[i]} for i in range(48)]}}


def main():
    p = argparse.ArgumentParser()
    for name in ('raw', 'student', 'panel', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    require(not args.output.exists(), 'Refusing existing analysis output')
    result = analyze(args.raw, args.student, args.panel)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                           encoding='utf-8')
    print(json.dumps({k: result['paired'][k] for k in
                      ('games', 'raw_only', 'student_only', 'tie',
                       'mean_success_difference', 'game_bootstrap_95ci')}))


if __name__ == '__main__':
    main()
