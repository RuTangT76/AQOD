"""Audit the frozen 60-game V4 Gate T experiment and apply its preregistered rule."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

PANEL_SHA = '8970df8a9692a57d2f48206efb8d83ae2c992a9d4382f6127d90f9a39e71cea5'
EVAL_SHA = '1116c0bcd53d1caf8e6ebe12e85cc38d5eb80fc0a381b5467bb826a7977083a7'
PROMPT_SHA = 'c49bd3e20908c2830d5147462ed9647583d71eb8f958af4cbf384c09b1e70884'
SEED = 2026092501
MANIP = {'pick_clean_then_place_in_recep', 'pick_cool_then_place_in_recep',
         'pick_heat_then_place_in_recep', 'pick_two_obj_and_place'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check(ok, reason):
    if not ok:
        raise ValueError(reason)


def trace_counts(path, episode):
    counts = Counter()
    for line in path.read_text(encoding='utf-8').splitlines():
        event = json.loads(line)
        if event.get('event') != 'generation':
            continue
        counts['generations'] += 1
        if event.get('context_limit'):
            counts['context_limits'] += 1
        else:
            counts['decisions'] += 1
            if event.get('command') is None:
                counts['format_errors'] += 1
            else:
                counts['executed'] += 1
                check(event.get('listed_as_admissible') in (True, False),
                      f'Missing admissibility: {path}')
                counts['unlisted'] += event['listed_as_admissible'] is False
    check(counts['executed'] == episode['steps'] and
          counts['unlisted'] == episode['unlisted_commands'],
          f'Trace/episode actions mismatch: {path}')
    stop = episode['stop_reason']
    check(stop in {'step_limit', 'environment_done', 'format_error', 'context_limit'},
          f'Unknown stop: {path}')
    check((stop == 'format_error') == (counts['format_errors'] == 1) and
          (stop == 'context_limit') == (counts['context_limits'] == 1),
          f'Trace/stop mismatch: {path}')
    check(counts['generations'] <= 40 and counts['format_errors'] <= 1 and
          counts['context_limits'] <= 1, f'Invalid generation count: {path}')
    return counts


def load_report(path, expected_role, panel):
    state, config, episodes = (read(path / name) for name in
                               ('state.json', 'run.json', 'episodes.json'))
    check(state.get('complete') is True and state.get('stage') == 'complete' and
          state.get('completed_games') == 60 and len(episodes) == 60,
          f'Incomplete report: {path}')
    check(config.get('role') == expected_role, f'Wrong role: {path}')
    check(config.get('selection_manifest_sha256') == PANEL_SHA and
          config.get('train_manifest_sha256') == PANEL_SHA and
          config.get('selection') == panel['eval'], f'Wrong panel: {path}')
    check(config.get('source_sha256') == EVAL_SHA and
          config.get('prompt_source_sha256') == PROMPT_SHA,
          f'Evaluator/prompt source changed: {path}')
    protocol = {'seed': SEED, 'max_steps': 40, 'max_new_tokens': 32,
                'context_length': 8192, 'prompt_format': 'current_commands',
                'decoding': 'greedy', 'thinking': False}
    for key, value in protocol.items():
        check(config.get(key) == value, f'Protocol {key} changed: {path}')
    check(bool(config.get('adapter_sha256')) == (expected_role == 'teacher_rl'),
          f'Unexpected adapter: {path}')
    totals = Counter()
    types = Counter()
    for i, (item, episode) in enumerate(zip(panel['eval'], episodes)):
        check(episode.get('game') == item['game'] and
              episode.get('task_type') == item['task_type'],
              f'Game/order mismatch: {path}, {i}')
        types[item['task_type']] += 1
        trace = path / f'episode-{i:02d}.jsonl'
        check(trace.is_file(), f'Missing trace: {trace}')
        totals.update(trace_counts(trace, episode))
    check(len(types) == 6 and all(n == 10 for n in types.values()),
          f'Unbalanced task panel: {path}')
    wins = sum(bool(e['won']) for e in episodes)
    check(state.get('successes') == wins and
          state.get('invalid_actions') == totals['unlisted'] and
          state.get('total_steps') == totals['executed'],
          f'State/episode/trace totals mismatch: {path}')
    summary = {'n': 60, 'wins': wins,
               'wins_by_type': {t: sum(bool(e['won']) for e in episodes
                                        if e['task_type'] == t) for t in sorted(types)},
               'natural_failures': sum(not e['won'] and e['stop_reason'] in
                                       {'step_limit', 'environment_done'} for e in episodes),
               'counts': dict(totals),
               'unlisted_rate': (totals['unlisted'] / totals['executed']
                                 if totals['executed'] else None),
               'format_error_rate': (totals['format_errors'] / totals['decisions']
                                     if totals['decisions'] else None)}
    provenance = {name: sha(path / name) for name in
                  ('state.json', 'run.json', 'episodes.json')}
    return {'config': config, 'episodes': episodes,
            'summary': summary, 'provenance': provenance}


def compare(a, b, panel):
    groups = {}
    for i, item in enumerate(panel['eval']):
        groups.setdefault(item['task_type'], []).append(
            int(bool(b[i]['won'])) - int(bool(a[i]['won'])))
    check(len(groups) == 6 and all(len(x) == 10 for x in groups.values()),
          'Invalid paired bootstrap strata')
    differences = [value for group in groups.values() for value in group]
    rng = random.Random(SEED)
    replicates = []
    for _ in range(10000):
        replicates.append(sum(group[rng.randrange(10)] for group in groups.values()
                              for _ in range(10)) / 60)
    replicates.sort()
    return {'net_wins': sum(differences), 'mean_difference': sum(differences) / 60,
            'bootstrap_95ci': [replicates[250], replicates[9750]],
            'helped': differences.count(1), 'harmed': differences.count(-1),
            'tied': differences.count(0),
            'net_by_type': {key: sum(value) for key, value in groups.items()}}


def main():
    parser = argparse.ArgumentParser()
    for name in ('student', 'raw', 'rl', 'panel', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    check(not args.output.exists(), 'Refusing existing output directory')
    check(sha(args.panel) == PANEL_SHA, 'Frozen panel hash changed')
    panel = read(args.panel)
    check(len(panel['train']) == 180 and len(panel['eval']) == 60 and
          panel.get('teacher_or_student_outcomes_used') is False and
          not ({x['game'] for x in panel['train']} &
               {x['game'] for x in panel['eval']}), 'Invalid panel integrity')
    reports = {name: load_report(path, role, panel) for name, role, path in
               (('student', 'student_base', args.student),
                ('raw', 'teacher_raw', args.raw), ('rl', 'teacher_rl', args.rl))}
    check(reports['raw']['config']['model_config_sha256'] ==
          reports['rl']['config']['model_config_sha256'],
          'Raw/RL teacher base configs differ')
    delta = compare(reports['student']['episodes'], reports['rl']['episodes'], panel)
    raw_delta = compare(reports['raw']['episodes'], reports['rl']['episodes'], panel)
    rl = reports['rl']['summary']
    positive_types = [task for task, net in delta['net_by_type'].items() if net > 0]
    checks = {
        'effect_size': delta['net_wins'] >= 6,
        'uncertainty': delta['bootstrap_95ci'][0] > 0,
        'breadth': len(positive_types) >= 3 and bool(set(positive_types) & MANIP),
        'action_quality': rl['unlisted_rate'] is not None and
                          rl['unlisted_rate'] <= 0.20 and
                          rl['format_error_rate'] is not None and
                          rl['format_error_rate'] <= 0.05,
        'natural_failure_set': rl['natural_failures'] >= 6,
    }
    output = {'panel_sha256': PANEL_SHA, 'evaluator_sha256': EVAL_SHA,
              'prompt_sha256': PROMPT_SHA, 'bootstrap_seed': SEED,
              'reports': {name: {key: report[key] for key in ('summary', 'provenance')}
                          for name, report in reports.items()},
              'rl_vs_student': delta, 'rl_vs_raw_diagnostic': raw_delta,
              'positive_task_types': positive_types,
              'gate_t_checks': checks, 'gate_t_passed': all(checks.values())}
    args.output.mkdir(parents=True)
    with (args.output / 'metrics.json').open('x', encoding='utf-8') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(output, ensure_ascii=False))


if __name__ == '__main__':
    main()
