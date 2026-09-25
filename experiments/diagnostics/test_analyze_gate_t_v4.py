"""Synthetic integrity check for the preregistered V4 Gate T analyzer."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


HERE = Path(__file__).parent
PANEL = HERE / 'aqod_teacher_v4_budget_selection.json'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def create_report(root, role, games, wins):
    root.mkdir()
    config = {'role': role, 'selection_manifest_sha256': '8970df8a9692a57d2f48206efb8d83ae2c992a9d4382f6127d90f9a39e71cea5',
              'train_manifest_sha256': '8970df8a9692a57d2f48206efb8d83ae2c992a9d4382f6127d90f9a39e71cea5',
              'source_sha256': '1116c0bcd53d1caf8e6ebe12e85cc38d5eb80fc0a381b5467bb826a7977083a7',
              'prompt_source_sha256': 'c49bd3e20908c2830d5147462ed9647583d71eb8f958af4cbf384c09b1e70884',
              'selection': games, 'seed': 2026092501, 'max_steps': 40,
              'max_new_tokens': 32, 'context_length': 8192,
              'prompt_format': 'current_commands', 'decoding': 'greedy',
              'thinking': False,
              'model_config_sha256': '2B' if role == 'student_base' else '9B',
              'adapter_sha256': 'synthetic-final-adapter' if role == 'teacher_rl' else None}
    save(root / 'run.json', config)
    episodes = []
    for i, item in enumerate(games):
        success = i in wins
        episodes.append({'game': item['game'], 'task_type': item['task_type'],
                         'won': success, 'stop_reason': 'environment_done' if success else 'step_limit',
                         'steps': 1, 'unlisted_commands': 0})
        trace = [{'event': 'generation', 'text': 'look', 'command': 'look',
                  'listed_as_admissible': True}]
        (root / f'episode-{i:02d}.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in trace), encoding='utf-8')
    save(root / 'episodes.json', episodes)
    save(root / 'state.json', {'stage': 'complete', 'complete': True,
                              'completed_games': 60, 'successes': len(wins),
                              'invalid_actions': 0, 'total_steps': 60})


def main():
    games = json.loads(PANEL.read_text(encoding='utf-8'))['eval']
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        for role, name, wins in (
            ('student_base', 'student', set()),
            ('teacher_raw', 'raw', set()),
            ('teacher_rl', 'rl', {i for i in range(12)})):
            create_report(root / name, role, games, wins)
        def invoke(output):
            return subprocess.run([sys.executable, str(HERE / 'analyze_gate_t_v4.py'),
                '--student', str(root / 'student'), '--raw', str(root / 'raw'),
                '--rl', str(root / 'rl'), '--panel', str(PANEL), '--output', str(root / output)],
                capture_output=True, text=True)
        first = invoke('analysis-valid')
        assert first.returncode == 0, first.stderr
        result = json.loads((root / 'analysis-valid' / 'metrics.json').read_text())
        assert result['gate_t_passed'] and result['rl_vs_student']['net_wins'] == 12
        raw_config = json.loads((root / 'raw' / 'run.json').read_text())
        raw_config['seed'] += 1
        save(root / 'raw' / 'run.json', raw_config)
        second = invoke('analysis-invalid')
        assert second.returncode != 0 and 'Protocol seed changed' in second.stderr
        print('V4 synthetic qualification and protocol rejection passed')


if __name__ == '__main__':
    main()
