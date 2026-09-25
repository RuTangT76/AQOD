"""Freeze independent train-split Gate0/confirmation games before V4 outcomes."""
import argparse
import hashlib
import json
from pathlib import Path


TASKS = (
    'pick_and_place_simple',
    'pick_clean_then_place_in_recep',
    'pick_cool_then_place_in_recep',
    'pick_heat_then_place_in_recep',
    'pick_two_obj_and_place',
    'look_at_obj_in_light',
)
SEED = 2026092502


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-root', type=Path, required=True)
    p.add_argument('--v2-panel', type=Path, required=True)
    p.add_argument('--v4-panel', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    v2 = json.loads(args.v2_panel.read_text(encoding='utf-8'))
    v4 = json.loads(args.v4_panel.read_text(encoding='utf-8'))
    if not all(key in v2 for key in ('train', 'eval', 'excluded')):
        raise ValueError('Incomplete V2 exclusion panel')
    if not all(key in v4 for key in ('train', 'eval')):
        raise ValueError('Incomplete V4 exclusion panel')
    excluded = set(v2['excluded'])
    excluded.update(row['game'] for key in ('train', 'eval') for row in v2[key])
    excluded.update(row['game'] for key in ('train', 'eval') for row in v4[key])
    if len(excluded) != 342:
        raise ValueError(f'Expected 342 distinct excluded games, found {len(excluded)}')
    gate0, confirmation = [], []
    for task in TASKS:
        candidates = []
        for path in args.train_root.glob(f'{task}-*/trial_*/game.tw-pddl'):
            rel = path.relative_to(args.train_root).as_posix()
            if rel in excluded:
                continue
            rank = hashlib.sha256(f'{SEED}:{rel}'.encode()).hexdigest()
            candidates.append((rank, rel, path))
        candidates.sort()
        if len(candidates) < 10:
            raise ValueError(f'Insufficient unused games: {task}')
        for target, rows in ((gate0, candidates[:5]),
                             (confirmation, candidates[5:10])):
            for rank, rel, path in rows:
                target.append({'game': rel, 'task_type': task,
                               'sha256': sha(path), 'rank': rank})
    games = [row['game'] for row in gate0 + confirmation]
    if len(games) != 60 or len(set(games)) != 60 or set(games) & excluded:
        raise ValueError('Gate0/confirmation overlap')
    doc = {
        'purpose': 'Conditional same-family Qwen3.5-9B teacher Gate0 and sealed confirmation',
        'split': 'ALFWorld train',
        'seed': SEED,
        'selection_rule': 'SHA256(seed:relative_path) within task type; first 5 Gate0, next 5 confirmation',
        'v2_panel_sha256': sha(args.v2_panel),
        'v4_panel_sha256': sha(args.v4_panel),
        'selector_sha256': sha(Path(__file__)),
        'teacher_or_outcome_used_for_selection': False,
        'excluded_game_count': len(excluded),
        'gate0': gate0,
        'confirmation': confirmation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(doc, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'sha256': sha(args.output),
                      'gate0': len(gate0), 'confirmation': len(confirmation),
                      'excluded_game_count': len(excluded)}))


if __name__ == '__main__':
    main()
