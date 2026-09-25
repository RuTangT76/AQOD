"""Select an independent valid_seen control panel without model outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

TASKS = (
    'pick_and_place_simple', 'pick_clean_then_place_in_recep',
    'pick_cool_then_place_in_recep', 'pick_heat_then_place_in_recep',
    'pick_two_obj_and_place', 'look_at_obj_in_light',
)
PRIOR_SHA = 'd375b3bc26e67acec0e596b888521e276c0015d1076f1123fa3b4d5fff889016'
SEED = 2026092507
PER_TYPE = 8


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def select(root, prior_path):
    root = root.resolve(strict=True)
    if root.name != 'valid_seen' or sha(prior_path) != PRIOR_SHA:
        raise ValueError('Wrong valid_seen split or prior diagnostic panel')
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    excluded = {x['game'] for x in prior['selected']}
    if len(excluded) != 30:
        raise ValueError('Expected 30 prior diagnostic games')
    by_type = {}
    for task in TASKS:
        paths = []
        for p in root.glob(f'{task}-*/trial_*/game.tw-pddl'):
            rel = p.relative_to(root).as_posix()
            if rel not in excluded:
                paths.append((hashlib.sha256(f'{SEED}:{rel}'.encode()).hexdigest(), rel, p))
        paths.sort()
        if len(paths) < PER_TYPE:
            raise ValueError(f'Insufficient unused {task} games: {len(paths)}')
        by_type[task] = [{'game': rel, 'task_type': task, 'sha256': sha(path)}
                         for _, rel, path in paths[:PER_TYPE]]
    selected = [by_type[task][i] for i in range(PER_TYPE) for task in TASKS]
    if len(selected) != 48 or len({x['game'] for x in selected}) != 48 or \
            {x['game'] for x in selected} & excluded:
        raise ValueError('Panel count or exclusion mismatch')
    return {'purpose': 'Independent GPU1 same-prompt raw9B/vanilla2B valid_seen control; not Gate T or AQOD method selection',
            'split': 'ALFWorld valid_seen', 'seed': SEED,
            'selection_rule': 'SHA256(seed:path) within task type; first 8 unused; round-robin task order',
            'prior_selection_sha256': PRIOR_SHA,
            'selector_sha256': sha(__file__),
            'teacher_or_student_outcomes_used': False,
            'selected': selected}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--valid-seen-root', type=Path, required=True)
    p.add_argument('--prior-panel', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    doc = select(args.valid_seen_root, args.prior_panel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'output': str(args.output), 'sha256': sha(args.output),
                      'games': len(doc['selected'])}))


if __name__ == '__main__':
    main()
