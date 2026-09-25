"""Freeze the outcome-blind veRL restart panel; never read model results."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path('/root/trust_mvp')
    sources = {
        'aqod_gate_t_v2_selection.json': '02b89be2c985afacace26d7da5e02d2fe2996ae9b108667a230f21ba20931a48',
        'aqod_teacher_v4_budget_selection.json': '8970df8a9692a57d2f48206efb8d83ae2c992a9d4382f6127d90f9a39e71cea5',
        'aqod_teacher_v4_conditional_gate0_panels.json': 'c32ff2cd60fb54e52523689d18bb9149671ec7cf2124f0295ab8ceab795eb22b',
        'aqod_gate1_competence_selection.json': 'd4c1b99a11d452bcd5b05dafba0c5220f80cf419116fc416817b20842cc08524',
    }
    excluded = set()
    for name, expected in sources.items():
        path = root/'analysis'/name
        if sha(path) != expected:
            raise ValueError('Prior manifest changed: '+name)
        doc = json.loads(path.read_text())
        for key in ('excluded','train','eval','gate0','confirmation','gate1_train','snapshot_dev'):
            for item in doc.get(key,[]):
                excluded.add(item if isinstance(item,str) else item['game'])
    if len(excluded) != 642:
        raise ValueError(f'Prior exclusion count differs: {len(excluded)}')
    tasks = ('pick_and_place_simple','pick_clean_then_place_in_recep',
             'pick_cool_then_place_in_recep','pick_heat_then_place_in_recep',
             'pick_two_obj_and_place','look_at_obj_in_light')
    train_root = root/'assets/alfworld/json_2.1.1/train'
    by_type = {}
    for task in tasks:
        candidates = [(p.relative_to(train_root).as_posix(),p)
                      for p in train_root.glob(f'{task}-*/trial_*/game.tw-pddl')
                      if p.relative_to(train_root).as_posix() not in excluded]
        candidates.sort(key=lambda x:hashlib.sha256(('2026092510:'+x[0]).encode()).hexdigest())
        if len(candidates)<41:
            raise ValueError('Insufficient unused games')
        by_type[task] = [{'game':name,'task_type':task,'sha256':sha(p)} for name,p in candidates[:41]]
    result = {'purpose':'veRL teacher restart V5; independent qualification panel',
              'seed':2026092510,'teacher_or_student_outcomes_used':False,
              'prior_manifests':sources,'excluded_prior_games':len(excluded),
              'selector_sha256':sha(__file__),
              'train':[by_type[t][i] for i in range(30) for t in tasks],
              'eval':[by_type[t][i] for i in range(30,40) for t in tasks],
              'smoke':[by_type[t][40] for t in tasks]}
    paths=[x['game'] for k in ('train','eval','smoke') for x in result[k]]
    assert len(paths)==len(set(paths))==246 and not set(paths)&excluded
    output=root/'analysis/aqod_teacher_v5_verl_panel.json'
    with output.open('x') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    print(json.dumps({'path':str(output),'sha256':sha(output),'train':180,'eval':60,'smoke':6}))


if __name__=='__main__':
    main()
