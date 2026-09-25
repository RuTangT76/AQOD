"""Recompute Gate T comparisons from immutable per-task episode records."""
import argparse
import json
from pathlib import Path
import random
from .teacher_grpo_probe import save, sha


def load(path):
    config=json.loads((path/'run.json').read_text())
    episodes=json.loads((path/'episodes.json').read_text())
    if len(episodes)!=30 or len({x['game'] for x in episodes})!=30:
        raise ValueError(f'Incomplete Gate T report: {path}')
    if config['selection'] != [dict(game=x['game'],task_type=x['task_type'],
                                   sha256=item['sha256'])
                               for x,item in zip(episodes,config['selection'])]:
        raise ValueError('Episode and selection order differ')
    return config,episodes


def compare(a,b,seed):
    pairs=list(zip(a,b))
    diff=[int(y['won'])-int(x['won']) for x,y in pairs]
    rng=random.Random(seed)
    means=[]
    for _ in range(10000):
        means.append(sum(diff[rng.randrange(len(diff))] for _ in diff)/len(diff))
    means.sort()
    return {'mean_difference':sum(diff)/len(diff),
            'bootstrap_95ci':[means[250],means[9750]],
            'a_fail_b_success':sum(x>0 for x in diff),
            'a_success_b_fail':sum(x<0 for x in diff),
            'both_success':sum(x['won'] and y['won'] for x,y in pairs),
            'both_failure':sum(not x['won'] and not y['won'] for x,y in pairs)}


def summarize(xs):
    return {'n':len(xs),'successes':sum(x['won'] for x in xs),
            'invalid_actions':sum(x['unlisted_commands'] for x in xs),
            'total_steps':sum(x['steps'] for x in xs),
            'mean_steps':sum(x['steps'] for x in xs)/len(xs),
            'by_type':{t:{'n':sum(x['task_type']==t for x in xs),
                          'successes':sum(x['task_type']==t and x['won'] for x in xs)}
                       for t in sorted({x['task_type'] for x in xs})}}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--student',type=Path,required=True)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--rl',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise ValueError('Output exists')
    reports={}
    for role,path in [('student',args.student),('raw',args.raw),('rl',args.rl)]:
        config,episodes=load(path)
        reports[role]={'config':config,'episodes':episodes,'path':path}
    selections=[x['config']['selection'] for x in reports.values()]
    if not selections[0]==selections[1]==selections[2]:
        raise ValueError('Different evaluation tasks')
    if len({(x['config']['max_steps'],x['config']['max_new_tokens'],
             x['config']['context_length'],x['config']['decoding'],
             x['config']['thinking']) for x in reports.values()})!=1:
        raise ValueError('Unfair decoding or budget settings')
    s,r,t=[reports[k]['episodes'] for k in ('student','raw','rl')]
    result={'n':30,'selection_sha256':sha(args.student/'run.json'),
            'run_sha256':{k:sha(v['path']/'run.json') for k,v in reports.items()},
            'summaries':{k:summarize(v['episodes']) for k,v in reports.items()},
            'raw_vs_student':compare(s,r,20260923),
            'rl_vs_student':compare(s,t,20260923),
            'rl_vs_raw':compare(r,t,20260923),
            'gate_t_decision':'requires_scientific_review',
            'scientific_mvp_passed':False}
    args.output.mkdir(parents=True)
    save(args.output/'metrics.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__': main()
