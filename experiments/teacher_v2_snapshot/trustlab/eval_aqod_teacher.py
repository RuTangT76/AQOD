"""Fixed 30-game ALFWorld Gate T control evaluation (greedy, public input)."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from .alfworld_bridge import EnvironmentClient
from .alfworld_rollout import run_episode
from .modeling import encode_prompt, load_inference
from .aqod_mainline import assert_teacher_frozen
from .teacher_grpo_probe import save, sha


def run(args):
    import torch
    if args.output.exists():
        raise ValueError('Output exists; refusing to overwrite')
    args.output.mkdir(parents=True)
    source = json.loads(args.selection_manifest.read_text())
    if source.get('teacher_or_outcome_used_for_selection') is not False:
        raise ValueError('Evaluation selection must precede teacher outcomes')
    games = source['selected']
    if len(games) != 30 or len({x['game'] for x in games}) != 30:
        raise ValueError('Expected the frozen 30-game panel')
    train_selection = json.loads(args.train_manifest.read_text())['selection']
    train_games = {x['game'] for x in train_selection}
    if train_games.intersection(x['game'] for x in games):
        raise ValueError('Training and Gate T games overlap')
    selection = []
    for item in games:
        game = item['game']
        path = (args.train_root/game).resolve(strict=True)
        if not path.is_relative_to(args.train_root.resolve(strict=True)):
            raise ValueError('Game outside train root')
        selection.append({'game':game, 'task_type':item['task_type'], 'sha256':sha(path)})
    config = {'role': args.role, 'model': str(args.model),
              'adapter': str(args.adapter) if args.adapter else None,
              'model_config_sha256': sha(args.model/'config.json'),
              'adapter_sha256': sha(args.adapter/'adapter_model.safetensors') if args.adapter else None,
              'selection_manifest_sha256': sha(args.selection_manifest),
              'train_manifest_sha256': sha(args.train_manifest),
              'source_sha256': sha(__file__), 'selection':selection,
              'max_steps':args.max_steps, 'max_new_tokens':args.max_new_tokens,
              'context_length':args.context_length, 'seed':args.seed,
              'decoding':'greedy', 'thinking':False,
              'scientific_mvp_passed':False}
    save(args.output/'run.json', config)
    state = {'stage':'loading_model', 'complete':False, 'completed_games':0}
    save(args.output/'state.json', state)
    started = time.time()
    try:
        torch.manual_seed(args.seed)
        model, tokenizer = load_inference(str(args.model), str(args.adapter) if args.adapter else None,
                                           args.device)
        assert_teacher_frozen(model)
        outcomes=[]
        def generate(prompt):
            try:
                ids=encode_prompt(tokenizer,prompt,args.context_length,
                    args.max_new_tokens,enable_thinking=False)
            except ValueError as exc:
                if 'max_length=' not in str(exc): raise
                return {'context_limit':True,'text':'','error':str(exc)}
            tensor=torch.tensor([ids],device=args.device)
            with torch.inference_mode():
                out=model.generate(input_ids=tensor,attention_mask=torch.ones_like(tensor),
                    do_sample=False,max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,use_cache=True)
            tokens=out[0,len(ids):].tolist()
            return {'text':tokenizer.decode(tokens,skip_special_tokens=True),
                    'input_tokens':len(ids),'output_tokens':len(tokens)}
        with EnvironmentClient(args.environment_python,args.train_root) as env:
            for i,item in enumerate(selection):
                state.update(stage='evaluating',game_index=i)
                save(args.output/'state.json',state)
                with (args.output/f'episode-{i:02d}.jsonl').open('x',buffering=1) as f:
                    def emit(row): f.write(json.dumps(row,ensure_ascii=False)+'\n')
                    result=run_episode(env,item['game'],generate,args.max_steps,args.seed,
                                       emit,history_format='flat')
                result.update({'game':item['game'],'task_type':item['task_type']})
                outcomes.append(result)
                save(args.output/'episodes.json',outcomes)
                state.update(completed_games=len(outcomes),elapsed_seconds=time.time()-started)
                save(args.output/'state.json',state)
                print(json.dumps({'index':i,**result}),flush=True)
        by_type={task:{'n':sum(x['task_type']==task for x in outcomes),
                       'successes':sum(x['task_type']==task and x['won'] for x in outcomes)}
                 for task in sorted({x['task_type'] for x in outcomes})}
        metrics={'n':len(outcomes),'successes':sum(x['won'] for x in outcomes),
                 'success_rate':sum(x['won'] for x in outcomes)/len(outcomes),
                 'invalid_actions':sum(x['unlisted_commands'] for x in outcomes),
                 'total_steps':sum(x['steps'] for x in outcomes),
                 'mean_steps':sum(x['steps'] for x in outcomes)/len(outcomes),
                 'by_task_type':by_type,'elapsed_seconds':time.time()-started}
        save(args.output/'metrics.json',metrics)
        state.update(stage='complete',complete=True,**metrics)
        save(args.output/'state.json',state)
    except BaseException as exc:
        state.update(stage='failed',error=f'{type(exc).__name__}: {exc}',
                     elapsed_seconds=time.time()-started)
        save(args.output/'state.json',state)
        raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--role',choices=('student_base','teacher_raw','teacher_rl'),required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--adapter',type=Path)
    p.add_argument('--environment-python',type=Path,required=True)
    p.add_argument('--train-root',type=Path,required=True)
    p.add_argument('--selection-manifest',type=Path,required=True)
    p.add_argument('--train-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--max-steps',type=int,default=40)
    p.add_argument('--max-new-tokens',type=int,default=32)
    p.add_argument('--context-length',type=int,default=4096)
    p.add_argument('--seed',type=int,default=20260923)
    args=p.parse_args()
    if args.role == 'teacher_rl' and args.adapter is None:
        p.error('RL teacher requires a frozen adapter')
    if args.role != 'teacher_rl' and args.adapter is not None:
        p.error('Only the RL teacher may use an adapter')
    run(args)

if __name__=='__main__': main()
