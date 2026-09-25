"""Independent 48-game valid_seen raw9B/vanilla2B same-prompt controls."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from .alfworld_bridge_seen import EnvironmentClient
from .alfworld_rollout import run_episode
from .aqod_prompt import build_current_commands_prompt
from .modeling import encode_prompt, load_inference
from .aqod_mainline import assert_teacher_frozen
from .teacher_grpo_probe import save, sha


def run(args):
    import torch
    if args.output.exists():
        raise ValueError('Output exists; refusing to overwrite')
    source = json.loads(args.selection_manifest.read_text())
    if source.get('teacher_or_student_outcomes_used') is not False:
        raise ValueError('Evaluation selection must precede teacher outcomes')
    if sha(args.selection_manifest) != 'c98a4e7bc803abf9979f3e648dde87716507924bf2fd98285983e2113c74e7b5':
        raise ValueError('Unexpected frozen valid_seen control selection')
    if args.train_root.resolve(strict=True).name != 'valid_seen':
        raise ValueError('Controls must use official valid_seen split')
    games = source['selected']
    if len(games) != 48 or len({x['game'] for x in games}) != 48 or \
            sorted(x['task_type'] for x in games) != sorted(x for x in (
                'pick_and_place_simple', 'pick_clean_then_place_in_recep',
                'pick_cool_then_place_in_recep', 'pick_heat_then_place_in_recep',
                'pick_two_obj_and_place', 'look_at_obj_in_light') for _ in range(8)):
        raise ValueError('Expected 48 balanced distinct valid_seen games')
    selection = []
    for item in games:
        game = item['game']
        path = (args.train_root/game).resolve(strict=True)
        if not path.is_relative_to(args.train_root.resolve(strict=True)):
            raise ValueError('Game outside train root')
        digest = sha(path)
        if 'sha256' in item and item['sha256'] != digest:
            raise ValueError('Evaluation game hash mismatch')
        selection.append({'game':game, 'task_type':item['task_type'], 'sha256':digest})
    config = {'role': args.role, 'model': str(args.model),
              'adapter': None,
              'model_config_sha256': sha(args.model/'config.json'),
              'adapter_sha256': None,
              'selection_manifest_sha256': sha(args.selection_manifest),
              'source_sha256': sha(__file__),
              'prompt_source_sha256': sha(Path(__file__).parent/'aqod_prompt.py'),
              'prompt_format': args.prompt_format, 'selection':selection,
              'max_steps':args.max_steps, 'max_new_tokens':args.max_new_tokens,
              'context_length':args.context_length, 'seed':args.seed,
              'decoding':'greedy', 'thinking':False,
              'scientific_mvp_passed':False}
    args.output.mkdir(parents=True)
    save(args.output/'run.json', config)
    state = {'stage':'loading_model', 'complete':False, 'completed_games':0}
    save(args.output/'state.json', state)
    started = time.time()
    try:
        torch.manual_seed(args.seed)
        model, tokenizer = load_inference(str(args.model), None, args.device)
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
                                       emit,history_format='flat',
                                       prompt_builder=(build_current_commands_prompt if args.prompt_format == 'current_commands' else None))
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
    p.add_argument('--role',choices=('student_base','teacher_raw'),required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--environment-python',type=Path,required=True)
    p.add_argument('--train-root',type=Path,required=True)
    p.add_argument('--selection-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--prompt-format',choices=('current_commands',),default='current_commands')
    p.add_argument('--max-steps',type=int,default=40)
    p.add_argument('--max-new-tokens',type=int,default=32)
    p.add_argument('--context-length',type=int,default=8192)
    p.add_argument('--seed',type=int,default=2026092508)
    args=p.parse_args()
    if (args.max_steps,args.max_new_tokens,args.context_length,args.seed) != \
            (40,32,8192,2026092508):
        p.error('The valid_seen control protocol is frozen')
    expected_model = 'Qwen3.5-9B' if args.role == 'teacher_raw' else 'Qwen3.5-2B'
    if args.model.resolve(strict=True).name != expected_model:
        p.error('Control role and model path disagree')
    run(args)

if __name__=='__main__': main()
