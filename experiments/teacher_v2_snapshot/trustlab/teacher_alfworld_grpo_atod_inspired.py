"""ALFWorld LoRA-GRPO teacher with a bounded invalid-action penalty.

ATOD's released teacher recipe motivates tracking invalid actions. This is an
independent trajectory-level adaptation for one 9B model on one 4090; it does
not implement ATOD's distributed trainer or its student distillation method.
The frozen AQOD task split, prompt, and evaluation protocol stay unchanged.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

from .alfworld_bridge import EnvironmentClient
from .alfworld_rollout import build_prompt, parse_command
from .aqod_prompt import build_current_commands_prompt
from .modeling import encode_prompt, load_base_model, text_lora_targets
from .teacher_grpo_probe import save, sequence_logp, sha


def select_games(root, exclude_manifests, seed, per_type):
    from .alfworld_preflight import TASKS
    excluded = set()
    for path in exclude_manifests:
        if path.exists():
            payload = json.loads(path.read_text())
            items = payload.get('selection', payload.get('selected'))
            if items is None:
                raise ValueError(f'Unknown exclusion manifest schema: {path}')
            for item in items:
                game = item.get('relative_path', item.get('game'))
                if not game:
                    raise ValueError(f'Exclusion entry lacks game path: {path}')
                excluded.add(game)
    selected = []
    for task in TASKS:
        paths = [p for p in root.glob(f'{task}-*/trial_*/game.tw-pddl')
                 if str(p.relative_to(root)) not in excluded]
        paths.sort(key=lambda p: hashlib.sha256(
            f'{seed}:{p.relative_to(root)}'.encode()).hexdigest())
        if len(paths) < per_type:
            raise ValueError(f'Insufficient independent training games for {task}')
        for path in paths[:per_type]:
            selected.append({'game': str(path.relative_to(root)), 'task_type': task,
                             'sha256': sha(path)})
    return selected


def episode_seeds(seed, group_index, sample_index):
    return seed+1000*group_index, seed+1000*group_index+sample_index


def collect(model, tokenizer, args, game, environment_seed, sampling_seed):
    import torch
    torch.manual_seed(sampling_seed)
    history, actions = [], []
    invalid = 0
    stop = 'step_limit'
    with EnvironmentClient(args.environment_python, args.train_root) as env:
        result = env.call('reset', game=game, seed=environment_seed)
        history.append({'public': result['public']})
        for step in range(args.max_steps):
            try:
                prompt_fn = (build_current_commands_prompt if args.prompt_format == 'current_commands' else build_prompt)
                prompt = encode_prompt(tokenizer, prompt_fn(history),
                    args.context_length, args.max_new_tokens, enable_thinking=False)
            except ValueError as exc:
                if 'max_length=' not in str(exc):
                    raise
                stop = 'context_limit'
                break
            with torch.inference_mode():
                out = model.generate(input_ids=torch.tensor([prompt], device=args.device),
                    do_sample=True, temperature=args.temperature, top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens, use_cache=True,
                    pad_token_id=tokenizer.pad_token_id)
            completion = out[0, len(prompt):].tolist()
            text = tokenizer.decode(completion, skip_special_tokens=True)
            command = parse_command(text)
            action = {'prompt': prompt, 'completion': completion,
                      'text': text, 'command': command,
                      'listed': command in result['public']['admissible_commands']}
            actions.append(action)
            if command is None:
                stop = 'format_error'
                break
            invalid += not action['listed']
            result = env.call('step', action=command)
            history[-1]['action'] = command
            history.append({'public': result['public']})
            if result['audit']['done']:
                stop = 'environment_done'
                break
    return {'game': game, 'environment_seed': environment_seed,
            'sampling_seed': sampling_seed, 'won': bool(result['audit']['won']),
            'reward': float(result['audit']['won']), 'stop': stop,
            'steps': len(history)-1, 'invalid': invalid, 'actions': actions}


def group_relative_advantages(rewards):
    if len(rewards) < 2 or len(set(rewards)) == 1:
        return None
    mean = sum(rewards)/len(rewards)
    std = math.sqrt(sum((r-mean)**2 for r in rewards)/len(rewards))
    return [(r-mean)/(std+1e-8) for r in rewards]


def training_scores(episodes, invalid_penalty_coef):
    """Keep terminal success dominant while distinguishing invalid-heavy runs."""
    scores = []
    for episode in episodes:
        attempted = max(1, len(episode['actions']))
        invalid = episode['invalid'] + int(episode['stop'] == 'format_error')
        scores.append(episode['reward'] - invalid_penalty_coef * invalid / attempted)
    return scores


def optimize_group(model, optimizer, episodes, args):
    import torch
    rewards = [e['reward'] for e in episodes]
    scores = training_scores(episodes, args.invalid_penalty_coef)
    advantages = group_relative_advantages(scores)
    if advantages is None:
        return {'updated': False, 'reason': 'zero_group_variance',
                'rewards': rewards, 'training_scores': scores}
    # In terminal ties, the auxiliary validity signal must remain smaller than
    # a terminal-success update, even after group normalization.
    if len(set(rewards)) == 1:
        advantages = [args.invalid_only_advantage_scale * a for a in advantages]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    selected = [[a for a in e['actions'] if a['completion'] and
                 len(a['prompt']) + len(a['completion']) <= args.max_train_tokens]
                for e in episodes]
    if any(not row for row in selected):
        model.eval()
        return {'updated': False, 'reason': 'no_eligible_action', 'rewards': rewards,
                'training_scores': scores,
                'eligible_actions': list(map(len, selected))}
    losses = []
    for advantage, episode, rows in zip(advantages, episodes, selected):
        for action in rows:
            prompt, completion = action['prompt'], action['completion']
            with torch.no_grad():
                old = sequence_logp(model, prompt, completion, args.device)
                with model.disable_adapter():
                    reference = sequence_logp(model, prompt, completion, args.device)
            current = sequence_logp(model, prompt, completion, args.device)
            ratio = (current-old).exp()
            pg = -torch.minimum(ratio*advantage, ratio.clamp(0.8,1.2)*advantage)
            log_ratio = reference-current
            kl = log_ratio.exp()-log_ratio-1
            loss = (pg + args.beta*kl)/(len(episodes)*len(rows))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite GRPO loss')
            loss.backward()
            losses.append(float(loss.detach()))
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(grad_norm):
        raise RuntimeError('Nonfinite GRPO gradient')
    optimizer.step()
    model.eval()
    return {'updated': True, 'rewards': rewards, 'training_scores': scores,
            'advantages': advantages,
            'eligible_actions': list(map(len, selected)),
            'grad_norm': float(grad_norm), 'loss_sum': sum(losses)}


def run(args):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer
    if args.output.exists():
        raise ValueError('Output exists; refusing to overwrite')
    if args.group_size not in (2,4) or args.max_train_tokens > args.context_length:
        raise ValueError('This MVP supports group size 2 or 4 and bounded train tokens')
    if not (0 < args.invalid_penalty_coef < 1):
        raise ValueError('invalid penalty must be positive and below terminal reward')
    if not (0 < args.invalid_only_advantage_scale <= 1):
        raise ValueError('invalid-only advantage scale must be in (0, 1]')
    args.output.mkdir(parents=True)
    started = time.time()
    if args.selection_manifest is not None:
        selection_doc = json.loads(args.selection_manifest.read_text())
        games = selection_doc['train']
        if not games or len({x['game'] for x in games}) != len(games):
            raise ValueError('Frozen training selection must contain distinct games')
        for game in games:
            if sha(args.train_root/game['game']) != game['sha256']:
                raise ValueError('Frozen training game hash mismatch')
    else:
        games = select_games(args.train_root, args.exclude_manifest, args.seed,
                             args.games_per_type)
    if args.max_groups < 0:
        raise ValueError('max_groups must be nonnegative')
    run_games = games[:args.max_groups] if args.max_groups else games
    manifest = {'mode': 'teacher_alfworld_lora_grpo_invalid_penalty_v1',
        'reward': 'terminal ALFWorld won; train-only bounded invalid-action penalty',
        'inspiration': 'ATOD ALFWorld teacher GRPO invalid-action penalty; independent implementation',
        'reference_url': 'https://github.com/TanQitai/ATOD/blob/main/examples/grpo_teacher_trainer/run_alfworld_grpo_qwen3_4b.sh',
        'student_task_warmup': False, 'teacher_frozen_after_training': True,
        'selection': games, 'run_selection': run_games,
        'selection_manifest_sha256': sha(args.selection_manifest) if args.selection_manifest else None,
        'prompt_source_sha256': sha(Path(__file__).parent/'aqod_prompt.py'), 'arguments': {k: [str(x) for x in v] if isinstance(v,list)
             else str(v) for k,v in vars(args).items()},
        'source_sha256': sha(__file__), 'model_config_sha256': sha(args.model/'config.json'),
        'alfworld_bridge_sha256': sha(Path(__file__).parent/'alfworld_bridge.py'),
        'start_time': started, 'scientific_mvp_passed': False}
    save(args.output/'manifest.json', manifest)
    state = {'stage': 'loading', 'complete': False, 'groups': 0, 'updates': 0}
    save(args.output/'state.json', state)
    try:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = load_base_model(str(args.model), args.device)
        model.config.use_cache = False
        if hasattr(model.config, 'text_config'):
            model.config.text_config.use_cache = False
        targets = text_lora_targets(model, include_mlp=True)
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2*args.rank,
            lora_dropout=0.0, target_modules=targets, bias='none', task_type='CAUSAL_LM'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=args.lr, weight_decay=0.0)
        torch.cuda.reset_peak_memory_stats(args.device)
        model.eval()
        with (args.output/'groups.jsonl').open('x', buffering=1) as stream:
            for i, game in enumerate(run_games):
                state.update(stage='collecting', group_index=i)
                save(args.output/'state.json', state)
                episodes = []
                for j in range(args.group_size):
                    episode = collect(model, tokenizer, args, game['game'],
                                      *episode_seeds(args.seed,i,j))
                    episodes.append(episode)
                    with (args.output/f'group-{i:02d}-episode-{j}.json').open('x') as f:
                        json.dump(episode, f, ensure_ascii=False)
                state['stage'] = 'updating'
                save(args.output/'state.json', state)
                update = optimize_group(model, optimizer, episodes, args)
                state['groups'] += 1
                if update['updated']:
                    state['updates'] += 1
                    snapshot = args.output/'snapshots'/f'update-{state["updates"]:03d}'
                    snapshot.mkdir(parents=True)
                    model.save_pretrained(snapshot, safe_serialization=True)
                    tokenizer.save_pretrained(snapshot)
                    update['checkpoint'] = str(snapshot)
                    update['adapter_sha256'] = sha(snapshot/'adapter_model.safetensors')
                row = {'group': i, 'game': game, 'update': update,
                       'episode_summary': [{k:e[k] for k in ('won','reward','stop','steps','invalid')}
                                           for e in episodes],
                       'peak_allocated_bytes': torch.cuda.max_memory_allocated(args.device),
                       'peak_reserved_bytes': torch.cuda.max_memory_reserved(args.device),
                       'elapsed_seconds': time.time()-started}
                stream.write(json.dumps(row)+'\n')
                state.update(stage='collecting',
                    peak_allocated_bytes=row['peak_allocated_bytes'],
                    peak_reserved_bytes=row['peak_reserved_bytes'],
                    elapsed_seconds=row['elapsed_seconds'])
                save(args.output/'state.json', state)
                print(json.dumps(row), flush=True)
        state.update(stage='complete', complete=True, elapsed_seconds=time.time()-started)
        save(args.output/'state.json', state)
    except BaseException as exc:
        state.update(stage='failed', error=f'{type(exc).__name__}: {exc}',
                     elapsed_seconds=time.time()-started)
        save(args.output/'state.json', state)
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--environment-python', type=Path, required=True)
    p.add_argument('--train-root', type=Path, required=True)
    p.add_argument('--exclude-manifest', type=Path, action='append', default=[])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--selection-manifest', type=Path)
    p.add_argument('--max-groups', type=int, default=0)
    p.add_argument('--prompt-format', choices=('full','current_commands'), default='full')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--context-length', type=int, default=4096)
    p.add_argument('--max-new-tokens', type=int, default=32)
    p.add_argument('--max-train-tokens', type=int, default=1024)
    p.add_argument('--max-steps', type=int, default=40)
    p.add_argument('--group-size', type=int, default=2)
    p.add_argument('--games-per-type', type=int, default=1)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--beta', type=float, default=0.01)
    p.add_argument('--invalid-penalty-coef', type=float, default=0.1)
    p.add_argument('--invalid-only-advantage-scale', type=float, default=0.1)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--seed', type=int, default=20260923)
    run(p.parse_args())


if __name__ == '__main__':
    main()
