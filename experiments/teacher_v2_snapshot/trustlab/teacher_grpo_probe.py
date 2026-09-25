"""One-update BF16 9B LoRA group-policy-gradient memory probe.

Synthetic +/- advantages exercise the optimizer only. This is not ALFWorld
teacher training, a capability result, or a checkpoint for student use.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from .alfworld_bridge import EnvironmentClient
from .alfworld_rollout import build_prompt
from .modeling import encode_prompt, load_base_model, text_lora_targets


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True))


def sequence_logp(model, prompt, completion, device):
    import torch
    ids = torch.tensor([prompt + completion], dtype=torch.long, device=device)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                   use_cache=False).logits[:, len(prompt)-1:-1].float()
    targets = ids[:, len(prompt):]
    return logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean()


def run(args):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer
    if args.output.exists():
        raise ValueError('Output exists; refusing to overwrite an experiment')
    args.output.mkdir(parents=True)
    started = time.time()
    manifest_path = args.preflight / 'manifest.json'
    completed_path = args.preflight / 'completed.json'
    manifest = json.loads(manifest_path.read_text())
    completed = json.loads(completed_path.read_text())
    if not completed.get('completed') or not all(g['replay_equal'] for g in completed['games']):
        raise ValueError('ALFWorld replay preflight has not passed')
    game = manifest['selection'][0]
    if sha(args.train_root / game['relative_path']) != game['sha256']:
        raise ValueError('Game hash mismatch')
    run_data = {'mode': 'teacher_grpo_memory_probe', 'scientific_mvp_passed': False,
                'synthetic_advantages': [1.0, -1.0],
                'note': 'Engineering probe only; no adapter checkpoint is saved',
                'arguments': {k: str(v) for k, v in vars(args).items()},
                'source_sha256': sha(__file__), 'manifest_sha256': sha(manifest_path),
                'model_config_sha256': sha(Path(args.model) / 'config.json'),
                'seed': args.seed, 'start_time': started}
    save(args.output / 'run.json', run_data)
    state = {'stage': 'starting', 'complete': False}
    save(args.output / 'state.json', state)
    try:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        with EnvironmentClient(args.environment_python, args.train_root) as env:
            reset = env.call('reset', game=game['relative_path'], seed=args.seed)
        history = [{'public': reset['public']}]
        prompt = encode_prompt(tokenizer, build_prompt(history), args.context_length,
                               args.max_new_tokens, enable_thinking=False)
        if args.probe_prompt_tokens > len(prompt):
            prompt = prompt + (prompt[-1:] * (args.probe_prompt_tokens - len(prompt)))
        save(args.output / 'public_state.json', {'game': game['relative_path'],
             'public': reset['public'], 'prompt_sha256': hashlib.sha256(bytes(str(prompt), 'utf8')).hexdigest()})
        state['stage'] = 'loading_model'
        save(args.output / 'state.json', state)
        model = load_base_model(args.model, args.device)
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
        state['stage'] = 'sampling'
        save(args.output / 'state.json', state)
        model.eval()
        candidates = []
        for i in range(2):
            torch.manual_seed(args.seed + i)
            with torch.inference_mode():
                out = model.generate(input_ids=torch.tensor([prompt], device=args.device),
                    do_sample=True, temperature=0.8, top_p=0.95,
                    max_new_tokens=args.max_new_tokens, use_cache=True,
                    pad_token_id=tokenizer.pad_token_id)
            completion = out[0, len(prompt):].tolist()
            if not completion:
                raise RuntimeError('Empty completion')
            candidates.append(completion)
        save(args.output / 'candidates.json', [{'text': tokenizer.decode(x, skip_special_tokens=True),
             'tokens': len(x)} for x in candidates])
        state['stage'] = 'one_update'
        save(args.output / 'state.json', state)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for advantage, completion in zip((1.0, -1.0), candidates):
            with torch.no_grad():
                old = sequence_logp(model, prompt, completion, args.device)
                with model.disable_adapter():
                    reference = sequence_logp(model, prompt, completion, args.device)
            current = sequence_logp(model, prompt, completion, args.device)
            ratio = (current - old).exp()
            clipped = ratio.clamp(0.8, 1.2)
            pg = -torch.minimum(ratio * advantage, clipped * advantage)
            log_ratio = reference - current
            kl = log_ratio.exp() - log_ratio - 1
            loss = (pg + args.beta * kl) / 2
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            losses.append(float(loss.detach()))
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError('Nonfinite gradient')
        optimizer.step()
        state.update(stage='complete', complete=True, losses=losses,
            grad_norm=float(grad_norm), prompt_tokens=len(prompt),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            elapsed_seconds=time.time()-started)
        save(args.output / 'state.json', state)
        print(json.dumps(state), flush=True)
    except BaseException as exc:
        state.update(stage='failed', error=f'{type(exc).__name__}: {exc}',
                     elapsed_seconds=time.time()-started)
        save(args.output / 'state.json', state)
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--environment-python', type=Path, required=True)
    p.add_argument('--train-root', type=Path, required=True)
    p.add_argument('--preflight', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--context-length', type=int, default=4096)
    p.add_argument('--max-new-tokens', type=int, default=32)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--probe-prompt-tokens', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--beta', type=float, default=0.01)
    p.add_argument('--seed', type=int, default=20260923)
    run(p.parse_args())


if __name__ == '__main__':
    main()
