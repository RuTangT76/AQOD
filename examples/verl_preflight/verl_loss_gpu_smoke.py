"""Technical veRL GRPO loss and Qwen3.5-2B LoRA update smoke; no AQOD outcome."""

import json
import random

import numpy as np
import torch
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, compute_policy_loss_vanilla


MODEL = "/root/shared-nvme/models/Qwen3.5-2B"
SEED = 2026092511


def token_logprobs(model, ids, prompt_len):
    logits = model(
        input_ids=ids[:, :-1], attention_mask=torch.ones_like(ids[:, :-1]), use_cache=False
    ).logits
    token_logits = logits[:, prompt_len - 1 :, :].float()
    return token_logits.log_softmax(-1).gather(-1, ids[:, prompt_len:].unsqueeze(-1)).squeeze(-1)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True
    )
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
    model.to("cuda:0")
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in name for name, _ in trainable)
    target = next(p for name, p in trainable if "lora_B" in name)
    before = target.detach().clone()
    prompt = tok("You are testing an optimizer. Reply briefly.", return_tensors="pt")["input_ids"].to("cuda:0")
    prompts = prompt.repeat(2, 1)
    model.eval()
    with torch.no_grad():
        ids = model.generate(
            input_ids=prompts, attention_mask=torch.ones_like(prompts),
            do_sample=True, temperature=1.0, top_p=1.0,
            max_new_tokens=4, min_new_tokens=4, pad_token_id=tok.pad_token_id,
        )
        old_logprob = token_logprobs(model, ids, prompts.shape[1]).detach()
    assert old_logprob.shape == (2, 4)
    # Synthetic contrast verifies the optimizer path only. It is never used for research selection.
    rewards = torch.zeros_like(old_logprob)
    rewards[0, -1], rewards[1, -1] = 1.0, -1.0
    mask = torch.ones_like(old_logprob)
    advantage, _ = compute_grpo_outcome_advantage(
        rewards, mask, np.array(["smoke", "smoke"]), norm_adv_by_std_in_grpo=False
    )
    model.train()
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=1e-4)
    logprob = token_logprobs(model, ids, prompts.shape[1])
    loss, metrics = compute_policy_loss_vanilla(
        old_logprob, logprob, advantage, mask,
        config=OmegaConf.create({
            "clip_ratio": 0.2, "clip_ratio_low": None, "clip_ratio_high": None,
            "clip_ratio_c": 3.0, "global_batch_info": {},
        }),
    )
    assert torch.isfinite(loss)
    loss.backward()
    grad_norm = torch.linalg.vector_norm(torch.stack([
        p.grad.float().norm() for _, p in trainable if p.grad is not None
    ]))
    assert torch.isfinite(grad_norm) and grad_norm > 0
    assert target.grad is not None and target.grad.float().norm() > 0
    optimizer.step()
    delta = (target.detach() - before).abs().max()
    assert torch.isfinite(delta) and delta > 0
    print(json.dumps({
        "technical_smoke_only": True, "seed": SEED, "model": MODEL,
        "verl_grpo_and_ppo_loss": True, "optimizer_step": 1,
        "loss": float(loss.detach()), "grad_norm": float(grad_norm),
        "gradient_bearing_lora_max_delta": float(delta), "policy_metrics": metrics,
        "peak_cuda_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
    }), flush=True)


if __name__ == "__main__":
    main()
