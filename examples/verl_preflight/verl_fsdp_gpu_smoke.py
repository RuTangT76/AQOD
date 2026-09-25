"""Technical veRL FSDP LoRA update smoke; never use its weights as an AQOD model."""

import json
import math
import os

import torch
import torch.distributed as dist
from verl.trainer.config import CheckpointConfig
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


MODEL = os.environ.get("AQOD_SMOKE_MODEL", "/root/shared-nvme/models/Qwen3.5-2B")
SEED = int(os.environ.get("AQOD_SMOKE_SEED", "2026092512"))


def main():
    torch.manual_seed(SEED)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if world_size == 1:
        dist.init_process_group("nccl", init_method="tcp://127.0.0.1:29527", rank=0, world_size=1,
                                device_id=torch.device(f"cuda:{local_rank}"))
    else:
        dist.init_process_group("nccl", init_method="env://",
                                device_id=torch.device(f"cuda:{local_rank}"))
    try:
        model_config = HFModelConfig(
            path=MODEL, trust_remote_code=True, enable_gradient_checkpointing=False,
            use_remove_padding=False, lora_rank=4, lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            override_config={"attn_implementation": "sdpa"},
        )
        engine_config = FSDPEngineConfig(
            strategy="fsdp", dtype="bfloat16", model_dtype="bf16",
            use_remove_padding=False, use_dynamic_bsz=False,
            micro_batch_size_per_gpu=1, infer_micro_batch_size_per_gpu=1,
            use_torch_compile=False, fsdp_size=world_size, use_orig_params=True,
        )
        optimizer_config = FSDPOptimizerConfig(lr=1e-4, total_training_steps=1, clip_grad=1.0)
        engine = FSDPEngineWithLMHead(
            model_config, engine_config, optimizer_config, CheckpointConfig()
        )
        engine.initialize()
        token_ids = model_config.tokenizer("A short technical test.", return_tensors="pt")["input_ids"].to(local_rank)
        attention_mask = torch.ones_like(token_ids)
        engine.optimizer_zero_grad()
        with engine.train_mode():
            output = engine.module(input_ids=token_ids, attention_mask=attention_mask, use_cache=False)
            logits = output.logits[:, :-1].float()
            labels = token_ids[:, 1:]
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            assert torch.isfinite(loss)
            loss.backward()
            grad_values = [p.grad.float().norm().item() for p in engine.module.parameters() if p.grad is not None]
            print(json.dumps({"rank": rank, "stage": "before_optimizer", "loss": float(loss.detach()),
                              "grad_tensor_count": len(grad_values),
                              "max_grad_norm": max(grad_values, default=0.0)}), flush=True)
            grad_norm = engine.optimizer_step()
            print(json.dumps({"rank": rank, "stage": "after_optimizer", "grad_norm": grad_norm}), flush=True)
        assert math.isfinite(grad_norm) and grad_norm > 0
        print(json.dumps({
            "technical_smoke_only": True, "seed": SEED, "model": MODEL,
            "rank": rank, "world_size": world_size,
            "verl_fsdp_engine_step": 1, "loss": float(loss.detach()),
            "grad_norm": grad_norm,
            "peak_cuda_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        }), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
