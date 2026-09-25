"""AQOD Gate1 trainer on the same Ray/FSDP/vLLM roles as ATOD.

Only framework services are inherited. No ATOD/SOD trainer or objective is
imported. The prerequisite gates still determine whether training may start.
"""

import json
from pathlib import Path
import numpy as np
import torch
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.metric import reduce_metrics

from .objective import sampled_opd_signal


class AQODRayTrainer(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Upstream otherwise uses the student's base model whenever LoRA is on.
        self.ref_in_actor = False
        if not self.use_reference_policy or self.use_critic:
            raise ValueError('Gate1 requires an external teacher and no critic')
        self.log_path = Path(self.config.trainer.default_local_dir) / 'metrics.jsonl'

    def log(self, event, **values):
        row = {'event': event, 'global_step': self.global_steps, **values}
        with self.log_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)

    def train_batch(self, batch):
        """Score the *same* generated tokens with student and frozen teacher."""
        width = batch.batch['responses'].shape[1]
        mask = batch.batch['attention_mask'][:, -width:].clone()
        if 'loss_mask' in batch.batch:
            mask = mask * batch.batch['loss_mask'][:, -width:]
        batch.batch['response_mask'] = mask
        padded, n_pad = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)
        old = self.actor_rollout_wg.compute_log_prob(padded)
        teacher = self.ref_policy_wg.compute_ref_log_prob(padded)
        old = unpad_dataproto(old, n_pad)
        teacher = unpad_dataproto(teacher, n_pad)
        student_lp = old.batch['old_log_probs'].detach().float().cpu()
        teacher_lp = teacher.batch['ref_log_prob'].detach().float().cpu()
        signal = sampled_opd_signal(student_lp.numpy(), teacher_lp.numpy(), mask.cpu().numpy())
        batch.batch['old_log_probs'] = student_lp
        batch.batch['advantages'] = torch.from_numpy(signal)
        batch.batch['returns'] = batch.batch['advantages'].clone()
        # Preserve all real turns. Upstream adjust_batch can duplicate/drop
        # turns; here scheduling padding carries zero policy-gradient signal.
        size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        training, extra = pad_dataproto_to_divisor(batch, size)
        if extra:
            for name in ('response_mask', 'advantages', 'returns'):
                training.batch[name][-extra:] = 0
            # Keep the copied attention mask nonempty. The upstream actor
            # derives its reduction mask from attention_mask; clearing it can
            # cause a 0/0 masked mean on a padding-only rank. Zero advantages
            # yield zero gradients because entropy/KL/auxiliary losses are off.
        training.meta_info['global_token_num'] = training.batch['attention_mask'].sum(-1).tolist()
        training.meta_info['temperature'] = self.config.actor_rollout_ref.rollout.temperature
        training.meta_info['multi_turn'] = False  # each collector row is one turn
        output = self.actor_rollout_wg.update_actor(training)
        metrics = reduce_metrics(output.meta_info['metrics'])
        active = mask.cpu().numpy().astype(bool)
        metrics.update({
            'opd/teacher_student_logp_gap': float(signal[active].mean()),
            'opd/teacher_scored_tokens': int(active.sum()),
            'opd/turns': len(batch), 'opd/padding_turns': extra,
            'cost/train_environment_steps': self.envs.environment_steps,
            'cost/train_episodes': self.envs.reset_count,
        })
        for key in ('success_rate',):
            if key in batch.non_tensor_batch:
                metrics[f'rollout/{key}'] = float(np.mean(batch.non_tensor_batch[key]))
        return metrics

    def _save_checkpoint(self):
        super()._save_checkpoint()
        path = Path(self.config.trainer.default_local_dir) / f'global_step_{self.global_steps}'
        state = {'global_step': self.global_steps,
                 'train_environment_steps': self.envs.environment_steps,
                 'train_episodes': self.envs.reset_count,
                 'development_environment_steps': self.val_envs.environment_steps,
                 'development_episodes': self.val_envs.reset_count}
        temporary = path / 'aqod_state.json.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
        temporary.replace(path / 'aqod_state.json')

    def _load_checkpoint(self):
        super()._load_checkpoint()
        if self.config.trainer.resume_mode == 'resume_path':
            path = Path(self.config.trainer.resume_from_path) / 'aqod_state.json'
            state = json.loads(path.read_text(encoding='utf-8'))
            if state['global_step'] != self.global_steps:
                raise ValueError('AQOD cost state and model checkpoint differ')
            for env, prefix in ((self.envs, 'train'), (self.val_envs, 'development')):
                env.environment_steps = state[f'{prefix}_environment_steps']
                env.reset_count = state[f'{prefix}_episodes']

    def fit(self):
        self.global_steps = 0
        self._load_checkpoint()
        try:
            if self.config.trainer.val_before_train:
                self.log('validation', metrics=self._validate())
            while self.global_steps < self.total_training_steps:
                for raw in self.train_dataloader:
                    if self.global_steps >= self.total_training_steps:
                        break
                    gen = DataProto.from_single_dict(raw)
                    gen.meta_info.update({
                        'eos_token_id': self.tokenizer.eos_token_id,
                        'pad_token_id': self.tokenizer.pad_token_id,
                        'do_sample': True, 'validate': False,
                    })
                    batch = self.traj_collector.multi_turn_loop(
                        gen_batch=gen, actor_rollout_wg=self.actor_rollout_wg,
                        envs=self.envs, is_train=True)
                    metrics = self.train_batch(batch)
                    self.global_steps += 1
                    self.log('student_update', metrics=metrics)
                    last = self.global_steps == self.total_training_steps
                    if last or self.global_steps % self.config.trainer.test_freq == 0:
                        self.log('validation', metrics=self._validate())
                    if last or self.global_steps % self.config.trainer.save_freq == 0:
                        self._save_checkpoint()
            self.log('complete', scientific_mvp_passed=False)
        except BaseException as exc:
            self.log('failed', error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            self.envs.close()
            self.val_envs.close()
