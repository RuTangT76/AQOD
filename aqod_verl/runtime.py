"""Ray driver assembly. Heavy dependencies load only for an explicit launch."""

import json
from pathlib import Path
import random

import numpy as np
import ray
import torch
from omegaconf import OmegaConf

from .contracts import assert_same_tokenizer, validate_recipe, verify_panel
from .environment import AQODEnvironmentManager


class ManifestDataset(torch.utils.data.Dataset):
    """Actual game identities, not ATOD's placeholder geometry data rows."""
    def __init__(self, games, seed):
        self.games, self.seed = games, seed

    def __len__(self):
        return len(self.games)

    def __getitem__(self, index):
        # verl-agent replaces this dummy token with the current public prompt.
        return {
            'input_ids': torch.zeros(1, dtype=torch.long),
            'attention_mask': torch.ones(1, dtype=torch.long),
            'position_ids': torch.zeros(1, dtype=torch.long),
            'raw_prompt_ids': np.asarray([0]),
            'raw_prompt': np.asarray([{'role': 'user', 'content': ''}], dtype=object),
            'data_source': 'aqod_alfworld_train',
            'env_kwargs': {'game': self.games[index]['game'], 'seed': self.seed},
        }


def run_driver(config):
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from verl.single_controller.ray import RayWorkerGroup
    from verl.utils import hf_tokenizer
    from verl.utils.dataset.rl_dataset import collate_fn
    from agent_system.multi_turn_rollout import TrajectoryCollector
    from agent_system.reward_manager import EpisodeRewardManager
    from trustlab.alfworld_bridge import EnvironmentClient
    from trustlab.alfworld_rollout import parse_command
    from trustlab.aqod_prompt import build_current_commands_prompt
    from .workers import AQODWorker
    from .trainer import AQODRayTrainer

    plain = OmegaConf.to_container(config, resolve=True)
    validate_recipe(plain)
    seed = config.aqod.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    aq = config.aqod
    train, dev = verify_panel(aq.panel.path, aq.panel.sha256, aq.train_root)
    student = hf_tokenizer(config.actor_rollout_ref.model.path)
    teacher = hf_tokenizer(config.actor_rollout_ref.ref.model.path)
    assert_same_tokenizer(student, teacher)

    def manager(split):
        log = Path(config.trainer.default_local_dir) / f'{split}_rollouts.jsonl'
        def emit(row):
            with log.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        return AQODEnvironmentManager(
            lambda: EnvironmentClient(aq.environment_python, aq.train_root),
            build_current_commands_prompt, parse_command, emit)

    worker = ray.remote(AQODWorker)
    roles = {Role.ActorRollout: worker, Role.RefPolicy: worker}
    pool = ResourcePoolManager(
        resource_pool_spec={'aqod': [config.trainer.n_gpus_per_node]},
        mapping={role: 'aqod' for role in roles})
    collector = TrajectoryCollector(config=config, tokenizer=student, processor=None)
    reward = EpisodeRewardManager(tokenizer=student, num_examine=0, normalize_by_length=False)
    trainer = AQODRayTrainer(
        config=config, tokenizer=student, role_worker_mapping=roles,
        resource_pool_manager=pool, ray_worker_group_cls=RayWorkerGroup,
        reward_fn=reward, val_reward_fn=reward,
        train_dataset=ManifestDataset(train, aq.seed),
        val_dataset=ManifestDataset(dev, aq.seed), collate_fn=collate_fn,
        traj_collector=collector, envs=manager('train'), val_envs=manager('development'),
        device_name='cuda')
    trainer.init_workers()
    trainer.fit()


def launch(config):
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    ray.init(runtime_env=get_ppo_ray_runtime_env(), **dict(config.ray_init))
    try:
        ray.get(ray.remote(num_cpus=1)(run_driver).remote(config))
    finally:
        ray.shutdown()
