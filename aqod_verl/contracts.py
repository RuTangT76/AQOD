"""CPU-only contracts shared by the launcher, environment and trust adapter."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def teacher_worker_config(config, role):
    """Do not let the inherited LoRA-reference shortcut select the student."""
    result = deepcopy(config)
    if role == 'ref':
        teacher = result['ref']['model']['path']
        if not teacher or teacher == result['model']['path']:
            raise ValueError('An independent frozen teacher path is required')
        result['model']['lora_rank'] = 0
        result['model']['path'] = teacher
    return result


def assert_same_tokenizer(student, teacher):
    """Sampled-token OPD needs the same token IDs, not just equal vocab size."""
    if student.get_vocab() != teacher.get_vocab():
        raise ValueError('Teacher/student vocabulary IDs differ; token OPD is invalid')
    for name in ('bos_token_id', 'eos_token_id', 'pad_token_id'):
        if getattr(student, name) != getattr(teacher, name):
            raise ValueError(f'Teacher/student {name} differs')
    if student.chat_template != teacher.chat_template:
        raise ValueError('Teacher/student chat templates differ')


def verify_panel(path, expected_sha, root):
    if sha256(path) != expected_sha:
        raise ValueError('Frozen Gate1 panel hash differs')
    panel = read_json(path)
    if panel.get('teacher_or_student_outcomes_used_for_selection') is not False:
        raise ValueError('Panel must be selected independently of outcomes')
    root = Path(root).resolve(strict=True)
    if root.name != 'train':
        raise ValueError('Gate1 requires the official train split')
    groups = [panel['gate1_train'], panel['snapshot_dev']]
    seen = set()
    for group in groups:
        if not group:
            raise ValueError('Training and development panels must be nonempty')
        for row in group:
            file = (root / row['game']).resolve(strict=True)
            if not file.is_relative_to(root) or file in seen:
                raise ValueError('Escaping, duplicate, or overlapping game path')
            if sha256(file) != row['sha256']:
                raise ValueError(f'Game hash differs: {row["game"]}')
            seen.add(file)
    return groups


def validate_recipe(config):
    """Validate the supported migration stage before loading heavyweight code.

    Gate1 is deliberately pure OPD. The online controller remains a reusable
    component until Gates 2-4 define the actual exploration training protocol.
    """
    aq = config['aqod']
    if aq['stage'] != 'gate1_opd':
        raise ValueError('Only Gate1 OPD is wired to GPU training in this migration')
    ar = config['actor_rollout_ref']
    if ar['model']['path'] != aq['vanilla_student_model']:
        raise ValueError('Student initialization must be the declared vanilla model')
    if ar['model']['lora_rank'] <= 0:
        raise ValueError('Gate1 updates student LoRA only')
    if ar['actor']['strategy'] != 'fsdp' or ar['rollout']['name'] != 'vllm':
        raise ValueError('This adapter supports FSDP + vLLM')
    if ar['rollout']['mode'] != 'sync':
        raise ValueError('The on-policy collector requires synchronous rollout')
    if ar['rollout']['n'] != 1 or config['env']['rollout']['n'] != 1:
        raise ValueError('Gate1 uses one unassisted student rollout per game')
    if ar['actor']['use_kl_loss'] or config['algorithm']['use_kl_in_reward']:
        raise ValueError('Gate1 has only the teacher OPD training signal')
    if ar['actor']['entropy_coeff'] != 0 or ar['actor']['use_invalid_action_penalty']:
        raise ValueError('No entropy or task reward auxiliary objective in Gate1')
    if any(ar['actor'].get(name, False) for name in ('use_sdl_loss', 'use_sdar_loss')):
        raise ValueError('No inherited self-distillation auxiliary losses in Gate1')
    if config['algorithm'].get('atod') or config['algorithm'].get('sod'):
        raise ValueError('ATOD/SOD algorithm settings cannot enter the AQOD recipe')
    if config['algorithm']['filter_groups']['enable']:
        raise ValueError('Outcome-based rollout filtering changes the protocol')
    if config['data']['truncation'] != 'error':
        raise ValueError('Public history must not be silently truncated')
    if config['trainer']['resume_mode'] not in ('disable', 'resume_path'):
        raise ValueError('Resume must identify an explicit AQOD checkpoint')
    if config['trainer']['nnodes'] != 1:
        raise ValueError('The initial adapter supports one server')
    for name in ('total_training_steps', 'save_freq', 'test_freq', 'n_gpus_per_node'):
        if type(config['trainer'][name]) is not int or config['trainer'][name] <= 0:
            raise ValueError(f'Positive trainer.{name} required')
    if ar['actor']['loss_agg_mode'] != 'token-mean' or ar['actor']['ppo_epochs'] != 1:
        raise ValueError('Gate1 uses one pass and token-mean reduction')
    if ar['rollout']['temperature'] != 1.0 or ar['rollout']['top_p'] != 1.0:
        raise ValueError('Sampled OPD requires untruncated on-policy sampling at temperature 1')
    if ar['rollout'].get('top_k', -1) not in (-1, 0) or ar['rollout'].get('use_fire_sampling', False):
        raise ValueError('Additional rollout sampling transforms are not supported')
    if config['data']['seed'] != aq['seed'] or ar['rollout']['seed'] != aq['seed']:
        raise ValueError('Data and rollout seeds must match the logged run seed')
    n = config['trainer']['n_gpus_per_node']
    if config['data']['train_batch_size'] % n or ar['actor']['ppo_mini_batch_size'] % n:
        raise ValueError('Training and optimizer batches must be divisible by GPU count')
    teacher_worker_config(ar, 'ref')


def verify_gates(aq):
    """Require immutable evidence; this does not create or change gate results."""
    for name, flag in (('gate_t', 'gate_t_passed'), ('gate0', 'gate0_passed')):
        evidence = aq[name]
        if sha256(evidence['path']) != evidence['sha256']:
            raise ValueError(f'{name} evidence hash differs')
        if read_json(evidence['path']).get(flag) is not True:
            raise ValueError(f'{name} has not passed; student training is blocked')
    # A merged teacher export must carry an audited linkage to the qualified
    # base+adapter. Do not silently use an arbitrary HF directory as teacher.
    linkage = aq['teacher_export_audit']
    if sha256(linkage['path']) != linkage['sha256']:
        raise ValueError('Teacher export audit hash differs')
    doc = read_json(linkage['path'])
    if doc.get('equivalence_passed') is not True:
        raise ValueError('Merged teacher equivalence has not been verified')
    if doc.get('gate_t_sha256') != aq['gate_t']['sha256'] or \
            doc.get('gate0_sha256') != aq['gate0']['sha256']:
        raise ValueError('Teacher export is linked to different gate evidence')
    return doc
