"""Preflight or explicitly launch the local AQOD veRL adapter.

python -m aqod_verl.main --config recipe.json --check
python -m aqod_verl.main --config recipe.json --upstream /path/to/ATOD --run
"""

import argparse
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

from .contracts import read_json, sha256, validate_recipe, verify_gates, verify_panel


def preflight(recipe, upstream):
    report = {'ready': False, 'checks': {}, 'errors': []}
    for name in ('torch', 'ray', 'omegaconf', 'hydra', 'vllm', 'verl', 'agent_system', 'trustlab'):
        report['checks'][name] = importlib.util.find_spec(name) is not None
    missing = [name for name, present in report['checks'].items() if not present]
    if missing:
        report['errors'].append('Missing runtime modules: ' + ', '.join(missing))
    if platform.system() != 'Linux':
        report['errors'].append('GPU runtime requires Linux; CPU contract tests work here')
    if report['checks']['torch']:
        import torch
        if torch.cuda.device_count() < recipe['trainer']['n_gpus_per_node']:
            report['errors'].append('Insufficient visible CUDA GPUs for this recipe')
    if upstream is None or not (upstream / 'verl/trainer/config/ppo_trainer.yaml').is_file():
        report['errors'].append('Provide a local ATOD checkout with --upstream')
    try:
        validate_recipe(recipe)
        aq = recipe['aqod']
        verify_panel(aq['panel']['path'], aq['panel']['sha256'], aq['train_root'])
        evidence = verify_gates(aq)
        teacher = Path(recipe['actor_rollout_ref']['ref']['model']['path']).resolve(strict=True)
        weights = evidence['export_files']
        if not weights or not any(name.endswith('.safetensors') for name in weights):
            raise ValueError('Teacher export audit must identify model weight shards')
        actual_shards = {file.name for file in teacher.glob('*.safetensors')}
        if actual_shards != {name for name in weights if name.endswith('.safetensors')}:
            raise ValueError('Teacher export audit must cover every model shard')
        for name, expected in weights.items():
            file = (teacher / name).resolve(strict=True)
            if not file.is_relative_to(teacher) or sha256(file) != expected:
                raise ValueError('Teacher export file hash differs: ' + name)
        if upstream:
            spec = importlib.util.find_spec('verl')
            if spec and not Path(spec.origin).resolve().is_relative_to(upstream.resolve()):
                raise ValueError('Installed verl is not the supplied ATOD checkout')
    except (ValueError, KeyError, TypeError, OSError) as exc:
        report['errors'].append(str(exc))
    report['ready'] = not report['errors']
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--upstream', type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check', action='store_true')
    action.add_argument('--run', action='store_true')
    args = parser.parse_args()
    recipe = read_json(args.config)
    report = preflight(recipe, args.upstream)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report['ready']:
        return 2
    if args.check:
        return 0
    from omegaconf import OmegaConf
    config = OmegaConf.merge(
        OmegaConf.load(args.upstream / 'verl/trainer/config/ppo_trainer.yaml'),
        OmegaConf.create(recipe))
    plain = OmegaConf.to_container(config, resolve=True)
    validate_recipe(plain)
    output = Path(config.trainer.default_local_dir)
    sources = {
        str(p.relative_to(args.upstream)): sha256(p)
        for name in ('verl', 'agent_system') for p in (args.upstream / name).rglob('*.py')}
    own_sources = {str(p.relative_to(Path(__file__).parent)): sha256(p)
                   for p in Path(__file__).parent.rglob('*.py')}
    own_sources['../aqod_online_trust.py'] = sha256(Path(__file__).parent.parent / 'aqod_online_trust.py')
    import trustlab.alfworld_bridge as bridge
    import trustlab.alfworld_rollout as rollout
    import trustlab.aqod_prompt as prompt
    own_sources.update({f'trustlab/{m.__name__}': sha256(m.__file__)
                        for m in (bridge, rollout, prompt)})
    versions = {name: importlib.metadata.version(name)
                for name in ('torch', 'ray', 'vllm', 'transformers', 'peft', 'hydra-core')}
    provenance = {'config': plain, 'framework_sources': sources, 'aqod_sources': own_sources,
                  'versions': versions, 'command': sys.argv,
                  'scientific_mvp_passed': False}
    if config.trainer.resume_mode == 'disable':
        output.mkdir(parents=True, exist_ok=False)
        (output / 'run.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    else:
        previous = read_json(output / 'run.json')
        def identity(c):
            copy = json.loads(json.dumps(c))
            copy['trainer'].pop('resume_mode', None)
            copy['trainer'].pop('resume_from_path', None)
            return copy
        if identity(previous['config']) != identity(plain) or \
                previous['framework_sources'] != sources or previous['aqod_sources'] != own_sources or \
                previous['versions'] != versions:
            raise ValueError('Resume requires identical configuration, source and runtime versions')
        ckpt = Path(config.trainer.resume_from_path).resolve(strict=True)
        if ckpt.parent != output.resolve() or not (ckpt / 'aqod_state.json').is_file() or \
                not (ckpt / 'data.pt').is_file() or not (ckpt / 'actor').is_dir():
            raise ValueError('Resume must use a complete checkpoint from this output directory')
    try:
        from .runtime import launch
        launch(config)
    except BaseException as exc:
        with (output / 'failures.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'type': type(exc).__name__, 'message': str(exc),
                                     'command': sys.argv}) + '\n')
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
