"""Read-only linkage audit from V4 completed training to frozen Gate T teacher."""
import argparse
import hashlib
import json
from pathlib import Path


PANEL_SHA = '8970df8a9692a57d2f48206efb8d83ae2c992a9d4382f6127d90f9a39e71cea5'
TRAINER_SHA = 'b0eb95aef682e311cf258034ca5e4a738b89cacce9862584a3b51e03ac49d8f4'
PROMPT_SHA = 'c49bd3e20908c2830d5147462ed9647583d71eb8f958af4cbf384c09b1e70884'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--rl', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite audit')
    require(digest(args.panel) == PANEL_SHA, 'Frozen panel changed')
    panel = load(args.panel)
    manifest = load(args.training / 'manifest.json')
    state = load(args.training / 'state.json')
    frozen = load(args.training / 'frozen_checkpoint.json')
    evaluation = load(args.rl / 'run.json')
    eval_state = load(args.rl / 'state.json')
    require(state.get('complete') is True and state.get('stage') == 'complete' and
            state.get('groups') == 180 and state.get('updates', 0) >= 1,
            'Training did not complete 180 groups')
    require(manifest.get('source_sha256') == TRAINER_SHA and
            manifest.get('prompt_source_sha256') == PROMPT_SHA and
            manifest.get('selection_manifest_sha256') == PANEL_SHA and
            manifest.get('run_selection') == panel['train'],
            'Training code/prompt/selection mismatch')
    expected = (args.training.resolve() / 'snapshots' /
                f"update-{state['updates']:03d}")
    checkpoint = Path(frozen['checkpoint']).resolve()
    require(checkpoint == expected and frozen.get('all_parameters_finite') is True,
            'Final checkpoint was not the validated last update')
    weights = checkpoint / 'adapter_model.safetensors'
    require(weights.is_file(), 'Adapter weights missing')
    adapter_sha = digest(weights)
    require(adapter_sha == frozen.get('adapter_sha256'),
            'Frozen checkpoint weight hash mismatch')
    require(eval_state.get('complete') is True and
            evaluation.get('role') == 'teacher_rl' and
            evaluation.get('adapter_sha256') == adapter_sha and
            Path(evaluation['adapter']).resolve() == checkpoint and
            evaluation.get('selection_manifest_sha256') == PANEL_SHA,
            'RL Gate T evaluation did not use the frozen final adapter/panel')
    output = {'training_state_sha256': digest(args.training / 'state.json'),
              'training_manifest_sha256': digest(args.training / 'manifest.json'),
              'frozen_checkpoint_sha256': digest(args.training / 'frozen_checkpoint.json'),
              'rl_run_sha256': digest(args.rl / 'run.json'),
              'rl_state_sha256': digest(args.rl / 'state.json'),
              'groups': state['groups'], 'updates': state['updates'],
              'adapter_sha256': adapter_sha, 'last_update_finite': True,
              'linkage_passed': True}
    args.output.mkdir(parents=True)
    (args.output / 'audit.json').write_text(json.dumps(output, indent=2) + '\n',
                                            encoding='utf-8')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
