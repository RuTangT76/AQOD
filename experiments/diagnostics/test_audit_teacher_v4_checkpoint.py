"""Synthetic final-adapter linkage check for V4; no model or game outcomes used."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


HERE = Path(__file__).parent
PANEL = HERE / 'aqod_teacher_v4_budget_selection.json'


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def main():
    panel = json.loads(PANEL.read_text(encoding='utf-8'))
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        training = root / 'training'
        evaluation = root / 'rl'
        checkpoint = training / 'snapshots' / 'update-001'
        checkpoint.mkdir(parents=True)
        evaluation.mkdir()
        weights = checkpoint / 'adapter_model.safetensors'
        weights.write_bytes(b'synthetic finite-adapter placeholder')
        adapter_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
        write(training / 'manifest.json', {
            'source_sha256': 'b0eb95aef682e311cf258034ca5e4a738b89cacce9862584a3b51e03ac49d8f4',
            'prompt_source_sha256': 'c49bd3e20908c2830d5147462ed9647583d71eb8f958af4cbf384c09b1e70884',
            'selection_manifest_sha256': '8970df8a9692a57d2f48206efb8d83ae2c992a9d4382f6127d90f9a39e71cea5',
            'run_selection': panel['train']})
        write(training / 'state.json', {'complete': True, 'stage': 'complete',
                                        'groups': 180, 'updates': 1})
        write(training / 'frozen_checkpoint.json', {
            'checkpoint': str(checkpoint), 'all_parameters_finite': True,
            'adapter_sha256': adapter_hash})
        write(evaluation / 'run.json', {
            'role': 'teacher_rl', 'adapter': str(checkpoint),
            'adapter_sha256': adapter_hash,
            'selection_manifest_sha256': '8970df8a9692a57d2f48206efb8d83ae2c992a9d4382f6127d90f9a39e71cea5'})
        write(evaluation / 'state.json', {'complete': True})

        def invoke(name):
            return subprocess.run([sys.executable, str(HERE / 'audit_teacher_v4_checkpoint.py'),
                '--training', str(training), '--rl', str(evaluation),
                '--panel', str(PANEL), '--output', str(root / name)],
                capture_output=True, text=True)

        valid = invoke('valid')
        assert valid.returncode == 0, valid.stderr
        result = json.loads((root / 'valid' / 'audit.json').read_text())
        assert result['linkage_passed'] and result['adapter_sha256'] == adapter_hash
        run = json.loads((evaluation / 'run.json').read_text())
        run['adapter_sha256'] = 'wrong-checkpoint'
        write(evaluation / 'run.json', run)
        invalid = invoke('invalid')
        assert invalid.returncode != 0 and 'frozen final adapter' in invalid.stderr
        print('V4 final-adapter linkage accepts exact match and rejects mismatch')


if __name__ == '__main__':
    main()
