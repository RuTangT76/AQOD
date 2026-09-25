import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from trustlab.teacher_alfworld_grpo import select_games


class TestTeacherSelection(unittest.TestCase):
    def test_both_manifest_schemas_exclude_games(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for n in (1,2,3):
                p=root/f'simple-{n}'/'trial_a'/'game.tw-pddl'
                p.parent.mkdir(parents=True)
                p.write_text(str(n))
            a=root/'prior_selection.json'
            b=root/'prior_selected.json'
            a.write_text(json.dumps({'selection':[{'relative_path':'simple-1/trial_a/game.tw-pddl'}]}))
            b.write_text(json.dumps({'selected':[{'game':'simple-2/trial_a/game.tw-pddl'}]}))
            with patch('trustlab.alfworld_preflight.TASKS',('simple',)):
                chosen=select_games(root,[a,b],42,1)
            self.assertEqual(chosen[0]['game'],'simple-3/trial_a/game.tw-pddl')

    def test_unknown_schema_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            manifest=root/'unknown.json'
            manifest.write_text('{}')
            with patch('trustlab.alfworld_preflight.TASKS',('simple',)):
                with self.assertRaisesRegex(ValueError,'schema'):
                    select_games(root,[manifest],42,1)
