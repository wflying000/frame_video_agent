import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent import model, validation, workflow
from agent.common import read_json
from agent.store import Blocked, Project


class ObservationTests(unittest.TestCase):
    def test_typo_in_frame_key_is_retried_without_guessing_or_changing_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            with Project(Path(temporary)/'project', {'story': 'test'}) as project:
                image = project.root/'frame.png'
                image.write_bytes(b'actual pixels')
                directory = project.root/'exports/final/v001'
                bad = {'observations': [{'frame_index_index': 432, 'description': 'actual description'}]}
                good = {'observations': [{'frame_index': 432, 'description': 'actual description'}]}
                with patch.object(model, 'ask', side_effect=[bad, good]) as ask:
                    result = workflow.observe_batch(project, directory, 'observe_0', [(432, image)])
                self.assertEqual(result, good['observations'])
                self.assertEqual(project.data['calls'], 2)
                self.assertEqual(read_json(directory/'observe_0.json'), bad)
                self.assertEqual(read_json(directory/'observe_0.call.json')['status'], 'SCHEMA_ERROR')
                first, second = [call.args[0] for call in ask.call_args_list]
                self.assertEqual(first[1]['content'][1:], second[1]['content'][1:])
                context = json.loads(second[1]['content'][0]['text'])
                self.assertEqual(context['frame_indices'], [432])
                self.assertIn('observations[0].frame_index', context['format_error'])
                self.assertNotIn('story', context)

    def test_repeated_schema_failure_blocks_and_invalid_cache_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            with Project(Path(temporary)/'project', {'story': 'test'}) as project:
                image = project.root/'frame.png'
                image.write_bytes(b'actual pixels')
                directory = project.root/'exports/final/v001'
                with patch.object(model, 'ask', return_value={'observations': [{}]}) as ask:
                    with self.assertRaises(Blocked):
                        workflow.observe_batch(project, directory, 'observe_0', [(432, image)])
                    self.assertEqual(ask.call_count, 2)
                with patch.object(model, 'ask', return_value={'observations': [
                        {'frame_index': 432, 'description': 'visible object'}]}) as ask:
                    workflow.observe_batch(project, directory, 'observe_0', [(432, image)])
                    ask.assert_called_once()

    def test_coverage_and_integer_frame_ids_remain_strict(self):
        valid = {'observations': [{'frame_index': 0, 'description': 'a'},
                                  {'frame_index': 12, 'description': 'b'}]}
        for values in ([0, 0], [12, 0], [0, 13], [False, 12], ['0', 12]):
            data = copy.deepcopy(valid)
            for row, value in zip(data['observations'], values):
                row['frame_index'] = value
            with self.subTest(values=values), self.assertRaises(ValueError):
                validation.observations(data, [0, 12])


if __name__ == '__main__':
    unittest.main()
