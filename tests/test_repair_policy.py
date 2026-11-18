from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent import consistency, repair
from agent.common import write_json


class RepairPolicyTests(unittest.TestCase):
    @staticmethod
    def candidate(version, policy=None):
        candidate = {'version': version, 'path': f'segments/S01/v{version:03d}',
                     'report': {'target_id': 'S01_K01', 'checks': [],
                                'repair': {'instruction': f'feedback from v{version}'}}}
        if policy is not None:
            candidate['review_policy'] = consistency.review_policy(policy)
        return candidate

    def test_only_current_full_policy_becomes_segment_repair_history(self):
        attempts = [self.candidate(1), self.candidate(2, 'strict'), self.candidate(3, 'balanced'),
                    self.candidate(4, 'relaxed'), self.candidate(5, 'balanced')]
        attempts[4]['review_policy']['version'] = 'older-policy-version'
        original = deepcopy(attempts)
        project = SimpleNamespace(review_strictness='balanced',
                                  data={'tasks': {'segments/S01': {'attempts': attempts}}})
        self.assertEqual([row['version'] for row in repair.history(project, 'segments/S01', attempts[-1])], [3])
        self.assertEqual(attempts, original)
        project.review_strictness = 'strict'
        self.assertEqual([row['version'] for row in repair.history(project, 'segments/S01', attempts[-1])], [1, 2])
        project.review_strictness = 'relaxed'
        self.assertEqual([row['version'] for row in repair.history(project, 'segments/S01', attempts[-1])], [4])

    def test_reference_history_is_not_filtered_by_scene_review_policy(self):
        attempts = [self.candidate(1), self.candidate(2, 'strict')]
        project = SimpleNamespace(review_strictness='balanced',
                                  data={'tasks': {'assets/FISH': {'attempts': attempts}}})
        self.assertEqual([row['version'] for row in repair.history(project, 'assets/FISH', attempts[-1])], [1, 2])

    def test_policy_filter_runs_before_recent_history_limit(self):
        attempts = [self.candidate(i, 'balanced' if i <= 7 else 'strict') for i in range(1, 15)]
        project = SimpleNamespace(review_strictness='balanced',
                                  data={'tasks': {'segments/S01': {'attempts': attempts}}})
        self.assertEqual([row['version'] for row in repair.history(project, 'segments/S01', attempts[-1])],
                         [2, 3, 4, 5, 6, 7])

    def test_regression_reference_must_have_the_same_full_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempts = [self.candidate(1), self.candidate(2, 'balanced'),
                        self.candidate(3, 'balanced'), self.candidate(4, 'balanced')]
            attempts[2]['review_policy']['version'] = 'older-policy-version'
            for candidate in attempts:
                directory = root / candidate['path']
                directory.mkdir(parents=True)
                (directory / 'scene.html').write_text(f'drawBody({candidate["version"]});\n', encoding='utf-8')
                write_json(directory / 'check_K01.json', {'status': 'PASS'})
            project = SimpleNamespace(root=root, review_strictness='balanced',
                                      data={'tasks': {'segments/S01': {'attempts': attempts}}})
            current = root / attempts[-1]['path'] / 'scene.html'
            spec, plan = {'shot_id': 'S01'}, {'keyframes': [{'frame_id': 'K01', 'frame_index': 0}]}
            with patch.object(repair, 'passed_frame_evidence', return_value=[]):
                result = repair.regression_context(project, spec, plan, attempts[-1], current)
                self.assertEqual(result['last_passed_version'], 2)
                project.review_strictness = 'strict'
                result = repair.regression_context(project, spec, plan, attempts[-1], current)
                self.assertEqual(result['last_passed_version'], 1)
                project.review_strictness = 'relaxed'
                self.assertIsNone(repair.regression_context(project, spec, plan, attempts[-1], current))


if __name__ == '__main__':
    unittest.main()
