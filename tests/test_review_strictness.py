"""语义容差进入实际审核签名，不能改变原始检查项或技术证据门禁。"""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent import consistency, model, planning, rendering
from agent.common import digest, read_json, write_json
from agent.store import Rejected
from fixtures import FakeModel, REQ


class ReviewStrictnessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.image = self.root / 'actual.png'
        self.image.write_bytes(b'actual immutable frame pixels')
        self.images = [(f'实际画面 frame_index={i}', self.image) for i in (0, 1, 2)]

    def project(self, name='strict', mutate=None):
        fake = FakeModel()
        project = SimpleNamespace(root=self.root)
        if name is not None:
            project.review_strictness = name

        def call(directory, call_name, messages, extra_body=None):
            report = fake(messages)
            if mutate:
                mutate(report)
            write_json(directory / (call_name + '.json'), report)
            write_json(directory / (call_name + '.request.json'), model.recorded_messages(messages))
            write_json(directory / (call_name + '.call.json'), {
                'status': 'PASS', 'model': model.effective_settings(extra_body),
                'input_hash': digest({'messages': messages, 'settings': model.effective_settings(extra_body)})})
            return report

        project.call = call
        return project, fake

    def review(self, project, directory, version=1, **kwargs):
        return planning.review(project, directory, 'review', 'S01', version, REQ,
                               {'original_context': 'unchanged'}, self.images, [0, 1, 2], **kwargs)

    def test_policy_api_and_all_levels_reach_pixel_reviews_without_changing_requirements(self):
        self.assertEqual(consistency.REVIEW_LEVELS, ('strict', 'balanced', 'relaxed'))
        self.assertEqual(consistency.RENDER_MODES, ('exact', 'tolerant'))
        for task in ('segments/S01', 'checks/timeline', 'exports/final'):
            self.assertTrue(consistency.affects_review(task))
        for task in ('plans/story', 'plans/frames/S01', 'assets/P01', 'checks/other', 'segments', None):
            self.assertFalse(consistency.affects_review(task))
        for invalid in ('unknown', '', None, True):
            with self.assertRaises(ValueError):
                consistency.review_policy(invalid)
        expected = planning.depth_requirements(REQ, 'temporal')
        original = copy.deepcopy(REQ)
        for level in consistency.REVIEW_LEVELS:
            project, fake = self.project(level)
            directory = self.root / level / 'v001'
            self.assertEqual(self.review(project, directory, visual_scope='temporal')['status'], 'PASS')
            payload = fake.calls[0][1]
            self.assertEqual(payload['consistency_policy'], consistency.review_policy(level))
            self.assertEqual(payload['consistency_policy']['name'], level)
            self.assertEqual(payload['consistency_policy']['version'], 'consistency_v1')
            self.assertEqual(payload['requirements'], expected)
            self.assertEqual(payload['required_check_ids'], [row['id'] for row in expected])
            self.assertTrue(all(row['hard'] for row in payload['requirements']))
            self.assertEqual(payload['visual_depth_frame_indices'], [0, 1, 2])
            self.assertEqual(REQ, original)
        policy = consistency.review_policy('balanced'); policy['instruction'] = 'changed caller copy'
        self.assertNotEqual(policy, consistency.review_policy('balanced'))
        for level in ('balanced', 'relaxed'):
            text = consistency.review_policy(level)['instruction']
            for marker in ('原文', '关键动作', 'UNCERTAIN', '体积', '身份', '刚性平移', 'PASS'):
                self.assertIn(marker, text)

    def test_scene_cache_isolated_by_level_and_legacy_fixture_defaults_to_strict(self):
        project, fake = self.project(None)
        directory = self.root / 'segments/S01'
        first = self.review(project, directory / 'v001')
        self.assertEqual(fake.calls[-1][1]['consistency_policy']['name'], 'strict')
        self.assertNotIn('reused_from', first)
        results = []
        for version, level in enumerate(('balanced', 'relaxed', 'balanced', 'strict'), 2):
            project.review_strictness = level
            results.append(self.review(project, directory / f'v{version:03d}', version))
        self.assertEqual(len(fake.calls), 3)
        self.assertNotIn('reused_from', results[0])
        self.assertNotIn('reused_from', results[1])
        self.assertIn('reused_from', results[2])
        self.assertIn('reused_from', results[3])
        signatures = [read_json(directory / f'v{i:03d}/review.call.json')['input_hash'] for i in (1, 2, 3)]
        self.assertEqual(len(set(signatures)), 3)

    def test_plan_reference_observations_and_no_image_payloads_remain_identical(self):
        cases = {
            'plan': {'mode': 'plan', 'visual_scope': 'plan', 'images': self.images, 'frame_ids': None},
            'reference': {'visual_scope': 'reference', 'images': self.images, 'frame_ids': [0, 1, 2]},
            'observations': {'mode': 'observations', 'images': self.images, 'frame_ids': None},
            'no_image': {'mode': 'pixels', 'images': (), 'frame_ids': None},
        }
        for case, options in cases.items():
            requests = []
            for level in consistency.REVIEW_LEVELS:
                project, fake = self.project(level)
                folder = self.root / case / level / 'v001'
                planning.review(project, folder, 'review', 'target', 1, REQ,
                                {'original_context': 'untouched'}, **options)
                self.assertNotIn('consistency_policy', fake.calls[0][1])
                requests.append(read_json(folder / 'review.request.json'))
            self.assertEqual(requests[0], requests[1])
            self.assertEqual(requests[1], requests[2])

    def test_relaxed_semantics_never_bypass_failed_hard_checks_or_invalid_frame_evidence(self):
        for level in consistency.REVIEW_LEVELS:
            for fault in ('FAIL', 'UNCERTAIN', 'NOT_APPLICABLE', 'missing_check', 'wrong_frame'):
                def mutate(report):
                    if fault == 'missing_check':
                        report['checks'].pop()
                    elif fault == 'wrong_frame':
                        report['checks'][0]['evidence_frames'] = [99]
                    else:
                        report['checks'][0]['result'] = fault
                project, fake = self.project(level, mutate)
                folder = self.root / level / fault / 'v001'
                expected = ValueError if fault in ('missing_check', 'wrong_frame') else Rejected
                with self.subTest(level=level, fault=fault), self.assertRaises(expected):
                    self.review(project, folder, visual_scope='temporal')
                self.assertEqual(len(fake.calls), 2 if expected is ValueError else 1)
                self.assertTrue(all(payload['consistency_policy']['name'] == level for _, payload in fake.calls))

    def test_video_frame_count_and_timestamp_validation_is_independent_of_semantic_level(self):
        for level in consistency.REVIEW_LEVELS:
            consistency.review_policy(level)
            for fault in ('frame_count', 'timestamps'):
                frames = [{'best_effort_timestamp_time': value} for value in ('0', '.5')]
                if fault == 'frame_count':
                    frames.pop()
                else:
                    frames[-1]['best_effort_timestamp_time'] = '.6'
                metadata = {'streams': [{'width': 1280, 'height': 720, 'r_frame_rate': '2/1'}],
                            'frames': frames}
                with self.subTest(level=level, fault=fault), \
                        patch.object(rendering.subprocess, 'check_output', return_value=json.dumps(metadata).encode()), \
                        patch.object(rendering.subprocess, 'run') as decode, self.assertRaises(ValueError):
                    rendering.inspect_video(self.root / 'video.mp4', 2, 2)
                decode.assert_not_called()


if __name__ == '__main__':
    unittest.main()
