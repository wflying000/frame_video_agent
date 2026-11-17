"""验证立体感要求的传递与验收门槛；模型和渲染替身不证明实际画质。"""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import model, planning, prompts, rendering, workflow
from agent.common import file_hash, read_json
from agent.store import Blocked, Project
from fixtures import REQ, STORY, FakeModel
import test_workflow as workflow_tests


class VisualDepthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'out'

    def fake_renderers(self):
        return workflow_tests.WorkflowTests().fake_renderers()

    def assert_depth_requirements(self, payload, expected):
        checks = {row['id']: row for row in payload['requirements']}
        self.assertTrue(expected <= checks.keys())
        self.assertTrue(all(checks[key]['hard'] is True for key in expected))
        self.assertEqual(payload['required_check_ids'], list(checks))

    @staticmethod
    def fail_depth(report, check_id, status='FAIL'):
        # 故意保留整体 PASS 和所有原有检查的 PASS，单独验证新增硬门槛。
        check = next(row for row in report['checks'] if row['requirement_id'] == check_id)
        check.update(result=status, evidence='主体仍是平涂纸片，缺少可辨认的体积与空间层次')
        return report

    def test_fixed_depth_checks_reach_plans_assets_frames_motion_and_encoded_video(self):
        fake = FakeModel()
        with patch.object(model, 'ask', side_effect=fake), self.fake_renderers(), \
                patch.object(rendering, 'sample_video',
                             side_effect=workflow_tests.WorkflowTests.sample_stub) as sample:
            result = workflow.generate(STORY, self.root, 4, 2)
        self.assertTrue(result.exists())
        by_target = {}
        for system, payload in fake.calls:
            if system == prompts.REVIEW:
                by_target.setdefault(payload['target_id'], []).append(payload)
        base = {'DEPTH_VOLUME', 'DEPTH_ILLUSTRATION'}
        scene = base | {'DEPTH_SPACE'}
        for target in ('plans/visual', 'plans/storyboard', 'plans/frames/S01', 'plans/frames/S02'):
            plans = [row for row in by_target[target] if row.get('visual_scope') == 'plan']
            self.assertTrue(plans, target)
            for payload in plans:
                self.assert_depth_requirements(payload, scene)
                self.assertEqual(payload['evidence_mode'], 'plan')
        reference = by_target['P01'][0]
        self.assertEqual(reference['visual_scope'], 'reference')
        self.assert_depth_requirements(reference, base | {'DEPTH_REFERENCE_SPACE'})
        for target in ('S01_K01', 'S01_K02', 'S02_K01', 'S02_K02'):
            payload = by_target[target][0]
            self.assertEqual(payload['visual_scope'], 'frame')
            self.assert_depth_requirements(payload, scene)
            self.assertEqual(payload['requirements'][0], REQ[0])
            self.assertEqual(payload['evidence_mode'], 'pixels')
        for shot_id in ('S01', 'S02'):
            temporal = next(row for row in by_target[shot_id] if 'next_shot_entry' in row)
            final = by_target[f'{shot_id}_appearance'][0]
            for payload in (temporal, final):
                self.assertEqual(payload['visual_scope'], 'temporal')
                self.assert_depth_requirements(payload, scene | {'DEPTH_STABILITY'})
                self.assertEqual(payload['evidence_mode'], 'pixels')
        # 成片审核必须来自编码后的候选视频，且后续镜头使用全局帧号。
        final_video = self.root / 'exports/final/v001/candidate_final.mp4'
        self.assertTrue(any(Path(call.args[0]).resolve() == final_video.resolve()
                            for call in sample.call_args_list))
        final = by_target['S02_appearance'][0]
        self.assertEqual([key['frame_index'] for key in final['frame_plan']['keyframes']], [4, 7])
        report = read_json(self.root / 'exports/final/v001/appearance_S02.json')
        for check in report['checks']:
            self.assertTrue(check['evidence_frames'])
            self.assertTrue(set(check['evidence_frames']) <= {4, 5, 6, 7})

    def test_nonpassing_depth_check_blocks_even_when_all_story_checks_pass(self):
        for status in ('FAIL', 'UNCERTAIN', 'NOT_APPLICABLE'):
            fake = FakeModel()
            root = self.root / status

            def ask(messages, **kwargs):
                report = fake(messages)
                if report.get('target_id') == 'S01_K01':
                    self.fail_depth(report, 'DEPTH_VOLUME', status)
                return report

            with self.subTest(status=status), patch.object(model, 'ask', side_effect=ask), self.fake_renderers():
                with self.assertRaises(Blocked):
                    workflow.generate(STORY, root, 4, 2)
            manifest = read_json(root / 'manifest.json')
            self.assertEqual(manifest['status'], 'BLOCKED')
            self.assertEqual(len(manifest['tasks']['segments/S01']['attempts']), 5)
            self.assertFalse((root / 'final.mp4').exists())
            self.assertTrue((root / 'segments/S01/v001/failure.json').exists())
            self.assertGreater(manifest['calls'], 0)
            self.assertLess(manifest['calls'], 120)

    def test_missing_depth_check_or_invalid_evidence_cannot_pass(self):
        for fault in ('missing', 'wrong_frame', 'comparison_frame', 'black_frame', 'single_temporal_frame'):
            root = self.root / fault
            fake = FakeModel()
            indices = [0, 1] if fault in ('comparison_frame', 'single_temporal_frame') else [0]
            if fault == 'black_frame':
                indices = [0, 1, 2]
            scope = 'temporal' if fault in ('black_frame', 'single_temporal_frame') else 'frame'
            payload = {'task': {'frame_index': 1 if fault == 'comparison_frame' else 0}}
            if fault == 'black_frame':
                payload['allowed_black_ranges'] = [[0, 1]]

            def ask(messages, **kwargs):
                report = fake(messages)
                if fault == 'missing':
                    report['checks'] = [row for row in report['checks']
                                        if row['requirement_id'] != 'DEPTH_VOLUME']
                else:
                    requirement = 'DEPTH_STABILITY' if fault == 'single_temporal_frame' else 'DEPTH_VOLUME'
                    next(row for row in report['checks']
                         if row['requirement_id'] == requirement)['evidence_frames'] = [99 if fault == 'wrong_frame' else 0]
                return report

            with self.subTest(fault=fault), Project(root, {'story': STORY}) as project:
                image = root / 'frame.png'
                image.write_bytes(b'frame-placeholder')
                directory = root / 'segments/S01/v001'
                directory.mkdir(parents=True)
                with patch.object(model, 'ask', side_effect=ask) as mocked:
                    with self.assertRaises(ValueError):
                        planning.review(project, directory, 'check_K01', 'S01_K01', 1, REQ,
                                        payload, [(f'实际画面 frame_index={i}', image) for i in indices],
                                        indices, visual_scope=scope)
                self.assertEqual(mocked.call_count, 2)
                self.assertFalse((root / 'final.mp4').exists())
                for _, payload in fake.calls:
                    self.assertIn('DEPTH_VOLUME', payload['required_check_ids'])

    def test_final_depth_failure_repairs_only_affected_shot_before_publish(self):
        fake = FakeModel()

        def ask(messages, **kwargs):
            report = fake(messages)
            if report.get('target_id') == 'S01_appearance' and report['target_version'] == 1:
                self.assertFalse((self.root / 'final.mp4').exists())
                self.fail_depth(report, 'DEPTH_SPACE')
                # 模型故意指错镜头；成片局部审核应绑定实际被审核的 S01。
                report['repair']['shot_ids'] = ['S02']
            return report

        with patch.object(model, 'ask', side_effect=ask), self.fake_renderers():
            result = workflow.generate(STORY, self.root, 4, 2)
        manifest = read_json(self.root / 'manifest.json')
        self.assertTrue(result.exists())
        self.assertEqual(manifest['status'], 'COMPLETED')
        self.assertEqual(manifest['global_repairs'], 1)
        self.assertEqual(len(manifest['tasks']['segments/S01']['attempts']), 2)
        self.assertEqual(len(manifest['tasks']['segments/S02']['attempts']), 1)
        self.assertEqual(len(manifest['tasks']['exports/final']['attempts']), 2)
        feedback = manifest['repair_requests']['S01']
        self.assertNotIn('S02', manifest['repair_requests'])
        self.assertEqual(read_json(self.root / 'exports/final/v001/appearance_S01.json')
                         ['repair']['shot_ids'], ['S02'])
        self.assertEqual(next(row['result'] for row in feedback['checks']
                              if row['requirement_id'] == 'DEPTH_SPACE'), 'FAIL')

    def test_invalid_depth_evidence_cannot_be_reused_by_a_later_candidate(self):
        for fault in ('black_frame', 'comparison_frame', 'single_temporal_frame'):
            root = self.root / fault
            fake = FakeModel()
            scope = 'frame' if fault == 'comparison_frame' else 'temporal'
            indices = [0, 1, 2] if fault == 'black_frame' else [0, 1]
            payload = {'task': {'frame_index': 1}} if fault == 'comparison_frame' else {}
            if fault == 'black_frame':
                payload['allowed_black_ranges'] = [[0, 1]]
            requirement = 'DEPTH_STABILITY' if fault == 'single_temporal_frame' else 'DEPTH_VOLUME'

            def invalid_response(messages, **kwargs):
                report = fake(messages)
                next(row for row in report['checks']
                     if row['requirement_id'] == requirement)['evidence_frames'] = [0]
                return report

            with self.subTest(fault=fault), Project(root, {'story': STORY}) as project:
                image = root / 'frame.png'
                image.write_bytes(b'unchanged-frame-placeholder')
                images = [(f'实际画面 frame_index={i}', image) for i in indices]
                first, second = root / 'segments/S01/v001', root / 'segments/S01/v002'
                first.mkdir(parents=True)
                second.mkdir(parents=True)
                with patch.object(model, 'ask', side_effect=invalid_response) as invalid:
                    with self.assertRaises(ValueError):
                        planning.review(project, first, 'review', 'S01', 1, REQ,
                                        payload, images, indices, visual_scope=scope)
                self.assertEqual(invalid.call_count, 2)
                # 模型的原始 PASS 必须保留，但证据非法，不能作为下一候选的复用来源。
                self.assertEqual(read_json(first / 'review.json')['status'], 'PASS')
                before = file_hash(first / 'review.json')
                with patch.object(model, 'ask', side_effect=FakeModel()) as valid:
                    result = planning.review(project, second, 'review', 'S01', 2, REQ,
                                             payload, images, indices, visual_scope=scope)
                self.assertEqual(valid.call_count, 1)
                self.assertNotIn('reused_from', result)
                self.assertEqual(result['status'], 'PASS')
                self.assertEqual(project.data['calls'], 3)
                self.assertEqual(file_hash(first / 'review.json'), before)

    def test_repeated_final_depth_failure_exhausts_bounded_repairs_without_publish(self):
        fake = FakeModel()

        def ask(messages, **kwargs):
            report = fake(messages)
            if report.get('target_id') == 'S02_appearance':
                self.fail_depth(report, 'DEPTH_SPACE')
            return report

        with patch.object(model, 'ask', side_effect=ask), self.fake_renderers():
            with self.assertRaises(Blocked):
                workflow.generate(STORY, self.root, 4, 2)
        manifest = read_json(self.root / 'manifest.json')
        self.assertEqual(manifest['status'], 'BLOCKED')
        self.assertEqual(manifest['global_repairs'], 2)
        self.assertEqual(len(manifest['tasks']['segments/S01']['attempts']), 1)
        self.assertEqual(len(manifest['tasks']['segments/S02']['attempts']), 3)
        self.assertEqual(len(manifest['tasks']['exports/final']['attempts']), 3)
        self.assertFalse((self.root / 'final.mp4').exists())
        self.assertGreater(manifest['calls'], 0)
        self.assertLess(manifest['calls'], 120)
        self.assertTrue((self.root / 'exports/final/v003/failure.json').exists())

    def test_declared_black_transition_is_exempt_without_fabricating_depth_evidence(self):
        fake = FakeModel()

        def ask(messages, **kwargs):
            result = fake(messages)
            if messages[0]['content'] == prompts.FRAME_PLAN and result['shot_id'] == 'S01':
                result['allowed_black_ranges'] = [[0, 1]]
                result['keyframes'][0].update(state='声明的首帧黑屏转场', covers=[],
                    requirements=[{'id': 'BLACK', 'hard': True, 'text': '本帧为声明的黑屏转场'}])
                result['intervals'] = [
                    {'start_frame': 0, 'end_frame': 1, 'method': 'cut', 'description': '黑屏转场'},
                    {'start_frame': 1, 'end_frame': 4, 'method': 'motion', 'description': '球向右移动'}]
            return result

        with patch.object(model, 'ask', side_effect=ask), self.fake_renderers():
            result = workflow.generate(STORY, self.root, 4, 2)
        self.assertTrue(result.exists())
        anchor = next(payload for _, payload in fake.calls if payload.get('target_id') == 'S01_K01')
        self.assertEqual(anchor['required_check_ids'], ['BLACK'])
        self.assertTrue(anchor['visual_depth_exemption'])
        for path in ('segments/S01/v001/temporal_review.json',
                     'exports/final/v001/appearance_S01.json'):
            report = read_json(self.root / path)
            for row in report['checks']:
                if row['requirement_id'].startswith('DEPTH_'):
                    self.assertTrue(row['evidence_frames'])
                    self.assertNotIn(0, row['evidence_frames'])
                    if row['requirement_id'] == 'DEPTH_STABILITY':
                        self.assertGreaterEqual(len(set(row['evidence_frames'])), 2)

    def test_new_depth_rules_invalidate_cache_without_erasing_history_or_budget(self):
        with patch.object(model, 'ask', side_effect=FakeModel()), self.fake_renderers():
            workflow.generate(STORY, self.root, 4, 2)
        before = read_json(self.root / 'manifest.json')
        report = self.root / 'segments/S01/v001/check_K01.json'
        report_hash = file_hash(report)
        final_hash = file_hash(self.root / 'final.mp4')
        with patch.object(model, 'ask', side_effect=AssertionError('unchanged rules should reuse')), self.fake_renderers():
            workflow.generate(STORY, self.root, 4, 2, resume=True)
        self.assertEqual(read_json(self.root / 'manifest.json')['calls'], before['calls'])

        changed = copy.deepcopy(prompts.DEPTH_CHECKS)
        changed['base']['DEPTH_VOLUME'] += ' 明确检查新规则中的局部体积阴影。'
        fake = FakeModel()
        with patch.object(prompts, 'DEPTH_CHECKS', changed), \
                patch.object(model, 'ask', side_effect=fake), self.fake_renderers():
            workflow.generate(STORY, self.root, 4, 2, resume=True)
        after = read_json(self.root / 'manifest.json')
        self.assertEqual(after['status'], 'COMPLETED')
        self.assertGreater(after['calls'], before['calls'])
        self.assertGreaterEqual(after['active_seconds'], before['active_seconds'])
        self.assertEqual(len(after['tasks']['segments/S01']['attempts']), 2)
        self.assertEqual(file_hash(report), report_hash)
        archived = list((self.root / 'exports').glob('previous_final_*.mp4'))
        self.assertTrue(archived)
        self.assertTrue(any(file_hash(path) == final_hash for path in archived))
        refreshed = next(payload for _, payload in fake.calls if payload.get('target_id') == 'S01_K01')
        self.assertIn('新规则', next(row['text'] for row in refreshed['requirements']
                                   if row['id'] == 'DEPTH_VOLUME'))


if __name__ == '__main__':
    unittest.main()
