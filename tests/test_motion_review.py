"""比较方向审核：不能把跨帧位移误判为终态方位，也不能据此自动放行。"""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent import model, planning, repair
from agent.common import read_json, write_json
from agent.store import Project, Rejected


MOTION = [{'id': 'MOTION', 'hard': True,
           'text': '相对K01，链条尾由左后上摆改为右下摆，身体保持游动波浪。'}]


class MotionReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'project'

    def test_motion_policy_is_not_added_to_static_location_requirement(self):
        self.assertIsNone(repair.motion_comparison_policy([
            {'id': 'LOCATION', 'text': '尾端位于躯干右下方。'}]))
        self.assertEqual(repair.motion_comparison_policy(MOTION)['requirement_ids'], ['MOTION'])

    def test_direction_review_keeps_requirements_and_can_still_reject(self):
        original = copy.deepcopy(MOTION)
        with Project(self.root, {'story': '游动'}) as project:
            first, second = self.root / 'first.png', self.root / 'second.png'
            first.write_bytes(b'first actual pixels')
            second.write_bytes(b'second actual pixels')
            for status in ('FAIL', 'UNCERTAIN'):
                report = {'target_id': 'S01_K02', 'target_version': 1, 'status': status,
                          'checks': [{'requirement_id': 'MOTION', 'result': status,
                                      'evidence': '身体波浪不清楚', 'evidence_frames': [0, 12]}]}
                with self.subTest(status=status), patch.object(model, 'ask', return_value=report) as ask:
                    with self.assertRaises(Rejected):
                        planning.review(project, self.root / status, 'check_K02', 'S01_K02', 1,
                            MOTION, {'task': {'frame_index': 12}},
                            [('实际画面 frame_index=0', first), ('实际画面 frame_index=12', second)], [0, 12])
                    ask.assert_called_once()
                    context = json.loads(ask.call_args.args[0][1]['content'][0]['text'])
                    self.assertEqual(context['requirements'], original)
                    scope = context['pixel_scope']
                    self.assertEqual(scope['comparison_frame_indices'], [0])
                    instruction = scope['motion_comparison']['instruction']
                    self.assertIn('delta=r当前-r前帧', instruction)
                    self.assertIn('仍可能留在躯干左侧', instruction)
                    self.assertIn('整体移动按要求中的屏幕坐标判断', instruction)
                    self.assertIn('起始和结束两处的“摆”都描述运动相位', instruction)
                    self.assertIn('单独一张前帧的静态方位不能证明或否定此前运动方向', instruction)
        self.assertEqual(MOTION, original)

    def test_legacy_direction_failure_is_rechecked_once_even_after_detail_review(self):
        project = SimpleNamespace(root=self.root)
        spec = {'shot_id': 'S01'}
        plan = {'keyframes': [{'frame_id': 'K02', 'frame_index': 12, 'requirements': MOTION}]}
        previous = {'path': 'segments/S01/v001', 'report': {
            'status': 'FAIL', 'target_id': 'S01_K02', 'detail_confirmation': {'status': 'FAIL'},
            'checks': [{'requirement_id': 'MOTION', 'result': 'FAIL'}]}}
        path = self.root / previous['path'] / 'check_K02.request.json'
        context = {'pixel_scope': {'current_frame_index': 12, 'comparison_frame_indices': [0]}}
        write_json(path, model.messages('审核', context))
        self.assertTrue(repair.needs_motion_confirmation(project, spec, plan, previous))
        context['pixel_scope']['motion_comparison'] = repair.motion_comparison_policy(MOTION)
        write_json(path, model.messages('审核', context))
        self.assertFalse(repair.needs_motion_confirmation(project, spec, plan, previous))
        context['pixel_scope'].pop('motion_comparison')
        context['pixel_scope']['comparison_frame_indices'] = []
        write_json(path, model.messages('审核', context))
        self.assertFalse(repair.needs_motion_confirmation(project, spec, plan, previous))

    def test_legacy_pass_without_pixel_scope_cannot_bypass_direction_review(self):
        with Project(self.root, {'story': '游动'}) as project:
            first, second = self.root / 'first.png', self.root / 'second.png'
            first.write_bytes(b'first actual pixels')
            second.write_bytes(b'second actual pixels')
            images = [('实际画面 frame_index=0', first), ('实际画面 frame_index=12', second)]
            payload = {'task': {'frame_index': 12}}
            context = dict(payload, target_id='S01_K02', target_version=1,
                           requirements=MOTION, required_check_ids=['MOTION'], evidence_mode='pixels')
            old_report = {'target_id': 'S01_K02', 'target_version': 1, 'status': 'PASS',
                          'checks': [{'requirement_id': 'MOTION', 'result': 'PASS',
                                      'evidence': '旧审核通过', 'evidence_frames': [0, 12]}]}
            with patch.object(model, 'ask', return_value=old_report):
                planning.checked_review(project, self.root / 'segments/S01/v001', 'check_K02',
                    'S01_K02', 1, MOTION, context, images, [0, 12])
            rejected = {'target_id': 'S01_K02', 'target_version': 2, 'status': 'UNCERTAIN',
                        'checks': [{'requirement_id': 'MOTION', 'result': 'UNCERTAIN',
                                    'evidence': '相对位移不清楚', 'evidence_frames': [0, 12]}]}
            with patch.object(model, 'ask', return_value=rejected) as ask:
                with self.assertRaises(Rejected):
                    planning.review(project, self.root / 'segments/S01/v002', 'check_K02',
                        'S01_K02', 2, MOTION, payload, images, [0, 12])
                ask.assert_called_once()

    def test_changed_motion_policy_does_not_reuse_old_review_but_new_policy_can_reuse(self):
        with Project(self.root, {'story': '游动'}) as project:
            first, second = self.root / 'first.png', self.root / 'second.png'
            first.write_bytes(b'first actual pixels')
            second.write_bytes(b'second actual pixels')
            images = [('实际画面 frame_index=0', first), ('实际画面 frame_index=12', second)]
            payload = {'task': {'frame_index': 12}}
            def report(version):
                return {'target_id': 'S01_K02', 'target_version': version, 'status': 'PASS',
                        'checks': [{'requirement_id': 'MOTION', 'result': 'PASS',
                                    'evidence': '两帧实际位移和身体波浪可见', 'evidence_frames': [0, 12]}]}
            folder = self.root / 'segments/S01/v001'
            with patch.object(repair, 'motion_comparison_policy', return_value=None), \
                    patch.object(model, 'ask', return_value=report(1)):
                planning.review(project, folder, 'check_K02', 'S01_K02', 1,
                                MOTION, payload, images, [0, 12])
            with patch.object(model, 'ask', return_value=report(2)) as ask:
                for version in (2, 3):
                    result = planning.review(project, self.root / f'segments/S01/v{version:03d}',
                        'check_K02', 'S01_K02', version, MOTION, payload, images, [0, 12])
                    self.assertEqual(result['target_version'], version)
                ask.assert_called_once()
                self.assertIn('reused_from', result)
            self.assertEqual(read_json(folder / 'check_K02.json')['target_version'], 1)


if __name__ == '__main__':
    unittest.main()
