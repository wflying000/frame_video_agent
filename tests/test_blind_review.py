"""像素复核隔离先验结论，诊断叙述变化不使相同证据的 PASS 失效。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent import drawing, model, planning, prompts, rendering
from agent.common import file_hash, read_json, write_json
from agent.store import Project, Rejected
from fixtures import BOARD, FakeModel, REQ, STORY, frames
import test_keyframe_observation
import test_workflow


class BlindReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'project'

    @staticmethod
    def crop(source, output, region):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b'real cropped pixels')

    def test_blind_detail_has_all_original_checks_and_no_initial_judgment(self):
        for second_status in ['PASS', 'FAIL', 'UNCERTAIN']:
            with self.subTest(status=second_status), Project(self.root / second_status, {'story': STORY}) as project:
                image = project.root / 'frame.png'; image.write_bytes(b'actual pixels')
                requirements = REQ + [{'id': 'CONTACT', 'hard': True, 'text': '接合关系与阴影可辨'}]
                fake = FakeModel()
                def ask(messages, **kwargs):
                    result = fake(messages)
                    count = len(fake.calls)
                    status = 'FAIL' if count == 1 else second_status
                    result['status'] = status
                    result['checks'][0 if count == 1 else 1].update(result=status, evidence_frames=[0],
                        evidence='INITIAL_VERDICT_SENTINEL' if count == 1 else 'actual region evidence',
                        region_xyxy=[.2,.3,.5,.7])
                    if count == 1:
                        result['repair']['instruction'] = 'INITIAL_REPAIR_SENTINEL'
                    return result
                folder = project.root / 'segments/S01/v001'
                with patch.object(model, 'ask', side_effect=ask) as called, \
                        patch.object(rendering, 'crop_evidence', side_effect=self.crop):
                    arguments = (project, folder, 'check_K01', 'S01_K01', 1, requirements, {},
                                 [('实际画面 frame_index=0', image)], [0])
                    if second_status == 'PASS':
                        self.assertEqual(planning.review(*arguments)['status'], 'PASS')
                    else:
                        with self.assertRaises(Rejected):
                            planning.review(*arguments)
                    self.assertEqual(called.call_count, 2)
                    message = called.call_args.args[0][1]['content']
                    context = json.loads(message[0]['text'])
                    self.assertEqual(context['requirements'], requirements)
                    self.assertEqual(context['required_check_ids'], ['R01', 'CONTACT'])
                    self.assertEqual(context['pixel_review_protocol'], 'blind_detail_v2')
                    self.assertNotIn('initial_report', context['detail_confirmation'])
                    self.assertNotIn('INITIAL_', json.dumps(message))
                    labels = [part['text'] for part in message[1:] if part['type'] == 'text']
                    self.assertIn('实际画面 frame_index=0', labels)
                    self.assertTrue(any(label.startswith('区域细节：实际画面 frame_index=0') and
                                        'region_xyxy=[0.2, 0.3, 0.5, 0.7]' in label for label in labels))
                    self.assertFalse(any('失败区域' in label for label in labels))
                    original = read_json(folder / 'check_K01_initial_result.json')
                    self.assertEqual(original['repair']['instruction'], 'INITIAL_REPAIR_SENTINEL')
                    self.assertEqual(read_json(folder / 'check_K01.json')['status'], 'FAIL')
                    proof = read_json(folder / 'check_K01_resolution.json')
                    self.assertEqual(proof['status'], second_status)
                    self.assertEqual(proof['review_protocol'], 'blind_detail_v2')
                    self.assertEqual(proof['initial_hash'], file_hash(folder / 'check_K01_initial_result.json'))

    def test_legacy_pass_is_not_reused_across_pixel_review_protocol(self):
        with Project(self.root, {'story': STORY}) as project:
            image = self.root / 'frame.png'; image.write_bytes(b'unchanged actual pixels')
            images = [('实际画面 frame_index=0', image)]
            fake = FakeModel()
            context = {'target_id': 'S01_K01', 'target_version': 1, 'requirements': REQ,
                       'required_check_ids': ['R01'], 'evidence_mode': 'pixels'}
            first = self.root / 'segments/S01/v001'
            with patch.object(model, 'ask', side_effect=fake):
                planning.checked_review(project, first, 'check_K01', 'S01_K01', 1,
                                        REQ, context, images, [0])
            original_hashes = {p: file_hash(p) for p in first.rglob('*') if p.is_file()}
            fake.calls.clear()
            with patch.object(model, 'ask', side_effect=fake) as ask:
                second = planning.review(project, self.root / 'segments/S01/v002', 'check_K01',
                                         'S01_K01', 2, REQ, {}, images, [0])
                third = planning.review(project, self.root / 'segments/S01/v003', 'check_K01',
                                        'S01_K01', 3, REQ, {}, images, [0])
                ask.assert_called_once()
                self.assertNotIn('reused_from', second)
                self.assertIn('reused_from', third)
            self.assertEqual({p: file_hash(p) for p in original_hashes}, original_hashes)

    def test_changing_observation_diagnostics_does_not_change_pixel_review_or_reuse(self):
        fake = FakeModel()
        def ask(messages, **kwargs):
            result = fake(messages)
            if messages[0]['content'] == prompts.OBSERVE:
                for row in result['observations']:
                    row['description'] = f'diagnostic variation {len(fake.calls)}'
            return result
        with Project(self.root, {'story': STORY, 'fps': 2}) as project, \
                patch.object(model, 'ask', side_effect=ask), \
                patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub), \
                patch.object(rendering, 'inspect_video', return_value={'passed': True}):
            fixture = test_keyframe_observation.KeyframeObservationTests()
            refs = fixture.seed(project)
            drawing.shot(project, BOARD['shots'][0], frames('S01'), refs)
            first = self.root / 'segments/S01/v001'
            project.invalidate('segments/S01')
            before = len([call for call in fake.calls if call[0] == prompts.REVIEW])
            drawing.shot(project, BOARD['shots'][0], frames('S01'), refs)
            second = self.root / 'segments/S01/v002'
            self.assertEqual(before, len([call for call in fake.calls if call[0] == prompts.REVIEW]))
            self.assertNotEqual(read_json(first / 'keyframe_observation_K01.json'),
                                read_json(second / 'keyframe_observation_K01.json'))
            def request_without_version(folder):
                request = read_json(folder / 'check_K01.request.json')
                payload = json.loads(request[1]['content'][0]['text'])
                payload.pop('target_version')
                self.assertNotIn('independent_observations', payload)
                self.assertNotIn('observation_context', payload)
                self.assertNotIn('diagnostic variation', json.dumps(payload))
                self.assertEqual(payload['pixel_evidence_context']['current_frame_index'], 0)
                self.assertEqual(payload['pixel_evidence_context']['comparison_frame_indices'], [])
                self.assertIn('标准资产用于身份与材质参考', payload['pixel_evidence_context']['instruction'])
                request[1]['content'][0]['text'] = payload
                return request
            self.assertEqual(request_without_version(first), request_without_version(second))
            self.assertIn('reused_from', read_json(second / 'check_K01.json'))
            self.assertEqual(read_json(second / 'frame_tasks.json')['review_protocol'], 'keyframe_reasoned_v3')


if __name__ == '__main__':
    unittest.main()
