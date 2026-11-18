"""关键帧一次性补充真实中间采样，保留静态门禁、原像素及通过缓存。"""
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent import drawing, model, prompts, rendering, validation
from agent.common import digest, file_hash, read_json
from agent.store import Blocked, Project
from fixtures import BOARD, FakeModel, STORY, frames
from test_model_logging import Response, event
import test_keyframe_observation
import test_workflow


class KeyframeMotionRecheckTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / 'project'
        self.config = {'story': STORY, 'fps': 24}
        self.spec = copy.deepcopy(BOARD['shots'][0])
        self.spec['end_frame'] = 240
        self.plan = frames('S01')
        template = self.plan['keyframes'][0]
        self.plan['keyframes'] = [dict(copy.deepcopy(template), frame_id=f'K{number:02d}', frame_index=index)
                                  for number, index in enumerate([0, 72, 144, 192, 239], 1)]
        self.plan['intervals'][-1]['end_frame'] = 240
        self.samples = [0, 72, 144, 192, 204, 216, 228, 239]

    def failing_model(self, later_status=None):
        fake = FakeModel()
        def ask(messages, **kwargs):
            report = fake(messages)
            payload = json.loads(messages[1]['content'][0]['text'])
            # 多帧比较仍须以当前帧证明静态体积，测试替身遵守既有像素门禁。
            current = payload.get('task', {}).get('frame_index')
            if current is not None and 'checks' in report:
                for row in report['checks']:
                    if row['requirement_id'] in payload.get('visual_depth_check_ids', []):
                        row['evidence_frames'] = [current]
            if report.get('target_id') == 'S01_K05':
                status = 'FAIL' if payload['target_version'] == 1 else later_status
                if status:
                    report['status'] = status
                    report['checks'][0].update(result=status, evidence='OLD_K05_FAILURE_NARRATIVE')
                    report['repair'].update(scope='frame', shot_ids=['S01'], instruction='OLD_REPAIR_NARRATIVE')
            return report
        return fake, ask

    def fail_original(self, project):
        refs = test_keyframe_observation.KeyframeObservationTests().seed(project)
        project.task_attempt_limits['segments/S01'] = 1
        with self.assertRaises(Blocked):
            drawing.shot(project, self.spec, self.plan, refs)
        previous = project.data['tasks']['segments/S01']['attempts'][-1]
        source = project.root / previous['path'] / 'scene.html'
        descriptor = {'source_hash': file_hash(source), 'target_review': 'check_K05',
                      'frame_indices': [204, 216, 228], 'reason': 'HUMAN_DIAGNOSIS_NOT_PIXEL_EVIDENCE'}
        return refs, previous, source, descriptor

    @staticmethod
    def without_version(directory, name):
        request = read_json(directory / (name + '.request.json'))
        context = json.loads(request[1]['content'][0]['text'])
        context.pop('target_version')
        request[1]['content'][0]['text'] = context
        return request

    def test_only_failed_keyframe_gets_middle_frames_and_other_passes_reuse_exactly(self):
        fake, ask = self.failing_model()
        with Project(self.root, self.config) as project, patch.object(model, 'ask', side_effect=ask), \
                patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub) as render, \
                patch.object(rendering, 'inspect_video', return_value={'passed': True}), \
                patch.object(validation, 'sample_indices', return_value=self.samples):
            refs, _, source, descriptor = self.fail_original(project)
            original = source.parent
            hashes = {path: file_hash(path) for path in original.rglob('*') if path.is_file()}
            project.data['repair_requests']['S01'] = {'keyframe_motion_recheck': descriptor}
            project.task_attempt_limits['segments/S01'] = 2
            before = len(fake.calls)
            result = drawing.shot(project, self.spec, self.plan, refs)
            directory = project.root / 'segments/S01/v002'
            self.assertEqual(result['version'], 2)
            self.assertEqual(file_hash(source), file_hash(directory / 'scene.html'))
            self.assertEqual(hashes, {path: file_hash(path) for path in hashes})
            self.assertEqual(read_json(original / 'failure.json')['status'], 'FAIL')
            for frame_id in ('K01', 'K02', 'K03', 'K04'):
                name = 'check_' + frame_id
                self.assertEqual(self.without_version(original, name), self.without_version(directory, name))
                self.assertIn('reused_from', read_json(directory / (name + '.json')))
            added = fake.calls[before:]
            self.assertFalse(any(system in (prompts.PATCH, prompts.CANVAS, prompts.FRAME_PLAN) for system, _ in added))
            self.assertFalse(any(system == prompts.REVIEW and payload.get('target_id') in
                                {'S01_K01', 'S01_K02', 'S01_K03', 'S01_K04'} for system, payload in added))
            request = read_json(directory / 'check_K05.request.json')
            old_request = read_json(original / 'check_K05.request.json')
            payload = json.loads(request[1]['content'][0]['text'])
            old_payload = json.loads(old_request[1]['content'][0]['text'])
            self.assertEqual(payload['requirements'], old_payload['requirements'])
            self.assertEqual(payload['required_check_ids'], old_payload['required_check_ids'])
            self.assertTrue(all(row['hard'] for row in payload['requirements']))
            self.assertEqual(payload['pixel_evidence_context']['current_frame_index'], 239)
            self.assertEqual(payload['pixel_scope']['current_frame_index'], 239)
            self.assertEqual(payload['pixel_evidence_context']['comparison_frame_indices'], [192, 204, 216, 228])
            labels = [row['text'] for row in request[1]['content'][1:] if row['type'] == 'text']
            self.assertEqual([label for label in labels if label.startswith('实际画面 frame_index=')],
                             [f'实际画面 frame_index={index}' for index in [192, 204, 216, 228, 239]])
            self.assertIn('标准资产 P01 v1', labels)
            # 原始完整图和引用全部保留，并保持其相对顺序。
            remaining = iter(request[1]['content'][1:])
            for old_part in old_request[1]['content'][1:]:
                self.assertTrue(any(part == old_part for part in remaining))
            self.assertEqual(len(request[1]['content']), len(old_request[1]['content']) + 6)
            self.assertIn('keyframe_motion_evidence', payload)
            serialized = json.dumps(request, ensure_ascii=False)
            for forbidden in (descriptor['reason'], 'OLD_K05_FAILURE_NARRATIVE', 'OLD_REPAIR_NARRATIVE'):
                self.assertNotIn(forbidden, serialized)
            added_render = next(call for call in render.call_args_list
                                if call.kwargs.get('sample_indices') == descriptor['frame_indices'])
            sample_folder = added_render.args[1]
            actual = rendering.preview_images(sample_folder)
            self.assertEqual([index for index, _ in actual], descriptor['frame_indices'])
            proof = read_json(directory / 'keyframe_motion_recheck_provenance.json')
            self.assertEqual(proof['source_hash'], descriptor['source_hash'])
            self.assertEqual(proof['target_review'], 'check_K05')
            proof_text = json.dumps(proof, ensure_ascii=False)
            for _, path in actual:
                self.assertIn(file_hash(path), proof_text)
                self.assertIn(str(path.relative_to(project.root)), proof_text)
            self.assertTrue((directory / 'temporal_review.json').exists())

    def test_all_keyframes_enable_reasoning_in_actual_http_body_and_bound_metadata(self):
        enabled = {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high'}
        defaults = {'thinking': {'type': 'disabled'}, 'max_tokens': 32768}
        _, response_report = self.failing_model()
        bodies = []

        def transport(request, **kwargs):
            body = json.loads(request.data)
            bodies.append(body)
            report = response_report(body['messages'])
            return Response(event({'content': json.dumps(report, ensure_ascii=False)}, 'stop') + 'data: [DONE]\n\n')

        with patch.dict(os.environ, {'LLM_API_KEY': 'test-only-key', 'LLM_MODEL': 'deepseek-flash',
                'LLM_BASE_URL': 'https://api.deepseek.com', 'LLM_EXTRA_BODY': json.dumps(defaults),
                'PATH': os.environ.get('PATH', '')}, clear=True), \
                patch.object(model, 'load_dotenv'), patch('sys.stdout', io.StringIO()), \
                Project(self.root, self.config) as project, patch.object(model, 'urlopen', side_effect=transport), \
                patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub), \
                patch.object(rendering, 'inspect_video', return_value={'passed': True}), \
                patch.object(validation, 'sample_indices', return_value=self.samples):
            refs, _, source, descriptor = self.fail_original(project)
            descriptor['frame_indices'] = [204, 216, 228, 237, 238]
            project.data['repair_requests']['S01'] = {'keyframe_motion_recheck': descriptor}
            project.task_attempt_limits['segments/S01'] = 2
            drawing.shot(project, self.spec, self.plan, refs)
            directory = project.root / 'segments/S01/v002'
            target_body = None
            initial_targets = set()
            observer_count = 0
            for body in bodies:
                if body['messages'][0]['content'] == prompts.OBSERVE:
                    self.assertEqual(body['thinking'], defaults['thinking'])
                    self.assertNotIn('reasoning_effort', body)
                    observer_count += 1
                if body['messages'][0]['content'] != prompts.REVIEW:
                    continue
                payload = json.loads(body['messages'][1]['content'][0]['text'])
                target = payload.get('target_id', '')
                if not target.startswith('S01_K'):
                    continue
                self.assertEqual(body['thinking'], enabled['thinking'])
                self.assertEqual(body['reasoning_effort'], 'high')
                self.assertEqual(body['max_tokens'], 32768)
                self.assertEqual(payload['visual_review_extra_body'], enabled)
                self.assertEqual(payload['keyframe_review_protocol'], 'keyframe_reasoned_v3')
                if target == 'S01_K05' and payload['target_version'] == 2:
                    self.assertIsNone(target_body)
                    target_body = body
                else:
                    self.assertEqual(payload['target_version'], 1)
                    initial_targets.add(target)
            self.assertEqual(initial_targets, {f'S01_K{number:02d}' for number in range(1, 6)})
            self.assertGreater(observer_count, 0)
            self.assertIsNotNone(target_body)
            effective = model.effective_settings(enabled)
            metadata = read_json(directory / 'check_K05.call.json')
            self.assertEqual(metadata['model'], effective)
            self.assertEqual(metadata['input_hash'], digest({'messages': target_body['messages'], 'settings': effective}))
            proof = read_json(directory / 'keyframe_motion_recheck_provenance.json')
            self.assertEqual(proof['effective_options'], effective)
            self.assertEqual(proof['frame_indices'], descriptor['frame_indices'])
            for number in range(1, 5):
                name = f'check_K{number:02d}'
                self.assertEqual(self.without_version(source.parent, name), self.without_version(directory, name))
                self.assertIn('reused_from', read_json(directory / (name + '.json')))
            self.assertEqual(model.settings()['extra_body'], defaults)

    def test_source_target_and_middle_frame_guards_reject_before_new_calls(self):
        _, ask = self.failing_model()
        with Project(self.root, self.config) as project, patch.object(model, 'ask', side_effect=ask), \
                patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub), \
                patch.object(validation, 'sample_indices', return_value=self.samples):
            _, previous, source, descriptor = self.fail_original(project)
            invalid = [dict(descriptor, source_hash='0' * 64), dict(descriptor, target_review='check_K04'),
                       dict(descriptor, target_review='check_K01'), dict(descriptor, target_review='../failure'),
                       dict(descriptor, target_review='check_K99')]
            invalid += [dict(descriptor, frame_indices=value) for value in
                        ([], [204] * 2, [193, 194, 195, 196, 197, 198, 199], [192], [239],
                         [0], [240], [204.0], [True])]
            calls = project.data['calls']
            for number, bad in enumerate(invalid):
                with self.subTest(number=number):
                    directory = project.root / 'segments/S01' / f'bad_{number}'
                    directory.mkdir()
                    with self.assertRaises((Blocked, ValueError)):
                        drawing.keyframe_motion_recheck_source(project, directory, previous, bad, 'S01', self.plan)
            self.assertEqual(project.data['calls'], calls)
            # 删除上一目标实际审核文件后不能只凭文字描述复验。
            (source.parent / 'check_K05.json').unlink()
            directory = project.root / 'segments/S01/missing_review'; directory.mkdir()
            with self.assertRaises((Blocked, ValueError)):
                drawing.keyframe_motion_recheck_source(project, directory, previous, descriptor, 'S01', self.plan)

    def test_repeat_evidence_is_rejected_but_running_candidate_resume_is_allowed(self):
        _, ask = self.failing_model()
        with Project(self.root, self.config) as project, patch.object(model, 'ask', side_effect=ask), \
                patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub), \
                patch.object(validation, 'sample_indices', return_value=self.samples):
            _, previous, source, descriptor = self.fail_original(project)
            first = project.root / 'segments/S01/v002'; first.mkdir()
            html, bound = drawing.keyframe_motion_recheck_source(project, first, previous, descriptor, 'S01', self.plan)
            self.assertEqual(html, source.read_text())
            self.assertEqual(bound['frame_indices'], [204, 216, 228])
            self.assertEqual(drawing.keyframe_motion_recheck_source(
                project, first, previous, descriptor, 'S01', self.plan)[0], html)
            second = project.root / 'segments/S01/v003'; second.mkdir()
            for frames_requested in ([204, 216, 228], [228, 216, 204], [204], [200, 212, 224]):
                with self.subTest(frames=frames_requested), self.assertRaises((Blocked, ValueError)):
                    drawing.keyframe_motion_recheck_source(project, second, previous,
                        dict(descriptor, frame_indices=frames_requested), 'S01', self.plan)
            self.assertEqual(drawing.keyframe_motion_recheck_source(project, second, previous,
                dict(descriptor, frame_indices=[204, 216, 228, 237, 238]), 'S01', self.plan)[0], html)

    def test_motion_recheck_is_mutually_exclusive_with_all_other_repair_modes(self):
        for mode in ('local_source', 'local_patch', 'keyframe_recheck', 'temporal_recheck', 'temporal_detail_recheck'):
            _, ask = self.failing_model()
            with self.subTest(mode=mode), Project(self.root / mode, self.config) as project, \
                    patch.object(model, 'ask', side_effect=ask), \
                    patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub), \
                    patch.object(validation, 'sample_indices', return_value=self.samples):
                refs, _, _, descriptor = self.fail_original(project)
                project.data['repair_requests']['S01'] = {'keyframe_motion_recheck': descriptor, mode: {}}
                calls = project.data['calls']
                with self.assertRaises((Blocked, ValueError)):
                    drawing.shot(project, self.spec, self.plan, refs)
                self.assertEqual(project.data['calls'], calls)

    def test_failed_or_uncertain_motion_recheck_blocks_temporal_encoding_and_publication(self):
        for status in ('FAIL', 'UNCERTAIN'):
            fake, ask = self.failing_model(later_status=status)
            with self.subTest(status=status), Project(self.root / status, self.config) as project, \
                    patch.object(model, 'ask', side_effect=ask), \
                    patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub), \
                    patch.object(validation, 'sample_indices', return_value=self.samples), \
                    patch.object(rendering, 'inspect_video') as inspect:
                refs, _, source, descriptor = self.fail_original(project)
                failure = (source.parent / 'failure.json').read_bytes()
                project.data['repair_requests']['S01'] = {'keyframe_motion_recheck': descriptor}
                project.task_attempt_limits['segments/S01'] = 2
                before = len(fake.calls)
                with self.assertRaises(Blocked):
                    drawing.shot(project, self.spec, self.plan, refs)
                directory = project.root / 'segments/S01/v002'
                self.assertEqual(read_json(directory / 'failure.json')['status'], status)
                self.assertEqual((source.parent / 'failure.json').read_bytes(), failure)
                self.assertFalse((directory / 'temporal_review.request.json').exists())
                self.assertFalse((directory / 'preview/video.mp4').exists())
                self.assertFalse((directory / 'result.json').exists())
                self.assertFalse((project.root / 'final.mp4').exists())
                self.assertFalse(any(payload.get('temporal_review_stage') for _, payload in fake.calls[before:]))
                inspect.assert_not_called()


if __name__ == '__main__':
    unittest.main()
