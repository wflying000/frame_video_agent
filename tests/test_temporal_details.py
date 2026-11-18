"""绑定真实源码的有限时序细节复验，不把诊断或旧失败结论注入像素审核。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent import drawing, model, planning, prompts, rendering, validation
from agent.common import file_hash, read_json
from agent.store import Blocked, Project
from fixtures import BOARD, FakeModel, STORY, frames
import test_keyframe_observation
import test_workflow


class TemporalDetailTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / 'project'
        self.config = {'story': STORY, 'fps': 24}
        self.spec = copy.deepcopy(BOARD['shots'][0])
        self.spec['end_frame'] = 36
        self.plan = frames('S01')
        self.plan['keyframes'][-1]['frame_index'] = 35
        self.plan['intervals'][-1]['end_frame'] = 36
        self.samples = list(range(36))

    @staticmethod
    def crop(source, destination, region):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())

    def failing_model(self, second_status=None):
        fake = FakeModel()
        def ask(messages, **kwargs):
            report = fake(messages)
            payload = json.loads(messages[1]['content'][0]['text'])
            if payload.get('temporal_window', {}).get('number') == 3:
                status = 'FAIL' if payload['target_version'] == 1 else second_status
                if status:
                    report['status'] = status
                    report['checks'][0].update(result=status, evidence='OLD_FAILURE_NARRATIVE')
                    report['repair']['instruction'] = 'OLD_REPAIR_NARRATIVE'
            return report
        return fake, ask

    def fail_original(self, project):
        refs = test_keyframe_observation.KeyframeObservationTests().seed(project)
        project.task_attempt_limits['segments/S01'] = 1
        with self.assertRaises(Blocked):
            drawing.shot(project, self.spec, self.plan, refs)
        previous = project.data['tasks']['segments/S01']['attempts'][-1]
        source = project.root / previous['path'] / 'scene.html'
        descriptor = {'target_review': 'temporal_window_003', 'source_hash': file_hash(source),
                      'reason': 'HUMAN_REASON_NOT_A_PIXEL_VERDICT',
                      'regions': [{'frame_index': 20, 'region_xyxy': [0.1, 0.2, 0.5, 0.7]},
                                  {'frame_index': 31, 'region_xyxy': [0.2, 0.1, 0.7, 0.6]}]}
        return refs, previous, source, descriptor

    @staticmethod
    def request_without_version(directory, name):
        request = read_json(directory / (name + '.request.json'))
        payload = json.loads(request[1]['content'][0]['text'])
        payload.pop('target_version')
        request[1]['content'][0]['text'] = payload
        return request

    def test_only_target_gets_new_pixels_and_original_passes_reuse_without_source_changes(self):
        fake, ask = self.failing_model()
        with Project(self.root, self.config) as project, patch.object(model, 'ask', side_effect=ask) as remote, \
                patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub), \
                patch.object(rendering, 'inspect_video', return_value={'passed': True}), \
                patch.object(validation, 'sample_indices', return_value=self.samples), \
                patch.object(rendering, 'crop_evidence', side_effect=self.crop) as crops:
            refs, previous, source, descriptor = self.fail_original(project)
            old = source.parent
            historical = {path: file_hash(path) for path in old.rglob('*') if path.is_file()}
            project.data['repair_requests']['S01'] = {'temporal_detail_recheck': descriptor}
            project.task_attempt_limits['segments/S01'] = 2
            before = len(fake.calls)
            result = drawing.shot(project, self.spec, self.plan, refs)
            current = project.root / 'segments/S01/v002'
            self.assertEqual(result['version'], 2)
            self.assertEqual(file_hash(current / 'scene.html'), file_hash(source))
            self.assertEqual(historical, {path: file_hash(path) for path in historical})
            self.assertEqual(read_json(old / 'failure.json')['status'], 'FAIL')
            for name in ('temporal_window_001', 'temporal_window_002'):
                self.assertEqual(self.request_without_version(old, name), self.request_without_version(current, name))
                self.assertIn('reused_from', read_json(current / (name + '.json')))
            added_calls = fake.calls[before:]
            self.assertFalse(any(system in (prompts.PATCH, prompts.CANVAS) for system, _ in added_calls))
            self.assertFalse(any(payload.get('temporal_window', {}).get('number') in (1, 2)
                                 for _, payload in added_calls))
            request = read_json(current / 'temporal_window_003.request.json')
            original_request = read_json(old / 'temporal_window_003.request.json')
            payload = json.loads(request[1]['content'][0]['text'])
            old_payload = json.loads(original_request[1]['content'][0]['text'])
            self.assertEqual(payload['requirements'], old_payload['requirements'])
            self.assertEqual(payload['required_check_ids'], old_payload['required_check_ids'])
            self.assertTrue(all(row['hard'] for row in payload['requirements']))
            self.assertEqual(payload['sample_timing'], old_payload['sample_timing'])
            # 全部原图、首尾对照与参考资产继续按原顺序出现，仅追加新裁剪。
            old_parts = original_request[1]['content'][1:]
            self.assertEqual(request[1]['content'][1:1 + len(old_parts)], old_parts)
            self.assertEqual(len(request[1]['content']), len(original_request[1]['content']) + 4)
            labels = [row['text'] for row in request[1]['content'][1 + len(old_parts):] if row['type'] == 'text']
            self.assertTrue(all('frame_index=' in label and 'region' in label for label in labels))
            self.assertTrue(all('原图' in label or '全图' in label for label in labels))
            serialized = json.dumps(request, ensure_ascii=False)
            for forbidden in ('OLD_FAILURE_NARRATIVE', 'OLD_REPAIR_NARRATIVE', descriptor['reason']):
                self.assertNotIn(forbidden, serialized)
            self.assertEqual(crops.call_count, 2)
            self.assertEqual([call.args[2] for call in crops.call_args_list],
                             [row['region_xyxy'] for row in descriptor['regions']])
            aggregate = read_json(current / 'temporal_aggregate.json')
            self.assertEqual(aggregate['status'], 'PASS')
            provenance = read_json(current / 'temporal_detail_recheck_provenance.json')
            self.assertEqual(provenance['source_hash'], file_hash(source))
            self.assertEqual(provenance['target_review'], descriptor['target_review'])
            proof = json.dumps(provenance, ensure_ascii=False)
            self.assertIn(file_hash(crops.call_args_list[0].args[0]), proof)
            self.assertIn(file_hash(crops.call_args_list[0].args[1]), proof)
            global_payload = json.loads(read_json(current / 'temporal_review.request.json')[1]['content'][0]['text'])
            original_requirements = self.spec['requirements'] + [planning.req('TEMPORAL', '动作完整、接触持续、身份稳定、因果顺序正确，无意外闪烁或漂移')]
            self.assertEqual(global_payload['requirements'], planning.depth_requirements(original_requirements, 'temporal'))
            # 新证据不应重复已接受的首两窗模型调用。
            self.assertTrue(remote.called)

    def test_binding_rejects_bad_source_target_frame_or_region_before_new_model_calls(self):
        _, ask = self.failing_model()
        with Project(self.root, self.config) as project, patch.object(model, 'ask', side_effect=ask), \
                patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub), \
                patch.object(validation, 'sample_indices', return_value=self.samples):
            _, previous, source, descriptor = self.fail_original(project)
            before = project.data['calls']
            invalid = []
            row = copy.deepcopy(descriptor); row['source_hash'] = '0' * 64; invalid.append(row)
            row = copy.deepcopy(descriptor); row['target_review'] = 'temporal_window_002'; invalid.append(row)
            row = copy.deepcopy(descriptor); row['target_review'] = '../failure'; invalid.append(row)
            row = copy.deepcopy(descriptor); row['regions'][0]['frame_index'] = 0; invalid.append(row)
            row = copy.deepcopy(descriptor); row['regions'] = []; invalid.append(row)
            row = copy.deepcopy(descriptor); row['regions'] *= 4; invalid.append(row)
            row = copy.deepcopy(descriptor); row['regions'][0]['region_xyxy'] = [0.5, 0.2, 0.2, 0.7]; invalid.append(row)
            row = copy.deepcopy(descriptor); row['regions'][0]['region_xyxy'] = [-0.1, 0.2, 0.5, 0.7]; invalid.append(row)
            for number, bad in enumerate(invalid):
                with self.subTest(number=number):
                    directory = project.root / 'segments/S01' / f'invalid{number}'
                    directory.mkdir()
                    with self.assertRaises((Blocked, ValueError)):
                        drawing.temporal_detail_recheck_source(project, directory, previous, bad)
            self.assertEqual(project.data['calls'], before)
            # 旧帧发生改变时，原绑定不能作为新的真实像素证据。
            aggregate = read_json(source.parent / 'temporal_aggregate.json')
            actual = next(row for row in aggregate['failed_review']['images'] if row['frame_index'] == 20)
            (project.root / actual['path']).write_bytes(b'changed old pixels')
            directory = project.root / 'segments/S01/changed_pixels'; directory.mkdir()
            with self.assertRaises((Blocked, ValueError)):
                drawing.temporal_detail_recheck_source(project, directory, previous, descriptor)

    def test_same_evidence_recheck_is_one_time_but_current_candidate_can_resume(self):
        _, ask = self.failing_model()
        with Project(self.root, self.config) as project, patch.object(model, 'ask', side_effect=ask), \
                patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub), \
                patch.object(validation, 'sample_indices', return_value=self.samples):
            _, previous, source, descriptor = self.fail_original(project)
            first = project.root / 'segments/S01/v002'; first.mkdir()
            html, details = drawing.temporal_detail_recheck_source(project, first, previous, descriptor)
            self.assertEqual(html, source.read_text())
            self.assertEqual(details['target_review'], descriptor['target_review'])
            self.assertTrue(all(row['source_frame_hash'] for row in details['regions']))
            # 同一RUNNING候选允许网络中断后的继续，不额外产生候选或证据。
            self.assertEqual(drawing.temporal_detail_recheck_source(project, first, previous, descriptor)[0], html)
            second = project.root / 'segments/S01/v003'; second.mkdir()
            for requested in (descriptor, dict(descriptor, regions=list(reversed(descriptor['regions']))),
                              dict(descriptor, regions=descriptor['regions'][:1])):
                with self.assertRaises((Blocked, ValueError)):
                    drawing.temporal_detail_recheck_source(project, second, previous, requested)
            fresh = copy.deepcopy(descriptor)
            fresh['regions'][0]['region_xyxy'] = [0.05, 0.1, 0.4, 0.6]
            self.assertEqual(drawing.temporal_detail_recheck_source(project, second, previous, fresh)[0], html)

    def test_detail_recheck_conflicts_with_other_mutation_or_recheck_modes(self):
        for mode in ('local_patch', 'local_source', 'keyframe_recheck', 'temporal_recheck'):
            _, ask = self.failing_model()
            with self.subTest(mode=mode), Project(self.root / mode, self.config) as project, \
                    patch.object(model, 'ask', side_effect=ask), \
                    patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub), \
                    patch.object(validation, 'sample_indices', return_value=self.samples):
                refs, _, _, descriptor = self.fail_original(project)
                project.data['repair_requests']['S01'] = {'temporal_detail_recheck': descriptor, mode: {}}
                calls = project.data['calls']
                with self.assertRaises((Blocked, ValueError)):
                    drawing.shot(project, self.spec, self.plan, refs)
                self.assertEqual(project.data['calls'], calls)

    def test_nonpass_detail_window_still_blocks_following_windows_global_and_encoding(self):
        for status in ('FAIL', 'UNCERTAIN'):
            fake, ask = self.failing_model(second_status=status)
            with self.subTest(status=status), Project(self.root / status, self.config) as project, \
                    patch.object(model, 'ask', side_effect=ask), \
                    patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub), \
                    patch.object(validation, 'sample_indices', return_value=self.samples), \
                    patch.object(rendering, 'crop_evidence', side_effect=self.crop), \
                    patch.object(rendering, 'inspect_video') as inspect:
                refs, _, source, descriptor = self.fail_original(project)
                old_failure = (source.parent / 'failure.json').read_bytes()
                project.data['repair_requests']['S01'] = {'temporal_detail_recheck': descriptor}
                project.task_attempt_limits['segments/S01'] = 2
                before = len(fake.calls)
                with self.assertRaises(Blocked):
                    drawing.shot(project, self.spec, self.plan, refs)
                self.assertEqual((source.parent / 'failure.json').read_bytes(), old_failure)
                current = project.root / 'segments/S01/v002'
                self.assertEqual(read_json(current / 'temporal_aggregate.json')['status'], status)
                self.assertEqual(read_json(current / 'failure.json')['status'], status)
                self.assertFalse((current / 'temporal_window_004.request.json').exists())
                self.assertFalse((current / 'temporal_review.request.json').exists())
                self.assertFalse((current / 'preview/video.mp4').exists())
                self.assertFalse((current / 'result.json').exists())
                self.assertFalse((project.root / 'final.mp4').exists())
                inspect.assert_not_called()
                self.assertFalse(any(payload.get('temporal_review_stage') == 'global' for _, payload in fake.calls[before:]))


if __name__ == '__main__':
    unittest.main()
