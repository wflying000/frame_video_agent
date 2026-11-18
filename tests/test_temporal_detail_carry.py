"""同源正式通过窗口的补充像素可继承，历史结论仍须通过完整审核签名。"""
import copy
import io
import json
import os
from contextlib import contextmanager
import unittest
from unittest.mock import patch

from agent import drawing, model, prompts, rendering, temporal, validation, workflow
from agent.common import file_hash, read_json
from agent.store import Blocked, Project
from fixtures import FACTS, FakeModel
import test_keyframe_observation
import test_temporal_details
import test_workflow


class TemporalDetailCarryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_temporal_details.TemporalDetailTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.config = self.fixture.root, dict(self.fixture.config, seconds=1.5)
        self.spec, self.plan = self.fixture.spec, self.fixture.plan
        self.options = {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high'}
        for context in (patch.dict(os.environ, {
                'LLM_BASE_URL': 'https://api.deepseek.com', 'LLM_MODEL': 'deepseek-flash',
                'LLM_EXTRA_BODY': json.dumps({'thinking': {'type': 'disabled'}, 'max_tokens': 32768})}),
                patch.object(model, 'load_dotenv'), patch('sys.stdout', io.StringIO())):
            context.start()
            self.addCleanup(context.stop)

    @contextmanager
    def harness(self, final_status=None):
        fake = FakeModel()
        def ask(messages, **kwargs):
            report = fake(messages)
            payload = json.loads(messages[1]['content'][0]['text'])
            window = payload.get('temporal_window', {}).get('number')
            version = payload.get('target_version')
            status = ('FAIL' if (version, window) in {(1, 1), (2, 3)} else
                      final_status if (version, window) == (3, 3) else None)
            if status:
                report['status'] = status
                report['checks'][0].update(result=status, evidence='OLD_FAILURE_NOT_NEW_PIXEL_EVIDENCE')
                report['repair'].update(scope='frame', shot_ids=['S01'], instruction='OLD_REPAIR_NOT_NEW_CONSTRAINT')
            return report
        with Project(self.root, self.config, review_strictness='balanced') as project, \
                patch.object(model, 'ask', side_effect=ask), \
                patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub), \
                patch.object(rendering, 'crop_evidence', side_effect=self.fixture.crop), \
                patch.object(rendering, 'inspect_video', return_value={'passed': True}), \
                patch.object(validation, 'sample_indices', return_value=self.fixture.samples):
            yield project, fake

    def first_detail_pass_later_window_fail(self, project):
        refs = test_keyframe_observation.KeyframeObservationTests().seed(project)
        project.task_attempt_limits['segments/S01'] = 1
        with self.assertRaises(Blocked):
            drawing.shot(project, self.spec, self.plan, refs)
        original = self.root / 'segments/S01/v001'
        first = {'target_review': 'temporal_window_001', 'source_hash': file_hash(original / 'scene.html'),
                 'relative_motion_reference': True, 'reason': 'OLD_HUMAN_REASON',
                 'regions': [{'frame_index': 0, 'region_xyxy': [0.1, 0.1, 0.8, 0.8]},
                             {'frame_index': 11, 'region_xyxy': [0.1, 0.1, 0.8, 0.8]}]}
        project.data['repair_requests']['S01'] = {'temporal_detail_recheck': first}
        project.task_attempt_limits['segments/S01'] = 2
        with self.assertRaises(Blocked):
            drawing.shot(project, self.spec, self.plan, refs)
        accepted = self.root / 'segments/S01/v002'
        self.assertEqual(read_json(accepted / 'temporal_window_001.json')['status'], 'PASS')
        self.assertEqual(read_json(accepted / 'temporal_window_003.json')['status'], 'FAIL')
        second = {'target_review': 'temporal_window_003', 'source_hash': file_hash(accepted / 'scene.html'),
                  'reason': 'NEW_HUMAN_REASON',
                  'regions': [{'frame_index': 20, 'region_xyxy': [0.2, 0.2, 0.6, 0.7]},
                              {'frame_index': 31, 'region_xyxy': [0.2, 0.2, 0.6, 0.7]}]}
        return refs, original, accepted, second

    def test_new_target_retains_accepted_pixels_cache_and_inherited_reference_through_final(self):
        with self.harness() as (project, fake):
            refs, original, accepted, details = self.first_detail_pass_later_window_fail(project)
            historical = {path: file_hash(path) for folder in (original, accepted)
                          for path in folder.rglob('*') if path.is_file()}
            project.data['repair_requests']['S01'] = {'temporal_detail_recheck': details}
            self.assertNotIn('relative_motion_reference', details)
            project.task_attempt_limits['segments/S01'] = 3
            before = len(fake.calls)
            result = drawing.shot(project, self.spec, self.plan, refs)
            current = self.root / 'segments/S01/v003'
            self.assertEqual(result['version'], 3)
            self.assertEqual(file_hash(current / 'scene.html'), details['source_hash'])
            self.assertEqual(historical, {path: file_hash(path) for path in historical})
            for name in ('temporal_window_001', 'temporal_window_002'):
                self.assertEqual(self.fixture.request_without_version(accepted, name),
                                 self.fixture.request_without_version(current, name))
                self.assertIn('reused_from', read_json(current / (name + '.json')))
            self.assertFalse(any(system == prompts.REVIEW and payload.get('temporal_window', {}).get('number') in (1, 2)
                                 for system, payload in fake.calls[before:]))
            request = read_json(current / 'temporal_window_001.request.json')
            context = json.loads(request[1]['content'][0]['text'])
            self.assertEqual(context['relative_motion_reference'], temporal.relative_motion_policy())
            self.assertEqual(len(context['temporal_region_details']['images']), 2)
            for sentinel in ('OLD_FAILURE_NOT_NEW_PIXEL_EVIDENCE', 'OLD_REPAIR_NOT_NEW_CONSTRAINT',
                             'OLD_HUMAN_REASON', 'NEW_HUMAN_REASON'):
                self.assertNotIn(sentinel, json.dumps(request))
            carry = read_json(current / 'temporal_detail_carry_provenance.json')
            self.assertEqual(set(carry['targets']), {'temporal_window_001'})
            proof = carry['targets']['temporal_window_001']
            self.assertTrue(proof['requires_full_review'])
            self.assertEqual(proof['effective_options'], model.effective_settings(self.options))
            self.assertEqual(proof['review_policy'], project.review_policy)
            for name, filename in (('original_provenance', 'temporal_detail_recheck_provenance.json'),
                                   ('passed_report', 'temporal_window_001.json'),
                                   ('request', 'temporal_window_001.request.json'),
                                   ('call', 'temporal_window_001.call.json')):
                self.assertEqual(proof[name]['hash'], file_hash(accepted / filename))
                self.assertEqual(proof[name]['path'], str((accepted / filename).relative_to(self.root)))
            self.assertEqual(read_json(current / 'temporal_detail_recheck_provenance.json')['target_review'],
                             'temporal_window_003')
            self.assertEqual(read_json(current / 'temporal_aggregate.json')['status'], 'PASS')
            self.assertTrue(read_json(current / 'temporal_aggregate.json')['relative_motion_reference'])
            self.assertTrue(result['relative_motion_reference'])
            self.assertEqual(result['relative_motion_policy_version'], temporal.RELATIVE_MOTION_POLICY_VERSION)
            self.assertTrue((current / 'preview/video.mp4').exists())
            # 参照来自有效继承；当前descriptor没有显式true，仍须传到实际编码成片事件验收。
            project.task('checks/timeline', {}, ['segments/S01'], lambda *args: {'status': 'PASS'})
            facts = dict(copy.deepcopy(FACTS), events=copy.deepcopy(FACTS['events'][:1]))
            before_final = len(fake.calls)
            with test_workflow.WorkflowTests().fake_renderers(), patch.object(project, 'task', wraps=project.task) as task:
                final = workflow.final_check(project, facts, {'shots': [self.spec]}, [result])
            self.assertEqual(final['status'], 'PASS')
            self.assertEqual(task.call_args.args[1]['relative_motion_references'],
                             {'S01': temporal.RELATIVE_MOTION_POLICY_VERSION})
            events = [payload for system, payload in fake.calls[before_final:]
                      if system == prompts.REVIEW and payload.get('target_id') == facts['events'][0]['event_id']]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]['relative_motion_reference'], temporal.relative_motion_policy())
            self.assertEqual(events[0]['relative_motion_reference_shot_ids'], ['S01'])

    def test_helper_rejects_tampered_sources_frames_crop_report_request_provenance_policy_and_options(self):
        with self.harness() as (project, _):
            _, original, accepted, details = self.first_detail_pass_later_window_fail(project)
            current = self.root / 'segments/S01/v003'; current.mkdir()
            html = current / 'scene.html'; html.write_bytes((accepted / 'scene.html').read_bytes())
            test_workflow.WorkflowTests.render_stub(html, current / 'preview', 2, 24, True, 36, self.fixture.samples)
            images = rendering.preview_images(current / 'preview')
            windows = temporal.sample_windows(images)
            def carried(options=None):
                return temporal.carried_detail_evidence(project, current, 'S01', windows,
                    self.options if options is None else options)
            self.assertEqual(set(carried()), {'temporal_window_001'})
            proof = read_json(accepted / 'temporal_detail_recheck_provenance.json')
            variants = [(html, b'changed current source'), (original / 'scene.html', b'changed bound source'),
                        (accepted / 'scene.html', b'changed reviewed source'),
                        (self.root / proof['detail_images'][0]['source_frame']['path'], b'changed old frame'),
                        (self.root / proof['detail_images'][0]['crop']['path'], b'changed old crop'),
                        (images[0][1], b'changed current frame')]
            for filename, change in (
                    ('temporal_window_001.json', lambda value: value.update(status='FAIL')),
                    ('temporal_window_001.json', lambda value: value.update(checks=value['checks'][1:])),
                    ('temporal_window_001.call.json', lambda value: value['model'].update(model='other')),
                    ('temporal_window_001.request.json', lambda value: value[0].update(content='changed system')),
                    ('temporal_detail_recheck_provenance.json', lambda value: value.update(request_hash='wrong'))):
                value = read_json(accepted / filename); change(value)
                variants.append((accepted / filename, json.dumps(value).encode()))
            for number, (path, content) in enumerate(variants):
                original_bytes = path.read_bytes()
                try:
                    path.write_bytes(content)
                    with self.subTest(number=number, path=path.name):
                        self.assertEqual(carried(), {})
                finally:
                    path.write_bytes(original_bytes)
            project.review_strictness = 'strict'
            self.assertEqual(carried(), {})
            project.review_strictness = 'balanced'
            self.assertEqual(carried(dict(self.options, reasoning_effort='low')), {})
            self.assertEqual(set(carried()), {'temporal_window_001'})

    def test_a_new_explicit_failure_still_blocks_remaining_windows_and_video(self):
        for status in ('FAIL', 'UNCERTAIN'):
            self.root = self.root / status
            with self.subTest(status=status), self.harness(final_status=status) as (project, fake):
                refs, _, accepted, details = self.first_detail_pass_later_window_fail(project)
                project.data['repair_requests']['S01'] = {'temporal_detail_recheck': details}
                project.task_attempt_limits['segments/S01'] = 3
                with patch.object(rendering, 'inspect_video') as inspect, self.assertRaises(Blocked):
                    drawing.shot(project, self.spec, self.plan, refs)
                current = self.root / 'segments/S01/v003'
                self.assertIn('reused_from', read_json(current / 'temporal_window_001.json'))
                self.assertEqual(read_json(current / 'temporal_aggregate.json')['status'], status)
                self.assertEqual(read_json(current / 'failure.json')['status'], status)
                for filename in ('temporal_window_004.request.json', 'temporal_review.request.json',
                                 'preview/video.mp4', 'result.json'):
                    self.assertFalse((current / filename).exists())
                inspect.assert_not_called()
                with self.assertRaises(Blocked):
                    project.publish(current / 'preview/video.mp4')

    def test_changed_recreated_crop_blocks_instead_of_claiming_inherited_pixels(self):
        with self.harness() as (project, _):
            refs, _, _, details = self.first_detail_pass_later_window_fail(project)
            project.data['repair_requests']['S01'] = {'temporal_detail_recheck': details}
            project.task_attempt_limits['segments/S01'] = 3
            def changed_crop(source, destination, region):
                self.fixture.crop(source, destination, region)
                if 'temporal_window_001_detail_evidence' in destination.parts:
                    destination.write_bytes(b'changed regenerated crop')
            with patch.object(rendering, 'crop_evidence', side_effect=changed_crop), self.assertRaises(Blocked):
                drawing.shot(project, self.spec, self.plan, refs)
            current = self.root / 'segments/S01/v003'
            self.assertFalse((current / 'temporal_window_001.request.json').exists())
            self.assertFalse((current / 'preview/video.mp4').exists())


if __name__ == '__main__':
    unittest.main()
