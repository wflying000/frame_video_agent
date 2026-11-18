"""显式相对运动参照必须绑定新像素证据，不能改变默认请求或放宽验收。"""
import base64
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from agent import drawing, model, rendering, temporal, validation, workflow
from agent.common import file_hash, read_json
from agent.store import Blocked, Project
from fixtures import BOARD, FACTS, STORY, FakeModel, frames
import test_final_sampling
import test_temporal_details
import test_workflow


class RelativeMotionReferenceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_temporal_details.TemporalDetailTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.names = [f'temporal_window_{number:03d}' for number in range(1, 5)] + ['temporal_review']

    def test_optional_flag_is_strict_bool_and_binds_source_request_hash_and_provenance(self):
        fixture = self.fixture
        _, ask = fixture.failing_model()
        with Project(self.root, fixture.config) as project, patch.object(model, 'ask', side_effect=ask), \
                patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub), \
                patch.object(validation, 'sample_indices', return_value=fixture.samples):
            _, previous, source, descriptor = fixture.fail_original(project)
            before = project.data['calls']
            for index, invalid in enumerate((0, 1, 'true', None, [], {})):
                directory = project.root / 'segments/S01' / f'invalid_flag_{index}'
                directory.mkdir()
                with self.subTest(invalid=invalid), self.assertRaises((ValueError, Blocked)):
                    drawing.temporal_detail_recheck_source(project, directory, previous,
                        dict(descriptor, relative_motion_reference=invalid))
            hashes = []
            for label, flag in (('absent', None), ('disabled', False), ('enabled', True)):
                directory = project.root / 'segments/S01' / ('v002' if flag is True else label)
                directory.mkdir()
                requested = copy.deepcopy(descriptor)
                if flag is not None:
                    requested['relative_motion_reference'] = flag
                html, details = drawing.temporal_detail_recheck_source(project, directory, previous, requested)
                proof = read_json(directory / 'temporal_detail_recheck_provenance.json')
                self.assertEqual(html, source.read_text())
                self.assertEqual(proof['source_hash'], file_hash(source))
                self.assertEqual(details['source_hash'], file_hash(source))
                hashes.append(proof['request_hash'])
                if flag is None:
                    self.assertNotIn('relative_motion_reference', proof)
                    self.assertNotIn('relative_motion_reference', details)
                else:
                    self.assertIs(proof['relative_motion_reference'], flag)
                    self.assertIs(details['relative_motion_reference'], flag)
            self.assertEqual(len(set(hashes)), 3)
            enabled = project.root / 'segments/S01/v002'
            with self.assertRaises((ValueError, Blocked)):
                drawing.temporal_detail_recheck_source(project, enabled, previous,
                    dict(descriptor, relative_motion_reference=False))
            bad_source = dict(descriptor, relative_motion_reference=True, source_hash='0' * 64)
            with self.assertRaises((ValueError, Blocked)):
                drawing.temporal_detail_recheck_source(project, enabled, previous, bad_source)
            self.assertEqual(project.data['calls'], before)

    def test_flag_cannot_bypass_existing_new_crop_guard(self):
        fixture = self.fixture
        _, ask = fixture.failing_model()
        with Project(self.root, fixture.config) as project, patch.object(model, 'ask', side_effect=ask), \
                patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub), \
                patch.object(validation, 'sample_indices', return_value=fixture.samples):
            _, previous, _, descriptor = fixture.fail_original(project)
            first = project.root / 'segments/S01/v002'; first.mkdir()
            drawing.temporal_detail_recheck_source(project, first, previous, descriptor)
            second = project.root / 'segments/S01/v003'; second.mkdir()
            enabled = dict(descriptor, relative_motion_reference=True)
            with self.assertRaises((ValueError, Blocked)):
                drawing.temporal_detail_recheck_source(project, second, previous, enabled)
            enabled['regions'] = []
            with self.assertRaises((ValueError, Blocked)):
                drawing.temporal_detail_recheck_source(project, second, previous, enabled)
            enabled = copy.deepcopy(dict(descriptor, relative_motion_reference=True))
            enabled['regions'][0]['region_xyxy'] = [0.05, 0.1, 0.4, 0.6]
            _, details = drawing.temporal_detail_recheck_source(project, second, previous, enabled)
            self.assertTrue(details['relative_motion_reference'])

    def test_enabled_context_reaches_all_windows_and_global_without_changing_pixels_or_checks(self):
        fixture = self.fixture
        saved = {}
        for label, enabled in (('ordinary', False), ('enabled', True)):
            _, ask = fixture.failing_model()
            with self.subTest(mode=label), Project(self.root / label, fixture.config) as project, \
                    patch.object(model, 'ask', side_effect=ask), \
                    patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub), \
                    patch.object(rendering, 'inspect_video', return_value={'passed': True}), \
                    patch.object(validation, 'sample_indices', return_value=fixture.samples), \
                    patch.object(rendering, 'crop_evidence', side_effect=fixture.crop):
                refs, _, source, descriptor = fixture.fail_original(project)
                old = source.parent
                old_files = {path: file_hash(path) for path in old.rglob('*') if path.is_file()}
                if enabled:
                    descriptor['relative_motion_reference'] = True
                project.data['repair_requests']['S01'] = {'temporal_detail_recheck': descriptor}
                project.task_attempt_limits['segments/S01'] = 2
                result = drawing.shot(project, fixture.spec, fixture.plan, refs)
                if enabled:
                    self.assertIs(result['relative_motion_reference'], True)
                    self.assertEqual(result['relative_motion_policy_version'], temporal.RELATIVE_MOTION_POLICY_VERSION)
                else:
                    self.assertNotIn('relative_motion_reference', result)
                    self.assertNotIn('relative_motion_policy_version', result)
                directory = project.root / 'segments/S01/v002'
                self.assertEqual(file_hash(directory / 'scene.html'), file_hash(source))
                self.assertEqual(old_files, {path: file_hash(path) for path in old_files})
                saved[label] = {}
                for name in self.names:
                    request = read_json(directory / (name + '.request.json'))
                    payload = json.loads(request[1]['content'][0]['text'])
                    signature = read_json(directory / (name + '.call.json'))['input_hash']
                    saved[label][name] = request, payload, signature
                    for forbidden in ('OLD_FAILURE_NARRATIVE', 'OLD_REPAIR_NARRATIVE', descriptor['reason']):
                        self.assertNotIn(forbidden, json.dumps(payload, ensure_ascii=False))
                    if enabled:
                        self.assertEqual(payload['relative_motion_reference']['method'],
                                         'same_endpoint_minus_stable_body_anchor')
                    else:
                        self.assertNotIn('relative_motion_reference', payload)
                if not enabled:
                    for name in self.names[:2]:
                        self.assertEqual(fixture.request_without_version(old, name),
                                         fixture.request_without_version(directory, name))
                        self.assertIn('reused_from', read_json(directory / (name + '.json')))
                self.assertNotIn('relative_motion_reference', json.loads(read_json(
                    directory / 'check_K01.request.json')[1]['content'][0]['text']))
        fixed_context = None
        for name in self.names:
            ordinary, original_payload, old_signature = saved['ordinary'][name]
            enabled, current_payload, new_signature = saved['enabled'][name]
            self.assertEqual(ordinary[1]['content'][1:], enabled[1]['content'][1:])
            for key in ('requirements', 'required_check_ids', 'sample_timing', 'frame_plan'):
                self.assertEqual(original_payload[key], current_payload[key])
            self.assertNotEqual(old_signature, new_signature)
            context = current_payload['relative_motion_reference']
            if fixed_context is None:
                fixed_context = context
            self.assertEqual(context, fixed_context)
        guidance = fixed_context['instruction'].replace(' ', '')
        self.assertIn('P-A', guidance)
        self.assertIn('半周期', guidance)
        self.assertIn('末端', guidance)
        self.assertIn('躯干', guidance)
        self.assertIn('UNCERTAIN', guidance)

    def test_final_event_reference_is_selective_and_preserves_encoded_pixels_and_requirements(self):
        saved = {}
        for enabled in (False, True):
            fake, requests = FakeModel(), {}
            root = self.root / ('final_enabled' if enabled else 'final_ordinary')
            config = {'story': STORY, 'fps': 2, 'seconds': 4}
            facts, board = copy.deepcopy(FACTS), copy.deepcopy(BOARD)
            def ask(messages, **kwargs):
                payload = json.loads(messages[1]['content'][0]['text'])
                if payload.get('target_id'):
                    requests[payload['target_id']] = copy.deepcopy(messages)
                return fake(messages)
            with self.subTest(enabled=enabled), Project(root, config) as project:
                shots = []
                for spec in board['shots']:
                    def segment(directory, version, previous, spec=spec):
                        video = directory / 'video.mp4'
                        video.write_bytes(b'encoded-shot')
                        test_workflow.WorkflowTests.render_stub(video, directory / 'source', 2, 2,
                                                               preview=True, frame_count=4, sample_indices=[0, 1, 2, 3])
                        result = {'shot_id': spec['shot_id'], 'version': version,
                                  'start_frame': spec['start_frame'], 'end_frame': spec['end_frame'],
                                  'video': str(video.relative_to(project.root)),
                                  'frame_plan': frames(spec['shot_id']),
                                  'samples': [{'frame_index': index, 'path': str(path.relative_to(project.root))}
                                              for index, path in rendering.preview_images(directory / 'source')]}
                        if enabled and spec['shot_id'] == 'S01':
                            result.update(relative_motion_reference=True,
                                          relative_motion_policy_version=temporal.RELATIVE_MOTION_POLICY_VERSION)
                        return result
                    shots.append(project.task('segments/' + spec['shot_id'], {'shot': spec}, [], segment))
                project.task('checks/timeline', {}, ['segments/S01', 'segments/S02'], lambda *args: {'status': 'PASS'})
                with test_workflow.WorkflowTests().fake_renderers(), \
                        patch.object(rendering, 'sample_video', side_effect=test_final_sampling.FinalSamplingTests.sample_video) as decode, \
                        patch.object(model, 'ask', side_effect=ask), \
                        patch.object(project, 'task', wraps=project.task) as task:
                    result = workflow.final_check(project, facts, board, shots)
                self.assertEqual(result['status'], 'PASS')
                inputs = task.call_args.args[1]
                if enabled:
                    self.assertEqual(inputs['relative_motion_references'],
                                     {'S01': temporal.RELATIVE_MOTION_POLICY_VERSION})
                else:
                    self.assertNotIn('relative_motion_references', inputs)
                self.assertEqual(decode.call_args.args[0], project.root / 'exports/final/v001/candidate_final.mp4')
                saved[enabled] = requests
                for event_id, indices in (('E01', range(4)), ('E02', range(4, 8))):
                    request = requests[event_id]
                    payload = json.loads(request[1]['content'][0]['text'])
                    event = next(row for row in facts['events'] if row['event_id'] == event_id)
                    self.assertEqual(payload['requirements'], [{'id': event_id, 'hard': True, 'text': event['must_show']}])
                    image_bytes = [base64.b64decode(row['image_url']['url'].split(',', 1)[1])
                                   for row in request[1]['content'] if row['type'] == 'image_url']
                    self.assertEqual(image_bytes, [f'decoded-final-frame:{index}'.encode() for index in indices])
                    if enabled and event_id == 'E01':
                        self.assertEqual(payload['relative_motion_reference'], temporal.relative_motion_policy())
                        self.assertEqual(payload['relative_motion_reference_shot_ids'], ['S01'])
                    else:
                        self.assertNotIn('relative_motion_reference', payload)
                        self.assertNotIn('relative_motion_reference_shot_ids', payload)
                for target in ('S01_appearance', 'S02_appearance'):
                    payload = json.loads(requests[target][1]['content'][0]['text'])
                    self.assertNotIn('relative_motion_reference', payload)
                    self.assertNotIn('relative_motion_reference_shot_ids', payload)
        for target in ('E01', 'E02', 'S01_appearance', 'S02_appearance'):
            ordinary, enabled = saved[False][target], saved[True][target]
            self.assertEqual(ordinary[1]['content'][1:], enabled[1]['content'][1:])
            original_payload = json.loads(ordinary[1]['content'][0]['text'])
            enabled_payload = json.loads(enabled[1]['content'][0]['text'])
            if target == 'E01':
                enabled_payload.pop('relative_motion_reference')
                enabled_payload.pop('relative_motion_reference_shot_ids')
            self.assertEqual(original_payload, enabled_payload)

    def test_relative_reference_never_overrides_local_or_global_nonpass(self):
        fixture = self.fixture
        for stage, status in (('window', 'FAIL'), ('global', 'UNCERTAIN')):
            fake, original_ask = fixture.failing_model(second_status=status if stage == 'window' else None)
            def ask(messages, **kwargs):
                result = original_ask(messages, **kwargs)
                payload = json.loads(messages[1]['content'][0]['text'])
                if stage == 'global' and payload.get('target_version') == 2 and payload.get('temporal_review_stage') == 'global':
                    result['status'] = status
                    result['checks'][0].update(result=status, evidence='当前实际像素未通过')
                return result
            with self.subTest(stage=stage), Project(self.root / stage, fixture.config) as project, \
                    patch.object(model, 'ask', side_effect=ask), \
                    patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub), \
                    patch.object(rendering, 'inspect_video') as inspect, \
                    patch.object(validation, 'sample_indices', return_value=fixture.samples), \
                    patch.object(rendering, 'crop_evidence', side_effect=fixture.crop):
                refs, _, source, descriptor = fixture.fail_original(project)
                old_failure = (source.parent / 'failure.json').read_bytes()
                descriptor['relative_motion_reference'] = True
                project.data['repair_requests']['S01'] = {'temporal_detail_recheck': descriptor}
                project.task_attempt_limits['segments/S01'] = 2
                with self.assertRaises(Blocked):
                    drawing.shot(project, fixture.spec, fixture.plan, refs)
                directory = project.root / 'segments/S01/v002'
                self.assertEqual(read_json(directory / 'failure.json')['status'], status)
                self.assertEqual(read_json(directory / 'temporal_aggregate.json')['status'], status)
                self.assertFalse((directory / 'result.json').exists())
                self.assertFalse((directory / 'preview/video.mp4').exists())
                self.assertFalse((project.root / 'final.mp4').exists())
                self.assertEqual((source.parent / 'failure.json').read_bytes(), old_failure)
                inspect.assert_not_called()
                if stage == 'window':
                    self.assertFalse((directory / 'temporal_review.request.json').exists())
                    self.assertFalse((directory / 'temporal_window_004.request.json').exists())


if __name__ == '__main__':
    unittest.main()
