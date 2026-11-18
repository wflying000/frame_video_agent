"""同源码后续候选保留已通过的实际运动证据，不以历史结论代替重新绑定。"""
import copy
import io
import json
import os
from contextlib import contextmanager
import unittest
from unittest.mock import patch

from agent import drawing, model, prompts, rendering, validation
from agent.common import file_hash, read_json
from agent.store import Blocked, Project
import test_keyframe_motion_recheck
import test_temporal_details
import test_workflow


class KeyframeMotionCarryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_keyframe_motion_recheck.KeyframeMotionRecheckTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.config = self.fixture.root, self.fixture.config
        self.spec, self.plan = self.fixture.spec, self.fixture.plan
        self.options = {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high'}
        environment = patch.dict(os.environ, {
            'PATH': os.environ.get('PATH', ''),
            'LLM_API_KEY': 'test-only-key', 'LLM_MODEL': 'deepseek-flash',
            'LLM_BASE_URL': 'https://api.deepseek.com',
            'LLM_EXTRA_BODY': json.dumps({'thinking': {'type': 'disabled'}, 'max_tokens': 32768}),
        }, clear=True)
        environment.start(); self.addCleanup(environment.stop)
        dotenv = patch.object(model, 'load_dotenv')
        dotenv.start(); self.addCleanup(dotenv.stop)

    @contextmanager
    def harness(self):
        fake, ordinary = self.fixture.failing_model()

        def ask(messages, **kwargs):
            report = ordinary(messages, **kwargs)
            payload = json.loads(messages[1]['content'][0]['text'])
            if payload.get('temporal_window', {}).get('number') == 1 and payload['target_version'] == 2:
                report['status'] = 'FAIL'
                report['checks'][0].update(result='FAIL', evidence='OLD_TEMPORAL_FAILURE_NOT_EVIDENCE')
                report['repair'].update(scope='frame', shot_ids=['S01'], instruction='OLD_TEMPORAL_REPAIR')
            return report

        with Project(self.root, self.config) as project, patch.object(model, 'ask', side_effect=ask), \
                patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub), \
                patch.object(rendering, 'crop_evidence', side_effect=test_temporal_details.TemporalDetailTests.crop), \
                patch.object(rendering, 'inspect_video', return_value={'passed': True}), \
                patch.object(validation, 'sample_indices', return_value=self.fixture.samples), \
                patch('sys.stdout', io.StringIO()):
            yield project, fake

    def motion_pass_temporal_fail(self, project):
        refs, _, original, descriptor = self.fixture.fail_original(project)
        project.data['repair_requests']['S01'] = {'keyframe_motion_recheck': descriptor}
        project.task_attempt_limits['segments/S01'] = 2
        with self.assertRaises(Blocked):
            drawing.shot(project, self.spec, self.plan, refs)
        motion = project.root / 'segments/S01/v002'
        self.assertEqual(read_json(motion / 'check_K05.json')['status'], 'PASS')
        self.assertEqual(read_json(motion / 'temporal_window_001.json')['status'], 'FAIL')
        details = {'source_hash': file_hash(motion / 'scene.html'),
                   'target_review': 'temporal_window_001', 'reason': 'NEUTRAL_NEW_REGION_REQUEST',
                   'regions': [{'frame_index': 204, 'region_xyxy': [0.1, 0.1, 0.8, 0.8]}]}
        return refs, original.parent, motion, details

    def test_temporal_detail_candidate_carries_exact_passed_motion_request_and_images(self):
        with self.harness() as (project, fake):
            refs, original, motion, details = self.motion_pass_temporal_fail(project)
            historical = {path: file_hash(path) for folder in (original, motion)
                          for path in folder.rglob('*') if path.is_file()}
            project.data['repair_requests']['S01'] = {'temporal_detail_recheck': details}
            project.task_attempt_limits['segments/S01'] = 3
            before = len(fake.calls)
            result = drawing.shot(project, self.spec, self.plan, refs)
            current = project.root / 'segments/S01/v003'
            self.assertEqual(result['version'], 3)
            self.assertEqual(file_hash(current / 'scene.html'), details['source_hash'])
            self.assertEqual(historical, {path: file_hash(path) for path in historical})
            for number in range(1, 6):
                name = f'check_K{number:02d}'
                self.assertEqual(self.fixture.without_version(motion, name),
                                 self.fixture.without_version(current, name))
                self.assertIn('reused_from', read_json(current / (name + '.json')))
            self.assertFalse(any(system == prompts.REVIEW and payload.get('target_id', '').startswith('S01_K')
                                 for system, payload in fake.calls[before:]))
            payload = json.loads(read_json(current / 'check_K05.request.json')[1]['content'][0]['text'])
            self.assertEqual(payload['visual_review_extra_body'], self.options)
            self.assertEqual(payload['pixel_evidence_context']['current_frame_index'], 239)
            self.assertEqual(payload['pixel_evidence_context']['comparison_frame_indices'], [192, 204, 216, 228])
            self.assertEqual(payload['keyframe_motion_evidence']['additional_frame_indices'], [204, 216, 228])
            self.assertEqual(read_json(current / 'check_K05.call.json')['model'], model.effective_settings(self.options))
            serialized = json.dumps(payload, ensure_ascii=False)
            for text in ('OLD_K05_FAILURE_NARRATIVE', 'OLD_REPAIR_NARRATIVE',
                         'OLD_TEMPORAL_FAILURE_NOT_EVIDENCE', 'OLD_TEMPORAL_REPAIR', details['reason']):
                self.assertNotIn(text, serialized)
            carry = read_json(current / 'keyframe_motion_carry_provenance.json')
            self.assertEqual(carry['source_hash'], details['source_hash'])
            proof = carry['targets']['check_K05']
            self.assertEqual(proof['effective_options'], model.effective_settings(self.options))
            for field, filename in (('original_provenance', 'keyframe_motion_recheck_provenance.json'),
                                    ('passed_report', 'check_K05.json'),
                                    ('request', 'check_K05.request.json'), ('call', 'check_K05.call.json')):
                self.assertEqual(proof[field]['path'], str((motion / filename).relative_to(project.root)))
                self.assertEqual(proof[field]['hash'], file_hash(motion / filename))
            self.assertEqual([row['frame_index'] for row in proof['actual_frames']], [192, 204, 216, 228, 239])
            for row in proof['actual_frames']:
                self.assertEqual(file_hash(project.root / row['path']), row['hash'])
            self.assertEqual(read_json(current / 'temporal_aggregate.json')['status'], 'PASS')
            self.assertTrue((current / 'preview/video.mp4').is_file())

    def test_helper_skips_changed_source_nonpass_invalid_checks_pixels_and_model(self):
        with self.harness() as (project, _):
            _, original, motion, details = self.motion_pass_temporal_fail(project)
            current = project.root / 'segments/S01/v003'; current.mkdir()
            html = current / 'scene.html'; html.write_bytes((motion / 'scene.html').read_bytes())

            def carried(options=None):
                return drawing.carried_keyframe_motion(project, current, html, self.plan,
                                                       self.options if options is None else options)

            valid = carried()
            self.assertEqual(set(valid), {'check_K05'})
            self.assertEqual(valid['check_K05']['frame_indices'], [204, 216, 228])
            self.assertEqual([index for index, _ in valid['check_K05']['images']], [204, 216, 228])
            proof = read_json(motion / 'keyframe_motion_recheck_provenance.json')
            report = read_json(motion / 'check_K05.json')
            call = read_json(motion / 'check_K05.call.json')
            variants = [(html, b'changed current HTML'),
                        (original / 'scene.html', b'changed bound source HTML'),
                        (motion / 'scene.html', b'changed own candidate HTML')]
            for status in ('FAIL', 'UNCERTAIN'):
                changed = copy.deepcopy(report); changed['status'] = status
                changed['checks'][0]['result'] = status
                variants.append((motion / 'check_K05.json', json.dumps(changed).encode()))
            changed = copy.deepcopy(report); changed['checks'] = changed['checks'][1:]
            variants.append((motion / 'check_K05.json', json.dumps(changed).encode()))
            changed = copy.deepcopy(call); changed['model']['model'] = 'different-model'
            variants.append((motion / 'check_K05.call.json', json.dumps(changed).encode()))
            variants.append((project.root / proof['actual_frames'][1]['path'], b'tampered actual pixels'))
            changed = copy.deepcopy(proof); changed['source_hash'] = '0' * 64
            variants.append((motion / 'keyframe_motion_recheck_provenance.json', json.dumps(changed).encode()))
            for number, (path, content) in enumerate(variants):
                old = path.read_bytes()
                try:
                    path.write_bytes(content)
                    with self.subTest(case=number, path=path.name):
                        self.assertEqual(carried(), {})
                finally:
                    path.write_bytes(old)
            self.assertEqual(carried(dict(self.options, reasoning_effort='low')), {})
            self.assertEqual(set(carried()), {'check_K05'})

    def test_changed_current_anchor_blocks_carry_and_following_review_or_encoding(self):
        with self.harness() as (project, fake):
            refs, _, motion, details = self.motion_pass_temporal_fail(project)
            project.data['repair_requests']['S01'] = {'temporal_detail_recheck': details}
            project.task_attempt_limits['segments/S01'] = 3

            def changed_renderer(html, out, *args, **kwargs):
                test_workflow.WorkflowTests.render_stub(html, out, *args, **kwargs)
                if 'v003' in out.parts and kwargs.get('sample_indices') == [239]:
                    for _, path in rendering.preview_images(out):
                        path.write_bytes(bytes.fromhex('89504e470d0a1a0a0000000d494844520000000200000002080600000072b60d240000001149444154789c6338a111f01f8419600c004f5408fd2453bc0a0000000049454e44ae426082'))

            before = len(fake.calls)
            with patch.object(rendering, 'render_job', side_effect=changed_renderer), \
                    patch.object(rendering, 'inspect_video') as inspect, \
                    self.assertRaises((ValueError, Blocked)):
                drawing.shot(project, self.spec, self.plan, refs)
            current = project.root / 'segments/S01/v003'
            self.assertFalse((current / 'check_K05.request.json').exists())
            self.assertFalse((current / 'temporal_window_001.request.json').exists())
            self.assertFalse((current / 'preview/video.mp4').exists())
            self.assertFalse((current / 'result.json').exists())
            self.assertEqual(read_json(motion / 'check_K05.json')['status'], 'PASS')
            self.assertFalse(any(system == prompts.REVIEW and payload.get('target_id') == 'S01_K05'
                                 for system, payload in fake.calls[before:]))
            inspect.assert_not_called()


if __name__ == '__main__':
    unittest.main()
