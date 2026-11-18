"""显式单窗口像素测量严格绑定真实帧，只补证据，不替代语义验收。"""
import copy
import io
import json
import os
from contextlib import contextmanager
import unittest
from unittest.mock import patch

from agent import drawing, model, pixel_motion, prompts, rendering, temporal, validation
from agent.common import digest, file_hash, read_json, write_json
from agent.store import Blocked, Project
from fixtures import FakeModel
import test_keyframe_observation
import test_temporal_details


class TemporalPixelMotionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_temporal_details.TemporalDetailTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.config = self.fixture.root, self.fixture.config
        self.spec, self.plan = self.fixture.spec, self.fixture.plan
        self.motion = {'reference_frame': 20, 'frame_indices': [31, 20],
                       'points': [{'id': 'A', 'point_xy': [30, 30]}]}
        self.options = {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high'}
        for context in (patch.dict(os.environ, {'LLM_BASE_URL': 'https://api.deepseek.com',
                'LLM_MODEL': 'deepseek-flash', 'LLM_EXTRA_BODY': '{}'}),
                patch.object(model, 'load_dotenv'), patch('sys.stdout', io.StringIO())):
            context.start(); self.addCleanup(context.stop)

    @staticmethod
    def render(html, out, seconds, fps, preview=False, frame_count=None, sample_indices=None, **kwargs):
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for frame in sorted(set(sample_indices)):
            pixels = bytearray()
            for y in range(64):
                for x in range(64):
                    sx = (x - frame % 3) % 64
                    pixels += bytes(((sx * 17 + y * 11) % 256, (sx * sx + y * 13) % 256,
                                     (sx * 3 + y * y) % 256, 255))
            name = f'frame_{frame:03d}.png'
            (out / name).write_bytes(temporal.rgba_png(64, 64, pixels))
            rows.append({'frame_index': frame, 'path': name})
        write_json(out / 'preview_manifest.json', rows)
        if not preview:
            (out / 'video.mp4').write_bytes(b'encoded-video')

    @contextmanager
    def harness(self, measurement_status=None):
        fake = FakeModel()
        def ask(messages, **kwargs):
            result = fake(messages)
            payload = json.loads(messages[1]['content'][0]['text'])
            window, version = payload.get('temporal_window', {}).get('number'), payload.get('target_version')
            status = ('FAIL' if (version, window) in {(1, 3), (2, 3), (3, 4)} else
                      measurement_status if (version, window) == (3, 3) else None)
            if status:
                result['status'] = status
                result['checks'][0].update(result=status, evidence='OLD_FAILURE_SENTINEL')
            return result
        with Project(self.root, self.config, review_strictness='balanced') as project, \
                patch.object(model, 'ask', side_effect=ask), \
                patch.object(rendering, 'render_job', side_effect=self.render), \
                patch.object(rendering, 'crop_evidence', side_effect=self.fixture.crop), \
                patch.object(rendering, 'inspect_video', return_value={'passed': True}), \
                patch.object(validation, 'sample_indices', return_value=self.fixture.samples):
            yield project, fake

    def cropped_failure(self, project):
        refs = test_keyframe_observation.KeyframeObservationTests().seed(project)
        project.task_attempt_limits['segments/S01'] = 1
        with self.assertRaises(Blocked):
            drawing.shot(project, self.spec, self.plan, refs)
        source = self.root / 'segments/S01/v001/scene.html'
        descriptor = {'target_review': 'temporal_window_003', 'source_hash': file_hash(source),
                      'relative_motion_reference': True, 'reason': 'HUMAN_REASON_NOT_EVIDENCE',
                      'regions': [{'frame_index': 20, 'region_xyxy': [0.1, 0.1, 0.8, 0.8]},
                                  {'frame_index': 31, 'region_xyxy': [0.1, 0.1, 0.8, 0.8]}]}
        project.data['repair_requests']['S01'] = {'temporal_detail_recheck': descriptor}
        project.task_attempt_limits['segments/S01'] = 2
        with self.assertRaises(Blocked):
            drawing.shot(project, self.spec, self.plan, refs)
        descriptor = dict(descriptor, pixel_motion=copy.deepcopy(self.motion))
        return refs, project.data['tasks']['segments/S01']['attempts'][-1], descriptor

    def test_new_spec_is_bound_once_and_original_descriptor_hash_remains_compatible(self):
        with self.harness() as (project, _):
            _, previous, descriptor = self.cropped_failure(project)
            old = read_json(self.root / 'segments/S01/v002/temporal_detail_recheck_provenance.json')
            self.assertEqual(old['request_hash'], digest({'source_hash': old['source_hash'],
                'target_review': old['target_review'], 'regions': temporal.detail_regions(old),
                'relative_motion_reference': True}))
            target = self.root / 'segments/S01/v003'
            html, details = drawing.temporal_detail_recheck_source(project, target, previous, descriptor)
            proof = read_json(target / 'temporal_detail_recheck_provenance.json')
            normalized = pixel_motion.validate_spec(self.motion)
            self.assertEqual(proof['pixel_motion'], normalized)
            self.assertEqual(details['pixel_motion'], normalized)
            self.assertEqual([row['frame_index'] for row in proof['pixel_motion_frames']], [20, 31])
            for row in proof['pixel_motion_frames']:
                self.assertEqual(row['hash'], file_hash(self.root / row['path']))
            self.assertEqual(proof['request_hash'], digest({'source_hash': proof['source_hash'],
                'target_review': proof['target_review'], 'regions': temporal.detail_regions(proof),
                'relative_motion_reference': True, 'pixel_motion': normalized}))
            self.assertEqual(drawing.temporal_detail_recheck_source(project, target, previous, descriptor)[0], html)
            reordered = copy.deepcopy(descriptor); reordered['pixel_motion']['frame_indices'].reverse()
            with self.assertRaisesRegex(Blocked, '重复'):
                drawing.temporal_detail_recheck_source(project, self.root / 'segments/S01/v004', previous, reordered)
            for fault in ('source', 'frame', 'point'):
                invalid = copy.deepcopy(descriptor)
                if fault == 'source': invalid['source_hash'] = 'wrong'
                if fault == 'frame': invalid['pixel_motion']['frame_indices'].append(19)
                if fault == 'point': invalid['pixel_motion']['points'][0]['point_xy'] = [-1, 30]
                with self.subTest(fault=fault), self.assertRaises((ValueError, Blocked)):
                    drawing.temporal_detail_recheck_source(project, self.root / ('invalid_' + fault), previous, invalid)

    def test_measurements_are_canonical_and_carry_preserves_exact_request_and_pixels(self):
        with self.harness() as (project, fake):
            refs, _, descriptor = self.cropped_failure(project)
            old = self.root / 'segments/S01/v002'
            old_hashes = {path: file_hash(path) for path in old.rglob('*') if path.is_file()}
            descriptor['pixel_motion_measurement'] = {'fake_external_displacement': 999999}
            project.data['repair_requests']['S01'] = {'temporal_detail_recheck': descriptor}
            project.task_attempt_limits['segments/S01'] = 3
            with self.assertRaises(Blocked):
                drawing.shot(project, self.spec, self.plan, refs)
            measured = self.root / 'segments/S01/v003'
            name = 'temporal_window_003'
            context = json.loads(read_json(measured / (name + '.request.json'))[1]['content'][0]['text'])
            actual = dict(rendering.preview_images(measured / 'preview'))
            expected = pixel_motion.measure(self.motion, actual)
            self.assertEqual(context['pixel_motion_evidence'], temporal.pixel_motion_context(
                pixel_motion.validate_spec(self.motion), expected))
            self.assertEqual(read_json(measured / (name + '_pixel_motion.json')), expected)
            self.assertNotIn('999999', json.dumps(context))
            self.assertIn('误认', context['pixel_motion_evidence']['instruction'])
            self.assertEqual(context['required_check_ids'], json.loads(read_json(
                old / (name + '.request.json'))[1]['content'][0]['text'])['required_check_ids'])
            self.assertEqual(read_json(measured / (name + '.json'))['status'], 'PASS')
            self.assertEqual(old_hashes, {path: file_hash(path) for path in old_hashes})
            next_details = {'target_review': 'temporal_window_004', 'source_hash': descriptor['source_hash'],
                            'reason': 'NEXT_WINDOW_ONLY', 'relative_motion_reference': True,
                            'regions': [{'frame_index': 35, 'region_xyxy': [0.1, 0.1, 0.8, 0.8]}]}
            project.data['repair_requests']['S01'] = {'temporal_detail_recheck': next_details}
            project.task_attempt_limits['segments/S01'] = 4
            before = len(fake.calls)
            result = drawing.shot(project, self.spec, self.plan, refs)
            current = self.root / 'segments/S01/v004'
            self.assertEqual(result['version'], 4)
            self.assertEqual(self.fixture.request_without_version(measured, name),
                             self.fixture.request_without_version(current, name))
            self.assertIn('reused_from', read_json(current / (name + '.json')))
            self.assertFalse(any(system == prompts.REVIEW and payload.get('temporal_window', {}).get('number') == 3
                                 for system, payload in fake.calls[before:]))
            carry = read_json(current / 'temporal_detail_carry_provenance.json')['targets'][name]
            self.assertEqual(carry['pixel_motion_measurement']['hash'], file_hash(measured / (name + '_pixel_motion.json')))
            self.assertEqual(carry['current_pixel_motion_measurement']['hash'], file_hash(current / (name + '_pixel_motion.json')))
            self.assertEqual(read_json(current / (name + '_pixel_motion.json')), expected)

    def test_measurement_and_bound_frames_tampering_cannot_be_carried(self):
        with self.harness() as (project, _):
            refs, _, descriptor = self.cropped_failure(project)
            project.data['repair_requests']['S01'] = {'temporal_detail_recheck': descriptor}
            project.task_attempt_limits['segments/S01'] = 3
            with self.assertRaises(Blocked): drawing.shot(project, self.spec, self.plan, refs)
            measured = self.root / 'segments/S01/v003'
            current = self.root / 'segments/S01/v004'; current.mkdir()
            html = current / 'scene.html'; html.write_bytes((measured / 'scene.html').read_bytes())
            self.render(html, current / 'preview', 2, 24, True, 36, self.fixture.samples)
            windows = temporal.sample_windows(rendering.preview_images(current / 'preview'))
            def carried(): return temporal.carried_detail_evidence(project, current, 'S01', windows, self.options)
            self.assertIn('temporal_window_003', carried())
            proof = read_json(measured / 'temporal_detail_recheck_provenance.json')
            paths = [measured / 'temporal_window_003_pixel_motion.json',
                     self.root / proof['pixel_motion_frames'][0]['path']]
            for path in paths:
                saved = path.read_bytes()
                try:
                    path.write_bytes(b'changed measured evidence')
                    self.assertNotIn('temporal_window_003', carried())
                finally: path.write_bytes(saved)

    def test_measurement_does_not_override_fail_or_uncertain(self):
        for status in ('FAIL', 'UNCERTAIN'):
            self.root = self.root / status
            with self.subTest(status=status), self.harness(status) as (project, _):
                refs, _, descriptor = self.cropped_failure(project)
                project.data['repair_requests']['S01'] = {'temporal_detail_recheck': descriptor}
                project.task_attempt_limits['segments/S01'] = 3
                with patch.object(rendering, 'inspect_video') as inspect, self.assertRaises(Blocked):
                    drawing.shot(project, self.spec, self.plan, refs)
                current = self.root / 'segments/S01/v003'
                self.assertTrue((current / 'temporal_window_003_pixel_motion.json').exists())
                self.assertEqual(read_json(current / 'temporal_window_003.json')['status'], status)
                self.assertFalse((current / 'temporal_window_004.request.json').exists())
                self.assertFalse((current / 'preview/video.mp4').exists())
                inspect.assert_not_called()
                with self.assertRaises(Blocked): project.publish(current / 'preview/video.mp4')


if __name__ == '__main__':
    unittest.main()
