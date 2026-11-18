"""关键帧推理协议升级保留同一源码、完整门禁和可验证的参数来源。"""
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent import drawing, model, planning, prompts, rendering
from agent.common import file_hash, read_json, write_json
from agent.store import Blocked, Project
from fixtures import BOARD, HTML, STORY, FakeModel, frames
import test_keyframe_observation
import test_workflow


OPTIONS = {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high'}
DEFAULT = {'thinking': {'type': 'disabled'}, 'max_tokens': 32768}


class KeyframeReasonedTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / 'project'
        for context in (patch.dict(os.environ, {
                'LLM_BASE_URL': 'https://api.deepseek.com', 'LLM_MODEL': 'deepseek-flash',
                'LLM_EXTRA_BODY': json.dumps(DEFAULT)}), patch.object(model, 'load_dotenv'),
                patch('sys.stdout', io.StringIO())):
            context.start()
            self.addCleanup(context.stop)

    def source(self, protocol='keyframe_pixels_v2'):
        previous = {'version': 1, 'path': 'segments/S01/v001'}
        directory = self.root / previous['path']
        directory.mkdir(parents=True, exist_ok=True)
        source = directory / 'scene.html'
        source.write_text(HTML, encoding='utf-8')
        write_json(directory / 'frame_tasks.json', {'review_protocol': protocol})
        write_json(directory / 'failure.json', {'status': 'FAIL', 'reason': 'original report'})
        write_json(directory / 'check_K01.call.json', {'status': 'PASS', 'model': model.settings()})
        return SimpleNamespace(root=self.root), source, previous, {'source_hash': file_hash(source)}

    def test_v2_upgrade_records_source_protocol_and_actual_call_options(self):
        project, source, previous, descriptor = self.source()
        original = {path: file_hash(path) for path in source.parent.iterdir() if path.is_file()}
        target = self.root / 'segments/S01/v002'
        self.assertEqual(drawing.keyframe_recheck_source(project, target, previous, descriptor), HTML)
        proof = read_json(target / 'keyframe_recheck_provenance.json')
        self.assertEqual(proof['from_protocol'], 'keyframe_pixels_v2')
        self.assertEqual(proof['to_protocol'], 'keyframe_reasoned_v3')
        self.assertEqual(proof['source_hash'], file_hash(source))
        self.assertEqual(proof['source_version'], 1)
        self.assertEqual(proof['effective_options'], model.effective_settings(OPTIONS))
        self.assertEqual(proof['visual_review_extra_body'], OPTIONS)
        self.assertEqual(proof['options_source'], 'model.visual_review_options')
        self.assertEqual(proof['previous_review_calls'][0]['settings']['extra_body'], DEFAULT)
        self.assertEqual(proof['previous_review_calls'][0]['hash'], file_hash(source.parent / 'check_K01.call.json'))
        self.assertEqual(original, {path: file_hash(path) for path in original})
        self.assertEqual(model.settings()['extra_body'], DEFAULT)

    def test_old_observation_protocol_and_unversioned_observations_can_upgrade(self):
        for protocol in ('keyframe_observation_v1', None):
            with self.subTest(protocol=protocol):
                project, source, previous, descriptor = self.source(protocol)
                write_json(source.parent / 'keyframe_observation_K01.json', {'observations': []})
                target = self.root / 'checks' / str(protocol)
                self.assertEqual(drawing.keyframe_recheck_source(project, target, previous, descriptor), HTML)
                self.assertEqual(read_json(target / 'keyframe_recheck_provenance.json')['from_protocol'],
                                 'keyframe_observation_v1')

    def test_current_or_unknown_protocol_and_source_hash_mismatch_are_rejected(self):
        for protocol in ('keyframe_reasoned_v3', 'future_protocol'):
            with self.subTest(protocol=protocol):
                project, source, previous, descriptor = self.source(protocol)
                with self.assertRaisesRegex(Blocked, '不能重复协议复验'):
                    drawing.keyframe_recheck_source(project, self.root / 'next', previous, descriptor)
        project, source, previous, descriptor = self.source()
        with self.assertRaisesRegex(Blocked, '源版本不匹配'):
            drawing.keyframe_recheck_source(project, self.root / 'next', previous, {'source_hash': 'wrong'})
        self.assertFalse((self.root / 'next/keyframe_recheck_provenance.json').exists())

    def test_same_bound_candidate_can_resume_but_another_upgrade_cannot_repeat(self):
        project, source, previous, descriptor = self.source()
        target = self.root / 'segments/S01/v002'
        drawing.keyframe_recheck_source(project, target, previous, descriptor)
        proof_hash = file_hash(target / 'keyframe_recheck_provenance.json')
        drawing.keyframe_recheck_source(project, target, previous, descriptor)
        self.assertEqual(file_hash(target / 'keyframe_recheck_provenance.json'), proof_hash)
        with self.assertRaisesRegex(Blocked, '不能重复协议复验'):
            drawing.keyframe_recheck_source(project, self.root / 'segments/S01/v003', previous, descriptor)
        with self.assertRaisesRegex(Blocked, '绑定已改变'):
            drawing.keyframe_recheck_source(project, target, previous, descriptor, {})

    def exercise_upgrade(self, passed_source):
        fake = FakeModel()
        with Project(self.root, {'story': STORY, 'fps': 2}) as project, \
                patch.object(model, 'ask', side_effect=fake) as ask, \
                patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub), \
                patch.object(rendering, 'inspect_video', return_value={'passed': True}):
            fixture = test_keyframe_observation.KeyframeObservationTests()
            refs = fixture.seed(project)
            spec, plan = BOARD['shots'][0], frames('S01')
            if passed_source:
                def old_action(directory, version, previous):
                    source = directory / 'scene.html'
                    source.write_text(HTML, encoding='utf-8')
                    write_json(directory / 'frame_tasks.json', {'frame_plan': plan,
                        'review_protocol': 'keyframe_pixels_v2'})
                    folder = directory / 'keyframes/K01'
                    test_workflow.WorkflowTests.render_stub(source, folder, 2, 2, True, 4, [0])
                    images = [('标准资产 P01 v1', project.root / refs['P01']['image'])] + \
                             drawing.image_inputs(rendering.preview_images(folder))
                    planning.review(project, directory, 'check_K01', 'S01_K01', version,
                        plan['keyframes'][0]['requirements'], {'shot': spec, 'task': plan['keyframes'][0],
                        'fixed_assets': refs, 'allowed_black_ranges': [],
                        'keyframe_review_protocol': 'keyframe_pixels_v2'}, images, [0], visual_scope='frame')
                    return {'version': version}
                project.task('segments/S01', {'review_protocol': 'keyframe_pixels_v2'},
                    ['plans/storyboard', 'plans/frames/S01', 'assets/P01'], old_action)
                source = self.root / 'segments/S01/v001/scene.html'
                self.assertEqual(project.data['tasks']['segments/S01']['status'], 'PASS')
                self.assertNotIn('S01', project.data['repair_requests'])
            else:
                source = fixture.old_candidate(project, refs, protocol='keyframe_pixels_v2')
            old_hashes = {path: file_hash(path) for path in source.parent.rglob('*') if path.is_file()}
            fake.calls.clear(); ask.reset_mock()
            result = drawing.shot(project, spec, plan, refs)
            self.assertEqual(result['version'], 2)
            current = self.root / 'segments/S01/v002'
            self.assertEqual(file_hash(current / 'scene.html'), file_hash(source))
            self.assertEqual(old_hashes, {path: file_hash(path) for path in old_hashes})
            self.assertFalse((current / 'generated.json').exists())
            self.assertFalse(any(system in {prompts.CANVAS, prompts.PATCH, prompts.FRAME_PLAN}
                                 for system, _ in fake.calls))
            proof = read_json(current / 'keyframe_recheck_provenance.json')
            self.assertEqual(proof['from_protocol'], 'keyframe_pixels_v2')
            self.assertEqual(proof['to_protocol'], 'keyframe_reasoned_v3')
            checks = []
            for call in ask.call_args_list:
                payload = json.loads(call.args[0][1]['content'][0]['text'])
                if call.args[0][0]['content'] != prompts.REVIEW or '_K' not in payload.get('target_id', ''):
                    continue
                checks.append(payload)
                self.assertEqual(call.kwargs['extra_body'], {**DEFAULT, **OPTIONS})
                self.assertEqual(payload['visual_review_extra_body'], OPTIONS)
                self.assertEqual(payload['keyframe_review_protocol'], 'keyframe_reasoned_v3')
                self.assertIn('DEPTH_VOLUME', payload['required_check_ids'])
            self.assertEqual(len(checks), len(plan['keyframes']))
            self.assertNotIn('reused_from', read_json(current / 'check_K01.json'))
            self.assertEqual(read_json(current / 'frame_tasks.json')['review_protocol'], 'keyframe_reasoned_v3')

    def test_explicit_v2_failure_upgrade_rechecks_all_frames_without_rewriting(self):
        self.exercise_upgrade(passed_source=False)

    def test_old_pass_automatically_upgrades_without_patch_on_resume(self):
        self.exercise_upgrade(passed_source=True)


if __name__ == '__main__':
    unittest.main()
