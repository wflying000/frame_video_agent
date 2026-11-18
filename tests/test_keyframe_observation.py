"""关键帧独立观察隔离资产和预期状态，协议升级只复验一次且不改旧审核。"""
import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from agent import drawing, model, planning, prompts, rendering
from agent.common import file_hash, read_json, write_json
from agent.store import Project, Blocked, Rejected
from fixtures import BOARD, FACTS, HTML, STORY, VISUAL, FakeModel, frames
import test_workflow


class KeyframeObservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'out'
        self.config = {'story': STORY, 'fps': 2}

    def seed(self, project):
        project.task('plans/story', {}, [], lambda *args: FACTS)
        project.task('plans/visual', {}, ['plans/story'], lambda *args: VISUAL)
        project.task('plans/storyboard', {}, ['plans/story', 'plans/visual'], lambda *args: BOARD)
        def asset(directory, *_):
            image = directory / 'reference.png'
            image.write_bytes(b'ASSET REFERENCE - not the actual frame')
            return {'version': 1, 'image': str(image.relative_to(project.root))}
        reference = project.task('assets/P01', {}, ['plans/visual'], asset)
        project.task('plans/frames/S01', {}, ['plans/storyboard'], lambda *args: frames('S01'))
        return {'P01': reference}

    def old_candidate(self, project, refs, protocol=None):
        spec, plan = BOARD['shots'][0], frames('S01')
        def action(directory, version, previous):
            html = directory / 'scene.html'
            html.write_text(HTML)
            write_json(directory / 'frame_tasks.json', {'frame_plan': plan,
                **({'review_protocol': protocol} if protocol is not None else {})})
            folder = directory / 'keyframes/K01'
            test_workflow.WorkflowTests.render_stub(html, folder, 2, 2, True, 4, [0])
            images = [('标准资产 P01 v1', project.root / refs['P01']['image'])] + drawing.image_inputs(rendering.preview_images(folder))
            planning.review(project, directory, 'check_K01', 'S01_K01', version, plan['keyframes'][0]['requirements'],
                {'shot': spec, 'task': plan['keyframes'][0], 'fixed_assets': refs, 'allowed_black_ranges': [],
                 **({'keyframe_review_protocol': protocol} if protocol is not None else {})},
                images, [0], visual_scope='frame')
            raise Rejected({'status': 'FAIL', 'target_id': 'S01_K02', 'checks': [],
                            'repair': {'scope': 'frame', 'shot_ids': ['S01'], 'instruction': 'old report'}})
        with self.assertRaises(Rejected):
            project.task('segments/S01', {'old_protocol': True},
                ['plans/storyboard', 'plans/frames/S01', 'assets/P01'], action, retry=False)
        source = project.root / 'segments/S01/v001/scene.html'
        project.data['repair_requests']['S01'] = {'keyframe_recheck': {'source_hash': file_hash(source)}}
        return source

    def test_observer_receives_actual_frames_without_reference_or_desired_state(self):
        fake = FakeModel()
        with Project(self.root, self.config) as project, patch.object(model, 'ask', side_effect=fake) as ask:
            batch = []
            for index in [48, 96]:
                path = project.root / f'{index}.png'
                path.write_bytes(f'actual-pixels-{index}'.encode())
                batch.append((index, path))
            rows = drawing.observe_keyframes(project, project.root / 'candidate', 'keyframe_observation_K03', batch)
            self.assertEqual([row['frame_index'] for row in rows], [48, 96])
            system, payload = fake.calls[0]
            self.assertEqual(system, prompts.OBSERVE)
            self.assertEqual(set(payload), {'frame_indices', 'observation_task', 'output_contract'})
            self.assertIn('接触、空隙、连接物、遮挡', payload['observation_task'])
            self.assertIn('近似位置', payload['observation_task'])
            self.assertIn('不预设', payload['observation_task'])
            self.assertIn('未知', payload['observation_task'])
            content = ask.call_args.args[0][1]['content']
            self.assertEqual([part['text'] for part in content[1:] if part['type'] == 'text'],
                             ['实际画面 frame_index=48', '实际画面 frame_index=96'])
            self.assertEqual([part['image_url']['url'] for part in content if part['type'] == 'image_url'],
                ['data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode() for _, path in batch])
            calls = project.data['calls']
            drawing.observe_keyframes(project, project.root / 'candidate', 'keyframe_observation_K03', batch)
            self.assertEqual(project.data['calls'], calls)

    def test_upgrade_keeps_old_failure_and_cannot_reuse_old_protocol_pass(self):
        fake = FakeModel()
        with Project(self.root, self.config) as project, patch.object(model, 'ask', side_effect=fake) as ask, \
                patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub), \
                patch.object(rendering, 'inspect_video', return_value={'passed': True}):
            refs = self.seed(project)
            source = self.old_candidate(project, refs)
            old_hashes = {path: file_hash(path) for path in source.parent.rglob('*') if path.is_file()}
            fake.calls.clear(); ask.reset_mock()
            result = drawing.shot(project, BOARD['shots'][0], frames('S01'), refs)
            self.assertEqual(result['version'], 2)
            folder = project.root / 'segments/S01/v002'
            self.assertEqual((folder / 'scene.html').read_bytes(), source.read_bytes())
            self.assertEqual({path: file_hash(path) for path in old_hashes}, old_hashes)
            self.assertEqual(read_json(source.parent / 'failure.json')['status'], 'FAIL')
            self.assertTrue(read_json(folder / 'keyframe_recheck_provenance.json')['requires_full_review'])
            self.assertEqual([system for system, _ in fake.calls],
                [prompts.OBSERVE, prompts.REVIEW, prompts.OBSERVE, prompts.REVIEW, prompts.REVIEW, prompts.REVIEW])
            observed = [payload['frame_indices'] for system, payload in fake.calls if system == prompts.OBSERVE]
            self.assertEqual(observed, [[0], [0, 3]])
            checks = [payload for system, payload in fake.calls if system == prompts.REVIEW]
            for check in checks[:2]:
                self.assertEqual(check['keyframe_review_protocol'], 'keyframe_reasoned_v3')
                self.assertEqual(check['evidence_mode'], 'pixels')
                self.assertIn('DEPTH_VOLUME', check['required_check_ids'])
                self.assertNotIn('independent_observations', check)
                self.assertNotIn('observation_context', check)
            for call in ask.call_args_list:
                if call.args[0][0]['content'] == prompts.REVIEW:
                    labels = [part.get('text', '') for part in call.args[0][1]['content'][1:]]
                    self.assertTrue(any(label.startswith('标准资产 P01 v1') for label in labels))
            self.assertNotIn('reused_from', read_json(folder / 'check_K01.json'))
            self.assertTrue((folder / 'technical_check.json').exists())
            # Segment 输入协议变化不会改写规划或资产候选。
            self.assertTrue(all(len(project.data['tasks'][name]['attempts']) == 1
                for name in ['plans/story', 'plans/visual', 'plans/storyboard', 'plans/frames/S01', 'assets/P01']))

    def test_failed_new_protocol_review_remains_fail_and_cannot_recheck_again(self):
        fake = FakeModel()
        def ask(messages, **kwargs):
            report = fake(messages)
            if report.get('target_id') == 'S01_K01' and report.get('target_version') == 2:
                report['status'] = 'FAIL'
                row = next(row for row in report['checks'] if row['requirement_id'] == 'DEPTH_VOLUME')
                row.update(result='FAIL', evidence='实际体积不足')
            return report
        with Project(self.root, self.config) as project, patch.object(model, 'ask', side_effect=ask), \
                patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub), \
                patch.object(rendering, 'inspect_video') as technical:
            refs = self.seed(project)
            source = self.old_candidate(project, refs)
            old_failure = (source.parent / 'failure.json').read_bytes()
            with self.assertRaisesRegex(Blocked, '不能重复协议复验'):
                drawing.shot(project, BOARD['shots'][0], frames('S01'), refs)
            folder = project.root / 'segments/S01/v002'
            self.assertEqual(read_json(folder / 'failure.json')['status'], 'FAIL')
            self.assertEqual((source.parent / 'failure.json').read_bytes(), old_failure)
            self.assertFalse((folder / 'result.json').exists())
            self.assertFalse((project.root / 'final.mp4').exists())
            technical.assert_not_called()

    def test_recheck_bad_hash_or_conflicting_source_rejected_before_model(self):
        for fault in ['hash', 'local_source', 'local_patch']:
            with self.subTest(fault=fault), Project(self.root / fault, self.config) as project, \
                    patch.object(model, 'ask', side_effect=FakeModel()):
                refs = self.seed(project)
                self.old_candidate(project, refs)
                feedback = project.data['repair_requests']['S01']
                if fault == 'hash':
                    feedback['keyframe_recheck']['source_hash'] = '0' * 64
                else:
                    feedback[fault] = {'path': 'authored.html', 'hash': 'not-used'}
                before = project.data['calls']
                with self.assertRaises((Blocked, ValueError)):
                    drawing.shot(project, BOARD['shots'][0], frames('S01'), refs)
                self.assertEqual(project.data['calls'], before)

    def test_bad_observation_schema_stops_after_one_format_retry(self):
        fake = FakeModel()
        def ask(messages, **kwargs):
            fake(messages)
            return {'observations': [{'frame_index': 99, 'description': 'incorrect frame'}]}
        with Project(self.root, self.config) as project, patch.object(model, 'ask', side_effect=ask):
            image = project.root / 'actual.png'; image.write_bytes(b'actual pixels')
            with self.assertRaisesRegex(Blocked, '字段连续无效'):
                drawing.observe_keyframes(project, project.root / 'candidate', 'keyframe_observation_K01', [(0, image)])
            self.assertEqual(project.data['calls'], 2)
            self.assertTrue(all(system == prompts.OBSERVE for system, _ in fake.calls))
            self.assertTrue((project.root / 'candidate/keyframe_observation_K01_format_retry.schema_error.json').exists())

if __name__ == '__main__':
    unittest.main()
