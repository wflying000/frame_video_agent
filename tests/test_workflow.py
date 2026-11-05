import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import model, planning, rendering, repair, validation, workflow
from agent.common import file_hash, read_json, write_json
from agent.drawing import patched_html
from agent.store import Blocked, Project, Rejected
from fixtures import BOARD, FACTS, REQ, STORY, FakeModel, frames


class ValidationTests(unittest.TestCase):
    def test_timeline_rejects_gaps_overlaps_missing_events_and_wrong_order(self):
        for field, value in [('start_frame', 5), ('start_frame', 3), ('event_ids', ['E01'])]:
            board = copy.deepcopy(BOARD)
            board['shots'][1][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validation.storyboard(board, FACTS, 8)
        bad = copy.deepcopy(BOARD)
        bad['shots'][0]['event_ids'], bad['shots'][1]['event_ids'] = ['E02'], ['E01']
        with self.assertRaises(ValueError):
            validation.storyboard(bad, FACTS, 8)

    def test_missing_hard_evidence_cannot_pass(self):
        report = {'target_id': 'S01', 'target_version': 1, 'status': 'PASS',
                  'checks': [{'requirement_id': 'R01', 'result': 'UNCERTAIN', 'evidence': '看不清'}]}
        self.assertFalse(validation.review(report, 'S01', 1, REQ))
        report['checks'][0]['result'] = 'NOT_APPLICABLE'
        self.assertFalse(validation.review(report, 'S01', 1, REQ))
        with self.assertRaises(ValueError):
            validation.review(report, 'S01', 2, REQ)
        report['checks'] = []
        with self.assertRaises(ValueError):
            validation.review(report, 'S01', 1, REQ)

    def test_review_modes_distinguish_plans_from_actual_pixels(self):
        from agent import planning
        from unittest.mock import Mock
        report = {'target_id': 'S01', 'target_version': 1, 'status': 'PASS',
                  'checks': [{'requirement_id': 'R01', 'result': 'PASS', 'evidence': '文本方案完整'}]}
        project = Mock()
        project.call.return_value = report
        planning.review(project, Path('/unused'), 'review', 'S01', 1, REQ, {})
        messages = project.call.call_args.args[2]
        payload = json.loads(messages[1]['content'][0]['text'])
        self.assertEqual(payload['evidence_mode'], 'plan')
        with self.assertRaises(ValueError):
            planning.review(project, Path('/unused'), 'review', 'S01', 1, REQ, {}, frame_ids=[0], mode='pixels')

    def test_review_format_retry_does_not_relax_required_checks(self):
        from agent import planning
        from unittest.mock import Mock
        project = Mock()
        correct = {'target_id': 'S01', 'target_version': 1, 'status': 'PASS',
                   'checks': [{'requirement_id': 'R01', 'result': 'PASS', 'evidence': '方案完整'}]}
        malformed = copy.deepcopy(correct)
        malformed['checks'].append({'requirement_id': 'UNREQUESTED', 'result': 'PASS', 'evidence': '多余'})
        project.call.side_effect = [malformed, correct]
        planning.review(project, Path('/unused'), 'review', 'S01', 1, REQ, {})
        self.assertEqual(project.call.call_count, 2)
        request = json.loads(project.call.call_args.args[2][1]['content'][0]['text'])
        self.assertEqual(request['required_check_ids'], ['R01'])
        self.assertIn('format_error', request)

    def test_identical_visual_review_reuses_pass_but_changed_pixels_require_new_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'project'
            report = {'target_id': 'S01_K01', 'target_version': 1, 'status': 'PASS',
                      'checks': [{'requirement_id': 'R01', 'result': 'PASS', 'evidence': '球可见',
                                  'evidence_frames': [0]}]}
            with Project(root, {'story': STORY}) as project:
                image = root / 'frame.png'
                image.write_bytes(b'original pixels')
                first = root / 'segments/S01/v001'
                first.mkdir(parents=True)
                images = [('实际画面 frame_index=0', image)]
                with patch.object(model, 'ask', return_value=report) as ask:
                    planning.review(project, first, 'check_K01', 'S01_K01', 1, REQ, {}, images, [0])
                    result = planning.review(project, root / 'segments/S01/v002', 'check_K01',
                                             'S01_K01', 2, REQ, {}, images, [0])
                    self.assertEqual(ask.call_count, 1)
                    self.assertEqual(result['target_version'], 2)
                    self.assertEqual(result['reused_from']['sha256'], file_hash(first / 'check_K01.json'))
                    image.write_bytes(b'changed pixels')
                    ask.return_value = dict(report, target_version=3)
                    third = root / 'segments/S01/v003'
                    third.mkdir(parents=True)
                    result = planning.review(project, third, 'check_K01', 'S01_K01', 3, REQ, {}, images, [0])
                    self.assertEqual(ask.call_count, 2)
                    self.assertNotIn('reused_from', result)

    def test_visual_zoom_uses_actual_frame_and_declared_coordinate_region(self):
        from agent.repair import audit_evidence
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requirements = [{'id': 'R', 'hard': True, 'text': '爪心位于(0.64,0.43)'}]
            with patch.object(rendering, 'crop_evidence') as crop:
                images = audit_evidence(root, requirements, [('标准资产 C01', root / 'ref.png'),
                    ('实际画面 frame_index=68', root / '68.png'),
                    ('实际画面 frame_index=92', root / '92.png')])
            self.assertEqual(crop.call_count, 2)
            self.assertEqual(crop.call_args_list[1].args[0], root / '92.png')
            self.assertIn('原图', images[1][0])

    def test_keyframe_review_distinguishes_current_frame_from_reference_pose(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'project'
            with Project(root, {'story': STORY}) as project:
                image = root / 'frame.png'
                image.write_bytes(b'pixels')
                folder = root / 'segments/S01/v001'
                folder.mkdir(parents=True)
                report = {'target_id': 'S01_K02', 'target_version': 1, 'status': 'PASS',
                          'checks': [{'requirement_id': 'R01', 'result': 'PASS',
                                      'evidence': '当前帧球可见', 'evidence_frames': [3]}]}
                with patch.object(model, 'ask', return_value=report) as ask:
                    planning.review(project, folder, 'check_K02', 'S01_K02', 1, REQ,
                        {'task': {'frame_index': 3}}, [('标准资产 C01', image),
                        ('实际画面 frame_index=0', image), ('实际画面 frame_index=3', image)], [0, 3])
                context = json.loads(ask.call_args.args[0][1]['content'][0]['text'])
                self.assertEqual(context['pixel_scope']['current_frame_index'], 3)
                self.assertEqual(context['pixel_scope']['comparison_frame_indices'], [0])
                self.assertEqual(context['requirements'], REQ)

    def test_localized_failure_gets_one_detail_review_without_relaxing_requirements(self):
        for status in ['PASS', 'FAIL']:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / 'project'
                with Project(root, {'story': STORY}) as project:
                    image = root / 'frame.png'
                    image.write_bytes(b'original pixels')
                    folder = root / 'segments/S01/v001'
                    folder.mkdir(parents=True)
                    initial = {'target_id': 'S01_K01', 'target_version': 1, 'status': 'FAIL',
                               'checks': [{'requirement_id': 'R01', 'result': 'FAIL',
                                           'evidence': '局部不清楚', 'evidence_frames': [0],
                                           'region_xyxy': [0.2, 0.3, 0.4, 0.5]}]}
                    detail = copy.deepcopy(initial)
                    detail['status'] = detail['checks'][0]['result'] = status
                    def crop(source, output, region):
                        output.parent.mkdir(parents=True, exist_ok=True)
                        output.write_bytes(b'actual region pixels')
                    with patch.object(model, 'ask', side_effect=[initial, detail]) as ask, \
                            patch.object(rendering, 'crop_evidence', side_effect=crop):
                        if status == 'FAIL':
                            with self.assertRaises(Rejected):
                                planning.review(project, folder, 'check_K01', 'S01_K01', 1, REQ,
                                    {}, [('实际画面 frame_index=0', image)], [0])
                        else:
                            result = planning.review(project, folder, 'check_K01', 'S01_K01', 1, REQ,
                                {}, [('实际画面 frame_index=0', image)], [0])
                            self.assertEqual(result['status'], 'PASS')
                        self.assertEqual(ask.call_count, 2)
                        payload = json.loads(ask.call_args.args[0][1]['content'][0]['text'])
                        self.assertEqual(payload['requirements'], REQ)
                        self.assertEqual(payload['required_check_ids'], ['R01'])
                        self.assertEqual(payload['detail_confirmation']['initial_report']['status'], 'FAIL')
                        self.assertEqual(read_json(folder / 'check_K01.json')['status'], 'FAIL')
                        self.assertEqual(read_json(folder / 'check_K01_resolution.json')['status'], status)
                        if status == 'PASS':
                            reused = planning.review(project, root / 'segments/S01/v002',
                                'check_K01', 'S01_K01', 2, REQ, {},
                                [('实际画面 frame_index=0', image)], [0])
                            self.assertEqual(ask.call_count, 2)
                            self.assertIn('detail_resolution', reused['reused_from'])

    def test_temporal_details_are_bounded_and_bound_to_real_frame_numbers(self):
        from agent.repair import temporal_evidence
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = {'keyframes': [{'frame_id': f'K{i}', 'frame_index': i,
                'requirements': [{'id': 'R', 'hard': True, 'text': '接触点(0.64,0.43)'}]} for i in range(10)]}
            actual = [(f'实际画面 frame_index={i}', root / f'{i}.png') for i in range(10)]
            with patch.object(rendering, 'crop_evidence') as crop:
                images = temporal_evidence(root, plan, actual)
            self.assertEqual(len(images), 4)
            self.assertEqual(crop.call_count, 4)
            self.assertTrue(any('frame_index=9' in label for label, _ in images))

    def test_frame_plan_requires_coverage_and_critical_neighborhoods(self):
        plan = frames('S01')
        self.assertEqual(validation.sample_indices(plan, 4), [0, 1, 2, 3])
        plan['keyframes'][0]['covers'] = []
        plan['keyframes'][1]['covers'] = []
        with self.assertRaises(ValueError):
            validation.frame_plan(plan, BOARD['shots'][0])

    def test_motion_review_includes_interval_interior_not_only_keyframes(self):
        plan = {'keyframes': [{'frame_index': 0}, {'frame_index': 32},
                              {'frame_index': 64}, {'frame_index': 119}],
                'critical_frames': [32, 64], 'intervals': [
                    {'start_frame': 0, 'end_frame': 32, 'method': 'motion'},
                    {'start_frame': 32, 'end_frame': 64, 'method': 'motion'},
                    {'start_frame': 64, 'end_frame': 120, 'method': 'hold'}]}
        indices = validation.sample_indices(plan, 120)
        self.assertTrue({40, 48, 55, 92} <= set(indices))
        self.assertEqual(indices, sorted(set(indices)))
        self.assertTrue(all(0 <= i < 120 for i in indices))

    def test_empty_state_frame_gets_hard_check_without_mutating_model_output(self):
        plan = frames('S01')
        key = plan['keyframes'][0]
        key.update(covers=[], requirements=[], state='红球位于画面左侧')
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_json(directory / 'generated.json', plan)
            original_hash = file_hash(directory / 'generated.json')
            result = planning.prepare_frame_plan(directory, plan)
            validation.frame_plan(result, BOARD['shots'][0])
            self.assertEqual(plan['keyframes'][0]['requirements'], [])
            self.assertEqual(result['keyframes'][0]['covers'], [])
            check = result['keyframes'][0]['requirements'][0]
            self.assertTrue(check['hard'])
            self.assertIn(key['state'], check['text'])
            self.assertEqual(file_hash(directory / 'generated.json'), original_hash)
            self.assertEqual(read_json(directory / 'normalized.json'), result)
            self.assertEqual(read_json(directory / 'normalization.json')['changes'][0]['source'],
                             'keyframes[0].state')

    def test_invalid_frame_requirements_still_fail_with_field_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            for requirements in [None, {}, [], [{'id': 'SOFT', 'text': '球可见', 'hard': False}]]:
                plan = frames('S01')
                plan['keyframes'][0]['requirements'] = requirements
                with self.subTest(requirements=requirements):
                    result = planning.prepare_frame_plan(Path(temporary), plan)
                    with self.assertRaisesRegex(ValueError, r'keyframes.K01.requirements'):
                        validation.frame_plan(result, BOARD['shots'][0])
            for value, message in [([], '非空列表'), ({}, '必须是列表')]:
                with self.assertRaisesRegex(ValueError, message):
                    validation.items(value, 'requirements')

    def test_patch_rejects_ambiguous_or_whole_file_replacements(self):
        with self.assertRaises(ValueError):
            patched_html('aaa', {'patches': [{'old': 'a', 'new': 'b'}]})
        with self.assertRaises(ValueError):
            patched_html('aaa', {'patches': [{'old': 'aaa', 'new': 'b'}]})
        self.assertEqual(patched_html('abc', {'patches': [{'old': 'b', 'new': 'd'}]}), 'adc')


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'project'
        self.config = {'story': STORY}

    def action(self, directory, version, previous):
        (directory / 'asset.txt').write_text(f'version {version}')
        return {'version': version}

    def test_hash_change_invalidates_dependents_but_retains_unrelated_output(self):
        with Project(self.root, self.config) as project:
            project.task('a', {}, [], self.action)
            project.task('b', {}, ['a'], self.action)
            project.task('c', {}, [], self.action)
            (self.root / 'a/v001/asset.txt').write_text('changed')
            project.refresh()
            self.assertEqual(project.data['tasks']['a']['status'], 'STALE')
            self.assertEqual(project.data['tasks']['b']['status'], 'STALE')
            self.assertEqual(project.data['tasks']['c']['status'], 'PASS')
            with self.assertRaises(Blocked):
                project.publish(self.root / 'a/v001/asset.txt')

    def test_resume_reuses_passed_tasks_and_rejects_changed_story(self):
        with Project(self.root, self.config) as project:
            project.task('a', {}, [], self.action)
        with Project(self.root, self.config, resume=True) as project:
            result = project.task('a', {}, [], lambda *args: self.fail('passed task was rerun'))
            self.assertEqual(result['version'], 1)
        with self.assertRaises(ValueError):
            with Project(self.root, {'story': 'different'}, resume=True):
                pass

    def test_budget_survives_resume(self):
        with Project(self.root, self.config, max_calls=1) as project:
            with patch.object(model, 'ask', return_value={'ok': True}):
                project.call(self.root, 'first', [])
        with Project(self.root, self.config, resume=True, max_calls=1) as project:
            self.assertEqual(project.call(self.root, 'first', []), {'ok': True})
            with self.assertRaises(Blocked):
                project.call(self.root, 'second', [{'role': 'user', 'content': 'new'}])

    def test_explicit_candidate_limit_preserves_history_and_passed_tasks(self):
        def fail(*args):
            raise rendering.RenderError('重画不一致')
        with Project(self.root, self.config) as project:
            project.task('plans/story', {}, [], self.action)
            with self.assertRaises(Blocked):
                project.task('assets/L01', {}, ['plans/story'], fail)
        failure = self.root / 'assets/L01/v003/failure.json'
        before = file_hash(failure)
        with Project(self.root, self.config, resume=True) as project:
            with self.assertRaises(Blocked):
                project.task('assets/L01', {}, ['plans/story'], self.action)
        with Project(self.root, self.config, resume=True,
                     task_attempt_limits={'assets/L01': 4}) as project:
            project.task('plans/story', {}, [], lambda *args: self.fail('不应重做已通过任务'))
            result = project.task('assets/L01', {}, ['plans/story'], self.action)
            self.assertEqual(result['version'], 4)
            self.assertEqual([a['status'] for a in project.data['tasks']['assets/L01']['attempts']],
                             ['FAIL', 'FAIL', 'FAIL', 'PASS'])
            with self.assertRaises(Blocked):
                project.task('assets/OTHER', {}, [], fail)
            self.assertEqual(len(project.data['tasks']['assets/OTHER']['attempts']), 3)
        self.assertEqual(file_hash(failure), before)
        with Project(self.root, self.config, resume=True) as project:
            self.assertEqual(project.task_attempt_limits, {'assets/L01': 4})
            with self.assertRaises(Blocked):
                project.task('assets/L01', {'changed': True}, ['plans/story'], self.action)

    def test_candidate_limit_rejects_unknown_task_and_requires_resume(self):
        with Project(self.root, self.config):
            pass
        before = file_hash(self.root / 'manifest.json')
        with self.assertRaisesRegex(ValueError, '不存在'):
            with Project(self.root, self.config, resume=True, task_attempt_limits={'typo': 4}):
                pass
        self.assertEqual(file_hash(self.root / 'manifest.json'), before)
        self.assertFalse((self.root / '.run.lock').exists())
        with self.assertRaisesRegex(ValueError, '--resume'):
            Project(self.root, self.config, task_attempt_limits={'a': 4})
        with self.assertRaisesRegex(ValueError, '正整数'):
            Project(self.root, self.config, resume=True, task_attempt_limits={'a': 0})

    def test_second_writer_cannot_open_same_project(self):
        with Project(self.root, self.config):
            with self.assertRaises(Blocked):
                with Project(self.root, self.config, resume=True):
                    pass


class RepairTests(unittest.TestCase):
    def test_regression_context_recovers_code_from_last_passed_keyframe(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempts = [{'version': i, 'path': f'segments/S01/v{i:03d}'} for i in [1, 2]]
            for attempt in attempts:
                (root / attempt['path']).mkdir(parents=True)
            old = root / attempts[0]['path']
            current = root / attempts[1]['path'] / 'scene.html'
            (old / 'scene.html').write_text('scaleBody();\ndrawPaw();\n')
            current.write_text('drawPaw();\n')
            write_json(old / 'check_K01.json', {'status': 'PASS'})
            attempts[1]['report'] = {'target_id': 'S01_K01', 'status': 'FAIL'}
            project = SimpleNamespace(root=root, data={'tasks': {'segments/S01': {'attempts': attempts}}})
            result = repair.regression_context(project, BOARD['shots'][0], frames('S01'), attempts[1], current)
            self.assertEqual(result['last_passed_version'], 1)
            self.assertIn('-scaleBody();', result['changes_since_pass'])
            self.assertEqual(result['passed_hash'], file_hash(old / 'scene.html'))

    def test_legacy_local_failure_is_rechecked_once_before_redrawing(self):
        previous = {'report': {'status': 'FAIL', 'checks': [{'result': 'FAIL',
                    'evidence_frames': [32], 'region_xyxy': [0.4, 0.5, 0.6, 0.7]}],
                    'review_images': [{'label': '实际画面 frame_index=32'}]}}
        self.assertTrue(repair.needs_detail_confirmation(previous))
        previous['report']['detail_confirmation'] = {'status': 'FAIL'}
        self.assertFalse(repair.needs_detail_confirmation(previous))
        previous['report'].pop('detail_confirmation')
        previous['report']['checks'][0]['evidence_frames'] = [99]
        self.assertFalse(repair.needs_detail_confirmation(previous))

    def test_repair_history_retains_earlier_spatial_failure_after_position_failure(self):
        from types import SimpleNamespace
        attempts = [{'version': 6, 'report': {'target_id': 'S03', 'checks': [
                        {'requirement_id': 'R09', 'result': 'FAIL', 'evidence': '没有穿过开口'}]}},
                    {'version': 7, 'report': {'target_id': 'S03_K03', 'checks': [
                        {'requirement_id': 'K_R09', 'result': 'FAIL', 'evidence': '偏离指定位置'}]}},
                    {'version': 8}]
        project = SimpleNamespace(data={'tasks': {'segments/S03': {'attempts': attempts}}})
        result = repair.history(project, 'segments/S03', attempts[1])
        self.assertEqual([row['version'] for row in result], [6, 7])
        self.assertIn('开口', result[0]['checks'][0]['evidence'])
        self.assertIn('位置', result[1]['checks'][0]['evidence'])

    def test_late_repair_uses_actual_source_and_effective_plan(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempts = [{'version': v, 'path': f'segments/S01/v{v:03d}'} for v in range(1, 7)]
            base = root / attempts[4]['path']
            base.mkdir(parents=True)
            html = '<html>already patched scene</html>'
            (base / 'scene.html').write_text(html)
            effective = frames('S01')
            effective['keyframes'][1]['requirements'] = [
                {'id': 'CONTACT', 'hard': True, 'text': '接触区必须可见'}]
            write_json(base / 'frame_tasks.json', {'frame_plan': effective})
            write_json(base / 'generated.json', {'patches': [{'old': 'a', 'new': 'b'}]})
            project = SimpleNamespace(root=root, data={'tasks': {'segments/S01': {'attempts': attempts}}})
            # v006 补丁失败、没有 scene.html 时仍从最近有效代码 v005 修，而非重画或读 patch JSON。
            source, plan, strategy = repair.patch_context(project, root / 'segments/S01/v007',
                BOARD['shots'][0], frames('S01'), attempts[5])
            self.assertEqual(source.read_text(), html)
            self.assertEqual(plan, effective)
            self.assertEqual(strategy['strategy'], 'patch_existing_scene')
            self.assertEqual(strategy['base_version'], 5)

    def test_evidence_crops_only_failed_regions_and_keeps_full_image(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'frame.png'
            source.write_bytes(b'frame')
            report = {'checks': [{'result': 'FAIL', 'region_xyxy': [0.6, 0.3, 0.8, 0.5]},
                                 {'result': 'PASS', 'region_xyxy': [0, 0, 1, 1]}],
                      'review_images': [{'label': '实际画面 frame_index=92', 'path': 'frame.png',
                                         'hash': file_hash(source)},
                                        {'label': '测量辅助：实际画面 frame_index=92，局部放大图',
                                         'path': 'old_zoom.png', 'hash': 'not-an-original-frame'}]}
            with patch.object(rendering, 'crop_evidence') as crop:
                images = repair.evidence(SimpleNamespace(root=root), root / 'repair', [report, report])
            self.assertEqual(len(images), 2)
            self.assertEqual(images[0][1], source)
            self.assertIn('frame_index=92', images[1][0])
            self.assertEqual(crop.call_count, 1)
            self.assertNotIn('old_zoom.png', [str(path) for _, path in images])
            source.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, '过期'):
                repair.evidence(SimpleNamespace(root=root), root / 'repair', [report])

    def test_large_temporal_failure_keeps_evidence_request_bounded(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'frame.png'
            source.write_bytes(b'frame')
            images = []
            for i in range(30):
                path = root / f'frame_{i}.png'
                path.write_bytes(b'frame')
                images.append({'label': f'实际画面 frame_index={i}', 'path': path.name, 'hash': file_hash(path)})
            report = {'checks': [{'result': 'FAIL', 'region_xyxy': [0.6, 0.3, 0.8, 0.5],
                                  'evidence_frames': [12, 13, 14, 15, 16]}], 'review_images': images}
            with patch.object(rendering, 'crop_evidence') as crop:
                selected = repair.evidence(SimpleNamespace(root=root), root / 'repair', [report])
            self.assertLessEqual(crop.call_count, 6)
            self.assertLessEqual(len(selected), 12)
            self.assertTrue(any('frame_index=12' in label for label, _ in selected))
            self.assertTrue(any('frame_index=29' in label for label, _ in selected))


class PacingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'project'
        self.config = {'story': STORY, 'seconds': 4, 'fps': 2}

    @staticmethod
    def seed(project, board):
        project.task('plans/storyboard', board, [], lambda *args: board)
        project.task('assets/P01', {}, [], lambda *args: {'asset_id': 'P01'})

    def test_text_pacing_keeps_full_timeline_without_rendering_and_resumes(self):
        board = copy.deepcopy(BOARD)
        board['shots'][0]['action'] = '球缓慢向右移动。' * 40 + '最后停在右侧。'
        fake = FakeModel()
        with Project(self.root, self.config) as project, \
             patch.object(model, 'ask', side_effect=fake) as ask, \
             patch.object(rendering, 'render_job', side_effect=AssertionError('文字检查不应渲染')):
            self.seed(project, board)
            text_path = workflow.check_pacing(project, board, {'P01': {}})
            text = text_path.read_text(encoding='utf-8')
            self.assertIn(board['shots'][0]['action'], text)
            self.assertIn('S01 | 0.000–2.000 秒 | 时长 2.000 秒 | 帧 [0, 4)', text)
            self.assertIn('S02 | 2.000–4.000 秒 | 时长 2.000 秒 | 帧 [4, 8)', text)
            self.assertIn('R01（硬性）', text)
            content = ask.call_args.args[0][1]['content']
            self.assertEqual([row['type'] for row in content], ['text'])
            payload = json.loads(content[0]['text'])
            self.assertEqual(payload['evidence_mode'], 'plan')
            self.assertEqual(payload['timeline_text'], text)
            self.assertEqual(payload['storyboard'], board)
            self.assertEqual(project.data['calls'], 1)
            self.assertTrue(project.artifacts_valid(project.data['tasks']['plans/animatic']))
        self.assertFalse(list(self.root.rglob('*.png')))
        self.assertFalse(list(self.root.rglob('*.mp4')))
        self.assertFalse(list(self.root.rglob('*.html')))
        text_path.unlink()  # 汇总文本可从同一份已审核分镜重新导出。
        with Project(self.root, self.config, resume=True) as project, \
             patch.object(model, 'ask', side_effect=AssertionError('不应重复调用模型')):
            workflow.check_pacing(project, board, {'P01': {}})
            self.assertEqual(project.data['calls'], 1)
            self.assertEqual(len(project.data['tasks']['plans/animatic']['attempts']), 1)
        self.assertEqual(text_path.read_text(encoding='utf-8'), text)

    def legacy_action(self, project):
        def action(directory, version, previous):
            render = directory / 'render'
            render.mkdir()
            (render / 'preview_00.png').write_bytes(b'legacy text card')
            (render / 'video.mp4').write_bytes(b'legacy animatic')
            planning.review(project, directory, 'review', 'animatic', version,
                            [planning.req('PACING', '镜头顺序、时长和节奏可实现')],
                            {'storyboard': BOARD, 'type': '粗略节奏预览'},
                            [('实际画面 frame_index=0', render / 'preview_00.png')], [0])
            return {'video': str((render / 'video.mp4').relative_to(project.root))}
        return action

    def seed_legacy(self, project):
        self.seed(project, BOARD)
        payload = {'shots': BOARD['shots'], 'fps': 2, 'seconds': 4}
        project.task('plans/animatic', payload, ['plans/storyboard', 'assets/P01'],
                     self.legacy_action(project))
        return copy.deepcopy(project.data['tasks']['plans/animatic'])

    def test_legacy_pass_reuses_review_and_retains_all_historical_artifacts(self):
        with Project(self.root, self.config) as project, patch.object(model, 'ask', side_effect=FakeModel()):
            legacy = self.seed_legacy(project)
            (self.root / 'animatic.mp4').write_bytes(b'legacy root copy')
            calls = project.data['calls']
        with Project(self.root, self.config, resume=True) as project, \
             patch.object(model, 'ask', side_effect=AssertionError('不应重新审核旧版 PASS')), \
             patch.object(rendering, 'render_job', side_effect=AssertionError('不应渲染旧版占位图')):
            path = workflow.check_pacing(project, BOARD, {'P01': {}})
            self.assertTrue(path.exists())
            self.assertEqual(project.data['tasks']['plans/animatic'], legacy)
            self.assertEqual(project.data['calls'], calls)
            self.assertTrue(project.artifacts_valid(legacy))
        self.assertEqual((self.root / 'animatic.mp4').read_bytes(), b'legacy root copy')
        self.assertFalse((self.root / 'plans/animatic/v002').exists())

    def test_changed_legacy_storyboard_requires_a_new_text_review(self):
        with Project(self.root, self.config) as project, patch.object(model, 'ask', side_effect=FakeModel()):
            legacy = self.seed_legacy(project)
        board = copy.deepcopy(BOARD)
        board['shots'][0]['action'] = '向右缓慢移动'
        fake = FakeModel()
        with Project(self.root, self.config, resume=True) as project, \
             patch.object(model, 'ask', side_effect=fake), \
             patch.object(rendering, 'render_job', side_effect=AssertionError('不应渲染占位图')):
            self.seed(project, board)
            workflow.check_pacing(project, board, {'P01': {}})
            self.assertEqual(project.data['tasks']['plans/animatic']['version'], 2)
            self.assertEqual(project.data['calls'], 2)
            self.assertTrue(project.artifacts_valid(legacy))
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0][1]['storyboard'], board)
        self.assertEqual(fake.calls[0][1]['evidence_mode'], 'plan')
        self.assertFalse((self.root / 'plans/animatic/v002/render').exists())


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'out'

    @staticmethod
    def render_stub(html, out, seconds, fps, preview=False, frame_count=None, sample_indices=None):
        out.mkdir(parents=True, exist_ok=True)
        count = frame_count or seconds * fps
        indices = sorted(set(sample_indices or [0, count - 1]))
        rows = []
        for i, frame in enumerate(indices):
            name = f'preview_{i:02d}.png'
            (out / name).write_bytes(b'actual-png-placeholder')
            rows.append({'frame_index': frame, 'path': name})
        write_json(out / 'preview_manifest.json', rows)
        if not preview:
            (out / 'video.mp4').write_bytes(b'video')

    @staticmethod
    def encode_stub(paths, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b'composed-video')

    @staticmethod
    def sample_stub(path, out, indices):
        WorkflowTests.render_stub(path, out, 4, 2, True, 8, indices)
        return rendering.preview_images(out)

    def fake_renderers(self):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(rendering, 'render_job', side_effect=self.render_stub))
        stack.enter_context(patch.object(rendering, 'inspect_video', return_value={'passed': True}))
        stack.enter_context(patch.object(rendering, 'encode_segments', side_effect=self.encode_stub))
        stack.enter_context(patch.object(rendering, 'sample_video', side_effect=self.sample_stub))
        stack.enter_context(patch.object(rendering, 'boundary_changes', return_value={'boundary_change': 0.1}))
        return stack

    def test_keyframe_failure_repairs_only_one_shot_and_resume_uses_no_model(self):
        fake = FakeModel(fail_key=True)
        with patch.object(model, 'ask', side_effect=fake), self.fake_renderers():
            result = workflow.generate(STORY, self.root, 4, 2)
            manifest = read_json(self.root / 'manifest.json')
            self.assertEqual(manifest['status'], 'COMPLETED')
            self.assertEqual(len(manifest['tasks']['segments/S01']['attempts']), 1)
            self.assertEqual(len(manifest['tasks']['segments/S02']['attempts']), 2)
            before = file_hash(result)
            temporal = [payload for _, payload in fake.calls
                        if payload.get('target_id') == 'S01' and 'next_shot_entry' in payload][0]
            self.assertEqual(temporal['next_shot_entry']['shot_id'], 'S02')
            self.assertIn('checks/timeline', temporal['verification_scope'])
            boundary = next(payload for _, payload in fake.calls if payload.get('target_id') == 'S01_S02')
            self.assertEqual(boundary['boundary_frames'], [3, 4])
            self.assertEqual(boundary['required_check_ids'], ['CONTINUITY', 'MOTION_PHASE'])
            self.assertEqual([row['frame_index'] for row in read_json(
                self.root / 'checks/timeline/v001/boundary_1_frames/left/preview_manifest.json')], [1, 2, 3])
            self.assertEqual([row['frame_index'] for row in read_json(
                self.root / 'checks/timeline/v001/boundary_1_frames/right/preview_manifest.json')], [0, 1, 2])
        with patch.object(model, 'ask', side_effect=AssertionError('should reuse')), self.fake_renderers():
            resumed = workflow.generate(STORY, self.root, 4, 2, resume=True)
        self.assertEqual(file_hash(resumed), before)

    def test_final_review_routes_repair_to_target_shot(self):
        with patch.object(model, 'ask', side_effect=FakeModel(fail_final=True)), self.fake_renderers():
            workflow.generate(STORY, self.root, 4, 2)
        manifest = read_json(self.root / 'manifest.json')
        self.assertEqual(manifest['global_repairs'], 1)
        self.assertEqual(len(manifest['tasks']['segments/S01']['attempts']), 1)
        self.assertEqual(len(manifest['tasks']['segments/S02']['attempts']), 2)
        self.assertEqual(len(manifest['tasks']['exports/final']['attempts']), 2)

    def test_explicit_plan_repair_reviews_new_plan_and_existing_pixels_before_redrawing(self):
        from agent import prompts
        with patch.object(model, 'ask', side_effect=FakeModel()), self.fake_renderers():
            workflow.generate(STORY, self.root, 4, 2)
        source_hash = file_hash(self.root / 'segments/S02/v001/scene.html')
        with Project(self.root, read_json(self.root / 'project_config.json'), resume=True) as project:
            project.data['repair_requests']['S02'] = {
                'replan_source_version': 1, 'repair': {'scope': 'plan',
                'instruction': '修订有冲突的进入约束，保留原分镜硬要求并重新审核。'}}
            project.invalidate('segments/S02')
        fake = FakeModel()
        def ask(messages, **kwargs):
            result = copy.deepcopy(fake(messages))
            if messages[0]['content'] == prompts.FRAME_PLAN:
                result['keyframes'][0]['requirements'][0]['text'] = '球的停止状态承接上一镜头末帧'
                self.assertTrue(any(row['type'] == 'image_url' for row in messages[1]['content']))
            payload = json.loads(messages[1]['content'][0]['text'])
            if payload.get('target_id') == 'S02_K01':
                self.assertEqual(payload['entry_reference']['shot_id'], 'S01')
                self.assertTrue(any('上一镜头实际末帧' in row.get('text', '')
                                    for row in messages[1]['content']))
            return result
        with patch.object(model, 'ask', side_effect=ask), self.fake_renderers():
            workflow.generate(STORY, self.root, 4, 2, resume=True)
        folder = self.root / 'segments/S02/v002'
        self.assertEqual(file_hash(folder / 'scene.html'), source_hash)
        self.assertEqual(read_json(folder / 'replan_review.json')['status'], 'PASS')
        self.assertEqual(read_json(folder / 'check_K01.json')['status'], 'PASS')
        self.assertTrue(any(p.get('target_id') == 'S02_K01' for _, p in fake.calls))
        self.assertFalse(any(system == prompts.PATCH for system, _ in fake.calls))

    def test_network_resume_retains_generated_candidate(self):
        fake = FakeModel(network=True)
        with patch.object(model, 'ask', side_effect=fake), self.fake_renderers():
            with self.assertRaises(URLError):
                workflow.generate(STORY, self.root, 4, 2)
            html = self.root / 'segments/S02/v001/scene.html'
            before = file_hash(html)
            calls = read_json(self.root / 'manifest.json')['calls']
            workflow.generate(STORY, self.root, 4, 2, resume=True)
        self.assertEqual(file_hash(html), before)
        manifest = read_json(self.root / 'manifest.json')
        self.assertEqual(len(manifest['tasks']['segments/S02']['attempts']), 1)
        self.assertGreater(manifest['calls'], calls)

    def test_uncertainty_stops_before_drawing_and_never_publishes(self):
        with patch.object(model, 'ask', side_effect=FakeModel(uncertain=True)), self.fake_renderers():
            with self.assertRaises(Blocked):
                workflow.generate(STORY, self.root, 4, 2)
        self.assertFalse((self.root / 'final.mp4').exists())
        self.assertFalse((self.root / 'assets').exists())
        self.assertEqual(read_json(self.root / 'manifest.json')['status'], 'BLOCKED')

    def test_text_pacing_hard_failures_block_shots_and_final_export(self):
        for status in ('FAIL', 'UNCERTAIN', 'NOT_APPLICABLE'):
            fake = FakeModel()
            root = self.root / status
            def ask(messages, **kwargs):
                result = fake(messages)
                if result.get('target_id') == 'animatic':
                    result['status'] = status
                    result['checks'][0]['result'] = status
                return result
            with self.subTest(status=status), patch.object(model, 'ask', side_effect=ask), self.fake_renderers():
                with self.assertRaises(Blocked):
                    workflow.generate(STORY, root, 4, 2)
                manifest = read_json(root / 'manifest.json')
                self.assertEqual(manifest['status'], 'BLOCKED')
                self.assertEqual(len(manifest['tasks']['plans/animatic']['attempts']), 3)
                self.assertFalse((root / 'segments').exists())
                self.assertFalse((root / 'final.mp4').exists())
                self.assertFalse((root / 'plans/pacing.txt').exists())
                self.assertTrue((root / 'plans/animatic/v001/pacing.txt').exists())

    def test_normalized_anchor_is_audited_and_uncertainty_blocks_export(self):
        fake = FakeModel()
        seen = []
        def ask(messages, **kwargs):
            from agent import prompts
            result = fake(messages)
            payload = json.loads(messages[1]['content'][0]['text'])
            if messages[0]['content'] == prompts.FRAME_PLAN:
                self.assertEqual(payload['generation_contract']['frame_index_range'], [0, 3])
                result['keyframes'][0].update(covers=[], requirements=[])
            if messages[0]['content'] == prompts.PATCH:
                result = {'patches': [{'old': 'ctx.resetTransform();', 'new': 'ctx.resetTransform();\n'}]}
            if payload.get('target_id') == 'S01_K01':
                seen.append(payload)
                self.assertEqual(payload['required_check_ids'], ['STATE'])
                self.assertTrue(payload['requirements'][0]['hard'])
                self.assertEqual(payload['evidence_mode'], 'pixels')
                result['status'] = 'UNCERTAIN'
                result['checks'][0]['result'] = 'UNCERTAIN'
            return result
        with patch.object(model, 'ask', side_effect=ask), self.fake_renderers():
            with self.assertRaises(Blocked):
                workflow.generate(STORY, self.root, 4, 2)
        self.assertEqual(len(seen), 5)
        self.assertFalse((self.root / 'final.mp4').exists())
        generated = read_json(self.root / 'plans/frames/S01/v001/generated.json')
        result = read_json(self.root / 'plans/frames/S01/v001/result.json')
        self.assertEqual(generated['keyframes'][0]['requirements'], [])
        self.assertEqual(result['keyframes'][0]['requirements'][0]['id'], 'STATE')

    def test_bad_animation_is_repaired_without_redrawing_other_shots(self):
        def render(*args, **kwargs):
            if 'segments/S02/v001/' in str(args[0]):
                raise rendering.RenderError('frame 函数抛出异常')
            return self.render_stub(*args, **kwargs)
        with patch.object(model, 'ask', side_effect=FakeModel()), self.fake_renderers(), \
             patch.object(rendering, 'render_job', side_effect=render):
            workflow.generate(STORY, self.root, 4, 2)
        tasks = read_json(self.root / 'manifest.json')['tasks']
        self.assertEqual(len(tasks['segments/S01']['attempts']), 1)
        self.assertEqual(len(tasks['segments/S02']['attempts']), 2)

    def test_real_two_shot_render_and_final_decoding(self):
        with patch.object(model, 'ask', side_effect=FakeModel()):
            result = workflow.generate(STORY, self.root, 4, 2)
        self.assertTrue(result.exists())
        report = rendering.inspect_video(result, 8, 2)
        self.assertEqual(report['frame_count'], 8)
        self.assertTrue((self.root / 'plans/pacing.txt').exists())
        self.assertFalse((self.root / 'animatic.mp4').exists())
        self.assertFalse(list((self.root / 'plans/animatic').rglob('*.png')))
        self.assertEqual(read_json(self.root / 'manifest.json')['status'], 'COMPLETED')


if __name__ == '__main__':
    unittest.main()
