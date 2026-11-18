"""窗口覆盖、全局硬门槛、恢复复用及一次性时序协议升级。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent import drawing, model, planning, prompts, rendering, temporal, validation
from agent.common import file_hash, read_json, write_json
from agent.store import Blocked, Project, Rejected
from fixtures import BOARD, FakeModel, HTML, REQ, STORY, frames
import test_keyframe_observation
import test_workflow


class WindowSelectionTests(unittest.TestCase):
    def test_all_frames_and_adjacent_samples_are_covered_with_bounded_overlapping_windows(self):
        for count in range(1, 101):
            images = [(index * 3, Path(str(index))) for index in range(count)]
            windows = temporal.sample_windows(images)
            self.assertTrue(all(len(window) <= 12 for window in windows))
            self.assertEqual({row for window in windows for row in window}, set(images))
            pairs = {pair for window in windows for pair in zip(window, window[1:])}
            self.assertTrue(set(zip(images, images[1:])) <= pairs)
            for first, second in zip(windows, windows[1:]):
                self.assertEqual(first[-2:], second[:2])
            self.assertEqual(windows, temporal.sample_windows(images))
        for bad in ([], [(2, Path('x')), (1, Path('y'))], [(1, Path('x')), (1, Path('y'))]):
            with self.assertRaises(ValueError):
                temporal.sample_windows(bad)

    def test_representatives_keep_keys_and_fill_time_gaps_instead_of_dense_sample_indices(self):
        images = [(index, Path(str(index))) for index in sorted(set(range(0, 40)) | set(range(48, 240, 12)) | {239})]
        plan = {'keyframes': [{'frame_index': index} for index in [0, 48, 96, 168, 239]]}
        selected = temporal.representative_images(images, plan)
        indices = [index for index, _ in selected]
        self.assertEqual(len(selected), 16)
        self.assertTrue({0, 48, 96, 168, 239} <= set(indices))
        self.assertTrue(any(96 < index < 168 for index in indices))
        self.assertTrue(any(0 < index < 48 for index in indices))
        self.assertTrue(set(selected) <= set(images))
        self.assertEqual(indices, sorted(set(indices)))
        with self.assertRaises(ValueError):
            temporal.representative_images(images, {'keyframes': [{'frame_index': 999}]})
        with self.assertRaises(ValueError):
            temporal.representative_images(images, {'keyframes': [{'frame_index': i} for i in range(17)]})

    def test_local_plan_keeps_nearest_outside_anchors_without_claiming_actual_images(self):
        plan = frames('S01')
        plan['keyframes'] = [{'frame_id': str(i), 'frame_index': i} for i in [0, 48, 96, 168, 239]]
        plan['intervals'] = [{'start_frame': 0, 'end_frame': 96}, {'start_frame': 96, 'end_frame': 240}]
        original = copy.deepcopy(plan)
        selected = temporal.window_plan(plan, 108, 144)
        self.assertEqual([row['frame_index'] for row in selected['keyframes']], [96, 168])
        self.assertEqual(selected['intervals'], [plan['intervals'][1]])
        self.assertEqual(plan, original)


class WindowReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'project'
        self.config = {'story': STORY, 'fps': 24}
        self.spec = copy.deepcopy(BOARD['shots'][0])
        self.spec['end_frame'] = 240
        self.plan = frames('S01')
        key = self.plan['keyframes'][0]
        self.plan['keyframes'] = [dict(copy.deepcopy(key), frame_id=f'K{number:02d}', frame_index=index)
                                  for number, index in enumerate([0, 48, 96, 168, 239], 1)]
        self.plan['intervals'][-1]['end_frame'] = 240
        self.samples = sorted(set(range(0, 240, 4)) | {1, 2, 238, 239})

    def contexts(self, fake):
        return [data for system, data in fake.calls if system == prompts.REVIEW and data.get('target_id') == 'S01']

    def test_all_windows_then_global_keep_requirements_and_bind_actual_evidence(self):
        fake = FakeModel()
        with Project(self.root, self.config) as project, \
                patch.object(model, 'ask', side_effect=fake), \
                patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub), \
                patch.object(rendering, 'inspect_video', return_value={'passed': True}), \
                patch.object(validation, 'sample_indices', return_value=self.samples):
            refs = test_keyframe_observation.KeyframeObservationTests().seed(project)
            result = drawing.shot(project, self.spec, self.plan, refs)
            contexts = self.contexts(fake)
            self.assertEqual([row['temporal_review_stage'] for row in contexts], ['window'] * 7 + ['global'])
            union, pairs = set(), set()
            for context in contexts[:-1]:
                indices = [row['frame_index'] for row in context['sample_timing']]
                union.update(indices)
                pairs.update(zip(indices, indices[1:]))
                self.assertLessEqual(len(indices), 12)
                self.assertEqual(indices, context['temporal_window']['actual_frame_indices'])
                self.assertIn('不得豁免本区间适用的硬要求', context['verification_scope'])
                self.assertIn('窗外最近前后锚点只是计划文本', context['verification_scope'])
                self.assertIn('标准资产只用于身份与材质参照', context['verification_scope'])
            self.assertEqual(union, set(self.samples))
            self.assertTrue(set(zip(self.samples, self.samples[1:])) <= pairs)
            global_context = contexts[-1]
            self.assertLessEqual(global_context['actual_full_frame_count'], 16)
            self.assertEqual(global_context['frame_plan'], self.plan)
            self.assertIn('仍须完整验证所有运动区间', global_context['verification_scope'])
            original = self.spec['requirements'] + [planning.req('TEMPORAL', '动作完整、接触持续、身份稳定、因果顺序正确，无意外闪烁或漂移')]
            self.assertEqual(global_context['requirements'], planning.depth_requirements(original, 'temporal'))
            self.assertTrue(all(row['hard'] for row in global_context['requirements']))
            for context in contexts[:-1]:
                self.assertEqual(context['original_shot_requirements'], original)
                self.assertEqual(context['requirements'], planning.depth_requirements(temporal.local_requirements(), 'temporal'))
                self.assertTrue(all(row['hard'] for row in context['requirements']))
                self.assertTrue({'R01', 'TEMPORAL'}.isdisjoint(context['required_check_ids']))
                self.assertIn('应分离阶段不得要求提前连接', context['requirements'][2]['text'])
            self.assertNotIn('original_shot_requirements', global_context)
            for context in contexts:
                self.assertEqual(context['temporal_evidence_protocol'], 'windowed_reasoned_v5')
                self.assertEqual(context['fps'], 24)
                self.assertIn('DEPTH_STABILITY', context['required_check_ids'])
            for summary in global_context['temporal_window_evidence']:
                self.assertEqual(set(summary), {'number', 'frame_range', 'frame_indices', 'status', 'image_hashes', 'evidence_signature'})
            directory = self.root / 'segments/S01/v001'
            aggregate = read_json(directory / 'temporal_aggregate.json')
            self.assertEqual(aggregate['status'], 'PASS')
            self.assertEqual(len(aggregate['windows']), 7)
            self.assertEqual([row['frame_index'] for row in aggregate['actual_samples']], self.samples)
            self.assertEqual(aggregate['source']['hash'], file_hash(directory / 'scene.html'))
            for row in aggregate['windows'] + [aggregate['global']]:
                self.assertEqual(row['status'], 'PASS')
                self.assertTrue(row['artifacts'])
                for item in row['artifacts'] + row['images']:
                    self.assertEqual(item['hash'], file_hash(self.root / item['path']))
            self.assertEqual(read_json(directory / 'frame_tasks.json')['temporal_evidence_protocol'], 'windowed_reasoned_v5')
            self.assertEqual(len(result['samples']), 64)
            # 相同像素、要求和窗口不因新候选号改变审核签名，保持恢复预算有界。
            project.invalidate('segments/S01')
            calls = len([row for row in fake.calls if row[0] == prompts.REVIEW])
            drawing.shot(project, self.spec, self.plan, refs)
            self.assertEqual(calls, len([row for row in fake.calls if row[0] == prompts.REVIEW]))
            self.assertIn('reused_from', read_json(self.root / 'segments/S01/v002/temporal_review.json'))

    def test_nonpass_window_or_global_blocks_remaining_reviews_and_encoding(self):
        for stage in ('window', 'global'):
            for status in ('FAIL', 'UNCERTAIN'):
                with self.subTest(stage=stage, status=status):
                    root = self.root / (stage + status)
                    fake = FakeModel()
                    def ask(messages, **kwargs):
                        report = fake(messages)
                        payload = json.loads(messages[1]['content'][0]['text'])
                        if payload.get('temporal_review_stage') == stage:
                            report['status'] = status
                            report['checks'][0].update(result=status, evidence='实际时序证据未通过')
                        return report
                    with Project(root, self.config) as project, \
                            patch.object(model, 'ask', side_effect=ask), \
                            patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub), \
                            patch.object(rendering, 'inspect_video') as inspect, \
                            patch.object(validation, 'sample_indices', return_value=self.samples):
                        refs = test_keyframe_observation.KeyframeObservationTests().seed(project)
                        project.task_attempt_limits['segments/S01'] = 1
                        with self.assertRaises(Blocked):
                            drawing.shot(project, self.spec, self.plan, refs)
                        contexts = self.contexts(fake)
                        self.assertEqual(len(contexts), 1 if stage == 'window' else 8)
                        self.assertEqual(contexts[-1]['temporal_review_stage'], stage)
                        aggregate = read_json(root / 'segments/S01/v001/temporal_aggregate.json')
                        self.assertEqual(aggregate['status'], status)
                        self.assertEqual(aggregate['failed_review']['status'], status)
                        self.assertFalse((root / 'segments/S01/v001/result.json').exists())
                        self.assertFalse((root / 'segments/S01/v001/preview/video.mp4').exists())
                        self.assertFalse((root / 'final.mp4').exists())
                        inspect.assert_not_called()

    def test_nonpass_return_cannot_be_aggregated_as_pass_even_if_review_fails_to_raise(self):
        with Project(self.root, self.config) as project:
            directory = self.root / 'candidate'
            directory.mkdir()
            (directory / 'scene.html').write_text(HTML)
            image = directory / 'frame.png'
            image.write_bytes(bytes.fromhex('89504e470d0a1a0a0000000d494844520000000200000002080600000072b60d240000001149444154789c63105030f80fc20c3006002bb4057d5db26f420000000049454e44ae426082'))
            plan = frames('S01')
            for key in plan['keyframes']:
                key['frame_index'] = 0
            with patch.object(temporal, 'review', return_value={'status': 'UNCERTAIN', 'checks': []}) as review:
                with self.assertRaises(Rejected):
                    temporal.review_windows(project, directory, 'S01', 1, REQ,
                                            {'frame_plan': plan, 'fps': 24}, [], [(0, image)])
                review.assert_called_once()
            self.assertEqual(read_json(directory / 'temporal_aggregate.json')['status'], 'UNCERTAIN')

    def old_candidate(self, project, protocol='sample_timing_v1'):
        def action(directory, *_):
            (directory / 'scene.html').write_text(HTML)
            write_json(directory / 'frame_tasks.json', {'temporal_evidence_protocol': protocol})
            raise Rejected({'status': 'FAIL', 'checks': [], 'repair': {'scope': 'shot', 'instruction': '旧审核'}})
        with self.assertRaises(Rejected):
            project.task('segments/S01', {'temporal_evidence_protocol': protocol},
                         ['plans/storyboard', 'plans/frames/S01', 'assets/P01'], action, retry=False)
        source = project.root / 'segments/S01/v001/scene.html'
        project.data['repair_requests']['S01'] = {'temporal_recheck': {'source_hash': file_hash(source)}}
        return source

    def test_one_time_recheck_keeps_source_and_old_failure_and_rejects_further_upgrade(self):
        for old_protocol in ('sample_timing_v1', 'windowed_v2', 'windowed_local_v3', 'windowed_comparison_v4'):
            fake = FakeModel()
            with self.subTest(old_protocol=old_protocol), Project(self.root / old_protocol, self.config) as project, \
                    patch.object(model, 'ask', side_effect=fake), \
                    patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub), \
                    patch.object(rendering, 'inspect_video', return_value={'passed': True}), \
                    patch.object(validation, 'sample_indices', return_value=self.samples):
                refs = test_keyframe_observation.KeyframeObservationTests().seed(project)
                source = self.old_candidate(project, old_protocol)
                old = {path: file_hash(path) for path in source.parent.rglob('*') if path.is_file()}
                result = drawing.shot(project, self.spec, self.plan, refs)
                self.assertEqual(result['version'], 2)
                directory = project.root / 'segments/S01/v002'
                self.assertEqual(file_hash(source), file_hash(directory / 'scene.html'))
                self.assertEqual(old, {path: file_hash(path) for path in old})
                self.assertEqual(read_json(source.parent / 'failure.json')['status'], 'FAIL')
                proof = read_json(directory / 'temporal_recheck_provenance.json')
                self.assertEqual(proof['source_hash'], file_hash(source))
                self.assertEqual(proof['from_protocol'], old_protocol)
                self.assertEqual(proof['to_protocol'], 'windowed_reasoned_v5')
                self.assertFalse(any(system in (prompts.PATCH, prompts.CANVAS) for system, _ in fake.calls))
                project.invalidate('segments/S01')
                calls = project.data['calls']
                with self.assertRaisesRegex(Blocked, '不能重复协议复验'):
                    drawing.shot(project, self.spec, self.plan, refs)
                self.assertEqual(project.data['calls'], calls)

    def test_recheck_rejects_source_mismatch_other_protocol_and_conflicting_mutations(self):
        for fault in ('hash', 'windowed_reasoned_v5', 'missing_protocol', 'local_patch', 'local_source', 'keyframe_recheck'):
            with self.subTest(fault=fault), Project(self.root / fault, self.config) as project, \
                    patch.object(model, 'ask', side_effect=FakeModel()):
                refs = test_keyframe_observation.KeyframeObservationTests().seed(project)
                self.old_candidate(project, fault if fault in ('windowed_reasoned_v5', 'missing_protocol') else 'sample_timing_v1')
                feedback = project.data['repair_requests']['S01']
                if fault == 'hash':
                    feedback['temporal_recheck']['source_hash'] = '0' * 64
                elif fault in ('local_patch', 'local_source', 'keyframe_recheck'):
                    feedback[fault] = {}
                calls = project.data['calls']
                with self.assertRaises((Blocked, ValueError)):
                    drawing.shot(project, self.spec, self.plan, refs)
                self.assertEqual(project.data['calls'], calls)

    def test_old_temporal_pass_cannot_cross_protocol_signature(self):
        fake = FakeModel()
        with Project(self.root, self.config) as project, patch.object(model, 'ask', side_effect=fake):
            path = self.root / 'frame.png'
            path.write_bytes(b'actual frame')
            images = [('实际画面 frame_index=0', path)]
            for version, protocol in enumerate(('sample_timing_v1', 'windowed_v2', 'windowed_local_v3', 'windowed_comparison_v4', 'windowed_reasoned_v5'), 1):
                result = planning.review(project, self.root / f'segments/S01/v{version:03d}',
                                         'temporal_review', 'S01', version, REQ,
                                         {'temporal_evidence_protocol': protocol}, images, [0])
                self.assertNotIn('reused_from', result)
            self.assertEqual(len(fake.calls), 5)


if __name__ == '__main__':
    unittest.main()
