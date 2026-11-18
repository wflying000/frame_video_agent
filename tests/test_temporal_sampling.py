"""稀疏时序证据携带实际采样时间，不把跨帧空档伪装为连续帧。"""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent import drawing, model, prompts, rendering, validation
from agent.common import read_json
from agent.store import Blocked, Project
from fixtures import BOARD, FakeModel, STORY, frames
import test_keyframe_observation
import test_workflow


class TemporalSamplingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'project'
        self.config = {'story': STORY, 'fps': 24}
        self.shot = copy.deepcopy(BOARD['shots'][0])
        self.shot['end_frame'] = 48
        self.plan = frames('S01')
        self.plan['keyframes'][-1]['frame_index'] = 47
        self.plan['intervals'][-1]['end_frame'] = 48
        self.samples = [0, 6, 22, 23, 47]

    def test_review_uses_actual_sample_times_and_retains_all_hard_checks(self):
        fake = FakeModel()
        with Project(self.root, self.config) as project, \
                patch.object(model, 'ask', side_effect=fake), \
                patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub), \
                patch.object(rendering, 'inspect_video', return_value={'passed': True}), \
                patch.object(validation, 'sample_indices', return_value=self.samples):
            refs = test_keyframe_observation.KeyframeObservationTests().seed(project)
            with patch.object(project, 'task', wraps=project.task) as task:
                result = drawing.shot(project, self.shot, self.plan, refs)
            self.assertEqual(task.call_args.args[1]['temporal_evidence_protocol'], 'windowed_reasoned_v5')
            payload = next(data for system, data in fake.calls
                           if system == prompts.REVIEW and data.get('target_id') == 'S01')
            self.assertEqual(payload['fps'], 24)
            self.assertEqual(payload['temporal_evidence_protocol'], 'windowed_reasoned_v5')
            manifest = read_json(self.root / 'segments/S01/v001/preview/preview_manifest.json')
            actual = [row['frame_index'] for row in manifest]
            self.assertEqual([row['frame_index'] for row in payload['sample_timing']], actual)
            self.assertEqual([row['time_seconds'] for row in payload['sample_timing']],
                             [index / 24 for index in actual])
            self.assertEqual([row['gap_from_previous_frames'] for row in payload['sample_timing']],
                             [None, 6, 16, 1, 24])
            guidance = payload['temporal_sampling_context']
            self.assertIn('非连续采样不能当作连续帧', guidance)
            self.assertIn('相邻真实帧', guidance)
            self.assertIn('证据不足则UNCERTAIN', guidance)
            self.assertIn('不能根据计划臆测中间正常', guidance)
            self.assertTrue({'LOCAL_STATE', 'LOCAL_CONTINUITY', 'LOCAL_IDENTITY_CONTACT',
                             'DEPTH_VOLUME', 'DEPTH_SPACE', 'DEPTH_STABILITY'}
                            <= set(payload['required_check_ids']))
            self.assertTrue({'R01', 'TEMPORAL'} <=
                            {row['id'] for row in payload['original_shot_requirements']})
            self.assertTrue(all(row['hard'] for row in payload['requirements']))
            task_record = read_json(self.root / 'segments/S01/v001/frame_tasks.json')
            self.assertEqual(task_record['temporal_evidence_protocol'], 'windowed_reasoned_v5')
            self.assertEqual([row['frame_index'] for row in result['samples']], actual)
            self.assertTrue(all(len(project.data['tasks'][name]['attempts']) == 1 for name in
                ['plans/story', 'plans/visual', 'plans/storyboard', 'plans/frames/S01', 'assets/P01']))

    def test_uncertain_temporal_evidence_still_blocks_encoding_and_publication(self):
        fake = FakeModel()
        def ask(messages, **kwargs):
            result = fake(messages)
            if result.get('target_id') == 'S01':
                result['status'] = 'UNCERTAIN'
                check = next(row for row in result['checks'] if row['requirement_id'] == 'LOCAL_CONTINUITY')
                check.update(result='UNCERTAIN', evidence='采样空档不足以判断瞬时跳变')
            return result
        with Project(self.root, self.config) as project, \
                patch.object(model, 'ask', side_effect=ask), \
                patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub), \
                patch.object(rendering, 'inspect_video') as technical, \
                patch.object(validation, 'sample_indices', return_value=self.samples):
            refs = test_keyframe_observation.KeyframeObservationTests().seed(project)
            project.task_attempt_limits['segments/S01'] = 1
            with self.assertRaises(Blocked):
                drawing.shot(project, self.shot, self.plan, refs)
            report = read_json(self.root / 'segments/S01/v001/failure.json')
            self.assertEqual(report['status'], 'UNCERTAIN')
            self.assertEqual(next(row for row in report['checks'] if row['requirement_id'] == 'LOCAL_CONTINUITY')['result'], 'UNCERTAIN')
            technical.assert_not_called()
            self.assertFalse((self.root / 'segments/S01/v001/preview/video.mp4').exists())
            self.assertFalse((self.root / 'segments/S01/v001/result.json').exists())
            self.assertFalse((self.root / 'final.mp4').exists())


if __name__ == '__main__':
    unittest.main()
