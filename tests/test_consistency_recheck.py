"""切换一致性配置只重验已有源码，保留失败历史、累计预算和正式门禁。"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent import consistency, drawing, model, prompts, rendering
from agent.common import file_hash, read_json
from agent.store import Blocked, Project
from fixtures import BOARD, FakeModel, HTML, STORY, frames
import test_local_shot_source
import test_workflow


class ConsistencyRecheckTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / 'project'
        self.config = {'story': STORY, 'fps': 2}

    def exercise_migration(self, second_status=None):
        fake = FakeModel()

        def ask(messages, **kwargs):
            report = fake(messages)
            payload = json.loads(messages[1]['content'][0]['text'])
            if payload.get('temporal_review_stage') == 'window':
                status = 'FAIL' if payload['target_version'] == 1 else second_status
                if status:
                    report['status'] = status
                    report['checks'][0].update(result=status, evidence='真实帧动作未通过')
                    report['repair'].update(scope='interval', shot_ids=['S01'], instruction='原始失败保留')
            return report

        with patch.object(model, 'ask', side_effect=ask), \
                patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub), \
                patch.object(rendering, 'inspect_video', return_value={'passed': True}) as inspect:
            with Project(self.root, self.config, review_strictness='strict',
                         max_calls=40, max_seconds=90) as project:
                refs, _ = test_local_shot_source.LocalShotSourceTests().seed(project)
                project.task_attempt_limits['segments/S01'] = 1
                with self.assertRaises(Blocked):
                    drawing.shot(project, BOARD['shots'][0], frames('S01'), refs)
            old = self.root / 'segments/S01/v001'
            original_files = {path: file_hash(path) for path in old.rglob('*') if path.is_file()}
            manifest = read_json(self.root / 'manifest.json')
            calls_before = manifest['calls']
            seconds_before = manifest['active_seconds']
            self.assertEqual(manifest['tasks']['segments/S01']['attempts'][0]['status'], 'FAIL')
            self.assertEqual(read_json(old / 'failure.json')['status'], 'FAIL')
            inspect.assert_not_called()
            previous_calls = len(fake.calls)

            with Project(self.root, self.config, resume=True, review_strictness='balanced',
                         task_attempt_limits={'segments/S01': 2}, max_calls=40, max_seconds=90) as project:
                self.assertEqual(project.data['calls'], calls_before)
                self.assertGreaterEqual(project.data['active_seconds'], seconds_before)
                self.assertNotIn('local_source', project.data['repair_requests'].get('S01', {}))
                self.assertIn('local_source', project.data['policy_history'][-1]['repair_requests']['S01'])
                if second_status:
                    with self.assertRaises(Blocked):
                        drawing.shot(project, BOARD['shots'][0], frames('S01'), refs)
                    with self.assertRaises(Blocked):
                        project.publish(self.root / 'segments/S01/v002/preview/video.mp4')
                else:
                    result = drawing.shot(project, BOARD['shots'][0], frames('S01'), refs)
                    self.assertEqual(result['version'], 2)
                self.assertGreater(project.data['calls'], calls_before)
                self.assertEqual(project.data['calls'], len(fake.calls))
                attempts = project.data['tasks']['segments/S01']['attempts']
                self.assertEqual([row['version'] for row in attempts], [1, 2])
                self.assertEqual([row['status'] for row in attempts], ['FAIL', 'FAIL' if second_status else 'PASS'])
                self.assertEqual(attempts[0]['review_policy']['name'], 'strict')
                self.assertEqual(attempts[1]['review_policy']['name'], 'balanced')
                self.assertEqual(project.data['limits']['max_calls'], 40)
                self.assertTrue(all(len(node['attempts']) == 1 for task, node in project.data['tasks'].items()
                                    if task.startswith(('plans/', 'assets/'))))

            current = self.root / 'segments/S01/v002'
            self.assertEqual(original_files, {path: file_hash(path) for path in original_files})
            self.assertEqual(file_hash(old / 'scene.html'), file_hash(current / 'scene.html'))
            self.assertEqual((current / 'scene.html').read_text(), HTML)
            self.assertFalse((current / 'generated.json').exists())
            self.assertFalse((self.root / 'final.mp4').exists())
            proof = read_json(current / 'consistency_recheck_provenance.json')
            self.assertEqual(proof['source_hash'], file_hash(old / 'scene.html'))
            self.assertEqual(proof['source_version'], 1)
            self.assertEqual(proof['from_review_policy'], consistency.review_policy('strict'))
            self.assertEqual(proof['to_review_policy'], consistency.review_policy('balanced'))
            self.assertTrue(proof['requires_full_review'])
            tasks = read_json(current / 'frame_tasks.json')
            self.assertEqual(tasks['consistency_policy'], consistency.review_policy('balanced'))
            added = fake.calls[previous_calls:]
            self.assertFalse(any(system in (prompts.CANVAS, prompts.PATCH, prompts.FRAME_PLAN) for system, _ in added))
            reviews = [payload for system, payload in added if system == prompts.REVIEW]
            self.assertTrue(reviews)
            self.assertTrue(all(payload['consistency_policy'] == consistency.review_policy('balanced') for payload in reviews))
            self.assertTrue(all('DEPTH_VOLUME' in payload['required_check_ids'] for payload in reviews))
            self.assertTrue(all(row['hard'] for payload in reviews for row in payload['requirements']))
            self.assertTrue(any(payload.get('temporal_review_stage') == 'window' for payload in reviews))
            if second_status:
                self.assertEqual(read_json(current / 'failure.json')['status'], second_status)
                self.assertEqual(read_json(current / 'temporal_aggregate.json')['status'], second_status)
                self.assertFalse((current / 'temporal_review.request.json').exists())
                self.assertFalse((current / 'result.json').exists())
                self.assertFalse((current / 'preview/video.mp4').exists())
                inspect.assert_not_called()
            else:
                self.assertTrue(any(payload.get('temporal_review_stage') == 'global' for payload in reviews))
                self.assertEqual(read_json(current / 'temporal_aggregate.json')['status'], 'PASS')
                self.assertTrue((current / 'preview/video.mp4').exists())
                inspect.assert_called_once()

    def test_strict_failure_resumes_as_balanced_by_rechecking_identical_source(self):
        self.exercise_migration()

    def test_policy_change_cannot_turn_a_new_failure_into_pass_or_publish(self):
        self.exercise_migration(second_status='FAIL')

    def test_render_mode_change_clones_source_but_unchanged_settings_do_not_request_clone(self):
        self.root.mkdir()
        source = self.root / 'old/scene.html'
        source.parent.mkdir(); source.write_text(HTML)
        project = SimpleNamespace(root=self.root, review_strictness='balanced', render_consistency='exact')
        previous = {'version': 3, 'review_policy': consistency.review_policy('balanced'),
                    'render_consistency': 'exact'}
        self.assertFalse(drawing.changed_review_settings(project, None))
        self.assertFalse(drawing.changed_review_settings(project, previous))
        self.assertEqual(drawing.render_options(project), {})
        project.render_consistency = 'tolerant'
        self.assertTrue(drawing.changed_review_settings(project, previous))
        self.assertEqual(drawing.render_options(project), {'consistency': 'tolerant'})
        original_hash = file_hash(source)
        directory = self.root / 'next'
        html = drawing.preserve_for_policy_review(project, directory, source, previous)
        self.assertEqual(html, HTML)
        self.assertEqual(file_hash(source), original_hash)
        proof = read_json(directory / 'consistency_recheck_provenance.json')
        self.assertEqual(proof['source_hash'], original_hash)
        self.assertEqual(proof['from_render_consistency'], 'exact')
        self.assertEqual(proof['to_render_consistency'], 'tolerant')
        self.assertEqual(proof['from_review_policy'], proof['to_review_policy'])
        previous['render_consistency'] = 'tolerant'
        self.assertFalse(drawing.changed_review_settings(project, previous))


if __name__ == '__main__':
    unittest.main()
