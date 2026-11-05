"""服务繁忙须等待、有限重试并保存同一候选，不能当作画面失败。"""
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

from agent import model, store
from agent.common import read_json
from agent.store import Blocked, Project


class ServiceRetryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'project'
        settings = patch.object(model, 'settings', return_value={'model': 'test'})
        settings.start()
        self.addCleanup(settings.stop)

    def run_task(self, project):
        def action(directory, version, previous):
            project.call(directory, 'generated', [{'role': 'user', 'content': '生成'}])
            return project.call(directory, 'review', [{'role': 'user', 'content': '审核'}])
        return project.task('segments/S01', {}, [], action)

    def test_busy_service_pauses_then_resumes_review_without_regenerating(self):
        busy = lambda: HTTPError('https://example.invalid', 503, 'busy', {}, None)
        with patch.object(model, 'ask', side_effect=[{'html': 'source'}, busy(), busy()]), \
                patch.object(store.time, 'sleep') as sleep:
            with self.assertRaisesRegex(Blocked, 'HTTP 503'):
                with Project(self.root, {'story': 'test'}) as project:
                    self.run_task(project)
            sleep.assert_called_once_with(5)
        manifest = read_json(self.root / 'manifest.json')
        self.assertEqual(manifest['calls'], 3)
        self.assertEqual(manifest['tasks']['segments/S01']['attempts'][0]['status'], 'RETRY')
        with Project(self.root, {'story': 'test'}, resume=True) as project, \
                patch.object(model, 'ask', return_value={'status': 'PASS'}) as ask:
            self.assertEqual(self.run_task(project), {'status': 'PASS'})
            ask.assert_called_once()
            self.assertEqual(project.data['calls'], 4)
            self.assertEqual(len(project.data['tasks']['segments/S01']['attempts']), 1)

    def test_retry_after_seconds_is_respected_and_long_wait_pauses(self):
        for seconds in (9, 60):
            with self.subTest(seconds=seconds):
                root = self.root / str(seconds)
                error = HTTPError('https://example.invalid', 429, 'busy', {'Retry-After': str(seconds)}, None)
                with Project(root, {'story': 'test'}) as project, \
                        patch.object(model, 'ask', side_effect=[error, {'status': 'PASS'}]) as ask, \
                        patch.object(store.time, 'sleep') as sleep:
                    if seconds <= 30:
                        self.assertEqual(project.call(root, 'review', []), {'status': 'PASS'})
                        sleep.assert_called_once_with(seconds)
                        self.assertEqual(ask.call_count, 2)
                    else:
                        with self.assertRaisesRegex(Blocked, '稍后'):
                            project.call(root, 'review', [])
                        sleep.assert_not_called()
                        ask.assert_called_once()


if __name__ == '__main__':
    unittest.main()
