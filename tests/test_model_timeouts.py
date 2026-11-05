"""服务端保活不能让无有效输出的请求永久占用同一候选。"""
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent import model
from agent.common import read_json
from agent.store import Blocked, Project
from test_model_logging import Response, event


class ModelTimeoutTests(unittest.TestCase):
    def setUp(self):
        self.clock = [0.0]
        for context in (patch.dict(os.environ, {'LLM_API_KEY': 'test-key',
                'LLM_MODEL': 'test', 'LLM_BASE_URL': 'https://example.invalid',
                'LLM_IDLE_TIMEOUT_SECONDS': '3'}, clear=True),
                patch.object(model, 'load_dotenv'),
                patch.object(model.time, 'monotonic', side_effect=lambda: self.clock[0]),
                patch('sys.stdout', io.StringIO())):
            context.start()
            self.addCleanup(context.stop)

    def stalled(self):
        clock = self.clock

        class Stalled(Response):
            def __iter__(self):
                yield from event({'content': '{"html":'}).encode().splitlines(keepends=True)
                clock[0] += 2
                yield from b': keepalive\n\n'.splitlines(keepends=True)
                yield from event().encode().splitlines(keepends=True)
                clock[0] += 2
                yield b'data: {"choices": [], "usage": {}}\n\n'

        return Stalled('')

    def test_stall_retries_once_preserves_partial_output_and_resumes_same_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'project'
            def run(project):
                return project.task('segments/S01', {}, [],
                    lambda folder, version, previous: project.call(folder, 'generated', []))
            with patch.object(model, 'urlopen', side_effect=[self.stalled(), self.stalled()]) as ask:
                with self.assertRaisesRegex(Blocked, '连续超时'):
                    with Project(root, {'story': 'test'}) as project:
                        run(project)
                self.assertEqual(ask.call_count, 2)
            manifest = read_json(root / 'manifest.json')
            attempts = manifest['tasks']['segments/S01']['attempts']
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0]['status'], 'RETRY')
            self.assertEqual(manifest['calls'], 2)
            traces = list(root.glob('segments/S01/v001/calls/generated/*'))
            self.assertEqual(len(traces), 2)
            for trace in traces:
                self.assertEqual(read_json(trace / 'response.json')['status'], 'TIMEOUT')
                self.assertEqual((trace / 'response.txt').read_text(), '{"html":')
            self.assertFalse((root / 'final.mp4').exists())
            with Project(root, {'story': 'test'}, resume=True) as project, \
                    patch.object(model, 'urlopen', return_value=Response(event({'content': '{"html":"ok"}'}, 'stop'))):
                self.assertEqual(run(project), {'html': 'ok'})
                self.assertEqual(project.data['calls'], 3)
                self.assertEqual(len(project.data['tasks']['segments/S01']['attempts']), 1)

    def test_real_progress_resets_idle_timer_and_finish_does_not_wait_for_close(self):
        clock = self.clock

        class Progress(Response):
            def __iter__(self):
                for delta in ({'reasoning_content': '思考'}, {'reasoning_content': '继续'},
                              {'content': '{"ok":'}, {'content': 'true}'}):
                    clock[0] += 2
                    yield from event(delta).encode().splitlines(keepends=True)
                yield from event(finish='stop').encode().splitlines(keepends=True)
                raise AssertionError('完成后仍在等待服务端关闭')

        with patch.object(model, 'urlopen', return_value=Progress('')):
            self.assertEqual(model.ask([]), {'ok': True})
        self.assertGreater(self.clock[0], 3)

    def test_idle_timeout_must_be_finite_and_positive(self):
        for value in ('0', '-1', 'nan', 'inf'):
            with self.subTest(value=value), patch.dict(os.environ, {'LLM_IDLE_TIMEOUT_SECONDS': value}), \
                    patch.object(model, 'urlopen') as request:
                with self.assertRaisesRegex(ValueError, '有限正数'):
                    model.ask([])
                request.assert_not_called()


if __name__ == '__main__':
    unittest.main()
