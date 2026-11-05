"""模型流式响应、等待日志及失败处理；不调用远程 API。"""
import io
import json
import os
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import model as agent
from agent import common


class Response(io.BytesIO):
    def __init__(self, data, content_type='text/event-stream'):
        super().__init__(data.encode())
        self.headers = {'Content-Type': content_type}


def event(delta=None, finish=None):
    return 'data: ' + json.dumps({'choices': [{
        'delta': delta or {}, 'finish_reason': finish,
    }]}, ensure_ascii=False) + '\n\n'


class ModelLoggingTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            'LLM_API_KEY': 'test-private-key', 'LLM_MODEL': 'test-model',
            'LLM_BASE_URL': 'https://example.invalid',
        }, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.dotenv = patch.object(agent, 'load_dotenv')
        self.dotenv.start()
        self.addCleanup(self.dotenv.stop)

    def test_stream_reassembles_unicode_and_skips_keepalive_and_usage(self):
        data = ': keepalive\n\n' + event({'role': 'assistant'})
        data += event({'reasoning_content': '检查'})
        data += event({'content': '{"plan": ["橘猫'})
        data += event({'content': '睡觉"], "html": "<canvas/>"}'})
        data += event(finish='stop')
        data += 'data: {"choices": [], "usage": {"completion_tokens": 12}}\n\n'
        data += 'data: [DONE]\n\n'
        output = io.StringIO()
        with patch.object(agent, 'urlopen', return_value=Response(data)) as request, \
             patch('sys.stdout', output):
            result = agent.ask([{'role': 'user', 'content': '故事'}])
        self.assertEqual(result['plan'], ['橘猫睡觉'])
        body = json.loads(request.call_args.args[0].data)
        self.assertIs(body['stream'], True)
        self.assertIn('收到首段正文', output.getvalue())
        self.assertIn('JSON 解析成功', output.getvalue())
        self.assertNotIn('test-private-key', output.getvalue())
        agent.load_dotenv.assert_called_once_with(agent.PROJECT_ROOT / '.env', override=False)

    def test_full_output_is_visible_before_stream_finishes(self):
        output = io.StringIO()

        class LiveResponse:
            headers = {'Content-Type': 'text/event-stream'}

            def __iter__(self):
                yield from event({'content': '{"passed":'}).encode().splitlines(keepends=True)
                # 只有接收下一段前已经打印，才是真正的流式显示。
                self_seen = output.getvalue()
                if '{"passed":' not in self_seen:
                    raise AssertionError('首段输出被缓存')
                yield from event({'content': 'true}'}, 'stop').encode().splitlines(keepends=True)
                yield b'data: [DONE]\n\n'

        with patch('sys.stdout', output):
            choice = agent.read_completion(LiveResponse(), {}, True)
        self.assertEqual(json.loads(choice['message']['content']), {'passed': True})

    def test_non_streaming_service_still_works(self):
        os.environ['LLM_EXTRA_BODY'] = '{"stream":false}'
        data = json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': '{"ok":true}'}}]})
        with patch.object(agent, 'urlopen', return_value=Response(data, 'application/json')) as request, \
             patch('sys.stdout', io.StringIO()):
            self.assertEqual(agent.ask([]), {'ok': True})
        self.assertIs(json.loads(request.call_args.args[0].data)['stream'], False)

    def test_incomplete_or_truncated_stream_is_not_accepted(self):
        cases = [
            (event({'content': '{"ok":true}'}), RuntimeError, '提前结束'),
            (event({'content': '{"ok":true}'}, 'length'), ValueError, '截断'),
            ('data: {"error":{"message":"quota exceeded"}}\n\n', RuntimeError, 'quota exceeded'),
        ]
        for data, error_type, message in cases:
            with self.subTest(message=message), \
                 patch.object(agent, 'urlopen', return_value=Response(data)), \
                 patch('sys.stdout', io.StringIO()), \
                 self.assertRaisesRegex(error_type, message):
                agent.ask([])

    def test_http_failure_logs_status_and_redacts_key(self):
        error = HTTPError('https://example.invalid', 401, 'Unauthorized', {},
                          io.BytesIO(b'{"error":"invalid test-private-key"}'))
        output = io.StringIO()
        with patch.object(agent, 'urlopen', side_effect=error), \
             patch('sys.stdout', output), self.assertRaises(HTTPError):
            agent.ask([])
        self.assertIn('HTTP 401', output.getvalue())
        self.assertIn('[REDACTED]', output.getvalue())
        self.assertNotIn('test-private-key', output.getvalue())

    def test_heartbeat_reports_during_wait_and_stops_on_error(self):
        reported = threading.Event()
        with patch.object(common, 'log', side_effect=lambda *args: reported.set()) as log:
            with self.assertRaisesRegex(ValueError, 'failed'):
                with agent.heartbeat('模型', lambda: '等待首段输出', interval=0.01):
                    self.assertTrue(reported.wait(1), '等待时没有进度日志')
                    raise ValueError('failed')
            count = log.call_count
            reported.clear()
            self.assertFalse(reported.wait(0.04), '请求结束后仍在打印等待日志')
            self.assertEqual(log.call_count, count)


if __name__ == '__main__':
    unittest.main()
