"""格式错误保留原文、有限重试与恢复；模拟网络，实际执行客户端和调度器。"""
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent import model
from agent.common import digest, read_json, write_json
from agent.store import Blocked, Project
from test_model_logging import Response, event


class ModelResponseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'project'
        self.messages = [{'role': 'system', 'content': '返回 JSON 对象'},
                         {'role': 'user', 'content': '生成 HTML'}]
        for context in (patch.dict(os.environ, {
                'LLM_API_KEY': 'test-private-key', 'LLM_MODEL': 'test-model',
                'LLM_BASE_URL': 'https://example.invalid'}, clear=True),
                patch.object(model, 'load_dotenv'), patch('sys.stdout', io.StringIO())):
            context.start()
            self.addCleanup(context.stop)

    @staticmethod
    def response(content, finish='stop'):
        return Response(event({'content': content}, finish) + 'data: [DONE]\n\n')

    def task(self, project):
        return project.task('segments/S01', {}, [],
            lambda directory, version, previous: project.call(directory, 'generated', self.messages))

    def test_malformed_json_is_retried_with_evidence_in_same_candidate(self):
        raw = '{\n  "html": "test-private-key",\n  broken: true\n}'
        html = '<script>const a="中文";\nconst b="\\\\";</script>'
        before = copy.deepcopy(self.messages)
        with patch.object(model, 'urlopen', side_effect=[self.response(raw),
                self.response(json.dumps({'html': html}))]) as request:
            with Project(self.root, {'story': 'test'}) as project:
                self.assertEqual(self.task(project), {'html': html})
                self.assertEqual(project.data['calls'], 2)
                self.assertEqual(len(project.data['tasks']['segments/S01']['attempts']), 1)
                self.assertTrue(project.artifacts_valid(project.data['tasks']['segments/S01']))
        folder = self.root / 'segments/S01/v001'
        first = folder / 'calls/generated/000001'
        self.assertEqual((first / 'response.txt').read_text(), raw.replace('test-private-key', '[REDACTED]'))
        detail = read_json(first / 'response.json')
        self.assertEqual((detail['stage'], detail['line'], detail['column']), ('content', 3, 3))
        self.assertEqual(detail['finish_reason'], 'stop')
        sent = json.loads(request.call_args.args[0].data)['messages']
        feedback = json.loads(sent[-1]['content'])
        self.assertIn('broken', feedback['previous_response'])
        self.assertIn('禁止注释', feedback['instruction'])
        self.assertEqual(self.messages, before)
        self.assertEqual(read_json(folder / 'generated.call.json')['status'], 'PASS')
        for path in folder.rglob('*'):
            if path.is_file():
                self.assertNotIn('test-private-key', path.read_text())
        with Project(self.root, {'story': 'test'}, resume=True) as project, \
                patch.object(model, 'urlopen', side_effect=AssertionError('应复用成功缓存')):
            self.assertEqual(self.task(project), {'html': html})
            self.assertEqual(project.data['calls'], 2)

    def test_repeated_format_failure_pauses_and_resume_keeps_candidate_and_budget(self):
        with patch.object(model, 'urlopen', side_effect=[self.response(' '), self.response('{bad}')]) as call:
            with self.assertRaisesRegex(Blocked, '响应格式连续失败'):
                with Project(self.root, {'story': 'test'}) as project:
                    self.task(project)
            self.assertEqual(call.call_count, 2)
        manifest = read_json(self.root / 'manifest.json')
        attempts = manifest['tasks']['segments/S01']['attempts']
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]['status'], 'RETRY')
        self.assertEqual(manifest['status'], 'BLOCKED')
        self.assertFalse((self.root / 'final.mp4').exists())
        self.assertFalse((self.root / 'segments/S01/v001/generated.json').exists())
        with Project(self.root, {'story': 'test'}, resume=True) as project, \
                patch.object(model, 'urlopen', return_value=self.response('{"html":"ok"}')):
            self.task(project)
            self.assertEqual(project.data['calls'], 3)
            self.assertEqual(len(project.data['tasks']['segments/S01']['attempts']), 1)
        self.assertEqual(len(list(self.root.rglob('response.txt'))), 3)

    def test_format_retry_cannot_exceed_global_call_budget(self):
        with patch.object(model, 'urlopen', return_value=self.response('{bad}')) as call:
            with self.assertRaisesRegex(Blocked, '调用预算已用尽'):
                with Project(self.root, {'story': 'test'}, max_calls=1) as project:
                    self.task(project)
            self.assertEqual(call.call_count, 1)
        with Project(self.root, {'story': 'test'}, resume=True, max_calls=1) as project, \
                patch.object(model, 'urlopen') as call:
            with self.assertRaises(Blocked):
                self.task(project)
            call.assert_not_called()

    def test_failed_call_metadata_cannot_validate_an_old_output_file(self):
        with Project(self.root, {'story': 'test'}) as project:
            signature = digest({'messages': self.messages, 'settings': model.settings()})
            write_json(self.root / 'generated.json', {'html': 'stale output'})
            for status in ('RUNNING', 'FORMAT_ERROR'):
                write_json(self.root / 'generated.call.json', {
                    'input_hash': signature, 'status': status})
                with self.subTest(status=status), patch.object(model, 'urlopen',
                        return_value=self.response('{"html":"fresh output"}')) as request:
                    result = project.call(self.root, 'generated', self.messages)
                    self.assertEqual(result, {'html': 'fresh output'})
                    request.assert_called_once()

    def test_protocol_json_and_content_json_have_distinct_diagnostics(self):
        cases = [('sse_event', Response('data: {broken}\n\n')),
                 ('response_envelope', Response('{broken}', 'application/json')),
                 ('content', self.response('[]')),
                 ('content', self.response(' \n  '))]
        for i, (stage, response) in enumerate(cases):
            directory = self.root / str(i)
            with self.subTest(stage=stage, i=i), \
                    patch.object(model, 'urlopen', return_value=response), \
                    self.assertRaises(model.ResponseFormatError):
                model.ask(self.messages, diagnostics=directory)
            detail = read_json(directory / 'response.json')
            self.assertEqual(detail['stage'], stage)
            self.assertEqual(detail['status'], 'FORMAT_ERROR')
            self.assertTrue((directory / 'response.txt').exists())

    def test_truncated_response_is_saved_but_never_parsed_as_success(self):
        directory = self.root / 'truncated'
        with patch.object(model, 'urlopen', return_value=self.response('{"ok":true}', 'length')), \
                self.assertRaisesRegex(ValueError, '截断'):
            model.ask(self.messages, diagnostics=directory)
        self.assertEqual(read_json(directory / 'response.json')['status'], 'TRUNCATED')
        self.assertEqual((directory / 'response.txt').read_text(), '{"ok":true}')


if __name__ == '__main__':
    unittest.main()
