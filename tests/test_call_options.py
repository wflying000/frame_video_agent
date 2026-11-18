"""逐次推理参数绑定实际请求、缓存和审核复用；默认配置与预算保持不变。"""
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent import model, planning, rendering
from agent.common import digest, file_hash, read_json
from agent.store import Blocked, Project
from fixtures import FakeModel, REQ, STORY
from test_model_logging import Response, event


ENABLED = {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high'}
DEFAULT_BODY = {'thinking': {'type': 'disabled'}, 'max_tokens': 32768, 'top_p': 0.9}


class CallOptionsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / 'project'
        self.messages = [{'role': 'system', 'content': '返回严格JSON'},
                         {'role': 'user', 'content': '检查实际证据'}]
        for context in (patch.dict(os.environ, {
                'LLM_API_KEY': 'test-only-key', 'LLM_MODEL': 'deepseek-flash',
                'LLM_BASE_URL': 'https://api.deepseek.com',
                'LLM_EXTRA_BODY': json.dumps(DEFAULT_BODY)}, clear=True),
                patch.object(model, 'load_dotenv'), patch('sys.stdout', io.StringIO())):
            context.start()
            self.addCleanup(context.stop)

    @staticmethod
    def response(content):
        return Response(event({'content': content}, 'stop') + 'data: [DONE]\n\n')

    def test_effective_settings_merge_without_changing_defaults_or_caller_options(self):
        defaults = copy.deepcopy(model.settings())
        options = copy.deepcopy(ENABLED)
        effective = model.effective_settings(options)
        self.assertEqual(effective['model'], defaults['model'])
        self.assertEqual(effective['base_url'], defaults['base_url'])
        self.assertEqual(effective['extra_body'], {**DEFAULT_BODY, **ENABLED})
        self.assertEqual(effective['extra_body']['max_tokens'], 32768)
        self.assertEqual(model.effective_settings(), defaults)
        self.assertEqual(model.effective_settings({}), defaults)
        self.assertEqual(options, ENABLED)
        effective['extra_body']['thinking']['type'] = 'mutated-result'
        self.assertEqual(model.settings(), defaults)
        self.assertEqual(options, ENABLED)

    def test_visual_review_options_only_target_the_official_deepseek_hostname(self):
        allowed = ('https://api.deepseek.com', 'https://api.deepseek.com/v1/',
                   'https://API.DEEPSEEK.COM/custom/path')
        denied = ('https://example.invalid', 'https://api.deepseek.com.evil.invalid',
                  'https://example.invalid/api.deepseek.com',
                  'https://api.deepseek.com@example.invalid/v1', 'http://localhost:8000/v1')
        for base_url in allowed + denied:
            with self.subTest(base_url=base_url), patch.dict(os.environ, {'LLM_BASE_URL': base_url}):
                before = model.settings()
                self.assertEqual(model.visual_review_options(), ENABLED if base_url in allowed else {})
                self.assertEqual(model.settings(), before)

    def test_actual_http_body_uses_the_same_effective_extra_body_and_keeps_defaults(self):
        before = model.settings()
        with patch.object(model, 'urlopen', return_value=self.response('{"ok":true}')) as request:
            self.assertEqual(model.ask(self.messages, extra_body=ENABLED), {'ok': True})
        body = json.loads(request.call_args.args[0].data)
        expected = model.effective_settings(ENABLED)
        self.assertEqual(body['model'], expected['model'])
        self.assertEqual(body['messages'], self.messages)
        self.assertEqual(request.call_args.args[0].full_url, 'https://api.deepseek.com/chat/completions')
        for key, value in expected['extra_body'].items():
            self.assertEqual(body[key], value)
        self.assertEqual(body['response_format'], {'type': 'json_object'})
        self.assertTrue(body['stream'])
        self.assertEqual(model.settings(), before)
        with patch.object(model, 'urlopen', return_value=self.response('{"ok":true}')) as request:
            model.ask(self.messages)
        default_body = json.loads(request.call_args.args[0].data)
        self.assertEqual(default_body['thinking'], {'type': 'disabled'})
        self.assertNotIn('reasoning_effort', default_body)

    def test_call_cache_isolated_by_effective_settings_and_reused_for_identical_options(self):
        with Project(self.root, {'story': STORY}) as project, \
                patch.object(model, 'ask', side_effect=[{'value': 'default'}, {'value': 'enabled'}]) as ask:
            self.assertEqual(project.call(self.root, 'review', self.messages), {'value': 'default'})
            first = read_json(self.root / 'review.call.json')
            self.assertEqual(first['model'], model.settings())
            self.assertEqual(first['input_hash'], digest({'messages': self.messages, 'settings': model.settings()}))
            enabled = project.call(self.root, 'review', self.messages, extra_body=ENABLED)
            self.assertEqual(enabled, {'value': 'enabled'})
            metadata = read_json(self.root / 'review.call.json')
            expected = model.effective_settings(ENABLED)
            self.assertEqual(metadata['model'], expected)
            self.assertEqual(metadata['input_hash'], digest({'messages': self.messages, 'settings': expected}))
            self.assertNotEqual(metadata['input_hash'], first['input_hash'])
            self.assertEqual(model.effective_settings(ask.call_args.kwargs.get('extra_body')), expected)
            self.assertEqual(project.call(self.root, 'review', self.messages, extra_body=copy.deepcopy(ENABLED)), enabled)
            # 显式给出相同的完整有效参数与只给覆盖项应复用同一缓存。
            self.assertEqual(project.call(self.root, 'review', self.messages,
                                          extra_body=expected['extra_body']), enabled)
            self.assertEqual(ask.call_count, 2)
            self.assertEqual(project.data['calls'], 2)
            self.assertEqual(model.settings()['extra_body'], DEFAULT_BODY)

    def test_format_retry_preserves_options_then_resume_keeps_candidate_and_accumulated_budget(self):
        def task(project):
            return project.task('segments/S01', {'review_options': ENABLED}, [],
                lambda directory, version, previous: project.call(directory, 'review', self.messages,
                                                                  extra_body=ENABLED))
        with patch.object(model, 'urlopen', side_effect=[self.response('{bad}'), self.response('{bad}')]) as request:
            with self.assertRaisesRegex(Blocked, '响应格式连续失败'):
                with Project(self.root, {'story': STORY}) as project:
                    task(project)
            self.assertEqual(request.call_count, 2)
            for call in request.call_args_list:
                body = json.loads(call.args[0].data)
                self.assertEqual(body['thinking'], ENABLED['thinking'])
                self.assertEqual(body['reasoning_effort'], 'high')
                self.assertEqual(body['max_tokens'], 32768)
        manifest = read_json(self.root / 'manifest.json')
        self.assertEqual(manifest['calls'], 2)
        self.assertEqual(len(manifest['tasks']['segments/S01']['attempts']), 1)
        directory = self.root / 'segments/S01/v001'
        metadata = read_json(directory / 'review.call.json')
        self.assertEqual(metadata['status'], 'FORMAT_ERROR')
        self.assertEqual(metadata['model'], model.effective_settings(ENABLED))
        self.assertFalse((directory / 'review.json').exists())
        self.assertFalse((self.root / 'final.mp4').exists())
        with Project(self.root, {'story': STORY}, resume=True) as project, \
                patch.object(model, 'urlopen', return_value=self.response('{"ok":true}')) as request:
            self.assertEqual(task(project), {'ok': True})
            self.assertEqual(project.data['calls'], 3)
            self.assertEqual(len(project.data['tasks']['segments/S01']['attempts']), 1)
            self.assertEqual(json.loads(request.call_args.args[0].data)['thinking'], ENABLED['thinking'])

    def test_format_retry_cannot_bypass_global_call_limit_with_reasoning_enabled(self):
        with patch.object(model, 'urlopen', return_value=self.response('{bad}')) as request:
            with self.assertRaisesRegex(Blocked, '调用预算已用尽'):
                with Project(self.root, {'story': STORY}, max_calls=1) as project:
                    project.call(self.root, 'review', self.messages, extra_body=ENABLED)
            request.assert_called_once()
        self.assertEqual(read_json(self.root / 'manifest.json')['calls'], 1)
        self.assertFalse((self.root / 'review.json').exists())

    def test_review_reuse_cannot_cross_reasoning_configuration(self):
        with Project(self.root, {'story': STORY}) as project, patch.object(model, 'ask', side_effect=FakeModel()) as ask:
            source = self.root / 'frame.png'
            source.write_bytes(b'unchanged actual pixels')
            images = [('实际画面 frame_index=0', source)]
            def review(version, options=None):
                return planning.review(project, self.root / f'segments/S01/v{version:03d}', 'review',
                                       'S01', version, REQ, {}, images, [0], extra_body=options)
            self.assertNotIn('reused_from', review(1))
            first = self.root / 'segments/S01/v001'
            first_hashes = {path: file_hash(path) for path in first.rglob('*') if path.is_file()}
            self.assertNotIn('reused_from', review(2, ENABLED))
            third = review(3, ENABLED)
            self.assertEqual(third['reused_from']['path'], 'segments/S01/v002/review.json')
            fourth = review(4)
            self.assertEqual(fourth['reused_from']['path'], 'segments/S01/v001/review.json')
            self.assertEqual(ask.call_count, 2)
            self.assertEqual(project.data['calls'], 2)
            self.assertEqual(read_json(self.root / 'segments/S01/v003/review.call.json')['model'],
                             model.effective_settings(ENABLED))
            self.assertEqual(first_hashes, {path: file_hash(path) for path in first_hashes})

    def test_schema_retry_and_detail_confirmation_keep_the_same_call_options(self):
        for case in ('schema', 'detail'):
            with self.subTest(case=case), Project(self.root / case, {'story': STORY}) as project:
                image = project.root / 'frame.png'
                image.write_bytes(b'actual frame pixels')
                fake = FakeModel()
                def ask(messages, **kwargs):
                    result = fake(messages)
                    if len(fake.calls) == 1:
                        if case == 'schema':
                            result['checks'] = []
                        else:
                            result['status'] = 'FAIL'
                            result['checks'][0].update(result='FAIL', region_xyxy=[.2, .2, .6, .6],
                                                       evidence_frames=[0])
                    return result
                def crop(source, output, region):
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(b'actual crop pixels')
                with patch.object(model, 'ask', side_effect=ask) as called, \
                        patch.object(rendering, 'crop_evidence', side_effect=crop):
                    result = planning.review(project, project.root / 'segments/S01/v001', 'review',
                                             'S01', 1, REQ, {}, [('实际画面 frame_index=0', image)], [0],
                                             extra_body=ENABLED)
                self.assertEqual(result['status'], 'PASS')
                self.assertEqual(called.call_count, 2)
                for call in called.call_args_list:
                    self.assertEqual(model.effective_settings(call.kwargs.get('extra_body')),
                                     model.effective_settings(ENABLED))
                for metadata in (project.root / 'segments/S01/v001').glob('*.call.json'):
                    self.assertEqual(read_json(metadata)['model'], model.effective_settings(ENABLED))
                self.assertEqual(model.settings()['extra_body'], DEFAULT_BODY)


if __name__ == '__main__':
    unittest.main()
