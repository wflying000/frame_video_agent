"""Explicit reference inference upgrades preserve pixels, history and every formal gate."""
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent import drawing, model, prompts, rendering
from agent.common import digest, file_hash, read_json, write_json
from agent.store import Blocked, Project
from fixtures import FACTS, HTML, STORY, VISUAL, FakeModel
import test_workflow
from test_model_logging import Response, event

OPTIONS = {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high'}
DEFAULT = {'thinking': {'type': 'disabled'}, 'max_tokens': 32768}


class ReferenceReasonedTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve() / 'project'
        self.fake = FakeModel()
        for context in (
            patch.dict(os.environ, {'LLM_API_KEY': 'test-only-key', 'LLM_BASE_URL': 'https://api.deepseek.com',
                       'LLM_MODEL': 'deepseek-flash', 'LLM_EXTRA_BODY': json.dumps(DEFAULT)}, clear=True),
            patch.object(model, 'load_dotenv'), patch('sys.stdout', io.StringIO()),
            patch.object(rendering, 'render_job', side_effect=test_workflow.WorkflowTests.render_stub)):
            context.start(); self.addCleanup(context.stop)

    def seed(self, project, fail=False):
        project.task('plans/story', {}, [], lambda *args: FACTS)
        project.task('plans/visual', {}, ['plans/story'], lambda *args: VISUAL)
        def ordinary(messages, **kwargs):
            result = self.fake(messages, **kwargs)
            payload = json.loads(messages[1]['content'][0]['text'])
            if messages[0]['content'] == prompts.CANVAS:
                result['html'] = HTML.replace('\n', '\r\n')
            if fail and messages[0]['content'] == prompts.REVIEW and payload.get('visual_scope') == 'reference':
                result['status'] = 'FAIL'
                for row in result['checks']:
                    row['result'] = 'FAIL'
            return result
        with patch.object(model, 'ask', side_effect=ordinary):
            if fail:
                project.task_attempt_limits['assets/P01'] = 1
                with self.assertRaises(Blocked):
                    drawing.assets(project, FACTS, VISUAL)
                project.task_attempt_limits['assets/P01'] = 3
            else:
                drawing.assets(project, FACTS, VISUAL)
        source = self.root / 'assets/P01/v001/reference.html'
        descriptor = {'source_hash': file_hash(source), 'reason': 'Explicit same-pixel inference upgrade'}
        project.data['repair_requests']['P01'] = {'reference_reasoned_recheck': descriptor}
        return source, descriptor

    def test_upgrade_preserves_failed_bytes_and_history_without_generation_or_observation_anchoring(self):
        with Project(self.root, {'story': STORY, 'fps': 2}) as project:
            source, descriptor = self.seed(project, fail=True)
            old = {path: file_hash(path) for path in source.parent.rglob('*') if path.is_file()}
            calls = []
            def ask(messages, **kwargs):
                calls.append((messages, copy.deepcopy(kwargs)))
                return self.fake(messages, **kwargs)
            with patch.object(model, 'ask', side_effect=ask):
                result = drawing.assets(project, FACTS, VISUAL)['P01']
            current = self.root / 'assets/P01/v002'
            self.assertEqual(result['version'], 2)
            self.assertEqual(source.read_bytes(), (current / 'reference.html').read_bytes())
            self.assertIn(b'\r\n', source.read_bytes())
            self.assertEqual(old, {path: file_hash(path) for path in old})
            self.assertFalse((current / 'generated.json').exists())
            self.assertTrue((current / 'independent_observations.json').is_file())
            formal = []
            for messages, kwargs in calls:
                self.assertNotIn(messages[0]['content'], (prompts.CANVAS, prompts.PATCH))
                payload = json.loads(messages[1]['content'][0]['text'])
                if messages[0]['content'] == prompts.REVIEW:
                    formal.append(payload)
                    self.assertEqual(kwargs['extra_body'], {**DEFAULT, **OPTIONS})
                    self.assertNotIn('independent_observations', payload)
                    self.assertNotIn('observation_context', payload)
                    self.assertNotIn('previous_review', payload)
                    self.assertEqual(payload['reference_review_protocol'], drawing.REFERENCE_REASONED_PROTOCOL)
                    self.assertEqual(set(payload['required_check_ids']),
                        {'R01', 'DEPTH_VOLUME', 'DEPTH_ILLUSTRATION', 'DEPTH_REFERENCE_SPACE'})
                else:
                    self.assertNotIn('extra_body', kwargs)
            self.assertTrue(formal)
            proof = read_json(current / 'reference_reasoned_recheck_provenance.json')
            self.assertEqual(proof['source_hash'], descriptor['source_hash'])
            self.assertEqual(proof['source_version'], 1)
            self.assertEqual(proof['source_request_hash'], file_hash(source.parent / 'review.request.json'))
            self.assertEqual(proof['source_failure_hash'], file_hash(source.parent / 'failure.json'))
            self.assertEqual(proof['source_pixels'][0]['hash'], file_hash(source.parent / 'preview/preview_00.png'))
            self.assertEqual(proof['effective_settings'], model.effective_settings(OPTIONS))
            self.assertEqual(proof['current_review_requests'][0]['hash'], file_hash(current / 'review.request.json'))
            self.assertEqual(model.settings()['extra_body'], DEFAULT)
            before = project.data['calls']
            with patch.object(model, 'ask') as again:
                drawing.assets(project, FACTS, VISUAL)
            again.assert_not_called(); self.assertEqual(project.data['calls'], before)

    def test_actual_http_options_and_task_signature_are_bound(self):
        with Project(self.root, {'story': STORY, 'fps': 2}) as project:
            source, descriptor = self.seed(project)
            bodies, inputs = [], {}
            def transport(request, **kwargs):
                body = json.loads(request.data); bodies.append(body)
                result = self.fake(body['messages'])
                return Response(event({'content': json.dumps(result)}, 'stop') + 'data: [DONE]\n\n')
            original = project.task
            def task(task_id, task_inputs, *args, **kwargs):
                inputs[task_id] = copy.deepcopy(task_inputs)
                return original(task_id, task_inputs, *args, **kwargs)
            with patch.object(model, 'urlopen', side_effect=transport), patch.object(project, 'task', side_effect=task):
                drawing.assets(project, FACTS, VISUAL)
            formal = [b for b in bodies if b['messages'][0]['content'] == prompts.REVIEW]
            self.assertTrue(formal)
            for body in formal:
                self.assertEqual(body['thinking'], OPTIONS['thinking'])
                self.assertEqual(body['reasoning_effort'], 'high')
                self.assertEqual(body['max_tokens'], 32768)
            for body in bodies:
                if body['messages'][0]['content'] == prompts.OBSERVE:
                    self.assertEqual(body['thinking'], DEFAULT['thinking'])
                    self.assertNotIn('reasoning_effort', body)
            captured = inputs['assets/P01']
            self.assertEqual(captured['review_protocol'], drawing.REFERENCE_REASONED_PROTOCOL)
            binding = captured['reference_reasoned_binding']
            self.assertEqual(binding['source_hash'], file_hash(source))
            self.assertEqual(binding['source_version'], 1)
            self.assertEqual(binding['source_request_hash'], file_hash(source.parent / 'review.request.json'))
            self.assertEqual(binding['effective_settings'], model.effective_settings(OPTIONS))
            node = project.data['tasks']['assets/P01']
            signature = {'inputs': captured, 'dependencies': node['input_versions'], 'model': model.settings(),
                         'prompts': {k:v for k,v in vars(prompts).items() if k.isupper()},
                         'render_consistency': project.render_consistency}
            self.assertEqual(node['input_hash'], digest(signature))
            changed = copy.deepcopy(signature)
            changed['inputs']['reference_reasoned_binding']['effective_settings']['extra_body']['reasoning_effort'] = 'low'
            self.assertNotEqual(node['input_hash'], digest(changed))

    def test_source_tampering_is_rejected_before_new_calls(self):
        with Project(self.root, {'story': STORY, 'fps': 2}) as project:
            source, _ = self.seed(project)
            source.write_bytes(source.read_bytes() + b'\nchanged')
            with patch.object(model, 'ask') as ask, self.assertRaisesRegex(Blocked, '源版本不匹配'):
                drawing.assets(project, FACTS, VISUAL)
            ask.assert_not_called()

    def test_unsupported_endpoint_missing_unknown_and_already_reasoned_sources_are_rejected(self):
        with Project(self.root, {'story': STORY, 'fps': 2}) as project:
            source, descriptor = self.seed(project)
            call = source.parent / 'review.call.json'
            original = read_json(call)
            for old_settings in ({}, model.effective_settings(OPTIONS),
                                 {'base_url': 'https://api.deepseek.com', 'model': 'deepseek-flash', 'extra_body': {}}):
                write_json(call, dict(original, model=old_settings))
                with self.subTest(settings=old_settings), self.assertRaises(Blocked):
                    drawing.reference_reasoned_binding(project, source, descriptor, OPTIONS)
            write_json(call, original)
            with self.assertRaises(Blocked):
                drawing.reference_reasoned_binding(project, source, descriptor, {})
            request = source.parent / 'review.request.json'; saved = read_json(request)
            for protocol in ('future_reference', drawing.REFERENCE_REASONED_PROTOCOL):
                messages = copy.deepcopy(saved); payload = json.loads(messages[1]['content'][0]['text'])
                payload['reference_review_protocol'] = protocol
                messages[1]['content'][0]['text'] = json.dumps(payload);write_json(request, messages)
                with self.subTest(protocol=protocol), self.assertRaises(Blocked):
                    drawing.reference_reasoned_binding(project, source, descriptor, OPTIONS)
            write_json(request, saved);call.unlink()
            with self.assertRaises(Blocked):
                drawing.reference_reasoned_binding(project, source, descriptor, OPTIONS)

    def test_same_candidate_provenance_resume_is_stable_but_new_candidate_duplicate_is_rejected(self):
        with Project(self.root, {'story': STORY, 'fps': 2}) as project:
            source, descriptor = self.seed(project)
            binding = drawing.reference_reasoned_binding(project, source, descriptor, OPTIONS)
            previous = project.data['tasks']['assets/P01']['attempts'][-1]
            target = self.root / 'assets/P01/v002'
            first = drawing.reference_reasoned_recheck_source(project, target, previous, descriptor, OPTIONS, binding)
            proof_hash = file_hash(target / 'reference_reasoned_recheck_provenance.json')
            self.assertEqual(first, drawing.reference_reasoned_recheck_source(project, target, previous, descriptor, OPTIONS, binding))
            self.assertEqual(proof_hash, file_hash(target / 'reference_reasoned_recheck_provenance.json'))
            with self.assertRaisesRegex(Blocked, '不能重复协议复验'):
                drawing.reference_reasoned_recheck_source(project, self.root / 'assets/P01/v003', previous, descriptor, OPTIONS, binding)
            altered = copy.deepcopy(binding);altered['source_request_hash']='changed'
            with self.assertRaisesRegex(Blocked, '来源或有效参数改变'):
                drawing.reference_reasoned_recheck_source(project, target, previous, descriptor, OPTIONS, altered)

    def test_formal_failure_still_blocks_and_does_not_reroll_current_protocol(self):
        with Project(self.root, {'story': STORY, 'fps': 2}) as project:
            self.seed(project)
            def reject(messages, **kwargs):
                result = self.fake(messages, **kwargs)
                if messages[0]['content'] == prompts.REVIEW:
                    result['status'] = 'FAIL'
                    result['checks'][0]['result'] = 'FAIL'
                return result
            with patch.object(model, 'ask', side_effect=reject), self.assertRaises(Blocked):
                drawing.assets(project, FACTS, VISUAL)
            self.assertFalse((self.root / 'final.mp4').exists())
            self.assertEqual(read_json(self.root / 'assets/P01/v002/failure.json')['status'], 'FAIL')
            before = project.data['calls']
            with patch.object(model, 'ask') as ask, self.assertRaises(Blocked):
                drawing.assets(project, FACTS, VISUAL)
            ask.assert_not_called();self.assertEqual(before, project.data['calls'])

    def test_interrupted_candidate_resumes_same_source_protocol_and_cached_observation(self):
        config = {'story': STORY, 'fps': 2}
        with self.assertRaisesRegex(Blocked, 'simulated interruption'):
            with Project(self.root, config) as project:
                self.seed(project)
                def interrupt(messages, **kwargs):
                    if messages[0]['content'] == prompts.REVIEW:
                        raise Blocked('simulated interruption')
                    return self.fake(messages, **kwargs)
                with patch.object(model, 'ask', side_effect=interrupt):
                    drawing.assets(project, FACTS, VISUAL)
        folder = self.root / 'assets/P01/v002'
        source_hash = file_hash(folder / 'reference.html')
        old_calls = read_json(self.root / 'manifest.json')['calls']
        with Project(self.root, config, resume=True) as project, \
                patch.object(model, 'ask', side_effect=self.fake) as ask:
            result = drawing.assets(project, FACTS, VISUAL)['P01']
            self.assertEqual(result['version'], 2)
            self.assertEqual(len(project.data['tasks']['assets/P01']['attempts']), 2)
            self.assertEqual(project.data['calls'], old_calls + 1)
            self.assertEqual(ask.call_count, 1)
            self.assertEqual(ask.call_args.args[0][0]['content'], prompts.REVIEW)
            self.assertEqual(ask.call_args.kwargs['extra_body'], {**DEFAULT, **OPTIONS})
        self.assertEqual(file_hash(folder / 'reference.html'), source_hash)

    def test_old_pixels_request_report_and_failure_tampering_are_rejected(self):
        with Project(self.root, {'story': STORY, 'fps': 2}) as project:
            source, descriptor = self.seed(project, fail=True)
            mutations = [
                (source.parent / 'preview/preview_00.png', lambda path: path.write_bytes(path.read_bytes()+b'changed')),
                (source.parent / 'review.request.json', lambda path: path.write_text(path.read_text().replace('reference', 'changed', 1))),
                (source.parent / 'review.json', lambda path: write_json(path, dict(read_json(path), target_version=99))),
                (source.parent / 'failure.json', lambda path: write_json(path, dict(read_json(path), extra='changed'))),
            ]
            for path, mutate in mutations:
                before = path.read_bytes()
                with self.subTest(path=path.name):
                    mutate(path)
                    with self.assertRaises(Blocked):
                        drawing.reference_reasoned_binding(project, source, descriptor, OPTIONS)
                    path.write_bytes(before)

    def test_changed_new_render_is_blocked_before_observation_or_formal_call(self):
        with Project(self.root, {'story': STORY, 'fps': 2}) as project:
            self.seed(project)
            def changed(*args, **kwargs):
                test_workflow.WorkflowTests.render_stub(*args, **kwargs)
                path = args[1] / 'preview_00.png'
                path.write_bytes(path.read_bytes()+b'changed-raster')
            before = project.data['calls']
            with patch.object(rendering, 'render_job', side_effect=changed), \
                    patch.object(model, 'ask') as ask, self.assertRaisesRegex(Blocked, '重画像素'):
                drawing.assets(project, FACTS, VISUAL)
            ask.assert_not_called();self.assertEqual(project.data['calls'], before)

    def test_jpeg_transport_must_bind_both_original_png_and_actual_request(self):
        with Project(self.root, {'story': STORY, 'fps': 2}) as project:
            source, descriptor = self.seed(project)
            png = source.parent / 'preview/preview_00.png'
            encoded = source.parent / 'review_transport/0000.jpg'
            encoded.parent.mkdir();encoded.write_bytes(b'jpeg-transport-fixture')
            request_path = source.parent / 'review.request.json'
            request = read_json(request_path)
            url = model.messages('', {}, [('', encoded)])[1]['content'][-1]['image_url']['url']
            for part in request[1]['content']:
                if part.get('type') == 'image_url':
                    part['image_url'] = {'url': url}
            call_path = source.parent / 'review.call.json';call=read_json(call_path)
            call['input_hash'] = digest({'messages': request, 'settings': call['model']})
            write_json(request_path, model.recorded_messages(request));write_json(call_path, call)
            encoder = {'codec':'mjpeg', 'q':2, 'pixel_format':'yuvj444p', 'resize':False, 'crop':False}
            record = {'original_path':str(png.resolve()),'original_sha256':file_hash(png),
                      'encoded_path':str(encoded.resolve()),'encoded_sha256':file_hash(encoded),'encoder':encoder}
            manifest=encoded.parent/'manifest.json';write_json(manifest, {'encoder':encoder,'images':[record]})
            binding=drawing.reference_reasoned_binding(project, source, descriptor, OPTIONS)
            self.assertEqual(binding['source_pixels'][0]['hash'],file_hash(png))
            record['original_sha256']='changed';write_json(manifest, {'encoder':encoder,'images':[record]})
            with self.assertRaises(Blocked):
                drawing.reference_reasoned_binding(project, source, descriptor, OPTIONS)

    def test_mutually_exclusive_requests_and_normal_inputs_are_unchanged(self):
        with Project(self.root, {'story': STORY, 'fps': 2}) as project:
            _, descriptor = self.seed(project)
            for name in ('local_patch', 'reference_recheck'):
                project.data['repair_requests']['P01'] = {'reference_reasoned_recheck': descriptor, name: {}}
                with self.assertRaisesRegex(ValueError, '不能同时指定'):
                    drawing.assets(project, FACTS, VISUAL)
            project.data['repair_requests'].clear()
            captured = []
            original = project.task
            def task(task_id, inputs, *args, **kwargs):
                captured.append(inputs);return original(task_id, inputs, *args, **kwargs)
            with patch.object(project, 'task', side_effect=task), patch.object(model, 'ask') as ask:
                drawing.assets(project, FACTS, VISUAL)
            ask.assert_not_called()
            self.assertEqual(captured, [{'asset': VISUAL['assets'][0], 'review_protocol': 'reference_observation_v1'}])


if __name__ == '__main__':
    unittest.main()
