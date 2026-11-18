"""审核JSON的唯一多闭合括号恢复：值保真、证据可复验，严格审核仍是门槛。"""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent import model, planning, prompts
from agent.common import digest, read_json, write_json
from agent.store import Blocked, Project, Rejected
from test_model_logging import Response, event


class ReviewJSONRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / 'project'
        self.requirements = [{'id': 'R01', 'hard': True, 'text': '原始动作要求必须成立'}]
        self.messages = model.messages(prompts.REVIEW, {
            'target_id': 'S01', 'target_version': 1, 'requirements': self.requirements})
        for context in (patch.dict(os.environ, {
                'LLM_API_KEY': 'test-only-key', 'LLM_MODEL': 'test-model',
                'LLM_BASE_URL': 'https://example.invalid'}, clear=True),
                patch.object(model, 'load_dotenv'), patch('sys.stdout', io.StringIO())):
            context.start()
            self.addCleanup(context.stop)

    @staticmethod
    def response(raw):
        return Response(event({'content': raw}, 'stop') + 'data: [DONE]\n\n')

    @staticmethod
    def report(status='PASS'):
        return {'target_id': 'S01', 'target_version': 1, 'status': status,
                'checks': [{'requirement_id': 'R01', 'result': status,
                            'evidence': '真实像素证据，文字中的 } , "repair": 不是结构',
                            'evidence_frames': [0]}]}

    @staticmethod
    def malformed(prefix, repair=None):
        repair = {'scope': 'frame', 'shot_ids': ['S01'], 'instruction': '原修复建议保持不变'} if repair is None else repair
        return (' \n' + json.dumps(prefix, ensure_ascii=False) + ' , "repair": ' +
                json.dumps(repair, ensure_ascii=False) + '}\n')

    def test_recovery_preserves_pass_fail_uncertain_values_and_verifiable_raw_proof(self):
        repair = {'scope': 'shot', 'shot_ids': ['S01'], 'instruction': '保留原判断',
                  'nested': {'values': [None, True, 1, 0.125, '中文']}}
        for status in ('PASS', 'FAIL', 'UNCERTAIN'):
            with self.subTest(status=status):
                prefix = self.report(status)
                raw = self.malformed(prefix, repair)
                directory = self.root / status
                expected = dict(copy.deepcopy(prefix), repair=copy.deepcopy(repair))
                with patch.object(model, 'urlopen', return_value=self.response(raw)) as request:
                    result = model.ask(self.messages, diagnostics=directory)
                request.assert_called_once()
                self.assertEqual(result, expected)
                self.assertEqual((directory / 'response.txt').read_text(), raw)
                self.assertEqual(read_json(directory / 'normalized_response.json'), expected)
                response = read_json(directory / 'response.json')
                proof = read_json(directory / 'recovery_proof.json')
                self.assertEqual(response['recovery'], proof)
                self.assertEqual(proof['kind'], 'review_repair_extra_closing_brace_v1')
                self.assertEqual(proof['raw_sha256'], hashlib.sha256(raw.encode()).hexdigest())
                normalized = (directory / 'normalized_response.json').read_bytes()
                self.assertEqual(proof['normalized_sha256'], hashlib.sha256(normalized).hexdigest())
                self.assertEqual(proof['normalized_file'], 'normalized_response.json')
                self.assertEqual(normalized.decode(), json.dumps(expected, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
                split = proof['prefix_end']
                self.assertEqual(json.loads(raw[:split]), prefix)
                self.assertEqual(json.loads('{' + raw[split:].strip()[1:]), {'repair': repair})
                self.assertEqual({key: result[key] for key in prefix}, prefix)

    def test_ordinary_valid_json_keeps_existing_path_without_recovery_artifacts(self):
        for number, report in enumerate((dict(self.report(), repair={'scope': 'none'}), {'html': '<canvas/>'})):
            raw = json.dumps(report, ensure_ascii=False, indent=1)
            directory = self.root / str(number)
            messages = self.messages if number == 0 else model.messages(prompts.CANVAS, {'kind': 'shot'})
            with patch.object(model, 'urlopen', return_value=self.response(raw)):
                self.assertEqual(model.ask(messages, diagnostics=directory), report)
            self.assertEqual((directory / 'response.txt').read_text(), raw)
            self.assertNotIn('recovery', read_json(directory / 'response.json'))
            self.assertFalse((directory / 'normalized_response.json').exists())
            self.assertFalse((directory / 'recovery_proof.json').exists())

    def test_ambiguous_or_different_json_damage_is_rejected_without_normalization(self):
        prefix = self.report()
        text = json.dumps(prefix, ensure_ascii=False)
        missing = {key: value for key, value in prefix.items() if key != 'checks'}
        cases = {
            'missing_field': self.malformed(missing),
            'extra_prefix_field': self.malformed(dict(prefix, other=1)),
            'prefix_already_has_repair': self.malformed(dict(prefix, repair={})),
            'duplicate_prefix_key': text.replace('"status": "PASS"', '"status": "FAIL", "status": "PASS"') + ',"repair":{}}',
            'duplicate_nested_check_key': text.replace('"requirement_id": "R01"', '"requirement_id": "R02", "requirement_id": "R01"') + ',"repair":{}}',
            'duplicate_repair_key': text + ',"repair":{},"repair":{}}',
            'duplicate_inside_repair': text + ',"repair":{"scope":"frame","scope":"shot"}}',
            'extra_suffix_field': text + ',"repair":{},"other":1}',
            'non_object_repair': self.malformed(prefix, ['not', 'an', 'object']),
            'null_repair': text + ',"repair":null}',
            'no_comma': text + ' "repair":{}}',
            'trailing_text': self.malformed(prefix) + ' explanation',
            'second_json_value': self.malformed(prefix) + '{}',
            'extra_closing_brace': self.malformed(prefix) + '}',
            'missing_final_brace': text + ',"repair":{}',
            'wrong_suffix_name': text + ',"repairs":{}}',
            'nonstandard_constant': text + ',"repair":{"x":NaN}}',
            'array_prefix': '[1,2],"repair":{}}',
            'generic_syntax_error': '{"target_id":"S01",broken}',
        }
        for name, raw in cases.items():
            with self.subTest(case=name):
                directory = self.root / name
                with patch.object(model, 'urlopen', return_value=self.response(raw)), \
                        self.assertRaises(model.ResponseFormatError):
                    model.ask(self.messages, diagnostics=directory)
                self.assertEqual((directory / 'response.txt').read_text(), raw)
                self.assertEqual(read_json(directory / 'response.json')['status'], 'FORMAT_ERROR')
                self.assertFalse((directory / 'normalized_response.json').exists())
                self.assertFalse((directory / 'recovery_proof.json').exists())

    def test_recovery_is_limited_to_exact_review_requests(self):
        raw = self.malformed(self.report())
        for number, system in enumerate((prompts.CANVAS, prompts.OBSERVE, prompts.REVIEW + '\n')):
            with self.subTest(system=number), patch.object(model, 'urlopen', return_value=self.response(raw)), \
                    self.assertRaises(model.ResponseFormatError):
                model.ask(model.messages(system, {'kind': 'not-review'}), diagnostics=self.root / str(number))

    def test_recovered_json_still_requires_original_hard_checks_and_valid_pixel_evidence(self):
        cases = [('FAIL', 'FAIL', Rejected), ('UNCERTAIN', 'UNCERTAIN', Rejected),
                 ('PASS', 'FAIL', Rejected), ('PASS', 'missing', ValueError),
                 ('PASS', 'wrong_frame', ValueError)]
        for status, check_status, exception in cases:
            with self.subTest(status=status, check_status=check_status):
                root = self.root / (status + check_status)
                report = self.report(status)
                if check_status == 'missing':
                    report['checks'] = []
                elif check_status == 'wrong_frame':
                    report['checks'][0]['evidence_frames'] = [99]
                else:
                    report['checks'][0]['result'] = check_status
                raw = self.malformed(report)
                with Project(root, {'story': 'test'}) as project:
                    image = root / 'frame.png'
                    image.write_bytes(b'actual-pixel-fixture')
                    with patch.object(model, 'urlopen', side_effect=lambda *a, **k: self.response(raw)) as request, \
                            self.assertRaises(exception):
                        planning.review(project, root / 'candidate', 'review', 'S01', 1,
                                        self.requirements, {}, [('实际画面 frame_index=0', image)], [0])
                    self.assertEqual(request.call_count, 2 if exception is ValueError else 1)
                    self.assertFalse((root / 'candidate/result.json').exists())
                    self.assertFalse((root / 'final.mp4').exists())
                    self.assertEqual(read_json(root / 'candidate/review.json')['checks'], report['checks'])

    def test_unrecoverable_responses_keep_one_format_retry_and_accumulated_budget(self):
        invalid = self.malformed(self.report(), ['invalid repair'])
        with Project(self.root, {'story': 'test'}) as project, \
                patch.object(model, 'urlopen', side_effect=lambda *a, **k: self.response(invalid)) as request:
            with self.assertRaisesRegex(Blocked, '响应格式连续失败'):
                project.call(self.root, 'review', self.messages)
            self.assertEqual(request.call_count, 2)
            self.assertEqual(project.data['calls'], 2)
            self.assertEqual(read_json(self.root / 'review.call.json')['status'], 'FORMAT_ERROR')
            self.assertFalse(list(self.root.rglob('normalized_response.json')))
        with Project(self.root, {'story': 'test'}, resume=True) as project, \
                patch.object(model, 'urlopen', return_value=self.response(self.malformed(self.report()))) as request:
            self.assertEqual(project.call(self.root, 'review', self.messages)['status'], 'PASS')
            request.assert_called_once()
            self.assertEqual(project.data['calls'], 3)

    def test_old_format_error_cache_is_not_recovered_without_a_new_api_response(self):
        old_raw = self.malformed(self.report('PASS'))
        fresh_raw = self.malformed(self.report('FAIL'))
        with Project(self.root, {'story': 'test'}) as project:
            project.data['calls'] = 1
            old_trace = self.root / 'calls/review/000001'
            old_trace.mkdir(parents=True)
            (old_trace / 'response.txt').write_text(old_raw)
            write_json(old_trace / 'response.json', {'status': 'FORMAT_ERROR'})
            write_json(self.root / 'review.json', dict(self.report('PASS'), repair={}))
            write_json(self.root / 'review.call.json', {
                'status': 'FORMAT_ERROR', 'call_number': 1,
                'input_hash': digest({'messages': self.messages, 'settings': model.settings()}),
                'model': model.settings()})
            before = {path: path.read_bytes() for path in old_trace.iterdir()}
            with patch.object(model, 'urlopen', return_value=self.response(fresh_raw)) as request:
                result = project.call(self.root, 'review', self.messages)
            request.assert_called_once()
            self.assertEqual(result['status'], 'FAIL')
            self.assertEqual(project.data['calls'], 2)
            self.assertEqual(before, {path: path.read_bytes() for path in old_trace.iterdir()})
            self.assertFalse((old_trace / 'normalized_response.json').exists())
            self.assertEqual(read_json(self.root / 'calls/review/000002/normalized_response.json')['status'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
