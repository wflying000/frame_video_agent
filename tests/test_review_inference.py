"""复杂视觉审核的推理参数阶段路由；模型与渲染均为本地测试替身。"""
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import model, prompts, workflow
from agent.common import file_hash, read_json
from agent.store import Project
from fixtures import FakeModel, STORY
import test_workflow as workflow_tests


ENABLED = {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high'}
DEFAULT_BODY = {'thinking': {'type': 'disabled'}, 'max_tokens': 32768}


class ReviewInferenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / 'out'
        for context in (patch.dict(os.environ, {
                'LLM_BASE_URL': 'https://api.deepseek.com', 'LLM_MODEL': 'deepseek-flash',
                'LLM_EXTRA_BODY': json.dumps(DEFAULT_BODY)}),
                patch.object(model, 'load_dotenv'), patch('sys.stdout', io.StringIO())):
            context.start()
            self.addCleanup(context.stop)

    def generate(self):
        fake, calls, inputs = FakeModel(), [], {}
        original_task = Project.task

        def ask(messages, **kwargs):
            payload = json.loads(messages[1]['content'][0]['text'])
            calls.append((messages[0]['content'], payload, copy.deepcopy(kwargs.get('extra_body'))))
            return fake(messages)

        def task(project, task_id, task_inputs, *args, **kwargs):
            inputs[task_id] = copy.deepcopy(task_inputs)
            return original_task(project, task_id, task_inputs, *args, **kwargs)

        with patch.object(model, 'ask', side_effect=ask), \
                patch.object(Project, 'task', new=task), \
                workflow_tests.WorkflowTests().fake_renderers():
            result = workflow.generate(STORY, self.root, 4, 2)
        return result, calls, inputs

    def test_all_shot_pixel_review_stages_receive_the_override(self):
        result, calls, _ = self.generate()
        self.assertTrue(result.exists())
        seen = set()
        for system, payload, options in calls:
            target = payload.get('target_id', '')
            stage = payload.get('temporal_review_stage')
            if system == prompts.REVIEW and stage in {'window', 'global'}:
                group, expected = f'temporal_{stage}', ENABLED
            elif system == prompts.REVIEW and target == 'S01_S02':
                group, expected = 'boundary', ENABLED
            elif system == prompts.REVIEW and target.endswith('_appearance'):
                group, expected = 'final_appearance', ENABLED
            elif system == prompts.REVIEW and target in {'E01', 'E02'}:
                group, expected = 'final_event', ENABLED
            elif system == prompts.OBSERVE:
                group, expected = 'observer', None
            elif system == prompts.REVIEW and target == 'final_story':
                group, expected = 'story_only', None
            elif system == prompts.REVIEW and '_K' in target:
                group, expected = 'keyframe', ENABLED
                self.assertEqual(payload['keyframe_review_protocol'], 'keyframe_reasoned_v3')
            elif system == prompts.REVIEW and target == 'P01':
                group, expected = 'asset', None
            elif system in {prompts.STORY, prompts.VISUAL, prompts.STORYBOARD, prompts.FRAME_PLAN} or (
                    system == prompts.REVIEW and (target.startswith('plans/') or target == 'animatic')):
                group, expected = 'plan', None
            else:
                group, expected = 'other_default', None
            seen.add(group)
            with self.subTest(group=group, target=target):
                self.assertEqual(options, {**DEFAULT_BODY, **expected} if expected is not None else None)
                if expected is not None:
                    self.assertEqual(payload['visual_review_extra_body'], ENABLED)
        self.assertTrue({'temporal_window', 'temporal_global', 'boundary', 'final_appearance',
                         'final_event', 'observer', 'story_only', 'keyframe', 'asset', 'plan'} <= seen)
        self.assertEqual(model.settings()['extra_body'], DEFAULT_BODY)

    def test_stage_inputs_and_artifacts_record_options_and_same_config_resumes(self):
        result, _, inputs = self.generate()
        for task_id in ['segments/S01', 'segments/S02', 'checks/timeline', 'exports/final']:
            with self.subTest(task_id=task_id):
                self.assertEqual(inputs[task_id]['visual_review_extra_body'], ENABLED)
                if task_id.startswith('segments/'):
                    self.assertEqual(inputs[task_id]['review_protocol'], 'keyframe_reasoned_v3')
        for task_id, payload in inputs.items():
            if task_id.startswith(('plans/', 'assets/')):
                self.assertNotIn('visual_review_extra_body', payload)
        for shot_id in ['S01', 'S02']:
            for filename in ['frame_tasks.json', 'temporal_aggregate.json']:
                artifact = read_json(self.root / f'segments/{shot_id}/v001' / filename)
                self.assertEqual(artifact['visual_review_extra_body'], ENABLED)
                if filename == 'frame_tasks.json':
                    self.assertEqual(artifact['review_protocol'], 'keyframe_reasoned_v3')
        sampling = read_json(self.root / 'exports/final/v001/final_sampling.json')
        self.assertEqual(sampling['visual_review_extra_body'], ENABLED)
        before = read_json(self.root / 'manifest.json')
        checksum = file_hash(result)
        with patch.object(model, 'ask', side_effect=AssertionError('相同推理配置恢复不应重新调用模型')) as ask, \
                workflow_tests.WorkflowTests().fake_renderers():
            resumed = workflow.generate(STORY, self.root, 4, 2, resume=True)
        ask.assert_not_called()
        after = read_json(self.root / 'manifest.json')
        self.assertEqual(after['calls'], before['calls'])
        self.assertEqual(file_hash(resumed), checksum)
        self.assertEqual({key: len(node['attempts']) for key, node in after['tasks'].items()},
                         {key: len(node['attempts']) for key, node in before['tasks'].items()})


if __name__ == '__main__':
    unittest.main()
