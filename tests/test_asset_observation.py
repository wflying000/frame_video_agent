"""参考图先独立看图再验收，协议升级不会放松硬门槛或无限复核。"""
import base64
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import drawing, model, prompts, rendering
from agent.common import file_hash, read_json
from agent.store import Blocked, Project
from fixtures import FACTS, STORY, VISUAL, FakeModel
import test_asset_local_patch


class AssetObservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'out'
        self.config = {'story': STORY, 'fps': 2}

    def seed_plans(self, project):
        project.task('plans/story', {}, [], lambda *args: FACTS)
        project.task('plans/visual', {}, ['plans/story'], lambda *args: VISUAL)

    def test_observer_receives_only_actual_pixels_and_neutral_task(self):
        fake = FakeModel()
        with Project(self.root, self.config) as project, \
                patch.object(model, 'ask', side_effect=fake) as ask, \
                patch.object(rendering, 'render_job', side_effect=test_asset_local_patch.AssetLocalPatchTests.render_stub):
            self.seed_plans(project)
            drawing.assets(project, FACTS, VISUAL)
            self.assertEqual([system for system, _ in fake.calls],
                             [prompts.CANVAS, prompts.OBSERVE, prompts.REVIEW])
            observed = fake.calls[1][1]
            self.assertEqual(set(observed), {'frame_indices', 'observation_task', 'output_contract'})
            self.assertEqual(observed['frame_indices'], [0])
            self.assertNotIn(VISUAL['style'], json.dumps(observed, ensure_ascii=False))
            self.assertNotIn(VISUAL['assets'][0]['fixed'], json.dumps(observed, ensure_ascii=False))
            content = ask.call_args_list[1].args[0][1]['content']
            image = next(row for row in content if row['type'] == 'image_url')
            actual = self.root / 'assets/P01/v001/preview/frame.png'
            self.assertEqual(image['image_url']['url'], 'data:image/png;base64,' +
                             base64.b64encode(actual.read_bytes()).decode())
            checked = fake.calls[2][1]
            self.assertEqual(checked['requirements'][0], VISUAL['assets'][0]['requirements'][0])
            self.assertIn('DEPTH_VOLUME', checked['required_check_ids'])
            self.assertEqual(checked['evidence_mode'], 'pixels')
            self.assertEqual(checked['independent_observations'],
                             [{'frame_index': 0, 'description': '实际图像中有球'}])
            folder = self.root / 'assets/P01/v001'
            self.assertTrue((folder / 'reference_observation.request.json').exists())
            self.assertEqual(read_json(folder / 'reference_observation.call.json')['call_number'], 2)

    def test_interrupted_review_reuses_successful_observation(self):
        fake = FakeModel()
        fail = True
        def ask(messages, **kwargs):
            if messages[0]['content'] == prompts.REVIEW and fail:
                raise Blocked('模拟审核中断')
            return fake(messages)
        with Project(self.root, self.config) as project, \
                patch.object(model, 'ask', side_effect=ask), \
                patch.object(rendering, 'render_job', side_effect=test_asset_local_patch.AssetLocalPatchTests.render_stub):
            self.seed_plans(project)
            with self.assertRaises(Blocked):
                drawing.assets(project, FACTS, VISUAL)
            fail = False
            drawing.assets(project, FACTS, VISUAL)
            self.assertEqual([system for system, _ in fake.calls].count(prompts.OBSERVE), 1)
            self.assertEqual(len(project.data['tasks']['assets/P01']['attempts']), 1)
            self.assertEqual(project.data['tasks']['assets/P01']['status'], 'PASS')

    def test_bad_observation_schema_has_two_calls_and_preserves_candidate(self):
        fake = FakeModel()
        def ask(messages, **kwargs):
            value = fake(messages)
            if messages[0]['content'] == prompts.OBSERVE:
                return {'observations': [{'frame_index': 2, 'description': '错误帧号'}]}
            return value
        with Project(self.root, self.config) as project, \
                patch.object(model, 'ask', side_effect=ask), \
                patch.object(rendering, 'render_job', side_effect=test_asset_local_patch.AssetLocalPatchTests.render_stub):
            self.seed_plans(project)
            with self.assertRaisesRegex(Blocked, '字段格式连续无效'):
                drawing.assets(project, FACTS, VISUAL)
            self.assertEqual([system for system, _ in fake.calls].count(prompts.OBSERVE), 2)
            self.assertNotIn(prompts.REVIEW, [system for system, _ in fake.calls])
            self.assertEqual(len(project.data['tasks']['assets/P01']['attempts']), 1)
            self.assertEqual(project.data['calls'], 3)
            self.assertTrue((self.root / 'assets/P01/v001/reference_observation_format_retry.schema_error.json').exists())

    def test_recheck_is_bound_to_old_protocol_source_and_cannot_repeat(self):
        fake = FakeModel()
        with Project(self.root, self.config) as project, \
                patch.object(model, 'ask', side_effect=fake), \
                patch.object(rendering, 'render_job', side_effect=test_asset_local_patch.AssetLocalPatchTests.render_stub):
            self.seed_plans(project)
            # 模拟升级前已保存的候选：拥有源图与审核，没有新协议观察记录。
            with patch.object(drawing, 'observe_reference', return_value=[]):
                drawing.assets(project, FACTS, VISUAL)
            source = self.root / 'assets/P01/v001/reference.html'
            recheck = {'source_hash': file_hash(source), 'reason': '新增中性独立观察协议'}
            project.data['repair_requests']['P01'] = {'reference_recheck': recheck}
            fake.calls.clear()
            drawing.assets(project, FACTS, VISUAL)
            folder = self.root / 'assets/P01/v002'
            self.assertEqual((folder / 'reference.html').read_bytes(), source.read_bytes())
            self.assertEqual([system for system, _ in fake.calls], [prompts.OBSERVE, prompts.REVIEW])
            proof = read_json(folder / 'reference_recheck.json')
            self.assertEqual(proof['provenance'], 'review_protocol_upgrade')
            self.assertTrue(proof['requires_full_review'])
            self.assertEqual(proof['source_hash'], file_hash(source))
            calls = project.data['calls']
            recheck['reason'] = '要求再次复核同一图'
            with self.assertRaisesRegex(Blocked, '已经独立观察审核'):
                drawing.assets(project, FACTS, VISUAL)
            self.assertEqual(project.data['calls'], calls)

    def test_recheck_rejects_wrong_source_hash_before_observer(self):
        fake = FakeModel()
        with Project(self.root, self.config) as project, \
                patch.object(model, 'ask', side_effect=fake), \
                patch.object(rendering, 'render_job', side_effect=test_asset_local_patch.AssetLocalPatchTests.render_stub):
            self.seed_plans(project)
            with patch.object(drawing, 'observe_reference', return_value=[]):
                drawing.assets(project, FACTS, VISUAL)
            project.data['repair_requests']['P01'] = {'reference_recheck':
                {'source_hash': '0' * 64, 'reason': '协议升级'}}
            calls = project.data['calls']
            with self.assertRaisesRegex(Blocked, '源版本不匹配'):
                drawing.assets(project, FACTS, VISUAL)
            self.assertEqual(project.data['calls'], calls)

if __name__ == '__main__':
    unittest.main()
