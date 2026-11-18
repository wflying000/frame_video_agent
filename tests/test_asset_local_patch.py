"""本地参考图修复保留代码来源、哈希绑定和独立美术验收。"""
import base64
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import drawing, model, prompts, rendering
from agent.common import file_hash, read_json, write_json
from agent.store import Blocked, Project
from fixtures import FACTS, STORY, VISUAL, FakeModel


class AssetLocalPatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'out'
        self.config = {'story': STORY, 'fps': 2}

    @staticmethod
    def render_stub(source, out, *args, **kwargs):
        # 该单元测试的图片替身随源代码改变，避免错误复用上一候选的视觉审核。
        out.mkdir(parents=True, exist_ok=True)
        (out / 'frame.png').write_bytes(b'image-placeholder:' + source.read_bytes())
        write_json(out / 'preview_manifest.json', [{'frame_index': 0, 'path': 'frame.png'}])

    def seed(self, project):
        project.task('plans/story', {}, [], lambda *args: FACTS)
        project.task('plans/visual', {}, ['plans/story'], lambda *args: VISUAL)
        drawing.assets(project, FACTS, VISUAL)
        source = project.root / 'assets/P01/v001/reference.html'
        patch_path = project.root / 'repairs/P01.json'
        write_json(patch_path, {'patches': [{'old': "'#ffd166'", 'new': "'#ffaa44'"}]})
        descriptor = {'path': 'repairs/P01.json', 'hash': file_hash(patch_path),
                      'source_hash': file_hash(source)}
        project.data['repair_requests']['P01'] = {'local_patch': descriptor}
        return source, patch_path, descriptor

    def test_asset_patch_skips_generation_but_keeps_pixel_review_and_provenance(self):
        fake = FakeModel()
        with Project(self.root, self.config) as project, \
                patch.object(model, 'ask', side_effect=fake), \
                patch.object(rendering, 'render_job', side_effect=self.render_stub):
            source, _, descriptor = self.seed(project)
            original_hash = file_hash(source)
            prior_calls = project.data['calls']
            fake.calls.clear()
            result = drawing.assets(project, FACTS, VISUAL)
            self.assertEqual(result['P01']['version'], 2)
            self.assertEqual(project.data['calls'], prior_calls + 2)
            self.assertEqual([system for system, _ in fake.calls], [prompts.OBSERVE, prompts.REVIEW])
            payload = fake.calls[1][1]
            self.assertEqual(payload['visual_scope'], 'reference')
            self.assertEqual(payload['evidence_mode'], 'pixels')
            self.assertIn('DEPTH_VOLUME', payload['required_check_ids'])
            folder = project.root / 'assets/P01/v002'
            proof = read_json(folder / 'applied_local_patch.json')
            self.assertEqual(proof['provenance'], 'local_code_edit')
            self.assertTrue(proof['requires_full_review'])
            self.assertEqual(proof['patch'], descriptor)
            self.assertEqual(proof['source_hash'], original_hash)
            self.assertEqual(file_hash(source), original_hash)
            self.assertIn("'#ffaa44'", (folder / 'reference.html').read_text())
            self.assertFalse((folder / 'generated.json').exists())
            calls = project.data['calls']
            drawing.assets(project, FACTS, VISUAL)
            self.assertEqual(project.data['calls'], calls)
            self.assertEqual(len(project.data['tasks']['assets/P01']['attempts']), 2)

    def test_asset_patch_hash_or_source_mismatch_blocks_before_model_call(self):
        for fault in ('patch', 'source'):
            with self.subTest(fault=fault), Project(self.root / fault, self.config) as project, \
                    patch.object(model, 'ask', side_effect=FakeModel()), \
                    patch.object(rendering, 'render_job', side_effect=self.render_stub):
                _, patch_path, descriptor = self.seed(project)
                calls = project.data['calls']
                if fault == 'patch':
                    patch_path.write_text('{}')
                else:
                    descriptor['source_hash'] = '0' * 64
                with self.assertRaises(Blocked):
                    drawing.assets(project, FACTS, VISUAL)
                self.assertEqual(project.data['calls'], calls)
                self.assertFalse((project.root / 'assets/P01/v002/review.json').exists())

    def test_asset_patch_cannot_bypass_failed_volume_review(self):
        fake = FakeModel()

        def ask(messages, **kwargs):
            report = fake(messages)
            if report.get('target_id') == 'P01' and report.get('target_version') == 2:
                report['status'] = 'FAIL'
                check = next(row for row in report['checks'] if row['requirement_id'] == 'DEPTH_VOLUME')
                check.update(result='FAIL', evidence='仍然扁平')
            return report

        with Project(self.root, self.config) as project, \
                patch.object(model, 'ask', side_effect=ask), \
                patch.object(rendering, 'render_job', side_effect=self.render_stub):
            self.seed(project)
            fake.calls.clear()
            with self.assertRaises(Blocked):
                drawing.assets(project, FACTS, VISUAL)
            self.assertFalse((project.root / 'assets/P01/v002/result.json').exists())
            failure = read_json(project.root / 'assets/P01/v002/failure.json')
            self.assertEqual(failure['status'], 'FAIL')
            self.assertEqual([system for system, _ in fake.calls], [prompts.OBSERVE, prompts.REVIEW])
            self.assertFalse((project.root / 'final.mp4').exists())

    def test_asset_review_regions_keep_original_and_crop_only_actual_preview(self):
        fake = FakeModel()

        def crop_stub(source, output, region):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b'crop-placeholder:' + source.read_bytes() + str(region).encode())

        with Project(self.root, self.config) as project, \
                patch.object(model, 'ask', side_effect=fake) as ask, \
                patch.object(rendering, 'render_job', side_effect=self.render_stub), \
                patch.object(rendering, 'crop_evidence', side_effect=crop_stub) as crop:
            _, _, descriptor = self.seed(project)
            regions = [[0.1, 0.6, 0.3, 0.9], [0.6, 0.5, 0.9, 0.8]]
            descriptor['review_regions'] = regions
            drawing.assets(project, FACTS, VISUAL)
            actual = project.root / 'assets/P01/v002/preview/frame.png'
            self.assertEqual(crop.call_count, 2)
            self.assertEqual([call.args[0] for call in crop.call_args_list], [actual, actual])
            self.assertEqual([call.args[2] for call in crop.call_args_list], regions)
            content = ask.call_args.args[0][1]['content']
            labels = [row['text'] for row in content[1:] if row['type'] == 'text']
            self.assertEqual(labels[0], '实际画面 frame_index=0')
            for region, label in zip(regions, labels[1:]):
                self.assertIn('frame_index=0', label)
                self.assertIn(str(region), label)
                self.assertIn('不是新帧', label)
            encoded = [row['image_url']['url'] for row in content if row['type'] == 'image_url']
            paths = [actual] + [call.args[1] for call in crop.call_args_list]
            self.assertEqual(encoded, ['data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode()
                                       for path in paths])
            # 独立观察与验收收到同一全图和真实局部，观察不另带生成意图。
            observation_call = ask.call_args_list[-2].args[0]
            self.assertEqual(observation_call[0]['content'], prompts.OBSERVE)
            observer_content = observation_call[1]['content']
            observer_images = [row['image_url']['url'] for row in observer_content if row['type'] == 'image_url']
            self.assertEqual(observer_images, encoded)
            payload = fake.calls[-1][1]
            self.assertEqual(payload['visual_scope'], 'reference')
            self.assertEqual(payload['requirements'][0], VISUAL['assets'][0]['requirements'][0])
            self.assertIn('DEPTH_VOLUME', payload['required_check_ids'])
            proof = read_json(project.root / 'assets/P01/v002/applied_local_patch.json')
            self.assertEqual(proof['patch']['review_regions'], regions)
            # 无效区域在候选/调用前拒绝，不截断为前两个或放宽坐标。
            calls = project.data['calls']
            for invalid in ([regions[0]] * 3, [[-0.1, 0.1, 0.4, 0.5]], 'invalid'):
                descriptor['review_regions'] = invalid
                with self.assertRaises(ValueError):
                    drawing.assets(project, FACTS, VISUAL)
            self.assertEqual(project.data['calls'], calls)
            self.assertEqual(len(project.data['tasks']['assets/P01']['attempts']), 2)


if __name__ == '__main__':
    unittest.main()
