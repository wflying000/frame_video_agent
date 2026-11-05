"""使用真实编码帧的连续采样，避免边界附近的动作重置被稀疏采样掩盖。"""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent import drawing, model, rendering, repair
from agent.common import file_hash, read_json, write_json
from agent.store import Blocked, Project


class BoundaryChangeTests(unittest.TestCase):
    @staticmethod
    def frames(values):
        return b''.join(bytes([value]) * (160 * 90 * 3) for value in values)

    def test_continuous_motion_and_reset_have_distinct_boundary_metrics(self):
        for right, expected in (([13, 14, 15], 1), ([100, 101, 102], 88)):
            with self.subTest(right=right), patch.object(rendering.subprocess, 'check_output',
                    side_effect=[self.frames([10, 11, 12]), self.frames(right)]) as decode:
                result = rendering.boundary_changes(Path('left.mp4'), Path('right.mp4'), 96, 96)
                self.assertEqual(result['boundary_change'], expected)
                self.assertEqual(result['boundary_to_neighbor_ratio'], expected)
                self.assertEqual(result['boundary_pair_index'], 2)
                self.assertIn('93', decode.call_args_list[0].args[0][6])
                self.assertIn('95', decode.call_args_list[0].args[0][6])
                self.assertNotIn('passed', result)  # 数值不绕过视觉审核，也不自动禁止合理切镜。

    def test_missing_decoded_frames_are_rejected(self):
        with patch.object(rendering.subprocess, 'check_output', return_value=self.frames([10])):
            with self.assertRaisesRegex(rendering.RenderError, '解码缺帧'):
                rendering.boundary_changes(Path('left.mp4'), Path('right.mp4'), 96, 96)


class LocalPatchTests(unittest.TestCase):
    def test_missing_entry_reference_allows_one_bounded_recheck(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            previous = {'path': 'segments/S02/v001', 'report': {
                'target_id': 'S02_K01', 'status': 'UNCERTAIN'}}
            plan = {'keyframes': [{'frame_id': 'K01', 'requirements': [
                {'id': 'ENTRY', 'text': '首帧承接上一镜头的构图'}]}]}
            request = root / previous['path'] / 'check_K01.request.json'
            write_json(request, model.messages('review', {}))
            project = SimpleNamespace(root=root)
            self.assertTrue(repair.needs_entry_confirmation(project, {'shot_id': 'S02'}, plan, previous))
            write_json(request, model.messages('review', {'entry_reference': {'image_hash': 'actual-image'}}))
            self.assertFalse(repair.needs_entry_confirmation(project, {'shot_id': 'S02'}, plan, previous))

    def test_patch_is_bound_to_source_and_preserves_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, patch_path = root / 'old.html', root / 'fix.json'
            source.write_text('frame(t) { return Math.sin(t); }')
            write_json(patch_path, {'patches': [{'old': 'Math.sin(t)', 'new': 'Math.sin(t + 4)'}]})
            descriptor = {'path': 'fix.json', 'hash': file_hash(patch_path),
                          'source_hash': file_hash(source)}
            result = drawing.local_patch(SimpleNamespace(root=root), root / 'candidate', source, descriptor)
            self.assertIn('Math.sin(t + 4)', result)
            self.assertEqual(file_hash(source), descriptor['source_hash'])
            record = read_json(root / 'candidate/applied_local_patch.json')
            self.assertTrue(record['requires_full_review'])
            self.assertEqual(record['provenance'], 'local_code_edit')
            self.assertFalse((root / 'candidate/generated.call.json').exists())
            source.write_text('a different source version')
            with self.assertRaisesRegex(Blocked, '源版本不匹配'):
                drawing.local_patch(SimpleNamespace(root=root), root / 'another', source, descriptor)

    def test_changed_patch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, patch_path = root / 'old.html', root / 'fix.json'
            source.write_text('original')
            write_json(patch_path, {'patches': [{'old': 'original', 'new': 'changed'}]})
            descriptor = {'path': 'fix.json', 'hash': file_hash(patch_path),
                          'source_hash': file_hash(source)}
            write_json(patch_path, {'patches': [{'old': 'original', 'new': 'other'}]})
            with self.assertRaisesRegex(Blocked, '补丁内容已改变'):
                drawing.local_patch(SimpleNamespace(root=root), root / 'candidate', source, descriptor)

    def test_stale_patch_does_not_exhaust_candidate_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            with Project(Path(temporary) / 'project', {'story': 'test'}) as project:
                source, patch_path = project.root / 'scene.html', project.root / 'fix.json'
                source.write_text('original')
                write_json(patch_path, {'patches': [{'old': 'original', 'new': 'changed'}]})
                descriptor = {'path': 'fix.json', 'hash': file_hash(patch_path), 'source_hash': 'outdated'}
                with self.assertRaises(Blocked):
                    project.task('segments/S01', descriptor, [],
                        lambda directory, *_: drawing.local_patch(project, directory, source, descriptor),
                        max_attempts=5)
                self.assertEqual(len(project.data['tasks']['segments/S01']['attempts']), 1)


if __name__ == '__main__':
    unittest.main()
