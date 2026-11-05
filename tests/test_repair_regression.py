"""修复回退时保留长单行代码的局部差异及绑定原审核的历史像素。"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from agent import model, repair
from agent.common import write_json


class RepairRegressionTests(unittest.TestCase):
    def test_large_single_line_retains_small_exact_changes(self):
        repeated = ''.join(f'function part{i}(){{draw({i},0.5);}}' for i in range(1000))
        before = repeated + 'function background(){var farA=0.92;drawFar(farA);}'
        after = repeated + 'function background(){var farA=0.16;drawFar(farA);}'
        delta = repair.scene_changes(before, after, 'old', 'new')
        self.assertLess(len(delta), 24000)
        spans = json.loads(delta)['exact_text_changes']
        self.assertTrue(any('farA=0.92' in row['before'] and 'farA=0.16' in row['after'] for row in spans))
        self.assertTrue(all(row['before'] in before and row['after'] in after for row in spans))
        self.assertEqual(repair.scene_changes(before, before, 'old', 'new'), '')

    def historical_frame(self, root):
        directory = root / 'segments/S01/v001'
        folder = directory / 'keyframes/K01'
        folder.mkdir(parents=True)
        image = folder / 'preview_00.png'
        image.write_bytes(b'original approved frame pixels')
        write_json(folder / 'preview_manifest.json', [{'frame_index': 0, 'path': image.name}])
        request = model.recorded_messages(model.messages('', {}, [('实际画面 frame_index=0', image)]))
        write_json(directory / 'check_K01.request.json', request)
        return directory, image

    def test_historical_image_must_match_request_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, image = self.historical_frame(root)
            project = SimpleNamespace(root=root)
            key = {'frame_id': 'K01', 'frame_index': 0}
            evidence = repair.passed_frame_evidence(project, directory, key, 'check_K01')
            self.assertEqual(len(evidence), 1)
            self.assertEqual(root / evidence[0]['path'], image)
            image.write_bytes(b'altered frame')
            self.assertEqual(repair.passed_frame_evidence(project, directory, key, 'check_K01'), [])

    def test_too_large_diff_does_not_discard_verified_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, image = self.historical_frame(root)
            (directory / 'scene.html').write_text('a' * 25000)
            write_json(directory / 'check_K01.json', {'status': 'PASS'})
            current = root / 'current.html'
            current.write_text('b' * 25000)
            previous = {'version': 2, 'report': {'target_id': 'S01_K01', 'status': 'FAIL'}}
            attempts = [{'version': 1, 'path': 'segments/S01/v001'}, previous]
            project = SimpleNamespace(root=root, data={'tasks': {'segments/S01': {'attempts': attempts}}})
            result = repair.regression_context(project, {'shot_id': 'S01'},
                {'keyframes': [{'frame_id': 'K01', 'frame_index': 0}]}, previous, current)
            self.assertEqual(result['changes_since_pass'], '')
            self.assertEqual(root / result['passed_images'][0]['path'], image)


if __name__ == '__main__':
    unittest.main()
