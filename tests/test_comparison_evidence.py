"""首尾并排证据保持原像素，差分只作度量而不替代严格审核。"""
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zlib

from agent import model, prompts, temporal
from agent.common import file_hash, read_json
from agent.store import Project, Rejected
from fixtures import FakeModel, HTML, REQ, STORY, frames


def write_png(path, width, height, pixels):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    rows = b''.join(b'\0' + pixels[y * width * 4:(y + 1) * width * 4] for y in range(height))
    path.write_bytes(b'\x89PNG\r\n\x1a\n' +
                     chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)) +
                     chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))


def decode(path):
    return subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-frames:v', '1',
                                    '-f', 'rawvideo', '-pix_fmt', 'rgba', '-'], timeout=30)


class ComparisonEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_full_resolution_panels_copy_every_pixel_and_labels_never_cover_sources(self):
        width, height = 1280, 720
        first_pixels = bytes((10, 20, 30, 255)) * width * height
        last_pixels = bytearray(first_pixels)
        for x, y, color in ((13, 7, (11, 20, 30, 255)), (1279, 719, (10, 20, 30, 125))):
            position = (y * width + x) * 4
            last_pixels[position:position + 4] = bytes(color)
        first, last = self.root / 'first.png', self.root / 'last.png'
        write_png(first, width, height, first_pixels)
        write_png(last, width, height, last_pixels)
        original_hashes = file_hash(first), file_hash(last)
        label, output, metadata = temporal.window_comparison(self.root, 'window', [(0, first), (17, last)], 24)
        self.assertEqual(metadata['source_size'], [1280, 720])
        self.assertEqual(metadata['comparison_size'], [2568, 748])
        self.assertEqual(metadata['pixel_difference']['changed_pixels'], 2)
        self.assertEqual(metadata['pixel_difference']['bbox_xyxy'], [13, 7, 1280, 720])
        self.assertEqual(metadata['pixel_difference']['bbox_coordinates'],
                         'original_frame_pixels; right/bottom exclusive')
        actual = decode(output)
        output_width = metadata['comparison_size'][0]
        for panel, expected in zip(metadata['panels'], (first_pixels, bytes(last_pixels))):
            x, y, right, bottom = panel['image_region_xyxy']
            self.assertEqual((right - x, bottom - y), (width, height))
            restored = b''.join(actual[((row + y) * output_width + x) * 4:
                                      ((row + y) * output_width + right) * 4] for row in range(height))
            self.assertEqual(restored, expected)
        band = actual[:output_width * 28 * 4]
        self.assertIn(bytes((255, 255, 255, 255)), band)  # 标签只占顶部独立区域。
        self.assertEqual((file_hash(first), file_hash(last)), original_hashes)
        self.assertIn('frame_index=0', label)
        self.assertIn('frame_index=17', label)
        self.assertIn('排版x偏移不是主体位移', metadata['coordinate_rule'])
        self.assertIn('不能自动证明动作', metadata['interpretation'])
        self.assertIn('固定光源下的刚性平移不要求表面光照必然变化', metadata['interpretation'])
        self.assertNotIn('PASS', json.dumps(metadata))
        self.assertEqual(read_json(self.root / 'window_comparison.json'), metadata)

    def test_equal_frames_report_zero_change_and_mismatched_sources_are_rejected(self):
        first, last = self.root / 'first.png', self.root / 'last.png'
        write_png(first, 2, 2, bytes((30, 40, 50, 255)) * 4)
        _, _, metadata = temporal.window_comparison(self.root, 'equal', [(1, first), (2, first)], 24)
        self.assertEqual(metadata['pixel_difference']['changed_pixels'], 0)
        self.assertIsNone(metadata['pixel_difference']['bbox_xyxy'])
        write_png(last, 3, 2, bytes((30, 40, 50, 255)) * 6)
        with self.assertRaisesRegex(ValueError, '尺寸必须一致'):
            temporal.window_comparison(self.root, 'mismatch', [(1, first), (2, last)], 24)
        last.write_bytes(b'not actual pixels')
        with self.assertRaisesRegex(ValueError, '真实PNG'):
            temporal.window_comparison(self.root, 'invalid', [(1, first), (2, last)], 24)

    def test_comparison_first_then_every_actual_frame_then_references_and_all_checks_remain(self):
        with Project(self.root / 'project', {'story': STORY, 'fps': 24}) as project:
            directory = project.root / 'candidate'
            directory.mkdir()
            (directory / 'scene.html').write_text(HTML)
            images = []
            for index in [0, 1, 3]:
                path = directory / f'frame_{index}.png'
                write_png(path, 4, 3, bytes((index, 20, 30, 255)) * 12)
                images.append((index, path))
            reference = directory / 'reference.png'
            write_png(reference, 4, 3, bytes((50, 60, 70, 255)) * 12)
            fake = FakeModel()
            with patch.object(model, 'ask', side_effect=fake) as ask:
                aggregate = temporal.review_windows(project, directory, 'S01', 1, REQ,
                    {'frame_plan': frames('S01'), 'fps': 24}, [('标准资产 P01 v1', reference)], images)
            calls = ask.call_args_list
            local_messages = calls[0].args[0]
            content = local_messages[1]['content']
            payload = json.loads(content[0]['text'])
            labels = [row['text'] for row in content[1:] if row['type'] == 'text']
            self.assertTrue(labels[0].startswith('实际首尾对照 actual'))
            self.assertEqual(labels[1:4], [f'实际画面 frame_index={index}' for index, _ in images])
            self.assertTrue(labels[4].startswith('标准资产 P01 v1'))
            self.assertIn('reference', labels[4])
            self.assertEqual(len([row for row in content if row['type'] == 'image_url']), 5)
            self.assertEqual([row['frame_index'] for row in payload['sample_timing']], [0, 1, 3])
            self.assertEqual(payload['original_shot_requirements'], REQ)
            self.assertTrue({'LOCAL_STATE', 'LOCAL_CONTINUITY', 'LOCAL_IDENTITY_CONTACT', 'DEPTH_STABILITY'}
                            <= set(payload['required_check_ids']))
            global_payload = json.loads(calls[-1].args[0][1]['content'][0]['text'])
            self.assertEqual(global_payload['requirements'][0], REQ[0])
            self.assertNotIn('endpoint_comparison', global_payload)
            self.assertEqual(aggregate['status'], 'PASS')
            proof = aggregate['windows'][0]['comparison']
            self.assertEqual(proof['image']['hash'], file_hash(project.root / proof['image']['path']))
            self.assertEqual(proof['metadata'], payload['endpoint_comparison'])

    def test_nonzero_change_never_overrides_failed_or_uncertain_review(self):
        for status in ('FAIL', 'UNCERTAIN'):
            with self.subTest(status=status), Project(self.root / status, {'story': STORY, 'fps': 24}) as project:
                directory = project.root / 'candidate'
                directory.mkdir()
                (directory / 'scene.html').write_text(HTML)
                images = []
                for index in (0, 3):
                    path = directory / f'{index}.png'
                    write_png(path, 2, 2, bytes((index, 20, 30, 255)) * 4)
                    images.append((index, path))
                with patch.object(temporal, 'review', return_value={'status': status, 'checks': []}) as review:
                    with self.assertRaises(Rejected):
                        temporal.review_windows(project, directory, 'S01', 1, REQ,
                                                {'frame_plan': frames('S01'), 'fps': 24}, [], images)
                    review.assert_called_once()
                report = read_json(directory / 'temporal_aggregate.json')
                self.assertEqual(report['status'], status)
                self.assertEqual(report['failed_review']['comparison']['metadata']['pixel_difference']['changed_pixels'], 4)
                self.assertNotIn('global', report)


if __name__ == '__main__':
    unittest.main()
