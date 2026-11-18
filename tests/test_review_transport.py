"""大批量审核图压缩仅影响传输，保留分辨率、全部帧、原图证据和硬门槛。"""
import base64
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import model, planning, rendering
from agent.common import file_hash, read_json
from agent.store import Project, Rejected
from fixtures import FakeModel, REQ


class ReviewTransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def image(self, name='original.png'):
        path = self.root / name
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i',
                        'testsrc2=size=320x180', '-frames:v', '1', '-update', '1', str(path)],
                       check=True, timeout=30)
        return path

    def test_small_calls_preserve_exact_paths_labels_and_png_message(self):
        source = self.root / 'small.png'
        source.write_bytes(b'small actual png')
        images = [('实际画面 frame_index=8', source)]
        directory = self.root / 'transport'
        with patch.object(rendering.subprocess, 'run', side_effect=AssertionError('不应编码')):
            self.assertEqual(rendering.compact_review_images(directory, images, max_bytes=source.stat().st_size), images)
        self.assertFalse(directory.exists())
        content = model.messages('test', {}, images)[1]['content']
        self.assertEqual(len(content), 3)
        self.assertEqual(content[1]['text'], images[0][0])
        self.assertEqual(content[2]['image_url']['url'],
                         'data:image/png;base64,' + base64.b64encode(source.read_bytes()).decode())

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), '需要 FFmpeg')
    def test_real_transport_keeps_every_image_full_size_and_provenance(self):
        source = self.image()
        original_bytes = source.read_bytes()
        images = [(f'实际画面 frame_index={i}', source) for i in [0, 7, 23, 48]]
        images.insert(0, ('标准资产 C02', source))
        directory = self.root / 'candidate' / 'review_transport'
        encoded = rendering.compact_review_images(directory, images, max_bytes=1)
        self.assertEqual([label for label, _ in encoded], [label for label, _ in images])
        self.assertEqual(len(encoded), len(images))
        self.assertEqual(source.read_bytes(), original_bytes)
        manifest = read_json(directory / 'manifest.json')
        self.assertEqual(len(manifest['images']), len(images))
        for (label, path), record in zip(encoded, manifest['images']):
            self.assertEqual(record['label'], label)
            self.assertEqual(record['original_path'], str(source.resolve()))
            self.assertEqual(record['original_sha256'], file_hash(source))
            self.assertEqual(record['encoded_sha256'], file_hash(path))
            self.assertEqual((record['width'], record['height']), (320, 180))
            self.assertEqual(record['encoder']['q'], 2)
            self.assertFalse(record['encoder']['crop'])
            self.assertFalse(record['encoder']['resize'])
            self.assertTrue(path.read_bytes().startswith(b'\xff\xd8'))
            info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
                'stream=width,height,pix_fmt,codec_name', '-of', 'json', str(path)], timeout=30))['streams'][0]
            self.assertEqual((info['width'], info['height'], info['pix_fmt'], info['codec_name']),
                             (320, 180, 'yuvj444p', 'mjpeg'))
        with patch.object(rendering.subprocess, 'run', side_effect=AssertionError('应复用哈希匹配副本')):
            self.assertEqual(rendering.compact_review_images(directory, images, max_bytes=1), encoded)
        content = model.messages('test', {}, encoded)[1]['content']
        actual_labels = [row['text'] for row in content if row['type'] == 'text' and
                         row['text'].startswith('实际画面 frame_index=')]
        self.assertEqual(actual_labels, [label for label, _ in images if label.startswith('实际画面')])
        self.assertEqual(len([row for row in content if row['type'] == 'image_url']), len(images))
        self.assertTrue(all(row['image_url']['url'].startswith('data:image/jpeg;base64,')
                            for row in content if row['type'] == 'image_url'))
        self.assertTrue(any('原分辨率保留' in row.get('text', '') for row in content))

    def test_jpeg_extension_uses_correct_mime_without_changing_source(self):
        for suffix in ('.jpg', '.jpeg', '.JPEG'):
            path = self.root / ('frame' + suffix)
            path.write_bytes(b'jpeg-source-payload')
            content = model.messages('test', {}, [('实际画面 frame_index=0', path)])[1]['content']
            self.assertEqual(content[-1]['image_url']['url'],
                'data:image/jpeg;base64,' + base64.b64encode(b'jpeg-source-payload').decode())
            self.assertEqual(content[1]['text'], '实际画面 frame_index=0')
            self.assertEqual(path.read_bytes(), b'jpeg-source-payload')

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), '需要 FFmpeg')
    def test_review_preserves_all_frame_ids_hard_checks_and_rejection(self):
        source = self.image()
        fake = FakeModel()
        compact = rendering.compact_review_images
        def ask(messages, **kwargs):
            report = fake(messages)
            report['status'] = 'FAIL'
            row = next(row for row in report['checks'] if row['requirement_id'] == 'DEPTH_VOLUME')
            row.update(result='FAIL', evidence='实际体积不足')
            return report
        with Project(self.root / 'project', {'story': 'test'}) as project, \
                patch.object(model, 'ask', side_effect=ask) as called, \
                patch.object(rendering, 'compact_review_images', side_effect=lambda d, images: compact(d, images, 1)):
            images = [(f'实际画面 frame_index={i}', source) for i in [0, 3, 9]]
            with self.assertRaises(Rejected):
                planning.review(project, project.root / 'checks/sample/v002', 'review', 'S01', 2,
                    REQ, {}, images, [0, 3, 9], visual_scope='temporal')
            payload = fake.calls[-1][1]
            self.assertEqual(payload['requirements'][0], REQ[0])
            self.assertEqual(payload['visual_depth_frame_indices'], [0, 3, 9])
            self.assertIn('DEPTH_STABILITY', payload['required_check_ids'])
            self.assertEqual(payload['evidence_mode'], 'pixels')
            sent = called.call_args.args[0][1]['content']
            self.assertEqual(sum(row['type'] == 'image_url' for row in sent), 3)
            self.assertTrue((project.root / 'checks/sample/v002/review_transport/manifest.json').exists())
            self.assertFalse((project.root / 'checks/sample/v001').exists())
            self.assertFalse((project.root / 'final.mp4').exists())

if __name__ == '__main__':
    unittest.main()
