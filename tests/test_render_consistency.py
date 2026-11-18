"""像素容差只接受微小抗锯齿变化，不能覆盖已审核预览或隐藏可见运动。"""
import hashlib
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib

from agent import rendering
from agent.common import read_json


def png(width, height, pixels):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    rows = b''.join(b'\0' + pixels[y * width * 4:(y + 1) * width * 4] for y in range(height))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)) +
            chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))


def edited_png(width, height, base, edits):
    pixels = bytearray(bytes(base) * width * height)
    for x, y, color in edits:
        offset = (y * width + x) * 4
        pixels[offset:offset + 4] = bytes(color)
    return png(width, height, pixels)


def scene(rectangle):
    return '''<!doctype html><canvas id="stage" width="1280" height="720"></canvas><script>
const c=document.getElementById('stage'),g=c.getContext('2d');
window.film={duration:2,frame(t){g.fillStyle='rgb(128,128,128)';g.fillRect(0,0,1280,720);
''' + rectangle + '''}};window.film.ready=Promise.resolve();</script>'''


class RenderConsistencyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_sparse_antialias_difference_passes_tolerant_but_exact_rejects(self):
        before = edited_png(64, 64, (128, 128, 128, 255), [])
        after = edited_png(64, 64, (128, 128, 128, 255), [(5, 8, (136, 132, 128, 255))])
        self.assertEqual(rendering.compare_render_frames(before, after, 'exact')['status'], 'FAIL')
        measured = rendering.compare_render_frames(before, after, 'tolerant')
        self.assertEqual(measured['status'], 'PASS')
        self.assertEqual(measured['metrics']['max_channel_difference'], 8)
        self.assertEqual(measured['metrics']['significant_pixels'], 1)
        self.assertEqual(measured['expected_sha256'], hashlib.sha256(before).hexdigest())
        self.assertEqual(measured['actual_sha256'], hashlib.sha256(after).hexdigest())
        self.assertEqual(measured['limits']['significant_pixel_ratio'], .001)
        self.assertEqual(rendering.compare_render_frames(before, before)['status'], 'PASS')

    def test_small_high_contrast_motion_cannot_hide_in_full_frame_mean(self):
        before = edited_png(1280, 720, (0, 0, 0, 255), [(5, 8, (255, 255, 255, 255))])
        after = edited_png(1280, 720, (0, 0, 0, 255), [(6, 8, (255, 255, 255, 255))])
        measured = rendering.compare_render_frames(before, after, 'tolerant')
        self.assertLess(measured['metrics']['mean_absolute_channel_difference'], .02)
        self.assertEqual(measured['metrics']['max_channel_difference'], 255)
        self.assertEqual(measured['status'], 'FAIL')

    def test_broad_low_amplitude_flicker_and_clustered_change_are_rejected(self):
        before = edited_png(64, 64, (128, 128, 128, 255), [])
        flicker = edited_png(64, 64, (129, 129, 129, 255), [])
        measured = rendering.compare_render_frames(before, flicker, 'tolerant')
        self.assertEqual(measured['metrics']['max_channel_difference'], 1)
        self.assertEqual(measured['metrics']['significant_pixels'], 0)
        self.assertGreater(measured['metrics']['mean_absolute_channel_difference'], .02)
        self.assertEqual(measured['status'], 'FAIL')
        clustered = edited_png(64, 64, (128, 128, 128, 255),
                               [(x, y, (131, 131, 131, 255)) for x in range(5) for y in range(5)])
        measured = rendering.compare_render_frames(before, clustered, 'tolerant')
        self.assertLess(measured['metrics']['mean_absolute_channel_difference'], .02)
        self.assertGreater(measured['metrics']['significant_pixel_ratio'], .001)
        self.assertEqual(measured['status'], 'FAIL')

    def test_dimensions_and_mode_are_not_relaxed(self):
        before = edited_png(2, 2, (128, 128, 128, 255), [])
        after = edited_png(3, 2, (128, 128, 128, 255), [])
        for mode in ('exact', 'tolerant'):
            self.assertEqual(rendering.compare_render_frames(before, after, mode)['reason'], 'dimensions_mismatch')
            self.assertEqual(rendering.compare_render_frames(before, before, mode, (1280, 720))['status'], 'FAIL')
        for invalid in ('off', 'balanced', None, ''):
            with self.assertRaises(ValueError):
                rendering.compare_render_frames(before, before, invalid)
        with self.assertRaises(ValueError):
            rendering.compare_render_frames(b'bad PNG', b'bad PNG', 'tolerant')

    def test_worker_receives_explicit_mode_and_still_has_secrets_removed(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-secret', 'SOME_TOKEN': 'test-token'}), \
             patch.object(rendering.subprocess, 'run') as run:
            run.return_value.returncode = 0
            rendering.render_job(self.root / 'scene.html', self.root / 'out', 2, 24,
                                 preview=True, consistency='tolerant')
        command = run.call_args.args[0]
        self.assertEqual(command[command.index('--render-consistency') + 1], 'tolerant')
        self.assertIn('--worker', command)
        self.assertNotIn('OPENAI_API_KEY', run.call_args.kwargs['env'])
        self.assertNotIn('SOME_TOKEN', run.call_args.kwargs['env'])

    def test_failed_reload_preserves_all_previously_reviewed_preview_files(self):
        html, out = self.root / 'scene.html', self.root / 'preview'
        html.write_text(scene(''))
        rendering.render(html, out, 2, 2, preview=True, frame_count=2, sample_indices=[0, 1])
        original = {path.name: path.read_bytes() for path in out.glob('preview_*.png')}
        html.write_text(scene("g.fillStyle=t<.5?'rgb(133,128,128)':'rgb(255,255,255)';g.fillRect(5,8,1,1);"))
        with self.assertRaisesRegex(ValueError, '重新加载后预览不一致'):
            rendering.render(html, out, 2, 2, frame_count=2, sample_indices=[0, 1], consistency='tolerant')
        self.assertEqual(original, {path.name: path.read_bytes() for path in out.glob('preview_*.png')})
        report = read_json(out / 'render_consistency.json')
        self.assertEqual([row['status'] for row in report['runs']], ['PASS', 'FAIL'])
        comparisons = [row for row in report['runs'][-1]['comparisons'] if row['phase'] == 'reload_preview']
        self.assertEqual([row['status'] for row in comparisons], ['PASS', 'FAIL'])
        self.assertEqual(comparisons[-1]['frame_index'], 1)
        self.assertFalse((out / 'video.mp4').exists())

    def test_tolerant_success_preserves_reviewed_pixels_and_encodes_actual_frames(self):
        html, out = self.root / 'scene.html', self.root / 'preview'
        html.write_text(scene(''))
        rendering.render(html, out, 2, 2, preview=True, frame_count=2, sample_indices=[0, 1])
        original = (out / 'preview_00.png').read_bytes()
        html.write_text(scene("g.fillStyle='rgb(133,128,128)';g.fillRect(5,8,1,1);"))
        rendering.render(html, out, 2, 2, frame_count=2, sample_indices=[0, 1], consistency='tolerant')
        self.assertEqual((out / 'preview_00.png').read_bytes(), original)
        self.assertNotEqual((out / 'frames/000000.png').read_bytes(), original)
        self.assertTrue((out / 'video.mp4').is_file())
        report = read_json(out / 'render_consistency.json')
        self.assertEqual(report['runs'][-1]['status'], 'PASS')
        self.assertEqual(report['runs'][-1]['mode'], 'tolerant')
        self.assertEqual(len(report['runs'][-1]['comparisons']), 4)


if __name__ == '__main__':
    unittest.main()
