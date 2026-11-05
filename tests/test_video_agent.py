"""离线测试：模型响应模拟；浏览器绘帧及 FFmpeg 编码实际执行。"""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import rendering as agent

HTML = '''<!doctype html><canvas id="stage" width="1280" height="720"></canvas>
<script>const c=document.getElementById('stage'),ctx=c.getContext('2d');
window.film={duration:2,frame(t){
ctx.resetTransform();ctx.fillStyle='#102030';ctx.fillRect(0,0,1280,720);
ctx.fillStyle='#ffd166';ctx.fillRect(100+t*200,300,80,80);
}};</script>'''


@unittest.skipUnless(importlib.util.find_spec('playwright') and shutil.which('ffmpeg')
                     and shutil.which('ffprobe'), '需要 Playwright、Chromium、FFmpeg')
class RendererTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_declared_black_frames_use_half_open_ranges(self):
        html = self.root / 'black.html'
        html.write_text('''<canvas id="stage" width="1280" height="720"></canvas><script>
const x=document.getElementById('stage').getContext('2d');
window.film={frame(t){x.fillStyle=t<1?'black':'white';x.fillRect(0,0,1280,720);}};</script>''')
        out = self.root / 'black'
        agent.render_job(html, out, 2, 2)
        report = agent.inspect_video(out / 'video.mp4', 4, 2, [[0, 2]])
        self.assertEqual(report['black_frames'], [0, 1])
        with self.assertRaisesRegex(ValueError, '计划外黑屏'):
            agent.inspect_video(out / 'video.mp4', 4, 2)

    def test_real_renderer_produces_four_frames_and_mp4(self):
        html = self.root / 'scene.html'
        html.write_text(HTML)
        out = self.root / 'rendered'
        agent.render_job(html, out, 2, 2)
        self.assertEqual(len(list((out / 'frames').glob('*.png'))), 4)
        self.assertEqual(len(list(out.glob('preview_*.png'))), 4)
        probe = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json',
            str(out / 'video.mp4')]))
        stream = probe['streams'][0]
        self.assertEqual((stream['width'], stream['height'], stream['nb_frames']), (1280, 720, '4'))
        self.assertAlmostEqual(float(probe['format']['duration']), 2.0, places=2)
        self.assertNotEqual((out / 'frames/000000.png').read_bytes(), (out / 'frames/000003.png').read_bytes())

    def test_stateful_animation_is_rejected(self):
        html = self.root / 'stateful.html'
        html.write_text(HTML.replace('window.film=', 'let n=0;window.film=')
                       .replace('100+t*200', '100+(n++)*10'))
        out = self.root / 'bad'
        with self.assertRaisesRegex(RuntimeError, '重画不一致'):
            agent.render_job(html, out, 2, 2, preview=True)
        self.assertFalse((out / 'video.mp4').exists())

    def test_canvas_readback_stays_identical_across_redraw_and_reload(self):
        html = self.root / 'readback.html'
        html.write_text('''<canvas id="stage" width="1280" height="720"></canvas><script>
const ctx=document.getElementById('stage').getContext('2d',{willReadFrequently:false});
if(!ctx.getContextAttributes().willReadFrequently)throw Error('未固定绘制模式');
window.film={frame(t){
ctx.fillStyle='#e8dccb';ctx.fillRect(0,0,1280,720);
ctx.strokeStyle='#a9744c';ctx.lineWidth=3;
for(let i=0;i<6;i++){ctx.beginPath();ctx.moveTo(90+i*220,600);
ctx.lineTo(140+i*220,720);ctx.stroke();}
ctx.save();ctx.beginPath();ctx.rect(288,96,704,388);ctx.clip();
ctx.fillStyle='#a6bbc6';ctx.beginPath();ctx.ellipse(400,160,86,26,0,0,Math.PI*2);ctx.fill();
ctx.restore();
}};</script>''')
        out = self.root / 'readback'
        agent.render_job(html, out, 2, 2, preview=True)
        before = (out / 'preview_00.png').read_bytes()
        agent.render_job(html, out, 2, 2)
        for path in (out / 'frames').glob('*.png'):
            self.assertEqual(path.read_bytes(), before)

    def test_changed_picture_on_reload_is_rejected(self):
        html = self.root / 'scene.html'
        html.write_text(HTML)
        out = self.root / 'rendered'
        agent.render_job(html, out, 2, 2, preview=True)
        # 模拟预览之后、正式加载时画面发生变化，不能导出与审核不同的画面。
        html.write_text(HTML.replace('#102030', '#203040'))
        with self.assertRaisesRegex(RuntimeError, '重新加载后预览不一致'):
            agent.render_job(html, out, 2, 2)
        self.assertFalse((out / 'video.mp4').exists())


if __name__ == '__main__':
    unittest.main()
