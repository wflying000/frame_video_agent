"""确定性 Canvas 渲染、镜头拼接与编码检查。"""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import WIDTH, HEIGHT, log, write_json, read_json


class RenderError(RuntimeError):
    """候选动画无法渲染，可将诊断送回当前镜头修复。"""

def render(html, out, seconds, fps, preview=False, frame_count=None, sample_indices=None):
    """只渲染本地 HTML；模型不能执行 Python，也接触不到 API Key。"""
    from playwright.sync_api import sync_playwright
    out.mkdir(parents=True, exist_ok=True)
    errors = []
    log("渲染", "启动 Chromium，加载动画页面")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={"width": WIDTH, "height": HEIGHT},
                                      service_workers="block", accept_downloads=False)
        context.route("**/*", lambda route: route.abort())
        context.route_web_socket("**/*", lambda route: route.close())
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        # 必须在生成代码首次 getContext 前设置。Chromium 在反复读回 PNG 后
        # 可能从 GPU 切换到 CPU，导致同一条抗锯齿边缘的像素发生变化。
        # 固定读回模式，不放宽后面的逐字节一致性检查，也不重置动画自身状态。
        page.evaluate("""() => {
            const getContext = HTMLCanvasElement.prototype.getContext;
            HTMLCanvasElement.prototype.getContext = function(type, options) {
                return getContext.call(this, type, type === '2d'
                    ? {...options, willReadFrequently: true} : options);
            };
        }""")
        page.set_content(html.read_text(encoding="utf-8"), wait_until="load")
        page.evaluate("async () => { await window.film.ready; await document.fonts.ready; }")

        def frame(t):
            encoded = page.evaluate("""async t => {
                await window.film.frame(t);
                const c = document.getElementById('stage');
                if (c.width !== 1280 || c.height !== 720) throw Error('画布尺寸必须为1280x720');
                return c.toDataURL('image/png').split(',')[1];
            }""", t)
            if errors:
                raise ValueError("; ".join(errors))
            return base64.b64decode(encoded)

        frame_count = frame_count or seconds * fps
        indices = sorted(set(sample_indices if sample_indices is not None else
                             [round(i * (frame_count - 1) / 11) for i in range(12)]))
        if not indices or any(type(i) is not int or not 0 <= i < frame_count for i in indices):
            raise ValueError("采样帧号必须落在镜头的整数帧区间内。")
        times = [i / fps for i in indices]
        samples = []
        for i, t in enumerate(times):
            samples.append(frame(t))
            log("预览", f"采样 {i + 1}/{len(times)}，frame={indices[i]}，t={t:.2f}s")
        log("预览", "逆序重画，检查画面一致性")
        for i in reversed(range(len(times))):
            if frame(times[i]) != samples[i]:
                raise ValueError(f"frame({times[i]:.2f}) 重画不一致，请去掉累计状态。")
        for i, data in enumerate(samples):
            path = out / f"preview_{i:02d}.png"
            if not preview and path.exists() and path.read_bytes() != data:
                raise ValueError("重新加载后预览不一致，请去掉初始化时的随机数或时钟。")
            path.write_bytes(data)
        write_json(out / "preview_manifest.json", [
            {"frame_index": index, "path": f"preview_{i:02d}.png"}
            for i, index in enumerate(indices)])
        log("预览", f"{len(times)} 张预览帧已保存，一致性检查通过")
        if not preview:
            folder = out / "frames"
            folder.mkdir(exist_ok=True)
            for i in range(frame_count):
                (folder / f"{i:06d}.png").write_bytes(frame(i / fps))
                if (i + 1) % fps == 0 or i + 1 == frame_count:
                    log("渲染", f"{i + 1}/{frame_count} 帧（{(i + 1) / frame_count:.0%}）")
        browser.close()
    if not preview:
        temp = out / "video.part.mp4"
        log("编码", "FFmpeg 开始合成 H.264 MP4")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps),
                        "-i", str(out / "frames" / "%06d.png"), "-frames:v", str(frame_count),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(temp)], check=True, timeout=180)
        temp.replace(out / "video.mp4")
        log("编码", f"视频已保存：{out / 'video.mp4'}")


def render_job(html, out, seconds, fps, preview=False, frame_count=None, sample_indices=None):
    # 子进程让错误动画（如死循环）也有总超时；不把 API Key 交给浏览器进程。
    command = [sys.executable, "-u", str(Path(__file__).with_name("video_agent.py").resolve()), "render", str(html),
               "--out", str(out), "--seconds", str(seconds), "--fps", str(fps), "--worker"]
    env = {k: v for k, v in os.environ.items() if not any(x in k.upper() for x in ("KEY", "TOKEN", "SECRET"))}
    command += ["--preview"] if preview else []
    if frame_count is not None:
        command += ["--frames", str(frame_count)]
    if sample_indices is not None:
        command += ["--samples", ",".join(map(str, sample_indices))]
    started = time.monotonic()
    # stdout 直接继承终端，实时显示进度；仅收集 stderr 供失败后的修正反馈使用。
    result = subprocess.run(command, env=env, stderr=subprocess.PIPE, text=True,
                            timeout=180 if preview else 900)
    if result.returncode:
        raise RenderError(result.stderr[-2500:] or f"渲染子进程退出：{result.returncode}")
    log("渲染", f"{'预览' if preview else '视频导出'}完成，耗时 {time.monotonic() - started:.1f}s")




def preview_images(directory, offset=0):
    return [(row['frame_index'] + offset, directory / row['path'])
            for row in read_json(directory / 'preview_manifest.json')]


def crop_evidence(source, output, region):
    """从真实帧裁切并放大审核区域，仅作辅助证据；全图仍一并提供。"""
    import math
    from .validation import require
    require(isinstance(region, list) and len(region) == 4 and
            all(type(v) in (int, float) and 0 <= v <= 1 for v in region) and
            region[0] < region[2] and region[1] < region[3], '修复区域坐标无效')
    x0 = max(0, math.floor(region[0] * WIDTH) - 12)
    y0 = max(0, math.floor(region[1] * HEIGHT) - 12)
    x1 = min(WIDTH, math.ceil(region[2] * WIDTH) + 12)
    y1 = min(HEIGHT, math.ceil(region[3] * HEIGHT) + 12)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(source), '-vf',
                    f'crop={x1-x0}:{y1-y0}:{x0}:{y0},scale=512:512:force_original_aspect_ratio=decrease',
                    '-frames:v', '1', str(output)], check=True, timeout=30)


def encode_segments(paths, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    listing = output.with_suffix('.concat.txt')
    listing.write_text(''.join("file '" + str(Path(path).resolve()).replace("'", "'\\''") + "'\n"
                               for path in paths), encoding='utf-8')
    log('合成', f'按时间线拼接 {len(paths)} 段视频')
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0',
                    '-i', str(listing), '-c', 'copy', '-movflags', '+faststart', str(output)],
                   check=True, timeout=180)


def boundary_changes(left_video, right_video, left_count, right_count, context=3):
    """从实际编码视频量化切点前后变化；仅为视觉审核提供线索，不自动判定剪切合法性。"""
    size = 160 * 90 * 3
    def decode(path, start, stop):
        expression = f'select=between(n\\,{start}\\,{stop-1}),scale=160:90'
        raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-vf', expression,
            '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], timeout=180)
        if len(raw) != (stop - start) * size:
            raise RenderError('边界诊断解码缺帧')
        return [raw[i:i + size] for i in range(0, len(raw), size)]
    left = decode(left_video, max(0, left_count - context), left_count)
    right = decode(right_video, 0, min(context, right_count))
    frames = left + right
    changes = [sum(abs(a - b) for a, b in zip(previous, current)) / size
               for previous, current in zip(frames, frames[1:])]
    seam = len(left) - 1
    adjacent = changes[:seam] + changes[seam + 1:]
    maximum = max(adjacent, default=0)
    return {'method': 'decoded RGB mean absolute difference at 160x90 (0–255)',
            'pair_changes': changes, 'boundary_pair_index': seam, 'boundary_change': changes[seam],
            'neighbor_max_change': maximum,
            'boundary_to_neighbor_ratio': changes[seam] / maximum if maximum else None,
            'instruction': '该数值只定位异常变化，不能单凭阈值判断失败或通过；'
                '必须结合实际相邻帧和分镜判断变化是否合理。'}


def inspect_video(path, frame_count, fps, allowed_black_ranges=()):
    """检查实际解码帧数、尺寸、帧率、时长、时间戳和未声明黑屏。"""
    from fractions import Fraction
    import re
    from .validation import require
    result = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams', '-show_frames',
        '-show_entries', 'stream=width,height,r_frame_rate:frame=best_effort_timestamp_time',
        '-of', 'json', str(path)], timeout=180))
    stream = result['streams'][0]
    frames = result['frames']
    require((stream['width'], stream['height']) == (WIDTH, HEIGHT), '编码分辨率不匹配')
    require(Fraction(stream['r_frame_rate']) == fps, '编码帧率不匹配')
    require(len(frames) == frame_count, '解码后帧数不匹配')
    require(all(abs(float(frame['best_effort_timestamp_time']) - i / fps) < 0.002
                for i, frame in enumerate(frames)), '时间戳不连续或帧时序错误')
    decoded = subprocess.run(['ffmpeg', '-v', 'info', '-xerror', '-i', str(path),
                              '-vf', 'blackframe=amount=99:threshold=24', '-f', 'null', '-'],
                             text=True, capture_output=True, timeout=180, check=True)
    # YUV limited-range 的纯黑亮度是 16；逐帧检查，避免区间末端的半开边界误报。
    black = sorted(set(int(i) for i in re.findall(r'\bframe:(\d+)\s+pblack:', decoded.stderr)))
    for index in black:
        require(any(a <= index < b for a, b in allowed_black_ranges),
                f'发现计划外黑屏：frame={index}，t={index / fps:.3f}s')
    return {'passed': True, 'width': WIDTH, 'height': HEIGHT, 'fps': fps,
            'frame_count': frame_count, 'duration': frame_count / fps, 'black_frames': black}


def sample_video(path, out, indices):
    out.mkdir(parents=True, exist_ok=True)
    indices = sorted(set(indices))
    expression = '+'.join(f'eq(n\\,{i})' for i in indices)
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(path), '-vf', f'select={expression}',
                    '-vsync', '0', '-start_number', '0', str(out / 'preview_%04d.png')],
                   check=True, timeout=180)
    rows = [{'frame_index': index, 'path': f'preview_{i:04d}.png'} for i, index in enumerate(indices)]
    if any(not (out / row['path']).exists() for row in rows):
        raise ValueError('成片采样缺帧')
    write_json(out / 'preview_manifest.json', rows)
    return [(row['frame_index'], out / row['path']) for row in rows]
