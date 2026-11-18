"""确定性 Canvas 渲染、镜头拼接与编码检查。"""
import base64
import json
import hashlib
import struct
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import WIDTH, HEIGHT, file_hash, log, write_json, read_json


class RenderError(RuntimeError):
    """候选动画无法渲染，可将诊断送回当前镜头修复。"""

def render_consistency_mode(value):
    if value not in {'exact', 'tolerant'}:
        raise ValueError('render consistency 必须为 exact 或 tolerant')
    return value


def png_dimensions(data):
    if len(data) < 33 or data[:8] != b'\x89PNG\r\n\x1a\n' or data[12:16] != b'IHDR':
        raise ValueError('一致性比较必须使用有效 PNG 画面')
    size = struct.unpack('>II', data[16:24])
    if min(size) < 1:
        raise ValueError('一致性比较画面尺寸无效')
    return size


def decode_rgba(data, size):
    pixels = subprocess.run(['ffmpeg', '-v', 'error', '-i', 'pipe:0', '-frames:v', '1',
                             '-f', 'rawvideo', '-pix_fmt', 'rgba', 'pipe:1'],
                            input=data, capture_output=True, check=True, timeout=30).stdout
    if len(pixels) != size[0] * size[1] * 4:
        raise ValueError('一致性比较解码长度与画面尺寸不符')
    return pixels


def compare_render_frames(expected, actual, consistency='exact', expected_size=None):
    """只容忍极小抗锯齿差异；局部最大差防止小型明显运动被全图均值掩盖。"""
    mode = render_consistency_mode(consistency)
    first_size, second_size = png_dimensions(expected), png_dimensions(actual)
    result = {'mode': mode, 'byte_identical': expected == actual,
              'expected_sha256': hashlib.sha256(expected).hexdigest(),
              'actual_sha256': hashlib.sha256(actual).hexdigest(),
              'expected_size': list(first_size), 'actual_size': list(second_size)}
    if first_size != second_size or (expected_size is not None and first_size != tuple(expected_size)):
        return dict(result, status='FAIL', reason='dimensions_mismatch')
    if mode == 'exact':
        return dict(result, status='PASS' if result['byte_identical'] else 'FAIL',
                    reason='identical_bytes' if result['byte_identical'] else 'bytes_changed')
    limits = {'max_channel_difference': 8, 'mean_absolute_channel_difference': 0.02,
              'significant_channel_difference': 2, 'significant_pixel_ratio': 0.001}
    changed = significant = total_difference = maximum = 0
    if not result['byte_identical']:
        left, right = decode_rgba(expected, first_size), decode_rgba(actual, first_size)
        for offset in range(0, len(left), 4):
            differences = [abs(left[offset + i] - right[offset + i]) for i in range(4)]
            pixel_maximum = max(differences)
            total_difference += sum(differences)
            maximum = max(maximum, pixel_maximum)
            changed += pixel_maximum > 0
            significant += pixel_maximum > limits['significant_channel_difference']
    count = first_size[0] * first_size[1]
    metrics = {'max_channel_difference': maximum,
               'mean_absolute_channel_difference': total_difference / (count * 4),
               'changed_pixels': changed, 'significant_pixels': significant,
               'significant_pixel_ratio': significant / count}
    passed = (maximum <= limits['max_channel_difference'] and
              metrics['mean_absolute_channel_difference'] <= limits['mean_absolute_channel_difference'] and
              metrics['significant_pixel_ratio'] <= limits['significant_pixel_ratio'])
    return dict(result, status='PASS' if passed else 'FAIL',
                reason='within_pixel_tolerance' if passed else 'pixel_tolerance_exceeded',
                limits=limits, metrics=metrics)


def render(html, out, seconds, fps, preview=False, frame_count=None, sample_indices=None,
           consistency='exact'):
    """只渲染本地 HTML；模型不能执行 Python，也接触不到 API Key。"""
    from playwright.sync_api import sync_playwright
    consistency = render_consistency_mode(consistency)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / 'render_consistency.json'
    report = read_json(report_path) if report_path.exists() else {'version': 'pixel_consistency_v1', 'runs': []}
    if report.get('version') != 'pixel_consistency_v1' or not isinstance(report.get('runs'), list):
        raise ValueError('已有渲染一致性记录格式无效，不能覆盖')
    run = {'run_id': time.time_ns(), 'source_sha256': file_hash(html), 'mode': consistency,
           'preview_only': preview, 'status': 'RUNNING', 'comparisons': []}
    report['runs'].append(run)
    write_json(report_path, report)

    def checked_frame(expected, actual, index, phase, message):
        try:
            comparison = compare_render_frames(expected, actual, consistency, (WIDTH, HEIGHT))
        except (ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            run['status'] = 'ERROR'
            run['comparisons'].append({'frame_index': index, 'phase': phase,
                                       'status': 'ERROR', 'error': str(error)[:500]})
            write_json(report_path, report)
            raise
        run['comparisons'].append(dict(comparison, frame_index=index, phase=phase))
        if comparison['status'] != 'PASS':
            run['status'] = 'FAIL'
        write_json(report_path, report)
        if comparison['status'] != 'PASS':
            raise ValueError(message + f'（{consistency}: {comparison["reason"]}）')
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
        # 固定读回模式；像素容差由独立 consistency 配置决定，不重置动画自身状态。
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
            checked_frame(samples[i], frame(times[i]), indices[i], 'reverse_repaint',
                          f"frame({times[i]:.2f}) 重画不一致，请去掉累计状态。")
        # 先校验全部旧预览，再写新文件，避免后续失败留下部分替换的审核证据。
        if not preview:
            for i, data in enumerate(samples):
                path = out / f"preview_{i:02d}.png"
                if path.exists():
                    checked_frame(path.read_bytes(), data, indices[i], 'reload_preview',
                                  "重新加载后预览不一致，请去掉初始化时的随机数或时钟。")
        run['status'] = 'PASS'
        write_json(report_path, report)
        for i, data in enumerate(samples):
            path = out / f"preview_{i:02d}.png"
            # 已审核预览保留原始字节；重载产生的实际帧仍用于后续视频及成片验收。
            if preview or not path.exists():
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


def render_job(html, out, seconds, fps, preview=False, frame_count=None, sample_indices=None,
               consistency='exact'):
    consistency = render_consistency_mode(consistency)
    # 子进程让错误动画（如死循环）也有总超时；不把 API Key 交给浏览器进程。
    command = [sys.executable, "-u", str(Path(__file__).with_name("video_agent.py").resolve()), "render", str(html),
               "--out", str(out), "--seconds", str(seconds), "--fps", str(fps),
               "--render-consistency", consistency, "--worker"]
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



def compact_review_images(directory, images, max_bytes=12 * 1024 * 1024):
    """大批次仅换用原尺寸高质量 JPEG 传输副本；保留全部帧和原始证据。"""
    images = list(images)
    if sum(Path(path).stat().st_size for _, path in images) <= max_bytes:
        return images
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    encoder = {'codec': 'mjpeg', 'q': 2, 'pixel_format': 'yuvj444p',
               'resize': False, 'crop': False}
    manifest_path = directory / 'manifest.json'
    prior_manifest = read_json(manifest_path) if manifest_path.exists() else {}
    prior_rows = prior_manifest.get('images', []) if prior_manifest.get('encoder') == encoder else []

    def probe(path):
        data = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name,width,height,pix_fmt', '-of', 'json', str(path)],
            timeout=30))
        stream = data['streams'][0]
        return stream

    result, records = [], []
    for index, (label, original) in enumerate(images):
        original = Path(original).resolve()
        original_hash = file_hash(original)
        output = (directory / f'{index:04d}.jpg').resolve()
        cached = prior_rows[index] if index < len(prior_rows) else {}
        reusable = (cached.get('label') == label and
                    cached.get('original_path') == str(original) and
                    cached.get('original_sha256') == original_hash and
                    cached.get('encoded_path') == str(output) and output.is_file() and
                    cached.get('encoded_sha256') == file_hash(output))
        if reusable:
            record = cached
        else:
            original_info = probe(original)
            subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(original),
                '-map', '0:v:0', '-frames:v', '1', '-c:v', 'mjpeg', '-q:v', '2',
                '-pix_fmt', 'yuvj444p', '-update', '1', str(output)], check=True, timeout=30)
            encoded_info = probe(output)
            if (encoded_info['width'], encoded_info['height']) != (
                    original_info['width'], original_info['height']):
                raise RenderError('审核传输副本尺寸改变，拒绝使用。')
            if encoded_info.get('codec_name') != 'mjpeg' or encoded_info.get('pix_fmt') != 'yuvj444p':
                raise RenderError('审核传输副本编码格式不符合原尺寸 JPEG 4:4:4 要求。')
            record = {'label': label, 'original_path': str(original), 'original_sha256': original_hash,
                      'encoded_path': str(output), 'encoded_sha256': file_hash(output),
                      'width': original_info['width'], 'height': original_info['height'],
                      'encoder': encoder}
        records.append(record)
        # 精确帧标签供审核、局部证据及修复代码解析，传输说明由 model.messages 单独附上。
        result.append((label, output))
    write_json(manifest_path, {'encoder': encoder, 'images': records})
    log('传输', f'{len(images)} 张审核图保留原尺寸，使用 JPEG 4:4:4 副本')
    return result


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
