"""有界的完整采样分窗审核；全局验收始终保留全部原始硬要求。"""
from copy import deepcopy
import json
from functools import lru_cache
import struct
import subprocess
import zlib

from . import consistency, model, pixel_motion, rendering, validation
from .common import digest, file_hash, read_json, write_json
from .planning import req, review, validate_review_evidence
from .store import Rejected

PROTOCOL = 'windowed_reasoned_v5'
RELATIVE_MOTION_POLICY_VERSION = 'body_relative_motion_v1'


def relative_motion_policy():
    """显式补证启用的参照定义，不提供动作已发生的判断。"""
    return {
        'version': RELATIVE_MOTION_POLICY_VERSION, 'method': 'same_endpoint_minus_stable_body_anchor',
        'instruction': '部件摆动须始终选择同一可辨末端P与躯干同一稳定可见参照点A，比较相对向量r=P-A。'
            '使用原始全图坐标；先在证据中逐一列出本次每个实际帧的P、A、r及其相位变化，再判断动作。'
            'r的水平分量增加为相对向右，减少为相对向左；不能以尾巴始终位于躯干左侧否定相对左右摆动。'
            '预期方向以已审核计划中本时间范围所在半周期为准，不可把其他半周期的方向强加于本窗；'
            '仅短于完整周期的局部窗口不要求完成全次往返，周期回归相似姿态本身也不证明hold。'
            '全局或事件证据覆盖多个周期时仍必须核对实际交替；单向运动或整体刚性平移不能代替部件摆动。'
            '转向极值附近的合理减速或单个相邻帧对变化微小，不能据此断言整个窗口hold；'
            '仍须结合全部实际帧验证应有运动，不能用极值解释整段不动。'
            '计划文字不能代替实际观测，窗口内相邻实际帧无可辨变化时不能据计划自动PASS；'
            '仍须检查本区间适用的全部动作、连接和体积光影要求。'
            '要求运动时若实际无可见变化须FAIL，无法辨认或证据不足则UNCERTAIN，不能据此规则自动PASS。'}


@lru_cache(maxsize=8)
def rgba_pixels(data):
    """使用现有FFmpeg解码真实PNG；有界缓存只按原文件字节复用。"""
    validation.require(data[:8] == b'\x89PNG\r\n\x1a\n' and data[12:16] == b'IHDR',
                       '首尾对照证据必须来自真实PNG帧')
    width, height = struct.unpack('>II', data[16:24])
    pixels = subprocess.run(['ffmpeg', '-v', 'error', '-i', 'pipe:0', '-frames:v', '1',
                             '-f', 'rawvideo', '-pix_fmt', 'rgba', 'pipe:1'],
                            input=data, capture_output=True, check=True, timeout=30).stdout
    validation.require(len(pixels) == width * height * 4, '首尾对照PNG解码尺寸不一致')
    return width, height, pixels


def rgba_png(width, height, pixels):
    """无损编码原RGBA字节，不缩放、过滤或修改面板内像素。"""
    def chunk(kind, body):
        return struct.pack('>I', len(body)) + kind + body + struct.pack('>I', zlib.crc32(kind + body))
    rows = b''.join(b'\0' + pixels[y * width * 4:(y + 1) * width * 4] for y in range(height))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))


def comparison_label(pixels, width, band, x, text):
    """仅在图外标签带画最小ASCII数字字形，绝不覆盖真实画面。"""
    glyphs = {
        '0': ('111','101','101','101','111'), '1': ('010','110','010','010','111'),
        '2': ('111','001','111','100','111'), '3': ('111','001','111','001','111'),
        '4': ('101','101','111','001','001'), '5': ('111','100','111','001','111'),
        '6': ('111','100','111','101','111'), '7': ('111','001','001','001','001'),
        '8': ('111','101','111','101','111'), '9': ('111','101','111','001','111'),
        'F': ('111','100','110','100','100'), 'R': ('110','101','110','101','101'),
        'A': ('010','101','111','101','101'), 'M': ('101','111','111','101','101'),
        'E': ('111','100','110','100','111'), 'T': ('111','010','010','010','010'),
        'I': ('111','010','010','010','111'), '.': ('000','000','000','000','010'),
    }
    for number, char in enumerate(text):
        for y, row in enumerate(glyphs.get(char, ())):
            for column, value in enumerate(row):
                if value == '1':
                    for dy in range(3):
                        for dx in range(3):
                            px, py = x + number * 12 + column * 3 + dx, 6 + y * 3 + dy
                            if 0 <= px < width and 0 <= py < band:
                                offset = (py * width + px) * 4
                                pixels[offset:offset + 4] = b'\xff\xff\xff\xff'


def window_comparison(directory, name, batch, fps):
    """首尾原像素并排对照及独立RGBA绝对差度量；不产生通过结论。"""
    first, last = batch[0], batch[-1]
    left = rgba_pixels(first[1].read_bytes())
    right = rgba_pixels(last[1].read_bytes())
    width, height, original = left
    validation.require(right[:2] == left[:2], '首尾对照原图尺寸必须一致')
    band, gap = 28, 8
    output_width = width * 2 + gap
    pixels = bytearray(bytes((24, 28, 32, 255)) * output_width * (height + band))
    for y in range(height):
        for offset, source in ((0, original), (width + gap, right[2])):
            start = ((y + band) * output_width + offset) * 4
            pixels[start:start + width * 4] = source[y * width * 4:(y + 1) * width * 4]
    for x, (index, _) in ((4, first), (width + gap + 4, last)):
        comparison_label(pixels, output_width, band, x, f'FRAME {index} TIME {index / fps:.3f}')
    changed, x0, y0, x1, y1 = 0, width, height, -1, -1
    for offset in range(0, len(original), 4):
        if original[offset:offset + 4] != right[2][offset:offset + 4]:
            y, x = divmod(offset // 4, width)
            changed += 1
            x0, y0, x1, y1 = min(x0, x), min(y0, y), max(x1, x), max(y1, y)
    output = directory / f'{name}_comparison.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(rgba_png(output_width, height + band, pixels))
    metadata = {
        'source_size': [width, height], 'comparison_size': [output_width, height + band],
        'panels': [{'frame_index': index, 'time_seconds': index / fps, 'source_hash': file_hash(path),
                    'image_region_xyxy': [x, band, x + width, band + height]}
                   for x, (index, path) in ((0, first), (width + gap, last))],
        'comparison_hash': file_hash(output),
        'pixel_difference': {'method': 'absolute_RGBA_channel_difference_gt_zero',
                             'changed_pixels': changed,
                             'bbox_xyxy': [x0, y0, x1 + 1, y1 + 1] if changed else None,
                             'bbox_coordinates': 'original_frame_pixels; right/bottom exclusive'},
        'coordinate_rule': '比较各面板内部对应原图的坐标；归一化位置用面板内x/原图宽、y/原图高。'
            '右面板的排版x偏移不是主体位移；顶部标签和中间分隔不属于原图，也不参与差分。',
        'interpretation': '首尾对照直接复制真实帧原像素，不是新时间点；先列各可辨部件首末位置与姿态，再判断局部推进。'
            'changed_pixels和bbox仅度量原图像素变化，不能自动证明动作、接触、连续性或体积光影通过。'
            '固定光源下的刚性平移不要求表面光照必然变化；仍须按实际姿态、体积、接触与照明一致性严格审核。'
            '采样时间、hold/motion阶段及连续帧证据规则保持不变，不能从首尾图臆测中间过程。',
    }
    write_json(directory / f'{name}_comparison.json', metadata)
    label = (f'实际首尾对照 actual：左 frame_index={first[0]} time_seconds={first[0] / fps:.6f}；'
             f'右 frame_index={last[0]} time_seconds={last[0] / fps:.6f}；原像素面板，非新增帧')
    return label, output, metadata


def local_requirements():
    """局部验收只检查本时段必须成立的状态；整段结果由全局原要求审核。"""
    return [
        req('LOCAL_STATE', '本窗口每个实际采样帧符合该时段已审核计划的状态与推进，'
            '仅要求本时间范围内适用的动作部分，不要求提前完成窗外剧情。'
            '窗外计划锚点只是预期文本，不得据此断言实际画面或补全看不见的状态。'),
        req('LOCAL_CONTINUITY', '比较本窗口相邻实际采样的局部动作进度和因果顺序，'
            '按计划区分hold与motion时段；动作仅要求本窗适用部分。'
            '采样空档不能当作连续帧，瞬移或闪烁判断须有相邻连续真实帧证据；证据不足则UNCERTAIN。'),
        req('LOCAL_IDENTITY_CONTACT', '本窗口实际画面中的身份保持稳定，分离、对接或连接符合各时刻的已审核计划状态；'
            '应分离阶段不得要求提前连接，应对接阶段必须实际对接，已建立连接不得无故断开。'
            '原镜头要求中本时段适用的身份、接触和连接约束必须成立；标准资产仅作身份与材质参照。'),
    ]


def sample_windows(images, size=12, overlap=2):
    """按实际采样顺序分窗，验证全部帧及相邻采样对均有共同审核窗。"""
    indices = [index for index, _ in images]
    validation.require(indices and indices == sorted(set(indices)), '时序采样必须非空、按帧号排序且不重复')
    validation.require(type(size) is int and type(overlap) is int and 1 <= overlap < size,
                       '时序窗口必须保留重叠采样')
    windows = []
    for start in range(0, len(images), size - overlap):
        windows.append(images[start:start + size])
        if start + size >= len(images):
            break
    covered = {index for window in windows for index, _ in window}
    pairs = {(a[0], b[0]) for window in windows for a, b in zip(window, window[1:])}
    validation.require(covered == set(indices) and set(zip(indices, indices[1:])) <= pairs,
                       '时序窗口遗漏实际帧或跨窗相邻采样对')
    return windows


def representative_images(images, frame_plan, limit=16):
    """先保留所有关键帧与首尾，再填最大时间空隙，始终选真实采样帧。"""
    actual = dict(images)
    selected = {key['frame_index'] for key in frame_plan['keyframes']}
    selected.update((images[0][0], images[-1][0]))
    validation.require(selected <= actual.keys(), '全局时序代表帧缺少计划关键帧的真实采样')
    validation.require(len(selected) <= limit, '关键帧和首尾数量超过全局时序代表帧上限')
    while len(selected) < min(limit, len(images)):
        gaps = []
        ordered = sorted(selected)
        for left, right in zip(ordered, ordered[1:]):
            interior = [index for index in actual if left < index < right and index not in selected]
            if interior:
                middle = min(interior, key=lambda index: (abs(index - (left + right) / 2), index))
                gaps.append((right - left, -left, middle))
        validation.require(gaps, '无法补齐真实时序代表帧')
        selected.add(max(gaps)[2])
    return [(index, actual[index]) for index in sorted(selected)]


def timing(images, fps):
    return [{'frame_index': index, 'time_seconds': index / fps,
             'gap_from_previous_frames': index - images[position - 1][0] if position else None}
            for position, (index, _) in enumerate(images)]


def window_plan(frame_plan, first, last):
    """保留相交动作区间和最近前后锚点；窗外锚点仅是计划文本。"""
    plan = deepcopy(frame_plan)
    keys = frame_plan['keyframes']
    selected = [key for key in keys if first <= key['frame_index'] <= last]
    before = [key for key in keys if key['frame_index'] < first]
    after = [key for key in keys if key['frame_index'] > last]
    if before:
        selected.insert(0, max(before, key=lambda key: key['frame_index']))
    if after:
        selected.append(min(after, key=lambda key: key['frame_index']))
    plan['keyframes'] = deepcopy(selected)
    plan['intervals'] = [deepcopy(interval) for interval in frame_plan['intervals']
                         if interval['start_frame'] <= last and interval['end_frame'] > first]
    plan['critical_frames'] = [index for index in frame_plan.get('critical_frames', []) if first <= index <= last]
    return plan



def detail_regions(descriptor):
    """验证并规范化显式真实区域；排序保证同一区域换顺序不能重复复验。"""
    validation.require(isinstance(descriptor, dict), 'temporal_detail_recheck 必须是对象')
    validation.require('relative_motion_reference' not in descriptor or
                       type(descriptor['relative_motion_reference']) is bool,
                       'relative_motion_reference 必须是布尔值')
    if 'pixel_motion' in descriptor:
        pixel_motion.validate_spec(descriptor['pixel_motion'])
    target = descriptor.get('target_review', '')
    validation.require(isinstance(target, str) and target.startswith('temporal_window_') and
                       target.removeprefix('temporal_window_').isdecimal(), '细节复验必须指定实际时序窗口')
    regions = descriptor.get('regions')
    validation.require(isinstance(regions, list) and 1 <= len(regions) <= 6, '细节复验必须提供1至6个真实区域')
    result = []
    for row in regions:
        validation.require(isinstance(row, dict) and type(row.get('frame_index')) is int and
                           row['frame_index'] >= 0, '细节复验帧号必须是非负整数')
        region = row.get('region_xyxy')
        validation.require(isinstance(region, list) and len(region) == 4 and
                           all(type(v) in (int, float) and 0 <= v <= 1 for v in region) and
                           region[0] < region[2] and region[1] < region[3], '细节复验区域坐标无效')
        result.append({'frame_index': row['frame_index'], 'region_xyxy': [float(v) for v in region]})
    result.sort(key=lambda row: (row['frame_index'], row['region_xyxy']))
    validation.require(len({digest(row) for row in result}) == len(result), '细节复验区域不能重复')
    return result


def pixel_motion_context(spec, measurements):
    return {'spec': spec, 'measurements': measurements,
            'instruction': '数据由本窗口实际PNG像素的确定性图像块匹配计算，坐标为原图像素坐标；'
                '只辅助定位同一图像块的近似位移，不是动作、身份、接触或剧情通过结论。'
                '匹配可能误认相似纹理、背景或变形部件，必须结合全部实际全图、区域图和采样顺序'
                '确认每个匹配点属于哪个可见对象，以及位移是否代表要求的语义动作。'
                '误差与像素变化不能自动证明任何检查项通过；仍逐项审核全部原始要求，'
                '明确违反须FAIL，关键事实证据不足须UNCERTAIN。'}


def carried_detail_evidence(project, directory, shot_id, windows, review_options):
    """只继承同源码、同帧、同审核配置下正式通过的补充像素，不继承结论。"""
    result = {}
    source_hash = file_hash(directory / 'scene.html')
    effective = model.effective_settings(review_options)
    policy = consistency.review_policy(getattr(project, 'review_strictness', 'strict'))
    batches = {f'temporal_window_{number:03d}': batch for number, batch in enumerate(windows, 1)}

    def bound_path(row, within):
        path = (project.root / row['path']).resolve()
        validation.require(path.is_relative_to(within) and path.is_file() and file_hash(path) == row['hash'],
                           '历史补充证据文件绑定无效')
        return path

    def binding(path):
        return {'path': str(path.resolve().relative_to(project.root)), 'hash': file_hash(path)}

    for path in sorted(directory.parent.glob('v*/temporal_detail_recheck_provenance.json'), reverse=True):
        if path.parent.name >= directory.name:
            continue
        try:
            proof = read_json(path)
            name = proof['target_review']
            if name in result or name not in batches or proof['source_hash'] != source_hash:
                continue
            regions = detail_regions(proof)
            relative_flag = ({'relative_motion_reference': proof['relative_motion_reference']}
                             if 'relative_motion_reference' in proof else {})
            motion_spec = ({'pixel_motion': pixel_motion.validate_spec(proof['pixel_motion'])}
                           if 'pixel_motion' in proof else {})
            if proof['request_hash'] != digest({'source_hash': source_hash, 'target_review': name,
                                                'regions': regions, **relative_flag, **motion_spec}):
                continue
            source = (project.root / proof['source']).resolve()
            if (not source.is_relative_to(project.root) or not source.is_file() or
                    file_hash(source) != source_hash or file_hash(path.parent / 'scene.html') != source_hash):
                continue
            failed_path = bound_path(proof['failed_report'], project.root)
            if read_json(failed_path).get('status') not in {'FAIL', 'UNCERTAIN'}:
                continue
            aggregate_path = path.parent / 'temporal_aggregate.json'
            aggregate = read_json(aggregate_path)
            if (aggregate.get('temporal_evidence_protocol') != PROTOCOL or
                    aggregate.get('target_id') != shot_id or
                    bound_path(aggregate['source'], path.parent) != path.parent / 'scene.html'):
                continue
            row = next(row for row in aggregate['windows'] if row['name'] == name and row['status'] == 'PASS')
            indices = [index for index, _ in batches[name]]
            if row['frame_indices'] != indices or [item['frame_index'] for item in row['images']] != indices:
                continue
            historical_frames = {item['frame_index']: item for item in row['images']}
            for item, (index, current) in zip(row['images'], batches[name]):
                bound_path(item, path.parent)
                validation.require(item['hash'] == file_hash(current), '当前窗口与历史实际帧不同')
            artifacts = {item['path']: item for item in row['artifacts']}
            files = {key: path.parent / (name + suffix) for key, suffix in (
                ('passed_report', '.json'), ('request', '.request.json'), ('call', '.call.json'),
                ('detail_evidence', '_detail_evidence.json'))}
            if motion_spec:
                files['pixel_motion_measurement'] = path.parent / (name + '_pixel_motion.json')
            for artifact in files.values():
                bound_path(artifacts[str(artifact.relative_to(project.root))], path.parent)
            report, request, call = (read_json(files[key]) for key in ('passed_report', 'request', 'call'))
            context = json.loads(request[1]['content'][0]['text'])
            if (call.get('status') != 'PASS' or call.get('model') != effective or report.get('status') != 'PASS' or
                    context.get('consistency_policy') != policy or
                    context.get('target_id') != shot_id or context.get('target_version') != report.get('target_version') or
                    context.get('temporal_evidence_protocol') != PROTOCOL or
                    context.get('temporal_review_stage') != 'window' or
                    context['temporal_window']['actual_frame_indices'] != indices or
                    context.get('visual_review_extra_body') != review_options):
                continue
            if not validation.review(report, shot_id, context['target_version'], context['requirements']):
                continue
            validate_review_evidence(report, indices, context)
            recorded = deepcopy(request)
            stable_context = deepcopy(context)
            stable_context.pop('target_version', None)
            recorded[1]['content'][0]['text'] = json.dumps(stable_context, ensure_ascii=False, sort_keys=True)
            if row.get('evidence_signature') != digest({'messages': recorded, 'model': effective}):
                continue
            detail = read_json(files['detail_evidence'])
            rows = proof['detail_images']
            if (detail.get('source_hash') != source_hash or detail.get('target_review') != name or
                    rows != detail.get('images') or rows != row.get('detail_images') or
                    detail_regions({'target_review': name, 'regions': rows}) != regions):
                continue
            neutral, bound_regions = [], []
            initial_regions = {digest(region): original for region, original in zip(regions, proof['regions'])}
            for item in rows:
                region = {'frame_index': item['frame_index'], 'region_xyxy': item['region_xyxy']}
                original = initial_regions[digest(region)]
                frame = historical_frames[item['frame_index']]
                source_frame = bound_path(item['source_frame'], path.parent)
                crop = bound_path(item['crop'], path.parent)
                validation.require(item['source_frame'] == {'path': frame['path'], 'hash': frame['hash']} and
                                   original['source_frame_hash'] == frame['hash'], '历史裁剪与全图绑定不一致')
                neutral.append(dict(region, source_frame_hash=file_hash(source_frame), crop_hash=file_hash(crop)))
                bound_regions.append(dict(region, source_frame_hash=file_hash(source_frame)))
            if context.get('temporal_region_details', {}).get('images') != neutral:
                continue
            uses_relative = context.get('relative_motion_reference') == relative_motion_policy()
            if (context.get('relative_motion_reference') not in (None, relative_motion_policy()) or
                    ('relative_motion_reference' in proof and proof['relative_motion_reference'] != uses_relative)):
                continue
            motion_binding, measured_context = {}, None
            if motion_spec:
                spec = motion_spec['pixel_motion']
                motion_frames = proof['pixel_motion_frames']
                if [item['frame_index'] for item in motion_frames] != spec['frame_indices']:
                    continue
                source_frames = {}
                for item in motion_frames:
                    frame = historical_frames[item['frame_index']]
                    if item['hash'] != frame['hash']:
                        raise ValueError('测量帧与正式窗口实际帧不一致')
                    bound_path(item, project.root)
                    source_frames[item['frame_index']] = bound_path(frame, path.parent)
                measured = pixel_motion.measure(spec, source_frames)
                measurement_path = files['pixel_motion_measurement']
                if (proof.get('pixel_motion_measurement') != binding(measurement_path) or
                        measured != read_json(measurement_path)):
                    continue
                measured_context = pixel_motion_context(spec, measured)
                if context.get('pixel_motion_evidence') != measured_context:
                    continue
                motion_binding = dict(motion_spec, pixel_motion_frames=motion_frames)
            elif 'pixel_motion_evidence' in context:
                continue
            result[name] = {'details': {'target_review': name, 'source_hash': source_hash,
                'regions': bound_regions, **relative_flag, **motion_binding},
                'neutral_context': context['temporal_region_details'],
                **({'pixel_motion_context': measured_context} if motion_spec else {}),
                'provenance': {'original_provenance': binding(path), 'aggregate': binding(aggregate_path),
                    **{key: binding(value) for key, value in files.items()},
                    'actual_frames': row['images'], 'detail_images': rows,
                    'review_policy': policy, 'effective_options': effective,
                    'relative_motion_reference': uses_relative}}
        except (OSError, ValueError, KeyError, TypeError, StopIteration, AttributeError):
            continue  # 不完整、篡改或非正式PASS的历史证据不能取得继承资格。
    return result


def review_windows(project, directory, shot_id, version, requirements, payload, refs, images, details=None):
    """每窗严格PASS后才全局审核；所有请求、报告及像素记录到汇总证据。"""
    windows = sample_windows(images)
    representatives = representative_images(images, payload['frame_plan'])
    fps = payload['fps']
    review_options = model.visual_review_options()
    payload = dict(payload, visual_review_extra_body=review_options)
    carried = carried_detail_evidence(project, directory, shot_id, windows, review_options)
    if details is not None:
        carried.pop(details['target_review'], None)  # 当前显式补证始终优先。
        if details.get('relative_motion_reference') is False:
            carried = {key: value for key, value in carried.items()
                       if not value['provenance']['relative_motion_reference']}
    if any(value['provenance']['relative_motion_reference'] for value in carried.values()):
        payload = dict(payload, relative_motion_reference=relative_motion_policy())
    carry_proof = {'source_hash': file_hash(directory / 'scene.html'), 'targets': {}}
    if details is not None:
        requested = detail_regions(details)
        validation.require(details.get('source_hash') == file_hash(directory / 'scene.html'),
                           '细节复验源码哈希不匹配')
        batches = {f'temporal_window_{number:03d}': dict(batch) for number, batch in enumerate(windows, 1)}
        validation.require(details['target_review'] in batches, '细节复验目标窗口不存在')
        actual = batches[details['target_review']]
        bound = {digest({'frame_index': row['frame_index'], 'region_xyxy': row['region_xyxy']}): row
                 for row in details['regions']}
        for region in requested:
            validation.require(region['frame_index'] in actual, '细节复验帧不属于目标窗口实际采样')
            validation.require(file_hash(actual[region['frame_index']]) ==
                               bound[digest(region)].get('source_frame_hash'), '细节复验实际帧哈希已改变')
        if 'pixel_motion' in details:
            spec = pixel_motion.validate_spec(details['pixel_motion'])
            proof = read_json(directory / 'temporal_detail_recheck_provenance.json')
            validation.require(proof.get('pixel_motion') == spec and
                               proof.get('pixel_motion_frames') == details.get('pixel_motion_frames') and
                               proof.get('source_hash') == details['source_hash'] and
                               proof.get('target_review') == details['target_review'], '像素测量来源绑定不匹配')
            frames = details['pixel_motion_frames']
            validation.require([row['frame_index'] for row in frames] == spec['frame_indices'],
                               '像素测量帧清单与规范方案不一致')
            for row in frames:
                frame = row['frame_index']
                validation.require(frame in actual and file_hash(actual[frame]) == row['hash'],
                                   '像素测量当前帧与绑定实际帧不一致')
        if details.get('relative_motion_reference'):
            proof = read_json(directory / 'temporal_detail_recheck_provenance.json')
            validation.require(proof.get('relative_motion_reference') is True and
                               proof['source_hash'] == details['source_hash'] and
                               proof['target_review'] == details['target_review'], '相对运动参照来源绑定不匹配')
            payload = dict(payload, relative_motion_reference=relative_motion_policy())

    def binding(path):
        return {'path': str(path.resolve().relative_to(project.root)), 'hash': file_hash(path)}

    def evidence(batch):
        return [{'frame_index': index, **binding(path)} for index, path in batch]

    aggregate = {'temporal_evidence_protocol': PROTOCOL, 'target_id': shot_id, 'target_version': version,
                 'status': 'RUNNING', 'fps': fps, 'visual_review_extra_body': review_options,
                 'relative_motion_reference': payload.get('relative_motion_reference') == relative_motion_policy(),
                 'source': binding(directory / 'scene.html'),
                 'actual_samples': evidence(images),
                 'references': [{'label': label, **binding(path)} for label, path in refs],
                 'representative_frames': [index for index, _ in representatives], 'windows': []}
    aggregate_path = directory / 'temporal_aggregate.json'
    write_json(aggregate_path, aggregate)

    def run(name, batch, context, check_requirements, comparison=None):
        row = {'name': name, 'frame_indices': [index for index, _ in batch],
               'frame_range': [batch[0][0], batch[-1][0]], 'images': evidence(batch), 'status': 'RUNNING'}
        actual_images = [(f'实际画面 frame_index={index}', path) for index, path in batch]
        review_images = refs + actual_images
        if comparison is not None:
            label, path, metadata = comparison
            row['comparison'] = {'image': binding(path), 'metadata': metadata}
            review_images = [(label, path)] + actual_images + [
                (label + '（reference，仅身份与材质；非当前实际帧）', path) for label, path in refs]
        inherited = carried.get(name)
        region_details = details if details is not None and name == details['target_review'] else (
            inherited['details'] if inherited else None)
        if region_details is not None:
            actual = dict(batch)
            rows, neutral = [], []
            for number, region in enumerate(region_details['regions'], 1):
                frame, box = region['frame_index'], region['region_xyxy']
                source = actual[frame]
                crop = directory / (name + '_detail_evidence') / f'crop_{number:02d}.png'
                rendering.crop_evidence(source, crop, box)
                label = (f'区域细节 actual：原图 frame_index={frame}，region_xyxy={box}；'
                         '同帧真实像素裁剪放大，非新帧；所有归一化坐标仍对应原始全图，不能使用裁剪图坐标')
                review_images.append((label, crop))
                rows.append({'frame_index': frame, 'region_xyxy': box,
                             'source_frame': binding(source), 'crop': binding(crop)})
                neutral.append({'frame_index': frame, 'region_xyxy': box,
                                'source_frame_hash': file_hash(source), 'crop_hash': file_hash(crop)})
            row['detail_images'] = rows
            context = dict(context, temporal_region_details={
                'images': neutral,
                'instruction': '区域图来自本窗口已提供的实际全图，仅增加可辨像素细节。'
                    '逐项检查原始全部硬要求，所有全图、首尾对照及其他采样仍须审核；'
                    '裁剪不是新时间点，不能替代全图空间关系，不可从看不清处补全动作；证据不足则UNCERTAIN。'})
            write_json(directory / (name + '_detail_evidence.json'), {
                'target_review': name, 'source_hash': region_details['source_hash'], 'images': rows})
            if inherited:
                validation.require(context['temporal_region_details'] == inherited['neutral_context'],
                                   '重新裁剪的实际像素或中性上下文改变，不能继承历史补充证据')
                carry_proof['targets'][name] = dict(inherited['provenance'],
                    current_detail_images=rows, requires_full_review=True)
                write_json(directory / 'temporal_detail_carry_provenance.json', carry_proof)
            else:
                provenance = directory / 'temporal_detail_recheck_provenance.json'
                proof = read_json(provenance)
                validation.require(proof['source_hash'] == region_details['source_hash'] and proof['target_review'] == name,
                                   '细节复验来源记录不匹配')
                validation.require(proof.get('relative_motion_reference', False) is
                                   region_details.get('relative_motion_reference', False), '相对运动参照标志已改变')
                proof['detail_images'] = rows
                write_json(provenance, proof)
            if 'pixel_motion' in region_details:
                spec = pixel_motion.validate_spec(region_details['pixel_motion'])
                bound_frames = {item['frame_index']: item for item in region_details['pixel_motion_frames']}
                for frame in spec['frame_indices']:
                    validation.require(frame in actual and file_hash(actual[frame]) == bound_frames[frame]['hash'],
                                       '像素测量当前实际帧绑定已改变')
                measurements = pixel_motion.measure(spec, {frame: actual[frame] for frame in spec['frame_indices']})
                measurement_path = directory / (name + '_pixel_motion.json')
                write_json(measurement_path, measurements)
                measured_context = pixel_motion_context(spec, measurements)
                context = dict(context, pixel_motion_evidence=measured_context)
                row['pixel_motion_measurement'] = binding(measurement_path)
                if inherited:
                    validation.require(inherited.get('pixel_motion_context') == measured_context,
                                       '重新测量的实际数据改变，不能继承历史像素测量证据')
                    carry_proof['targets'][name]['current_pixel_motion_measurement'] = binding(measurement_path)
                    write_json(directory / 'temporal_detail_carry_provenance.json', carry_proof)
                else:
                    proof['pixel_motion_measurement'] = binding(measurement_path)
                    write_json(provenance, proof)
        try:
            report = review(project, directory, name, shot_id, version, check_requirements, context,
                            review_images,
                            [index for index, _ in batch], visual_scope='temporal', extra_body=review_options)
            if report.get('status') != 'PASS' or any(
                    check.get('result') in {'FAIL', 'UNCERTAIN'} for check in report.get('checks', [])):
                raise Rejected(report)
        except Rejected as error:
            row['status'] = error.report['status']
            row['artifacts'] = [binding(path) for path in sorted(directory.glob(name + '*.json'))]
            aggregate['failed_review'] = row
            aggregate['status'] = row['status']
            write_json(aggregate_path, aggregate)
            raise
        row['status'] = report['status']
        row['artifacts'] = [binding(path) for path in sorted(directory.glob(name + '*.json'))]
        # 仅请求证据摘要进入全局上下文；不携带局部自由文本结论，也不引入候选号波动。
        recorded = read_json(directory / (name + '.request.json'))
        request_context = json.loads(recorded[1]['content'][0]['text'])
        request_context.pop('target_version', None)
        recorded[1]['content'][0]['text'] = json.dumps(request_context, ensure_ascii=False, sort_keys=True)
        row['evidence_signature'] = digest({'messages': recorded,
                                            'model': model.effective_settings(review_options)})
        return row

    summaries = []
    for number, batch in enumerate(windows, 1):
        first, last = batch[0][0], batch[-1][0]
        local_plan = window_plan(payload['frame_plan'], first, last)
        name = f'temporal_window_{number:03d}'
        comparison = window_comparison(directory, name, batch, fps)
        context = dict(payload, temporal_evidence_protocol=PROTOCOL, temporal_review_stage='window',
                       endpoint_comparison=comparison[2],
                       frame_plan=local_plan, sample_timing=timing(batch, fps),
                       original_shot_requirements=deepcopy(requirements),
                       temporal_window={'number': number, 'count': len(windows),
                           'frame_range': [first, last], 'actual_frame_indices': [index for index, _ in batch],
                           'actual_full_frame_count': len(batch),
                           'context_only_anchor_indices': [key['frame_index'] for key in local_plan['keyframes']
                                                          if not first <= key['frame_index'] <= last]},
                       verification_scope='逐帧检查本窗口全部实际采样及相邻样本之间的连续性、身份、接触、体积和照明。'
                           '本窗口按LOCAL_STATE、LOCAL_CONTINUITY、LOCAL_IDENTITY_CONTACT及体积光影硬检查验收。'
                           'original_shot_requirements保留完整原要求作为适用约束，不要求窗内完成全段剧情或窗外终态；'
                           '不得要求本窗口提前完成窗外剧情，也不得豁免本区间适用的硬要求。'
                           '该时刻应有的身份、分离、对接、连接、接触和稳定性在实际帧不满足时必须FAIL；'
                           '计划尚处于分离或靠近阶段时，不能因尚未完成连接而判失败。'
                           '窗外最近前后锚点只是计划文本，不是已提供的实际画面；不能用它们证明动作正常。'
                           '标准资产只用于身份与材质参照，不能代替当前实际动作或接触证据。'
                           '本窗PASS不代表整段剧情通过，全部窗口之后另做保留完整原要求的全局验收。')
        row = run(name, batch, context, local_requirements(), comparison)
        aggregate['windows'].append(row)
        summaries.append({'number': number, 'frame_range': row['frame_range'],
                          'frame_indices': row['frame_indices'], 'status': row['status'],
                          'image_hashes': [item['hash'] for item in row['images']],
                          'evidence_signature': row['evidence_signature']})
        write_json(aggregate_path, aggregate)
    global_context = dict(payload, temporal_evidence_protocol=PROTOCOL, temporal_review_stage='global',
                          sample_timing=timing(representatives, fps),
                          temporal_window_evidence=summaries,
                          representative_sampling_context='附图是全部关键帧、首尾及按最大时间空隙补充的真实代表帧。'
                              '窗口记录仅证明各局部证据已审核，不能代替当前全局像素判定，也不能自动使任何要求PASS。'
                              '仍须完整核对全部原要求、动作顺序和结果及体积光影；证据不足则UNCERTAIN。'
                              '完整实际采样帧数与引用图、同帧局部放大图分别统计；放大图不是额外时间点。',
                          actual_full_frame_count=len(representatives))
    aggregate['global'] = run('temporal_review', representatives, global_context, requirements)
    aggregate['status'] = 'PASS'
    write_json(aggregate_path, aggregate)
    return aggregate
