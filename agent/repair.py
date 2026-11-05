"""用上一版实际代码和真实局部图像修复，保留已通过的状态与镜头。"""
import re
import difflib
import json
from . import model, rendering, validation
from .common import file_hash, read_json, write_json


def source_candidate(project, task_id, previous):
    if previous is None:
        return None
    for candidate in reversed(project.data['tasks'][task_id]['attempts']):
        if candidate['version'] <= previous['version']:
            directory = project.root / candidate['path']
            if (directory / 'scene.html').exists():
                return candidate
    return None


def history(project, task_id, previous):
    """累计约束防止修复 A 时撤回 B；不能只看上一轮的最后一个错误。"""
    if previous is None:
        return []
    records = []
    for candidate in project.data['tasks'][task_id]['attempts']:
        if candidate['version'] > previous['version'] or not candidate.get('report'):
            continue
        report = candidate['report']
        records.append({'version': candidate['version'], 'target': report.get('target_id'),
                        'error': report.get('error'), 'checks': report.get('checks', []),
                        'repair': report.get('repair', {})})
    return records[-6:]


def evidence(project, directory, reports):
    images, seen = [], set()
    for report in reports:
        regions = []
        focus_frames = set()
        for check in report.get('checks', []):
            region = check.get('region_xyxy')
            if check.get('result') in {'FAIL', 'UNCERTAIN'} and region and region not in regions:
                regions.append(region)
            if check.get('result') in {'FAIL', 'UNCERTAIN'} and region:
                focus_frames.update(check.get('evidence_frames', []))
        # 标准资产由调用方单独提供。修复只需失败帧和首尾上下文，不能把
        # 数十张时序全图各放大两次后再次全部发送，造成请求体和等待时间膨胀。
        candidates = [item for item in report.get('review_images', [])
                      if re.fullmatch(r'实际画面 frame_index=\d+', item['label'])]
        if len(candidates) > 6:
            focused = [item for item in candidates if
                       int(re.search(r'frame_index=(\d+)', item['label'])[1]) in focus_frames]
            if not focused:
                focused = candidates
            selected = [focused[round(i * (len(focused) - 1) / 3)] for i in range(4)]
            selected += [candidates[0], candidates[-1]]
            candidates = [item for item in candidates if item in selected]
        for item in candidates:
            path = project.root / item['path']
            validation.require(file_hash(path) == item['hash'], '修复参考图已改变，不能使用过期审核证据')
            if item['path'] in seen:
                continue
            seen.add(item['path'])
            images.append(('待修复/邻接证据：' + item['label'], path))
            if 'frame_index=' in item['label']:
                for region in regions[:2]:
                    crop = directory / 'repair_evidence' / f'crop_{len(images):03d}.png'
                    rendering.crop_evidence(path, crop, region)
                    images.append((f'原图局部放大，{item["label"]}，region_xyxy={region}；'
                                   '坐标对应原始全图，放大图不是新帧', crop))
    return images


def audit_evidence(directory, requirements, images, max_centers=2):
    """按检查项中的归一化位置补充细节图；不推断物体是否真的在该位置。"""
    centers = []
    for requirement in requirements:
        for x, y in re.findall(r'[（(]\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*[,，]\s*'
                               r'(0(?:\.\d+)?|1(?:\.0+)?)\s*[)）]', requirement['text']):
            center = float(x), float(y)
            if center not in centers:
                centers.append(center)
    extras = []
    for label, source in images:
        if not label.startswith('实际画面 frame_index='):
            continue
        for x, y in centers[:max_centers]:
            region = [max(0, x - 0.075), max(0, y - 0.09), min(1, x + 0.075), min(1, y + 0.09)]
            output = directory / f'zoom_{len(extras):02d}.png'
            rendering.crop_evidence(source, output, region)
            extras.append((f'测量辅助：{label}，关注原图归一化区域 {region}（裁切包含周边像素）；'
                           '所有坐标仍以原图左上角为原点，不能使用放大图自身坐标', output))
    return extras


def temporal_evidence(directory, frame_plan, images):
    """保留时序全景，同时给最多四个动作关键帧提供细节，限制请求大小。"""
    keys = frame_plan.get('keyframes', [])[1:]  # 首帧在逐关键帧审核中已检查。
    if len(keys) > 4:
        keys = [keys[round(i * (len(keys) - 1) / 3)] for i in range(4)]
    result = []
    for key in keys:
        label = f'实际画面 frame_index={key["frame_index"]}'
        actual = [(name, path) for name, path in images if name == label]
        result += audit_evidence(directory / key['frame_id'], key['requirements'], actual, max_centers=1)
    return result


def review_details(directory, report, images, frame_ids):
    """仅为定位明确的关键帧失败增加新像素细节；由调用方限制为一次复核。"""
    actual = {int(label.split('=')[1]): path for label, path in images
              if re.fullmatch(r'实际画面 frame_index=\d+', label)}
    requested = []
    for check in report.get('checks', []):
        region = check.get('region_xyxy')
        if (check.get('result') not in {'FAIL', 'UNCERTAIN'} or
                not isinstance(region, list) or len(region) != 4 or
                not all(type(v) in (int, float) and 0 <= v <= 1 for v in region) or
                region[0] >= region[2] or region[1] >= region[3]):
            continue
        evidence = check.get('evidence_frames')
        if not isinstance(evidence, list):
            continue
        for index in evidence:
            if type(index) is int and index in frame_ids and index in actual:
                item = index, region
                if item not in requested:
                    requested.append(item)
    result = []
    for index, region in requested[:4]:
        output = directory / f'detail_{len(result):02d}.png'
        rendering.crop_evidence(actual[index], output, region)
        result.append((f'失败区域复核：实际画面 frame_index={index}，原图 region_xyxy={region}；'
                       '这是同一帧真实像素的放大图，非新绘制画面，坐标仍以全图为准', output))
    return result


def needs_detail_confirmation(previous):
    """旧版只看全图就失败的关键帧先复核；新版已复核的失败直接修图。"""
    report = (previous or {}).get('report', {})
    if report.get('detail_confirmation') or report.get('status') not in {'FAIL', 'UNCERTAIN'}:
        return False
    actual = {int(item['label'].split('=')[1]) for item in report.get('review_images', [])
              if re.fullmatch(r'实际画面 frame_index=\d+', item['label'])}
    if not 1 <= len(actual) <= 2:
        return False
    return any(check.get('result') in {'FAIL', 'UNCERTAIN'} and
               isinstance(check.get('region_xyxy'), list) and len(check['region_xyxy']) == 4 and
               all(type(v) in (int, float) and 0 <= v <= 1 for v in check['region_xyxy']) and
               check['region_xyxy'][0] < check['region_xyxy'][2] and
               check['region_xyxy'][1] < check['region_xyxy'][3] and
               isinstance(check.get('evidence_frames'), list) and
               any(type(i) is int and i in actual for i in check['evidence_frames'])
               for check in report.get('checks', []))


def motion_comparison_policy(requirements):
    """比较运动的方向与终态方位采用不同参照，不从要求推断审核结论。"""
    ids = [item['id'] for item in requirements
           if re.search(r'相对|相比|对比|前帧|上一帧', item.get('text', '')) and
           re.search(r'左|右|上|下|前|后', item.get('text', '')) and
           re.search(r'摆|移动|位移|运动|抬|落|升|降', item.get('text', ''))]
    if not ids:
        return None
    return {
        'version': 2, 'requirement_ids': ids,
        'instruction': '先区分跨帧运动方向与当前帧的位置方位。'
            '“相对前帧向右下摆”检查同一部位的右向、下向位移，不能追加“末端必须位于躯干右下方”的要求；'
            '向右移动的末端仍可能留在躯干左侧。若原文明确要求“位于某物右下方”，仍须检查当前帧的相对位置。'
            '“由右下摆改为左后上摆”等句式中，起始和结束两处的“摆”都描述运动相位；'
            '不能只对结束相位按位移判断，却把起始“右下摆”改成“前帧末端位于躯干右下方”。'
            '核对当前相对前帧的摆动方向；单独一张前帧的静态方位不能证明或否定此前运动方向。'
            '局部摆动应在两帧定位同一末端P与同一主体稳定锚点A，比较r=P-A，'
            '再计算delta=r当前-r前帧；不能用整个主体或相机的平移冒充局部摆动。'
            '整体移动按要求中的屏幕坐标判断。画面x向右、y向下；右下位移要求delta.x>0且delta.y>0，'
            '变化必须在真实像素中清晰可辨，不能把定位误差当动作。'
            '证据应列出两帧末端和锚点的近似坐标、相对位移及判断理由；'
            '坐标以原始全图为准，不能直接比较不同裁切图的局部坐标。'
            '只凭当前帧相对方位不能否定跨帧方向；其余姿态、位置、身份、背景要求仍逐项检查，'
            '看不清或参照不明确时返回UNCERTAIN，不根据计划文字虚构坐标。',
    }


def needs_motion_confirmation(project, spec, plan, previous):
    """旧方向审核未明确参照系时，仅在下一候选先复核一次原图。"""
    report = (previous or {}).get('report', {})
    if report.get('status') not in {'FAIL', 'UNCERTAIN'}:
        return False
    key = next((key for key in plan['keyframes']
                if report.get('target_id') == f'{spec["shot_id"]}_{key["frame_id"]}'), None)
    policy = motion_comparison_policy(key['requirements']) if key else None
    if not policy or not any(check.get('result') in {'FAIL', 'UNCERTAIN'} and
                             check.get('requirement_id') in policy['requirement_ids']
                             for check in report.get('checks', [])):
        return False
    request = project.root / previous['path'] / f'check_{key["frame_id"]}.request.json'
    if not request.exists():
        return False
    import json
    context = json.loads(read_json(request)[1]['content'][0]['text'])
    scope = context.get('pixel_scope', {})
    return bool(scope.get('comparison_frame_indices')) and scope.get('motion_comparison') != policy


def needs_entry_confirmation(project, spec, plan, previous):
    """首帧要求比较上一镜头却未收到真实参考时，补充证据后只复核一次。"""
    report = (previous or {}).get('report', {})
    key = plan['keyframes'][0]
    if report.get('status') not in {'FAIL', 'UNCERTAIN'} or \
            report.get('target_id') != f'{spec["shot_id"]}_{key["frame_id"]}':
        return False
    if not any(word in item['text'] for item in key['requirements']
               for word in ('上一镜头', '前一镜头')):
        return False
    request = project.root / previous['path'] / f'check_{key["frame_id"]}.request.json'
    if not request.exists():
        return False
    context = json.loads(read_json(request)[1]['content'][0]['text'])
    return not context.get('entry_reference')


def scene_changes(before, after, before_name, after_name, limit=24000):
    """单行 Canvas 代码也保留局部差异，避免整行 diff 超长后丢失修复依据。"""
    if before == after:
        return ''
    changes = ''.join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile=before_name, tofile=after_name, n=4))
    if len(changes) <= limit:
        return changes
    # 仅用于定位文本变化，不解析或改写 JavaScript；片段拼接后仍是原始字节文本。
    chunks = lambda text: re.findall(r'[^;{}]+[;{}]?|[;{}]', text)
    left, right = chunks(before), chunks(after)
    spans = []
    for group in difflib.SequenceMatcher(None, left, right, autojunk=False).get_grouped_opcodes(1):
        spans.append({'before': ''.join(left[group[0][1]:group[-1][2]]),
                      'after': ''.join(right[group[0][3]:group[-1][4]])})
    compact = json.dumps({'from': before_name, 'to': after_name, 'exact_text_changes': spans},
                         ensure_ascii=False, indent=2)
    return compact if len(compact) <= limit else ''


def passed_frame_evidence(project, directory, key, check_name):
    """历史帧须匹配当时实际发送的图片哈希，仅作修复参考。"""
    request = directory / f'{check_name}.request.json'
    preview = directory / 'keyframes' / key['frame_id']
    if not request.exists() or not (preview / 'preview_manifest.json').exists():
        return []
    label = f'实际画面 frame_index={key["frame_index"]}'
    content = read_json(request)[1]['content']
    expected = [content[i + 1].get('image_url', {}).get('sha256')
                for i, item in enumerate(content[:-1]) if item.get('text') == label]
    for index, path in rendering.preview_images(preview):
        if index != key['frame_index']:
            continue
        recorded = model.recorded_messages(model.messages('', {}, [(label, path)]))
        if recorded[1]['content'][-1]['image_url']['sha256'] in expected:
            return [{'label': f'历史通过画面 {directory.name}，{label}；仅用于定位回退，不能代替新候选审核',
                     'path': str(path.relative_to(project.root)), 'hash': file_hash(path)}]
    return []


def regression_context(project, spec, plan, previous, current_html):
    """失败帧曾通过时，提供其通过版本到当前代码的差异，帮助恢复被误删的变换。"""
    report = (previous or {}).get('report', {})
    key = next((key for key in plan['keyframes']
                if report.get('target_id') == f'{spec["shot_id"]}_{key["frame_id"]}'), None)
    if key is None:
        return None
    for candidate in reversed(project.data['tasks'][f'segments/{spec["shot_id"]}']['attempts']):
        if candidate['version'] >= previous['version']:
            continue
        directory = project.root / candidate['path']
        source, check = directory / 'scene.html', directory / f'check_{key["frame_id"]}.json'
        if not source.exists() or not check.exists():
            continue
        accepted = read_json(check)
        check_name = check.stem
        if accepted.get('status') != 'PASS':
            detail = directory / f'check_{key["frame_id"]}_detail.json'
            if not detail.exists() or read_json(detail).get('status') != 'PASS':
                continue
            check_name = detail.stem
        before, after = source.read_text(encoding='utf-8'), current_html.read_text(encoding='utf-8')
        if before == after:
            continue
        changes = scene_changes(before, after, f'passed_v{candidate["version"]}', f'current_v{previous["version"]}')
        images = passed_frame_evidence(project, directory, key, check_name)
        if not changes and not images:
            continue
        return {'target_id': report['target_id'], 'frame_index': key['frame_index'],
                'last_passed_version': candidate['version'],
                'passed_source': str(source.relative_to(project.root)), 'passed_hash': file_hash(source),
                'changes_since_pass': changes,
                'passed_images': images,
                'instruction': '当前失败帧曾在该版本通过。检查差异中被删除的缩放、平移、姿态或图层，'
                    '优先恢复引入回退的局部变化；不要整份回滚，也不要撤销之后已经修好的其他动作。'
                    '新画面仍需按原始要求完整验收。'}


def patch_context(project, directory, spec, frames, previous):
    base = source_candidate(project, f'segments/{spec["shot_id"]}', previous)
    if base is None:
        return None, frames, {}
    old_directory = project.root / base['path']
    plan_path = old_directory / 'frame_tasks.json'
    plan = read_json(plan_path)['frame_plan'] if plan_path.exists() else frames
    validation.frame_plan(plan, spec)
    html = old_directory / 'scene.html'
    context = {'base_version': base['version'], 'base_html': str(html.relative_to(project.root)),
               'base_hash': file_hash(html), 'strategy': 'patch_existing_scene',
               'instruction': '只修改审核指出的动作函数、遮挡层次或局部几何。保持其他已通过内容、'
                   '有效帧计划、相机、背景和身份。动作修复必须在实际像素上清晰可见，'
                   '不能只改变量名、注释、光晕或无关装饰。相邻关键帧应呈现要求的姿态变化。'
                   '同时满足 repair_history 中已有要求和当前反馈；不得把位置改远来修空间关系，'
                   '下一轮又改回位置而撤销空间修复。若位置与空间关系都固定，修正物体之间的'
                   '开口、遮挡和透视表达，使两项同时成立。已通过的接触、姿态变化仍要保留。'}
    context['instruction'] += ('若同一动作反复失败，检查其图形表达是否含糊：落地爪应有清楚的承重面，'
        '收爪和转头应有足够可见的轮廓变化，避免只移动少量像素或给原来朝前的爪垫换色。'
        '以实际附图核对反馈，不能照抄反馈中与像素不符的描述；仍须满足全部原始检查项。')
    context['instruction'] += ('同一主体改变姿态时不要交叉淡化两幅完整身体来替代动作，否则会出现'
        '双头和半透明肢体。用同一组身体、头和肢体的位置/形状插值以及明确遮挡完成过渡；'
        '需要替换绘制组件时，在轮廓重合或被遮挡的时刻切换，保持实体和动作连续。')
    context['instruction'] += ('时序问题必须修复整个失败区间，不能只清理一张关键帧，'
        '也不能把重影挪到两个采样点之间。若反馈指出重复主体，移除旧/新整身的成对透明度混合，'
        '必要时为该区间重写单一主体的过渡绘制函数；区间外已通过的帧应保持原像素。')
    context['instruction'] += ('修复背景显隐时分别控制远景、中景与近景；“仅见远景”仍要求远景可辨，'
        '不能把所有机械背景一起淡出。检查叠加透明度、暗角、遮挡及相机缩放后的屏幕位置，'
        '避免代码有图形但实际出画或不可见。后拉应扩大取景并显露中近景，同时保留首帧远景，'
        '不能通过删除首帧背景来制造展开效果。')
    regression = regression_context(project, spec, plan, previous, html)
    if regression:
        context['regression'] = regression
    write_json(directory / 'repair_strategy.json', context)
    return html, plan, context
