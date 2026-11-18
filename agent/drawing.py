"""标准资产、镜头锚定帧、逐关键帧审核与局部代码修复。"""
from pathlib import Path
import json

from . import consistency, model, pixel_motion, prompts, rendering, repair, temporal, validation
from .common import digest, file_hash, log, read_json, write_json
from .planning import frame_plan_context, prepare_frame_plan, prior, req, review, validate_review_evidence

KEYFRAME_REVIEW_PROTOCOL = 'keyframe_reasoned_v3'
REFERENCE_REASONED_PROTOCOL = 'reference_reasoned_v1'


def relative(project, path):
    return str(path.relative_to(project.root))


def render_options(project):
    mode = getattr(project, 'render_consistency', 'exact')
    return {'consistency': mode} if mode != 'exact' else {}


def changed_review_settings(project, previous):
    if not previous:
        return False
    old_policy = previous.get('review_policy', consistency.review_policy('strict'))
    new_policy = consistency.review_policy(getattr(project, 'review_strictness', 'strict'))
    return (old_policy != new_policy or previous.get('render_consistency', 'exact') !=
            getattr(project, 'render_consistency', 'exact'))


def preserve_for_policy_review(project, directory, source, previous, kind='scene'):
    source = source.resolve()
    validation.require(source.is_relative_to(project.root) and source.is_file(), '策略复验源码无效')
    write_json(directory / 'consistency_recheck_provenance.json', {
        'source': relative(project, source), 'source_hash': file_hash(source),
        'source_version': previous['version'], 'kind': kind,
        'from_review_policy': previous.get('review_policy', consistency.review_policy('strict')),
        'to_review_policy': consistency.review_policy(getattr(project, 'review_strictness', 'strict')),
        'from_render_consistency': previous.get('render_consistency', 'exact'),
        'to_render_consistency': getattr(project, 'render_consistency', 'exact'),
        'provenance': 'configuration_change_recheck', 'requires_full_review': True})
    return source.read_text(encoding='utf-8')


def image_inputs(images):
    return [(f'实际画面 frame_index={index}', path) for index, path in images]


def observe_reference(project, directory, images):
    """先中性看图，再按要求验收；独立观察不能代替真实像素或硬规则。"""
    context = {'frame_indices': [0], 'observation_task':
        '仅描述实际可见的几何形状、表面明暗及其位置、物体之间的遮挡、接触关系与材质线索。'
        '看不清的细节写“未知”；不推断创作意图，不评价是否满足要求。',
        'output_contract': {'observations': '给定帧一个对象',
            'required_fields': {'frame_index': '整数，必须使用这个精确字段名', 'description': '非空文本'}}}
    for attempt in range(2):
        name = 'reference_observation' if attempt == 0 else 'reference_observation_format_retry'
        observed = project.call(directory, name, model.messages(prompts.OBSERVE, context, images))
        try:
            rows = validation.observations(observed, [0])
            write_json(directory / 'independent_observations.json', rows)
            return rows
        except ValueError as error:
            diagnosis = {'error': str(error), 'expected_frames': [0], 'response': observed}
            write_json(directory / f'{name}.schema_error.json', diagnosis)
            cache = directory / f'{name}.call.json'
            if cache.exists():
                metadata = read_json(cache)
                metadata.update(status='SCHEMA_ERROR', error=str(error))
                write_json(cache, metadata)
                if 'call_number' in metadata:
                    write_json(directory / 'calls' / name /
                               f'{metadata["call_number"]:06d}' / 'schema_error.json', diagnosis)
            if attempt:
                from .store import Blocked
                raise Blocked('参考图独立观察字段格式连续无效，已保留候选和诊断；'
                              '修复原因后可 --resume，已消耗预算不清零。') from error
            context.update(format_error=str(error), previous_observations=observed)
            log('格式', f'参考图观察：{error}；同一候选内有限修正一次')


def validate_reference_source_evidence(project, source, messages, metadata, payload, pixels):
    """Rebuild the recorded request from preserved image files and verify its original call signature."""
    from .store import Blocked
    try:
        urls = {}
        for path in source.parent.rglob('*'):
            if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                url = model.messages('', {}, [('', path)])[1]['content'][-1]['image_url']['url']
                urls[digest(url)] = url
        actual = json.loads(json.dumps(messages))
        requested = set()
        for message in actual:
            if isinstance(message.get('content'), list):
                for part in message['content']:
                    if part.get('type') == 'image_url':
                        key = part['image_url']['sha256']
                        requested.add(key)
                        part['image_url'] = {'url': urls[key]}
        validation.require(requested, '旧参考审核没有实际图像')
        validation.require(metadata.get('input_hash') == digest({'messages': actual, 'settings': metadata['model']}),
                           '旧参考审核请求或有效配置与原调用签名不匹配')
        transport = source.parent / 'review_transport/manifest.json'
        rows = read_json(transport).get('images', []) if transport.exists() else []
        encoder = {'codec': 'mjpeg', 'q': 2, 'pixel_format': 'yuvj444p', 'resize': False, 'crop': False}
        for _, path in pixels:
            url = model.messages('', {}, [('', path)])[1]['content'][-1]['image_url']['url']
            if digest(url) in requested:
                continue
            matched = False
            for row in rows:
                encoded = Path(row.get('encoded_path', '')).resolve()
                if (row.get('original_path') == str(path.resolve()) and
                    row.get('original_sha256') == file_hash(path) and row.get('encoder') == encoder and
                    encoded.is_relative_to(source.parent) and encoded.is_file() and
                    row.get('encoded_sha256') == file_hash(encoded)):
                    encoded_url = model.messages('', {}, [('', encoded)])[1]['content'][-1]['image_url']['url']
                    matched = matched or digest(encoded_url) in requested
            validation.require(matched, '旧参考PNG与正式审核请求/传输来源不匹配')
        report = read_json(source.parent / 'review.json')
        validation.require(payload['required_check_ids'] == [item['id'] for item in payload['requirements']],
                           '旧参考审核检查项列表不完整')
        validation.review(report, payload['target_id'], payload['target_version'], payload['requirements'])
        validate_review_evidence(report, [index for index, _ in pixels], payload)
        failure = source.parent / 'failure.json'
        if failure.exists():
            attempts = project.data.get('tasks', {}).get(f"assets/{payload['target_id']}", {}).get('attempts', [])
            previous = next(item for item in attempts if item['version'] == payload['target_version'])
            validation.require(previous.get('report') == read_json(failure), '旧失败报告与保存的候选记录不匹配')
    except (ValueError, KeyError, TypeError, IndexError, AttributeError, OSError, StopIteration) as error:
        raise Blocked('参考图推理复验的旧像素、请求或报告证据不完整或已改变。') from error


def reference_reasoned_binding(project, source, descriptor, options):
    """Only upgrade a verifiable ordinary reference call with genuinely different inference settings."""
    from .store import Blocked
    if (source is None or not source.is_file() or not source.resolve().is_relative_to(project.root)
            or file_hash(source) != descriptor['source_hash']):
        raise Blocked('参考图推理复验的源版本不匹配，不自动消耗后续候选。')
    if (source.parent / 'reference_reasoned_recheck_provenance.json').exists():
        raise Blocked('来源已使用当前参考推理协议，不能重复协议复验。')
    request = source.parent / 'review.request.json'
    call = source.parent / 'review.call.json'
    if not request.is_file() or not call.is_file():
        raise Blocked('参考图推理复验缺少可验证的旧正式审核请求或参数。')
    try:
        messages = read_json(request)
        payload = json.loads(messages[1]['content'][0]['text'])
        metadata = read_json(call)
        settings = metadata['model']
        protocol = payload.get('reference_review_protocol', 'reference_observation_v1')
        source_version = int(source.parent.name.removeprefix('v'))
        valid = (messages[0]['content'] == prompts.REVIEW and
                 payload.get('visual_scope') == 'reference' and
                 payload.get('evidence_mode') == 'pixels' and
                 payload.get('target_id') == source.parent.parent.name and
                 payload.get('target_version') == source_version and
                 protocol == 'reference_observation_v1' and
                 metadata.get('status') == 'PASS' and isinstance(settings, dict) and
                 isinstance(settings.get('extra_body'), dict) and
                 all(isinstance(settings.get(key), str) and settings[key]
                     for key in ('base_url', 'model')))
    except (ValueError, KeyError, TypeError, IndexError, AttributeError):
        valid = False
    if not valid:
        raise Blocked('参考图推理复验仅允许来源完整的旧普通参考审核，不能重复协议复验。')
    effective = model.effective_settings(options)
    if (not options or settings == effective or
            settings['extra_body'].get('thinking') != {'type': 'disabled'}):
        raise Blocked('旧参考审核已启用推理或端点不支持有效覆盖，不能无意义重复复验。')
    if any(settings[key] != effective[key] for key in ('base_url', 'model')):
        raise Blocked('参考图推理复验不能同时更换审核端点或模型。')
    report = source.parent / 'review.json'
    failure = source.parent / 'failure.json'
    try:
        pixels = rendering.preview_images(source.parent / 'preview')
        if not report.is_file() or not pixels or any(not path.is_file() for _, path in pixels):
            raise ValueError('missing report or pixels')
        pixel_hashes = [{'frame_index': index, 'path': relative(project, path), 'hash': file_hash(path)}
                        for index, path in pixels]
    except (ValueError, KeyError, TypeError, OSError):
        raise Blocked('参考图推理复验缺少实际旧像素或正式报告，不能验证来源。')
    validate_reference_source_evidence(project, source, messages, metadata, payload, pixels)
    return {'source': relative(project, source), 'source_hash': descriptor['source_hash'],
            'source_version': source_version, 'source_pixels': pixel_hashes,
            'source_report_hash': file_hash(report),
            'source_failure_hash': file_hash(failure) if failure.exists() else None,
            'source_request_hash': file_hash(request), 'source_call_hash': file_hash(call),
            'from_protocol': protocol, 'to_protocol': REFERENCE_REASONED_PROTOCOL,
            'previous_effective_settings': settings,
            'visual_review_extra_body': options, 'effective_settings': effective}


def reference_reasoned_task_binding(project, asset_id, descriptor, options):
    """Keep resumed task signatures bound to the original request, even after the new candidate exists."""
    attempts = project.data.get('tasks', {}).get(f'assets/{asset_id}', {}).get('attempts', [])
    previous = attempts[-1] if attempts else None
    source = project.root / previous['path'] / 'reference.html' if previous else None
    if previous:
        proof = source.parent / 'reference_reasoned_recheck_provenance.json'
        if proof.exists():
            saved = read_json(proof)
            if saved.get('source_hash') == descriptor['source_hash']:
                source = project.root / saved['source']
        elif previous['status'] in {'RUNNING', 'RETRY'}:
            earlier = attempts[-2] if len(attempts) > 1 else None
            source = project.root / earlier['path'] / 'reference.html' if earlier else None
    return reference_reasoned_binding(project, source, descriptor, options)


def reference_reasoned_recheck_source(project, directory, previous, descriptor, options, binding):
    """One source-preserving explicit inference upgrade; never reroll the same source/protocol."""
    from .store import Blocked
    source = project.root / previous['path'] / 'reference.html' if previous else None
    actual = reference_reasoned_binding(project, source, descriptor, options)
    if actual != binding:
        raise Blocked('参考图推理复验来源或有效参数改变，不能覆盖来源绑定。')
    provenance = directory / 'reference_reasoned_recheck_provenance.json'
    for path in source.parent.parent.glob('v*/reference_reasoned_recheck_provenance.json'):
        saved = read_json(path)
        if path.resolve() == provenance.resolve():
            if (any(saved.get(key) != value for key, value in binding.items()) or
                    saved.get('reason') != descriptor['reason']):
                raise Blocked('当前候选参考图推理复验绑定已改变，不能覆盖来源记录。')
        elif (saved.get('source_hash') == descriptor['source_hash'] and
              saved.get('to_protocol') == REFERENCE_REASONED_PROTOCOL):
            raise Blocked('相同源码已升级当前参考推理协议，不能重复协议复验。')
    if not provenance.exists():
        write_json(provenance, {**binding, 'source_version': previous['version'],
            'reason': descriptor['reason'], 'options_source': 'model.visual_review_options',
            'provenance': 'review_protocol_upgrade', 'requires_full_review': True})
    return source.read_bytes()


def assets(project, facts, bible):
    results = {}
    for asset in bible['assets']:
        asset_id = asset['asset_id']
        feedback = project.data['repair_requests'].get(asset_id, {})
        descriptor = feedback.get('local_patch')
        recheck = feedback.get('reference_recheck')
        reasoned = feedback.get('reference_reasoned_recheck')
        validation.require(sum(item is not None for item in (descriptor, recheck, reasoned)) <= 1,
                           '参考图代码补丁、观察协议复验与推理复验不能同时指定')
        options, binding = None, None
        if reasoned is not None:
            validation.require(isinstance(reasoned, dict), '参考图推理复验描述必须是对象')
            validation.text(reasoned.get('source_hash'), 'reference_reasoned_recheck.source_hash')
            validation.text(reasoned.get('reason'), 'reference_reasoned_recheck.reason')
            options = model.visual_review_options()
            binding = reference_reasoned_task_binding(project, asset_id, reasoned, options)
        if recheck:
            validation.require(isinstance(recheck, dict), '参考图协议复验描述必须是对象')
            validation.text(recheck.get('source_hash'), 'reference_recheck.source_hash')
            validation.text(recheck.get('reason'), 'reference_recheck.reason')
        regions = (descriptor or recheck or reasoned or {}).get('review_regions', [])
        validation.require(isinstance(regions, list) and len(regions) <= 2,
                           '资产审核局部区域必须是列表且最多两个')
        for region in regions:
            validation.require(isinstance(region, list) and len(region) == 4 and
                               all(type(v) in (int, float) and 0 <= v <= 1 for v in region) and
                               region[0] < region[2] and region[1] < region[3],
                               '资产审核局部区域坐标无效')
        def action(directory, version, previous, asset=asset, asset_id=asset_id,
                   descriptor=descriptor, regions=regions, recheck=recheck, reasoned=reasoned,
                   options=options, binding=binding):
            html_bytes = None
            if reasoned is not None:
                html_bytes = reference_reasoned_recheck_source(
                    project, directory, previous, reasoned, options, binding)
                log('复核', f'{asset_id}：保留参考图源码，按显式推理协议完整复验一次')
            elif recheck:
                source = project.root / previous['path'] / 'reference.html' if previous else None
                from .store import Blocked
                if source is None or not source.is_file() or file_hash(source) != recheck['source_hash']:
                    raise Blocked('参考图协议复验的源版本不匹配，不自动消耗后续候选。')
                if (source.parent / 'reference_observation.json').exists():
                    raise Blocked('该参考图已经独立观察审核，不能重复协议复验；请修复实际失败原因。')
                html_text = source.read_text(encoding='utf-8')
                write_json(directory / 'reference_recheck.json', {
                    'source': relative(project, source), 'source_hash': recheck['source_hash'],
                    'reason': recheck['reason'], 'provenance': 'review_protocol_upgrade',
                    'requires_full_review': True})
                log('复核', f'{asset_id}：保留实际画面，按独立观察协议完整复验一次')
            elif descriptor:
                source = project.root / previous['path'] / 'reference.html' if previous else None
                html_text = local_patch(project, directory, source, descriptor)
                log('修复', f'{asset_id}：应用绑定源版本的本地参考图补丁，随后完整审核')
            elif previous and previous.get('render_consistency', 'exact') != getattr(project, 'render_consistency', 'exact'):
                source = project.root / previous['path'] / 'reference.html'
                html_text = preserve_for_policy_review(project, directory, source, previous, 'reference')
                log('复核', f'{asset_id}：保留参考图源码，按新的渲染一致性设置复验')
            else:
                value = project.call(directory, 'generated', model.messages(prompts.CANVAS, {
                    'kind': 'reference', 'style': bible['style'], 'asset': asset,
                    'duration': 1, **prior(project, previous)}))
                validation.text(value.get('html'), 'html')
                html_text = value['html']
            html = directory / 'reference.html'
            if html_bytes is not None:
                html.write_bytes(html_bytes)
            else:
                html.write_text(html_text, encoding='utf-8')
            preview = directory / 'preview'
            if not (preview / 'preview_manifest.json').exists():
                rendering.render_job(html, preview, 2, project.config['fps'], preview=True,
                                     frame_count=1, sample_indices=[0], **render_options(project))
            images = rendering.preview_images(preview)
            if reasoned is not None:
                expected = [(row['frame_index'], row['hash']) for row in binding['source_pixels']]
                if [(index, file_hash(path)) for index, path in images] != expected:
                    from .store import Blocked
                    raise Blocked('参考图推理复验重画像素与绑定旧图不同，不能把变化后的画面当作同源复验。')
            review_images = image_inputs(images)
            for index, region in enumerate(regions):
                crop = directory / 'review_evidence' / f'crop_{index:02d}.png'
                rendering.crop_evidence(images[0][1], crop, region)
                review_images.append((f'原图局部放大：实际画面 frame_index=0，region_xyxy={region}；'
                                      '坐标对应原始全图，放大图不是新帧，不能替代全图审核', crop))
            observations = observe_reference(project, directory, review_images)
            payload = {'asset': asset, 'style': bible['style']}
            if reasoned is not None:
                # Keep observations on disk without anchoring formal acceptance to their descriptions.
                payload.update(reference_review_protocol=REFERENCE_REASONED_PROTOCOL,
                               reference_reasoned_binding=binding)
            else:
                payload.update(independent_observations=observations,
                    observation_context='独立观察仅为看图辅助，不能代替实际像素证据、修改或覆盖任何审核要求。'
                        '审核仍须逐项对照全图与局部；观察与画面冲突时以真实画面为准。')
            try:
                review(project, directory, 'review', asset_id, version, asset['requirements'],
                       payload, review_images, [0], visual_scope='reference',
                       **({'extra_body': options} if reasoned is not None else {}))
            finally:
                if reasoned is not None:
                    # Retain hashes even when the formal result rejects the unchanged pixels.
                    proof_path = directory / 'reference_reasoned_recheck_provenance.json'
                    proof = read_json(proof_path)
                    proof['current_review_requests'] = [
                        {'path': relative(project, path), 'hash': file_hash(path),
                         'call': read_json(path.with_name(path.name.replace('.request.json', '.call.json')))}
                        for path in sorted(directory.glob('review*.request.json'))
                        if path.with_name(path.name.replace('.request.json', '.call.json')).exists()]
                    write_json(proof_path, proof)
            return {'asset_id': asset_id, 'version': version, 'image': relative(project, images[0][1]),
                    'html': relative(project, html), 'fixed': asset['fixed'], 'variable': asset['variable']}
        inputs = {'asset': asset, 'review_protocol': 'reference_observation_v1'}
        if descriptor:
            inputs['local_patch'] = descriptor
        if recheck:
            inputs['reference_recheck'] = recheck
        if reasoned is not None:
            inputs.update(review_protocol=REFERENCE_REASONED_PROTOCOL,
                          reference_reasoned_recheck=reasoned, reference_reasoned_binding=binding)
        from .store import Blocked, Rejected
        try:
            results[asset_id] = project.task(f'assets/{asset_id}', inputs,
                                            ['plans/story', 'plans/visual'], action,
                                            **({'retry': False} if reasoned is not None else {}))
        except Rejected as error:
            if reasoned is not None:
                raise Blocked('参考图推理复验未通过；已保留同源候选及正式报告，请修复实际失败原因。') from error
            raise
    write_json(project.root / 'assets/approved.json', results)
    return results


def patched_html(original, changes):
    patches = validation.items(changes.get('patches'), 'patches')
    for patch in patches:
        old, new = patch['old'], patch['new']
        validation.text(old, 'patch.old')
        validation.require(isinstance(new, str) and original.count(old) == 1,
                           '代码修复片段必须在原代码中唯一匹配')
        validation.require(old != original, '局部修复不能替换整份 HTML')
        original = original.replace(old, new, 1)
    return original


def local_patch(project, directory, source, descriptor):
    """绑定确切源文件的人工代码补丁，仍须经过全部画面审核。"""
    path = (project.root / descriptor['path']).resolve()
    validation.require(path.is_relative_to(project.root), '局部补丁必须保存在项目内')
    from .store import Blocked
    if file_hash(path) != descriptor['hash']:
        raise Blocked('局部补丁内容已改变；请更新绑定后恢复，不自动消耗后续候选。')
    if source is None or not source.is_file() or file_hash(source) != descriptor['source_hash']:
        raise Blocked('局部补丁的源版本不匹配；请基于当前代码更新补丁后恢复，不自动消耗后续候选。')
    changes = read_json(path)
    original = source.read_text(encoding='utf-8')
    result = patched_html(original, changes)
    validation.require(result != original, '局部补丁没有改变代码')
    write_json(directory / 'applied_local_patch.json', {
        'source': relative(project, source), 'source_hash': descriptor['source_hash'],
        'patch': descriptor, 'changes': changes,
        'provenance': 'local_code_edit', 'requires_full_review': True})
    return result


def local_source(project, directory, previous, descriptor):
    """首个镜头候选的显式本地源码；后续仍执行原有全部画面与技术验收。"""
    from .store import Blocked
    if previous is not None:
        raise Blocked('本地镜头源码仅允许首个候选；后续请基于已保存候选修复。')
    source_path = Path(descriptor['path'])
    source = (project.root / source_path).resolve()
    if source_path.is_absolute() or not source.is_relative_to(project.root) or not source.is_file():
        raise Blocked('本地镜头源码必须是项目内实际文件的相对路径。')
    if file_hash(source) != descriptor['hash']:
        raise Blocked('本地镜头源码哈希不匹配；请更新绑定后恢复，不自动消耗后续候选。')
    html_text = source.read_text(encoding='utf-8')
    validation.text(html_text, 'local_source.html')
    write_json(directory / 'local_source_provenance.json', {
        'path': relative(project, source), 'hash': descriptor['hash'],
        'provenance': 'local_code_authoring', 'requires_full_review': True})
    return html_text


def observe_keyframes(project, directory, name, batch):
    """隔离资产与预期状态，只观察当前和邻接实际帧；不作通过判断。"""
    indices = [frame for frame, _ in batch]
    context = {'frame_indices': indices, 'observation_task':
        '逐帧描述实际可见物体及近似位置，以及物体间的接触、空隙、连接物、遮挡与表面明暗。'
        '只记录能够看见的关系，不预设接触或空隙一定存在；看不清写“未知”。'
        '不推断创作意图，不评价是否满足要求，不作通过判断。',
        'output_contract': {'observations': '每个给定帧一个对象，保持原顺序',
            'required_fields': {'frame_index': '整数，必须使用这个精确字段名', 'description': '非空文本'}}}
    images = rendering.compact_review_images(directory / (name + '_transport'), image_inputs(batch))
    for attempt in range(2):
        call_name = name if attempt == 0 else name + '_format_retry'
        observed = project.call(directory, call_name, model.messages(prompts.OBSERVE, context, images))
        try:
            return validation.observations(observed, indices)
        except ValueError as error:
            diagnosis = {'error': str(error), 'expected_frames': indices, 'response': observed}
            write_json(directory / f'{call_name}.schema_error.json', diagnosis)
            cache = directory / f'{call_name}.call.json'
            if cache.exists():
                metadata = read_json(cache)
                metadata.update(status='SCHEMA_ERROR', error=str(error))
                write_json(cache, metadata)
                if 'call_number' in metadata:
                    write_json(directory / 'calls' / call_name /
                               f'{metadata["call_number"]:06d}' / 'schema_error.json', diagnosis)
            if attempt:
                from .store import Blocked
                raise Blocked(f'{name} 独立观察字段连续无效，已保留候选与诊断；'
                              '修复后可恢复，预算不清零。') from error
            context.update(format_error=str(error), previous_observations=observed)
            log('格式', f'{name}：{error}；同一候选内有限修正一次')


def keyframe_recheck_source(project, directory, previous, descriptor, review_options=None):
    """绑定旧协议的确切源码，只允许升级到当前关键帧推理审核协议一次。"""
    from .store import Blocked
    source = project.root / previous['path'] / 'scene.html' if previous else None
    if source is None or not source.is_file() or file_hash(source) != descriptor['source_hash']:
        raise Blocked('关键帧协议复验的源版本不匹配，不自动消耗后续候选。')
    tasks_path = source.parent / 'frame_tasks.json'
    tasks = read_json(tasks_path) if tasks_path.exists() else {}
    source_protocol = tasks.get('review_protocol')
    if source_protocol not in {None, 'keyframe_observation_v1', 'keyframe_pixels_v2'}:
        raise Blocked('关键帧协议复验仅允许旧观察或像素审核协议，不能重复协议复验；请修复实际失败原因。')
    if source_protocol is None:
        source_protocol = ('keyframe_observation_v1' if any(source.parent.glob('keyframe_observation_*.json'))
                           else 'legacy_unversioned')
    options = model.visual_review_options() if review_options is None else review_options
    effective = model.effective_settings(options)
    provenance = directory / 'keyframe_recheck_provenance.json'
    binding = {'source': relative(project, source), 'source_hash': descriptor['source_hash'],
               'from_protocol': source_protocol, 'to_protocol': KEYFRAME_REVIEW_PROTOCOL,
               'effective_options': effective}
    for path in source.parent.parent.glob('v*/keyframe_recheck_provenance.json'):
        saved = read_json(path)
        if path.resolve() == provenance.resolve():
            if any(saved.get(key) != value for key, value in binding.items()):
                raise Blocked('当前候选关键帧协议复验绑定已改变，不能覆盖来源记录。')
        elif (saved.get('source_hash') == descriptor['source_hash'] and
              saved.get('to_protocol') == KEYFRAME_REVIEW_PROTOCOL):
            raise Blocked('相同源码已升级当前关键帧协议，不能重复协议复验；请修复实际失败原因。')
    previous_calls = [{'path': relative(project, path), 'hash': file_hash(path),
                       'settings': read_json(path).get('model')}
                      for path in sorted(source.parent.glob('check_*.call.json'))]
    if not provenance.exists():
        write_json(provenance, {
            **binding, 'source_version': previous['version'],
            'visual_review_extra_body': options, 'options_source': 'model.visual_review_options',
            'previous_review_calls': previous_calls,
            'reason': descriptor.get('reason', '升级关键帧逐项推理审核协议'),
            'provenance': 'review_protocol_upgrade', 'requires_full_review': True})
    return source.read_text(encoding='utf-8')


def temporal_recheck_source(project, directory, previous, descriptor):
    """只将确切的旧采样协议候选升级一次；历史源码及失败报告保持不变。"""
    from .store import Blocked
    source = project.root / previous['path'] / 'scene.html' if previous else None
    if source is None or not source.is_file() or file_hash(source) != descriptor['source_hash']:
        raise Blocked('时序协议复验的源版本不匹配，不自动消耗后续候选。')
    tasks_path = source.parent / 'frame_tasks.json'
    tasks = read_json(tasks_path) if tasks_path.exists() else {}
    source_protocol = tasks.get('temporal_evidence_protocol')
    if source_protocol not in {'sample_timing_v1', 'windowed_v2', 'windowed_local_v3', 'windowed_comparison_v4'}:
        raise Blocked('时序协议复验仅允许 sample_timing_v1、windowed_v2、windowed_local_v3 或 windowed_comparison_v4 旧候选，'
                      '不能重复协议复验；请修复实际失败原因。')
    write_json(directory / 'temporal_recheck_provenance.json', {
        'source': relative(project, source), 'source_hash': descriptor['source_hash'],
        'from_protocol': source_protocol, 'to_protocol': temporal.PROTOCOL,
        'reason': descriptor.get('reason', '完整实际采样分窗审核'),
        'provenance': 'review_protocol_upgrade', 'requires_full_review': True})
    return source.read_text(encoding='utf-8')



def temporal_detail_recheck_source(project, directory, previous, descriptor):
    """只用新真实区域复验上一失败窗口，原源码、报告与通过证据均保留。"""
    from .store import Blocked
    regions = temporal.detail_regions(descriptor)
    motion_reference = ({'relative_motion_reference': descriptor['relative_motion_reference']}
                        if 'relative_motion_reference' in descriptor else {})
    motion_spec = ({'pixel_motion': pixel_motion.validate_spec(descriptor['pixel_motion'])}
                   if 'pixel_motion' in descriptor else {})
    source = project.root / previous['path'] / 'scene.html' if previous else None
    if source is None or not source.is_file() or file_hash(source) != descriptor['source_hash']:
        raise Blocked('时序细节复验的源版本不匹配，不自动消耗后续候选。')
    target = descriptor['target_review']
    aggregate_path = source.parent / 'temporal_aggregate.json'
    report_path = source.parent / (target + '.json')
    aggregate = read_json(aggregate_path) if aggregate_path.is_file() else {}
    failed = aggregate.get('failed_review', {})
    if (failed.get('name') != target or failed.get('status') not in {'FAIL', 'UNCERTAIN'} or
            aggregate.get('source', {}).get('hash') != descriptor['source_hash'] or
            not report_path.is_file() or read_json(report_path).get('status') not in {'FAIL', 'UNCERTAIN'}):
        raise Blocked('时序细节复验必须对应上一候选的实际失败窗口。')
    actual = {row['frame_index']: row for row in failed.get('images', [])}
    bound = []
    for region in regions:
        if region['frame_index'] not in actual:
            raise Blocked('细节复验帧不属于上一失败窗口的实际采样。')
        row = actual[region['frame_index']]
        path = (project.root / row['path']).resolve()
        if not path.is_relative_to(project.root) or not path.is_file() or file_hash(path) != row['hash']:
            raise Blocked('细节复验原始帧证据已改变，不能使用。')
        bound.append(dict(region, source_frame_hash=row['hash']))
    motion_frames = []
    for frame in motion_spec.get('pixel_motion', {}).get('frame_indices', []):
        if frame not in actual:
            raise Blocked('像素测量帧必须属于当前失败窗口的实际采样。')
        row = actual[frame]
        path = (project.root / row['path']).resolve()
        if not path.is_relative_to(project.root) or not path.is_file() or file_hash(path) != row['hash']:
            raise Blocked('像素测量原始帧绑定已改变。')
        motion_frames.append({'frame_index': frame, 'path': row['path'], 'hash': row['hash']})
    motion_binding = dict(motion_spec, pixel_motion_frames=motion_frames) if motion_spec else {}
    request_hash = digest({'source_hash': descriptor['source_hash'], 'target_review': target,
                           'regions': regions, **motion_reference, **motion_spec})
    provenance = directory / 'temporal_detail_recheck_provenance.json'
    seen, seen_motion = set(), set()
    for saved_path in source.parent.parent.glob('v*/temporal_detail_recheck_provenance.json'):
        saved = read_json(saved_path)
        if saved_path.resolve() == provenance.resolve():
            if saved.get('request_hash') != request_hash:
                raise Blocked('当前候选细节复验绑定已改变，不能覆盖来源记录。')
            continue  # 相同RUNNING候选恢复不额外申请一次复验。
        if saved.get('source_hash') == descriptor['source_hash'] and saved.get('target_review') == target:
            seen.update(digest(row) for row in temporal.detail_regions(saved))
            if 'pixel_motion' in saved:
                seen_motion.add(digest(pixel_motion.validate_spec(saved['pixel_motion'])))
            if saved.get('request_hash') == request_hash:
                raise Blocked('相同源码、窗口和区域不能重复细节复验。')
    if motion_spec and digest(motion_spec['pixel_motion']) in seen_motion:
        raise Blocked('相同源码、窗口和规范测量方案不能重复像素测量复验。')
    if not {digest(region) for region in regions} - seen and not motion_spec:
        raise Blocked('细节复验至少需要一个此前未提供的新有效区域或新的像素测量方案。')
    if not provenance.exists():
        write_json(provenance, {
            'source': relative(project, source), 'source_hash': descriptor['source_hash'],
            'target_review': target, 'reason': descriptor['reason'], 'regions': bound,
            **motion_reference, **motion_binding,
            'request_hash': request_hash, 'failed_report': {'path': relative(project, report_path),
                                                        'hash': file_hash(report_path)},
            'provenance': 'additional_actual_pixel_evidence', 'requires_full_review': True})
    return source.read_text(encoding='utf-8'), {
        'target_review': target, 'source_hash': descriptor['source_hash'], 'regions': bound,
        **motion_reference, **motion_binding}



def keyframe_motion_frames(descriptor):
    validation.require(isinstance(descriptor, dict), 'keyframe_motion_recheck 必须是对象')
    for field in ('source_hash', 'target_review', 'reason'):
        validation.text(descriptor.get(field), 'keyframe_motion_recheck.' + field)
    indices = descriptor.get('frame_indices')
    validation.require(isinstance(indices, list) and 1 <= len(indices) <= 6 and
                       all(type(index) is int for index in indices) and len(set(indices)) == len(indices),
                       '关键帧运动复验必须提供1至6个不重复的整数帧号')
    return sorted(indices)


def keyframe_motion_recheck_source(project, directory, previous, descriptor, shot_id, frames):
    """绑定上一失败关键帧，只补新实际中间帧，保留原源码和全部审核门槛。"""
    from .store import Blocked
    indices = keyframe_motion_frames(descriptor)
    source = project.root / previous['path'] / 'scene.html' if previous else None
    if source is None or not source.is_file() or file_hash(source) != descriptor['source_hash']:
        raise Blocked('关键帧运动复验的源版本不匹配。')
    target = descriptor['target_review']
    keys = frames['keyframes']
    positions = [i for i, key in enumerate(keys) if 'check_' + key['frame_id'] == target]
    if not positions or positions[0] == 0:
        raise Blocked('关键帧运动复验必须指定有前一关键帧的实际目标。')
    position = positions[0]
    if not all(keys[position - 1]['frame_index'] < index < keys[position]['frame_index'] for index in indices):
        raise Blocked('补充运动帧必须严格位于前一关键帧和当前目标之间。')
    failure_path = source.parent / 'failure.json'
    failure = read_json(failure_path) if failure_path.is_file() else {}
    if (failure.get('target_id') != shot_id + '_' + keys[position]['frame_id'] or
            failure.get('status') not in {'FAIL', 'UNCERTAIN'} or not (source.parent / (target + '.json')).is_file()):
        raise Blocked('运动复验必须对应上一候选实际失败的关键帧。')
    request_hash = digest({'source_hash': descriptor['source_hash'], 'target_review': target, 'frame_indices': indices})
    provenance = directory / 'keyframe_motion_recheck_provenance.json'
    seen = set()
    for path in source.parent.parent.glob('v*/keyframe_motion_recheck_provenance.json'):
        saved = read_json(path)
        if path.resolve() == provenance.resolve():
            if saved.get('request_hash') != request_hash:
                raise Blocked('当前候选运动复验绑定已改变，不能覆盖来源记录。')
            continue
        if saved.get('source_hash') == descriptor['source_hash'] and saved.get('target_review') == target:
            seen.update(saved['frame_indices'])
    if not seen <= set(indices):
        raise Blocked('运动复验必须保留同源码和目标已提供的全部中间帧。')
    if not set(indices) - seen:
        raise Blocked('相同源码和目标不能重复运动复验；必须提供新的实际中间帧。')
    if not provenance.exists():
        write_json(provenance, {
            'source': relative(project, source), 'source_hash': descriptor['source_hash'],
            'target_review': target, 'frame_indices': indices, 'reason': descriptor['reason'],
            'request_hash': request_hash, 'failed_report': {'path': relative(project, failure_path),
                                                         'hash': file_hash(failure_path)},
            'provenance': 'additional_actual_motion_frames', 'requires_full_review': True})
    return source.read_text(encoding='utf-8'), {
        'source_hash': descriptor['source_hash'], 'target_review': target, 'frame_indices': indices}


def keyframe_motion_context(images, fps, indices, options):
    return {'visual_review_extra_body': options, 'keyframe_motion_evidence': {
        'fps': fps, 'additional_frame_indices': indices,
        'sample_timing': temporal.timing(images, fps),
        'instruction': '当前关键帧静态状态仍只依据当前实际帧。额外实际中间帧用于检查该区间的动作变化与连续性，'
            '逐一比较真实位置与姿态；所有原始要求仍须满足，不能根据计划文字补全未给出的时间点。'
            '非连续采样间的瞬时跳变仍需邻接真实帧证据；无法确认则UNCERTAIN，不能自动判定通过。'}}


def carried_keyframe_motion(project, directory, html, frames, review_options):
    """只保留同源码且已正式通过的实际运动证据；是否复用仍由完整审核签名决定。"""
    result, source_hash = {}, file_hash(html)
    keys = frames['keyframes']
    for path in sorted(directory.parent.glob('v*/keyframe_motion_recheck_provenance.json'), reverse=True):
        if path.parent.name >= directory.name:
            continue
        try:
            proof = read_json(path)
            target = proof['target_review']
            if target in result or proof['source_hash'] != source_hash:
                continue
            indices = keyframe_motion_frames(proof)
            position = next(i for i, key in enumerate(keys) if 'check_' + key['frame_id'] == target)
            if not position or not all(keys[position - 1]['frame_index'] < i < keys[position]['frame_index']
                                       for i in indices):
                continue
            sources = [path.parent / 'scene.html', (project.root / proof['source']).resolve()]
            if any(not p.is_relative_to(project.root) or not p.is_file() or file_hash(p) != source_hash
                   for p in sources):
                continue
            if proof['request_hash'] != digest({'source_hash': source_hash, 'target_review': target,
                                                'frame_indices': indices}):
                continue
            actual = proof['actual_frames']
            expected = [keys[position - 1]['frame_index']] + indices + [keys[position]['frame_index']]
            if [row['frame_index'] for row in actual] != expected:
                continue
            images = [(row['frame_index'], (project.root / row['path']).resolve()) for row in actual]
            if any(not image.is_relative_to(path.parent) or not image.is_file() or file_hash(image) != row['hash']
                   for row, (_, image) in zip(actual, images)):
                continue
            report_path, request_path, call_path = [path.parent / (target + suffix)
                                                   for suffix in ('.json', '.request.json', '.call.json')]
            report, request, call = read_json(report_path), read_json(request_path), read_json(call_path)
            context = json.loads(request[1]['content'][0]['text'])
            motion_context = keyframe_motion_context(images, project.config['fps'], indices, review_options)
            if (report.get('status') != 'PASS' or context['task'] != keys[position] or
                    any(context.get(key) != value for key, value in motion_context.items()) or
                    proof['effective_options'] != model.effective_settings(review_options) or
                    call.get('model') != proof['effective_options'] or call.get('status') != 'PASS'):
                continue
            if not validation.review(report, context['target_id'], context['target_version'], context['requirements']):
                continue
            validate_review_evidence(report, expected, context)
            bindings = {name: {'path': relative(project, p), 'hash': file_hash(p)} for name, p in
                        [('original_provenance', path), ('passed_report', report_path),
                         ('request', request_path), ('call', call_path)]}
            result[target] = {'frame_indices': indices, 'context': motion_context, 'images': images[1:-1],
                'provenance': dict(bindings, source_hash=source_hash, target_review=target,
                                   actual_frames=actual, effective_options=proof['effective_options'])}
        except (OSError, ValueError, KeyError, TypeError, StopIteration):
            continue  # 不完整、已变更或未通过的旧证据不能取得继承资格。
    return result


def shot(project, spec, frames, references, previous_shot=None, next_shot=None):
    shot_id = spec['shot_id']
    count, fps = spec['end_frame'] - spec['start_frame'], project.config['fps']
    refs = [(f'标准资产 {key} v{references[key]["version"]}', project.root / references[key]['image'])
            for key in spec['asset_ids']]
    # 镜头共享已通过标准资产；镜头边界单独复验，避免一个局部修复级联重画整片。
    deps = ['plans/storyboard', f'plans/frames/{shot_id}'] + [f'assets/{key}' for key in spec['asset_ids']]
    feedback = project.data['repair_requests'].get(shot_id)
    review_options = model.visual_review_options()
    authored = (feedback or {}).get('local_source')
    recheck = (feedback or {}).get('keyframe_recheck')
    temporal_recheck = (feedback or {}).get('temporal_recheck')
    detail_recheck = (feedback or {}).get('temporal_detail_recheck')
    motion_recheck = (feedback or {}).get('keyframe_motion_recheck')
    if motion_recheck is not None:
        keyframe_motion_frames(motion_recheck)
        validation.require(all((feedback or {}).get(key) is None for key in
                           ('local_patch', 'local_source', 'keyframe_recheck', 'temporal_recheck', 'temporal_detail_recheck')),
                           '关键帧运动复验不能与本地修改或其他复验同时指定')
    if detail_recheck is not None:
        temporal.detail_regions(detail_recheck)
        validation.text(detail_recheck.get('source_hash'), 'temporal_detail_recheck.source_hash')
        validation.text(detail_recheck.get('reason'), 'temporal_detail_recheck.reason')
        validation.require(all((feedback or {}).get(key) is None for key in
                           ('local_patch', 'local_source', 'keyframe_recheck', 'temporal_recheck')),
                           '时序细节复验不能与本地修改或其他协议复验同时指定')
    if temporal_recheck is not None:
        validation.require(isinstance(temporal_recheck, dict), 'temporal_recheck 必须是对象')
        validation.text(temporal_recheck.get('source_hash'), 'temporal_recheck.source_hash')
        validation.require(authored is None and recheck is None and (feedback or {}).get('local_patch') is None,
                           '时序协议复验不能与关键帧复验、本地源码或局部补丁同时指定')
    if recheck is not None:
        validation.require(isinstance(recheck, dict), 'keyframe_recheck 必须是对象')
        validation.text(recheck.get('source_hash'), 'keyframe_recheck.source_hash')
        validation.require(authored is None and not feedback.get('local_patch'),
                           '关键帧协议复验不能与本地源码或局部补丁同时指定')
    if authored is not None:
        validation.require(isinstance(authored, dict), 'local_source 必须是对象')
        validation.text(authored.get('path'), 'local_source.path')
        validation.text(authored.get('hash'), 'local_source.hash')
        validation.require(not feedback.get('local_patch'), '本地镜头源码与局部补丁不能同时指定')

    def action(directory, version, previous):
        authored_html = local_source(project, directory, previous, authored) if authored is not None else None
        rechecked_html = (keyframe_recheck_source(project, directory, previous, recheck, review_options)
                          if recheck is not None else None)
        temporal_html = temporal_recheck_source(project, directory, previous, temporal_recheck) if temporal_recheck is not None else None
        detail_html, temporal_details = (temporal_detail_recheck_source(
            project, directory, previous, detail_recheck) if detail_recheck is not None else (None, None))
        motion_html, motion_details = (keyframe_motion_recheck_source(
            project, directory, previous, motion_recheck, shot_id, frames) if motion_recheck is not None else (None, None))
        old_html, local_plan, strategy = repair.patch_context(project, directory, spec, frames, previous)
        if (previous and previous.get('status') == 'PASS' and old_html is not None and old_html.is_file()
                and all((feedback or {}).get(key) is None for key in (
                    'local_source', 'local_patch', 'keyframe_recheck', 'temporal_recheck',
                    'temporal_detail_recheck', 'keyframe_motion_recheck'))):
            saved_tasks = old_html.parent / 'frame_tasks.json'
            old_protocol = read_json(saved_tasks).get('review_protocol') if saved_tasks.is_file() else None
            if old_protocol in {None, 'keyframe_observation_v1', 'keyframe_pixels_v2'}:
                rechecked_html = keyframe_recheck_source(project, directory, previous, {
                    'source_hash': file_hash(old_html),
                    'reason': '已通过镜头保留原源码，升级当前关键帧推理审核协议'}, review_options)
        policy_recheck = changed_review_settings(project, previous)
        payload = {'kind': 'shot', 'consistency_policy': consistency.review_policy(
                       getattr(project, 'review_strictness', 'strict')),
                   'shot': spec, 'frame_plan': local_plan, 'duration': count / fps,
                   'fps': fps, 'frame_count': count, 'last_time': (count - 1) / fps,
                   'assets': references, 'global_feedback': feedback, **prior(project, previous),
                   'repair_strategy': strategy,
                   'repair_history': repair.history(project, f'segments/{shot_id}', previous)}
        repair_images = repair.evidence(project, directory,
            [feedback or {}, previous.get('report', {}) if previous else {}])
        for item in strategy.get('regression', {}).get('passed_images', []):
            path = project.root / item['path']
            validation.require(file_hash(path) == item['hash'], '历史通过帧已改变，不能用于修复')
            repair_images.append((item['label'], path))
        # 仅明确的计划错误才重规划；候选版本号不能决定将局部动作升级为整镜头重画。
        replan = bool(not policy_recheck and detail_recheck is None and motion_recheck is None and previous and (
            previous.get('report', {}).get('repair', {}).get('scope') == 'plan' or
            (feedback and feedback.get('repair', {}).get('scope') == 'plan' and
             feedback.get('replan_source_version') == previous['version'])))
        if replan:
            plan_images = refs + repair_images
            if previous_shot:
                plan_images += [('上一镜头实际末帧：新计划的进入状态必须连续承接',
                                 project.root / previous_shot['last_frame'])]
            local_plan = project.call(directory, 'replanned', model.messages(prompts.FRAME_PLAN, {
                **frame_plan_context(spec, fps), 'previous_frame_plan': local_plan,
                'feedback': payload.get('feedback'), 'global_feedback': feedback}, plan_images))
            local_plan = prepare_frame_plan(directory, local_plan)
            validation.frame_plan(local_plan, spec)
            review(project, directory, 'replan_review', shot_id, version, spec['requirements'],
                   {'shot': spec, 'candidate_frame_plan': local_plan}, visual_scope='plan')
            payload['frame_plan'] = local_plan
        if previous_shot:
            previous_image = project.root / previous_shot['last_frame']
            repair_images.append(('上一镜头已通过的最后画面；本镜头首帧必须承接其位置、比例和背景', previous_image))
            payload['continuity'] = {'shot_id': previous_shot['shot_id'], 'version': previous_shot['version'],
                                     'image': previous_shot['last_frame'], 'image_hash': file_hash(previous_image)}
            payload['continuity'].update(
                previous_last_time=(previous_shot['end_frame'] - previous_shot['start_frame'] - 1) / fps,
                previous_next_time=(previous_shot['end_frame'] - previous_shot['start_frame']) / fps,
                instruction='本镜头的局部t=0不代表动作从零开始。持续动作应承接前镜头结束时的'
                    '身体形变、部件姿态、摆动相位、头部伸出、相机变换与背景状态。'
                    '先按上一镜头的时间和代码计算进入状态，再连续插值到本镜头的动作参数；'
                    '不能把两幅完整身体交叉淡化，也不能仅因机位描述不同就重置动作或让背景瞬间消失。')
            if feedback and (feedback.get('kind') == 'temporal_boundary' or
                             feedback.get('target_id') == f'{previous_shot["shot_id"]}_{shot_id}'):
                previous_source = project.root / previous_shot['html']
                payload['continuity']['previous_html'] = previous_source.read_text(encoding='utf-8')
                payload['continuity']['previous_html_hash'] = file_hash(previous_source)
            if old_html is None:
                old_html = project.root / previous_shot['html']
                payload['repair_strategy'] = {
                    'strategy': 'continue_previous_scene',
                    'instruction': '使用上一镜头实际代码作为场景和造型基础，只替换当前镜头的动作时间线。'
                        '本镜头 t=0 需要承接前镜头末帧；不要再次播放前镜头动作。保持背景、主体比例与资产身份。'}
            write_json(directory / 'continuity_reference.json', payload['continuity'])
        write_json(directory / 'frame_tasks.json', {
            'shot_id': shot_id, 'version': version, 'frame_plan': local_plan,
            'review_protocol': KEYFRAME_REVIEW_PROTOCOL,
            'temporal_evidence_protocol': temporal.PROTOCOL,
            'visual_review_extra_body': review_options,
            'consistency_policy': consistency.review_policy(getattr(project, 'review_strictness', 'strict')),
            'render_consistency': getattr(project, 'render_consistency', 'exact'),
            'references': [references[key] for key in spec['asset_ids']],
            'input_versions': project.data['tasks'][f'segments/{shot_id}']['input_versions']})
        motion_confirmation = repair.needs_motion_confirmation(project, spec, local_plan, previous)
        entry_confirmation = bool(previous_shot and repair.needs_entry_confirmation(
            project, spec, local_plan, previous))
        if authored_html is not None:
            html_text = authored_html
            log('绘制', f'{shot_id}：使用已绑定的本地镜头源码，随后完整审核')
        elif rechecked_html is not None:
            html_text = rechecked_html
            log('复核', f'{shot_id}：保留实际代码，按关键帧推理审核协议完整复验一次')
        elif temporal_html is not None:
            html_text = temporal_html
            log('复核', f'{shot_id}：保留实际代码，按完整采样分窗协议复验一次')
        elif detail_html is not None:
            html_text = detail_html
            log('复核', f'{shot_id}：保留原代码，仅为指定失败窗口补充真实区域细节')
        elif motion_html is not None:
            html_text = motion_html
            log('复核', f'{shot_id}：保留原代码，为指定失败关键帧补充实际中间运动帧')
        elif feedback and feedback.get('local_patch'):
            html_text = local_patch(project, directory, old_html, feedback['local_patch'])
            log('修复', f'{shot_id}：应用绑定源版本的本地代码补丁，随后完整审核')
        elif policy_recheck and old_html and old_html.exists():
            html_text = preserve_for_policy_review(project, directory, old_html, previous)
            log('复核', f'{shot_id}：保留已有动画，按新的审核或渲染一致性设置完整复验')
        elif old_html and old_html.exists() and (replan or repair.needs_detail_confirmation(previous) or
                                                motion_confirmation or entry_confirmation):
            html_text = old_html.read_text(encoding='utf-8')
            reason = ('已独立审核修订后的帧计划：先按新要求完整验证现有画面，再决定是否修改代码。'
                      if replan else
                      '旧首帧审核缺少上一镜头实际图：补充进入状态参考后复核原画面。'
                      if entry_confirmation else
                      '旧方向审核未区分跨帧位移与当前方位：按明确参照系复核原画面，再决定是否修改代码。'
                      if motion_confirmation else '旧报告尚无区域复核：保留原代码，在新候选内按全部要求验证真实像素。')
            write_json(directory / 'repair_strategy.json', dict(strategy,
                strategy='verify_existing_scene',
                instruction=reason))
            log('复核', f'{shot_id}：{reason}')
        elif old_html and old_html.exists():
            payload['html'] = old_html.read_text(encoding='utf-8')
            payload.pop('previous_candidate', None)  # 实际 HTML 才是补丁基准，不能把上一轮 patch JSON 当代码。
            changes = project.call(directory, 'generated', model.messages(prompts.PATCH, payload, refs + repair_images))
            html_text = patched_html(payload['html'], changes)
            validation.require(html_text != payload['html'], '修复没有改变任何代码')
        else:
            value = project.call(directory, 'generated', model.messages(prompts.CANVAS, payload, refs + repair_images))
            validation.text(value.get('html'), 'html')
            html_text = value['html']
        html = directory / 'scene.html'
        html.write_text(html_text, encoding='utf-8')
        carried_motion = carried_keyframe_motion(project, directory, html, local_plan, review_options)
        carried_proof = {'source_hash': file_hash(html), 'targets': {}}
        approved = []
        # 锚定帧先审核，再渲染和审核下一关键帧；比较约束始终传入真实参考与前帧。
        for key in local_plan['keyframes']:
            target = f'{shot_id}_{key["frame_id"]}'
            folder = directory / 'keyframes' / key['frame_id']
            if not (folder / 'preview_manifest.json').exists():
                rendering.render_job(html, folder, 2, fps, preview=True, frame_count=count,
                                     sample_indices=[key['frame_index']], **render_options(project))
            current = rendering.preview_images(folder)
            comparisons = approved[-1:] if approved else []
            motion_context = {}
            if motion_details is not None and motion_details['target_review'] == 'check_' + key['frame_id']:
                validation.require(file_hash(html) == motion_details['source_hash'], '运动复验源码已改变')
                validation.require(comparisons and all(comparisons[-1][0] < index < key['frame_index']
                    for index in motion_details['frame_indices']), '补充运动帧不属于当前关键帧间隔')
                motion_folder = folder / 'motion_recheck'
                if not (motion_folder / 'preview_manifest.json').exists():
                    rendering.render_job(html, motion_folder, 2, fps, preview=True, frame_count=count,
                                         sample_indices=motion_details['frame_indices'], **render_options(project))
                intermediate = rendering.preview_images(motion_folder)
                validation.require([index for index, _ in intermediate] == motion_details['frame_indices'],
                                   '运动复验实际采样与绑定帧号不一致')
                comparisons += intermediate
                motion_context = keyframe_motion_context(comparisons + current, fps,
                                                        motion_details['frame_indices'], review_options)
                provenance = directory / 'keyframe_motion_recheck_provenance.json'
                proof = read_json(provenance)
                proof['effective_options'] = model.effective_settings(review_options)
                proof['actual_frames'] = [{'frame_index': index, 'path': relative(project, path), 'hash': file_hash(path)}
                                          for index, path in comparisons + current]
                write_json(provenance, proof)
            elif 'check_' + key['frame_id'] in carried_motion:
                name = 'check_' + key['frame_id']
                carried = carried_motion[name]
                bound = {row['frame_index']: row['hash'] for row in carried['provenance']['actual_frames']}
                validation.require(all(bound.get(index) == file_hash(path) for index, path in comparisons + current),
                                   '同源码关键帧与已通过的运动证据像素不一致，不能继承')
                comparisons += carried['images']
                motion_context = carried['context']
                carried_proof['targets'][name] = carried['provenance']
                write_json(directory / 'keyframe_motion_carry_provenance.json', carried_proof)
                log('继承证据', f'{target} 保留同源码已通过的实际运动补帧，仍核对完整审核签名')
            entry_images, entry_context = [], {}
            if previous_shot and not approved and any(
                    word in item['text'] for item in key['requirements']
                    for word in ('上一镜头', '前一镜头')):
                entry_images = [('上一镜头实际末帧，仅作进入状态比较；不是本镜头当前帧', previous_image)]
                entry_context = {'entry_reference': {
                    'shot_id': previous_shot['shot_id'], 'version': previous_shot['version'],
                    'image': previous_shot['last_frame'], 'image_hash': file_hash(previous_image),
                    'instruction': '涉及上一镜头尺寸/姿态/背景承接时，比较该实际参考图与当前帧；'
                        '当前帧的位置和状态仍只依据当前帧。不得从计划文字臆测上一镜头画面。'}}
            # 独立描述保留为诊断文件，不将其可能错误的叙述注入像素审核或其复用签名。
            observe_keyframes(project, directory,
                f'keyframe_observation_{key["frame_id"]}', comparisons + current)
            review(project, directory, f'check_{key["frame_id"]}', target, version, key['requirements'],
                   {'shot': spec, 'task': key, 'fixed_assets': references,
                    'keyframe_review_protocol': KEYFRAME_REVIEW_PROTOCOL,
                    'visual_review_extra_body': review_options,
                    'pixel_evidence_context': {
                        'current_frame_index': key['frame_index'],
                        'comparison_frame_indices': [i for i, _ in comparisons],
                        'instruction': '当前状态只由当前实际帧判断；邻帧用于比较；标准资产用于身份与材质参考，'
                            '不能替代当前画面或动作证据。'},
                    'allowed_black_ranges': local_plan.get('allowed_black_ranges', []), **entry_context, **motion_context},
                   refs + entry_images + image_inputs(comparisons + current), [i for i, _ in comparisons + current],
                   visual_scope='frame', extra_body=review_options)
            approved += current
        indices = validation.sample_indices(local_plan, count)
        preview = directory / 'preview'
        if not (preview / 'preview_manifest.json').exists():
            rendering.render_job(html, preview, 2, fps, preview=True, frame_count=count, sample_indices=indices, **render_options(project))
        images = rendering.preview_images(preview)
        temporal_result = temporal.review_windows(project, directory, shot_id, version,
               spec['requirements'] + [req('TEMPORAL', '动作完整、接触持续、身份稳定、因果顺序正确，无意外闪烁或漂移')],
               {'shot': spec, 'frame_plan': local_plan,
                'temporal_evidence_protocol': temporal.PROTOCOL, 'fps': fps,
                'sample_timing': [
                    {'frame_index': index, 'time_seconds': index / fps,
                     'gap_from_previous_frames': index - images[position - 1][0] if position else None}
                    for position, (index, _) in enumerate(images)],
                'temporal_sampling_context': 'sample_timing来自本次实际附图清单，时间以当前镜头局部帧号除以fps计算。'
                    '非连续采样不能当作连续帧；跨多个帧的画面差异本身不能证明瞬时跳变。'
                    '判断瞬时跳变、闪烁须有相邻真实帧的证据；证据不足则UNCERTAIN，'
                    '不能根据计划臆测中间正常，也不能忽略实际画面已显示的身份、连接或动作缺失。'
                    '其余动作、接触、身份、体积与光影要求仍须完整审核。',
                'next_shot_entry': {'shot_id': next_shot['shot_id'], 'entry_state': next_shot['entry_state']}
                    if next_shot else None,
                'verification_scope': '本阶段检查当前镜头实际帧；涉及下一镜头衔接时，'
                    '检查当前末帧是否满足当前 exit_state 及已审核的 next_shot_entry。'
                    '下一镜头尚未绘制，不能因缺少其实际图片判失败，也不能声称见到了其画面。'
                    'next_shot_entry 用于核对身份和动作状态；若两镜头 camera 不同，'
                    '不能强制其屏幕坐标与当前末帧完全相同，本镜头自身的 exit_state 仍须满足。'
                    '动作发生时段以已审核frame_plan的intervals和keyframes为准：'
                    '逐项比较动作区间内的真实帧，不能因为首尾为静止姿态就推断中间没有动作，'
                    '也不能要求计划中的进入、保持或落定阶段持续保持中间运动姿态。'
                    '仍须完整验证所有运动区间、实际位移和动作结果；要求全程成立的连接或身份'
                    '在任何采样帧不满足时仍必须失败。'
                    '两侧实际像素的衔接由所有镜头生成后的 checks/timeline 独立审核，'
                    '任何边界未通过都不能发布 final.mp4。'},
               refs, images, details=temporal_details)
        if not (preview / 'video.mp4').exists():
            rendering.render_job(html, preview, 2, fps, frame_count=count, sample_indices=indices, **render_options(project))
        technical = rendering.inspect_video(preview / 'video.mp4', count, fps, local_plan.get('allowed_black_ranges', []))
        write_json(directory / 'technical_check.json', technical)
        return {'shot_id': shot_id, 'version': version, 'video': relative(project, preview / 'video.mp4'),
                'html': relative(project, html), 'frame_plan': local_plan,
                'start_frame': spec['start_frame'], 'end_frame': spec['end_frame'],
                'first_frame': relative(project, approved[0][1]), 'last_frame': relative(project, approved[-1][1]),
                **({'relative_motion_reference': True,
                    'relative_motion_policy_version': temporal.RELATIVE_MOTION_POLICY_VERSION}
                   if temporal_result.get('relative_motion_reference') is True else {}),
                'samples': [{'frame_index': i, 'path': relative(project, path)} for i, path in images]}

    return project.task(f'segments/{shot_id}', {'shot': spec, 'frames': frames, 'repair': feedback,
                        'review_protocol': KEYFRAME_REVIEW_PROTOCOL,
                        'temporal_evidence_protocol': temporal.PROTOCOL,
                        'visual_review_extra_body': review_options},
                        deps, action, max_attempts=5)
