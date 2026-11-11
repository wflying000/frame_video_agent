"""标准资产、镜头锚定帧、逐关键帧审核与局部代码修复。"""
from . import model, prompts, rendering, repair, validation
from .common import file_hash, log, read_json, write_json
from .planning import frame_plan_context, prepare_frame_plan, prior, req, review


def relative(project, path):
    return str(path.relative_to(project.root))


def image_inputs(images):
    return [(f'实际画面 frame_index={index}', path) for index, path in images]


def assets(project, facts, bible):
    results = {}
    for asset in bible['assets']:
        asset_id = asset['asset_id']
        def action(directory, version, previous, asset=asset, asset_id=asset_id):
            value = project.call(directory, 'generated', model.messages(prompts.CANVAS, {
                'kind': 'reference', 'style': bible['style'], 'asset': asset,
                'duration': 1, **prior(project, previous)}))
            validation.text(value.get('html'), 'html')
            html = directory / 'reference.html'
            html.write_text(value['html'], encoding='utf-8')
            preview = directory / 'preview'
            if not (preview / 'preview_manifest.json').exists():
                rendering.render_job(html, preview, 2, project.config['fps'], preview=True,
                                     frame_count=1, sample_indices=[0])
            images = rendering.preview_images(preview)
            review(project, directory, 'review', asset_id, version, asset['requirements'],
                   {'asset': asset, 'style': bible['style']}, image_inputs(images), [0])
            return {'asset_id': asset_id, 'version': version, 'image': relative(project, images[0][1]),
                    'html': relative(project, html), 'fixed': asset['fixed'], 'variable': asset['variable']}
        results[asset_id] = project.task(f'assets/{asset_id}', asset,
                                        ['plans/story', 'plans/visual'], action)
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


def shot(project, spec, frames, references, previous_shot=None, next_shot=None):
    shot_id = spec['shot_id']
    count, fps = spec['end_frame'] - spec['start_frame'], project.config['fps']
    refs = [(f'标准资产 {key} v{references[key]["version"]}', project.root / references[key]['image'])
            for key in spec['asset_ids']]
    # 镜头共享已通过标准资产；镜头边界单独复验，避免一个局部修复级联重画整片。
    deps = ['plans/storyboard', f'plans/frames/{shot_id}'] + [f'assets/{key}' for key in spec['asset_ids']]
    feedback = project.data['repair_requests'].get(shot_id)

    def action(directory, version, previous):
        old_html, local_plan, strategy = repair.patch_context(project, directory, spec, frames, previous)
        payload = {'kind': 'shot', 'shot': spec, 'frame_plan': local_plan, 'duration': count / fps,
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
        replan = bool(previous and (
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
                   {'shot': spec, 'candidate_frame_plan': local_plan})
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
            'references': [references[key] for key in spec['asset_ids']],
            'input_versions': project.data['tasks'][f'segments/{shot_id}']['input_versions']})
        motion_confirmation = repair.needs_motion_confirmation(project, spec, local_plan, previous)
        entry_confirmation = bool(previous_shot and repair.needs_entry_confirmation(
            project, spec, local_plan, previous))
        if feedback and feedback.get('local_patch'):
            html_text = local_patch(project, directory, old_html, feedback['local_patch'])
            log('修复', f'{shot_id}：应用绑定源版本的本地代码补丁，随后完整审核')
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
        approved = []
        # 锚定帧先审核，再渲染和审核下一关键帧；比较约束始终传入真实参考与前帧。
        for key in local_plan['keyframes']:
            target = f'{shot_id}_{key["frame_id"]}'
            folder = directory / 'keyframes' / key['frame_id']
            if not (folder / 'preview_manifest.json').exists():
                rendering.render_job(html, folder, 2, fps, preview=True, frame_count=count,
                                     sample_indices=[key['frame_index']])
            current = rendering.preview_images(folder)
            comparisons = approved[-1:] if approved else []
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
            review(project, directory, f'check_{key["frame_id"]}', target, version, key['requirements'],
                   {'shot': spec, 'task': key, 'fixed_assets': references, **entry_context},
                   refs + entry_images + image_inputs(comparisons + current), [i for i, _ in comparisons + current])
            approved += current
        indices = validation.sample_indices(local_plan, count)
        preview = directory / 'preview'
        if not (preview / 'preview_manifest.json').exists():
            rendering.render_job(html, preview, 2, fps, preview=True, frame_count=count, sample_indices=indices)
        images = rendering.preview_images(preview)
        review(project, directory, 'temporal_review', shot_id, version,
               spec['requirements'] + [req('TEMPORAL', '动作完整、接触持续、身份稳定、因果顺序正确，无意外闪烁或漂移')],
               {'shot': spec, 'frame_plan': local_plan,
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
               refs + image_inputs(images), [i for i, _ in images])
        if not (preview / 'video.mp4').exists():
            rendering.render_job(html, preview, 2, fps, frame_count=count, sample_indices=indices)
        technical = rendering.inspect_video(preview / 'video.mp4', count, fps, local_plan.get('allowed_black_ranges', []))
        write_json(directory / 'technical_check.json', technical)
        return {'shot_id': shot_id, 'version': version, 'video': relative(project, preview / 'video.mp4'),
                'html': relative(project, html), 'frame_plan': local_plan,
                'start_frame': spec['start_frame'], 'end_frame': spec['end_frame'],
                'first_frame': relative(project, approved[0][1]), 'last_frame': relative(project, approved[-1][1]),
                'samples': [{'frame_index': i, 'path': relative(project, path)} for i, path in images]}

    return project.task(f'segments/{shot_id}', {'shot': spec, 'frames': frames, 'repair': feedback},
                        deps, action, max_attempts=5)
