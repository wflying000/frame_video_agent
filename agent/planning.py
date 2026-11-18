"""规划阶段：解析与审核分离，先通过结构检查再进行语义验收。"""
from copy import deepcopy
import json
from . import consistency, model, prompts, rendering, validation
from .common import digest, file_hash, log, read_json, write_json
from .store import Rejected


def req(key, text):
    return {'id': key, 'hard': True, 'text': text}


def depth_requirements(requirements, scope):
    """追加程序持有的美术硬门槛，不依赖生成器自行列出，也不覆盖原检查项。"""
    validation.require(scope in {'plan', 'reference', 'frame', 'temporal'}, '未知体积与纵深审核范围')
    result = list(requirements)
    used = {item['id'] for item in result}
    groups = ['base', 'reference' if scope == 'reference' else 'scene']
    if scope == 'temporal':
        groups.append('temporal')
    for group in groups:
        for key, text in prompts.DEPTH_CHECKS[group].items():
            # 模型若恰好使用同名 ID，也保留它的原要求；程序规则仍独立审核。
            check_id = key
            while check_id in used:
                check_id += '_POLICY'
            used.add(check_id)
            result.append(req(check_id, text))
    return result


def depth_frame_indices(frame_ids, allowed_black_ranges):
    """已声明的黑屏转场不能充当物体体积或空间纵深的证据。"""
    return [frame for frame in frame_ids
            if not any(start <= frame < end for start, end in allowed_black_ranges)]


def validate_review_evidence(report, frame_ids, context):
    """新审核与跨候选复用使用同一证据门槛，格式失败不能在恢复时变成 PASS。"""
    available = set(frame_ids)
    for row in report['checks']:
        if row['result'] != 'PASS':
            continue
        evidence = row.get('evidence_frames')
        validation.require(isinstance(evidence, list) and evidence and
                           all(type(i) is int and i in available for i in evidence),
                           '视觉审核 PASS 必须引用实际提供的帧号')
        if row['requirement_id'] in context.get('visual_depth_check_ids', []):
            depth_frames = context['visual_depth_frame_indices']
            validation.require(set(evidence) <= set(depth_frames),
                               '体积与纵深 PASS 不能引用黑屏转场作为证据')
            if context.get('visual_scope') == 'frame':
                validation.require(context['task']['frame_index'] in evidence,
                                   '关键帧体积与纵深 PASS 必须引用当前帧')
            if row['requirement_id'].startswith('DEPTH_STABILITY'):
                validation.require(len(set(evidence)) >= min(2, len(depth_frames)),
                                   '光影与体积稳定性 PASS 必须比较实际非黑屏帧')


def reuse_review(project, directory, name, messages, target, version, requirements, frame_ids, extra_body=None):
    """只复用相同要求、真实像素与配置的 PASS，显式绑定原审核证据。"""
    def signature(recorded):
        recorded = deepcopy(recorded)
        context = json.loads(recorded[1]['content'][0]['text'])
        context.pop('target_version', None)
        recorded[1]['content'][0]['text'] = json.dumps(context, ensure_ascii=False, sort_keys=True)
        return digest(recorded)
    recorded = model.recorded_messages(messages)
    evidence_context = json.loads(recorded[1]['content'][0]['text'])
    expected, settings = signature(recorded), model.effective_settings(extra_body)
    for prior_dir in sorted(directory.parent.glob('v*'), reverse=True):
        if prior_dir == directory:
            continue
        source = prior_dir / f'{name}.json'
        request = prior_dir / f'{name}.request.json'
        call = prior_dir / f'{name}.call.json'
        if not all(path.is_file() for path in (source, request, call)):
            continue
        metadata, report = read_json(call), read_json(source)
        prior_request = read_json(request)
        if metadata.get('model') != settings or signature(prior_request) != expected:
            continue
        resolution_proof = None
        if report.get('status') != 'PASS':
            resolution = prior_dir / f'{name}_resolution.json'
            detail = prior_dir / f'{name}_detail.json'
            detail_request = prior_dir / f'{name}_detail.request.json'
            detail_call = prior_dir / f'{name}_detail.call.json'
            if not all(path.is_file() for path in (resolution, detail, detail_request, detail_call)):
                continue
            proof = read_json(resolution)
            if (proof.get('status') != 'PASS' or
                    proof.get('review_protocol') != evidence_context.get('pixel_review_protocol') or
                    read_json(detail_call).get('model') != settings):
                continue
            detailed = read_json(detail_request)
            base_context = json.loads(detailed[1]['content'][0]['text'])
            base_context.pop('detail_confirmation', None)
            detailed[1]['content'][0]['text'] = json.dumps(base_context, ensure_ascii=False)
            detailed[1]['content'] = detailed[1]['content'][:len(prior_request[1]['content'])]
            if signature(detailed) != expected or any(
                    not (project.root / row['path']).is_file() or
                    file_hash(project.root / row['path']) != row['hash']
                    for row in proof.get('detail_images', [])):
                continue
            source, report = detail, read_json(detail)
            resolution_proof = {'path': str(resolution.resolve().relative_to(project.root)),
                                'sha256': file_hash(resolution)}
        if report.get('status') != 'PASS':
            continue
        try:
            if not validation.review(report, target, report['target_version'], requirements):
                continue
            validate_review_evidence(report, frame_ids, evidence_context)
        except (ValueError, KeyError, TypeError):
            continue
        result = dict(report, target_version=version, reused_from={
            'path': str(source.resolve().relative_to(project.root)), 'sha256': file_hash(source),
            'target_version': report['target_version'], 'evidence_signature': expected})
        if resolution_proof:
            result['reused_from']['detail_resolution'] = resolution_proof
        write_json(directory / f'{name}.json', result)
        write_json(directory / f'{name}.request.json', recorded)
        write_json(directory / f'{name}.call.json', {
            'input_hash': digest({'messages': messages, 'settings': settings}), 'model': settings,
            'status': 'PASS', 'reused_from': result['reused_from']})
        log('复用审核', f'{target} 的要求与图片完全一致，沿用 v{report["target_version"]} 的通过证据')
        return result


def review(project, directory, name, target, version, requirements, payload, images=(), frame_ids=None,
           mode=None, visual_scope=None, extra_body=None):
    images = rendering.compact_review_images(directory / (name + '_transport'), images)
    depth_frames = None
    if visual_scope is not None and frame_ids is not None:
        black_ranges = payload.get('allowed_black_ranges', payload.get('frame_plan', {}).get('allowed_black_ranges', []))
        depth_frames = depth_frame_indices(frame_ids, black_ranges)
        current = payload.get('task', {}).get('frame_index')
        if visual_scope == 'frame' and current in frame_ids and current not in depth_frames:
            # 只豁免已声明的黑屏当前帧；原状态检查和其他实际画面的美术门槛仍保留。
            payload = dict(payload, visual_depth_exemption={
                'reason': 'declared_black_transition', 'frame_index': current})
            visual_scope = None
        elif visual_scope == 'temporal' and not depth_frames:
            intervals = payload.get('frame_plan', {}).get('intervals', [])
            if intervals and not depth_frame_indices(
                    range(intervals[0]['start_frame'], intervals[-1]['end_frame']), black_ranges):
                payload = dict(payload, visual_depth_exemption={
                    'reason': 'declared_black_transition', 'frame_indices': list(frame_ids)})
                visual_scope = None
    if visual_scope is not None:
        original_count = len(requirements)
        requirements = depth_requirements(requirements, visual_scope)
        payload = dict(payload, visual_scope=visual_scope,
                       visual_depth_check_ids=[item['id'] for item in requirements[original_count:]])
        if depth_frames is not None:
            validation.require(depth_frames, '体积与纵深审核必须有非黑屏的实际画面证据')
            payload['visual_depth_frame_indices'] = depth_frames
    context = dict(payload, target_id=target, target_version=version, requirements=requirements,
                   required_check_ids=[r['id'] for r in requirements],
                   evidence_mode=mode or ('pixels' if images else 'plan'))
    if images and context['evidence_mode'] == 'pixels' and visual_scope not in ('plan', 'reference'):
        context['consistency_policy'] = consistency.review_policy(getattr(project, 'review_strictness', 'strict'))
    if frame_ids is not None:
        # 协议属于请求证据，旧有带结论复核不能跨协议复用。
        context['pixel_review_protocol'] = 'blind_detail_v2'
        if payload.get('frame_plan'):
            context['spatial_coordinate_reference'] = {
                'version': 1,
                'instruction': '带坐标的位置要求须在全部证据帧使用同一定义的可见参考点。'
                    '原文明示部位或接触点时必须测量该部位；只给角色/鱼体中心坐标时，'
                    '使用主要躯干的中心，不采用包含摆动尾鳍、背鳍或伸出工具的整体包围框中心。'
                    '不能因尾部伸展或转向就改变中心定义，也不能把包围框变化当主体位移。'
                    '从真实画面定位起点、关键过渡与终点，列出同一躯干参考点的近似归一化坐标；'
                    '若像素不足以定位则返回UNCERTAIN，不能用计划坐标替代观测，不能为了通过换参考点。'}
        from .repair import motion_comparison_policy
        task = payload.get('task', {})
        policy = (motion_comparison_policy(requirements)
                  if task.get('frame_index') in frame_ids and
                  any(i != task['frame_index'] for i in frame_ids) else None)
        # 旧版未记录 pixel_scope 的审核也不能绕过新增的方向参照规则。
        if policy is None:
            reused = reuse_review(project, directory, name, model.messages(prompts.REVIEW, context, images),
                                  target, version, requirements, frame_ids, extra_body=extra_body)
            if reused:
                return reused
        # 关键帧检查经常涉及几十像素的小接触区；提供实际局部图，避免从全景估计精确坐标。
        if len(frame_ids) <= 2:
            from .repair import audit_evidence
            images = list(images) + audit_evidence(directory / (name + '_evidence'), requirements, images)
        elif payload.get('frame_plan'):
            from .repair import temporal_evidence
            images = list(images) + temporal_evidence(directory / (name + '_evidence'), payload['frame_plan'], images)
        if task.get('frame_index') in frame_ids:
            context['pixel_scope'] = {
                'current_frame_index': task['frame_index'],
                'comparison_frame_indices': [i for i in frame_ids if i != task['frame_index']],
                'instruction': '静态检查项只能依据 current_frame_index 的实际画面。'
                    '前帧只用于比较变化，不得把前帧姿态写成当前状态。'
                    '标准资产只用于核对造型身份，其姿态、位置、爪垫朝向不是当前帧证据。'
                    '先定位当前帧，再逐条描述可见证据；细节无法辨认时明确 UNCERTAIN，'
                    '不得根据计划文字补全看不见的内容。局部放大图与同帧全图是同一份证据。'}
            if policy:
                context['pixel_scope']['motion_comparison'] = policy
        reused = reuse_review(project, directory, name, model.messages(prompts.REVIEW, context, images),
                              target, version, requirements, frame_ids, extra_body=extra_body)
        if reused:
            return reused
    report, passed = checked_review(project, directory, name, target, version, requirements,
                                    context, images, frame_ids, extra_body=extra_body)
    if not passed and frame_ids is not None and len(frame_ids) <= 2:
        from .repair import review_details
        details = [(label.replace('失败区域复核：', '区域细节：', 1), path)
                   for label, path in review_details(
                       directory / (name + '_detail_evidence'), report, images, frame_ids)]
        if details:
            # 增加此前没有的区域细节后只复核一次，不能重复抽样直到 PASS。
            initial = directory / (name + '_initial_result.json')
            write_json(initial, report)
            # 初审判断和修复建议仅留在本地溯源文件；复核从原要求和像素独立判断。
            context = dict(context, detail_confirmation={
                'protocol': 'blind_detail_v2',
                'instruction': '依据原始全图和同一帧真实像素的区域放大图独立审核。'
                    '必须逐项核对全部 required_check_ids，不能只审局部或省略原要求。'
                    '局部图不代表额外帧，也不能代替全图的状态与空间证据；看不清则 UNCERTAIN。'})
            images = list(images) + details
            log('复核', f'{target} 增加真实区域细节，最多复核一次')
            report, passed = checked_review(project, directory, name + '_detail', target, version,
                                            requirements, context, images, frame_ids, extra_body=extra_body)
            resolution = {
                'review_protocol': 'blind_detail_v2',
                'initial_report': initial.name, 'initial_hash': file_hash(initial),
                'detail_report': name + '_detail.json', 'status': report['status'],
                'detail_images': [{'label': label, 'path': str(path.resolve().relative_to(project.root)),
                                   'hash': file_hash(path)} for label, path in details]}
            write_json(directory / (name + '_resolution.json'), resolution)
            report = dict(report, detail_confirmation=resolution)
    if not passed:
        if images:
            report['review_images'] = [
                {'label': label, 'path': str(path.resolve().relative_to(project.root)), 'hash': file_hash(path)}
                for label, path in images]
        raise Rejected(report)
    return report


def checked_review(project, directory, name, target, version, requirements, context, images, frame_ids, extra_body=None):
    """只修复响应格式，不重试语义结论。"""
    context = dict(context)
    for attempt in range(2):
        call_name = name if attempt == 0 else name + '_format_retry'
        report = project.call(directory, call_name, model.messages(prompts.REVIEW, context, images),
                              **({'extra_body': extra_body} if extra_body else {}))
        try:
            passed = validation.review(report, target, version, requirements)
            if frame_ids is not None:
                validate_review_evidence(report, frame_ids, context)
        except (ValueError, KeyError, TypeError) as error:
            if attempt:
                raise
            context.update(format_error=str(error), previous_review=report)
            continue
        return report, passed


def prior(project, previous):
    if not previous:
        return {}
    directory = project.root / previous['path']
    result = {'feedback': previous.get('report', {})}
    generated = directory / 'generated.json'
    if generated.exists():
        result['previous_candidate'] = read_json(generated)
    return result


def frame_plan_context(shot, fps):
    count = shot['end_frame'] - shot['start_frame']
    return {'shot': shot, 'frame_count': count, 'fps': fps,
            'generation_contract': {
                'frame_index_range': [0, count - 1], 'interval_end_boundary': count,
                'requirements_rule': '每个关键帧（含首帧和保持帧）必须有非空 requirements，'
                    '且至少一项 hard=true。没有已发生动作时检查 state 中的进入姿态、位置、身份；'
                    '这时 covers 可以为空，不得为填 covers 虚构动作已完成。'
            }}


def prepare_frame_plan(directory, value):
    """将没有动作覆盖的状态帧补成可审核任务；不删减要求或改写模型原文。"""
    validation.require(isinstance(value, dict), '帧计划必须是 JSON 对象')
    result = deepcopy(value)
    changes = []
    for index, key in enumerate(validation.items(result.get('keyframes'), 'keyframes')):
        validation.require(isinstance(key, dict), f'keyframes[{index}] 必须是对象')
        # 仅处理明确为空的两个列表。字段缺失、类型错误、声明覆盖动作却
        # 没有检查项等情况继续交给严格校验，不能替模型编造动作证据。
        if key.get('requirements') == [] and key.get('covers') == []:
            validation.identifier(key.get('frame_id'))
            validation.text(key.get('state'), f'keyframes.{key["frame_id"]}.state')
            check = req('STATE', '本帧画面必须符合以下计划状态（仅检查静态可见姿态、位置、身份和接触，'
                        '不以单帧证明速度或持续时间）：' + key['state'])
            key['requirements'] = [check]
            changes.append({'path': f'keyframes[{index}].requirements',
                            'source': f'keyframes[{index}].state',
                            'before': [], 'after': [check]})
    if changes:
        write_json(directory / 'normalization.json', {'changes': changes})
        write_json(directory / 'normalized.json', result)
        log('帧计划', f'为 {len(changes)} 个状态帧补充硬性画面检查；原始输出保留，仍需独立审核')
    return result


def plan_stage(project, task_id, deps, system, payload, validate, requirements, prepare=None,
               generation_context=None, visual_scope=None):
    def action(directory, version, previous):
        context = dict(payload, **prior(project, previous))
        if generation_context:
            context.update(generation_context)
        value = project.call(directory, 'generated', model.messages(system, context))
        if prepare:
            value = prepare(directory, value)
        validate(value)
        review(project, directory, 'review', task_id, version, requirements,
               {'source': payload, 'candidate': value}, visual_scope=visual_scope)
        return value
    return project.task(task_id, payload, deps, action)


def plan(project):
    config = project.config
    facts = plan_stage(project, 'plans/story', [], prompts.STORY,
                       {'story': config['story'], 'config': config},
                       lambda value: validation.story(value, config['story']),
                       [req('STORY', '事实完整忠实于原文，因果和结局正确，设计选择与事实分开，未删改关键情节')])
    write_json(project.root / 'plans/story_facts.json', facts)
    write_json(project.root / 'plans/event_graph.json', {'order': facts['event_order'], 'edges': facts['causal_links']})
    write_json(project.root / 'plans/story_state_timeline.json', [
        {'event_id': e['event_id'], 'before': e['precondition'], 'after': e['result']} for e in facts['events']])
    visual = plan_stage(project, 'plans/visual', ['plans/story'], prompts.VISUAL,
                        {'story': facts}, lambda value: validation.visual(value, facts),
                        [req('VISUAL', '全部实体的固定造型与可变状态清楚，一致且不与原文冲突')],
                        visual_scope='plan')
    write_json(project.root / 'plans/visual_bible.json', visual)
    board = plan_stage(project, 'plans/storyboard', ['plans/story', 'plans/visual'], prompts.STORYBOARD,
                       {'story': facts, 'visual': visual, 'total_frames': config['seconds'] * config['fps'],
                        'fps': config['fps']},
                       lambda value: validation.storyboard(value, facts, config['seconds'] * config['fps']),
                       [req(event['event_id'], event['must_show']) for event in facts['events']] +
                       [req('ORDER', '事件顺序、因果、镜头边界状态和节奏正确，动作有充分可见时间')],
                       visual_scope='plan')
    write_json(project.root / 'plans/storyboard.json', board)
    write_json(project.root / 'plans/event_shot_coverage.json', {
        e['event_id']: [s['shot_id'] for s in board['shots'] if e['event_id'] in s['event_ids']]
        for e in facts['events']})
    return facts, visual, board


def plan_frames(project, shot):
    value = plan_stage(project, f'plans/frames/{shot["shot_id"]}', ['plans/storyboard'], prompts.FRAME_PLAN,
                       {'shot': shot, 'frame_count': shot['end_frame'] - shot['start_frame'], 'fps': project.config['fps']},
                       lambda data: validation.frame_plan(data, shot),
                       shot['requirements'] + [req('TIME', '全部动作阶段与状态切换有可审核的关键帧和时间点，区间覆盖完整')],
                       prepare=prepare_frame_plan,
                       # 生成时补充帧结构契约；美术硬要求由独立审核统一追加。
                       generation_context=frame_plan_context(shot, project.config['fps']),
                       visual_scope='plan')
    return value
