"""规划阶段：解析与审核分离，先通过结构检查再进行语义验收。"""
from copy import deepcopy
import json
from . import model, prompts, validation
from .common import digest, file_hash, log, read_json, write_json
from .store import Rejected


def req(key, text):
    return {'id': key, 'hard': True, 'text': text}


def reuse_review(project, directory, name, messages, target, version, requirements, frame_ids):
    """只复用相同要求、真实像素与配置的 PASS，显式绑定原审核证据。"""
    def signature(recorded):
        recorded = deepcopy(recorded)
        context = json.loads(recorded[1]['content'][0]['text'])
        context.pop('target_version', None)
        recorded[1]['content'][0]['text'] = json.dumps(context, ensure_ascii=False, sort_keys=True)
        return digest(recorded)
    recorded = model.recorded_messages(messages)
    expected, settings = signature(recorded), model.settings()
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
            if proof.get('status') != 'PASS' or read_json(detail_call).get('model') != settings:
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
        if not validation.review(report, target, report['target_version'], requirements):
            continue
        if any(not isinstance(check.get('evidence_frames'), list) or not check['evidence_frames'] or
               any(type(i) is not int or i not in frame_ids for i in check['evidence_frames'])
               for check in report['checks'] if check['result'] == 'PASS'):
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


def review(project, directory, name, target, version, requirements, payload, images=(), frame_ids=None, mode=None):
    context = dict(payload, target_id=target, target_version=version, requirements=requirements,
                   required_check_ids=[r['id'] for r in requirements],
                   evidence_mode=mode or ('pixels' if images else 'plan'))
    if frame_ids is not None:
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
                                  target, version, requirements, frame_ids)
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
                              target, version, requirements, frame_ids)
        if reused:
            return reused
    report, passed = checked_review(project, directory, name, target, version, requirements,
                                    context, images, frame_ids)
    if not passed and frame_ids is not None and len(frame_ids) <= 2:
        from .repair import review_details
        details = review_details(directory / (name + '_detail_evidence'), report, images, frame_ids)
        if details:
            # 增加此前没有的区域细节后只复核一次，不能重复抽样直到 PASS。
            initial = directory / (name + '_initial_result.json')
            write_json(initial, report)
            context = dict(context, detail_confirmation={
                'initial_report': report,
                'instruction': '依据新增的失败区域真实放大图和原始全图复核。'
                    '首次报告可能正确也可能误读，必须重新核对全部 required_check_ids。'
                    '不要把第一次判断当作事实，不降低要求；仍看不清则 UNCERTAIN。'})
            images = list(images) + details
            log('复核', f'{target} 增加真实区域细节，最多复核一次')
            report, passed = checked_review(project, directory, name + '_detail', target, version,
                                            requirements, context, images, frame_ids)
            resolution = {
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


def checked_review(project, directory, name, target, version, requirements, context, images, frame_ids):
    """只修复响应格式，不重试语义结论。"""
    context = dict(context)
    for attempt in range(2):
        call_name = name if attempt == 0 else name + '_format_retry'
        report = project.call(directory, call_name, model.messages(prompts.REVIEW, context, images))
        try:
            passed = validation.review(report, target, version, requirements)
            if frame_ids is not None:
                available = set(frame_ids)
                for row in report['checks']:
                    if row['result'] == 'PASS':
                        evidence = row.get('evidence_frames')
                        validation.require(isinstance(evidence, list) and evidence and
                                           all(type(i) is int and i in available for i in evidence),
                                           '视觉审核 PASS 必须引用实际提供的帧号')
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
               generation_context=None):
    def action(directory, version, previous):
        context = dict(payload, **prior(project, previous))
        if generation_context:
            context.update(generation_context)
        value = project.call(directory, 'generated', model.messages(system, context))
        if prepare:
            value = prepare(directory, value)
        validate(value)
        review(project, directory, 'review', task_id, version, requirements,
               {'source': payload, 'candidate': value})
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
                        [req('VISUAL', '全部实体的固定造型与可变状态清楚，一致且不与原文冲突')])
    write_json(project.root / 'plans/visual_bible.json', visual)
    board = plan_stage(project, 'plans/storyboard', ['plans/story', 'plans/visual'], prompts.STORYBOARD,
                       {'story': facts, 'visual': visual, 'total_frames': config['seconds'] * config['fps'],
                        'fps': config['fps']},
                       lambda value: validation.storyboard(value, facts, config['seconds'] * config['fps']),
                       [req(event['event_id'], event['must_show']) for event in facts['events']] +
                       [req('ORDER', '事件顺序、因果、镜头边界状态和节奏正确，动作有充分可见时间')])
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
                       # 明确已有的校验契约，不改变已审核任务的输入及语义要求。
                       # 在生成时补充指导，避免导致已通过的分镜和其他镜头失效。
                       generation_context=frame_plan_context(shot, project.config['fps']))
    return value
