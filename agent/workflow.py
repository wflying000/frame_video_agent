"""分步验收、局部修复和最终发布的调度流程。"""
from . import drawing, model, planning, prompts, rendering, validation
from .common import WIDTH, HEIGHT, log, read_json, write_json
from .planning import req, review
from .store import Blocked, Project, Rejected


def pacing_text(board, fps, seconds):
    """完整保留分镜文字，时间由正式视频的整数帧边界计算。"""
    lines = ['分镜节奏计划（文字）',
             f'总时长：{seconds} 秒；帧率：{fps} fps；总帧数：{seconds * fps}',
             '帧区间采用 [开始帧, 结束帧)，不包含结束帧。',
             '此文件描述计划；实际动作与画面仍需通过后续视觉审核。']
    for shot in board['shots']:
        start, end = shot['start_frame'], shot['end_frame']
        lines += ['', f'{shot["shot_id"]} | {start / fps:.3f}–{end / fps:.3f} 秒'
                  f' | 时长 {(end - start) / fps:.3f} 秒 | 帧 [{start}, {end})',
                  '事件：' + ', '.join(shot['event_ids']),
                  '资产：' + ', '.join(shot['asset_ids'])]
        for label, field in [('目的', 'purpose'), ('机位', 'camera'), ('进入状态', 'entry_state'),
                             ('动作', 'action'), ('结束状态', 'exit_state')]:
            lines.append(f'{label}：{shot[field]}')
        lines.append('检查要求：')
        for item in shot['requirements']:
            level = '硬性' if item['hard'] else '可选'
            lines.append(f'  {item["id"]}（{level}）：{item["text"]}')
    return '\n'.join(lines) + '\n'


def check_pacing(project, board, assets):
    """文字分镜节奏检查；保留旧任务 ID、输入与依赖以兼容已通过项目。"""
    config = project.config
    payload = {'shots': board['shots'], 'fps': config['fps'], 'seconds': config['seconds']}
    text = pacing_text(board, config['fps'], config['seconds'])

    def action(directory, version, previous):
        log('节奏', f'文字检查 {len(board["shots"])} 个镜头，共 {config["seconds"]} 秒')
        path = directory / 'pacing.txt'
        path.write_text(text, encoding='utf-8')
        review(project, directory, 'pacing_review', 'animatic', version,
               [req('PACING', '根据文字分镜检查镜头顺序、时长和动作节奏是否可实现，'
                    '结束状态能否衔接；只判断计划，不把文字当作实际动作证据')],
               {'storyboard': board, 'fps': config['fps'], 'seconds': config['seconds'],
                'timeline_text': text, 'type': '文字分镜节奏检查'}, mode='plan')
        return {'text': drawing.relative(project, path), 'status': 'PASS'}

    # 旧版 PASS 已审核同一份完整分镜，可复用其审核；不改写历史版本或重绘占位图。
    project.task('plans/animatic', payload,
                 ['plans/storyboard'] + [f'assets/{key}' for key in assets], action)
    path = project.root / 'plans/pacing.txt'
    path.write_text(text, encoding='utf-8')
    log('节奏', f'文字时间线已保存：{path}')
    return path


def check_boundaries(project, board, shots):
    deps = [f'segments/{shot["shot_id"]}' for shot in shots]
    def action(directory, version, previous):
        for i in range(1, len(shots)):
            left, right = shots[i - 1], shots[i]
            left_count, right_count = left['end_frame'] - left['start_frame'], right['end_frame'] - right['start_frame']
            left_video, right_video = project.root / left['video'], project.root / right['video']
            # 稀疏的镜头采样可能跳过紧贴边界的帧；从编码视频提取两侧连续帧。
            folder = directory / f'boundary_{i}_frames'
            left_frames = rendering.sample_video(left_video, folder / 'left', list(range(max(0, left_count - 3), left_count)))
            right_frames = rendering.sample_video(right_video, folder / 'right', list(range(min(3, right_count))))
            images = [(index + left['start_frame'], path) for index, path in left_frames]
            images += [(index + right['start_frame'], path) for index, path in right_frames]
            changes = rendering.boundary_changes(left_video, right_video, left_count, right_count)
            write_json(directory / f'boundary_{i}_changes.json', changes)
            review(project, directory, f'boundary_{i}', f'{left["shot_id"]}_{right["shot_id"]}', version,
                   [req('CONTINUITY', '镜头边界人物、道具与状态符合分镜，动作和因果持续，无无故跳变或消失'),
                    req('MOTION_PHASE', '连续动作在边界承接前帧的部件姿态、形变和运动相位；'
                        '不得因新镜头局部时间归零而出现头部回缩、身体突然伸直、尾部跳位或背景无故消失')],
                   {'left_shot': board['shots'][i - 1], 'right_shot': board['shots'][i],
                    'continuity_review_version': 3,
                    'boundary_frames': [left['end_frame'] - 1, right['start_frame']],
                    'decoded_frame_changes': changes,
                    'boundary_phase_context': {
                        'left': {'sampled_local_frames': [index for index, _ in left_frames],
                                 'global_start_frame': left['start_frame'],
                                 'frame_plan': left['frame_plan']},
                        'right': {'sampled_local_frames': [index for index, _ in right_frames],
                                  'global_start_frame': right['start_frame'],
                                  'frame_plan': right['frame_plan']}},
                    'verification_scope': '先核对两侧分镜的 camera。若计划明确切换视角或景别，'
                        '屏幕坐标、尺寸和所见正背面可随机位变化，不能仅凭这些差异判角色转身跳变，'
                        '也不要求切镜之间存在单镜头式的转身动画。仍须核对剧情中的空间关系、'
                        '角色身份、物体开闭和动作结果；机位变化不能解释的道具状态改变必须失败。'
                        '机位描述不同不能自动豁免动作状态连续性；应明确区分合理投影变化与部件相位重置。'
                        '逐帧核对 boundary_frames 以及其前后连续帧，证据必须解释切点实际变化。'
                        'boundary_phase_context 给出已审核的局部帧计划及实际采样范围；'
                        '先将全局帧号换算为对应镜头局部帧号，再按该时刻应有的状态核对。'
                        '只判断所提供切点邻域内的连续性，不能要求计划稍后发生的动作或道具'
                        '提前出现在新镜头首三帧，也不能据此声称整段镜头漏掉动作。'
                        '后续完整动作仍由该镜头时序审核和成片事件审核验证；'
                        '若当前时刻已经应当出现的动作、连接关系或道具缺失，仍必须失败。'
                        '画面不足以判断边界连续性时用 UNCERTAIN，不按预设结论补全证据。'},
                   drawing.image_inputs(images), [frame for frame, _ in images])
        return {'status': 'PASS', 'boundaries': max(0, len(shots) - 1)}
    return project.task('checks/timeline', {'storyboard': board, 'continuity_review_version': 3}, deps, action,
                        max_attempts=3, retry=False)


def observe_batch(project, directory, name, batch):
    """独立观察的字段错误在同一候选内有限修复，不替模型猜测帧号。"""
    indices = [frame for frame, _ in batch]
    context = {'frame_indices': indices, 'output_contract': {
        'observations': '每个给定帧一个对象，保持原顺序',
        'required_fields': {'frame_index': '整数，必须使用这个精确字段名', 'description': '非空文本'}}}
    for attempt in range(2):
        call_name = name if attempt == 0 else name + '_format_retry'
        observed = project.call(directory, call_name,
            model.messages(prompts.OBSERVE, context, drawing.image_inputs(batch)))
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
                raise Blocked(f'{name} 独立观察字段格式连续无效，已保留候选和诊断；'
                              '修复原因后可 --resume，已消耗预算不清零。') from error
            context.update(format_error=str(error), previous_observations=observed)
            log('格式', f'{name}：{error}；同一候选内有限修正一次')


def final_check(project, facts, board, shots):
    config = project.config
    deps = ['checks/timeline'] + [f'segments/{shot["shot_id"]}' for shot in shots]
    def action(directory, version, previous):
        video = directory / 'candidate_final.mp4'
        if not video.exists():
            rendering.encode_segments([project.root / shot['video'] for shot in shots], video)
        allowed_black = [[a + shot['start_frame'], b + shot['start_frame']]
                         for shot in shots for a, b in shot['frame_plan'].get('allowed_black_ranges', [])]
        technical = rendering.inspect_video(video, config['seconds'] * config['fps'], config['fps'], allowed_black)
        write_json(directory / 'technical_check.json', technical)
        # 从实际编码成片取样，不能用生成器文字或未编码的源帧替代成片验收。
        indices = sorted({key['frame_index'] + shot['start_frame'] for shot in shots
                          for key in shot['frame_plan']['keyframes']} |
                         {i for shot in shots for i in (shot['start_frame'], shot['end_frame'] - 1)})
        if not (directory / 'frames/preview_manifest.json').exists():
            rendering.sample_video(video, directory / 'frames', indices)
        images = rendering.preview_images(directory / 'frames')
        observations = []
        for i in range(0, len(images), 12):
            batch = images[i:i + 12]
            observations += observe_batch(project, directory, f'observe_{i // 12}', batch)
        write_json(directory / 'independent_observations.json', observations)
        coverage = {}
        for event in facts['events']:
            relevant = [shot for shot in board['shots'] if event['event_id'] in shot['event_ids']]
            relevant_ids = {shot['shot_id'] for shot in relevant}
            evidence = [(frame, path) for frame, path in images
                        if any(shot['start_frame'] <= frame < shot['end_frame'] for shot in relevant)]
            report = review(project, directory, f'event_{event["event_id"]}', event['event_id'], version,
                            [req(event['event_id'], event['must_show'])],
                            {'original_story': config['story'], 'event': event,
                             'shots': relevant, 'independent_observations': observations,
                             'frame_plan': {'keyframes': [
                                 dict(key, frame_id=f'{rendered["shot_id"]}_{key["frame_id"]}',
                                      frame_index=key['frame_index'] + rendered['start_frame'])
                                 for rendered in shots if rendered['shot_id'] in relevant_ids
                                 for key in rendered['frame_plan']['keyframes']]}},
                            drawing.image_inputs(evidence), [frame for frame, _ in evidence])
            frames = report['checks'][0]['evidence_frames']
            coverage[event['event_id']] = {'frame_range': [min(frames), max(frames) + 1],
                                           'evidence_frames': frames, 'report': report}
        # 单事件通过后仍独立核对全片因果、顺序和结局。
        review(project, directory, 'story_review', 'final_story', version,
               [req('NARRATIVE', '独立观察与原文事件、因果、转折和结局一致，没有 unresolved 项或被字幕代替的动作')],
               {'original_story': config['story'], 'facts': facts, 'coverage': coverage,
                'independent_observations': observations}, mode='observations')
        write_json(directory / 'event_evidence.json', coverage)
        return {'video': drawing.relative(project, video), 'status': 'PASS', 'coverage': coverage}
    return project.task('exports/final', {'story': facts, 'storyboard': board}, deps, action,
                        max_attempts=3, retry=False)


def generate(story, out, seconds, fps, resume=False, max_calls=120, max_seconds=3600,
             task_attempt_limits=None):
    config = {'pipeline': 'staged-canvas-v1', 'story': story, 'seconds': seconds, 'fps': fps,
              'width': WIDTH, 'height': HEIGHT, 'audio': False, 'style': '2D Canvas finite animation'}
    with Project(out, config, resume, max_calls, max_seconds, task_attempt_limits) as project:
        log('规划', '先审核故事、视觉设定与分镜，再开始绘制')
        facts, visual, board = planning.plan(project)
        references = drawing.assets(project, facts, visual)
        check_pacing(project, board, references)
        frame_plans = {shot['shot_id']: planning.plan_frames(project, shot) for shot in board['shots']}
        while True:
            results = []
            for index, shot in enumerate(board['shots']):
                results.append(drawing.shot(project, shot, frame_plans[shot['shot_id']], references,
                                            results[-1] if results else None,
                                            board['shots'][index + 1] if index + 1 < len(board['shots']) else None))
            try:
                check_boundaries(project, board, results)
                result = final_check(project, facts, board, results)
                final = project.publish(project.root / result['video'])
                log('完成', f'{final}；项目已验收，模型累计调用 {project.data["calls"]} 次')
                return final
            except Rejected as failure:
                targets = failure.report.get('repair', {}).get('shot_ids', [])
                valid_ids = {shot['shot_id'] for shot in board['shots']}
                if not targets or not set(targets) <= valid_ids or project.data['global_repairs'] >= 2:
                    raise Blocked('全片检查未通过，无法可靠定位或已达到两次全片修复上限；查看 checks/exports 报告。')
                project.data['global_repairs'] += 1
                for shot_id in set(targets):
                    project.data['repair_requests'][shot_id] = failure.report
                    project.invalidate(f'segments/{shot_id}')
                project.checkpoint()
                log('局部修复', f'仅重做受影响镜头：{", ".join(targets)}；其他通过镜头复用')
