"""模型产物的结构、覆盖、整数时间线和审核硬门槛。"""
import re

STATES = {'PASS', 'FAIL', 'UNCERTAIN', 'NOT_APPLICABLE'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', value),
            f'非法标识符：{value!r}')
    return value


def items(value, name, nonempty=True):
    require(isinstance(value, list), f'{name} 必须是列表')
    require(value or not nonempty, f'{name} 必须是非空列表，不能为 []')
    return value


def text(value, name):
    require(isinstance(value, str) and bool(value.strip()), f'{name} 必须是非空文本')


def observations(value, frame_indices):
    require(isinstance(value, dict), '独立观察必须是 JSON 对象')
    rows = items(value.get('observations'), 'observations')
    for index, row in enumerate(rows):
        require(isinstance(row, dict), f'observations[{index}] 必须是对象')
        require(type(row.get('frame_index')) is int,
                f'observations[{index}].frame_index 必须是整数；字段名必须精确为 frame_index')
        text(row.get('description'), f'observations[{index}].description')
    require([row['frame_index'] for row in rows] == list(frame_indices),
            '独立观察必须按给定顺序逐帧覆盖，不得漏帧、重复、增加或改写帧号')
    return rows


def unique(records, field):
    ids = [identifier(record[field]) for record in records]
    require(len(ids) == len(set(ids)), f'{field} 重复')
    return set(ids)


def requirements(value, name='requirements'):
    rows = items(value, name)
    unique(rows, 'id')
    for row in rows:
        text(row.get('text'), f'{name}.{row["id"]}.text')
        require(type(row.get('hard')) is bool, f'{name}.{row["id"]}.hard 必须是布尔值')
    require(any(row['hard'] for row in rows), f'{name} 至少需要一项硬性要求')
    return rows


def story(data, original):
    entities = items(data['entities'], 'entities')
    entity_ids = unique(entities, 'id')
    for entity in entities:
        require(entity.get('kind') in {'character', 'prop', 'location'}, 'entity.kind 无效')
        text(entity.get('description'), 'entity.description')
    events = items(data['events'], 'events')
    ids = unique(events, 'event_id')
    for event in events:
        text(event.get('source_text'), 'source_text')
        require(event['source_text'] in original, '事件缺少原文依据')
        require(set(event['entity_ids']) <= entity_ids, '事件引用未知资产')
        for field in ('action', 'precondition', 'result', 'must_show'):
            text(event.get(field), field)
    require(data['event_order'] == [e['event_id'] for e in events], '事件顺序必须与 events 一致')
    for pair in items(data['causal_links'], 'causal_links', nonempty=False):
        require(isinstance(pair, list) and len(pair) == 2 and set(pair) <= ids, '因果边无效')
        require(data['event_order'].index(pair[0]) < data['event_order'].index(pair[1]), '因果顺序倒置')
    require(data['ending_event_id'] == events[-1]['event_id'], '结局必须对应最后事件')
    items(data['design_choices'], 'design_choices', nonempty=False)
    return data


def visual(data, facts):
    assets = items(data['assets'], 'assets')
    require(unique(assets, 'asset_id') == {e['id'] for e in facts['entities']}, '视觉设定必须覆盖全部资产')
    text(data.get('style'), 'style')
    for asset in assets:
        text(asset.get('fixed'), 'fixed')
        text(asset.get('variable'), 'variable')
        requirements(asset['requirements'])
    return data


def storyboard(data, facts, total_frames):
    shots = items(data['shots'], 'shots')
    require(len(shots) <= 12, '镜头数量超过 12 个')
    unique(shots, 'shot_id')
    event_ids = {e['event_id'] for e in facts['events']}
    asset_ids = {e['id'] for e in facts['entities']}
    cursor, covered, first = 0, set(), {}
    for shot in shots:
        start, end = shot['start_frame'], shot['end_frame']
        require(type(start) is int and type(end) is int and start == cursor and end > start,
                '镜头必须使用无空洞、无重叠的整数半开帧区间')
        cursor = end
        ids = set(items(shot['event_ids'], 'event_ids'))
        require(ids <= event_ids, '镜头引用未知事件')
        require(set(items(shot['asset_ids'], 'asset_ids')) <= asset_ids, '镜头引用未知资产')
        for eid in ids:
            first.setdefault(eid, start)
        covered |= ids
        for field in ('purpose', 'camera', 'entry_state', 'exit_state', 'action'):
            text(shot.get(field), field)
        requirements(shot['requirements'])
    require(cursor == total_frames, '分镜总帧数与目标不符')
    require(covered == event_ids, '分镜遗漏故事事件')
    require([first[e] for e in facts['event_order']] == sorted(first[e] for e in facts['event_order']),
            '分镜事件顺序倒置')
    require(facts['ending_event_id'] in shots[-1]['event_ids'], '最后镜头没有覆盖结局')
    return data


def frame_plan(data, shot):
    count = shot['end_frame'] - shot['start_frame']
    require(data['shot_id'] == shot['shot_id'], '帧计划镜头不匹配')
    keys = items(data['keyframes'], 'keyframes')
    require(len(keys) <= 16, '关键帧超过 16 张；请拆分复杂镜头')
    unique(keys, 'frame_id')
    indices = [key['frame_index'] for key in keys]
    for key in keys:
        index = key['frame_index']
        require(type(index) is int and 0 <= index < count,
                f'keyframes.{key["frame_id"]}.frame_index={index!r} 关键帧越界：'
                f'必须使用局部整数帧号 0～{count - 1}，不能使用全片帧号或结束边界 {count}')
    require(indices == sorted(set(indices)) and indices[0] == 0 and indices[-1] == count - 1,
            '关键帧需按帧号排列并包含首尾')
    for key in keys:
        text(key.get('state'), 'state')
        requirements(key.get('requirements'), f'keyframes.{key["frame_id"]}.requirements')
        require(set(items(key.get('covers'), 'covers', nonempty=False)) <=
                {r['id'] for r in shot['requirements']}, '关键帧覆盖了未知要求')
    require({r['id'] for r in shot['requirements'] if r['hard']} <=
            {rid for key in keys for rid in key['covers']}, '关键帧遗漏镜头硬性要求')
    for i in items(data['critical_frames'], 'critical_frames', nonempty=False):
        require(type(i) is int and 0 <= i < count, '动作接触或切换帧越界')
    ranges = items(data['intervals'], 'intervals')
    cursor = 0
    for interval in ranges:
        start, end = interval['start_frame'], interval['end_frame']
        require(type(start) is int and type(end) is int and start == cursor and end > start,
                '动作区间必须连续且不重叠')
        require(interval['method'] in {'motion', 'hold', 'cut'}, '动作区间方法无效')
        text(interval.get('description'), 'interval.description')
        cursor = end
    require(cursor == count, '动作区间没有覆盖整个镜头')
    for interval in items(data.get('allowed_black_ranges', []), 'allowed_black_ranges', nonempty=False):
        require(len(interval) == 2 and all(type(x) is int for x in interval)
                and 0 <= interval[0] < interval[1] <= count, '允许黑屏区间无效')
    return data


def review(report, target, version, reqs):
    require(report.get('target_id') == target and report.get('target_version') == version,
            '审核目标或版本不匹配')
    require(report.get('status') in STATES, '审核状态无效')
    checks = items(report.get('checks'), 'checks')
    require(len(checks) == len(reqs), '审核必须逐项覆盖原始要求')
    by_id = {check['requirement_id']: check for check in checks}
    require(set(by_id) == {row['id'] for row in reqs}, '审核项目缺失、重复或未知')
    for check in checks:
        require(check.get('result') in STATES, '检查结果无效')
        text(check.get('evidence'), 'evidence')
        region = check.get('region_xyxy')
        if region is not None:
            require(isinstance(region, list) and len(region) == 4 and
                    all(type(v) in (float, int) and 0 <= v <= 1 for v in region) and
                    region[0] < region[2] and region[1] < region[3], '修复区域坐标无效')
    passed = report['status'] == 'PASS' and all(
        by_id[row['id']]['result'] == 'PASS' if row['hard'] else
        by_id[row['id']]['result'] in {'PASS', 'NOT_APPLICABLE'} for row in reqs)
    return passed


def sample_indices(plan, count):
    result = {key['frame_index'] for key in plan['keyframes']}
    result.update([0, min(1, count - 1), max(0, count - 2), count - 1])
    for frame in plan['critical_frames']:
        result.update(range(max(0, frame - 2), min(count, frame + 3)))
    result.update(round(i * (count - 1) / 7) for i in range(8))
    for interval in plan['intervals']:
        start, last = interval['start_frame'], interval['end_frame'] - 1
        result.update([start, last])
        # 只验锚点和均匀时间点会漏掉两姿态之间的短暂重影/瞬移。
        fractions = (0.25, 0.5, 0.75) if interval['method'] == 'motion' else (0.5,)
        result.update(round(start + (last - start) * fraction) for fraction in fractions)
    return sorted(result)
