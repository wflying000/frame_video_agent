"""语义一致性容差与技术渲染模式的稳定配置接口。"""

REVIEW_LEVELS = ('strict', 'balanced', 'relaxed')
RENDER_MODES = ('exact', 'tolerant')


def affects_review(task_id):
    """仅场景执行与成片验收受语义严格度影响，规划和资产参考保持原约束。"""
    return isinstance(task_id, str) and (task_id.startswith('segments/') or
                                        task_id in ('checks/timeline', 'exports/final'))


def review_policy(name):
    """返回独立的新字典，作为生成、审核和缓存共同记录的语义策略。"""
    if name not in REVIEW_LEVELS:
        raise ValueError('review_strictness 必须是 strict、balanced 或 relaxed')
    tolerance = {
        'strict': '保持原有严格程度：逐项核对已审核计划的姿态、位置、比例、时间、运动相位、接触及光影连续性。'
            '不因偏差较小自动豁免；保留已有机位、采样范围及转向极值解释，不新增原要求之外的动作。',
        'balanced': '容忍不影响剧情理解、身份和关键连接的轻微姿态、位置、比例、时间、相位及局部光影偏差。'
            '自动计划中的非关键屏幕方向或精确运动曲线不能凌驾于原始故事可读性与自然动作。'
            '例如侧视鱼已有清楚、连贯的尾部变化和推进时，不额外要求用户未指定的完美屏幕水平左右摆动。'
            '微小细节不确定且不足以证明实质违反、核心要求已有实际证据支持时，可判PASS并解释适用容差；'
            '不能仅为细小非关键误差判FAIL或UNCERTAIN。',
        'relaxed': '主要核对剧情可读、必要动作和结果明确、角色身份稳定且没有明显异常。'
            '容忍不改变这些事实的非关键姿态、位置、比例、时间、相位、精确运动曲线及局部光影偏差，'
            '不以逐帧精确复刻自动计划或非关键屏幕方向作为通过条件。'
            '例如侧视鱼有可读、连贯的尾部变化与推进即可证明相应游动表现，不追加用户未要求的完美水平摆动。'
            '不确定的微小细节不足以说明实质违反且核心要求已有证据时，可判PASS并说明容差；'
            '不把非关键精度不足当成剧情失败。',
    }[name]
    return {
        'name': name, 'version': 'consistency_v1',
        'scope': 'scene_pixel_consistency',
        'instruction': '本策略定义一致性检查的语义容差，按所选容差逐项判断原required_check_ids；'
            '不得删除、增加或跳过检查项，也不根据策略名称自动判PASS。' + tolerance +
            '所有档位均保留原文和用户明确要求的事件、因果顺序、结果、角色身份、关键动作与连接关系。'
            '明显断裂、消失、身份改变、无故瞬移或关键动作缺失必须FAIL；关键动作或结果无法确认时用UNCERTAIN。'
            '整体刚性平移不能代替原文明确要求的部件动作，不能用计划文字或代码声称代替实际像素证据。'
            '仍须有可感知的体积和空间纵深，保持手绘/插画质感，禁止扁平纸片效果或3D建模/CG替代。'
            '本策略不改变实际帧证据绑定、检查项完整性、技术编码检查或渲染确定性要求。'}
