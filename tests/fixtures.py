"""可预测的模型替身，审核读取真实图片输入；不使用远程服务。"""
import copy
import json
from urllib.error import URLError
from agent import prompts

STORY = '红球向右移动，然后停住。'
FACTS = {'entities': [{'id': 'P01', 'kind': 'prop', 'description': '红球'}],
         'events': [{'event_id': 'E01', 'source_text': '红球向右移动', 'entity_ids': ['P01'],
                     'action': '移动', 'precondition': '左侧', 'result': '右侧', 'must_show': '球向右移动'},
                    {'event_id': 'E02', 'source_text': '然后停住', 'entity_ids': ['P01'],
                     'action': '停止', 'precondition': '移动中', 'result': '停止', 'must_show': '球停住'}],
         'event_order': ['E01', 'E02'], 'causal_links': [['E01', 'E02']],
         'ending_event_id': 'E02', 'design_choices': ['纯色背景']}
REQ = [{'id': 'R01', 'hard': True, 'text': '球的状态与要求一致'}]
VISUAL = {'style': '简洁二维', 'assets': [{'asset_id': 'P01', 'fixed': '红色圆球',
                                      'variable': '位置', 'requirements': REQ}]}
BOARD = {'shots': [{'shot_id': f'S0{i+1}', 'event_ids': [f'E0{i+1}'], 'asset_ids': ['P01'],
                   'start_frame': i*4, 'end_frame': (i+1)*4, 'purpose': '球的动作', 'camera': '固定',
                   'entry_state': '左侧' if i == 0 else '右侧', 'exit_state': '右侧',
                   'action': '向右移动' if i == 0 else '停止', 'requirements': REQ} for i in range(2)]}
HTML = '''<!doctype html><canvas id="stage" width="1280" height="720"></canvas>
<script>const c=document.getElementById('stage'),ctx=c.getContext('2d');
window.film={duration:2,frame(t){ctx.resetTransform();ctx.fillStyle='#102030';ctx.fillRect(0,0,1280,720);
ctx.fillStyle='#ffd166';ctx.beginPath();ctx.arc(100+t*100,360,50,0,Math.PI*2);ctx.fill();}};</script>'''


def frames(shot_id):
    return {'shot_id': shot_id,
            'keyframes': [{'frame_id': f'K0{i+1}', 'frame_index': frame, 'state': '球可见',
                           'covers': ['R01'], 'requirements': REQ} for i, frame in enumerate([0, 3])],
            'critical_frames': [1], 'intervals': [{'start_frame': 0, 'end_frame': 4,
                                                'method': 'motion', 'description': '移动或停留'}],
            'allowed_black_ranges': []}


class FakeModel:
    def __init__(self, fail_key=False, fail_final=False, network=False, uncertain=False):
        self.calls = []
        self.fail_key, self.fail_final, self.network, self.uncertain = fail_key, fail_final, network, uncertain
        self.network_failures = 0

    def __call__(self, messages, **kwargs):
        system = messages[0]['content']
        content = messages[1]['content']
        payload = json.loads(content[0]['text'])
        self.calls.append((system, copy.deepcopy(payload)))
        if system == prompts.STORY:
            return copy.deepcopy(FACTS)
        if system == prompts.VISUAL:
            return copy.deepcopy(VISUAL)
        if system == prompts.STORYBOARD:
            return copy.deepcopy(BOARD)
        if system == prompts.FRAME_PLAN:
            return frames(payload['shot']['shot_id'])
        if system == prompts.CANVAS:
            return {'html': HTML}
        if system == prompts.PATCH:
            old = "'#ffd166'" if "'#ffd166'" in payload['html'] else "'#ffaa44'"
            new = "'#ffaa44'" if old == "'#ffd166'" else "'#ffd166'"
            return {'patches': [{'old': old, 'new': new}]}
        if system == prompts.OBSERVE:
            assert any(row['type'] == 'image_url' for row in content)
            assert 'story' not in payload
            return {'observations': [{'frame_index': i, 'description': '实际图像中有球'} for i in payload['frame_indices']]}
        assert system == prompts.REVIEW
        target, version = payload['target_id'], payload['target_version']
        if self.network and target == 'S02_K01' and self.network_failures < 2:
            self.network_failures += 1
            raise URLError('simulated network outage')
        failed = ((self.fail_key and target == 'S02_K01' and version == 1) or
                  (self.fail_final and target == 'E02' and version == 1))
        status = 'UNCERTAIN' if self.uncertain else ('FAIL' if failed else 'PASS')
        indices = []
        for row in content[1:]:
            if row['type'] == 'text' and row['text'].startswith('实际画面 frame_index='):
                indices.append(int(row['text'].split('=')[1]))
        report = {'target_id': target, 'target_version': version, 'status': status,
                  'checks': [{'requirement_id': r['id'], 'result': status, 'evidence': '实际画面证据',
                              'evidence_frames': indices[:1]} for r in payload['requirements']],
                  'repair': {'scope': 'shot', 'shot_ids': ['S02'], 'instruction': '修正球的位置'}}
        return report
