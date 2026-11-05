"""故事 → 分步规划与审核 → 独立镜头 → 成片验收与断点恢复。"""
import argparse
import os
from pathlib import Path
import shutil
import sys

# 同时支持 python agent/video_agent.py 和 python -m agent。
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import rendering
from agent.workflow import generate
from agent.store import Blocked


def task_attempt_limit(value):
    task_id, separator, count = value.rpartition('=')
    if not separator or not task_id or not count.isdecimal() or int(count) < 1:
        raise argparse.ArgumentTypeError('格式为任务ID=总候选上限，例如 assets/L01=4。')
    return task_id, int(count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['generate', 'render'])
    parser.add_argument('input', type=Path, help='generate 读取故事 txt；render 读取已有 HTML')
    parser.add_argument('--out', type=Path, default=Path('output'))
    parser.add_argument('--seconds', type=int, default=20)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--resume', action='store_true', help='恢复同目录的新版 manifest，复用通过且版本匹配的任务')
    parser.add_argument('--max-calls', type=int, default=120, help='整个项目累计模型调用上限，恢复时不清零')
    parser.add_argument('--max-seconds', type=int, default=3600, help='项目累计活动耗时上限，恢复时不清零')
    parser.add_argument('--task-attempt-limit', type=task_attempt_limit, action='append', default=[],
                        metavar='TASK=COUNT', help='恢复时设置指定任务的总候选上限，可重复指定；历史候选保留')
    parser.add_argument('--preview', action='store_true', help='render 仅生成采样预览')
    parser.add_argument('--show-model-output', action='store_true', help='实时打印完整模型 JSON/代码')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--frames', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--samples', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.show_model_output:
        os.environ['LLM_SHOW_OUTPUT'] = '1'
    if args.mode == 'generate' and (args.preview or args.worker or args.frames is not None or args.samples):
        parser.error('--preview/--worker/--frames/--samples 仅用于 render 模式。')
    if args.mode == 'render' and args.resume:
        parser.error('--resume 仅用于 generate 模式。')
    if args.task_attempt_limit and (args.mode != 'generate' or not args.resume):
        parser.error('--task-attempt-limit 必须用于 generate --resume。')
    if not 2 <= args.seconds <= 120 or not 1 <= args.fps <= 60:
        parser.error('seconds 需为 2～120，fps 需为 1～60。')
    if args.max_calls < 1 or args.max_seconds < 1:
        parser.error('预算必须大于 0。')
    if args.frames is not None and not 1 <= args.frames <= 7200:
        parser.error('frames 需为 1～7200。')
    if not args.preview and (not shutil.which('ffmpeg') or not shutil.which('ffprobe')):
        parser.error('请安装 FFmpeg 和 ffprobe 并加入 PATH。')
    source = args.input.resolve()
    text = source.read_text(encoding='utf-8-sig').strip()
    if not text:
        parser.error('输入文件不能为空。')
    out = args.out.resolve()
    if args.mode == 'generate':
        try:
            print(generate(text, out, args.seconds, args.fps, args.resume, args.max_calls, args.max_seconds,
                           dict(args.task_attempt_limit)))
        except Blocked as error:
            parser.exit(1, f'生成暂停：{error}\n')
        return
    if not args.worker and out.exists() and any(out.iterdir()):
        parser.error('输出目录非空，请换一个 --out。')
    out.mkdir(parents=True, exist_ok=True)
    indices = [int(item) for item in args.samples.split(',')] if args.samples else None
    worker = rendering.render if args.worker else rendering.render_job
    worker(source, out, args.seconds, args.fps, args.preview, args.frames, indices)
    if not args.worker:
        print(out if args.preview else out / 'video.mp4')


if __name__ == '__main__':
    main()
