"""单进程任务调度：原子检查点、内容哈希、依赖失效及持久化预算。"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.error import HTTPError, URLError

from . import model, prompts
from .common import digest, file_hash, log, read_json, write_json
from .rendering import RenderError


class Blocked(RuntimeError):
    pass


class Rejected(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__(json.dumps(report, ensure_ascii=False))


class Project:
    def __init__(self, out, config, resume=False, max_calls=120, max_seconds=3600,
                 task_attempt_limits=None):
        self.root = Path(out).resolve()
        self.config = config
        self.resume = resume
        self.max_calls, self.max_seconds = max_calls, max_seconds
        self.task_attempt_limits = dict(task_attempt_limits or {})
        if self.task_attempt_limits and not resume:
            raise ValueError('指定任务候选上限必须同时使用 --resume。')
        if any(type(limit) is not int or limit < 1 for limit in self.task_attempt_limits.values()):
            raise ValueError('任务候选上限必须是正整数。')
        self.clock = time.monotonic()
        self.data = None

    def __enter__(self):
        if self.root.exists() and any(self.root.iterdir()) and not self.resume:
            raise ValueError('输出目录非空；使用新的 --out，或用 --resume 恢复此项目。')
        self.root.mkdir(parents=True, exist_ok=True)
        lock = self.root / '.run.lock'
        if lock.exists():
            try:
                pid = int(lock.read_text())
                os.kill(pid, 0)
            except ProcessLookupError:
                lock.unlink()
            except (ValueError, PermissionError):
                raise Blocked('无法确认项目锁的持有者；请检查 .run.lock。')
            else:
                raise Blocked(f'项目正由进程 {pid} 使用，不能同时写入。')
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'w') as stream:
            stream.write(str(os.getpid()))
        try:
            manifest = self.root / 'manifest.json'
            if self.resume:
                if not manifest.exists():
                    raise ValueError('目录没有新版 manifest.json；旧版结果请保留，使用新的 --out。')
                self.data = read_json(manifest)
                if self.data.get('schema_version') != 1 or self.data.get('config_hash') != digest(self.config):
                    raise ValueError('恢复参数或故事已改变，请使用新输出目录。')
                unknown = self.task_attempt_limits.keys() - self.data['tasks'].keys()
                if unknown:
                    raise ValueError('指定的任务不存在：' + ', '.join(sorted(unknown)))
                self.task_attempt_limits = {
                    **self.data.get('limits', {}).get('task_attempt_limits', {}),
                    **self.task_attempt_limits}
                self.refresh()
            else:
                self.data = {'schema_version': 1, 'config_hash': digest(self.config),
                             'status': 'RUNNING', 'tasks': {}, 'calls': 0, 'active_seconds': 0,
                             'repair_requests': {}, 'global_repairs': 0}
                write_json(self.root / 'project_config.json', self.config)
                (self.root / 'story.txt').write_text(self.config['story'], encoding='utf-8')
            self.data['limits'] = {'max_calls': self.max_calls, 'max_seconds': self.max_seconds,
                                   'task_attempt_limits': self.task_attempt_limits}
            self.data['status'] = 'RUNNING'
            self.checkpoint()
            return self
        except BaseException:
            lock.unlink(missing_ok=True)
            raise

    def __exit__(self, kind, error, traceback):
        try:
            if error is not None:
                self.data['status'] = 'BLOCKED'
                self.data['error'] = str(error)
            self.checkpoint()
        finally:
            (self.root / '.run.lock').unlink(missing_ok=True)

    def checkpoint(self):
        now = time.monotonic()
        self.data['active_seconds'] += now - self.clock
        self.clock = now
        write_json(self.root / 'manifest.json', self.data)

    def budget(self):
        self.check_time()
        if self.data['calls'] >= self.max_calls:
            raise Blocked(f"项目模型调用预算已用尽：{self.data['calls']}/{self.max_calls}；可调整预算后 --resume。")

    def check_time(self):
        elapsed = self.data['active_seconds'] + time.monotonic() - self.clock
        if elapsed >= self.max_seconds:
            raise Blocked(f'项目活动耗时预算已用尽：{elapsed:.1f}/{self.max_seconds}s；可调整预算后 --resume。')

    def artifacts_valid(self, node):
        return bool(node.get('artifacts')) and all(
            (self.root / path).is_file() and file_hash(self.root / path) == expected
            for path, expected in node['artifacts'].items())

    def refresh(self):
        for task_id, node in list(self.data['tasks'].items()):
            if node['status'] == 'PASS' and not self.artifacts_valid(node):
                self.invalidate(task_id)

    def invalidate(self, task_id):
        affected = {task_id}
        while True:
            more = {key for key, node in self.data['tasks'].items()
                    if affected.intersection(node.get('depends_on', []))}
            if more <= affected:
                break
            affected |= more
        for key in affected:
            if key in self.data['tasks']:
                self.data['tasks'][key]['status'] = 'STALE'
        final = self.root / 'final.mp4'
        if final.exists():
            previous = self.root / 'exports' / f'previous_final_{time.time_ns()}.mp4'
            previous.parent.mkdir(exist_ok=True)
            final.replace(previous)
        self.data['status'] = 'RUNNING'
        log('依赖', '需要重建或复验：' + ', '.join(sorted(affected)))

    def call(self, directory, name, messages):
        """候选内的每次调用可独立恢复，不因审核网络错误重新生成代码。"""
        path = directory / f'{name}.json'
        signature = digest({'messages': messages, 'settings': model.settings()})
        cache = directory / f'{name}.call.json'
        if path.exists() and cache.exists():
            cached = read_json(cache)
            if cached.get('status') == 'PASS' and cached.get('input_hash') == signature:
                return read_json(path)
        request_messages = messages
        for retry in range(2):
            self.budget()
            self.data['calls'] += 1
            # 保存提示词以便追踪；图片存哈希，真实图片与版本路径在任务产物中保留。
            recorded = model.recorded_messages(request_messages)
            write_json(directory / f'{name}.request.json', recorded)
            trace = directory / 'calls' / name / f'{self.data["calls"]:06d}'
            write_json(trace / 'request.json', recorded)
            write_json(cache, {'input_hash': signature, 'model': model.settings(),
                              'call_number': self.data['calls'], 'status': 'RUNNING'})
            self.checkpoint()
            try:
                value = model.ask(request_messages, diagnostics=trace)
                write_json(path, value)
                write_json(cache, {'input_hash': signature, 'model': model.settings(),
                                  'call_number': self.data['calls'], 'status': 'PASS'})
                return value
            except model.ResponseFormatError as error:
                report = {'status': 'FORMAT_ERROR', 'kind': 'model_response_format',
                          'error': str(error), 'diagnostics': str(trace.relative_to(self.root)),
                          **error.details}
                write_json(cache, {'input_hash': signature, 'model': model.settings(),
                                  'call_number': self.data['calls'], **report})
                log('格式', f'{name}：{error}；原始响应与位置见 {trace}')
                if retry:
                    raise Blocked(f'{name} 响应格式连续失败，已保留当前候选；见 {trace}。'
                                  '排查后可 --resume，已消耗调用仍计入项目总预算。') from error
                request_messages = model.retry_messages(messages, error)
                log('重试', f'{name} 附带格式诊断，在同一候选内有限重试一次')
            except (URLError, TimeoutError, ConnectionError) as error:
                transient = not isinstance(error, HTTPError) or error.code == 429 or error.code >= 500
                if retry and isinstance(error, TimeoutError):
                    raise Blocked(f'{name} 模型输出连续超时，已保留当前候选；诊断见 {trace}。'
                                  '服务恢复后可 --resume，候选次数不增加，已消耗调用仍计入总预算。') from error
                if retry and isinstance(error, HTTPError) and transient:
                    raise Blocked(f'{name} 模型服务连续不可用（HTTP {error.code}），已保留当前候选；'
                                  f'诊断见 {trace}。服务恢复后可 --resume，已完成的调用可复用。') from error
                if not transient or retry:
                    raise
                log('重试', f'{name} 网络错误，有限重试一次：{type(error).__name__}')
                if isinstance(error, HTTPError):
                    hint = (error.headers or {}).get('Retry-After', '')
                    delay = max(5, int(hint)) if hint.isdecimal() else 5
                    if delay > 30:
                        raise Blocked(f'{name} 模型服务要求等待 {delay}s，已保留当前候选；'
                                      f'请稍后 --resume。诊断见 {trace}。') from error
                    remaining = self.max_seconds - self.data['active_seconds'] - (time.monotonic() - self.clock)
                    if delay >= remaining:
                        raise Blocked(f'{name} 剩余时间预算不足以等待服务重试，已保留当前候选；'
                                      '调整预算后可 --resume。') from error
                    log('等待', f'模型服务 HTTP {error.code}，{delay}s 后重试')
                    time.sleep(delay)

    def task(self, task_id, inputs, deps, action, max_attempts=3, retry=True):
        max_attempts = self.task_attempt_limits.get(task_id, max_attempts)
        self.refresh()
        for dep in deps:
            if self.data['tasks'].get(dep, {}).get('status') != 'PASS':
                raise Blocked(f'{task_id} 的依赖 {dep} 未通过')
        versions = {dep: {'version': self.data['tasks'][dep]['version'],
                          'output_hash': self.data['tasks'][dep]['output_hash']} for dep in deps}
        signature = digest({'inputs': inputs, 'dependencies': versions, 'model': model.settings(),
                            'prompts': {key: value for key, value in vars(prompts).items() if key.isupper()}})
        node = self.data['tasks'].setdefault(task_id, {'status': 'PENDING', 'attempts': [],
                                                     'depends_on': deps})
        if node['status'] == 'PASS' and node.get('input_hash') == signature and self.artifacts_valid(node):
            log('复用', f'{task_id} v{node["version"]}')
            return read_json(self.root / node['result_path'])
        if node.get('input_hash') != signature and node['status'] == 'PASS':
            self.invalidate(task_id)
        node['depends_on'] = deps
        # 输入改变不重置累计修复次数，避免通过换版本绕过预算。
        while True:
            previous = node['attempts'][-1] if node['attempts'] else None
            continuing = (previous is not None and previous['status'] in {'RUNNING', 'RETRY'}
                          and previous['input_hash'] == signature)
            if not continuing:
                if len(node['attempts']) >= max_attempts:
                    node['status'] = 'BLOCKED'
                    self.checkpoint()
                    raise Blocked(f'{task_id} 已达到 {max_attempts} 个候选版本上限；见 manifest 和审核报告。'
                                  f'修复原因后可用 --resume --task-attempt-limit {task_id}={max_attempts + 1}'
                                  ' 增加一个候选；历史次数和项目预算不清零。')
                version = len(node['attempts']) + 1
                candidate = {'version': version, 'input_hash': signature, 'status': 'RUNNING',
                             'path': f'{task_id}/v{version:03d}'}
                node['attempts'].append(candidate)
            else:
                candidate = previous
                previous = node['attempts'][-2] if len(node['attempts']) > 1 else None
            directory = self.root / candidate['path']
            directory.mkdir(parents=True, exist_ok=True)
            node.update(status='RUNNING', input_hash=signature, input_versions=versions)
            candidate['status'] = 'RUNNING'
            self.checkpoint()
            log('任务', f'{task_id} v{candidate["version"]}')
            try:
                self.check_time()
                value = action(directory, candidate['version'], previous)
                self.check_time()
                # 单写者锁与再次核验绑定到同一版本，阻止旧输入产生的通过结果提交。
                if any(self.data['tasks'][dep]['status'] != 'PASS' or
                       self.data['tasks'][dep]['version'] != versions[dep]['version'] or
                       not self.artifacts_valid(self.data['tasks'][dep]) for dep in deps):
                    raise Blocked('任务执行期间输入版本变化，结果不能提交。')
                write_json(directory / 'result.json', value)
                artifacts = {str(path.relative_to(self.root)): file_hash(path)
                             for path in sorted(directory.rglob('*')) if path.is_file()}
                candidate['status'] = 'PASS'
                node.update(status='PASS', version=candidate['version'], artifacts=artifacts,
                            output_hash=digest(artifacts), result_path=str((directory / 'result.json').relative_to(self.root)))
                self.checkpoint()
                return value
            except (Rejected, ValueError, KeyError, TypeError, RenderError,
                    subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                report = error.report if isinstance(error, Rejected) else {
                    'status': 'FAIL', 'error': str(error), 'kind': 'structure_or_technical'}
                write_json(directory / 'failure.json', report)
                candidate.update(status='FAIL', report=report)
                node['status'] = report['status'] if report.get('status') in {'FAIL', 'UNCERTAIN'} else 'FAIL'
                self.checkpoint()
                log('修复', f'{task_id} 未通过，候选已保留')
                log('审核', json.dumps(report, ensure_ascii=False)[:1600])
                if not retry:
                    raise Rejected(report)
            except BaseException:
                candidate['status'] = 'RETRY'
                node['status'] = 'PENDING'
                self.checkpoint()
                raise

    def publish(self, candidate):
        self.check_time()
        self.refresh()
        if any(node['status'] != 'PASS' for node in self.data['tasks'].values()):
            raise Blocked('存在未通过或失效的必需任务，不能发布 final.mp4。')
        import shutil
        temporary = self.root / 'final.part.mp4'
        shutil.copyfile(candidate, temporary)
        temporary.replace(self.root / 'final.mp4')
        self.data['status'] = 'COMPLETED'
        self.data.pop('error', None)
        self.checkpoint()
        return self.root / 'final.mp4'
