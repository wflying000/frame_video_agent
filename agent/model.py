"""兼容 Chat Completions 的流式 JSON 模型客户端。"""
import base64
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from dotenv import load_dotenv
from .common import PROJECT_ROOT, digest, log, heartbeat, write_json


def redact(text):
    for name in ('LLM_API_KEY', 'DEEPSEEK_API_KEY', 'DASHSCOPE_API_KEY'):
        key = os.getenv(name)
        if key:
            text = text.replace(key, '[REDACTED]')
    return text


class ResponseFormatError(RuntimeError):
    """响应协议/正文格式失败，不属于画面审核失败。"""
    def __init__(self, raw, stage='content', error=None, finish_reason=None):
        self.raw = redact(raw)
        position = error.pos if isinstance(error, json.JSONDecodeError) else 0
        self.details = {
            'stage': stage, 'finish_reason': finish_reason, 'characters': len(raw),
            'error': redact(str(error) if error else '正文为空或只包含空白'),
            'line': getattr(error, 'lineno', None),
            'column': getattr(error, 'colno', None), 'position': position,
            'excerpt': redact(raw[max(0, position - 120):position + 240]),
        }
        super().__init__(f"模型响应格式错误（{stage}）：{self.details['error']}")


def save_response(directory, raw, **metadata):
    if directory is None:
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'response.txt').write_text(redact(raw), encoding='utf-8')
    write_json(directory / 'response.json', metadata)


def retry_messages(messages, error):
    """保留原要求与图像，只补充序列化诊断；不修改审核结论或代码内容。"""
    result = deepcopy(messages)
    feedback = {
        'instruction': '上次响应未能解析。请根据原始任务重新输出完整、严格有效的 JSON 对象。'
            '所有属性名和字符串用双引号；禁止注释、尾逗号、单引号键名、Markdown 和字符串拼接。'
            'HTML/JavaScript 必须完整放入 JSON 字符串，正确转义双引号、反斜杠和换行。'
            '仅修正输出格式，保留原任务全部要求；审核仍按真实证据判断。',
        'format_error': error.details,
    }
    if error.details['stage'] == 'content' and len(error.raw) <= 24000:
        feedback['previous_response'] = error.raw
    result.append({'role': 'user', 'content': json.dumps(feedback, ensure_ascii=False)})
    return result


def recorded_messages(messages):
    recorded = deepcopy(messages)
    for message in recorded:
        if isinstance(message.get('content'), list):
            for part in message['content']:
                if part.get('type') == 'image_url' and 'url' in part['image_url']:
                    part['image_url'] = {'sha256': digest(part['image_url']['url'])}
    return recorded

def stream_events(response, check_progress=None):
    """读取 SSE，忽略心跳注释，合并同一事件的 data 行。"""
    lines = []
    for raw in response:
        if check_progress:
            check_progress()
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if lines:
                yield "\n".join(lines)
                lines = []
        elif line.startswith("data:"):
            lines.append(line[5:].lstrip(" "))
    if lines:
        yield "\n".join(lines)


def read_completion(response, state, show_output, idle_timeout=180):
    if "text/event-stream" not in response.headers.get("Content-Type", ""):
        state["status"] = "接收非流式响应，等待完整结果"
        raw = response.read().decode('utf-8')
        try:
            choice = json.loads(raw)["choices"][0]
        except json.JSONDecodeError as error:
            raise ResponseFormatError(raw, 'response_envelope', error) from error
        if show_output:
            print(choice["message"]["content"], flush=True)
        return choice

    parts, finish = [], None
    chars, reasoning_chars = 0, 0
    last_progress = time.monotonic()

    def check_progress():
        idle = time.monotonic() - last_progress
        if idle >= idle_timeout:
            raise TimeoutError(f'模型连续 {idle:.1f}s 无新增正文或思考输出（上限 {idle_timeout:g}s）；'
                               'SSE 保活和空事件不算有效输出')

    try:
        for event in stream_events(response, check_progress):
            if event == "[DONE]":
                break
            try:
                chunk = json.loads(event)
            except json.JSONDecodeError as error:
                raise ResponseFormatError(event, 'sse_event', error) from error
            if chunk.get("error"):
                raise RuntimeError(f"模型流式响应错误：{chunk['error']}")
            if not chunk.get("choices"):
                continue
            choice = chunk["choices"][0]
            delta = choice.get("delta", {})
            reasoning = delta.get("reasoning_content") or ""
            reasoning_chars += len(reasoning)
            text = delta.get("content") or ""
            if text or reasoning:
                last_progress = time.monotonic()
            if text:
                if not parts:
                    log("模型", "收到首段正文，开始接收模型输出")
                parts.append(text)
                chars += len(text)
                if show_output:
                    print(text, end="", flush=True)
            state["status"] = (f"接收输出中，正文 {chars} 字符，思考 {reasoning_chars} 字符"
                               if chars or reasoning_chars else "连接已建立，等待模型首段输出")
            state["reasoning_chars"] = reasoning_chars
            finish = choice.get("finish_reason") or finish
            if finish:
                break  # 已完成的正文不应继续等待服务端关闭 SSE 连接。
    finally:
        state['content'] = ''.join(parts)
        if show_output and parts:
            print(flush=True)
    if finish is None:
        raise RuntimeError("模型流式响应提前结束，未收到完成标记；请重试。")
    return {"finish_reason": finish, "message": {"content": "".join(parts)}}


def ask(messages, *, diagnostics=None):
    """一个兼容 Chat Completions 的多模态模型同时负责编写与看图。"""
    # 仅在调用模型时加载，避免渲染子进程重新从 .env 读取已移除的密钥。
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise ValueError("请在项目 .env 或环境变量中设置 LLM_API_KEY、DEEPSEEK_API_KEY 或 DASHSCOPE_API_KEY。")
    base = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    body = {"model": os.getenv("LLM_MODEL", "deepseek-flash"), "messages": messages,
            "temperature": 0.2, "max_tokens": 12000,
            "response_format": {"type": "json_object"}, "stream": True}
    body.update(json.loads(os.getenv("LLM_EXTRA_BODY", '{}')))
    idle_timeout = float(os.getenv('LLM_IDLE_TIMEOUT_SECONDS', '180'))
    if not math.isfinite(idle_timeout) or idle_timeout <= 0:
        raise ValueError('LLM_IDLE_TIMEOUT_SECONDS 必须为有限正数')
    request = Request(base.rstrip("/") + "/chat/completions",
                      json.dumps(body).encode(),
                      {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    started = time.monotonic()
    state = {"status": "等待模型服务响应（连接、排队或首段输出）"}
    log("模型", f"发送请求：model={body['model']}，消息 {len(messages)} 条，stream={body.get('stream', False)}")
    try:
        with heartbeat("模型", lambda: state["status"]):
            with urlopen(request, timeout=min(300, idle_timeout)) as response:
                log("模型", f"连接已建立，耗时 {time.monotonic() - started:.1f}s")
                state["status"] = "连接已建立，等待模型首段输出"
                choice = read_completion(response, state, os.getenv("LLM_SHOW_OUTPUT") == "1", idle_timeout)
    except HTTPError as error:
        # 响应正文用于诊断协议、模型权限和配额错误；不打印请求头或密钥。
        detail = error.read(4096).decode("utf-8", errors="replace").replace(key, "[REDACTED]")
        log("模型", f"HTTP {error.code}，耗时 {time.monotonic() - started:.1f}s：{detail}")
        save_response(diagnostics, detail, status='HTTP_ERROR', http_status=error.code)
        raise
    except ResponseFormatError as error:
        save_response(diagnostics, error.raw, status='FORMAT_ERROR', **error.details)
        log('模型', str(error))
        raise
    except TimeoutError as error:
        log('模型', f'输出等待超时，耗时 {time.monotonic() - started:.1f}s：{redact(str(error))}')
        save_response(diagnostics, state.get('content', ''), status='TIMEOUT', error=redact(str(error)),
                      reasoning_chars=state.get('reasoning_chars', 0), idle_timeout_seconds=idle_timeout)
        raise
    except Exception as error:
        log("模型", f"请求失败，耗时 {time.monotonic() - started:.1f}s：{str(error).replace(key, '[REDACTED]')}")
        save_response(diagnostics, state.get('content', ''), status='ERROR', error=redact(str(error)))
        raise
    content = choice["message"]["content"] or ""
    log("模型", f"响应结束：正文 {len(content)} 字符，耗时 {time.monotonic() - started:.1f}s，结束原因 {choice.get('finish_reason')}")
    if choice.get("finish_reason") == "length":
        detail = (f"模型输出被截断（max_tokens={body['max_tokens']}，正文 {len(content)} 字符，"
                  f"思考 {state.get('reasoning_chars', 0)} 字符）；可在 LLM_EXTRA_BODY 中提高 max_tokens，"
                  "或在服务支持时关闭思考模式。")
        log("错误", detail)
        save_response(diagnostics, content, status='TRUNCATED', finish_reason='length', error=detail)
        raise ValueError(detail)
    try:
        if not content.strip():
            raise ResponseFormatError(content, finish_reason=choice.get('finish_reason'))
        result = json.loads(content)
        if not isinstance(result, dict):
            raise ResponseFormatError(content, error=ValueError('顶层必须是 JSON 对象'),
                                      finish_reason=choice.get('finish_reason'))
    except json.JSONDecodeError as error:
        failure = ResponseFormatError(content, error=error, finish_reason=choice.get('finish_reason'))
        save_response(diagnostics, content, status='FORMAT_ERROR', **failure.details)
        raise failure from error
    except ResponseFormatError as error:
        save_response(diagnostics, content, status='FORMAT_ERROR', **error.details)
        raise
    save_response(diagnostics, content, status='PASS', finish_reason=choice.get('finish_reason'),
                  characters=len(content))
    log("模型", "JSON 解析成功")
    return result



def settings():
    load_dotenv(PROJECT_ROOT / '.env', override=False)
    return {'model': os.getenv('LLM_MODEL', 'deepseek-flash'),
            'base_url': os.getenv('LLM_BASE_URL', 'https://api.deepseek.com'),
            'extra_body': json.loads(os.getenv('LLM_EXTRA_BODY', '{}'))}

def messages(system, payload, images=()):
    content = [{'type': 'text', 'text': json.dumps(payload, ensure_ascii=False)}]
    for label, path in images:
        content.append({'type': 'text', 'text': label})
        encoded = base64.b64encode(Path(path).read_bytes()).decode()
        content.append({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + encoded}})
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}]
