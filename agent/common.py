"""路径、日志和原子 JSON 写入。"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import threading
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 1280, 720

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def log(stage, message):
    print(f"[{time.strftime('%H:%M:%S')}] [{stage}] {message}", flush=True)


@contextmanager
def heartbeat(stage, status, interval=5):
    """即使连接或读取响应阻塞，也持续报告已经等待多久。"""
    started = time.monotonic()
    stopped = threading.Event()

    def report():
        while not stopped.wait(interval):
            log(stage, f"{status()}；已用时 {time.monotonic() - started:.1f}s")

    thread = threading.Thread(target=report, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()
