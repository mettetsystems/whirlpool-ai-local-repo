"""Isolate Hub transfers so pausing terminates workers without losing cached chunks.

Requests (including tokens) travel over stdin, never command-line arguments or files.
Only structured progress and sanitized errors are sent to the desktop over stdout.
"""
import importlib
import io
import itertools
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


class DownloadPaused(Exception):
    pass


def run_download(request, progress_callback, stop_event=None):
    process = subprocess.Popen([sys.executable, '-m', 'src.model_repo.download_process'],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, text=True,
                               cwd=str(Path(__file__).resolve().parents[2]))
    messages = queue.Queue()
    def read_output():
        for line in process.stdout:
            messages.put(line)
        messages.put(None)
    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    try:
        process.stdin.write(json.dumps(request))
        process.stdin.close()
        done = False
        error = None
        while True:
            if stop_event is not None and stop_event.is_set():
                raise DownloadPaused('Paused; downloaded chunks have been retained')
            try:
                line = messages.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                break
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event['type'] == 'progress':
                progress_callback(event)
            elif event['type'] == 'done':
                done = True
            elif event['type'] == 'error':
                error = event['message']
        code = process.wait()
        if code or not done:
            raise RuntimeError(error or f'Download worker exited unexpectedly ({code}); retry to resume')
        return request['local_dir']
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        reader.join(timeout=2)
        process.stdout.close()
        if not process.stdin.closed:
            process.stdin.close()


def child_main():
    from huggingface_hub import snapshot_download
    progress_module = importlib.import_module('huggingface_hub.utils.tqdm')
    from tqdm.auto import tqdm as original_tqdm
    request = json.load(sys.stdin)
    mutex = threading.RLock()
    counters = {}
    keys = itertools.count()
    files_done = 0
    baseline = request.get('completed_bytes', 0)
    start = time.monotonic()
    last_emit = 0.0

    def send(value):
        print(json.dumps(value), flush=True)

    class Progress(original_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs['file'] = io.StringIO()
            kwargs['disable'] = False
            kwargs.pop('name', None)
            self._bytes = kwargs.get('unit') == 'B'
            self._key = next(keys)
            self._initial = kwargs.get("initial", 0)
            self._reported = self._initial
            self._total_bytes = kwargs.get('total') or 0
            self._label = str(kwargs.get('desc', 'Downloading'))
            super().__init__(*args, **kwargs)
            self.report(force=True)

        def update(self, count=1):
            self._reported += count
            result = super().update(count)
            self.report()
            return result

        def report(self, force=False):
            nonlocal last_emit, files_done
            with mutex:
                if self._bytes:
                    counters[self._key] = (min(self._reported, self._total_bytes), self._initial)
                else:
                    files_done = int(self._reported)
                now = time.monotonic()
                completed = min(request['total_bytes'], baseline + sum(value[0] for value in counters.values()))
                if not force and now - last_emit < 0.2:
                    return
                last_emit = now
                speed = sum(max(0, value[0] - value[1]) for value in counters.values()) / max(now - start, 0.001)
                send({'type':'progress', 'bytes':completed, 'total':request['total_bytes'],
                      'files_done':files_done, 'files_total':request['files_total'],
                      'file':self._label if self._bytes else 'Checking / fetching files',
                      'speed':speed, 'eta':(request['total_bytes']-completed)/speed if speed else None})

        def close(self):
            if hasattr(self, '_reported'):
                self.report(force=True)
            super().close()

    # Scoped to this disposable process; other downloads and the GUI are unaffected.
    progress_module.tqdm = Progress
    try:
        snapshot_download(repo_id=request['model_id'], local_dir=request['local_dir'],
                          token=request['token'], revision=request['revision'],
                          tqdm_class=Progress, **request.get('options', {}))
        send({'type':'done'})
    except Exception as exc:
        message = str(exc)
        if request.get('token'):
            message = message.replace(request['token'], '[redacted]')
        send({'type':'error', 'message':message})
        raise SystemExit(1)


if __name__ == '__main__':
    child_main()
