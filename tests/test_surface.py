#!/usr/bin/env python3
"""Drive the public shim with a STUB worker.

Proves the behaviours the two-process design adds, without a checkpoint or a
GPU: nothing started at boot, one spawn per concurrent burst, the UDS proxy,
the bounded free-VRAM gate (refuse AND pass), a failed worker retried by the
next request, and the idle kill followed by a reload.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

BIN = os.path.expanduser("~/.local/bin")
SHIM = os.path.join(BIN, "qwen3-tts-server.py")
sys.path.insert(0, BIN)

failures = []


def check(name, condition, detail=""):
    if condition:
        print("PASS " + name)
    else:
        print("FAIL " + name + " :: " + repr(detail))
        failures.append(name)


def request(base, path, body=None, timeout=120):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers.get("Content-Type", "")


def as_json(raw):
    try:
        return json.loads(raw)
    except Exception:
        return {}


def wait_http(base, path, deadline_s, want=200):
    deadline = time.monotonic() + deadline_s
    status = None
    while time.monotonic() < deadline:
        try:
            status, payload, _ = request(base, path, timeout=5)
            if status == want:
                return status, payload
        except Exception:
            pass
        time.sleep(0.05)
    return status, b""


STUB = r'''#!/usr/bin/env python3
import json, os, socketserver
from http.server import BaseHTTPRequestHandler

SOCKET = os.environ["QWEN3_TTS_WORKER_SOCKET"]
READY_FD = os.environ.get("QWEN3_TTS_READY_FD")
COUNT = os.environ.get("STUB_SPAWN_COUNT")
if COUNT:
    with open(COUNT, "a") as fh:
        fh.write("%d\n" % os.getpid())


class H(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/voices":
            body = {"voices": [{"id": "vivian", "name": "vivian"}], "source": "model"}
        else:
            body = {"loaded": True}
        self._send(200, json.dumps(body).encode())

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self._send(200, b"RIFF" + b"stub-audio", "audio/wav")


class S(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


try:
    os.unlink(SOCKET)
except FileNotFoundError:
    pass
server = S(SOCKET, H)
if READY_FD:
    os.write(int(READY_FD), b"ready\n")
    os.close(int(READY_FD))
server.serve_forever()
'''


class Shim:
    def __init__(self, workdir, port, extra=None):
        self.worker = os.path.join(workdir, "stub-worker-%d.py" % port)
        with open(self.worker, "w") as fh:
            fh.write(STUB)
        os.chmod(self.worker, 0o755)
        self.socket = os.path.join(workdir, "worker-%d.sock" % port)
        self.count = os.path.join(workdir, "spawns-%d.txt" % port)
        self.log_path = os.path.join(workdir, "shim-%d.log" % port)
        env = dict(os.environ)
        env.update(
            {
                "QWEN3_TTS_PORT": str(port),
                "QWEN3_TTS_DEVICE": "cpu",
                "QWEN3_TTS_GPU_MIN_FREE_MIB": "0",
                "QWEN3_TTS_WORKER": self.worker,
                "QWEN3_TTS_WORKER_SOCKET": self.socket,
                "QWEN3_TTS_IDLE_UNLOAD_S": "2",
                "QWEN3_TTS_WORKER_STOP_S": "5",
                "QWEN3_TTS_LOAD_TIMEOUT_S": "15",
                "STUB_SPAWN_COUNT": self.count,
                "PYTHONUNBUFFERED": "1",
            }
        )
        env.update(extra or {})
        self.log = open(self.log_path, "w+b")
        self.proc = subprocess.Popen(
            [sys.executable, SHIM],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.base = "http://127.0.0.1:%d" % port

    def spawns(self):
        try:
            with open(self.count) as fh:
                return len([line for line in fh if line.strip()])
        except FileNotFoundError:
            return 0

    def tail(self, lines=12):
        try:
            with open(self.log_path, "rb") as fh:
                text = fh.read().decode("utf-8", "replace").strip().splitlines()
            return "\n".join(text[-lines:])
        except Exception:
            return ""

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
            self.proc.wait()
        self.log.close()


def main():
    workdir = tempfile.mkdtemp(prefix="qwen3-tts-surface-")
    shim = Shim(workdir, 18081)
    try:
        status, payload = wait_http(shim.base, "/v1/models", 60)
        check("the public API answers before any worker exists", status == 200, status)
        check("the served model id is named", as_json(payload).get("data", [{}])[0].get("id"), payload)

        status, raw, _ = request(shim.base, "/health", timeout=10)
        health = as_json(raw)
        check("cold start is /health 503 unloaded — the model is LAZY", status == 503 and health.get("status") == "unloaded", (status, health))
        check("no worker was spawned at start", shim.spawns() == 0, shim.spawns())

        status, raw, _ = request(shim.base, "/v1/audio/voices", timeout=10)
        voices = as_json(raw)
        check("the preset fallback list answers before the checkpoint loads", status == 200 and len(voices.get("voices", [])) == 9 and voices.get("source") == "fallback", voices)

        status, raw, _ = request(shim.base, "/v1/audio/speech", {"input": "x", "response_format": "opus"})
        body = as_json(raw)
        check("an unsupported format is a 4xx naming the supported set", 400 <= status < 500 and body.get("error", {}).get("code") == "unsupported_response_format", (status, body))

        status, raw, _ = request(shim.base, "/v1/audio/speech", {"input": "x", "model": "foreign"})
        body = as_json(raw)
        check("a foreign model is a 4xx naming the served one", 400 <= status < 500 and body.get("error", {}).get("code") == "model_not_served" and body.get("served_model"), (status, body))

        answers = []

        def fire():
            try:
                answers.append(request(shim.base, "/v1/audio/speech", {"input": "burst", "voice": "vivian"}, timeout=120)[0])
            except Exception as exc:
                answers.append(repr(exc)[:120])

        threads = [threading.Thread(target=fire) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        check("a concurrent burst all answers 200", answers == [200] * 6, answers)
        check("the burst spawned exactly ONE worker", shim.spawns() == 1, shim.spawns())

        status, raw, _ = request(shim.base, "/health", timeout=10)
        health = as_json(raw)
        check("/health is 200 ok once the worker serves", status == 200 and health.get("status") == "ok", (status, health))
        worker_pid = health.get("worker_pid")
        check("/health names the worker pid", isinstance(worker_pid, int) and worker_pid > 0, worker_pid)

        deadline = time.monotonic() + 30
        idle_state = {}
        while time.monotonic() < deadline:
            idle_state = as_json(request(shim.base, "/health", timeout=5)[1])
            if idle_state.get("loaded") is False:
                break
            time.sleep(0.2)
        check("after the idle TTL the worker is released (loaded=false)", idle_state.get("loaded") is False, idle_state)
        time.sleep(0.5)
        gone = False
        try:
            os.kill(worker_pid, 0)
        except ProcessLookupError:
            gone = True
        except Exception:
            gone = False
        check("the idle release KILLED the worker process (all VRAM released)", gone, worker_pid)

        status, raw, ctype = request(shim.base, "/v1/audio/speech", {"input": "reload", "voice": "vivian"}, timeout=120)
        check("the next request reloads and answers 200 audio", status == 200 and raw[:4] == b"RIFF" and ctype.startswith("audio/"), (status, raw[:8], ctype))
        check("the reload spawned a second worker", shim.spawns() == 2, shim.spawns())
    finally:
        print("--- shim log tail ---")
        print(shim.tail())
        shim.stop()

    # The gate REFUSES: an impossible floor with a bounded wait.
    low = os.path.join(workdir, "nvidia-smi-low")
    with open(low, "w") as fh:
        fh.write("#!/bin/sh\necho 1024\n")
    os.chmod(low, 0o755)
    gate = Shim(
        workdir,
        18082,
        {
            "QWEN3_TTS_DEVICE": "cuda:0",
            "QWEN3_TTS_GPU_MIN_FREE_MIB": "999999",
            "QWEN3_TTS_GPU_WAIT_SECONDS": "1",
            "QWEN3_TTS_GPU_POLL_S": "0.2",
            "QWEN3_TTS_NVIDIA_SMI": low,
            "QWEN3_TTS_IDLE_UNLOAD_S": "0",
        },
    )
    try:
        wait_http(gate.base, "/v1/models", 60)
        status, raw, _ = request(gate.base, "/v1/audio/speech", {"input": "wait", "voice": "vivian"}, timeout=30)
        body = as_json(raw)
        check("the VRAM gate refuses after its deadline with 503 gpu_memory_unavailable",
              status == 503 and body.get("error", {}).get("code") == "gpu_memory_unavailable", (status, body))
        check("the refusal names the last free reading", body.get("last_free_mib") == 1024, body)
        check("a refused gate never spawned a worker", gate.spawns() == 0, gate.spawns())
    finally:
        print("--- gate-refusal log tail ---")
        print(gate.tail())
        gate.stop()

    # The gate PASSES: enough free memory, and the wait is visible.
    high = os.path.join(workdir, "nvidia-smi-high")
    with open(high, "w") as fh:
        fh.write("#!/bin/sh\necho 8192\n")
    os.chmod(high, 0o755)
    passed = Shim(
        workdir,
        18083,
        {
            "QWEN3_TTS_DEVICE": "cuda:0",
            "QWEN3_TTS_GPU_MIN_FREE_MIB": "512",
            "QWEN3_TTS_GPU_WAIT_SECONDS": "10",
            "QWEN3_TTS_GPU_POLL_S": "0.2",
            "QWEN3_TTS_NVIDIA_SMI": high,
            "QWEN3_TTS_IDLE_UNLOAD_S": "0",
        },
    )
    try:
        wait_http(passed.base, "/v1/models", 60)
        status, raw, ctype = request(passed.base, "/v1/audio/speech", {"input": "gate", "voice": "vivian"}, timeout=120)
        check("the VRAM gate passes when the floor is met and serves audio", status == 200 and raw[:4] == b"RIFF", (status, raw[:8]))
        check("the gate path was actually taken", "gpu gate" in passed.tail(40), passed.tail(10))
    finally:
        print("--- gate-pass log tail ---")
        print(passed.tail())
        passed.stop()

    # A worker that fails to load: honest 503, retried by the NEXT request.
    exit1 = os.path.join(workdir, "exit1-worker.py")
    with open(exit1, "w") as fh:
        fh.write("#!/usr/bin/env python3\nimport sys\nprint('stub worker refuses to load', file=sys.stderr)\nsys.exit(2)\n")
    os.chmod(exit1, 0o755)
    broken = Shim(
        workdir,
        18084,
        {
            "QWEN3_TTS_WORKER": exit1,
            "QWEN3_TTS_IDLE_UNLOAD_S": "0",
            "QWEN3_TTS_LOAD_TIMEOUT_S": "5",
        },
    )
    try:
        wait_http(broken.base, "/v1/models", 60)
        status, raw, _ = request(broken.base, "/v1/audio/speech", {"input": "fail", "voice": "vivian"}, timeout=30)
        body = as_json(raw)
        check("a worker that dies before ready answers 503 model_not_loaded", status == 503 and body.get("error", {}).get("code") == "model_not_loaded", (status, body))
        check("the failure detail is carried", bool(body.get("detail")), body)
        before = as_json(request(broken.base, "/health", timeout=10)[1]).get("load_attempts", 0)
        request(broken.base, "/v1/audio/speech", {"input": "retry", "voice": "vivian"}, timeout=30)
        after = as_json(request(broken.base, "/health", timeout=10)[1]).get("load_attempts", 0)
        check("the NEXT request retries the failed start instead of wedging", after == before + 1, (before, after))
    finally:
        print("--- broken-worker log tail ---")
        print(broken.tail())
        broken.stop()

    if failures:
        print("FAILED %d assertion(s): %s" % (len(failures), failures))
        return 1
    print("shim surface, lazy start, single-flight, VRAM gate, idle kill and reload hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
