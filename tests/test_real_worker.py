#!/usr/bin/env python3
"""Prove the REAL worker boundary on CPU — no GPU and no stub.

This is the R10/RDD witness for the process split: the actual
`qwen3-tts-worker.py` imports torch, loads the checkpoint, serves uvicorn over a
Unix socket, writes the ready pipe, and dies with its parent (PR_SET_PDEATHSIG).
It needs the checkpoint in the HF cache; without it the test SKIPS visibly
(R8) instead of mocking the model.

Run it with the candy's own environment (the pixi env), e.g.:

    HF_HOME=~/.local/share/charly/models/huggingface \
      pixi run python tests/test_real_worker.py
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(HERE, "qwen3-tts-server.py")):
    ROOT = HERE
elif os.path.exists(os.path.join(os.path.dirname(HERE), "qwen3-tts-server.py")):
    ROOT = os.path.dirname(HERE)
else:
    ROOT = os.path.expanduser("~/.local/bin")
SHIM = os.path.join(ROOT, "qwen3-tts-server.py")
WORKER = os.path.join(ROOT, "qwen3-tts-worker.py")
MODEL = os.environ.get("QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")

failures = []


def check(name, condition, detail=""):
    if condition:
        print("PASS " + name)
    else:
        print("FAIL " + name + " :: " + repr(detail))
        failures.append(name)


def request(base, path, body=None, timeout=900):
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


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except Exception:
        return False


def cached_checkpoint():
    roots = []
    hf = os.environ.get("HF_HOME")
    if hf:
        roots.append(hf)
    roots.append(os.path.expanduser("~/.cache/huggingface"))
    safe = "models--" + MODEL.replace("/", "--")
    for root in roots:
        if os.path.isdir(os.path.join(root, "hub", safe)):
            return os.path.join(root, "hub", safe)
    return None


def main():
    checkpoint = cached_checkpoint()
    if checkpoint is None:
        print("SKIP: the checkpoint %r is not in the HF cache (set HF_HOME to a store that has it)" % MODEL)
        return 0

    workdir = tempfile.mkdtemp(prefix="qwen3-tts-real-")
    socket_path = os.path.join(workdir, "worker.sock")
    log_path = os.path.join(workdir, "shim.log")
    port = 18091
    env = dict(os.environ)
    env.update(
        {
            "QWEN3_TTS_PORT": str(port),
            "QWEN3_TTS_DEVICE": "cpu",
            "QWEN3_TTS_GPU_MIN_FREE_MIB": "0",
            "QWEN3_TTS_WORKER": WORKER,
            "QWEN3_TTS_WORKER_SOCKET": socket_path,
            "QWEN3_TTS_IDLE_UNLOAD_S": "15",
            "QWEN3_TTS_LOAD_TIMEOUT_S": "900",
            "QWEN3_TTS_SYNTH_TIMEOUT_S": "900",
            "QWEN3_TTS_LOG_LEVEL": "info",
            "PYTHONUNBUFFERED": "1",
        }
    )
    log = open(log_path, "w+b")
    shim = subprocess.Popen(
        [sys.executable, SHIM],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    base = "http://127.0.0.1:%d" % port
    worker_pid = None
    try:
        deadline = time.monotonic() + 120
        status = None
        while time.monotonic() < deadline:
            try:
                status, payload, _ = request(base, "/v1/models", timeout=5)
                if status == 200:
                    break
            except Exception:
                pass
            time.sleep(0.1)
        check("the shim answers /v1/models before any load", status == 200, status)

        status, raw, _ = request(base, "/health", timeout=10)
        check("cold start is 503 unloaded (nothing loaded)", status == 503 and as_json(raw).get("status") == "unloaded", (status, as_json(raw)))

        status, raw, ctype = request(base, "/v1/audio/speech", {"input": "hello from the real worker", "voice": "vivian"}, timeout=900)
        check("the REAL worker loads the checkpoint and returns RIFF audio", status == 200 and raw[:4] == b"RIFF" and ctype.startswith("audio/"), (status, raw[:8], ctype))

        status, raw, _ = request(base, "/health", timeout=10)
        health = as_json(raw)
        worker_pid = health.get("worker_pid")
        check("/health reports ok with the real worker pid", status == 200 and health.get("status") == "ok" and isinstance(worker_pid, int), (status, health))

        with open(log_path, "rb") as fh:
            shim_log = fh.read().decode("utf-8", "replace")
        check("the worker reported its own model load (torch + checkpoint)", "model ready" in shim_log, shim_log[-1200:])
        check("the ready pipe carried the worker's readiness", "speech worker ready" in shim_log, shim_log[-1200:])

        deadline = time.monotonic() + 120
        idle = {}
        while time.monotonic() < deadline:
            idle = as_json(request(base, "/health", timeout=5)[1])
            if idle.get("loaded") is False:
                break
            time.sleep(0.5)
        check("after the idle TTL the REAL worker is released", idle.get("loaded") is False, idle)
        time.sleep(0.5)
        check("the idle release killed the real worker process", not pid_alive(worker_pid), worker_pid)

        status, raw, ctype = request(base, "/v1/audio/speech", {"input": "reload the real worker", "voice": "vivian"}, timeout=900)
        check("the next request reloads the real checkpoint and serves audio", status == 200 and raw[:4] == b"RIFF", (status, raw[:8]))

        health = as_json(request(base, "/health", timeout=10)[1])
        worker_pid = health.get("worker_pid")
        check("a fresh real worker pid is running", isinstance(worker_pid, int) and pid_alive(worker_pid), worker_pid)

        os.kill(shim.pid, signal.SIGKILL)
        shim.wait(timeout=30)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and pid_alive(worker_pid):
            time.sleep(0.2)
        check("PR_SET_PDEATHSIG killed the worker when the shim was SIGKILLed", not pid_alive(worker_pid), worker_pid)
    finally:
        if shim.poll() is None:
            shim.terminate()
            try:
                shim.wait(timeout=20)
            except Exception:
                shim.kill()
                shim.wait()
        log.close()
        with open(log_path, "rb") as fh:
            tail = fh.read().decode("utf-8", "replace").strip().splitlines()
        print("--- shim/worker log tail ---")
        for line in tail[-15:]:
            print(line)

    if failures:
        print("FAILED %d assertion(s): %s" % (len(failures), failures))
        return 1
    print("the REAL worker boundary holds: torch load, uvicorn-on-UDS, ready pipe, idle kill, reload, PDEATHSIG")
    return 0


if __name__ == "__main__":
    sys.exit(main())
