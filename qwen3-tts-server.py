#!/usr/bin/env python3
"""OpenAI-compatible speech endpoint for Qwen3-TTS — the torch-free shim.

The public surface Open WebUI's `openai` TTS engine calls (POST
/v1/audio/speech, GET /v1/audio/voices) plus /v1/models and /health. It owns no
model: the checkpoint lives in an on-demand worker process (`qwen3-tts-worker.py`)
that this shim starts when a synthesis request arrives and kills after the idle
TTL. Killing the worker is what releases ALL of the pod's VRAM — weights, cached
blocks and the CUDA context alike.

Contracts this module keeps honest:

* **Nothing is loaded at service start.** `/v1/models` answers immediately,
  `/health` is 503 `unloaded`, and the pod holds no VRAM until a synthesis
  request asks for it.
* **A request waits for free NVRAM, bounded.** Before the worker is spawned the
  shim polls the GPU's free memory until the configured floor is met, up to
  `QWEN3_TTS_GPU_WAIT_SECONDS`; the deadline is the guard, not a blind retry.
  A missed deadline is a 503 `gpu_memory_unavailable` naming the last reading.
* **One spawn per burst.** Concurrent requests wait for the same start rather
  than spawning a worker each. A failed start is never sticky: the next request
  retries from scratch.
* **The request is validated BEFORE anything is started.** A foreign `model`,
  an unsupported `response_format` and an unknown voice are 4xx naming what this
  server actually serves — never a silent substitution.
* **Every knob is an env var** (see the README/SKILL table), with the same
  default in code so the script also works standalone.
"""
import asyncio
import collections
import os
import select
import shutil
import subprocess
import sys
import threading
import time
import traceback
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from qwen3_tts_common import (
    DEFAULT_VOICE,
    FALLBACK_VOICES,
    MODEL_ID_DEFAULT,
    SUPPORTED_ENCODINGS,
    SUPPORTED_FORMATS,
    error_response as _error,
    error_summary as _error_summary,
    media_type,
)

MODEL_ID = os.environ.get("QWEN3_TTS_MODEL", MODEL_ID_DEFAULT)
DEFAULT_VOICE_ENV = os.environ.get("QWEN3_TTS_VOICE", DEFAULT_VOICE)
LANGUAGE = os.environ.get("QWEN3_TTS_LANGUAGE") or None
INSTRUCT = os.environ.get("QWEN3_TTS_INSTRUCT") or None
LOG_LEVEL = os.environ.get("QWEN3_TTS_LOG_LEVEL", "info")

# The device the worker will use; the shim only needs it to decide whether the
# free-VRAM gate applies (a `cpu` deployment has no GPU to wait for).
DEVICE = os.environ.get("QWEN3_TTS_DEVICE", "cuda:0")
# Seconds with no synthesis before the worker is killed (0 = never).
IDLE_UNLOAD_S = float(os.environ.get("QWEN3_TTS_IDLE_UNLOAD_S", "600") or 0)
# Free-VRAM floor and the bounded wait before the worker is spawned.
GPU_MIN_FREE_MIB = int(os.environ.get("QWEN3_TTS_GPU_MIN_FREE_MIB", "4096") or 0)
GPU_WAIT_SECONDS = float(os.environ.get("QWEN3_TTS_GPU_WAIT_SECONDS", "600") or 0)
GPU_INDEX = os.environ.get("QWEN3_TTS_GPU_INDEX", "0")
GPU_POLL_S = float(os.environ.get("QWEN3_TTS_GPU_POLL_S", "5") or 5)
NVIDIA_SMI = os.environ.get("QWEN3_TTS_NVIDIA_SMI", "nvidia-smi")
# spawn -> ready deadline, covering a cold checkpoint download.
LOAD_TIMEOUT_S = float(os.environ.get("QWEN3_TTS_LOAD_TIMEOUT_S", "1800") or 1800)
WORKER = os.environ.get(
    "QWEN3_TTS_WORKER",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3-tts-worker.py"),
)
WORKER_SOCKET = os.environ.get("QWEN3_TTS_WORKER_SOCKET", "/tmp/qwen3-tts-worker.sock")
WORKER_STOP_S = float(os.environ.get("QWEN3_TTS_WORKER_STOP_S", "15") or 15)
SYNTH_TIMEOUT_S = float(os.environ.get("QWEN3_TTS_SYNTH_TIMEOUT_S", "600") or 600)

_state = {
    "phase": "unloaded",  # unloaded | waiting_for_vram | starting | ready | stopping | error
    "loaded": False,
    "loading": False,
    "attempts": 0,
    "last_error": None,
    "detail": "no speech model is loaded until the first synthesis request",
    "voices": None,
    "voices_source": None,
    "last_used": 0.0,
    "vram_free_mib": None,
    "worker_pid": None,
    "gpu_min_free_mib": GPU_MIN_FREE_MIB,
    "idle_unload_s": IDLE_UNLOAD_S,
}
_cond = threading.Condition(threading.Lock())
_active_lock = threading.Lock()
_active = 0
_proc = None
_last_exc = None
_logs = collections.deque(maxlen=40)
_log_lock = threading.Lock()


class LoadFailed(RuntimeError):
    """The worker is not serving; the request must answer 5xx, never 200."""


class GpuUnavailable(RuntimeError):
    """The deployment asks for CUDA but no usable NVIDIA tooling is present."""


class GpuMemoryUnavailable(RuntimeError):
    """The bounded wait for free VRAM expired."""

    def __init__(self, min_free_mib, last_free_mib, waited_s):
        self.min_free_mib = min_free_mib
        self.last_free_mib = last_free_mib
        self.waited_s = round(waited_s, 1)
        super().__init__(
            "no GPU with >= %d MiB free within %.0fs (last reading: %s MiB)"
            % (min_free_mib, waited_s, last_free_mib)
        )


def _log_tail():
    with _log_lock:
        return (" (last worker log: %s)" % _logs[-1]) if _logs else ""


def _snapshot():
    with _cond:
        return dict(_state)


def _touch():
    """Record that a synthesis request used the model (resets the idle TTL)."""
    with _cond:
        _state["last_used"] = time.monotonic()


def _gate_enabled():
    return DEVICE.startswith("cuda") and GPU_MIN_FREE_MIB > 0


def _free_vram_mib():
    """Read the configured GPU's free memory, in MiB. Raises GpuUnavailable."""
    if shutil.which(NVIDIA_SMI) is None:
        raise GpuUnavailable(
            "%s was not found, but QWEN3_TTS_DEVICE=%s requires an NVIDIA GPU "
            "(set QWEN3_TTS_GPU_MIN_FREE_MIB=0 for a deliberate CPU deployment)"
            % (NVIDIA_SMI, DEVICE)
        )
    try:
        proc = subprocess.run(
            [NVIDIA_SMI, "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(GPU_INDEX)],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired as exc:
        raise GpuUnavailable("%s did not answer within 10s: %r" % (NVIDIA_SMI, exc))
    if proc.returncode != 0:
        raise GpuUnavailable(
            "%s exited %d: %s" % (NVIDIA_SMI, proc.returncode, (proc.stderr or proc.stdout).strip()[:300])
        )
    lines = [line.strip() for line in (proc.stdout or "").strip().splitlines() if line.strip()]
    if not lines:
        raise GpuUnavailable("%s returned no free-memory reading" % NVIDIA_SMI)
    try:
        return int(float(lines[0]))
    except ValueError:
        raise GpuUnavailable("%s returned an unreadable reading: %r" % (NVIDIA_SMI, lines[0]))


def _free_vram_or_none():
    try:
        return _free_vram_mib()
    except Exception:
        return None


def _wait_for_vram():
    """Block until the GPU has the configured free floor, bounded by the deadline."""
    if not _gate_enabled():
        with _cond:
            _state["vram_free_mib"] = _free_vram_or_none()
            _cond.notify_all()
        return
    started = time.monotonic()
    deadline = started + GPU_WAIT_SECONDS
    while True:
        free = _free_vram_mib()
        with _cond:
            _state.update(
                phase="waiting_for_vram",
                loading=True,
                vram_free_mib=free,
                detail="waiting for %d MiB free on GPU %s (last reading: %d MiB)"
                % (GPU_MIN_FREE_MIB, GPU_INDEX, free),
            )
            _cond.notify_all()
        if free >= GPU_MIN_FREE_MIB:
            print(
                "qwen3-tts: gpu gate: %d MiB free (>= %d) — starting the speech worker"
                % (free, GPU_MIN_FREE_MIB),
                file=sys.stderr,
                flush=True,
            )
            with _cond:
                _state.update(phase="starting", detail="starting the speech worker")
                _cond.notify_all()
            return
        if time.monotonic() >= deadline:
            raise GpuMemoryUnavailable(GPU_MIN_FREE_MIB, free, time.monotonic() - started)
        time.sleep(GPU_POLL_S)


def _drain_logs(proc):
    """Forward the worker's output to this process's stderr and a bounded ring."""
    try:
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if not line:
                continue
            with _log_lock:
                _logs.append(line)
            print(line, file=sys.stderr, flush=True)
    except Exception:
        pass


def _read_ready_pipe(fd, proc):
    """Wait for the worker's one readiness line. Returns (outcome, detail)."""
    deadline = time.monotonic() + LOAD_TIMEOUT_S
    buf = b""
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "timeout", "the speech worker did not become ready within %.0fs%s" % (
                    LOAD_TIMEOUT_S,
                    _log_tail(),
                )
            readable, _, _ = select.select([fd], [], [], min(remaining, 1.0))
            if not readable:
                continue
            chunk = os.read(fd, 4096)
            if not chunk:
                return "eof", "the speech worker exited before it was ready (rc=%r)%s" % (
                    proc.poll(),
                    _log_tail(),
                )
            buf += chunk
            if b"\n" in buf:
                line, _, _ = buf.partition(b"\n")
                text = line.decode("utf-8", "replace").strip()
                if text == "ready":
                    return "ready", None
                if text.startswith("error:"):
                    return "error", text[len("error:") :].strip() or "the speech worker failed to load"
                return "error", text or "the speech worker reported an unknown readiness state"
    finally:
        try:
            os.close(fd)
        except Exception:
            pass


def _kill_proc(proc):
    if proc is None:
        return
    if proc.poll() is None:
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=WORKER_STOP_S)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass


def _fetch_voices():
    """Read the worker's authoritative speaker list; None keeps the fallback."""
    try:
        transport = httpx.HTTPTransport(uds=WORKER_SOCKET)
        with httpx.Client(transport=transport, base_url="http://worker", timeout=10) as client:
            data = client.get("/voices").json()
        return [v["id"] for v in data.get("voices", [])], data.get("source")
    except Exception as exc:
        print(
            "qwen3-tts: could not read the worker's speaker list (%r); serving the preset fallback" % (exc,),
            file=sys.stderr,
            flush=True,
        )
        return None, None


def _start_once():
    """Run exactly ONE worker start attempt and publish its outcome."""
    global _proc, _last_exc
    with _cond:
        _state["attempts"] += 1
        _last_exc = None
    try:
        _wait_for_vram()
    except (GpuUnavailable, GpuMemoryUnavailable) as exc:
        print("qwen3-tts: %s" % (exc,), file=sys.stderr, flush=True)
        with _cond:
            _last_exc = exc
            _state.update(
                phase="error",
                loading=False,
                last_error=traceback.format_exc(),
                detail=str(exc),
                vram_free_mib=_free_vram_or_none(),
            )
            _cond.notify_all()
        raise

    proc = None
    try:
        if not os.path.exists(WORKER):
            raise FileNotFoundError("the speech worker script is missing: %s" % WORKER)
        if os.path.exists(WORKER_SOCKET):
            os.unlink(WORKER_SOCKET)
        read_fd, write_fd = os.pipe()
        env = dict(os.environ)
        env.update(
            {
                "QWEN3_TTS_READY_FD": str(write_fd),
                "QWEN3_TTS_PARENT_PID": str(os.getpid()),
                "QWEN3_TTS_WORKER_SOCKET": WORKER_SOCKET,
            }
        )
        proc = subprocess.Popen(
            [sys.executable, WORKER],
            env=env,
            pass_fds=(write_fd,),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        os.close(write_fd)
        with _cond:
            _proc = proc
            _state["worker_pid"] = proc.pid
        threading.Thread(target=_drain_logs, args=(proc,), name="qwen3-tts-worker-logs", daemon=True).start()
        outcome, detail = _read_ready_pipe(read_fd, proc)
    except Exception as exc:
        failure = traceback.format_exc()
        _kill_proc(proc)
        with _cond:
            _proc = None
            _state.update(
                phase="error",
                loaded=False,
                loading=False,
                worker_pid=None,
                last_error=failure,
                detail="could not start the speech worker: %r" % (exc,),
                vram_free_mib=_free_vram_or_none(),
            )
            _cond.notify_all()
        return

    if outcome != "ready":
        _kill_proc(proc)
        with _cond:
            _proc = None
            _state.update(
                phase="error",
                loaded=False,
                loading=False,
                worker_pid=None,
                last_error=detail,
                detail=detail,
                vram_free_mib=_free_vram_or_none(),
            )
            _cond.notify_all()
        return

    voices, source = _fetch_voices()
    with _cond:
        _state.update(
            phase="ready",
            loaded=True,
            loading=False,
            last_error=None,
            detail=None,
            last_used=time.monotonic(),
            worker_pid=proc.pid,
            voices=voices,
            voices_source=source,
            vram_free_mib=_free_vram_or_none(),
        )
        _cond.notify_all()
    print(
        "qwen3-tts: speech worker ready (pid %d, %d speakers from %s)"
        % (proc.pid, len(voices or FALLBACK_VOICES), source or "fallback"),
        file=sys.stderr,
        flush=True,
    )


def ensure_worker():
    """Return the running worker, starting it once for a concurrent burst.

    One request runs at most ONE start attempt. A request that waited on an
    attempt which then failed answers that failure; the NEXT request retries, so
    a transient OOM or a busy GPU never wedges the process.
    """
    with _cond:
        waited = False
        while _state["phase"] != "ready":
            phase = _state["phase"]
            if phase in ("waiting_for_vram", "starting", "stopping"):
                _cond.wait()
                waited = True
                continue
            if waited and _state["phase"] == "error":
                raise _last_exc or LoadFailed(
                    _state.get("detail") or _error_summary(_state.get("last_error")) or "the speech model is not loaded"
                )
            _state.update(phase="starting", loading=True, last_error=None, detail="starting the speech worker")
            _cond.notify_all()
            break
        else:
            return _proc
    try:
        _start_once()
    finally:
        with _cond:
            if _state["phase"] == "starting":
                _state.update(phase="error", loading=False, detail="the speech worker failed to start")
                _cond.notify_all()
    with _cond:
        if _state["phase"] == "ready":
            return _proc
        raise _last_exc or LoadFailed(
            _state.get("detail") or _error_summary(_state.get("last_error")) or "the speech model is not loaded"
        )


def _stop_worker(reason):
    """Kill the worker (releasing ALL its VRAM) and publish the unloaded state."""
    global _proc, _last_exc
    with _cond:
        _last_exc = None
        proc = _proc
        if proc is None:
            _state.update(
                phase="unloaded",
                loaded=False,
                loading=False,
                worker_pid=None,
                voices=None,
                voices_source=None,
                detail="the speech model is not loaded",
            )
            _cond.notify_all()
            return
    before = _free_vram_or_none()
    _kill_proc(proc)
    after = _free_vram_or_none()
    with _cond:
        if _proc is proc:
            _proc = None
        _state.update(
            phase="unloaded",
            loaded=False,
            loading=False,
            worker_pid=None,
            voices=None,
            voices_source=None,
            last_error=None,
            detail="the speech model was released (%s)" % reason,
            vram_free_mib=after,
        )
        _cond.notify_all()
    print(
        "qwen3-tts: speech worker released (%s); free VRAM %s -> %s MiB"
        % (reason, before, after),
        file=sys.stderr,
        flush=True,
    )


def _mark_worker_dead(detail):
    global _proc, _last_exc
    with _cond:
        _last_exc = None
        proc = _proc
        _proc = None
        _state.update(
            phase="error",
            loaded=False,
            loading=False,
            worker_pid=None,
            voices=None,
            voices_source=None,
            last_error=detail,
            detail=detail,
            vram_free_mib=_free_vram_or_none(),
        )
        _cond.notify_all()
    _kill_proc(proc)


def _idle_loop():
    """Release the worker after IDLE_UNLOAD_S seconds without a synthesis."""
    interval = max(1.0, min(IDLE_UNLOAD_S / 2.0, 30.0))
    while True:
        time.sleep(interval)
        with _active_lock:
            if _active > 0:
                continue
            with _cond:
                if _state["phase"] != "ready":
                    continue
                idle_for = time.monotonic() - float(_state.get("last_used") or 0.0)
                if idle_for < IDLE_UNLOAD_S:
                    continue
                _state["phase"] = "stopping"
                _cond.notify_all()
        _stop_worker("idle for %.1fs" % idle_for)


def _shutdown():
    with _cond:
        proc = _proc
    if proc is not None:
        _stop_worker("shutdown")


def _proxied_synth(payload):
    transport = httpx.HTTPTransport(uds=WORKER_SOCKET)
    with httpx.Client(transport=transport, base_url="http://worker", timeout=SYNTH_TIMEOUT_S) as client:
        return client.post("/synth", json=payload)


def readiness(state):
    """Map a worker-state snapshot onto the /health answer: (status_code, body).

    Pure on purpose — the candy's plan check imports it and asserts every
    direction (ready -> 200, everything else -> 503) without a checkpoint.
    """
    loaded = bool(state.get("loaded"))
    phase = str(state.get("phase") or "unloaded")
    attempts = int(state.get("attempts") or 0)
    detail = state.get("detail") or _error_summary(state.get("last_error"))
    status = "ok" if loaded else {
        "waiting_for_vram": "waiting_for_vram",
        "starting": "loading",
        "stopping": "unloading",
        "error": "error",
    }.get(phase, "unloaded")
    body = {
        "status": status,
        "ready": loaded,
        "loaded": loaded,
        "model": MODEL_ID,
        "phase": phase,
        "load_attempts": attempts,
    }
    for key in ("vram_free_mib", "worker_pid", "gpu_min_free_mib", "idle_unload_s"):
        value = state.get(key)
        if value is not None:
            body[key] = value
    if loaded:
        body["voices"] = len(state.get("voices") or [])
        body["voices_source"] = state.get("voices_source")
        return 200, body
    body["detail"] = detail or (
        "the speech model is loading" if phase == "starting" else "the speech model has not been loaded yet"
    )
    return 503, body


def voices_payload(state):
    """Return the voice list the picker reads, and where it came from."""
    if state.get("loaded") and state.get("voices"):
        return {
            "voices": [{"id": name, "name": name} for name in state["voices"]],
            "source": state.get("voices_source"),
        }
    return {"voices": [{"id": name, "name": name} for name in FALLBACK_VOICES], "source": "fallback"}


def voice_validation(state, voice):
    """Return the 400 response for a voice the loaded checkpoint does not serve."""
    if not (state.get("loaded") and state.get("voices_source") == "model" and isinstance(voice, str)):
        return None
    served = state.get("voices") or []
    if voice.lower() in {name.lower() for name in served}:
        return None
    return _error(
        400,
        "invalid_request_error",
        "unsupported_voice",
        "voice %r is not a speaker of %r; supported: %s" % (voice, MODEL_ID, ", ".join(served)),
        param="voice",
        supported_voices=list(served),
    )


@asynccontextmanager
async def lifespan(_app):
    if IDLE_UNLOAD_S > 0:
        threading.Thread(target=_idle_loop, name="qwen3-tts-idle", daemon=True).start()
        print(
            "qwen3-tts: the speech worker is released after %.0fs without a synthesis request" % IDLE_UNLOAD_S,
            file=sys.stderr,
            flush=True,
        )
    else:
        print("qwen3-tts: idle unload disabled (QWEN3_TTS_IDLE_UNLOAD_S=0)", file=sys.stderr, flush=True)
    print(
        "qwen3-tts: no speech model is loaded until the first synthesis request "
        "(gpu gate: %s MiB free on GPU %s, wait %ss)"
        % (GPU_MIN_FREE_MIB, GPU_INDEX, int(GPU_WAIT_SECONDS)),
        file=sys.stderr,
        flush=True,
    )
    try:
        yield
    finally:
        _shutdown()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    status_code, body = readiness(_snapshot())
    return JSONResponse(body, status_code=status_code)


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}


@app.get("/v1/audio/voices")
def voices():
    return voices_payload(_snapshot())


@app.post("/v1/audio/speech")
async def speech(request: Request):
    global _active

    try:
        body = await request.json()
    except Exception:
        return _error(400, "invalid_request_error", "invalid_json", "the request body must be JSON")
    if not isinstance(body, dict):
        return _error(400, "invalid_request_error", "invalid_request", "the request body must be a JSON object")

    addressed = body.get("model")
    if isinstance(addressed, str) and not addressed.strip():
        addressed = None
    if addressed is not None and str(addressed) != MODEL_ID:
        return _error(
            400,
            "invalid_request_error",
            "model_not_served",
            "this server serves %r, not %r" % (MODEL_ID, addressed),
            param="model",
            served_model=MODEL_ID,
        )

    requested_format = body.get("response_format")
    format_name = "wav" if requested_format in (None, "") else str(requested_format).strip().lower()
    if format_name not in SUPPORTED_ENCODINGS:
        return _error(
            400,
            "invalid_request_error",
            "unsupported_response_format",
            "response_format %r is not supported; this server produces %s"
            % (requested_format, ", ".join(SUPPORTED_FORMATS)),
            param="response_format",
            supported_formats=SUPPORTED_FORMATS,
        )

    text = body.get("input")
    if not isinstance(text, str) or not text.strip():
        return _error(400, "invalid_request_error", "invalid_input", "input must be a non-empty string", param="input")
    text = text.strip()

    voice = body.get("voice") or DEFAULT_VOICE_ENV
    language = body.get("language") or LANGUAGE
    instruct = body.get("instruct") or INSTRUCT

    refusal = voice_validation(_snapshot(), voice)
    if refusal is not None:
        return refusal

    with _active_lock:
        _active += 1
    try:
        try:
            await asyncio.to_thread(ensure_worker)
        except GpuMemoryUnavailable as exc:
            return _error(
                503,
                "service_unavailable",
                "gpu_memory_unavailable",
                "no GPU has enough free memory for the speech model",
                detail=str(exc),
                min_free_mib=exc.min_free_mib,
                last_free_mib=exc.last_free_mib,
                waited_s=exc.waited_s,
                retry=True,
            )
        except GpuUnavailable as exc:
            return _error(
                503,
                "service_unavailable",
                "gpu_unavailable",
                "the GPU required by this deployment is not available",
                detail=str(exc),
                retry=True,
            )
        except LoadFailed as exc:
            return _error(
                503,
                "service_unavailable",
                "model_not_loaded",
                "the speech model is not loaded",
                detail=str(exc),
                model=MODEL_ID,
                retry=True,
            )

        print(
            "qwen3-tts: synth voice=%s format=%s chars=%d" % (voice, format_name, len(text)),
            file=sys.stderr,
            flush=True,
        )
        try:
            resp = await asyncio.to_thread(
                _proxied_synth,
                {
                    "input": text,
                    "voice": voice,
                    "language": language,
                    "instruct": instruct,
                    "response_format": format_name,
                },
            )
        except Exception as exc:
            _mark_worker_dead("the speech worker stopped answering: %r" % (exc,))
            return _error(
                503,
                "service_unavailable",
                "model_not_loaded",
                "the speech worker stopped answering",
                detail=str(exc),
                model=MODEL_ID,
                retry=True,
            )

        if resp.status_code != 200:
            try:
                payload = resp.json()
            except Exception:
                payload = {
                    "error": {
                        "message": (resp.text or "")[:400],
                        "type": "server_error",
                        "code": "synthesis_failed",
                    }
                }
            return JSONResponse(payload, status_code=resp.status_code)

        _touch()
        return Response(
            content=resp.content,
            media_type=resp.headers.get("content-type", media_type(format_name)),
        )
    finally:
        with _active_lock:
            _active -= 1


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("QWEN3_TTS_PORT", "8000")),
        log_level=LOG_LEVEL,
    )
