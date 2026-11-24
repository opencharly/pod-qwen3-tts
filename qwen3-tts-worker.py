#!/usr/bin/env python3
"""The on-demand Qwen3-TTS worker.

Started by the public shim ONLY when a synthesis request arrives, and killed by
it after the idle TTL. This process is the one that imports torch and owns the
CUDA context: killing it is what releases ALL of the pod's VRAM, including the
context the in-process design could never give back.

It binds a Unix socket (never a published port), loads the checkpoint before it
starts serving, and reports readiness over the pipe its parent passed. A parent
death (PR_SET_PDEATHSIG) terminates it, so a crashed shim can never leave an
orphan holding the card.
"""
import ctypes
import os
import signal
import sys
import threading
import traceback

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from qwen3_tts_common import (
    DEFAULT_VOICE,
    FALLBACK_VOICES,
    MODEL_ID_DEFAULT,
    SUPPORTED_ENCODINGS,
    encode_audio,
    error_response,
    error_summary as _error_summary,
    media_type,
)

MODEL_ID = os.environ.get("QWEN3_TTS_MODEL", MODEL_ID_DEFAULT)
DEVICE = os.environ.get("QWEN3_TTS_DEVICE", "cuda:0")
LANGUAGE = os.environ.get("QWEN3_TTS_LANGUAGE") or None
# Optional style/emotion instruction for the Qwen3-TTS CustomVoice path.
# A request's `instruct` field overrides it per call.
INSTRUCT = os.environ.get("QWEN3_TTS_INSTRUCT") or None
SOCKET = os.environ.get("QWEN3_TTS_WORKER_SOCKET", "/tmp/qwen3-tts-worker.sock")
LOG_LEVEL = os.environ.get("QWEN3_TTS_LOG_LEVEL", "info")
READY_FD = os.environ.get("QWEN3_TTS_READY_FD")
PARENT_PID = os.environ.get("QWEN3_TTS_PARENT_PID")


def _arm_parent_death():
    """Die with the shim, even if it is killed without a signal.

    Called BEFORE torch is imported so a dead parent cannot leave a GPU-holding
    orphan behind for the seconds a heavy import takes.
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        # PR_SET_PDEATHSIG = 1
        libc.prctl(1, signal.SIGTERM)
    except Exception as exc:
        print("qwen3-tts-worker: could not arm PR_SET_PDEATHSIG: %r" % (exc,), file=sys.stderr, flush=True)
    if PARENT_PID and str(os.getppid()) != str(PARENT_PID):
        print(
            "qwen3-tts-worker: the shim (pid %s) is already gone; exiting before loading anything"
            % PARENT_PID,
            file=sys.stderr,
            flush=True,
        )
        os._exit(3)


if PARENT_PID:
    _arm_parent_death()

_state = {
    "model": None,
    "voices": None,
    "voices_source": None,
    "loaded": False,
    "last_error": None,
}
_synth_lock = threading.Lock()
_ready_sent = False


def _ready(line):
    """Write the one readiness line to the shim, once, then close the pipe."""
    global _ready_sent
    if _ready_sent or not READY_FD:
        return
    _ready_sent = True
    try:
        os.write(int(READY_FD), (line + "\n").encode())
    except Exception as exc:
        print("qwen3-tts-worker: could not write the readiness pipe: %r" % (exc,), file=sys.stderr, flush=True)
    finally:
        try:
            os.close(int(READY_FD))
        except Exception:
            pass


def _resolve_voices(model):
    """Return (voices, source) from the checkpoint's own speaker list."""
    try:
        listed = model.get_supported_speakers()
    except Exception as exc:
        print(
            "qwen3-tts-worker: WARNING get_supported_speakers() failed (%r); falling back to the hardcoded preset list %s"
            % (exc, FALLBACK_VOICES),
            file=sys.stderr,
            flush=True,
        )
        return list(FALLBACK_VOICES), "fallback"
    if listed:
        voices = [str(name) for name in listed]
        if sorted(name.lower() for name in voices) != sorted(FALLBACK_VOICES):
            print(
                "qwen3-tts-worker: the checkpoint's own speaker list %s replaces the hardcoded %s"
                % (voices, FALLBACK_VOICES),
                file=sys.stderr,
                flush=True,
            )
        return voices, "model"
    print(
        "qwen3-tts-worker: WARNING the checkpoint exposes no speaker list; falling back to the hardcoded preset list %s"
        % (FALLBACK_VOICES,),
        file=sys.stderr,
        flush=True,
    )
    return list(FALLBACK_VOICES), "fallback"


def _build_model():
    import torch
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    dtype = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32
    print("qwen3-tts-worker: loading %s on %s (%s)" % (MODEL_ID, DEVICE, dtype), file=sys.stderr, flush=True)
    return Qwen3TTSModel.from_pretrained(MODEL_ID, device_map=DEVICE, dtype=dtype)


def _load():
    model = _build_model()
    voices, source = _resolve_voices(model)
    _state.update(model=model, voices=voices, voices_source=source, loaded=True, last_error=None)
    print(
        "qwen3-tts-worker: model ready (%d speakers from %s) on pid %d"
        % (len(voices), source, os.getpid()),
        file=sys.stderr,
        flush=True,
    )


def _render(text, voice, language, format_name, instruct):
    """Synthesize and encode. Callers run this off the event loop."""
    model = _state["model"]
    if model is None:
        raise RuntimeError("the model is not loaded")
    with _synth_lock:
        wavs, sample_rate = model.generate_custom_voice(
            text=text, language=language, speaker=voice, instruct=instruct
        )
    return encode_audio(np.asarray(wavs[0]), sample_rate, format_name)


app = FastAPI()


@app.get("/health")
def health():
    loaded = bool(_state["loaded"])
    body = {
        "status": "ok" if loaded else "unloaded",
        "loaded": loaded,
        "model": MODEL_ID,
        "pid": os.getpid(),
        "voices_source": _state.get("voices_source"),
    }
    return JSONResponse(body, status_code=200 if loaded else 503)


@app.get("/voices")
def voices():
    if _state["loaded"] and _state["voices"]:
        return {"voices": [{"id": n, "name": n} for n in _state["voices"]], "source": _state["voices_source"]}
    return {"voices": [{"id": n, "name": n} for n in FALLBACK_VOICES], "source": "fallback"}


@app.post("/synth")
async def synth(request: Request):
    import asyncio

    try:
        body = await request.json()
    except Exception:
        return error_response(400, "invalid_request_error", "invalid_json", "the request body must be JSON")
    if not isinstance(body, dict):
        return error_response(400, "invalid_request_error", "invalid_request", "the request body must be a JSON object")

    text = body.get("input")
    if not isinstance(text, str) or not text.strip():
        return error_response(400, "invalid_request_error", "invalid_input", "input must be a non-empty string", param="input")
    text = text.strip()

    format_name = body.get("response_format") or "wav"
    if format_name not in SUPPORTED_ENCODINGS:
        return error_response(
            400,
            "invalid_request_error",
            "unsupported_response_format",
            "response_format %r is not supported; this worker produces %s"
            % (format_name, ", ".join(sorted(SUPPORTED_ENCODINGS))),
            supported_formats=sorted(SUPPORTED_ENCODINGS),
        )

    voice = body.get("voice") or DEFAULT_VOICE
    language = body.get("language") or LANGUAGE
    instruct = body.get("instruct") or INSTRUCT

    served = _state.get("voices") or []
    if _state.get("voices_source") == "model" and served:  # boundary re-check
        if str(voice).lower() not in {name.lower() for name in served}:
            return error_response(
                400,
                "invalid_request_error",
                "unsupported_voice",
                "voice %r is not a speaker of %r; supported: %s" % (voice, MODEL_ID, ", ".join(served)),
                param="voice",
                supported_voices=list(served),
            )

    try:
        payload = await asyncio.to_thread(_render, text, voice, language, format_name, instruct)
    except Exception as exc:
        print("qwen3-tts-worker: synth failed: %r" % (exc,), file=sys.stderr, flush=True)
        return JSONResponse({"error": {"message": str(exc), "type": "server_error", "code": "synthesis_failed"}}, status_code=500)
    return Response(content=payload, media_type=media_type(format_name))


def _ready_when_started(server):
    """Signal ready only when uvicorn is listening AND the model is loaded."""
    while not server.started:
        threading.Event().wait(0.05)
    _ready("ready")


def main():
    try:
        os.unlink(SOCKET)
    except FileNotFoundError:
        pass
    try:
        _load()
    except BaseException:
        failure = traceback.format_exc()
        _state["last_error"] = failure
        print("qwen3-tts-worker: model load FAILED\n" + failure, file=sys.stderr, flush=True)
        _ready("error: " + (_error_summary(failure) or "the model load failed"))
        return 1

    import uvicorn

    config = uvicorn.Config(app, uds=SOCKET, log_level=LOG_LEVEL)
    server = uvicorn.Server(config)
    threading.Thread(target=_ready_when_started, args=(server,), name="qwen3-tts-ready", daemon=True).start()
    try:
        server.run()
    finally:
        try:
            os.unlink(SOCKET)
        except FileNotFoundError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
