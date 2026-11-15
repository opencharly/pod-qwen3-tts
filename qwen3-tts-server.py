#!/usr/bin/env python3
"""OpenAI-compatible speech endpoint for Qwen3-TTS.

Serves the routes Open WebUI's `openai` TTS engine calls — POST /v1/audio/speech
and GET /v1/audio/voices — plus /v1/models and /health, over the upstream
`qwen-tts` python API.

Four contracts this module keeps honest:

* The checkpoint loads in a BACKGROUND thread at service start, never on the
  event loop. uvicorn must answer /v1/models within milliseconds while the
  load runs, because the consumer pays the whole load inside its own timeout
  otherwise (measured: an 8-33 s load against a 41.8 s client). /health is the
  readiness signal — 503 with a machine-readable reason while warming or after
  a failed load, 200 once the model serves. A request that has to wait for a load
  waits on a WORKER thread, so the event loop never blocks on a load or on
  synthesis and stays answerable throughout.
* A failed load is never sticky. The next request that finds nothing loading
  performs one fresh attempt, so a transient CUDA OOM heals as soon as memory
  frees; the last failure stays readable at /health and in the 5xx body.
* The request is validated BEFORE anything is loaded: a foreign `model`, an
  unsupported `response_format` and an unknown `voice` are 4xx naming what this
  server actually serves — never a silent substitution.
* `response_format` is honoured from the set this environment can really
  produce, measured at import, and the served voices are the checkpoint's own
  speaker list once it has one.
"""
import io
import os
import sys
import threading
import traceback
from contextlib import asynccontextmanager

import asyncio

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

MODEL_ID = os.environ.get("QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
DEFAULT_VOICE = os.environ.get("QWEN3_TTS_VOICE", "vivian")
LANGUAGE = os.environ.get("QWEN3_TTS_LANGUAGE") or None

# The preset voices of a CustomVoice checkpoint. Used ONLY while the checkpoint
# has not handed us its own list — before the load finishes and after a failed
# load — and a fallback is always announced on stderr, never silent. Once the
# model is loaded /v1/audio/voices answers the checkpoint's own list and
# /health names where that list came from.
FALLBACK_VOICES = ["vivian", "serena", "uncle_fu", "dylan", "eric", "ryan", "aiden", "ono_anna", "sohee"]

# response_format -> (libsndfile major format, subtype, response media type).
# `pcm` is headerless 16-bit little-endian at the model's sample rate (what the
# OpenAI API means by `pcm`); RAW is libsndfile's headerless writer.
ENCODINGS = {
    "wav": ("WAV", "PCM_16", "audio/wav"),
    "pcm": ("RAW", "PCM_16", "audio/pcm"),
    "flac": ("FLAC", "PCM_16", "audio/flac"),
    "mp3": ("MP3", "MPEG_LAYER_III", "audio/mpeg"),
}


def _probe_encodings():
    """Keep only the response formats this libsndfile can actually WRITE.

    `sf.available_formats()` lists the formats the library knows about, which is
    not the same question as "can this build write them" — mp3 needs libmp3lame
    at libsndfile build time. So the advertised set is MEASURED here with a
    short write probe, and a format this environment cannot produce is never
    advertised (and never silently substituted by another one).
    """
    measured = {}
    for name, (fmt, subtype, media_type) in ENCODINGS.items():
        try:
            probe = io.BytesIO()
            sf.write(probe, np.zeros(2400, dtype="int16"), 24000, format=fmt, subtype=subtype)
        except Exception as exc:
            print(
                "qwen3-tts: response_format %r is unavailable in this environment (%r); not advertised"
                % (name, exc),
                file=sys.stderr,
                flush=True,
            )
            continue
        if probe.tell() > 0:
            measured[name] = (fmt, subtype, media_type)
    return measured


SUPPORTED_ENCODINGS = _probe_encodings()
SUPPORTED_FORMATS = sorted(SUPPORTED_ENCODINGS)
# Optional style/emotion instruction for the Qwen3-TTS CustomVoice path.
# Set QWEN3_TTS_INSTRUCT for a deployment-wide default; a request's `instruct`
# field overrides it per call. Passed straight to the model's `instruct` kwarg,
# so the value is free text describing the delivery (emotion, pace, character).
INSTRUCT = os.environ.get("QWEN3_TTS_INSTRUCT") or None

_state = {
    "model": None,
    "loaded": False,
    "loading": False,
    "attempts": 0,
    "last_error": None,
    "voices": None,
    "voices_source": None,
}
_cond = threading.Condition(threading.Lock())


class LoadFailed(RuntimeError):
    """The checkpoint is not loaded; the request must answer 5xx, never 200."""


def _error_summary(traceback_text):
    """The last non-empty traceback line, for a /health body and a 5xx body."""
    if not traceback_text:
        return None
    lines = [line.strip() for line in traceback_text.strip().splitlines() if line.strip()]
    return lines[-1][:400] if lines else None


def _snapshot():
    with _cond:
        return dict(_state)


def readiness(state):
    """Map a load-state snapshot onto the /health answer: (status_code, body).

    Pure on purpose — the candy's plan check imports it and asserts both
    directions (loaded -> 200, everything else -> 503) without a checkpoint.
    """
    loaded = bool(state.get("loaded"))
    loading = bool(state.get("loading"))
    attempts = int(state.get("attempts") or 0)
    detail = _error_summary(state.get("last_error"))
    body = {
        "status": "ok" if loaded else ("loading" if loading else ("error" if detail else "unloaded")),
        "ready": loaded,
        "loaded": loaded,
        "model": MODEL_ID,
        "load_attempts": attempts,
    }
    if loaded:
        body["voices"] = len(state.get("voices") or [])
        body["voices_source"] = state.get("voices_source")
        return 200, body
    body["detail"] = detail or ("the checkpoint is loading" if loading else "the checkpoint has not been loaded yet")
    return 503, body


def _resolve_voices(model):
    """Return (voices, source) from the checkpoint's own speaker list."""
    try:
        listed = model.get_supported_speakers()
    except Exception as exc:
        print(
            "qwen3-tts: WARNING get_supported_speakers() failed (%r); falling back to the hardcoded preset list %s"
            % (exc, FALLBACK_VOICES),
            file=sys.stderr,
            flush=True,
        )
        return list(FALLBACK_VOICES), "fallback"
    if listed:
        voices = [str(name) for name in listed]
        if sorted(name.lower() for name in voices) != sorted(FALLBACK_VOICES):
            print(
                "qwen3-tts: the checkpoint's own speaker list %s replaces the hardcoded %s"
                % (voices, FALLBACK_VOICES),
                file=sys.stderr,
                flush=True,
            )
        return voices, "model"
    print(
        "qwen3-tts: WARNING the checkpoint exposes no speaker list; falling back to the hardcoded preset list %s"
        % (FALLBACK_VOICES,),
        file=sys.stderr,
        flush=True,
    )
    return list(FALLBACK_VOICES), "fallback"


def _build_model():
    import torch
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    device = os.environ.get("QWEN3_TTS_DEVICE", "cuda:0")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    print("qwen3-tts: loading %s on %s (%s)" % (MODEL_ID, device, dtype), file=sys.stderr, flush=True)
    return Qwen3TTSModel.from_pretrained(MODEL_ID, device_map=device, dtype=dtype)


def _load_once():
    """Run exactly ONE load attempt, publish its outcome, and clear `loading`.

    The heavy work runs with NO lock held, so /health and /v1/models stay
    answerable while the checkpoint loads.
    """
    with _cond:
        _state["attempts"] += 1
        _state["last_error"] = None
    model = None
    failure = None
    try:
        model = _build_model()
    except BaseException:
        # A load attempt must never leave `loading` set, whatever it raises.
        failure = traceback.format_exc()
    voices = None
    source = None
    if model is not None:
        voices, source = _resolve_voices(model)
    with _cond:
        if model is not None:
            _state.update(model=model, loaded=True, voices=voices, voices_source=source)
        else:
            _state["last_error"] = failure
        _state["loading"] = False
        _cond.notify_all()
    if model is None:
        print("qwen3-tts: model load FAILED\n" + failure, file=sys.stderr, flush=True)
    else:
        print(
            "qwen3-tts: model ready (%d speakers from %s)" % (len(voices), source),
            file=sys.stderr,
            flush=True,
        )


def start_background_load():
    """Start the checkpoint load in the background — uvicorn never waits for it."""
    with _cond:
        if _state["loaded"] or _state["loading"]:
            return
        _state["loading"] = True
    try:
        threading.Thread(target=_load_once, name="qwen3-tts-load", daemon=True).start()
    except BaseException:
        with _cond:
            _state["loading"] = False
            _cond.notify_all()
        raise


def ensure_model():
    """Return the loaded model, waiting for an in-flight load instead of starting a second.

    One request performs at most ONE load attempt. A request that was waiting on
    an attempt which then failed answers that failure; the NEXT request retries,
    so a transient CUDA OOM never wedges the process for its lifetime.
    """
    with _cond:
        waited = False
        while not _state["loaded"]:
            if _state["loading"]:
                _cond.wait()
                waited = True
                continue
            if waited:
                raise LoadFailed(_error_summary(_state["last_error"]) or "the model is not loaded")
            _state["loading"] = True
            break
        else:
            return _state["model"]
    try:
        _load_once()
    finally:
        with _cond:
            if _state["loading"]:
                _state["loading"] = False
                _cond.notify_all()
    with _cond:
        if _state["loaded"]:
            return _state["model"]
        raise LoadFailed(_error_summary(_state["last_error"]) or "the model is not loaded")


def encode_audio(audio, sample_rate, format_name):
    """Encode one channel of model output into the requested response format."""
    fmt, subtype, _ = SUPPORTED_ENCODINGS[format_name]
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio), int(sample_rate), format=fmt, subtype=subtype)
    return buf.getvalue()


_synth_lock = threading.Lock()


def _render(text, voice, language, format_name, instruct=None):
    """Load-or-wait, synthesize and encode OFF the event loop.

    Both the load (minutes on a cold cache) and the model call are blocking, so they
    run on a worker thread: the event loop must keep answering /v1/models and /health
    while a request waits for a load. Synthesis stays serialized — the upstream model
    is not documented thread-safe — while the WAIT for an in-flight load is shared, so
    a burst costs one load attempt rather than one per request.
    """
    model = ensure_model()
    with _synth_lock:
        wavs, sample_rate = model.generate_custom_voice(
            text=text, language=language, speaker=voice, instruct=instruct
        )
    return encode_audio(np.asarray(wavs[0]), sample_rate, format_name)


def voice_validation(state, voice):
    """Return the 400 response for a voice the loaded checkpoint does not serve.

    Pure over a /health state snapshot, so the contract is testable without a
    checkpoint: validation only applies once the checkpoint's OWN speaker list is
    known (before that the preset list is the best information available).
    """
    if not (state.get("loaded") and state.get("voices_source") == "model" and isinstance(voice, str)):
        return None
    served = state.get("voices") or []
    if voice.lower() in {name.lower() for name in served}:
        return None
    return _error(
        400,
        "invalid_request_error",
        "unsupported_voice",
        "voice %r is not a speaker of %r; supported: %s"
        % (voice, MODEL_ID, ", ".join(served)),
        param="voice",
        supported_voices=list(served),
    )


def _error(status_code, error_type, code, message, param=None, **extra):
    body = {"error": {"message": message, "type": error_type, "code": code}}
    if param is not None:
        body["error"]["param"] = param
    body.update(extra)
    return JSONResponse(body, status_code=status_code)


@asynccontextmanager
async def lifespan(_app):
    start_background_load()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    status_code, body = readiness(_snapshot())
    return JSONResponse(body, status_code=status_code)


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}


def voices_payload(state):
    """Return the voice list the picker reads, and where it came from.

    Pure over a /health state snapshot: the checkpoint's own speaker list once it is
    known, the preset fallback before that — and `source` names which, so a client can
    tell an authoritative list from a provisional one.
    """
    if state.get("loaded") and state.get("voices"):
        return {
            "voices": [{"id": name, "name": name} for name in state["voices"]],
            "source": state.get("voices_source"),
        }
    return {"voices": [{"id": name, "name": name} for name in FALLBACK_VOICES], "source": "fallback"}


@app.get("/v1/audio/voices")
def voices():
    return voices_payload(_snapshot())


@app.post("/v1/audio/speech")
async def speech(request: Request):
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

    voice = body.get("voice") or DEFAULT_VOICE
    language = body.get("language") or LANGUAGE
    instruct = body.get("instruct") or INSTRUCT

    state = _snapshot()
    refusal = voice_validation(state, voice)
    if refusal is not None:
        return refusal

    print(
        "qwen3-tts: synth voice=%s format=%s chars=%d" % (voice, format_name, len(text)),
        file=sys.stderr,
        flush=True,
    )
    try:
        payload = await asyncio.to_thread(_render, text, voice, language, format_name, instruct)
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
    except Exception as exc:
        print("qwen3-tts: synth failed: %r" % (exc,), file=sys.stderr, flush=True)
        return _error(500, "server_error", "synthesis_failed", str(exc))
    return Response(content=payload, media_type=SUPPORTED_ENCODINGS[format_name][2])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("QWEN3_TTS_PORT", "8000")),
        log_level="info",
    )
