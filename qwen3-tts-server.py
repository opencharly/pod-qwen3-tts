#!/usr/bin/env python3
"""OpenAI-compatible speech endpoint for Qwen3-TTS.

Serves the two routes Open WebUI's `openai` TTS engine calls —
POST /v1/audio/speech and GET /v1/audio/voices — over the upstream
`qwen-tts` python API. The model loads lazily on the first synthesis request
(loading it in the constructor makes the service look dead for minutes).
"""
import io
import os
import sys
import threading
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
import numpy as np
import soundfile as sf

MODEL_ID = os.environ.get("QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
DEFAULT_VOICE = os.environ.get("QWEN3_TTS_VOICE", "vivian")
LANGUAGE = os.environ.get("QWEN3_TTS_LANGUAGE") or None

VOICES = ["vivian", "serena", "uncle_fu", "dylan", "eric", "ryan", "aiden", "ono_anna", "sohee"]

app = FastAPI()
_state = {"model": None, "error": None, "loaded": False}
_lock = threading.Lock()


def _load():
    global _state
    import torch
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    device = os.environ.get("QWEN3_TTS_DEVICE", "cuda:0")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    print(f"qwen3-tts: loading {MODEL_ID} on {device} ({dtype})", file=sys.stderr, flush=True)
    _state["model"] = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map=device, dtype=dtype)
    _state["loaded"] = True
    print("qwen3-tts: model ready", file=sys.stderr, flush=True)


def model():
    if _state["loaded"]:
        return _state["model"]
    with _lock:
        if not _state["loaded"] and _state["error"] is None:
            try:
                _load()
            except Exception:
                _state["error"] = traceback.format_exc()
                print("qwen3-tts: model load FAILED\n" + _state["error"], file=sys.stderr, flush=True)
    if _state["error"]:
        raise RuntimeError("model load failed: " + _state["error"].splitlines()[-1])
    return _state["model"]


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "loaded": _state["loaded"]}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}


@app.get("/v1/audio/voices")
def voices():
    return {"voices": [{"id": v, "name": v} for v in VOICES]}


@app.post("/v1/audio/speech")
async def speech(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    text = (body.get("input") or "").strip()
    if not text:
        return JSONResponse({"error": "input is required"}, status_code=400)
    voice = body.get("voice") or DEFAULT_VOICE
    language = body.get("language") or LANGUAGE
    print(f"qwen3-tts: synth voice={voice} chars={len(text)}", file=sys.stderr, flush=True)
    try:
        wavs, sr = model().generate_custom_voice(text=text, language=language, speaker=voice)
    except Exception as exc:
        print("qwen3-tts: synth failed: %r" % (exc,), file=sys.stderr, flush=True)
        return JSONResponse({"error": str(exc)}, status_code=500)
    audio = wavs[0]
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio), int(sr), format="WAV")
    return Response(content=buf.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("QWEN3_TTS_PORT", "8000")), log_level="info")
