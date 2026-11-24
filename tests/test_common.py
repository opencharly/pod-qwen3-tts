#!/usr/bin/env python3
"""Assert the pure contracts of the common module, the worker and the shim.

No checkpoint and no GPU: the encoding table is measured, the voice resolution
is driven by fake models, and the shim's readiness/voice state machine is
exercised directly.
"""
import importlib.util
import inspect
import json
import os
import sys

import numpy as np

BIN = os.path.expanduser("~/.local/bin")
sys.path.insert(0, BIN)

import qwen3_tts_common as common  # noqa: E402


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(BIN, filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


worker = load_module("qwen3_tts_worker", "qwen3-tts-worker.py")
shim = load_module("qwen3_tts_server", "qwen3-tts-server.py")

failures = []


def check(name, condition, detail=""):
    if condition:
        print("PASS " + name)
    else:
        print("FAIL " + name + " :: " + repr(detail))
        failures.append(name)


container = {
    "wav": lambda data: data[:4] == b"RIFF",
    "flac": lambda data: data[:4] == b"fLaC",
    "mp3": lambda data: len(data) > 1 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0,
    "pcm": lambda data: len(data) == 4800,
}

check(
    "the advertised formats are the ones this environment can produce",
    set(common.SUPPORTED_FORMATS) >= {"wav", "pcm", "flac", "mp3"},
    common.SUPPORTED_FORMATS,
)
audio = np.zeros(2400, dtype="int16")
for name in common.SUPPORTED_FORMATS:
    payload = common.encode_audio(audio, 24000, name)
    check(
        "response_format " + repr(name) + " encodes bytes in its own container",
        bool(payload) and container[name](payload),
        (name, len(payload), payload[:4].hex()),
    )
check("the media type follows the format", common.media_type("wav") == "audio/wav", common.media_type("wav"))


class FakeModel:
    def __init__(self, speakers=None, boom=False):
        self._speakers = speakers
        self._boom = boom

    def get_supported_speakers(self):
        if self._boom:
            raise RuntimeError("this checkpoint exposes no speaker table")
        return self._speakers


voices, source = worker._resolve_voices(FakeModel(["vivian", "ono_anna"]))
check(
    "the voice list is derived from the checkpoint's own speaker table",
    voices == ["vivian", "ono_anna"] and source == "model",
    (voices, source),
)
voices, source = worker._resolve_voices(FakeModel(boom=True))
check(
    "a checkpoint that exposes no speaker table falls back to the preset list and says so",
    source == "fallback" and sorted(voices) == sorted(common.FALLBACK_VOICES),
    (voices, source),
)
check(
    "the worker's renderer accepts a per-request style instruction",
    "instruct" in inspect.signature(worker._render).parameters,
    str(inspect.signature(worker._render)),
)

code, body = shim.readiness({"loaded": True, "phase": "ready", "voices": ["vivian"], "voices_source": "model"})
check("/health answers 200 once the worker serves", code == 200 and body.get("status") == "ok", (code, body))
code, body = shim.readiness({"loaded": False, "phase": "starting"})
check("/health answers 503 loading while the worker starts", code == 503 and body.get("status") == "loading", (code, body))
code, body = shim.readiness({"loaded": False, "phase": "waiting_for_vram", "vram_free_mib": 1024})
check(
    "/health answers 503 waiting_for_vram with the last reading",
    code == 503 and body.get("status") == "waiting_for_vram" and body.get("vram_free_mib") == 1024,
    (code, body),
)
code, body = shim.readiness({"loaded": False, "phase": "error", "detail": "no GPU with >= 4096 MiB free"})
check(
    "/health answers 503 error and carries the reason",
    code == 503 and body.get("status") == "error" and "4096" in (body.get("detail") or ""),
    (code, body),
)
code, body = shim.readiness({"loaded": False, "phase": "unloaded"})
check("/health answers 503 unloaded in the idle state", code == 503 and body.get("status") == "unloaded", (code, body))
code, body = shim.readiness({"loaded": False, "phase": "stopping"})
check("/health answers 503 unloading while the worker is released", code == 503 and body.get("status") == "unloading", (code, body))

payload = shim.voices_payload({"loaded": True, "voices_source": "model", "voices": ["vivian", "sohee"]})
check(
    "the voices route serves the checkpoint's list once it is known, and names the source",
    [v["id"] for v in payload["voices"]] == ["vivian", "sohee"] and payload["source"] == "model",
    payload,
)
payload = shim.voices_payload({"loaded": False})
check(
    "the voices route serves the preset fallback before the checkpoint loads, and names it",
    len(payload["voices"]) == len(common.FALLBACK_VOICES) and payload["source"] == "fallback",
    payload["source"],
)

loaded_state = {"loaded": True, "voices_source": "model", "voices": ["vivian", "ono_anna"]}
refusal = shim.voice_validation(loaded_state, "no-such-voice")
check(
    "a voice the checkpoint does not serve is refused with 400 unsupported_voice",
    refusal is not None and refusal.status_code == 400
    and json.loads(refusal.body)["error"]["code"] == "unsupported_voice",
    None if refusal is None else (refusal.status_code, refusal.body[:160]),
)
check("a voice the checkpoint serves is accepted", shim.voice_validation(loaded_state, "ono_anna") is None)
check(
    "voice validation stays out of the way until the checkpoint's own list is known",
    shim.voice_validation({"loaded": False}, "anything") is None,
)

if failures:
    print("FAILED %d assertion(s): %s" % (len(failures), failures))
    sys.exit(1)
print("formats, voice contract, readiness and state machine hold")
