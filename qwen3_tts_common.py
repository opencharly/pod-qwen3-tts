#!/usr/bin/env python3
"""Shared constants and helpers for the qwen3-tts shim and worker.

Imported by BOTH processes: the public FastAPI shim (which must never import
torch) and the on-demand worker (torch + qwen_tts). Nothing here imports torch
or qwen_tts, so importing this module never creates a CUDA context — the
property that lets the idle unload release ALL of the pod's VRAM.
"""
import io
import sys

import numpy as np
import soundfile as sf

MODEL_ID_DEFAULT = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_VOICE = "vivian"

# The preset voices of a CustomVoice checkpoint. Used ONLY while the checkpoint
# has not handed us its own list — before the load finishes and after a failed
# load — and a fallback is always announced on stderr, never silent.
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


def probe_encodings():
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


SUPPORTED_ENCODINGS = probe_encodings()
SUPPORTED_FORMATS = sorted(SUPPORTED_ENCODINGS)


def encode_audio(audio, sample_rate, format_name):
    """Encode one channel of model output into the requested response format."""
    fmt, subtype, _ = SUPPORTED_ENCODINGS[format_name]
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio), int(sample_rate), format=fmt, subtype=subtype)
    return buf.getvalue()


def media_type(format_name):
    """The response Content-Type for a supported format."""
    return SUPPORTED_ENCODINGS[format_name][2]
