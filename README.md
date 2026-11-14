# pod-qwen3-tts

The `qwen3-tts` candy of the OpenCharly candy library, as a standalone repo (kind-prefixed
naming): a **Qwen3-TTS speech service** served over the OpenAI-compatible speech API that
Open WebUI (and any OpenAI-compatible client) calls.

## What it provides

A supervised HTTP service on port 8000:

| Route | Purpose |
|---|---|
| `POST /v1/audio/speech` | synthesise `{input, voice, language}` → `audio/wav` bytes |
| `GET /v1/audio/voices` | the preset voice list Open WebUI's picker reads |
| `GET /health` | liveness + whether the checkpoint is loaded |
| `GET /v1/models` | the served model id |

The upstream `QwenLM/Qwen3-TTS` repo ships **no HTTP server** (a Gradio demo, a CLI and a python
API only), so this candy ships the server itself — a small FastAPI shim over the upstream
`qwen-tts` package.

## Properties

| Property | Value |
|---|---|
| Service | `qwen3-tts` (`%(ENV_HOME)s/.local/bin/qwen3-tts-server.py`) |
| Port | `8000` |
| Volume | `models` → `~/.cache/huggingface` |
| Requires | `layer-python`, `layer-supervisord` |
| Packages | `gcc`, `gcc-c++` (torch/triton JIT) |
| Default model | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` |
| Voices | vivian, serena, uncle_fu, dylan, eric, ryan, aiden, ono_anna, sohee |
| `env_provide` | `QWEN3_TTS_HOST`, `QWEN3_TTS_SERVED_MODEL`, `QWEN3_TTS_VOICE` |

## Quick start

```bash
charly config qwen3-tts
charly start qwen3-tts
curl -s http://127.0.0.1:$(charly status qwen3-tts | grep -oE 'Ports: *[0-9]+' | grep -oE '[0-9]+$')/health
```

The checkpoint downloads on the **first synthesis request** (lazy load) and is cached in the
`models` volume afterwards.

## Using it from Open WebUI

This candy `env_provide`s what `pod-openwebui` `env_accept`s, so charly wires the two without any
hand-edited environment:

```bash
charly config qwen3-tts            # Env provides injected: QWEN3_TTS_HOST=http://charly-qwen3-tts:8000/v1
charly config openwebui --update-all
```

The consumer's entrypoint turns that into `AUDIO_TTS_ENGINE=openai` plus the matching base URL,
model and voice.

## Base image note

Build on **`fedora`**, not `nvidia`: the `nvidia` base's `layer-cuda` pin predates the
`cuda-cccl-devel` fix (opencharly/distro-fedora#110), and torch's cu130 wheels plus CDI at runtime
make a separate CUDA layer unnecessary.

## Pin handles

The upstream `QwenLM/Qwen3-TTS` repo publishes **no releases and no tags**, so this candy pins the
two handles that DO exist: the PyPI distribution (`qwen-tts==0.1.1` in `pixi.toml`) and the
checkpoint repo id (`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, overridable via `QWEN3_TTS_MODEL`).
Torch is pinned through the pytorch cu130 index at the same floor the org's `python-ml` layer uses.

## Verification

```bash
charly box validate                                  # candy + skill shape
charly check box <image composing this candy>        # the baked checks
```
