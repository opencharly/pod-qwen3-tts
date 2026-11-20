# pod-qwen3-tts

The `qwen3-tts` candy of the OpenCharly candy library, as a standalone repo (kind-prefixed
naming): a **Qwen3-TTS speech service** served over the OpenAI-compatible speech API that
Open WebUI (and any OpenAI-compatible client) calls.

## What it provides

A supervised HTTP service on port 8000:

| Route | Purpose |
|---|---|
| `POST /v1/audio/speech` | synthesise `{input, voice, language, response_format}` → audio bytes in the requested format |
| `GET /v1/audio/voices` | the checkpoint's own speaker list (the hardcoded preset list only until it loads) |
| `GET /health` | readiness: `200` only when the model can synthesize, `503` + a machine-readable reason while it cannot |
| `GET /v1/models` | the served model id (answers `200` even during a warm-up) |

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
| Response formats | `wav` (default), `pcm`, `flac`, `mp3` — whatever the installed libsndfile really writes; anything else is a `4xx` |
| Voices | the checkpoint's own speaker list; fallback: vivian, serena, uncle_fu, dylan, eric, ryan, aiden, ono_anna, sohee |
| `env_provide` | `QWEN3_TTS_HOST`, `QWEN3_TTS_SERVED_MODEL`, `QWEN3_TTS_VOICE` |

## Quick start

```bash
charly config qwen3-tts
charly start qwen3-tts
curl -s http://127.0.0.1:$(charly status qwen3-tts | grep -oE 'Ports: *[0-9]+' | grep -oE '[0-9]+$')/health
```

The checkpoint loads in a **background thread at service start** and is cached in the `models`
volume afterwards. `/v1/models` answers immediately while the model warms; `/health` is `503` with
`{"status":"loading"}` until the model can synthesize and `200 {"status":"ok"}` after. A request
that arrives during the warm-up waits for that one in-flight load instead of starting a second
one, and a failed load is retried by the next request rather than wedging the service.

## Using it from Open WebUI

This candy `env_provide`s what `pod-openwebui` `env_accept`s, so charly wires the two without any
hand-edited environment:

```bash
charly config qwen3-tts            # Env provides injected: QWEN3_TTS_HOST=http://charly-qwen3-tts:8000/v1
charly config openwebui --update-all
```

The consumer's entrypoint turns that into `AUDIO_TTS_ENGINE=openai` plus the matching base URL,
model and voice, and declares `{"response_format":"mp3"}` as a preference. The server honours it
for real (libsndfile in this image writes MP3), so the consumer's `.mp3`-named speech cache stays
coherent instead of holding a mislabelled WAV.

## Base image note

The shipped box (`opencharly/distro-cachyos`, `box/qwen3-tts`) bases on **`cachyos`**, not `nvidia`:
torch's cu130 wheels carry their own CUDA runtime and the driver arrives through CDI at runtime, so
a separate CUDA layer is unnecessary — and CachyOS is the distro the host runs. **Do not base on
`nvidia` today** — that base's `layer-cuda` pin predates the `cuda-cccl-devel` fix and fails to
build (opencharly/distro-fedora#110).

## Pin handles

The upstream `QwenLM/Qwen3-TTS` repo publishes **no releases and no tags**. This candy consumes
the org's fork `opencharly/Qwen3-TTS` at its CalVer tag (`v2026.324.1131` in `pixi.toml`; the
committed `pixi.lock` records the tag's commit), which carries the public-export fixes and the
honest 0.6B `instruct` warning. The checkpoint repo id remains
`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, overridable via `QWEN3_TTS_MODEL`. Torch is pinned
through the pytorch cu130 index at the same floor the org's `python-ml` layer uses.

## Verification

```bash
charly box validate                                  # candy + skill shape
charly check box <image composing this candy>        # the baked checks
```

The baked `check:` steps run in a container with no checkpoint and no GPU, and prove the honesty
contract from there: `/v1/models` answering while `/health` is `503`, the `4xx` for an unsupported
`response_format` and for a foreign `model`, one load attempt per failed load (retriable, not
wedged), at most one attempt per concurrent burst, and every advertised format encoding real bytes.

## Style instruction

`QWEN3_TTS_INSTRUCT` (deployment default) and a request's `instruct` field (per call) are forwarded to the model's `instruct` kwarg — free text describing the delivery, e.g. emotion, pace, character. See the skill for the contract.
