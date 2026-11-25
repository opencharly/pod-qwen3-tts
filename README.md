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
| `GET /health` | readiness: `200` only when the on-demand worker serves; `503` with `unloaded`, `waiting_for_vram`, `loading`, `unloading` or `error` otherwise |
| `GET /v1/models` | the served model id (answers `200` even while unloaded) |

The upstream `QwenLM/Qwen3-TTS` repo ships **no HTTP server** (a Gradio demo, a CLI and a python
API only), so this candy ships the server itself — a torch-free FastAPI shim that starts an
on-demand worker over the upstream `qwen-tts` package.

## Properties

| Property | Value |
|---|---|
| Service | `qwen3-tts` (`%(ENV_HOME)s/.local/bin/qwen3-tts-server.py`) |
| Port | `8000` (the worker listens on a Unix socket, never a published port) |
| Volume | `models` → `~/.cache/huggingface`; bind it to `<charly data root>/models/huggingface` (default `~/.local/share/charly/models/huggingface`) so every pod shares ONE cached checkpoint |
| Requires | `layer-python`, `layer-supervisord` |
| Packages | `gcc`, `gcc-c++` (torch/triton JIT) |
| Default model | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` |
| Response formats | `wav` (default), `pcm`, `flac`, `mp3` — whatever the installed libsndfile really writes; anything else is a `4xx` |
| Voices | the checkpoint's own speaker list; fallback: vivian, serena, uncle_fu, dylan, eric, ryan, aiden, ono_anna, sohee |
| `env_provide` | `QWEN3_TTS_HOST`, `QWEN3_TTS_SERVED_MODEL`, `QWEN3_TTS_VOICE` |
| VRAM lifecycle | nothing at start; a request waits up to `QWEN3_TTS_GPU_WAIT_SECONDS` (600) for `QWEN3_TTS_GPU_MIN_FREE_MIB` (4096) free, then starts the worker; the worker is killed after `QWEN3_TTS_IDLE_UNLOAD_S` (600) idle |

**The gate reads NVML (`libnvidia-ml.so.1`), the library the container runtime injects** — not the
`nvidia-smi` CLI, which the runtime does not always provide (measured: the library and
`/dev/nvidia0` were present in a deployment that had no `nvidia-smi`). The image must not bake
`nvidia-utils` or any other NVIDIA binary; the pod/driver contract fails such a build. A deploy
without a device answers `503 gpu_unavailable`.

## Quick start

```bash
# bind the model cache to the shared store in the XDG data root (see "Model cache" below)
charly config qwen3-tts --bind models="$HOME/.local/share/charly/models/huggingface"
charly start qwen3-tts
curl -s http://127.0.0.1:$(charly status qwen3-tts | grep -oE 'Ports: *[0-9]+' | grep -oE '[0-9]+$')/health
```

**Model cache — one shared store.** The checkpoint is Hugging Face cache data, so it belongs in the
charly data root the resolver owns (`spec/storeroot`, DESIGN D-STORE-1/11/12) under
`models/huggingface` — `~/.local/share/charly/models/huggingface` with the default XDG roots.
Binding the `models` volume there in every deployment and check bed means the checkpoint is
downloaded once and read by every qwen3-tts pod. The shim and the on-demand worker inherit
`HF_HOME`, so they both read and download into that same store. Do not keep the default per-image
named volume for this cache: it duplicates the checkpoint per pod.

**Nothing loads at service start.** `/health` reports `503 unloaded` and the pod holds no VRAM.
The first synthesis request waits (bounded) for free NVRAM, starts the on-demand worker, loads the
checkpoint into the `models` volume and answers. Concurrent requests wait on that one spawn; a
failed start is retried by the next request rather than wedging the service. After ten idle minutes
the shim kills the worker, which releases the weights **and the CUDA context**, and the next
request reloads.


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
