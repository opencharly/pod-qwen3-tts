# AGENTS.md — pod-qwen3-tts

Standalone candy repo for the `qwen3-tts` candy — a Qwen3-TTS speech service on the
OpenAI-compatible speech API, served as a supervised HTTP service on port `8000`.

Canonical files:

- `charly.yml` — the `qwen3-tts:` candy entity (description, `require`, `package`, `env`,
  `env_provide`, `port`, `volume`, `service`, `plan`) and its routing-only `skill:` node.
- `qwen3-tts-server.py` — the charly-owned FastAPI shim over the upstream `qwen-tts` API.
- `pixi.toml` — the python environment (torch cu130 + `qwen-tts` + fastapi/uvicorn/soundfile).
- `.agents/skills/qwen3-tts/SKILL.md` — the owning skill's body (the `skill:` node is routing only).
- `.github/workflows/tag-on-merge.yml` — CalVer tag + `CHANGELOG/` on merge.
- `CHANGELOG/` — per-CalVer history.

## Load these skills first (R0)

- `/charly-qwen3-tts:qwen3-tts` — the owning skill (contract, knobs, verification).
- `/charly-openwebui:openwebui` — the consumer side that `env_accept`s this candy's provides.
- `/charly-image:layer` — candy authoring (`env_provide`, `service:`, `volume:`).
- `/charly-check:check` — the check/ADE framework for the baked `check:` steps.
- `/charly-internals:git-workflow` — before any git/PR action.

## Build / validate / test

- `charly box validate` at the repo root — structural check of the candy and its skill node.
- The live witness is a composing box's check bed; the candy's own `check:` steps assert the
  server script, and that the pixi env imports torch, soundfile and `qwen_tts`.
- The merge gate is the **org-wide** `charly/pr-validator` (required check `validate / validate`,
  defined in `opencharly/.github`); this repo has **no** per-repo candy gate.

## Modify this repo

- Edit the `qwen3-tts:` candy entity and `qwen3-tts-server.py` together — the server is the thing
  the service runs.
- Python dependency changes belong in `pixi.toml`, not in the plan.
- `env_provide` names are a CONTRACT with consumers: renaming one breaks `pod-openwebui`'s
  `env_accept` silently (the provide is filtered out and nothing is wired).
- The skill body lives in `.agents/skills/qwen3-tts/SKILL.md`; never inline it in `charly.yml`
  (an inline body that disagrees with the file is a hard error in the marketplace generator).

## Landing

- PR-only. Every change lands through a pull request; the org-required `charly/pr-validator`
  validates the diff and body and arms native auto-merge on PASS.
- History lives in `CHANGELOG/` (written by `tag-on-merge` at merge time); the PR body IS the
  changelog.
- The authoritative rulebook is the umbrella `AGENTS.md` in `opencharly/opencharly`. Do not
  restate its rules here.
