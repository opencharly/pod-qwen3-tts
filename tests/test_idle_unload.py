# Copyright 2026 opencharly
# SPDX-License-Identifier: Apache-2.0
"""Drives the idle-unload thread: the checkpoint is released after the TTL.

Runs inside the image (the candy copies it to ~/.local/bin and a plan check
executes it). It fails if `_idle_unload_loop` / `start_idle_unload` / `_touch`
are removed, so it is the in-diff witness for the server's new behaviour.
"""

import importlib.util
import os
import time

os.environ["QWEN3_TTS_IDLE_UNLOAD_S"] = "1"

server = os.path.expanduser("~/.local/bin/qwen3-tts-server.py")
spec = importlib.util.spec_from_file_location("qwen3_tts_server", server)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

assert module.IDLE_UNLOAD_S == 1.0, module.IDLE_UNLOAD_S

with module._cond:
    module._state.update(
        model=object(), loaded=True, voices=["vivian"], last_used=time.monotonic() - 10
    )

module.start_idle_unload()

deadline = time.monotonic() + 10
while time.monotonic() < deadline:
    with module._cond:
        if not module._state["loaded"]:
            break
    time.sleep(0.1)

with module._cond:
    assert module._state["loaded"] is False and module._state["model"] is None, module._state

code, body = module.readiness(module._snapshot())
assert code == 503 and body["status"] == "unloaded", (code, body)

module._touch()
with module._cond:
    assert module._state["last_used"] > 0

print("qwen3-tts idle-unload test OK: released, readiness=503 unloaded")
