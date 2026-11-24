#!/usr/bin/env python3
"""The shim must never import torch or qwen_tts.

That is the property that makes the idle unload total: the public process holds
no CUDA context to leak, so killing the worker returns every MiB. A later edit
that imports torch here must fail this check.
"""
import importlib.util
import os
import sys

BIN = os.path.expanduser("~/.local/bin")
sys.path.insert(0, BIN)

spec = importlib.util.spec_from_file_location("qwen3_tts_server", os.path.join(BIN, "qwen3-tts-server.py"))
module = importlib.util.module_from_spec(spec)
sys.modules["qwen3_tts_server"] = module
spec.loader.exec_module(module)

leaked = [name for name in ("torch", "qwen_tts") if name in sys.modules]
if leaked:
    print("FAIL the shim imported %s — the idle unload would leave a CUDA context behind" % ", ".join(leaked))
    sys.exit(1)
print("shim is torch-free: no torch, no qwen_tts in sys.modules")
