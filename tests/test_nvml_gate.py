#!/usr/bin/env python3
"""The free-VRAM gate must read NVML, and it must read it through the library.

The gate used to exec the `nvidia-smi` CLI. A deployed pod gets the driver
LIBRARY and the device nodes from the container runtime, but not necessarily the
CLI, so the gate refused every synthesis in exactly that shape. These checks pin
the replacement:

* the diagnostic override answers without touching NVML (so the surface tests can
  pin a reading on any host);
* a non-numeric override is refused, never silently treated as 0;
* the NVML path returns a real free-memory figure, cross-checked against
  `nvidia-smi` when that CLI happens to exist. Where the library or a device is
  absent the NVML case SKIPS VISIBLY (R8): a CPU-only host has no GPU to measure.
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import traceback

BIN = os.path.expanduser("~/.local/bin")
sys.path.insert(0, BIN)

failures = []


def load(name, env):
    """Load a fresh copy of the shim with `env` applied (config is read at import)."""
    saved = dict(os.environ)
    os.environ.update(env)
    for key in ("QWEN3_TTS_GPU_FREE_MIB",):
        if key not in env:
            os.environ.pop(key, None)
    try:
        spec = importlib.util.spec_from_file_location(name, os.path.join(BIN, "qwen3-tts-server.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        os.environ.clear()
        os.environ.update(saved)


def check(label, ok, detail=None):
    print("%s %s" % ("PASS" if ok else "FAIL", label))
    if not ok:
        if detail is not None:
            print("     %r" % (detail,))
        failures.append(label)


# 1. The override answers verbatim, and never loads NVML.
m = load("qwen3_tts_override", {"QWEN3_TTS_GPU_FREE_MIB": "1234"})
check("the diagnostic override answers the pinned reading", m._free_vram_mib() == 1234, m._free_vram_mib())
check("the override never touched NVML", m._nvml_lib_cache is None, m._nvml_lib_cache)

# 2. A non-numeric override is a named refusal, not a silent 0.
m = load("qwen3_tts_bad_override", {"QWEN3_TTS_GPU_FREE_MIB": "lots"})
try:
    m._free_vram_mib()
    check("a non-numeric override is refused", False, "no exception")
except m.GpuUnavailable as exc:
    check("a non-numeric override is refused", "not a number of MiB" in str(exc), str(exc))

# 3. The NVML path itself.
m = load("qwen3_tts_nvml", {})
try:
    free = m._free_vram_mib()
except m.GpuUnavailable as exc:
    print("SKIP the NVML reading is unavailable here: %s" % (exc,))
else:
    check("NVML reports a positive free-memory reading", isinstance(free, int) and free > 0, free)
    smi = shutil.which("nvidia-smi")
    if smi is None:
        print("SKIP the nvidia-smi cross-check: no CLI on this host (the case this fix is for)")
    else:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"],
                capture_output=True, text=True, timeout=20,
            )
            reading = int(float(out.stdout.strip().splitlines()[0]))
        except Exception:
            traceback.print_exc()
            check("the nvidia-smi cross-check ran", False, out.stderr if "out" in dir() else None)
        else:
            check(
                "NVML agrees with nvidia-smi within 1 GiB",
                abs(reading - free) <= 1024,
                (free, reading),
            )

if failures:
    print("%d check(s) failed" % len(failures))
    sys.exit(1)
print("the NVML gate holds")
