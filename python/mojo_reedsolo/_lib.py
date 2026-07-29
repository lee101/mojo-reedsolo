from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "reedsolo.mojo")
LIB = os.environ.get("MOJO_REEDSOLO_LIB") or os.path.join(
    ROOT, "dist", "libmojo-reedsolo.so"
)
I = ctypes.c_int64

_SIGNATURES = {
    "mrs_encode": ([I] * 7, I),
    "mrs_syndromes": ([I] * 8, I),
    "mrs_poly_mul": ([I] * 7, I),
    "mrs_find_errors": ([I] * 7, I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_REEDSOLO_LIB"):
        if os.path.isfile(LIB):
            return LIB
        raise BuildError(f"MOJO_REEDSOLO_LIB does not name a file: {LIB}")
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SRC):
        return LIB
    mojo = shutil.which("mojo")
    if not mojo:
        raise BuildError("mojo not found; run inside the Pixi environment")
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_handle: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _handle
    if _handle is None:
        _handle = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_handle, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _handle
