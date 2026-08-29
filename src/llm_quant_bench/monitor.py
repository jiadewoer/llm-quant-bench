"""Observe what Ollama actually did with a model.

Reads /api/ps rather than parsing `ollama ps` text output. The JSON gives
`size` and `size_vram` in bytes, and the PROCESSOR column you see in the
terminal is just size_vram/size rendered as a percentage -- so this is the
same number without the fragile text parsing.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass

import httpx

OLLAMA = "http://localhost:11434"
GIB = 1024**3


@dataclass(frozen=True)
class LoadedModel:
    name: str
    size_bytes: int
    size_vram_bytes: int

    @property
    def size_gb(self) -> float:
        return self.size_bytes / GIB

    @property
    def vram_gb(self) -> float:
        return self.size_vram_bytes / GIB

    @property
    def gpu_ratio(self) -> float:
        """Fraction resident on GPU. This is the PROCESSOR column."""
        return self.size_vram_bytes / self.size_bytes if self.size_bytes else 0.0

    def __str__(self) -> str:
        return f"{self.name} {self.size_gb:.2f}GB {self.gpu_ratio * 100:.0f}% GPU"


def ps(timeout: float = 10.0) -> list[LoadedModel]:
    """Currently loaded models. Empty list if nothing is loaded."""
    resp = httpx.get(f"{OLLAMA}/api/ps", timeout=timeout)
    resp.raise_for_status()
    return [
        LoadedModel(m["name"], m.get("size", 0), m.get("size_vram", 0))
        for m in resp.json().get("models", [])
    ]


def find_loaded(model: str) -> LoadedModel | None:
    """Ollama appends :latest to bare names, so match on the stem too."""
    stem = model.split(":")[0]
    for m in ps():
        if m.name == model or m.name.split(":")[0] == stem:
            return m
    return None


def stop_model(model: str, timeout: float = 30.0) -> None:
    """Force an unload so the next load applies a new num_ctx.

    Without this, Ollama keeps serving the already-loaded instance and
    silently ignores a changed num_ctx -- you get an identical column of
    numbers with no error to explain it.
    """
    try:
        httpx.post(
            f"{OLLAMA}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=60.0,
        )
    except httpx.HTTPError:
        pass

    deadline = time.time() + timeout
    while time.time() < deadline:
        if find_loaded(model) is None:
            return
        time.sleep(0.5)


def gpu_used_mib() -> float | None:
    """Total VRAM in use right now, or None when there is no NVIDIA GPU."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    return float(out.stdout.strip().splitlines()[0])


def gpu_total_mib() -> float | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    return float(out.stdout.strip().splitlines()[0])


def baseline_gb() -> float | None:
    """VRAM held by everything other than Ollama.

    Call with no model loaded. Subtract this from the card's capacity to see
    what Ollama has to work with. Measured on 2026-08-29 it ranged 1.13-1.75
    GiB depending on what was open, which is enough variation to change
    which layers get offloaded -- so record it alongside every measurement.
    """
    used = gpu_used_mib()
    return None if used is None else used / 1024
