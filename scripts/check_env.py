"""Environment checker for llm-quant-bench.

Cross-platform, stdlib-only (so it runs in CI before dependencies are
installed). Degrades to WARN instead of crashing when there is no GPU or no
Ollama, which is exactly what happens on a CI runner.

Usage:
    python scripts/check_env.py
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

# Windows consoles do not parse ANSI escapes until a VT-mode init happens.
# Calling os.system with an empty string triggers that init as a side effect.
if os.name == "nt":
    os.system("")

GREEN, YELLOW, RED, DIM, RESET = (
    "\033[92m",
    "\033[93m",
    "\033[91m",
    "\033[90m",
    "\033[0m",
)

_counts = {"OK": 0, "WARN": 0, "FAIL": 0}


def report(status: str, label: str, detail: str = "") -> None:
    color = {"OK": GREEN, "WARN": YELLOW, "FAIL": RED}[status]
    _counts[status] += 1
    print(f"  {color}[{status:<4}]{RESET} {label:<26} {DIM}{detail}{RESET}")


def section(title: str) -> None:
    print(f"\n{title}")
    print("  " + "-" * 68)


def run(cmd: list[str], timeout: int = 15) -> str | None:
    """Run a command, return stdout or None.

    encoding is forced to utf-8 with errors="replace": a Chinese Windows
    defaults to GBK and will raise UnicodeDecodeError halfway through
    nvidia-smi output otherwise.
    """
    if shutil.which(cmd[0]) is None:
        return None
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


# --------------------------------------------------------------------------


def check_python() -> None:
    section("Python")
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= (3, 10):
        report("OK", "version", f"{ver} ({platform.machine()})")
    else:
        report("FAIL", "version", f"{ver} --需要 >= 3.10")

    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv:
        report("OK", "virtualenv", sys.prefix)
    else:
        report("WARN", "virtualenv", "未激活 -- 先跑 .venv\\Scripts\\activate.bat")

    try:
        import llm_quant_bench  # noqa: F401

        report("OK", "package importable", "llm_quant_bench")
    except ImportError:
        report("WARN", "package importable", ' 未安装 -- uv pip install -e ".[dev]"')


def check_memory() -> None:
    section("Memory & Disk")
    if os.name == "nt":

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        # wmic is removed from recent Windows 11 builds; ctypes always works.
        stat = MemoryStatusEx()
        stat.dwLength = ctypes.sizeof(MemoryStatusEx)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        total = stat.ullTotalPhys / 1024**3
        avail = stat.ullAvailPhys / 1024**3
        detail = f"{total:.1f} GB total, {avail:.1f} GB free"
        report("OK" if total >= 16 else "WARN", "RAM", detail)
    else:
        report("WARN", "RAM", "非 Windows,跳过")

    for drive in ("C:\\", "D:\\") if os.name == "nt" else ("/",):
        try:
            usage = shutil.disk_usage(drive)
        except OSError:
            continue
        free = usage.free / 1024**3
        status = "OK" if free >= 50 else "WARN"
        report(status, f"disk {drive}", f"{free:.1f} GB free")


def check_gpu() -> None:
    section("GPU")
    out = run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out or not out.strip():
        report("WARN", "nvidia-smi", "未找到 GPU -- CI 环境下正常")
        return

    line = out.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        report("WARN", "nvidia-smi", f"输出无法解析: {line}")
        return

    name, total_mb, used_mb, driver = parts[0], parts[1], parts[2], parts[3]
    total_gb = float(total_mb) / 1024
    report("OK", "device", f"{name} (driver {driver})")
    detail = f"{total_gb:.2f} GiB total, {float(used_mb) / 1024:.2f} GiB in use"
    if total_gb < 10:
        report("WARN", "VRAM", f"{detail} -- 14B 及以上会触发 CPU 卸载")
    else:
        report("OK", "VRAM", detail)


def check_ollama() -> None:
    section("Ollama")
    ver = run(["ollama", "--version"])
    if ver:
        report("OK", "binary", ver.strip())
    else:
        report("FAIL", "binary", "未安装或不在 PATH")
        return

    models_dir = os.environ.get("OLLAMA_MODELS")
    if models_dir and os.path.isdir(models_dir):
        report("OK", "OLLAMA_MODELS", models_dir)
    elif models_dir:
        report("WARN", "OLLAMA_MODELS", f"{models_dir} 不存在")
    else:
        report("WARN", "OLLAMA_MODELS", "未设置,默认落在 C 盘")

    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as resp:
            models = json.load(resp).get("models", [])
        names = ", ".join(sorted(m["name"] for m in models)[:4])
        report("OK", "API :11434", f"{len(models)} 个模型 ({names}...)")
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        report("WARN", "API :11434", "服务未响应 -- 从开始菜单启动 Ollama")


def check_runtime_vars() -> None:
    """The four variables that decide whether Day 6 data is reproducible.

    If a benchmark result looks wrong later, this is the first place to look.
    """
    section("Ollama runtime variables")
    expected = {
        "OLLAMA_NUM_PARALLEL": ("1", "未设置时 Ollama 在 1 和 4 之间自选,KV Cache 差 4 倍"),
        "OLLAMA_KEEP_ALIVE": ("30m", "默认 5m,压测中途卸载会污染数据"),
        "OLLAMA_FLASH_ATTENTION": ("0", "开启后 KV Cache 布局改变,公式对不上"),
        "OLLAMA_KV_CACHE_TYPE": ("f16", "量化后每元素不足 2 字节,估算器会系统性偏大"),
    }
    for var, (want, why) in expected.items():
        got = os.environ.get(var)
        if got is None:
            report("WARN", var, f"未设置 (期望 {want}) -- {why}")
        elif got.strip().lower() == want.lower():
            report("OK", var, got)
        else:
            report("WARN", var, f"{got} (期望 {want}) -- {why}")

    if os.name == "nt":
        print(
            f"  {DIM}提示: setx 之后必须重开 CMD 窗口,并从托盘退出后重启 "
            f"Ollama 服务,否则两边都读不到新值。{RESET}"
        )


def main() -> int:
    print(f"\n{'=' * 72}")
    print("  llm-quant-bench 环境检查")
    print(f"  {platform.platform()}")
    print("=" * 72)

    check_python()
    check_memory()
    check_gpu()
    check_ollama()
    check_runtime_vars()

    print(f"\n{'=' * 72}")
    summary = (
        f"  {GREEN}OK {_counts['OK']}{RESET}   "
        f"{YELLOW}WARN {_counts['WARN']}{RESET}   "
        f"{RED}FAIL {_counts['FAIL']}{RESET}"
    )
    print(summary)
    print(f"{'=' * 72}\n")

    return 1 if _counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
