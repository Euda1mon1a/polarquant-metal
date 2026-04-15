#!/usr/bin/env python3
"""
Memory pressure watchdog for PolarQuant benchmarks.

Monitors vm.memory_pressure and Metal GPU wired memory on a background thread.
Sends SIGTERM to the target process when pressure becomes critical, giving it
a chance to flush logs before the OS panics.

Usage:
    # Start watchdog before launching benchmark:
    python3 benchmarks/watchdog.py --pid <benchmark_pid> &

    # Or import and use inline:
    from benchmarks.watchdog import start_watchdog
    start_watchdog(os.getpid())
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time

# Kill the watched process if vm.memory_pressure hits this level.
# macOS 26: 0=normal, 2=warn, 1=critical
KILL_ON_PRESSURE = 1  # critical only

# Also kill if Metal GPU wired pages exceed this many GB.
# M4 Pro has 24GB GPU wired limit; 35B model uses ~18GB.
# Set conservatively to catch runaway allocations.
KILL_ON_GPU_GB = 58.0  # ~90% of 64GB total unified memory

# Kill if MLX active + cache memory exceeds this many GB (0 = disabled).
# More precise than vm_stat for MLX workloads — tracks only our allocations.
# Set to 0.0 to disable (e.g. when running FP16 benchmarks that need full RAM).
KILL_ON_MLX_GB = 0.0

POLL_INTERVAL = 3.0  # seconds


def get_memory_pressure() -> int:
    try:
        r = subprocess.run(["sysctl", "-n", "vm.memory_pressure"],
                           capture_output=True, text=True, timeout=2)
        return int(r.stdout.strip())
    except Exception:
        return 0


def get_wired_gb() -> float:
    """Return wired (kernel + GPU) memory in GB from vm_stat."""
    try:
        r = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=2)
        for line in r.stdout.splitlines():
            if "wired down" in line.lower():
                pages = int(line.split(":")[1].strip().rstrip("."))
                return pages * 16384 / 1024 ** 3
    except Exception:
        pass
    return 0.0


def get_mlx_gb() -> float:
    """Return MLX active + cache memory in GB.

    Uses mx.get_active_memory() + mx.get_cache_memory() (MLX 0.31+).
    More precise than vm_stat for detecting runaway KV cache growth —
    tracks only MLX Metal allocations, not OS or other processes.
    Returns 0.0 if mlx is not importable in this thread context.
    """
    try:
        import mlx.core as mx
        return (mx.get_active_memory() + mx.get_cache_memory()) / 1024 ** 3
    except Exception:
        return 0.0


def get_swap_gb() -> float:
    try:
        r = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                           capture_output=True, text=True, timeout=2)
        # Format: total = Xm  used = Ym  free = Zm
        for part in r.stdout.split():
            if part.endswith("M") and "used" not in r.stdout.split()[r.stdout.split().index(part)-1:]:
                pass
        # Simpler parse
        import re
        m = re.search(r"used\s*=\s*([\d.]+)M", r.stdout)
        if m:
            return float(m.group(1)) / 1024
    except Exception:
        pass
    return 0.0


def watch(pid: int, log_file: str = None):
    """Block until process exits or memory thresholds are breached."""
    out = open(log_file, "a") if log_file else sys.stderr

    def log(msg):
        ts = time.strftime("%H:%M:%S")
        print(f"[watchdog {ts}] {msg}", file=out, flush=True)

    mlx_limit = f"{KILL_ON_MLX_GB}GB" if KILL_ON_MLX_GB > 0 else "disabled"
    log(f"Watching PID {pid} | kill_pressure={KILL_ON_PRESSURE} | kill_gpu={KILL_ON_GPU_GB}GB | kill_mlx={mlx_limit}")

    while True:
        # Check if process is still alive
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            log(f"PID {pid} exited cleanly.")
            break

        pressure = get_memory_pressure()
        wired_gb = get_wired_gb()
        swap_gb = get_swap_gb()
        mlx_gb = get_mlx_gb() if KILL_ON_MLX_GB > 0 else 0.0

        mlx_str = f" mlx={mlx_gb:.1f}GB" if KILL_ON_MLX_GB > 0 else ""
        log(f"pressure={pressure} wired={wired_gb:.1f}GB swap={swap_gb:.2f}GB{mlx_str}")

        if pressure == KILL_ON_PRESSURE:
            log(f"CRITICAL pressure ({pressure}) — sending SIGTERM to {pid}")
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            time.sleep(8)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            log("Process killed. GPU panic averted (hopefully).")
            break

        if KILL_ON_MLX_GB > 0 and mlx_gb >= KILL_ON_MLX_GB:
            log(f"MLX memory {mlx_gb:.1f}GB >= {KILL_ON_MLX_GB}GB — sending SIGTERM to {pid}")
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            time.sleep(8)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            log("Process killed.")
            break

        if wired_gb >= KILL_ON_GPU_GB:
            log(f"GPU wired memory {wired_gb:.1f}GB >= {KILL_ON_GPU_GB}GB — sending SIGTERM to {pid}")
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            time.sleep(8)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            log("Process killed.")
            break

        time.sleep(POLL_INTERVAL)

    if log_file:
        out.close()


def start_watchdog(pid: int, log_file: str = None) -> threading.Thread:
    """Start watchdog on a daemon thread. Returns the thread."""
    t = threading.Thread(target=watch, args=(pid, log_file), daemon=True,
                         name="benchmark-watchdog")
    t.start()
    return t


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory pressure watchdog")
    parser.add_argument("--pid", type=int, required=True, help="PID to watch")
    parser.add_argument("--log", default=None, help="Log file path (default: stderr)")
    args = parser.parse_args()
    watch(args.pid, args.log)
