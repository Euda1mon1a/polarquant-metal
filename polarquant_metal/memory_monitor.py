"""
Adaptive memory pressure monitor for PolarQuant Metal server.

Polls vm.memory_pressure on a daemon thread and maintains a tier with
hysteresis: downgrades immediately on pressure increase, requires sustained
improvement before upgrading (avoids thrashing).

Usage:
    from polarquant_metal.memory_monitor import start_monitor, get_controller

    start_monitor()  # call once at server startup

    tier = get_controller().tier  # QuantTier(name, bits, bits_v) — thread-safe
"""

import subprocess
import threading
import time
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuantTier:
    name: str
    bits: Optional[int]    # None = FP16 (no compression)
    bits_v: Optional[int]  # V cache bits — None = same as bits


# K is more sensitive than V per TurboQuant paper.
# Under pressure, keep V at 4-bit even when K drops to 3-bit.
TIERS: dict[str, QuantTier] = {
    "normal":   QuantTier("normal",   bits=None, bits_v=None),
    "warn":     QuantTier("warn",     bits=4,    bits_v=4),
    "critical": QuantTier("critical", bits=3,    bits_v=4),
}

TIER_SEVERITY: dict[str, int] = {"normal": 0, "warn": 1, "critical": 2}

# Models known to degrade with PolarQuant KV compression.
_INCOMPATIBLE_MODEL_SUBSTRINGS = ("phi-4-mini", "phi4-mini")


def is_compatible_model(model_id: str) -> bool:
    """Return False for models known to degrade with PolarQuant KV compression."""
    lower = model_id.lower()
    return not any(s in lower for s in _INCOMPATIBLE_MODEL_SUBSTRINGS)


def get_memory_pressure() -> str:
    """Return 'normal', 'warn', or 'critical'.

    vm.memory_pressure sysctl values on macOS 26+: 0=normal, 1=warn, 4=critical.
    Pre-26 sources documented 4=normal, 2=warn, 1=critical — kept for compatibility.
    Falls back to 'normal' on any error so inference is never blocked.
    """
    try:
        result = subprocess.run(
            ["sysctl", "-n", "vm.memory_pressure"],
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        level = int(result.stdout.strip())
        # 0 and 4 both treated as normal (0 observed on macOS 26, 4 from legacy docs).
        return {0: "normal", 4: "normal", 2: "warn", 1: "critical"}.get(level, "warn")
    except Exception:
        return "normal"


class AdaptiveTierController:
    """Thread-safe tier controller with hysteresis.

    Downgrades immediately when pressure increases.
    Requires `hysteresis_s` seconds of sustained improvement before upgrading.
    """

    def __init__(self, hysteresis_s: float = 5.0, poll_interval_s: float = 2.0):
        self._current = "normal"
        self._last_change = 0.0
        self._hysteresis = hysteresis_s
        self._poll_interval = poll_interval_s
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Start the background polling thread. No-op if already running."""
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="mem-pressure-monitor"
        )
        self._thread.start()
        logger.info("[MemPressure] Monitor started (poll=%.1fs, hysteresis=%.1fs)",
                    self._poll_interval, self._hysteresis)

    def stop(self) -> None:
        """Signal the background thread to exit on its next poll cycle."""
        self._running = False

    def _run(self) -> None:
        while self._running:
            self._maybe_update(get_memory_pressure())
            time.sleep(self._poll_interval)

    def _maybe_update(self, observed: str) -> None:
        now = time.monotonic()
        with self._lock:
            if observed == self._current:
                return
            is_downgrade = TIER_SEVERITY[observed] > TIER_SEVERITY[self._current]
            elapsed = now - self._last_change
            if is_downgrade or elapsed >= self._hysteresis:
                logger.info("[MemPressure] %s → %s", self._current, observed)
                self._current = observed
                self._last_change = now

    @property
    def tier(self) -> QuantTier:
        """Return the current QuantTier. Thread-safe."""
        with self._lock:
            return TIERS[self._current]

    @property
    def tier_name(self) -> str:
        """Return the current tier name string. Thread-safe."""
        with self._lock:
            return self._current

    def force_tier(self, name: str) -> None:
        """Override the current tier. Intended for testing only."""
        if name not in TIERS:
            raise ValueError(f"Unknown tier: {name!r}. Must be one of {list(TIERS)}")
        with self._lock:
            logger.warning("[MemPressure] Tier force-set to %r (testing only)", name)
            self._current = name
            self._last_change = time.monotonic()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_controller: Optional[AdaptiveTierController] = None
_init_lock = threading.Lock()


def start_monitor(
    hysteresis_s: float = 5.0,
    poll_interval_s: float = 2.0,
) -> AdaptiveTierController:
    """Start the background monitor singleton. Safe to call multiple times."""
    global _controller
    with _init_lock:
        if _controller is not None:
            return _controller
        _controller = AdaptiveTierController(
            hysteresis_s=hysteresis_s,
            poll_interval_s=poll_interval_s,
        )
        _controller.start()
    return _controller


def get_controller() -> AdaptiveTierController:
    """Return the singleton controller, starting it if not yet started."""
    if _controller is None:
        return start_monitor()
    return _controller


# ---------------------------------------------------------------------------
# Pressure simulation for testing
# ---------------------------------------------------------------------------

def simulate_pressure(gigabytes: float = 20.0, duration_s: float = 30.0) -> threading.Thread:
    """Allocate anonymous memory to trigger macOS memory pressure for testing.

    macOS does not expose a way to set vm.memory_pressure artificially.
    Allocating ~20 GB of resident memory forces the system into warn/critical.
    The buffer is held for `duration_s` seconds, then released.
    Runs in a daemon thread so it doesn't block the caller.

    Example::

        from polarquant_metal.memory_monitor import simulate_pressure
        t = simulate_pressure(gigabytes=20.0, duration_s=30.0)
        # Watch server logs for tier change events
        t.join()  # optional: wait for release

    Args:
        gigabytes: How many GB to allocate. Default 20 GB is enough on a
                   64 GB M4 Pro to trigger at least 'warn'.
        duration_s: How long to hold the allocation before releasing.

    Returns:
        The daemon Thread running the allocation (for optional join).
    """
    size = int(gigabytes * 1024 ** 3)
    logger.info("[MemPressure] simulate_pressure: allocating %.1f GB for %.0fs", gigabytes, duration_s)

    def _alloc() -> None:
        buf = bytearray(size)
        # Touch every page to ensure resident allocation (not just virtual).
        # Write one byte per 4K page so the OS can't optimize away the alloc.
        for i in range(0, size, 4096):
            buf[i] = 0xFF
        logger.info("[MemPressure] simulate_pressure: allocation resident, sleeping %.0fs", duration_s)
        time.sleep(duration_s)
        del buf
        logger.info("[MemPressure] simulate_pressure: allocation released")

    t = threading.Thread(target=_alloc, daemon=True, name="pressure-sim")
    t.start()
    return t
