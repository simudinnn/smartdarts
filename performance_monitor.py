"""Small, dependency-free Raspberry Pi health/performance sampler."""

from __future__ import annotations

import os
import platform
import subprocess
import time
from typing import Dict, Optional, Tuple


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _vcgencmd(*args: str) -> str:
    try:
        result = subprocess.run(
            ["vcgencmd", *args],
            capture_output=True,
            text=True,
            timeout=0.8,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def decode_throttled(raw: str) -> Dict[str, object]:
    value = 0
    try:
        token = str(raw or "").split("=", 1)[-1].strip()
        value = int(token, 0)
    except (TypeError, ValueError):
        pass
    current = []
    history = []
    for bit, label in (
        (0, "Niski napon"),
        (1, "CPU takt ograničen"),
        (2, "Throttling"),
        (3, "Temperaturno ograničenje"),
    ):
        if value & (1 << bit):
            current.append(label)
        if value & (1 << (bit + 16)):
            history.append(label)
    if current:
        status = "AKTIVNO UPOZORENJE"
        detail = ", ".join(current)
        level = "bad"
    elif history:
        status = "TRENUTNO OK"
        detail = "Ranije zabilježeno: " + ", ".join(history)
        level = "warn"
    elif raw:
        status = "OK"
        detail = "Nema undervoltagea ni throttlinga"
        level = "good"
    else:
        status = "NEDOSTUPNO"
        detail = "vcgencmd nije dostupan"
        level = "neutral"
    return {"raw": value, "status": status, "detail": detail, "level": level}


class PerformanceCollector:
    def __init__(self) -> None:
        self._last_cpu: Optional[Tuple[int, int]] = None

    def _cpu_percent(self) -> Optional[float]:
        line = _read_text("/proc/stat").splitlines()
        if not line or not line[0].startswith("cpu "):
            return None
        try:
            nums = [int(v) for v in line[0].split()[1:]]
        except ValueError:
            return None
        idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
        total = sum(nums)
        previous = self._last_cpu
        self._last_cpu = (total, idle)
        if previous is None or total <= previous[0]:
            return None
        busy_delta = (total - previous[0]) - (idle - previous[1])
        return max(0.0, min(100.0, 100.0 * busy_delta / (total - previous[0])))

    @staticmethod
    def _memory() -> Tuple[Optional[int], Optional[int]]:
        values: Dict[str, int] = {}
        for line in _read_text("/proc/meminfo").splitlines():
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            try:
                values[key] = int(raw.strip().split()[0]) * 1024
            except (ValueError, IndexError):
                continue
        total = values.get("MemTotal")
        available = values.get("MemAvailable")
        if not total or available is None:
            return None, None
        return total - available, total

    @staticmethod
    def _temperature() -> Optional[float]:
        raw = _read_text("/sys/class/thermal/thermal_zone0/temp")
        try:
            value = float(raw)
            return value / 1000.0 if value > 200.0 else value
        except ValueError:
            fallback = _vcgencmd("measure_temp")
            try:
                return float(fallback.split("=", 1)[1].split("'", 1)[0])
            except (ValueError, IndexError):
                return None

    @staticmethod
    def _cpu_clock_mhz() -> Optional[float]:
        raw = _read_text(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
        )
        try:
            return float(raw) / 1000.0
        except ValueError:
            fallback = _vcgencmd("measure_clock", "arm")
            try:
                return float(fallback.split("=", 1)[1]) / 1_000_000.0
            except (ValueError, IndexError):
                return None

    def collect(self) -> Dict[str, object]:
        used, total = self._memory()
        gpu_clock_raw = _vcgencmd("measure_clock", "core")
        try:
            gpu_clock = float(gpu_clock_raw.split("=", 1)[1]) / 1_000_000.0
        except (ValueError, IndexError):
            gpu_clock = None
        gpu_mem = _vcgencmd("get_mem", "gpu")
        throttled_raw = _vcgencmd("get_throttled")
        uptime = None
        try:
            uptime = float(_read_text("/proc/uptime").split()[0])
        except (ValueError, IndexError):
            pass
        return {
            "cpu_percent": self._cpu_percent(),
            "cpu_clock_mhz": self._cpu_clock_mhz(),
            "temperature_c": self._temperature(),
            "memory_used": used,
            "memory_total": total,
            "gpu_clock_mhz": gpu_clock,
            "gpu_memory": gpu_mem.split("=", 1)[-1] if gpu_mem else "",
            "throttled": decode_throttled(throttled_raw),
            "uptime_seconds": uptime,
            "platform": platform.machine() or "—",
            "sampled_at": time.time(),
        }
