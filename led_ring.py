"""GPIO power control for the dartboard LED ring relay."""

from __future__ import annotations

import atexit
import shutil
import subprocess
import threading
from typing import Optional


LED_RING_GPIO = 17


class LedRingPower:
    """Drive an active-high relay and remain harmless away from Raspberry Pi."""

    def __init__(self, pin: int = LED_RING_GPIO) -> None:
        self.pin = int(pin)
        self._device = None
        self._gpio_command: Optional[str] = None
        self._lock = threading.Lock()
        self._high = False
        self._desired_high = False
        self._last_error = ""

    def _run_gpio_command(self, high: bool) -> bool:
        command = self._gpio_command
        if not command:
            return False
        level = "dh" if high else "dl"
        try:
            result = subprocess.run(
                [command, "set", str(self.pin), "op", level],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=2.0,
            )
            if result.returncode == 0:
                return True
            raise RuntimeError((result.stderr or "").strip() or f"exit {result.returncode}")
        except Exception as exc:
            print(f"[led-ring] {command} nije uspio: {exc}", flush=True)
            self._gpio_command = None
            return False

    def _ensure_device(self) -> bool:
        if self._device is not None or self._gpio_command is not None:
            return True
        # Raspberry Pi OS provides one of these tools independently of the
        # Python virtual environment. Prefer it so GPIO does not depend on a
        # gpiozero pin-factory package being installed inside the venv.
        for name in ("pinctrl", "raspi-gpio"):
            command = shutil.which(name)
            if not command:
                continue
            self._gpio_command = command
            if self._run_gpio_command(False):
                print(f"[led-ring] backend={name}, BCM GPIO{self.pin}", flush=True)
                self._high = False
                self._last_error = ""
                return True
        try:
            from gpiozero import OutputDevice

            # Start LOW first so importing/initializing cannot pulse the relay.
            self._device = OutputDevice(
                self.pin,
                active_high=True,
                initial_value=False,
            )
            self._high = False
            if self._last_error:
                print(f"[led-ring] GPIO{self.pin} je sada dostupan", flush=True)
                self._last_error = ""
            return True
        except Exception as exc:
            error = str(exc)
            if error != self._last_error:
                print(
                    f"[led-ring] GPIO{self.pin} jos nije dostupan; pokusat cu ponovno: {exc}",
                    flush=True,
                )
                self._last_error = error
            return False

    def _apply_desired_state(self) -> bool:
        if not self._ensure_device():
            return False
        if self._gpio_command is not None:
            if not self._run_gpio_command(self._desired_high):
                # Command disappeared/failed: immediately try gpiozero.
                if not self._ensure_device():
                    return False
                if self._gpio_command is not None:
                    if not self._run_gpio_command(self._desired_high):
                        return False
                    if self._desired_high != self._high:
                        level = "HIGH" if self._desired_high else "LOW"
                        print(f"[led-ring] GPIO{self.pin}={level}", flush=True)
                    self._high = self._desired_high
                    return True
            else:
                if self._desired_high != self._high:
                    level = "HIGH" if self._desired_high else "LOW"
                    print(f"[led-ring] GPIO{self.pin}={level}", flush=True)
                self._high = self._desired_high
                return True
        if self._desired_high:
            self._device.on()
            if not self._high:
                print(f"[led-ring] GPIO{self.pin}=HIGH", flush=True)
            self._high = True
        else:
            self._device.off()
            if self._high:
                print(f"[led-ring] GPIO{self.pin}=LOW", flush=True)
            self._high = False
        return True

    def setup_low(self) -> bool:
        """Claim GPIO17 at application boot and guarantee an initial LOW."""
        with self._lock:
            self._desired_high = False
            if not self._apply_desired_state():
                return False
            print(f"[led-ring] GPIO{self.pin}=LOW (standby)", flush=True)
            return True

    def retry(self) -> bool:
        """Retry a boot-time GPIO failure without changing the requested state."""
        with self._lock:
            return self._apply_desired_state()

    def on(self) -> bool:
        with self._lock:
            self._desired_high = True
            if (self._device is not None or self._gpio_command is not None) and self._high:
                return True
            return self._apply_desired_state()

    def off(self) -> None:
        with self._lock:
            self._desired_high = False
            if self._gpio_command is not None:
                if not self._high:
                    return
                if self._run_gpio_command(False):
                    print(f"[led-ring] GPIO{self.pin}=LOW", flush=True)
                    self._high = False
                return
            device = self._device
            if device is None:
                return
            if not self._high:
                return
            try:
                device.off()
                if self._high:
                    print(f"[led-ring] GPIO{self.pin}=LOW", flush=True)
                self._high = False
            except Exception as exc:
                print(f"[led-ring] GPIO{self.pin} LOW nije uspio: {exc}", flush=True)

    def shutdown(self) -> None:
        with self._lock:
            if self._gpio_command is not None:
                self._run_gpio_command(False)
                if self._high:
                    print(f"[led-ring] GPIO{self.pin}=LOW", flush=True)
                self._high = False
                self._gpio_command = None
                return
            device = self._device
            if device is None:
                return
            try:
                device.off()
                if self._high:
                    print(f"[led-ring] GPIO{self.pin}=LOW", flush=True)
                self._high = False
            finally:
                self._device = None
                try:
                    device.close()
                except Exception:
                    pass


_controller: Optional[LedRingPower] = None


def led_ring_on() -> bool:
    global _controller
    if _controller is None:
        _controller = LedRingPower()
    return _controller.on()


def led_ring_setup_low() -> bool:
    global _controller
    if _controller is None:
        _controller = LedRingPower()
    return _controller.setup_low()


def led_ring_retry() -> bool:
    global _controller
    if _controller is None:
        _controller = LedRingPower()
    return _controller.retry()


def led_ring_off() -> None:
    if _controller is not None:
        _controller.off()


def led_ring_shutdown() -> None:
    if _controller is not None:
        _controller.shutdown()


atexit.register(led_ring_shutdown)
