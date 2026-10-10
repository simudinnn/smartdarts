"""Live UVC camera controls for Raspberry Pi (v4l2-ctl + OpenCV preview).

Run on the Pi while SmartDarts is NOT using the cameras. This tool opens all
three board cameras (/dev/video0, /dev/video2 and /dev/video4) together at
320x240, 15 FPS.

    python3 camera_controls.py

Keys:
  0/2/4  select camera
  s      save to camera_controls.json
  l      load from camera_controls.json
  a      apply current values to all three cameras
  w / x  previous / next control
  - / =  decrease / increase value (arrows also work)
  q / ESC  quit

Kiosk učitava camera_controls.json pri otvaranju kamera (ako datoteka postoji).
Every keyboard action is also available as a large touch button.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Raspberry Pi Desktop exports Wayland for the PySide kiosk, but the OpenCV
# wheel used by this standalone tool contains only the xcb HighGUI plugin.
# This must be selected before importing cv2.
if sys.platform.startswith("linux") and os.environ.get("QT_QPA_PLATFORM") == "wayland":
    os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np

from camera_capture_config import (
    CAPTURE_FPS,
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
)
from calibration import CameraManager
from led_ring import led_ring_off, led_ring_on

CAMERA_INDICES: Tuple[int, ...] = (0, 2, 4)
PREVIEW_WIDTH = 480
PREVIEW_HEIGHT = 360
UI_MIN_WIDTH = 960
CONTROLS_FILENAME = "camera_controls.json"
_INSTANCE_LOCK_PATH = "/tmp/smartdarts-camera-controls.lock"

# Preferred order / labels for the most useful UVC knobs.
_PREFERRED = (
    "exposure_auto",
    "auto_exposure",
    "exposure_time_absolute",
    "exposure_absolute",
    "exposure",
    "gain",
    "brightness",
    "contrast",
    "saturation",
    "gamma",
    "hue",
    "sharpness",
    "backlight_compensation",
    "white_balance_automatic",
    "white_balance_temperature_auto",
    "white_balance_temperature",
    "power_line_frequency",
)

@dataclass
class Ctrl:
    name: str
    kind: str  # int | bool | menu
    min_v: int
    max_v: int
    step: int
    default: int
    value: int
    menu: Dict[int, str] = field(default_factory=dict)


def controls_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), CONTROLS_FILENAME)


def _have_v4l2() -> bool:
    return shutil.which("v4l2-ctl") is not None


def _run_v4l2(args: List[str]) -> str:
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", *args],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=2,
        )
        return out
    except Exception as exc:
        print(f"[camctl] v4l2-ctl fail: {exc}", flush=True)
        return ""


def list_ctrls(index: int) -> Dict[str, Ctrl]:
    """Parse `v4l2-ctl --list-ctrls-menus` for /dev/video{index}."""
    text = _run_v4l2(["-d", f"/dev/video{index}", "--list-ctrls-menus"])
    ctrls: Dict[str, Ctrl] = {}
    current_menu: Optional[str] = None
    int_re = re.compile(
        r"^\s*(\S+)\s+0x[0-9a-fA-F]+\s+\((int|bool|menu)\)\s*:\s*"
        r"min=(-?\d+)\s+max=(-?\d+)\s+step=(-?\d+)\s+default=(-?\d+)\s+value=(-?\d+)",
    )
    menu_head_re = re.compile(
        r"^\s*(\S+)\s+0x[0-9a-fA-F]+\s+\(menu\)\s*:\s*"
        r"min=(-?\d+)\s+max=(-?\d+)(?:\s+default=(-?\d+))?(?:\s+value=(-?\d+))?"
    )
    menu_item_re = re.compile(r"^\s+(\d+):\s+(.+)$")
    for line in text.splitlines():
        m = int_re.match(line)
        if m:
            name, kind = m.group(1), m.group(2)
            ctrls[name] = Ctrl(
                name=name,
                kind=kind,
                min_v=int(m.group(3)),
                max_v=int(m.group(4)),
                step=max(1, int(m.group(5))),
                default=int(m.group(6)),
                value=int(m.group(7)),
            )
            current_menu = name if kind == "menu" else None
            continue
        m = menu_head_re.match(line)
        if m:
            name = m.group(1)
            default = int(m.group(4) or 0)
            value = int(m.group(5) or default)
            ctrls[name] = Ctrl(
                name=name,
                kind="menu",
                min_v=int(m.group(2)),
                max_v=int(m.group(3)),
                step=1,
                default=default,
                value=value,
            )
            current_menu = name
            continue
        m = menu_item_re.match(line)
        if m and current_menu and current_menu in ctrls:
            ctrls[current_menu].menu[int(m.group(1))] = m.group(2).strip()
    return ctrls


def get_ctrl(index: int, name: str) -> Optional[int]:
    text = _run_v4l2(["-d", f"/dev/video{index}", f"--get-ctrl={name}"])
    m = re.search(r":\s*(-?\d+)", text)
    return int(m.group(1)) if m else None


def set_ctrl(index: int, name: str, value: int) -> bool:
    if not _have_v4l2():
        return False
    text = _run_v4l2(["-d", f"/dev/video{index}", f"--set-ctrl={name}={int(value)}"])
    if "error" in text.lower() or "failed" in text.lower():
        print(f"[camctl] set {index} {name}={value} FAIL: {text.strip()}", flush=True)
        return False
    return True


def apply_values(index: int, values: Dict[str, int]) -> int:
    """Apply saved name→value map. Returns how many succeeded."""
    ok = 0
    # Auto-exposure / AWB first so manual exposure/wb can stick.
    first = [
        k
        for k in values
        if ("auto" in k and "exposure" in k)
        or k in ("white_balance_automatic", "white_balance_temperature_auto")
    ]
    rest = [k for k in values if k not in first]
    for name in first + rest:
        if set_ctrl(index, name, int(values[name])):
            ok += 1
    return ok


def apply_saved_controls(index: int) -> int:
    """Used by kiosk on camera open — no-op if JSON missing."""
    path = controls_path()
    if not os.path.isfile(path) or not _have_v4l2():
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except Exception:
        return 0
    values = data.get("all") or data.get(str(index)) or {}
    if not isinstance(values, dict) or not values:
        return 0
    n = apply_values(index, {str(k): int(v) for k, v in values.items()})
    if n:
        print(f"[camctl] applied {n} ctrls to cam{index}", flush=True)
    return n


def _ordered_names(ctrls: Dict[str, Ctrl]) -> List[str]:
    names = list(ctrls.keys())
    pref = [n for n in _PREFERRED if n in ctrls]
    extra = [n for n in names if n not in pref]
    extra.sort()
    return pref + extra


def _button(img: np.ndarray, rect: Tuple[int, int, int, int], text: str, *, active: bool = False) -> None:
    x1, y1, x2, y2 = rect
    fill = (0, 190, 220) if active else (30, 205, 238)
    border = (0, 235, 255)
    cv2.rectangle(img, (x1, y1), (x2, y2), fill, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), border, 2)
    scale = 0.62 if len(text) <= 10 else 0.50
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    tx = x1 + max(4, (x2 - x1 - tw) // 2)
    ty = y1 + (y2 - y1 + th) // 2
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (10, 10, 10), 2, cv2.LINE_AA)


def _draw_ui(
    frames: Dict[int, Optional[np.ndarray]],
    active: int,
    ctrls: Dict[str, Ctrl],
    names: List[str],
    sel: int,
    status: str = "",
) -> Tuple[np.ndarray, List[Tuple[Tuple[int, int, int, int], str]]]:
    cell_w, cell_h = PREVIEW_WIDTH, PREVIEW_HEIGHT
    gap = 8
    n = len(CAMERA_INDICES)
    content_w = n * cell_w + (n - 1) * gap
    total_w = max(UI_MIN_WIDTH, content_w)
    top = np.zeros((cell_h + 36, total_w, 3), dtype=np.uint8)
    top[:] = (24, 24, 28)
    for i, idx in enumerate(CAMERA_INDICES):
        x = (total_w - content_w) // 2 + i * (cell_w + gap)
        fr = frames.get(idx)
        if fr is None:
            tile = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            cv2.putText(tile, f"cam{idx} OFF", (220, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 200), 2)
        else:
            tile = cv2.resize(fr, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        if idx == active:
            cv2.rectangle(tile, (0, 0), (cell_w - 1, cell_h - 1), (0, 220, 220), 4)
        top[0:cell_h, x : x + cell_w] = tile
        live_state = "LIVE" if fr is not None else "OFF"
        selected_state = " / ODABRANA" if idx == active else ""
        label = f"Kamera {idx}  [{live_state}{selected_state}]"
        cv2.putText(top, label, (x + 8, cell_h + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

    row_h = 44
    toolbar_h = 66
    panel_h = toolbar_h + max(6, len(names)) * row_h + 12
    panel = np.zeros((panel_h, top.shape[1], 3), dtype=np.uint8)
    panel[:] = (18, 18, 22)
    hit_regions: List[Tuple[Tuple[int, int, int, int], str]] = []
    toolbar = [
        *((f"KAMERA {idx}", f"camera:{idx}") for idx in CAMERA_INDICES),
        ("SPREMI", "save"),
        ("UCITAJ", "load"),
        ("PRIMIJENI", "apply_all"),
        ("IZLAZ", "quit"),
    ]
    margin, gap = 8, 7
    bw = (panel.shape[1] - margin * 2 - gap * (len(toolbar) - 1)) // len(toolbar)
    for j, (text, action) in enumerate(toolbar):
        x1 = margin + j * (bw + gap)
        rect = (x1, 8, x1 + bw, 55)
        is_active = action == f"camera:{active}"
        _button(panel, rect, text, active=is_active)
        hit_regions.append((rect, action))
    for i, name in enumerate(names):
        y = toolbar_h + i * row_h
        c = ctrls[name]
        selected = i == sel
        col = (0, 220, 220) if selected else (200, 200, 200)
        minus_rect = (300, y + 4, 354, y + 39)
        bar_x1, bar_x2 = 365, top.shape[1] - 74
        plus_rect = (top.shape[1] - 63, y + 4, top.shape[1] - 9, y + 39)
        cv2.putText(panel, name, (12, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        _button(panel, minus_rect, "-", active=selected)
        _button(panel, plus_rect, "+", active=selected)
        hit_regions.append((minus_rect, f"minus:{i}"))
        hit_regions.append((plus_rect, f"plus:{i}"))
        hit_regions.append(((5, y, panel.shape[1] - 5, y + row_h - 1), f"select:{i}"))
        span = max(1, c.max_v - c.min_v)
        t = (c.value - c.min_v) / float(span)
        bx = int(bar_x1 + t * (bar_x2 - bar_x1))
        cv2.rectangle(panel, (bar_x1, y + 12), (bar_x2, y + 31), (50, 50, 58), -1)
        cv2.rectangle(panel, (bar_x1, y + 12), (max(bar_x1 + 1, bx), y + 31), col, -1)
        extra = ""
        if c.kind == "menu" and c.value in c.menu:
            extra = f" {c.menu[c.value]}"
        value_text = f"{c.value}{extra}"
        value_x = bar_x1 + 8
        cv2.putText(panel, value_text, (value_x, y + 29), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(panel, value_text, (value_x, y + 29), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if status:
        cv2.putText(top, status[:90], (10, cell_h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 235, 255), 2, cv2.LINE_AA)
    # Panel actions use panel coordinates; offset them below the preview.
    offset = top.shape[0]
    hit_regions = [((x1, y1 + offset, x2, y2 + offset), action)
                   for (x1, y1, x2, y2), action in hit_regions]
    return np.vstack([top, panel]), hit_regions


def _save(ctrls: Dict[str, Ctrl]) -> None:
    values = {name: int(c.value) for name, c in ctrls.items()}
    payload = {"all": values, "cameras": list(CAMERA_INDICES)}
    path = controls_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[camctl] saved {path} ({len(values)} ctrls)", flush=True)


def _load_into(ctrls: Dict[str, Ctrl], index: int) -> None:
    path = controls_path()
    if not os.path.isfile(path):
        print("[camctl] no camera_controls.json", flush=True)
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f) or {}
    values = data.get("all") or data.get(str(index)) or {}
    apply_values(index, {str(k): int(v) for k, v in values.items()})
    for name, c in ctrls.items():
        if name in values:
            c.value = int(values[name])
    print(f"[camctl] loaded into cam{index}", flush=True)


def _claim_single_instance():
    """Prevent repeated taps from opening several USB camera streams."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        import fcntl

        handle = open(_INSTANCE_LOCK_PATH, "w", encoding="utf-8")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.write(str(os.getpid()))
        handle.flush()
        return handle
    except (OSError, BlockingIOError):
        print("[camctl] Camera Controls je vec pokrenut.", flush=True)
        return False


def main() -> int:
    instance_lock = _claim_single_instance()
    if instance_lock is False:
        return 0
    if not _have_v4l2():
        print("v4l2-ctl nije u PATH — instaliraj v4l-utils (Raspberry Pi).", flush=True)
        print("  sudo apt install v4l-utils", flush=True)
        return 1

    # Camera Controls always uses the original low-bandwidth three-camera
    # profile, independently of the detector mode used by the kiosk process.
    import camera_capture_config as capture_cfg
    import calibration as calibration_module

    capture_cfg.SINGLE_CAMERA = False
    capture_cfg.CAPTURE_WIDTH = 320
    capture_cfg.CAPTURE_HEIGHT = 240
    calibration_module.CAPTURE_WIDTH = 320
    calibration_module.CAPTURE_HEIGHT = 240
    calibration_module.CAMERA_INDICES = CAMERA_INDICES
    calibration_module.CAMERA_LABELS = tuple(f"Kamera {idx}" for idx in CAMERA_INDICES)

    # Camera Controls is an active board mode, just like calibration. Use the
    # same GPIO17 controller as the main application.
    led_ring_on()

    # Reuse the capture manager with all three 320x240 streams.
    active = CAMERA_INDICES[0]
    camera_mgr = CameraManager()
    camera_mgr.start()
    camera_mgr.wait_for_frames(timeout_sec=12.0, min_cams=len(CAMERA_INDICES))
    opened = camera_mgr.opened_indices()
    if opened:
        active = int(opened[0])
    ctrls = list_ctrls(active)
    if not ctrls:
        print(f"[camctl] nema V4L2 kontrola na /dev/video{active}", flush=True)
    names = _ordered_names(ctrls)
    sel = 0

    win = "SmartDarts camera controls"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    click_queue: List[Tuple[int, int]] = []
    hit_regions: List[Tuple[Tuple[int, int, int, int], str]] = []

    def on_mouse(event, x, y, _flags, _userdata) -> None:
        if event == cv2.EVENT_LBUTTONUP:
            click_queue.append((int(x), int(y)))

    cv2.setMouseCallback(win, on_mouse)
    status = "Dodirni kameru ili + / -"
    status_until = 0.0
    print(f"[camctl] touch UI aktivan; kamere={opened}", flush=True)

    def set_status(text: str) -> None:
        nonlocal status, status_until
        status = text
        status_until = time.monotonic() + 3.0

    def switch_camera(index: int) -> None:
        nonlocal active, ctrls, names, sel
        if index == active:
            return
        active = int(index)
        ctrls = list_ctrls(active)
        names = _ordered_names(ctrls)
        sel = 0
        live = camera_mgr.grab_frames().get(active) is not None
        set_status(f"Kamera {active}: {'LIVE' if live else 'NEMA SIGNALA'}")

    def adjust_control(index: int, direction: int) -> None:
        nonlocal sel
        if not names or not (0 <= index < len(names)):
            return
        sel = index
        c = ctrls[names[sel]]
        step = max(c.step, max(1, (c.max_v - c.min_v) // 50))
        c.value = max(c.min_v, min(c.max_v, c.value + direction * step))
        if set_ctrl(active, c.name, c.value):
            set_status(f"{c.name}: {c.value}")
        else:
            set_status(f"Nije uspjelo: {c.name}")

    def run_action(action: str) -> bool:
        nonlocal sel
        if action.startswith("camera:"):
            switch_camera(int(action.split(":", 1)[1]))
        elif action.startswith("select:"):
            index = int(action.split(":", 1)[1])
            if 0 <= index < len(names):
                sel = index
        elif action.startswith("minus:"):
            adjust_control(int(action.split(":", 1)[1]), -1)
        elif action.startswith("plus:"):
            adjust_control(int(action.split(":", 1)[1]), 1)
        elif action == "save":
            _save(ctrls)
            values = {n: c.value for n, c in ctrls.items()}
            totals = [apply_values(idx, values) for idx in CAMERA_INDICES]
            set_status(f"Spremljeno i primijenjeno: {totals}")
        elif action == "load":
            _load_into(ctrls, active)
            set_status(f"Ucitano za kameru {active}")
        elif action == "apply_all":
            values = {n: c.value for n, c in ctrls.items()}
            totals = [apply_values(idx, values) for idx in CAMERA_INDICES]
            set_status(f"Primijenjeno na sve: {totals}")
        elif action == "quit":
            return False
        return True

    running = True
    while running:
        frames = camera_mgr.grab_frames()
        shown_status = status if time.monotonic() < status_until else "Kamere 0 / 2 / 4 - 320x240 - controls"
        img, hit_regions = _draw_ui(frames, active, ctrls, names, sel if names else 0, shown_status)
        cv2.imshow(win, img)
        key = cv2.waitKey(30) & 0xFF
        while click_queue and running:
            x, y = click_queue.pop(0)
            for (x1, y1, x2, y2), action in hit_regions:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    running = run_action(action)
                    break
        if key in (27, ord("q")):
            break
        if key in (ord("0"), ord("2"), ord("4")):
            switch_camera(int(chr(key)))
        elif key == ord("s"):
            run_action("save")
        elif key == ord("l"):
            run_action("load")
        elif key == ord("a"):
            run_action("apply_all")
        elif names and key in (82, ord("w")):
            sel = (sel - 1) % len(names)
        elif names and key in (84, ord("x")):
            sel = (sel + 1) % len(names)
        elif names and key in (81, ord("-"), ord("_")):
            adjust_control(sel, -1)
        elif names and key in (83, ord("="), ord("+")):
            adjust_control(sel, 1)

    camera_mgr.stop()
    cv2.destroyAllWindows()
    led_ring_off()
    return 0


if __name__ == "__main__":
    sys.exit(main())
