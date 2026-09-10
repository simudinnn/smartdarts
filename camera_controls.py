"""Live UVC camera controls for Raspberry Pi (v4l2-ctl + OpenCV preview).

Run on the Pi (kamere 0/2/4), while SmartDarts is NOT using the cameras:

    python3 camera_controls.py

Keys:
  s      save to camera_controls.json (and apply to all cams)
  l      load from camera_controls.json
  a      apply current values to ALL cameras
  0/2/4  switch preview camera
  w / x  previous / next control
  - / =  decrease / increase value (arrows also work)
  q / ESC  quit

Kiosk učitava camera_controls.json pri otvaranju kamera (ako datoteka postoji).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

CAMERA_INDICES: Tuple[int, ...] = (0, 2, 4)
CONTROLS_FILENAME = "camera_controls.json"

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


def _open_preview(index: int) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _draw_ui(
    frames: Dict[int, Optional[np.ndarray]],
    active: int,
    ctrls: Dict[str, Ctrl],
    names: List[str],
    sel: int,
) -> np.ndarray:
    cell_w, cell_h = 320, 240
    gap = 8
    n = len(CAMERA_INDICES)
    top = np.zeros((cell_h + 36, n * cell_w + (n - 1) * gap, 3), dtype=np.uint8)
    top[:] = (24, 24, 28)
    for i, idx in enumerate(CAMERA_INDICES):
        x = i * (cell_w + gap)
        fr = frames.get(idx)
        if fr is None:
            tile = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            cv2.putText(tile, f"cam{idx} OFF", (70, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 200), 2)
        else:
            tile = cv2.resize(fr, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        if idx == active:
            cv2.rectangle(tile, (0, 0), (cell_w - 1, cell_h - 1), (0, 220, 220), 4)
        top[0:cell_h, x : x + cell_w] = tile
        label = f"Kamera {idx}" + ("  [LIVE]" if idx == active else "")
        cv2.putText(top, label, (x + 8, cell_h + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

    row_h = 28
    panel_h = 80 + max(8, len(names)) * row_h
    panel = np.zeros((panel_h, top.shape[1], 3), dtype=np.uint8)
    panel[:] = (18, 18, 22)
    cv2.putText(
        panel,
        "s=save  l=load  a=apply-all  0/2/4=cam  w/x=ctrl  -/=value  q=quit",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
    )
    for i, name in enumerate(names):
        y = 56 + i * row_h
        c = ctrls[name]
        selected = i == sel
        col = (0, 220, 220) if selected else (200, 200, 200)
        bar_x1, bar_x2 = 360, top.shape[1] - 24
        cv2.putText(panel, name, (12, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        span = max(1, c.max_v - c.min_v)
        t = (c.value - c.min_v) / float(span)
        bx = int(bar_x1 + t * (bar_x2 - bar_x1))
        cv2.rectangle(panel, (bar_x1, y + 8), (bar_x2, y + 18), (50, 50, 58), -1)
        cv2.rectangle(panel, (bar_x1, y + 8), (max(bar_x1 + 1, bx), y + 18), col, -1)
        extra = ""
        if c.kind == "menu" and c.value in c.menu:
            extra = f" {c.menu[c.value]}"
        cv2.putText(
            panel,
            f"{c.value}{extra}",
            (bar_x1 - 120, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            col,
            1,
        )
    return np.vstack([top, panel])


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


def _mqtt_led_on() -> None:
    """Upali LED ring (isti broker/topic kao kiosk)."""
    try:
        import paho.mqtt.client as mqtt
        from mqtt import BROKER_HOST, BROKER_PORT, PASSWORD, TOPIC, USERNAME

        client = mqtt.Client()
        client.username_pw_set(USERNAME, PASSWORD)
        client.tls_set(
            ca_certs=None,
            certfile=None,
            keyfile=None,
            cert_reqs=ssl.CERT_REQUIRED,
            tls_version=ssl.PROTOCOL_TLS_CLIENT,
        )
        client.connect(BROKER_HOST, BROKER_PORT, 60)
        client.loop_start()
        info = client.publish(TOPIC, "on", qos=1)
        try:
            info.wait_for_publish(timeout=5.0)
        except Exception:
            pass
        client.loop_stop()
        client.disconnect()
        print(f"[camctl] MQTT 'on' -> {TOPIC}", flush=True)
    except Exception as exc:
        print(f"[camctl] MQTT on fail: {exc}", flush=True)


def main() -> int:
    if not _have_v4l2():
        print("v4l2-ctl nije u PATH — instaliraj v4l-utils (Raspberry Pi).", flush=True)
        print("  sudo apt install v4l-utils", flush=True)
        return 1

    threading.Thread(target=_mqtt_led_on, daemon=True).start()

    caps: Dict[int, Optional[cv2.VideoCapture]] = {}
    for idx in CAMERA_INDICES:
        cap = _open_preview(idx)
        caps[idx] = cap
        print(f"[camctl] cam{idx}: {'OK' if cap is not None else 'FAIL'}", flush=True)

    active = next((i for i in CAMERA_INDICES if caps.get(i) is not None), CAMERA_INDICES[0])
    ctrls = list_ctrls(active)
    if not ctrls:
        print(f"[camctl] nema V4L2 kontrola na /dev/video{active}", flush=True)
    names = _ordered_names(ctrls)
    sel = 0

    win = "SmartDarts camera controls"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("[camctl] s=save l=load a=all-cams  arrows=adjust  q=quit", flush=True)

    while True:
        frames: Dict[int, Optional[np.ndarray]] = {}
        for idx, cap in caps.items():
            if cap is None or not cap.isOpened():
                frames[idx] = None
                continue
            ok, fr = cap.read()
            frames[idx] = fr if ok else None
        img = _draw_ui(frames, active, ctrls, names, sel if names else 0)
        cv2.imshow(win, img)
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("0"):
            active = 0
            ctrls = list_ctrls(active)
            names = _ordered_names(ctrls)
            sel = 0
        elif key == ord("2"):
            active = 2
            ctrls = list_ctrls(active)
            names = _ordered_names(ctrls)
            sel = 0
        elif key == ord("4"):
            active = 4
            ctrls = list_ctrls(active)
            names = _ordered_names(ctrls)
            sel = 0
        elif key == ord("s"):
            _save(ctrls)
            values = {n: c.value for n, c in ctrls.items()}
            for idx in CAMERA_INDICES:
                apply_values(idx, values)
        elif key == ord("l"):
            _load_into(ctrls, active)
        elif key == ord("a"):
            values = {n: c.value for n, c in ctrls.items()}
            for idx in CAMERA_INDICES:
                n = apply_values(idx, values)
                print(f"[camctl] apply-all cam{idx}: {n}", flush=True)
        elif names and key in (82, ord("w")):
            sel = (sel - 1) % len(names)
        elif names and key in (84, ord("x")):
            sel = (sel + 1) % len(names)
        elif names and key in (81, ord("-"), ord("_")):
            c = ctrls[names[sel]]
            step = max(c.step, max(1, (c.max_v - c.min_v) // 50))
            c.value = max(c.min_v, c.value - step)
            set_ctrl(active, c.name, c.value)
        elif names and key in (83, ord("="), ord("+")):
            c = ctrls[names[sel]]
            step = max(c.step, max(1, (c.max_v - c.min_v) // 50))
            c.value = min(c.max_v, c.value + step)
            set_ctrl(active, c.name, c.value)

    for cap in caps.values():
        if cap is not None:
            cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
