#!/usr/bin/env python3
"""Jednokratni setup autostarta na Raspberry Pi.

Pokreni NA PI-JU (CRLF nije problem za .py):
  cd ~/smartdarts
  python3 deploy/install_autostart.py

SAMO JEDAN put pokretanja: labwc ~/.config/labwc/autostart
Ostalo (desktop autostart, user/system systemd) se gasi da ne diže 4 instance.

Launcher ide IZVAN projekta (~/.local/bin/smartdarts-kiosk),
pa update/zamjena foldera smartdarts NE kvari autostart.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
USER_HOME = Path.home()
LAUNCHER = USER_HOME / ".local" / "bin" / "smartdarts-kiosk"
LABWC_AUTOSTART = USER_HOME / ".config" / "labwc" / "autostart"
DESKTOP_AUTOSTART = USER_HOME / ".config" / "autostart" / "smartdarts-kiosk.desktop"
DESKTOP_AUTOSTART_OLD = USER_HOME / ".config" / "autostart" / "pikado-kiosk.desktop"
USER_SERVICE = USER_HOME / ".config" / "systemd" / "user" / "smartdarts-kiosk.service"
USER_SERVICE_OLD = USER_HOME / ".config" / "systemd" / "user" / "pikado-kiosk.service"
LOG_FILE = APP_DIR / "autostart.log"
LOCK_FILE = "/tmp/smartdarts-kiosk.lock"


def _write_lf(path: Path, text: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.replace("\r\n", "\n").replace("\r", "\n")
    if not data.endswith("\n"):
        data += "\n"
    path.write_bytes(data.encode("utf-8"))
    os.chmod(path, mode)


def _launcher_script(app_dir: Path) -> str:
    app = str(app_dir)
    log = str(app_dir / "autostart.log")
    return f"""#!/usr/bin/env bash
# SmartDarts kiosk launcher (IZVAN projekta — ne diraj pri updateu foldera)
set -e
APP="{app}"
LOG="{log}"
LOCK="{LOCK_FILE}"
cd "$APP"

# Jedna instanca — ako labwc/desktop/systemd ipak pokrenu više puta
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[smartdarts] already running, exit" | tee -a "$LOG"
  exit 0
fi

# Ako netko opet donese CRLF run.sh s Windowsa — popravi pa nastavi
if [ -f run.sh ]; then
  sed -i 's/\\r$//' run.sh 2>/dev/null || true
fi

if [ ! -x "venv/bin/python" ]; then
  echo "[smartdarts] creating venv..." | tee -a "$LOG"
  python3 -m venv venv
  ./venv/bin/pip install --upgrade pip
  ./venv/bin/pip install -r requirements.txt
fi

if ! ./venv/bin/python -c "import cv2, PySide6" 2>/dev/null; then
  echo "[smartdarts] installing requirements..." | tee -a "$LOG"
  ./venv/bin/pip install -r requirements.txt
fi

export HOME="{USER_HOME}"
export QT_QPA_PLATFORM="${{QT_QPA_PLATFORM:-wayland}}"
export QT_WAYLAND_DISABLE_WINDOWDECORATION="${{QT_WAYLAND_DISABLE_WINDOWDECORATION:-1}}"
# Pi GPU: prefer GLES for Qt widgets (safe with labwc/wayland; do not force eglfs).
export QT_OPENGL="${{QT_OPENGL:-es2}}"
export QT_QPA_EGLFS_INTEGRATION="${{QT_QPA_EGLFS_INTEGRATION:-eglfs_kms}}"

# Direktno python — ne ovisi o run.sh line endings
if [ -t 1 ]; then
  exec ./venv/bin/python main_manual.py --gui --kiosk "$@"
fi
exec ./venv/bin/python main_manual.py --gui --kiosk "$@" >>"$LOG" 2>&1
"""


def _labwc_autostart(launcher: Path) -> str:
    return f"""#!/bin/sh
export HOME="{USER_HOME}"
export QT_QPA_PLATFORM=wayland
export QT_WAYLAND_DISABLE_WINDOWDECORATION=1
export QT_OPENGL=es2
killall wf-panel-pi wf-panel lxpanel pcmanfm 2>/dev/null || true
pcmanfm --desktop-off 2>/dev/null || true
"{launcher}" &
"""


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=False)


def _strip_crlf_sh(root: Path) -> None:
    """Windows copy ostavlja CRLF u .sh — bash onda vidi $'\\r'."""
    for path in sorted(root.rglob("*.sh")):
        raw = path.read_bytes()
        if b"\r" not in raw:
            continue
        text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        path.write_bytes(text.encode("utf-8"))
        print(f"LF: {path}")


def _try_disable_user(name: str) -> None:
    r = subprocess.run(
        ["systemctl", "--user", "disable", "--now", name],
        capture_output=True,
        text=True,
    )
    if r.returncode == 0:
        print(f"+ disabled {name}")


def _rm(path: Path) -> None:
    if path.is_file():
        path.unlink()
        print(f"removed {path}")


def _splash_image() -> Path | None:
    for path in (
        APP_DIR / "assets" / "splash.png",
        APP_DIR / "splash.png",
        USER_HOME / "splash.png",
    ):
        if path.is_file():
            return path
    return None


def main() -> int:
    if not (APP_DIR / "main_manual.py").is_file():
        print(f"ERROR: nema main_manual.py u {APP_DIR}", file=sys.stderr)
        return 1

    print(f"APP_DIR = {APP_DIR}")
    print(f"LAUNCHER = {LAUNCHER}")

    _write_lf(LAUNCHER, _launcher_script(APP_DIR), mode=0o755)
    _write_lf(LABWC_AUTOSTART, _labwc_autostart(LAUNCHER), mode=0o755)

    # Popravi run.sh u projektu (opcionalno, za ručni bash run.sh)
    _write_lf(
        APP_DIR / "run.sh",
        f"""#!/usr/bin/env bash
set -e
exec "{LAUNCHER}" "$@"
""",
        mode=0o755,
    )

    # --- SAMO labwc — makni ostale autostart puteve (uzrok 4 instance) ---
    _rm(DESKTOP_AUTOSTART)
    _rm(DESKTOP_AUTOSTART_OLD)
    _rm(USER_SERVICE)
    _rm(USER_SERVICE_OLD)

    _strip_crlf_sh(APP_DIR / "deploy")
    _write_lf(
        APP_DIR / "deploy" / "install-splash.sh",
        """#!/usr/bin/env bash
# Wrapper — pravi installer je Python (CRLF s Windowsa lomi ovaj .sh).
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$DIR/install_splash.py" "$@"
""",
        mode=0o755,
    )

    _run(["systemctl", "--user", "daemon-reload"])
    _try_disable_user("smartdarts-kiosk.service")
    _try_disable_user("pikado-kiosk.service")
    subprocess.run(
        ["sudo", "systemctl", "disable", "--now", "pikado-kiosk.service"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print()
    print("OK — autostart: SAMO labwc (~/.config/labwc/autostart)")
    print("  Launcher:", LAUNCHER)
    print("  Lock:    ", LOCK_FILE, "(druga instanca odmah izlazi)")
    print("  Test:    ", LAUNCHER)
    print("  Log:     ", LOG_FILE)
    print()
    print("Boot slika (nema Pi loga / terminala) — jednom s sudo:")
    print("  (Python, ne bash — izbjegava CRLF s Windowsa)")
    img = _splash_image()
    if img is not None:
        print(f"  sudo python3 {APP_DIR / 'deploy' / 'install_splash.py'}")
        print(f"  slika: {img}")
    else:
        print("  stavi splash.png u assets/splash.png pa:")
        print(f"  sudo python3 {APP_DIR / 'deploy' / 'install_splash.py'}")
    print("  Zatim:   sudo reboot")
    print()
    print("Update foldera: ne treba ponovo install (osim nove putanje).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
