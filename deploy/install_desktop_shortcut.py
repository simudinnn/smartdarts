#!/usr/bin/env python3
"""Create a SmartDarts launcher on the current Raspberry Pi user's desktop."""

from __future__ import annotations

import os
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent.parent
DESKTOP_DIR = Path.home() / "Desktop"
SHORTCUT = DESKTOP_DIR / "SMARTDARTS.desktop"
OLD_SHORTCUT = DESKTOP_DIR / "SmartDarts.desktop"


def main() -> int:
    run_script = APP_DIR / "run.sh"
    if not run_script.is_file():
        print(f"ERROR: nema {run_script}")
        return 1

    DESKTOP_DIR.mkdir(parents=True, exist_ok=True)
    if OLD_SHORTCUT != SHORTCUT and OLD_SHORTCUT.is_file():
        OLD_SHORTCUT.unlink()
    shortcut = f"""[Desktop Entry]
Type=Application
Version=1.0
Name=SMARTDARTS
Comment=Pokreni SmartDarts kiosk
Exec=/bin/bash {run_script}
Path={APP_DIR}
Terminal=false
Icon={APP_DIR / 'assets' / 'logo.png'}
Categories=Game;
StartupNotify=false
"""
    SHORTCUT.write_text(shortcut, encoding="utf-8", newline="\n")
    os.chmod(SHORTCUT, 0o755)
    print(f"Desktop precac napravljen: {SHORTCUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
