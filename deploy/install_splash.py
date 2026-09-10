#!/usr/bin/env python3
"""Plymouth boot splash na Raspberry Pi.

Pokreni NA PI-JU (Python — CRLF nije problem):
  cd ~/smartdarts
  sudo python3 deploy/install_splash.py

Postavlja i kiosk autostart (labwc), inače nakon reboota ostane samo desktop.

Zadana slika:  assets/splash.png
Druga slika:   sudo python3 deploy/install_splash.py /putanja/do/slike.png
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent / "plymouth"
THEME_DIR = Path("/usr/share/plymouth/themes/smartdarts")
DEFAULT_IMG = APP_DIR / "assets" / "splash.png"


def _die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _write_lf(path: Path, text: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.replace("\r\n", "\n").replace("\r", "\n")
    if not data.endswith("\n"):
        data += "\n"
    path.write_bytes(data.encode("utf-8"))
    os.chmod(path, mode)


def _copy_text_lf(src: Path, dst: Path) -> None:
    text = src.read_text(encoding="utf-8", errors="replace")
    _write_lf(dst, text)


def _install_kiosk_autostart() -> None:
    """Splash sam ne diže aplikaciju — bez ovoga nakon reboota ostane samo desktop."""
    helper = APP_DIR / "deploy" / "install_autostart.py"
    if not helper.is_file():
        print("WARNING: nema deploy/install_autostart.py — kiosk se neće sam dignuti")
        return
    sudo_user = (os.environ.get("SUDO_USER") or "").strip()
    if sudo_user and sudo_user not in ("root",):
        cmd = ["sudo", "-u", sudo_user, "-H", sys.executable, str(helper)]
    else:
        cmd = [sys.executable, str(helper)]
    print("+ autostart:", " ".join(cmd))
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        print(
            "WARNING: autostart nije uspio — na Pi-ju kao običan user:\n"
            f"  python3 {helper}"
        )


def _need_root() -> None:
    if os.geteuid() == 0:
        return
    script = str(Path(__file__).resolve())
    print("Treba root — ponovo s sudo...")
    raise SystemExit(subprocess.call(["sudo", sys.executable, script, *sys.argv[1:]]))


def _resolve_image(arg: str | None) -> Path:
    candidates: list[Path] = []
    if arg:
        candidates.append(Path(arg).expanduser())
    candidates.extend(
        [
            DEFAULT_IMG,
            APP_DIR / "splash.png",
            Path("/home/user2/smartdarts/assets/splash.png"),
            Path("/home/user2/splash.png"),
        ]
    )
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path.resolve()
    _die(
        "nema splash slike. Stavi PNG u assets/splash.png ili predaj putanju:\n"
        f"  sudo python3 {Path(__file__).resolve()} /putanja/do/slike.png"
    )
    raise SystemExit(1)  # unreachable, keeps type checkers happy


def _first_existing(paths: list[str]) -> Path | None:
    for p in paths:
        path = Path(p)
        if path.is_file():
            return path
    return None


def _ensure_config_kv(path: Path, key: str, value: str) -> None:
    raw = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.split("\n")
    prefix = f"{key}="
    found = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(prefix) or stripped.startswith(f"#{prefix}"):
            lines[i] = f"{key}={value}"
            found = True
            break
    if not found:
        if lines and lines[-1] == "":
            lines.insert(-1, f"{key}={value}")
        else:
            lines.append(f"{key}={value}")
    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")
    print(f"  {path}: {key}={value}")


def _patch_cmdline(path: Path) -> None:
    raw = path.read_text(encoding="utf-8", errors="replace")
    line = raw.replace("\r", "").replace("\n", " ").strip()
    tokens = line.split()
    for tok in ("quiet", "splash", "logo.nologo", "vt.global_cursor_default=0"):
        if tok not in tokens:
            tokens.append(tok)
    new = " ".join(tokens) + "\n"
    path.write_text(new, encoding="utf-8")
    print(f"  {path}: quiet splash logo.nologo vt.global_cursor_default=0")


def main() -> int:
    _need_root()

    img = _resolve_image(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"APP_DIR  = {APP_DIR}")
    print(f"IMAGE    = {img}")
    print(f"THEME    = {THEME_DIR}")

    if not SRC_DIR.is_dir():
        _die(f"nema plymouth templatea: {SRC_DIR}")
    ply = SRC_DIR / "smartdarts.plymouth"
    script = SRC_DIR / "smartdarts.script"
    if not ply.is_file() or not script.is_file():
        _die(f"nedostaje {ply.name} ili {script.name} u {SRC_DIR}")

    THEME_DIR.mkdir(parents=True, exist_ok=True)
    _copy_text_lf(ply, THEME_DIR / "smartdarts.plymouth")
    _copy_text_lf(script, THEME_DIR / "smartdarts.script")
    shutil.copy2(img, THEME_DIR / "splash.png")
    for p in THEME_DIR.iterdir():
        if p.is_file():
            os.chmod(p, 0o644)
    print(f"  kopirano u {THEME_DIR}")

    if shutil.which("plymouth-set-default-theme"):
        print("+ plymouth-set-default-theme -R smartdarts")
        r = subprocess.run(["plymouth-set-default-theme", "-R", "smartdarts"], check=False)
        if r.returncode != 0:
            print("WARNING: plymouth-set-default-theme nije uspio — splash možda ostane stari.")
    else:
        print("plymouth-set-default-theme nije dostupan — zamjenjujem pix splash.png")
        pix = Path("/usr/share/plymouth/themes/pix/splash.png")
        if pix.is_file():
            bak = Path("/usr/share/plymouth/themes/pix/splash.png.bak")
            if not bak.is_file():
                shutil.copy2(pix, bak)
            shutil.copy2(img, pix)

    bootcfg = _first_existing(["/boot/firmware/config.txt", "/boot/config.txt"])
    if bootcfg:
        _ensure_config_kv(bootcfg, "disable_splash", "1")
        _ensure_config_kv(bootcfg, "avoid_warnings", "1")
    else:
        print("WARNING: nema config.txt — preskačem rainbow splash patch")

    cmdline = _first_existing(["/boot/firmware/cmdline.txt", "/boot/cmdline.txt"])
    if cmdline:
        _patch_cmdline(cmdline)
    else:
        print("WARNING: nema cmdline.txt — preskačem quiet/splash")

    print()
    _install_kiosk_autostart()
    print()
    print("OK — splash + autostart. Reboot:")
    print("  sudo reboot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
