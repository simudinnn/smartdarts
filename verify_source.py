"""Fail fast when an interrupted/corrupt update puts NUL bytes in Python files."""
from pathlib import Path
import py_compile
import sys

root = Path(__file__).resolve().parents[1]
files = [root / name for name in (
    "main_manual.py", "qt_ui.py", "kiosk_session.py", "dart_detection.py", "tournament.py"
)]
bad = []
for path in files:
    if not path.exists():
        bad.append(f"nedostaje: {path.name}")
        continue
    if b"\0" in path.read_bytes():
        bad.append(f"NUL bajt: {path.name}")
        continue
    try:
        py_compile.compile(str(path), doraise=True)
    except Exception as exc:
        bad.append(f"{path.name}: {exc}")
if bad:
    print("PROVJERA NIJE PROŠLA\n" + "\n".join(bad))
    sys.exit(1)
print("IZVORNE DATOTEKE SU ISPRAVNE")
