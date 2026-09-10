#!/usr/bin/env bash
# Tanki wrapper — pravi launcher je ~/.local/bin/smartdarts-kiosk (izvan foldera).
# Ako launcher postoji, koristi ga; inače fallback na venv python.
set -e
cd "$(dirname "$0")"
if [ -x "$HOME/.local/bin/smartdarts-kiosk" ]; then
  exec "$HOME/.local/bin/smartdarts-kiosk" "$@"
fi
if [ ! -x "venv/bin/python" ]; then
  python3 -m venv venv
  ./venv/bin/pip install --upgrade pip
  ./venv/bin/pip install -r requirements.txt
fi
exec ./venv/bin/python main_manual.py --gui --kiosk "$@"
