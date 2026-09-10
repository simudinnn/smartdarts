#!/usr/bin/env bash
# Wrapper — pravi installer je Python (CRLF s Windowsa lomi ovaj .sh).
# Na Pi-ju bolje: sudo python3 deploy/install_splash.py
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$DIR/install_splash.py" "$@"
