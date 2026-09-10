#!/usr/bin/env bash
# NA PI-JU: bash ovaj_blok  — ili zalijepi sadržaj u SSH.
# Ne kopiraj run.sh s Windowsa (CRLF lomi boot).
set -e
APP=/home/user2/smartdarts
cd "$APP"

# 1) run.sh bez CRLF
cat > "$APP/run.sh" << 'EOF'
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -x "venv/bin/python" ]; then
  echo "Creating venv..."
  python3 -m venv venv
  ./venv/bin/pip install --upgrade pip
  ./venv/bin/pip install -r requirements.txt
fi
if ! ./venv/bin/python -c "import cv2, PySide6" 2>/dev/null; then
  ./venv/bin/pip install -r requirements.txt
fi
exec ./venv/bin/python main_manual.py --gui --kiosk "$@"
EOF
chmod +x "$APP/run.sh"

# 2) Autostart
mkdir -p /home/user2/.config/labwc /home/user2/.config/autostart
cat > /home/user2/.config/labwc/autostart << 'EOF'
#!/bin/sh
export HOME=/home/user2
export QT_QPA_PLATFORM=wayland
export QT_WAYLAND_DISABLE_WINDOWDECORATION=1
export QT_OPENGL=es2
killall wf-panel-pi wf-panel lxpanel pcmanfm 2>/dev/null || true
pcmanfm --desktop-off 2>/dev/null || true
/bin/bash /home/user2/smartdarts/run.sh >> /home/user2/smartdarts/autostart.log 2>&1 &
EOF
chmod +x /home/user2/.config/labwc/autostart

cat > /home/user2/.config/autostart/pikado-kiosk.desktop << 'EOF'
[Desktop Entry]
Type=Application
Name=SmartDarts
Exec=/bin/bash /home/user2/smartdarts/run.sh
Path=/home/user2/smartdarts
Terminal=false
X-GNOME-Autostart-enabled=true
EOF

# 3) User systemd backup
mkdir -p /home/user2/.config/systemd/user
cat > /home/user2/.config/systemd/user/pikado-kiosk.service << 'EOF'
[Unit]
Description=SmartDarts
After=graphical-session.target
[Service]
Environment=HOME=/home/user2
Environment=QT_QPA_PLATFORM=wayland
Environment=QT_WAYLAND_DISABLE_WINDOWDECORATION=1
Environment=QT_OPENGL=es2
WorkingDirectory=/home/user2/smartdarts
ExecStartPre=/bin/sleep 3
ExecStart=/bin/bash /home/user2/smartdarts/run.sh
Restart=on-failure
RestartSec=3
StandardOutput=append:/home/user2/smartdarts/autostart.log
StandardError=append:/home/user2/smartdarts/autostart.log
[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable pikado-kiosk.service
sudo loginctl enable-linger user2 2>/dev/null || true

# 4) button mode
python3 - << 'PY'
import json, pathlib
p = pathlib.Path("/home/user2/smartdarts/kiosk_settings.json")
d = json.loads(p.read_text()) if p.exists() else {}
d["start_mode"] = "button"
p.write_text(json.dumps(d, indent=2) + "\n")
print("start_mode=", d["start_mode"])
PY

echo "=== TEST (bez reboota) ==="
pkill -f main_manual.py 2>/dev/null || true
bash "$APP/run.sh" &
sleep 4
if pgrep -f main_manual.py >/dev/null; then
  echo "OK — program radi. Sada: sudo reboot"
else
  echo "FAIL — pogledaj log:"
  tail -50 "$APP/autostart.log" 2>/dev/null || true
  bash "$APP/run.sh"
fi
