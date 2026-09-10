#!/usr/bin/env bash
# Kompletan setup na Pi-ju — zalijepi cijeli blok u SSH.
# Pretpostavka: /home/user2/smartdarts/run.sh postoji, assets/splash.png postoji.
set -e
APP=/home/user2/smartdarts
SPLASH=/home/user2/smartdarts/assets/splash.png
USER_HOME=/home/user2

if [ ! -f "$APP/run.sh" ]; then
  echo "NEMA $APP/run.sh"; exit 1
fi
if [ ! -f "$SPLASH" ]; then
  echo "NEMA $SPLASH — stavi svoju sliku tu"; exit 1
fi

sed -i 's/\r$//' "$APP/run.sh" 2>/dev/null || true
chmod +x "$APP/run.sh"

# --- Autostart (labwc + desktop) ---
mkdir -p "$USER_HOME/.config/labwc" "$USER_HOME/.config/autostart"

cat > "$USER_HOME/.config/labwc/autostart" << 'EOF'
#!/bin/sh
export HOME=/home/user2
export QT_QPA_PLATFORM=wayland
export QT_WAYLAND_DISABLE_WINDOWDECORATION=1
export QT_OPENGL=es2
killall wf-panel-pi 2>/dev/null || true
killall wf-panel 2>/dev/null || true
killall lxpanel 2>/dev/null || true
killall pcmanfm 2>/dev/null || true
pcmanfm --desktop-off 2>/dev/null || true
sleep 1
/bin/bash /home/user2/smartdarts/run.sh >> /home/user2/smartdarts/autostart.log 2>&1 &
EOF
chmod +x "$USER_HOME/.config/labwc/autostart"

cat > "$USER_HOME/.config/autostart/pikado-kiosk.desktop" << 'EOF'
[Desktop Entry]
Type=Application
Name=SmartDarts Kiosk
Exec=/bin/bash -c '/bin/bash /home/user2/smartdarts/run.sh >> /home/user2/smartdarts/autostart.log 2>&1'
Path=/home/user2/smartdarts
Terminal=false
X-GNOME-Autostart-enabled=true
EOF

# --- User systemd (backup ako labwc ne pokrene) ---
mkdir -p "$USER_HOME/.config/systemd/user"
cat > "$USER_HOME/.config/systemd/user/pikado-kiosk.service" << 'EOF'
[Unit]
Description=SmartDarts Kiosk
After=graphical-session.target
PartOf=graphical-session.target

[Service]
Type=simple
Environment=HOME=/home/user2
Environment=QT_QPA_PLATFORM=wayland
Environment=QT_WAYLAND_DISABLE_WINDOWDECORATION=1
Environment=QT_OPENGL=es2
WorkingDirectory=/home/user2/smartdarts
ExecStartPre=/bin/sleep 2
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
loginctl enable-linger user2 2>/dev/null || true

# --- Plymouth: SAMO tvoja slika (bez teksta / pi logo) ---
THEME=/usr/share/plymouth/themes/smartdarts
sudo mkdir -p "$THEME"
sudo cp "$SPLASH" "$THEME/splash.png"

sudo tee "$THEME/smartdarts.plymouth" >/dev/null << 'EOF'
[Plymouth Theme]
Name=SmartDarts
Description=Image only
ModuleName=script

[script]
ImageDir=/usr/share/plymouth/themes/smartdarts
ScriptFile=/usr/share/plymouth/themes/smartdarts/smartdarts.script
EOF

sudo tee "$THEME/smartdarts.script" >/dev/null << 'EOF'
img = Image("splash.png");
ww = Window.GetWidth();
hh = Window.GetHeight();
spr = Sprite(img.Scale(ww, hh));
spr.SetZ(-100);
EOF

if command -v plymouth-set-default-theme >/dev/null 2>&1; then
  sudo plymouth-set-default-theme smartdarts
  sudo update-initramfs -u
else
  # fallback: prepiši pix script da ne crta logo/tekst
  if [ -f /usr/share/plymouth/themes/pix/pix.script ]; then
    sudo cp /usr/share/plymouth/themes/pix/pix.script /usr/share/plymouth/themes/pix/pix.script.bak
    sudo cp /usr/share/plymouth/themes/pix/splash.png /usr/share/plymouth/themes/pix/splash.png.bak 2>/dev/null || true
    sudo cp "$SPLASH" /usr/share/plymouth/themes/pix/splash.png
    sudo tee /usr/share/plymouth/themes/pix/pix.script >/dev/null << 'EOF'
img = Image("splash.png");
ww = Window.GetWidth();
hh = Window.GetHeight();
spr = Sprite(img.Scale(ww, hh));
spr.SetZ(-100);
EOF
    sudo update-initramfs -u
  fi
fi

# Makni rainbow
for f in /boot/firmware/config.txt /boot/config.txt; do
  if [ -f "$f" ]; then
    if grep -q '^disable_splash=' "$f"; then
      sudo sed -i 's/^disable_splash=.*/disable_splash=1/' "$f"
    else
      echo 'disable_splash=1' | sudo tee -a "$f" >/dev/null
    fi
  fi
done

echo "OK. Test ručno: bash $APP/run.sh"
echo "Log nakon boot: $APP/autostart.log"
echo "Zatim: sudo reboot"
