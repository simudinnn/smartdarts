# Manual Kiosk Bundle (Raspberry Pi)

This folder is standalone for the darts kiosk app.

## Starting the app

Keep using the existing Raspberry Pi launcher/autostart. Folder updates,
including the external TV display, require no new setup or service installation.
`run.sh` delegates to the existing `~/.local/bin/smartdarts-kiosk` launcher.
Only on installations without that launcher does it use the local venv fallback.

## GUI

- Default: **PySide6 (Qt)** for UI, OpenCV only for cameras/detection/calibration preview
- Legacy OpenCV-drawn UI: `python3 main_manual.py --gui --kiosk --legacy-gui`

## USB cameras

Every camera opened by the kiosk, calibration, camera controls, or camera-index
utility requests **320x240 at 15 FPS** in the OpenCV/V4L2 open call, before its
first USB frame, and repeats those properties after opening. Pixel format is not
changed because some three-camera UVC hubs disconnect when MJPEG/YUYV is forced.
Driver-reported FPS is informational and is not used to reject a working feed.
Captured frames are normalized to 320x240 as a final safety limit.

Camera opening and calibration startup never wait on the Qt UI thread. An
already-open capture remains alive while Settings temporarily covers a game or
calibration screen, avoiding repeated USB close/open cycles. Returning to a
game schedules its clean motion reference on the vision thread. Camera shutdown
waits at most 250 ms on the UI thread; a blocked UVC read releases its own
device safely when it returns.

Calibration now processes the native 320x240 frame without a temporary
640x480 upscale. Calibration snapshots never exceed 320x240, final calibration
debug renders are 200x200, and saved motion-diff/fit images are 200x200. The
internal scoring geometry remains unchanged.

## External TV (Qt kiosk only)

In kiosk mode a second monitor automatically shows a read-only
SmartDarts display: logo/name during setup and QR pages, player cards and three
hit slots during play, and the leaderboard once the finishing turn is submitted.
The TV reuses the main UI wallpaper and game card components. The three large
hit slots fill the upper half horizontally. Player cards fill the lower half:
one large centred card for one player, two cards in one row, or a stable 2x2
grid for three and four players. The active card stays in its cell and gains
the normal coloured border; cards never resize or rotate between positions.
Cricket markers, Killer lives and Around progress keep their original internal
layout. TV animations are disabled; it redraws only when displayed state
changes. TV cards use a fully black rounded background and larger player initials.
Killer numbers remain compact while the life counter is enlarged for distance viewing.
X01 suggestions and BUST are enlarged for viewing from a distance. Standby
shows a large logo above a large SMARTDARTS title. The TV has no controls or QR codes. Non-kiosk and legacy
OpenCV modes do not open it.

The TV is updated only when game state changes. Its graphics views use minimal
dirty-region updates and avoid full-screen smooth resampling, which keeps the
main touchscreen responsive on Raspberry Pi. With only one monitor connected,
the TV window is not created and receives no game snapshots, so every game uses
the original single-screen rendering and scoring path.

Player-card turn changes on the main display are applied atomically. Deferred
card move/resize animations are disabled because rapid hit, undo and screen
updates could otherwise leave compact cards mounted in the active layout or
access Qt widgets already queued for deletion. Hit feedback, score updates and
the active-player border remain enabled.
Use an **extended desktop**, not display mirroring.

Controls always use HDMI1 (`HDMI-1`, `HDMI-A-1` or `HDMI1` in Qt).
The read-only display uses another output, normally HDMI2. If HDMI1 is missing,
the TV presentation stays hidden. Old touch pairing files and environment
monitor overrides are ignored; there is no pairing or new launcher setup.

## Auto / Manual

Settings → Auto / Manual selects the mode for the next game. Auto preserves
existing camera detection and the in-game input toggle. Manual starts directly
from player selection without cameras or board calibration, always shows the
keypad, and removes the detection toggle and in-game calibration button. The
mode is fixed for the duration of that game, even if settings change mid-game.
The preference persists in `kiosk_settings.json` as `input_mode`.

## LED ring relay

SmartDarts claims BCM GPIO17 (physical header pin 11) as LOW at application
startup. Outside tournaments it remains HIGH throughout game selection,
rules, player selection, clear-board calibration and play. During a tournament
it remains LOW on the bracket and switches HIGH only when a match starts. It
returns LOW on standby and before app exit, reboot, or system shutdown.
The relay control uses `gpiozero`; on non-Raspberry systems or when GPIO access
is unavailable, the application logs the condition and continues normally.
If the kiosk starts before the Raspberry Pi GPIO backend is ready, it retries
GPIO17 during the first 15 seconds while preserving the latest requested
HIGH/LOW state. A temporary boot-time failure therefore no longer disables the
relay for the rest of the application session.
The launcher dependency check includes `gpiozero`, including virtual
environments created before relay support was added. Because the installed
launcher lives outside the project folder, run `python3 deploy/install_autostart.py`
once after installing this GPIO fix to refresh it.
Use GPIO17 only as the relay control signal, with a 3.3 V-compatible input and
a common ground. Power the LED ring from its own correctly rated supply.

## Desktop shortcut

On the Raspberry Pi, create a desktop shortcut once with:

```bash
cd ~/smartdarts
python3 deploy/install_desktop_shortcut.py
```

The **SMARTDARTS** shortcut uses `assets/logo.png` and starts `run.sh`, which changes to the SmartDarts folder, uses the
project virtual environment, launches `main_manual.py --gui --kiosk --ml`, and keeps
the existing single-instance lock.

## Wi-Fi in kiosk mode

Settings includes a **WIFI** button. It opens a fullscreen modal that scans
NetworkManager networks, marks the active connection, shows signal/security,
and connects with an optional password. The password button appears after a
network is selected. Tapping it opens a separate fullscreen keyboard with
ABC/symbol, CAPS, space, delete, and a control that temporarily shows or hides
the entered password. Scan and connection commands run asynchronously through `nmcli`, so the kiosk UI
stays responsive. Raspberry Pi images must use NetworkManager and provide the
`nmcli` command; otherwise the modal reports that Wi-Fi management is unavailable.

## Lokalni turniri (prva verzija)

U Postavkama je gumb **TURNIR** koji otvara zaseban turnirski ekran. Host može
napraviti eliminacijski turnir za 2, 4, 6 ili 8
igrača, nastaviti spremljeni lokalni turnir ili pokrenuti slobodan meč. Stanje se
atomski sprema u `tournament_state.json`, pa restart aplikacije ne briše bracket.
Pri izradi se bira 301, 501, Cricket, Killer, Around ili Halve. Nakon leaderboarda pobjednik automatski napreduje, a
prekinuti meč ponovno postaje slobodan.

Za više ploča na hostu odaberite **NOVI TURNIR (HOST)**. Ostale ploče biraju
**PRONAĐI TURNIR (IP)** i upisuju IP hosta. Host sluša lokalni TCP port 8765 i
jedini mijenja bracket; zauzimanje meča i rezultat provjeravaju ploču i trenutno
stanje kako dvije ploče ne bi pokrenule isti meč. Ova verzija radi na postojećoj
LAN/Wi-Fi mreži. Automatsko stvaranje i gašenje SmartDarts hotspota još nije
uključeno jer prvo treba provjeriti bracket i oporavak mečeva na stvarnim pločama.
The network list uses fixed-height rows and touch scrolling, with a vertical
scrollbar when more networks are available than fit on screen.

## Raspberry Pi performance

Settings includes a **PERFORMANSE** button that opens a live system-health
view. It shows CPU load and clock, temperature, RAM use, GPU clock and reserved
memory, uptime, and Raspberry Pi power/throttling flags. Current faults and
faults recorded since boot are decoded separately so undervoltage or thermal
throttling remains visible after the condition clears. Measurements refresh in
the background every two seconds and can also be refreshed manually, without
blocking the touchscreen. Metrics that the current device does not expose are
shown as unavailable.

On Raspberry Pi/Wayland the TV is selected by Qt output identity, without relying
on global window coordinates. `[spectator]` lines in the existing `autostart.log`
report the backend, detected monitors and selected TV. If the TV stays on the
desktop, inspect those lines to distinguish monitor detection from placement
or an old application folder being launched.

## MQTT topic per kiosk

Edit `mqtt.py` and set:
- `TOPIC = "pikado/pikado1/cmd"` (Pi #1), or
- `TOPIC = "pikado/pikado2/cmd"` (Pi #2), or
- `TOPIC = "pikado/pikado3/cmd"` (Pi #3)
## ML pogodak izvan ploče

Ne-warpani surround služi za aktiviranje motion/wait faze, ali sam ne odlučuje
je li pogodak MISS jer shaft i flight pravilnog double pogotka često ostaju izvan
elipse. Nakon potvrđenog bacanja ML procjenjuje vrh na top-down fused slici.
Scoring geometrija vraća `MISS` samo kada je taj vrh izvan vanjskog double ruba.
Za rubne pogotke ML prihvaća dvije slabije top-down putanje od najmanje 8 piksela.

## Eksperimentalni single-camera način

Pokretanje `./run.sh --single` otvara samo prvi indeks kamere i traži 1280x720
pri 15 FPS. Kalibrira se samo ta kamera. Blob se analizira u izvornom kadru;
top-down transformira se samo procijenjena točka vrha radi postojećeg scoriranja.
Procjenjivač pretpostavlja da je kamera fiksno iznad segmenta 20. Od najvećeg
dijela strijelice prati samo lokalno povezani lanac flighta i shafta te bira
njegovu najnižu točku. Odvojeni odsjaji i šum na žicama zato ne mogu postati vrh
samo zato što su približno u istom stupcu. Native motion maske i označeni settle
frame spremaju se odgođeno nakon prikaza pogotka.

`python3 camera_controls.py` uvijek otvara sva tri capture uređaja
(`/dev/video0`, `/dev/video2` i `/dev/video4`) zajedno u štedljivom 320x240,
15 FPS načinu, neovisno o tome koristi li igra single detektor. Touch sučelje
prikazuje sva tri 4:3 pregleda, dopušta odabir kamere i može iste postavke
primijeniti na sve tri kamere.

### Eksperimentalni način s jednom kamerom

Pokrenite aplikaciju s `./run.sh --single`. Otvara se samo prvi capture uređaj
(`/dev/video0`) u 1280x720, 15 FPS MJPEG načinu. Poruka `OPEN OK` sada se
ispisuje tek nakon što kamera vrati stvarni kadar; `STREAM FAIL` znači da je
uređaj otvoren, ali odabrani video način ne daje slike.
