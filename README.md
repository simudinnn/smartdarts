# Manual Kiosk Bundle (Raspberry Pi)

This folder is standalone for the darts kiosk app.

## First run

```bash
cd manual
bash run.sh
```

`run.sh` will:
- create `venv` if needed
- install/update `requirements.txt`
- launch `main_manual.py --gui --kiosk` (**Qt GUI** + OpenCV vision)

## GUI

- Default: **PySide6 (Qt)** for UI, OpenCV only for cameras/detection/calibration preview
- Legacy OpenCV-drawn UI: `python3 main_manual.py --gui --kiosk --legacy-gui`

## MQTT topic per kiosk

Edit `mqtt.py` and set:
- `TOPIC = "pikado/pikado1/cmd"` (Pi #1), or
- `TOPIC = "pikado/pikado2/cmd"` (Pi #2), or
- `TOPIC = "pikado/pikado3/cmd"` (Pi #3)
