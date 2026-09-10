# ML tip detector (experimental)

Alternative to **fitLine intersection** for fused tip `(x, y)` only.
Multicamera fusion, warp, segment scoring, and debug overlays stay the same;
when **ML (ONNX)** is selected, only the fused tip position is replaced.

## Layout

```
ml_tip/
  preprocess.py      # fused motion → 200×200 [0,1] + board inverse map
  model.py           # tiny TipCNN (PyTorch, train/export only)
  train.py
  export_onnx.py
  label_tool.py
  infer_onnx.py      # ONNX Runtime class used by the kiosk
  runtime.py         # settings bridge (override tip / save dataset)
  tip_detector.onnx  # produced by export (not committed by default)
  checkpoints/       # train.py best .pt
  dataset/
    images/          # 000001.png …
    labels.csv       # filename,x,y
  cpp/
    tip_detector.h
    tip_detector.cpp # future / Pi native ONNX Runtime
  requirements-ml.txt
```

## Coordinate convention

| Stage | Format |
|--------|--------|
| Model input | `1×200×200` grayscale, float `[0, 1]` |
| Labels / model output | pixel `(x, y)` in the **200×200** image, origin top-left, ≈ `[0, 199]` |
| Board / scoring | warp-space pixels after inverse map |

**Resize policy (stretch):** the full fused motion map (`max` of per-cam warp `motion_raw`) is resized to 200×200 with `cv2.INTER_AREA` (down) / `INTER_LINEAR` (up). No letterbox.

```
board_x = pred_x * (src_w / 200)
board_y = pred_y * (src_h / 200)
```

With the current kiosk `DEBUG_WARP_SIZE = 200`, this is identity.

## Workflow

### 1. Collect images (kiosk)

Settings → **ML tip dataset save: On**. On each accepted hit the kiosk writes
`ml_tip/dataset/images/NNNNNN.png` (same 200×200 preprocessing as inference).
No labels are written at save time. You can keep **Tip detection: FitLine** while collecting.

### 2. Label

```bash
python -m ml_tip.label_tool
# or: python ml_tip/label_tool.py
```

Run from the repo root so `ml_tip` imports resolve. A bright **yellow/magenta** tip marker appears on click (auto-saves `labels.csv`). HUD shows `tip: x, y`.

If the window says **NEMA SLIKA / NO IMAGES**, put 200×200 PNGs in `ml_tip/dataset/images/` or enable **ML tip dataset save** in the kiosk.

| Key / tipka | Action |
|-------------|--------|
| Left click | set tip / postavi vrh (auto-save) |
| `n` / `→` | next / sljedeća |
| `p` / `←` | previous / prethodna |
| `d` / Delete | clear label / obriši |
| `s` | save (usually not needed — click auto-saves) |
| `q` / Esc | quit / izlaz |
| `+` / `-` | zoom |

### 3. Train

```bash
pip install -r ml_tip/requirements-ml.txt
python -m ml_tip.train --epochs 80 --batch-size 16 --lr 1e-3
# resume:
python -m ml_tip.train --resume
```

Best checkpoint: `ml_tip/checkpoints/tip_cnn_best.pt`.

### 4. Export ONNX

```bash
python -m ml_tip.export_onnx
# → ml_tip/tip_detector.onnx
```

### 5. Run in kiosk

Settings → **Tip detection: ML (ONNX)**. Requires `onnxruntime` and `ml_tip/tip_detector.onnx`.
If the model is missing, fails to load, looks untrained/collapsed (constant tip near
board center), or the predicted tip disagrees with motion mass, the kiosk **keeps the
FitLine tip** and logs a warning — debug lines are not pinned through a garbage ML tip.

**FitLine** remains the default and is bit-identical when selected (no tip override).
ML only works after a real train+export (labels → `train` → `export_onnx`); a smoke
export with random weights is not usable.

## Runtime deps

| Use | Packages |
|-----|----------|
| Kiosk inference | `onnxruntime` (+ existing OpenCV/numpy) — **no torch** |
| Train / export | `torch`, `onnx`, `onnxruntime` — see `requirements-ml.txt` |

Root `requirements.txt` stays Pi-light; install ONNX Runtime on devices that use ML tip.

## C++ (future / Pi native)

`ml_tip/cpp/tip_detector.{h,cpp}` — load model once, `PredictGrayU8` / `PredictGray01` → `(x,y)`.
Link ONNX Runtime C++ API. Not built by the Python kiosk today.
