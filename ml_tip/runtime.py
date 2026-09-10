"""Kiosk bridge: optional ML tip override + dataset image dump.

FitLine path is untouched when tip_detection == 'fitline'. This module only
runs after fuse_multicam_lines when settings request ML and/or dataset save.

When ML is selected but the ONNX model is missing, fails to load, looks
collapsed/untrained, or predicts a tip that disagrees with motion mass,
the FitLine tip is kept (no debug-line pin through a garbage tip).
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from ml_tip.infer_onnx import TipOnnxDetector, try_get_tip_detector
from ml_tip.paths import default_dataset_dir, ensure_dataset_dirs, next_image_filename
from ml_tip.preprocess import INPUT_SIZE, uint8_model_image

# Cached collapse probe: path -> (collapsed: bool, spread_px: float)
_collapse_probe: dict = {}
_collapse_warned_paths: set = set()


def _settings_tip_mode() -> str:
    try:
        from kiosk_settings import get_settings

        mode = str(getattr(get_settings(), "tip_detection", "fitline") or "fitline")
        return mode.strip().lower()
    except Exception:
        return "fitline"


def _settings_save_dataset() -> bool:
    try:
        from kiosk_settings import get_settings

        return bool(getattr(get_settings(), "ml_tip_save_dataset", False))
    except Exception:
        return False


def _fuse_motion_from_per_cam(per_cam: List[Any], out_size: int) -> Optional[np.ndarray]:
    """Max-fuse per-cam motion_raw (same idea as dart_detection._fuse_motion_raw_maps)."""
    fused_acc: Optional[np.ndarray] = None
    for r in per_cam:
        raw = getattr(r, "motion_raw", None)
        if raw is None:
            continue
        if raw.ndim == 3:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        if raw.shape[0] != out_size or raw.shape[1] != out_size:
            raw = cv2.resize(raw, (out_size, out_size), interpolation=cv2.INTER_AREA)
        if fused_acc is None:
            fused_acc = raw.astype(np.uint8, copy=True)
        else:
            np.maximum(fused_acc, raw, out=fused_acc)
    return fused_acc


def save_ml_dataset_image(fused_u8: np.ndarray, dataset_dir: Optional[str] = None) -> Optional[str]:
    """Save 200×200 preprocessing twin of inference input; no label required.

    No-op unless kiosk setting ``ml_tip_save_dataset`` is explicitly on
    (default off — prototype auto-save disabled during normal play).
    """
    if not _settings_save_dataset():
        return None
    _root, images_dir, _labels = ensure_dataset_dirs(dataset_dir or default_dataset_dir())
    u8, _info = uint8_model_image(fused_u8)
    name = next_image_filename(images_dir)
    path = os.path.join(images_dir, name)
    if cv2.imwrite(path, u8):
        print(f"[ml_tip] dataset image: {path}", flush=True)
        return path
    print(f"[ml_tip] failed to write dataset image: {path}", flush=True)
    return None


def _synthetic_blob_inputs() -> List[np.ndarray]:
    """Corner / mid-side blobs for collapse probe (200×200 float [0,1])."""
    outs: List[np.ndarray] = []
    boxes = (
        (20, 50, 20, 50),    # NW
        (20, 50, 150, 180),  # NE
        (150, 180, 20, 50),  # SW
        (150, 180, 150, 180),  # SE
        (80, 120, 20, 50),   # N
        (80, 120, 150, 180),  # S
    )
    for y0, y1, x0, x1 in boxes:
        img = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        img[y0:y1, x0:x1] = 1.0
        outs.append(img)
    return outs


def detector_prediction_spread(det: TipOnnxDetector) -> float:
    """Max pairwise distance (px) of predictions on synthetic off-center blobs."""
    preds = [det.predict_xy(img) for img in _synthetic_blob_inputs()]
    spread = 0.0
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            d = float(np.hypot(preds[i][0] - preds[j][0], preds[i][1] - preds[j][1]))
            if d > spread:
                spread = d
    return spread


def detector_looks_collapsed(
    det: TipOnnxDetector, *, min_spread_px: float = 20.0
) -> Tuple[bool, float]:
    """True when ONNX barely moves tip across distinct blob locations.

    Typical of random-weight smoke exports or mean-collapsed undertrained nets
    that always emit ~board center (~100,100 on 200×200).
    """
    path = os.path.abspath(getattr(det, "model_path", "") or "")
    cached = _collapse_probe.get(path)
    if cached is not None:
        return cached
    try:
        spread = detector_prediction_spread(det)
    except Exception as exc:
        print(f"[ml_tip] collapse probe failed ({exc}) — treating as unsafe", flush=True)
        result = (True, 0.0)
        if path:
            _collapse_probe[path] = result
        return result
    collapsed = spread < float(min_spread_px)
    result = (collapsed, spread)
    if path:
        _collapse_probe[path] = result
    return result


def _motion_mask(fused_u8: np.ndarray) -> np.ndarray:
    g = fused_u8
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    g = g.astype(np.uint8, copy=False)
    peak = int(g.max()) if g.size else 0
    if peak <= 0:
        return np.zeros(g.shape[:2], dtype=np.uint8)
    thr = max(10, int(0.15 * peak))
    return (g >= thr).astype(np.uint8)


def _motion_centroid_and_mass(mask: np.ndarray) -> Tuple[Optional[Tuple[float, float]], int]:
    ys, xs = np.nonzero(mask)
    n = int(xs.size)
    if n <= 0:
        return None, 0
    return (float(xs.mean()), float(ys.mean())), n


def _local_motion_count(
    mask: np.ndarray, tip: Tuple[float, float], radius: float
) -> int:
    h, w = mask.shape[:2]
    tx, ty = float(tip[0]), float(tip[1])
    r = max(1.0, float(radius))
    x0 = max(0, int(np.floor(tx - r)))
    x1 = min(w, int(np.ceil(tx + r)) + 1)
    y0 = max(0, int(np.floor(ty - r)))
    y1 = min(h, int(np.ceil(ty + r)) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (xx - tx) ** 2 + (yy - ty) ** 2 <= r * r
    return int(np.count_nonzero(mask[y0:y1, x0:x1][disk]))


def ml_tip_implausible_vs_motion(
    tip_xy: Tuple[float, float],
    fused_u8: np.ndarray,
    *,
    out_size: int,
) -> Tuple[bool, str]:
    """Return (True, reason) when ML tip disagrees with fused motion mass.

    Catches the bullseye cluster failure mode: tip ≈ board center while colored
    motion blobs (and their mass) sit elsewhere, so debug lines would radiate
    from center through blobs.
    """
    mask = _motion_mask(fused_u8)
    centroid, mass = _motion_centroid_and_mass(mask)
    if centroid is None or mass < 20:
        return False, ""

    tx, ty = float(tip_xy[0]), float(tip_xy[1])
    cx, cy = float(out_size) * 0.5, float(out_size) * 0.5
    tip_to_center = float(np.hypot(tx - cx, ty - cy))
    tip_to_mass = float(np.hypot(tx - centroid[0], ty - centroid[1]))
    local_r = max(8.0, float(out_size) / 12.0)
    local = _local_motion_count(mask, (tx, ty), local_r)
    local_frac = float(local) / float(mass)

    near_center_r = float(out_size) * 0.18
    mass_far_r = float(out_size) * 0.22
    min_local = max(8, int(0.02 * mass))

    if tip_to_center <= near_center_r and tip_to_mass >= mass_far_r:
        return (
            True,
            f"tip near board center ({tx:.1f},{ty:.1f}) but motion mass at "
            f"({centroid[0]:.1f},{centroid[1]:.1f}) dist={tip_to_mass:.1f}",
        )
    if local < min_local and tip_to_mass >= mass_far_r:
        return (
            True,
            f"almost no motion near tip ({tx:.1f},{ty:.1f}): local={local}/{mass} "
            f"mass_dist={tip_to_mass:.1f}",
        )
    if local_frac < 0.01 and tip_to_mass >= mass_far_r:
        return (
            True,
            f"tip isolated from motion mass local_frac={local_frac:.3f} "
            f"dist={tip_to_mass:.1f}",
        )
    return False, ""


def _point_on_line_closest(
    x0: float, y0: float, vx: float, vy: float, px: float, py: float
) -> Tuple[float, float]:
    nn = float(np.hypot(vx, vy)) or 1.0
    vx, vy = vx / nn, vy / nn
    t = (px - x0) * vx + (py - y0) * vy
    return float(x0 + t * vx), float(y0 + t * vy)


def _apply_tip_and_score(
    fused: Any,
    tip: Tuple[float, float],
    *,
    out_size: int,
    board_calibrator: Any = None,
    per_cam: Optional[List[Any]] = None,
) -> None:
    """Set fused tip, rescore, and pin per-cam debug tips onto lines through ML tip."""
    from dart_detection import score_topdown_point

    tx, ty = float(tip[0]), float(tip[1])
    fused.tip_xy = (tx, ty)

    ring_cals = None
    cams = list(per_cam or fused.per_cam or [])
    if board_calibrator is not None and cams:
        try:
            ring_cals = [board_calibrator.get(int(r.cam_idx)) for r in cams]
        except Exception:
            ring_cals = None

    number, zone, score = score_topdown_point(
        tx, ty, out_size=out_size, cals=ring_cals
    )
    fused.segment_number = int(number)
    fused.zone_name = str(zone)
    fused.score = int(score)
    if score <= 0:
        fused.zone_name = "miss"
        if fused.reject_reason not in ("off_board", "off_board_raw", "miss_majority"):
            fused.reject_reason = "miss"
    else:
        fused.reject_reason = ""

    cam_ids = set(int(c) for c in (fused.cam_indices or []))
    for r in cams:
        if not getattr(r, "found", False):
            continue
        if cam_ids and int(getattr(r, "cam_idx", -1)) not in cam_ids:
            continue
        if getattr(r, "reject_reason", "") == "off_board_raw":
            continue
        try:
            proj = _point_on_line_closest(
                float(r.line_x0),
                float(r.line_y0),
                float(r.line_vx),
                float(r.line_vy),
                tx,
                ty,
            )
            r.tip_xy = proj
            r.radius_tip = proj
            r.line_x0 = tx
            r.line_y0 = ty
        except Exception:
            continue


def apply_ml_tip_pipeline(
    fused: Any,
    per_cam: List[Any],
    *,
    out_size: int,
    board_calibrator: Any = None,
) -> Any:
    """After FitLine fuse: optional dataset dump + optional ML tip override.

    When tip_detection is 'fitline', fused tip/score are unchanged (bit-identical
    aside from optional dataset PNG side-effect if that setting is on).

    ML override is skipped (FitLine kept) when the detector is unavailable,
    collapsed/untrained, or the predicted tip is implausible vs motion.
    """
    want_ml = _settings_tip_mode() == "ml_onnx"
    want_save = _settings_save_dataset()
    if not want_ml and not want_save:
        return fused
    if fused is None or not getattr(fused, "found", False):
        return fused

    fused_map = _fuse_motion_from_per_cam(list(per_cam or fused.per_cam or []), out_size)
    if fused_map is None:
        if want_ml:
            print("[ml_tip] no fused motion_raw — keeping FitLine tip", flush=True)
        return fused

    if want_save:
        save_ml_dataset_image(fused_map)

    if not want_ml:
        return fused

    # FitLine tip already on fused — keep it unless ML passes safety checks.
    fitline_tip = (
        float(fused.tip_xy[0]),
        float(fused.tip_xy[1]),
    )

    det = try_get_tip_detector()
    if det is None:
        print("[ml_tip] falling back to FitLine tip", flush=True)
        return fused

    collapsed, spread = detector_looks_collapsed(det)
    if collapsed:
        model_path = os.path.abspath(getattr(det, "model_path", "") or "")
        if model_path not in _collapse_warned_paths:
            _collapse_warned_paths.add(model_path)
            print(
                f"[ml_tip] WARNING: ONNX tip model looks untrained/collapsed "
                f"(synthetic blob spread={spread:.1f}px < 20). "
                f"Keeping FitLine tip for all hits until you retrain+export. "
                f"Pipeline: label tips -> python -m ml_tip.train -> "
                f"python -m ml_tip.export_onnx",
                flush=True,
            )
        return fused

    try:
        board_xy, model_xy, info = det.predict_on_fused(fused_map)
    except Exception as exc:
        print(f"[ml_tip] infer failed ({exc}) — keeping FitLine tip", flush=True)
        return fused

    # Mapping sanity: 200×200 pred → board via stretch inverse (identity when
    # out_size/src == 200). Never substitute board center.
    bx, by = float(board_xy[0]), float(board_xy[1])
    print(
        f"[ml_tip] pred model=({model_xy[0]:.1f},{model_xy[1]:.1f}) "
        f"board=({bx:.1f},{by:.1f}) scale=({info.scale_x:.3f},{info.scale_y:.3f}) "
        f"fitline=({fitline_tip[0]:.1f},{fitline_tip[1]:.1f})",
        flush=True,
    )

    bad, reason = ml_tip_implausible_vs_motion(
        (bx, by), fused_map, out_size=out_size
    )
    if bad:
        print(
            f"[ml_tip] WARNING: ML tip implausible vs motion ({reason}) — "
            f"keeping FitLine tip=({fitline_tip[0]:.1f},{fitline_tip[1]:.1f})",
            flush=True,
        )
        return fused

    _apply_tip_and_score(
        fused,
        (bx, by),
        out_size=out_size,
        board_calibrator=board_calibrator,
        per_cam=per_cam,
    )
    print(
        f"[ml_tip] tip applied model=({model_xy[0]:.1f},{model_xy[1]:.1f}) "
        f"board=({bx:.1f},{by:.1f}) "
        f"score={fused.score} zone={fused.zone_name}",
        flush=True,
    )
    return fused
