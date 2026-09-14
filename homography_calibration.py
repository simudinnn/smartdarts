"""Experimental single-projective calibration from many real board landmarks.

Does not replace legacy Stage1+Stage2. Callers keep the existing path on
fallback. FitLine / Y-refine / scoring math are not used here.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import board_calibration as bc

HOMOGRAPHY_OUT_SIZE = int(bc.FINAL_GEOM_DEBUG_SIZE)  # 300
HOMOGRAPHY_MIN_LANDMARKS = 24
HOMOGRAPHY_MIN_INLIERS = 16
HOMOGRAPHY_RANSAC_REPROJ = 3.5
HOMOGRAPHY_MAX_RMS_PX = 4.0
HOMOGRAPHY_MAX_ERR_PX = 12.0
HOMOGRAPHY_MIN_INLIER_FRAC = 0.50
HOMOGRAPHY_INNER_MIN_WIRES = 12
HOMOGRAPHY_OPTIONAL_INNER = True
_FAMILY_ORDER = (
    "double_outer",
    "double_inner",
    "triple_outer",
    "triple_inner",
    "bull",
)
# WDF/PDC face radii (mm). Canonical dest = RING_* = these values.
_WDF_RING_MM = {
    "double_outer": 170.0,
    "double_inner": 162.0,
    "triple_outer": 107.0,
    "triple_inner": 99.0,
    "bull_outer": 15.9,
    "bull_inner": 6.35,
}
_FACE_MM = 170.0
_CODE_RING_MM = {
    "double_outer": _FACE_MM * float(bc.RING_DOUBLE_OUTER_FRAC),
    "double_inner": _FACE_MM * float(bc.RING_DOUBLE_INNER_FRAC),
    "triple_outer": _FACE_MM * float(bc.RING_TRIPLE_OUTER_FRAC),
    "triple_inner": _FACE_MM * float(bc.RING_TRIPLE_INNER_FRAC),
    "bull_outer": _FACE_MM * float(bc.RING_BULL_OUTER_FRAC),
    "bull_inner": _FACE_MM * float(bc.RING_BULL_INNER_FRAC),
}

_RING_FRAC = {
    "double_outer": float(bc.RING_DOUBLE_OUTER_FRAC),
    "double_inner": float(bc.RING_DOUBLE_INNER_FRAC),
    "triple_outer": float(bc.RING_TRIPLE_OUTER_FRAC),
    "triple_inner": float(bc.RING_TRIPLE_INNER_FRAC),
    "bull_outer": float(bc.RING_BULL_OUTER_FRAC),
    "bull_inner": float(bc.RING_BULL_INNER_FRAC),
}

_SOURCE_EDGE = {
    "double_outer": "RG cluster OUTER edge of the double colour band (sisal, not Stage1 ellipse)",
    "double_inner": "RG cluster INNER edge near double (sisal, not metal)",
    "triple_outer": "RG cluster OUTER edge near triple (sisal, not metal)",
    "triple_inner": "RG cluster INNER edge near triple (sisal, not metal)",
    "bull": "detected bull center",
}


def _perf_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _canonical_xy(cx: float, cy: float, radius: float, angle_deg: float) -> Tuple[float, float]:
    """Scoring convention: 0°=up, clockwise. x=cx+r*sin, y=cy-r*cos."""
    st, ct = bc._angle_sin_cos(float(angle_deg))
    return (float(cx) + float(radius) * st, float(cy) + float(radius) * ct)


def _H_from_tuple(vals: Tuple[float, ...]) -> np.ndarray:
    return np.array(vals, dtype=np.float64).reshape(3, 3)


def _H_to_tuple(H: np.ndarray) -> Tuple[float, ...]:
    return tuple(float(x) for x in np.asarray(H, dtype=np.float64).reshape(-1))


def scaled_landmark_H(cal: bc.BoardCalibration, out_size: int) -> Optional[np.ndarray]:
    """Camera -> dest homography at ``out_size`` (includes segment-20 dest rotation)."""
    return bc._scaled_landmark_H(cal, out_size)


def scale_landmark_H_source(H9: Tuple[float, ...], scale: float) -> Tuple[float, ...]:
    """Adjust H when camera coordinates are multiplied by ``scale``."""
    s = float(scale)
    if abs(s - 1.0) < 1e-9:
        return H9
    H = _H_from_tuple(H9)
    sinv = np.array(
        [[1.0 / s, 0.0, 0.0], [0.0, 1.0 / s, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return _H_to_tuple(H @ sinv)


def _clockwise_wires(cal: bc.BoardCalibration) -> List[Tuple[int, float, float]]:
    """(clockwise_idx, src_angle_deg, dst_angle_deg) for each detected wire."""
    n = min(len(cal.wire_angles_deg), bc.SEGMENT_COUNT)
    if n < 8:
        return []
    sorted_idx = sorted(range(n), key=lambda k: float(cal.wire_angles_deg[k]) % 360.0)
    off = int(cal.segment20_offset) % n
    out: List[Tuple[int, float, float]] = []
    for j, k in enumerate(sorted_idx):
        src_ang = float(cal.wire_angles_deg[k]) % 360.0
        dst_ang = bc._dst_angle_for_clockwise_wire(j, off, n)
        out.append((j, src_ang, dst_ang))
    return out


def _camera_rg_mask(
    frame: np.ndarray,
    *,
    close: bool,
    clip_ellipse: Optional[bc.Ellipse] = None,
) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red, green, _, _ = bc._board_color_masks(hsv)
    board = cv2.bitwise_or(red, green)
    if clip_ellipse is not None:
        # Stage1 / hint ellipse = scoring-face outer double. Cut red surround sponge.
        clip = bc._board_ellipse_mask(board.shape, clip_ellipse, shrink=1.02)
        board = cv2.bitwise_and(board, clip)
    if close:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        board = cv2.morphologyEx(board, cv2.MORPH_CLOSE, kernel)
    return board


def _ray_rg_clusters(
    board: np.ndarray,
    bull: Tuple[float, float],
    angle_deg: float,
    r_lo: float,
    r_hi: float,
    step: float = 0.4,
) -> List[Tuple[float, float]]:
    """RG (r_inner, r_outer) runs along a scoring-convention ray."""
    h, w = board.shape[:2]
    st, ct = bc._angle_sin_cos(float(angle_deg))
    bx, by = float(bull[0]), float(bull[1])
    clusters: List[Tuple[float, float]] = []
    in_run = False
    r0 = r1 = 0.0
    r = float(r_lo)
    hi = float(r_hi)
    while r <= hi + 1e-6:
        x = int(round(bx + r * st))
        y = int(round(by + r * ct))
        if x < 0 or x >= w or y < 0 or y >= h:
            break
        hit = bool(board[y, x] > 0)
        if hit:
            if not in_run:
                in_run = True
                r0 = float(r)
            r1 = float(r)
        elif in_run:
            clusters.append((r0, r1))
            in_run = False
        r += float(step)
    if in_run:
        clusters.append((r0, r1))
    return clusters


def _pick_cluster_radius(
    clusters: List[Tuple[float, float]],
    expected: float,
    which: str,
) -> Optional[float]:
    if not clusters:
        return None
    best: Optional[Tuple[float, float]] = None
    best_err = 1e9
    for r0, r1 in clusters:
        r = r1 if which == "outer" else r0
        err = abs(r - expected)
        if err < best_err:
            best_err = err
            best = (r0, r1)
    if best is None:
        return None
    r = best[1] if which == "outer" else best[0]
    if abs(r - expected) > max(4.0, expected * 0.18):
        return None
    return float(r)


def _fit_camera_ring_ellipse(
    points: List[Tuple[float, float]],
    bull: Tuple[float, float],
    expected_r: float,
) -> Optional[bc.Ellipse]:
    if len(points) < 16 or expected_r < 4.0:
        return None
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    try:
        ellipse = cv2.fitEllipse(pts)
    except cv2.error:
        return None
    (ecx, ecy), (width, height), _ang = ellipse
    semi = 0.25 * (float(width) + float(height))
    if abs(semi - expected_r) > max(5.0, expected_r * 0.28):
        return None
    if math.hypot(ecx - bull[0], ecy - bull[1]) > max(6.0, expected_r * 0.40):
        return None
    return ellipse


def _fit_ring_ellipse_on_camera(
    board: np.ndarray,
    bull: Tuple[float, float],
    double_ell: bc.Ellipse,
    frac: float,
    which: str,
    *,
    d_cap_frac: Optional[float] = None,
    r_lo_frac: Optional[float] = None,
) -> Optional[bc.Ellipse]:
    pts: List[Tuple[float, float]] = []
    radii: List[float] = []
    d_dists: List[float] = []
    for deg in range(0, 360, 3):
        dpt = bc._ray_to_ellipse_edge_f(bull, float(deg), double_ell)
        if dpt is None:
            continue
        d_dist = math.hypot(dpt[0] - bull[0], dpt[1] - bull[1])
        if d_dist < 8.0:
            continue
        exp = d_dist * float(frac)
        r_lo = max(2.0, exp * 0.80)
        if r_lo_frac is not None:
            r_lo = max(2.0, d_dist * float(r_lo_frac))
        r_hi = exp * 1.20
        if d_cap_frac is not None:
            r_hi = min(r_hi, d_dist * float(d_cap_frac))
        if r_hi <= r_lo + 1.0:
            continue
        clusters = _ray_rg_clusters(board, bull, float(deg), r_lo, r_hi)
        r = _pick_cluster_radius(clusters, exp, which)
        if r is None:
            continue
        st, ct = bc._angle_sin_cos(float(deg))
        pts.append((bull[0] + r * st, bull[1] + r * ct))
        radii.append(r)
        d_dists.append(d_dist)
    expected_r = float(np.median(radii)) if radii else (
        float(np.median(d_dists)) * float(frac) if d_dists else 0.0
    )
    return _fit_camera_ring_ellipse(pts, bull, expected_r)


def _intersect_wires(
    bull: Tuple[float, float],
    ellipse: Optional[bc.Ellipse],
    wires: List[Tuple[int, float, float]],
) -> List[Tuple[int, Tuple[float, float], float]]:
    """Per clockwise wire: (idx, src_xy, confidence). No scaled-double fallback."""
    if ellipse is None:
        return []
    out: List[Tuple[int, Tuple[float, float], float]] = []
    for j, src_ang, _dst_ang in wires:
        pt = bc._ray_to_ellipse_edge_f(bull, src_ang, ellipse)
        if pt is None:
            continue
        out.append((j, (float(pt[0]), float(pt[1])), 0.90))
    return out


def detect_homography_landmarks(
    frame: np.ndarray,
    cal: bc.BoardCalibration,
    *,
    out_size: int = HOMOGRAPHY_OUT_SIZE,
) -> Tuple[List[dict], Dict[str, int]]:
    """Detect wire×ring landmarks on the raw camera frame.

    Bull and ellipse *hints* enter through Stage1 ``cal.center`` /
    ``cal.double_ellipse`` (click-bull / click-ellipse). RG samples are clipped
    to that ellipse so a red surround sponge or an off-center board cannot
    steal outer-double clusters.
    """
    counts = {
        "double_outer": 0,
        "triple_outer": 0,
        "double_inner": 0,
        "triple_inner": 0,
        "bull": 0,
    }
    if frame is None or cal.double_ellipse is None or not cal.has_wires():
        return [], counts

    size = int(out_size) if out_size > 0 else HOMOGRAPHY_OUT_SIZE
    cx, _view_r, board_r = bc._warp_radii(size)
    cy = cx
    bull = (float(cal.center[0]), float(cal.center[1]))
    double_ell = cal.double_ellipse
    wires = _clockwise_wires(cal)
    if len(wires) < 8:
        return [], counts

    board_raw = _camera_rg_mask(frame, close=False, clip_ellipse=double_ell)

    landmarks: List[dict] = []

    def _add(ring: str, src: Tuple[float, float], dst_ang: float, conf: float, wire_idx: int) -> None:
        r = board_r * float(_RING_FRAC[ring])
        tx, ty = _canonical_xy(cx, cy, r, dst_ang)
        landmarks.append(
            {
                "source": (float(src[0]), float(src[1])),
                "target": (float(tx), float(ty)),
                "confidence": float(conf),
                "ring": ring,
                "wire_idx": int(wire_idx),
                "source_edge": _SOURCE_EDGE.get(ring, ring),
                "target_frac": float(_RING_FRAC[ring]),
            }
        )
        counts[ring] = counts.get(ring, 0) + 1

    # Bull (helps pin the projective center).
    landmarks.append(
        {
            "source": bull,
            "target": (float(cx), float(cy)),
            "confidence": 1.0,
            "ring": "bull",
            "wire_idx": -1,
            "source_edge": _SOURCE_EDGE["bull"],
            "target_frac": 0.0,
        }
    )
    counts["bull"] = 1

    # Inner/triple RG fits first — outer double uses scaled inner as the search guide.
    # Stage1 double_ellipse is systematically too large vs the RG colour rings, so
    # wire∩ellipse vs canonical board_r was a different physical contour (RANSAC 0/20).
    t_outer = _fit_ring_ellipse_on_camera(
        board_raw, bull, double_ell, _RING_FRAC["triple_outer"], "outer", d_cap_frac=0.92
    )
    d_inner = _fit_ring_ellipse_on_camera(
        board_raw, bull, double_ell, _RING_FRAC["double_inner"], "inner", d_cap_frac=0.995
    )
    t_inner = _fit_ring_ellipse_on_camera(
        board_raw, bull, double_ell, _RING_FRAC["triple_inner"], "inner", d_cap_frac=0.90
    )

    d_outer_guide = double_ell
    inner_over_outer = max(float(_RING_FRAC["double_inner"]), 1e-6)
    if d_inner is not None:
        d_outer_guide = bc._scale_ellipse_axes(d_inner, 1.0 / inner_over_outer)
    # Cap at Stage1/hint outer ellipse — never search into surround.
    d_outer = _fit_ring_ellipse_on_camera(
        board_raw,
        bull,
        d_outer_guide,
        _RING_FRAC["double_outer"],
        "outer",
        d_cap_frac=1.00,
        r_lo_frac=0.90,
    )
    if d_outer is None:
        d_outer = _fit_ring_ellipse_on_camera(
            board_raw,
            bull,
            double_ell,
            _RING_FRAC["double_outer"],
            "outer",
            d_cap_frac=1.00,
            r_lo_frac=0.86,
        )
    if d_outer is None and d_inner is not None:
        d_outer = bc._scale_ellipse_axes(d_inner, 1.0 / inner_over_outer)

    d_outer_hits = _intersect_wires(bull, d_outer, wires)
    if len(d_outer_hits) >= 8:
        for j, src, conf in d_outer_hits:
            _add("double_outer", src, wires[j][2], conf, j)

    for j, src, conf in _intersect_wires(bull, t_outer, wires):
        _add("triple_outer", src, wires[j][2], conf, j)

    if HOMOGRAPHY_OPTIONAL_INNER:
        d_hits = _intersect_wires(bull, d_inner, wires)
        t_hits = _intersect_wires(bull, t_inner, wires)
        if len(d_hits) >= HOMOGRAPHY_INNER_MIN_WIRES:
            for j, src, conf in d_hits:
                _add("double_inner", src, wires[j][2], conf, j)
        if len(t_hits) >= HOMOGRAPHY_INNER_MIN_WIRES:
            for j, src, conf in t_hits:
                _add("triple_inner", src, wires[j][2], conf, j)

    return landmarks, counts


def _project_xy(H: np.ndarray, x: float, y: float) -> Optional[Tuple[float, float]]:
    pts = np.array([[[float(x), float(y)]]], dtype=np.float32)
    try:
        dst = cv2.perspectiveTransform(pts, H.astype(np.float64))
    except cv2.error:
        return None
    return (float(dst[0, 0, 0]), float(dst[0, 0, 1]))


def _reprojection_stats(
    H: np.ndarray,
    landmarks: List[dict],
    inlier_mask: Optional[np.ndarray],
) -> Dict[str, float]:
    errs: List[float] = []
    in_errs: List[float] = []
    for i, lm in enumerate(landmarks):
        pred = _project_xy(H, lm["source"][0], lm["source"][1])
        if pred is None:
            continue
        e = math.hypot(pred[0] - lm["target"][0], pred[1] - lm["target"][1])
        errs.append(e)
        if inlier_mask is not None and int(inlier_mask[i]) != 0:
            in_errs.append(e)
    use = in_errs if in_errs else errs
    if not use:
        return {"rms": 1e9, "max": 1e9, "n": 0.0}
    arr = np.asarray(use, dtype=np.float64)
    rms = float(np.sqrt(np.mean(arr * arr)))
    return {"rms": rms, "max": float(np.max(arr)), "n": float(len(use))}


def _mean_or_nan(vals: List[float]) -> float:
    return float(np.mean(vals)) if vals else float("nan")


def _max_or_nan(vals: List[float]) -> float:
    return float(np.max(vals)) if vals else float("nan")


def _camera_family_fracs(landmarks: List[dict]) -> Dict[str, List[float]]:
    """Per-wire source radius / double_outer source radius (camera space)."""
    src_xy: Dict[str, Dict[int, Tuple[float, float]]] = {}
    for lm in landmarks:
        ring = str(lm.get("ring", ""))
        w = int(lm.get("wire_idx", -1))
        src_xy.setdefault(ring, {})[w] = (float(lm["source"][0]), float(lm["source"][1]))
    fracs: Dict[str, List[float]] = {k: [] for k in _FAMILY_ORDER}
    if -1 not in src_xy.get("bull", {}) or "double_outer" not in src_xy:
        return fracs
    bx, by = src_xy["bull"][-1]
    dout_r: Dict[int, float] = {}
    for w, (x, y) in src_xy["double_outer"].items():
        dout_r[w] = math.hypot(x - bx, y - by)
    for ring in _FAMILY_ORDER:
        if ring == "bull":
            continue
        for w, (x, y) in src_xy.get(ring, {}).items():
            den = dout_r.get(w)
            if den is None or den < 1e-6:
                continue
            fracs[ring].append(math.hypot(x - bx, y - by) / den)
    return fracs


def profile_homography_families(
    H: np.ndarray,
    landmarks: List[dict],
    mask: np.ndarray,
    *,
    out_size: int,
) -> Dict[str, dict]:
    """Per-family inlier counts + reprojection / radial errors using H + RANSAC mask."""
    cx, _, board_r = bc._warp_radii(int(out_size))
    cy = cx
    mask_u = np.asarray(mask).reshape(-1)
    out: Dict[str, dict] = {}
    for ring in _FAMILY_ORDER:
        idxs = [i for i, lm in enumerate(landmarks) if str(lm.get("ring")) == ring]
        n = len(idxs)
        n_in = 0
        reproj: List[float] = []
        radial: List[float] = []
        signed: List[float] = []
        for i in idxs:
            inlier = i < len(mask_u) and int(mask_u[i]) != 0
            if inlier:
                n_in += 1
            lm = landmarks[i]
            pred = _project_xy(H, lm["source"][0], lm["source"][1])
            tx, ty = float(lm["target"][0]), float(lm["target"][1])
            if pred is None:
                continue
            e = math.hypot(pred[0] - tx, pred[1] - ty)
            r_pred = math.hypot(pred[0] - cx, pred[1] - cy)
            r_tgt = math.hypot(tx - cx, ty - cy)
            srad = r_pred - r_tgt
            reproj.append(e)
            radial.append(abs(srad))
            signed.append(srad)
        canon = float(_RING_FRAC.get(ring, 0.0))
        out[ring] = {
            "n": n,
            "inliers": n_in,
            "outliers": n - n_in,
            "mean_reproj_px": _mean_or_nan(reproj),
            "max_reproj_px": _max_or_nan(reproj),
            "mean_radial_px": _mean_or_nan(radial),
            "signed_mean_radial_px": _mean_or_nan(signed),
            "canonical_frac": canon,
            "canonical_r_px": board_r * canon if ring != "bull" else 0.0,
            "source_edge": _SOURCE_EDGE.get(ring, ring),
        }
    out["_camera_fracs"] = _camera_family_fracs(landmarks)  # type: ignore[assignment]
    return out


def _fmt_n(x: float, nd: int = 2) -> str:
    if x != x:  # NaN
        return "nan"
    return f"{x:.{nd}f}"


def log_homography_inliers(
    *,
    cam_idx: Optional[int],
    profile: Dict[str, dict],
    ransac_thresh: float,
) -> None:
    print(f"[HOMOGRAPHY_INLIERS] cam={cam_idx} ransac_thresh={ransac_thresh:.2f}px", flush=True)
    for ring in _FAMILY_ORDER:
        row = profile.get(ring) or {}
        n = int(row.get("n", 0))
        n_in = int(row.get("inliers", 0))
        print(
            f"[HOMOGRAPHY_INLIERS] {ring} = {n_in}/{n}  "
            f"mean_reproj={_fmt_n(float(row.get('mean_reproj_px', float('nan'))))} "
            f"max_reproj={_fmt_n(float(row.get('max_reproj_px', float('nan'))))} "
            f"mean_radial={_fmt_n(float(row.get('mean_radial_px', float('nan'))))} "
            f"signed_mean_radial={_fmt_n(float(row.get('signed_mean_radial_px', float('nan'))))} "
            f"target_frac={float(row.get('canonical_frac', 0.0)):.4f} "
            f"target_r={_fmt_n(float(row.get('canonical_r_px', float('nan'))))}px",
            flush=True,
        )
    cam_fracs = profile.get("_camera_fracs") or {}
    if isinstance(cam_fracs, dict) and cam_fracs:
        parts = []
        for ring in ("double_outer", "double_inner", "triple_outer", "triple_inner"):
            vals = cam_fracs.get(ring) or []
            if not vals:
                continue
            parts.append(
                f"{ring}:src_frac={float(np.mean(vals)):.4f}"
                f"(canon={float(_RING_FRAC[ring]):.4f},n={len(vals)})"
            )
        if parts:
            print("[HOMOGRAPHY_INLIERS] camera_r/r_Dout " + " ".join(parts), flush=True)


def diagnose_dead_homography_families(
    *,
    cam_idx: Optional[int],
    profile: Dict[str, dict],
    landmarks: List[dict],
    H: np.ndarray,
    mask: np.ndarray,
    out_size: int,
) -> None:
    """If a whole family is ~0 inliers, explain source vs canonical target mismatch."""
    dead: List[str] = []
    for ring in _FAMILY_ORDER:
        if ring == "bull":
            row = profile.get(ring) or {}
            if int(row.get("n", 0)) >= 1 and int(row.get("inliers", 0)) == 0:
                dead.append(ring)
            continue
        row = profile.get(ring) or {}
        n = int(row.get("n", 0))
        n_in = int(row.get("inliers", 0))
        if n >= 12 and n_in <= max(2, int(round(0.15 * n))):
            dead.append(ring)
    if not dead:
        return

    cam_fracs = profile.get("_camera_fracs") or {}
    cx, _, board_r = bc._warp_radii(int(out_size))
    print(
        f"[HOMOGRAPHY_FAMILY_DIAG] cam={cam_idx} dead_families={dead} "
        f"(RANSAC kept a systematic family gap; not raising threshold)",
        flush=True,
    )
    print(
        "[HOMOGRAPHY_FAMILY_DIAG] source_vs_target: "
        "double_outer=Stage1 ellipse outer | "
        "double/triple inner=RG sisal INNER edge | "
        "triple_outer=RG sisal OUTER edge | "
        "canonical targets=WDF metal-wire radii (RING_*)",
        flush=True,
    )
    print(
        "[HOMOGRAPHY_FAMILY_DIAG] canonical_mm "
        + " ".join(
            f"{k}:code={_CODE_RING_MM[k]:.1f}/WDF={_WDF_RING_MM[k]:.1f}"
            for k in ("double_outer", "double_inner", "triple_outer", "triple_inner")
        ),
        flush=True,
    )

    # Band-center hypotheis: (inner+outer)/2.
    band_center_frac = {
        "double_inner": 0.5 * (_RING_FRAC["double_inner"] + _RING_FRAC["double_outer"]),
        "double_outer": 0.5 * (_RING_FRAC["double_inner"] + _RING_FRAC["double_outer"]),
        "triple_inner": 0.5 * (_RING_FRAC["triple_inner"] + _RING_FRAC["triple_outer"]),
        "triple_outer": 0.5 * (_RING_FRAC["triple_inner"] + _RING_FRAC["triple_outer"]),
    }

    for ring in dead:
        row = profile.get(ring) or {}
        src_fracs = cam_fracs.get(ring) if isinstance(cam_fracs, dict) else None
        src_mean = float(np.mean(src_fracs)) if src_fracs else float("nan")
        canon = float(_RING_FRAC.get(ring, 0.0))
        signed = float(row.get("signed_mean_radial_px", float("nan")))
        mean_e = float(row.get("mean_reproj_px", float("nan")))
        edge = _SOURCE_EDGE.get(ring, ring)
        why: List[str] = []
        if src_fracs:
            # Compare camera frac to other families' canonical fracs (swap?).
            others = {
                k: float(_RING_FRAC[k])
                for k in ("double_outer", "double_inner", "triple_outer", "triple_inner")
                if k != ring
            }
            nearest = min(others.items(), key=lambda kv: abs(kv[1] - src_mean))
            if abs(src_mean - canon) > 0.012 and abs(src_mean - nearest[1]) + 0.008 < abs(
                src_mean - canon
            ):
                why.append(
                    f"camera frac {src_mean:.4f} closer to {nearest[0]} "
                    f"({nearest[1]:.4f}) than to {ring} ({canon:.4f}) — possible family swap"
                )
            bc_frac = band_center_frac.get(ring)
            if bc_frac is not None and abs(src_mean - bc_frac) + 0.004 < abs(src_mean - canon):
                why.append(
                    f"camera frac {src_mean:.4f} closer to COLOR BAND CENTER "
                    f"({bc_frac:.4f}) than to {ring} wire ({canon:.4f})"
                )
            if abs(src_mean - canon) > 0.008:
                dpx = (src_mean - canon) * board_r
                why.append(
                    f"constant radial scale bias: src_frac-canon={src_mean - canon:+.4f} "
                    f"~ {dpx:+.2f}px at board_r={board_r:.1f}"
                )
        if signed == signed:
            direction = (
                "OUTSIDE canonical (source farther from bull than target)"
                if signed > 0
                else "INSIDE canonical (source closer to bull than target)"
            )
            why.append(
                f"after H: signed mean radial={signed:+.2f}px → projected points sit {direction}"
            )
        # inner/outer swap inside same ring
        if ring.endswith("_inner") and src_fracs:
            outer_name = ring.replace("_inner", "_outer")
            outer_canon = float(_RING_FRAC.get(outer_name, 1.0))
            if abs(src_mean - outer_canon) < 0.015:
                why.append(
                    f"inner family camera frac≈{src_mean:.4f} matches {outer_name} "
                    f"({outer_canon:.4f}) — inner/outer likely swapped"
                )
        if ring.endswith("_outer") and src_fracs:
            inner_name = ring.replace("_outer", "_inner")
            inner_canon = float(_RING_FRAC.get(inner_name, 1.0))
            if abs(src_mean - inner_canon) < 0.015:
                why.append(
                    f"outer family camera frac≈{src_mean:.4f} matches {inner_name} "
                    f"({inner_canon:.4f}) — inner/outer likely swapped"
                )
        mm_code = _CODE_RING_MM.get(ring)
        mm_wdf = _WDF_RING_MM.get(ring)
        if mm_code is not None and mm_wdf is not None:
            dmm = mm_code - mm_wdf
            dpx = (dmm / 170.0) * board_r
            why.append(
                f"code vs WDF radius: {mm_code:.1f}mm vs {mm_wdf:.1f}mm "
                f"({dmm:+.1f}mm ~ {dpx:+.2f}px) — too small to dump a whole family at "
                f"RANSAC {HOMOGRAPHY_RANSAC_REPROJ:.1f}px"
            )
        why.append(f"source measurement: {edge}")
        why.append(
            f"canonical target: circle r=board_r*{canon:.4f} "
            f"(metal-wire RING_* frac, not band center)"
        )
        if ring == "double_outer" and isinstance(cam_fracs, dict):
            di_vals = cam_fracs.get("double_inner") or []
            to_vals = cam_fracs.get("triple_outer") or []
            if di_vals:
                di_m = float(np.mean(di_vals))
                oversize = float(_RING_FRAC["double_inner"]) / max(di_m, 1e-6)
                extra_px = (oversize - 1.0) * board_r
                why.append(
                    f"RG double_inner src_frac={di_m:.4f} vs canon "
                    f"{float(_RING_FRAC['double_inner']):.4f}; Stage1 ellipse is "
                    f"{oversize:.3f}x that implied outer (~{extra_px:+.1f}px). "
                    f"double_outer should be RG outer-double, not the Stage1 ellipse."
                )
            if to_vals:
                to_m = float(np.mean(to_vals))
                why.append(
                    f"RG triple_outer src_frac={to_m:.4f} vs canon "
                    f"{float(_RING_FRAC['triple_outer']):.4f} (sisal outer edge vs metal wire)."
                )
        print(
            f"[HOMOGRAPHY_FAMILY_DIAG] {ring}: inliers={row.get('inliers')}/{row.get('n')} "
            f"mean_reproj={_fmt_n(mean_e)} src_frac={_fmt_n(src_mean, 4)} canon_frac={canon:.4f}",
            flush=True,
        )
        for line in why:
            print(f"[HOMOGRAPHY_FAMILY_DIAG]   - {line}", flush=True)

    # inner vs outer camera order check
    if isinstance(cam_fracs, dict):
        for pair in (("double_inner", "double_outer"), ("triple_inner", "triple_outer")):
            a = cam_fracs.get(pair[0]) or []
            b = cam_fracs.get(pair[1]) or []
            if a and b:
                ma, mb = float(np.mean(a)), float(np.mean(b))
                if ma >= mb:
                    print(
                        f"[HOMOGRAPHY_FAMILY_DIAG] ORDER BUG: {pair[0]} src_frac={ma:.4f} "
                        f">= {pair[1]} src_frac={mb:.4f} (inner should be smaller)",
                        flush=True,
                    )


def find_landmark_homography(
    landmarks: List[dict],
    *,
    cam_idx: Optional[int] = None,
    out_size: int = HOMOGRAPHY_OUT_SIZE,
) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, float]]]:
    if len(landmarks) < HOMOGRAPHY_MIN_LANDMARKS:
        return None
    src = np.array([lm["source"] for lm in landmarks], dtype=np.float64)
    dst = np.array([lm["target"] for lm in landmarks], dtype=np.float64)
    H, mask = cv2.findHomography(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(HOMOGRAPHY_RANSAC_REPROJ),
        maxIters=4000,
        confidence=0.995,
    )
    if H is None or mask is None:
        return None
    mask_u = mask.reshape(-1).astype(np.uint8)
    n_in = int(np.count_nonzero(mask_u))
    # Profile against the RANSAC H + mask (before inlier LS refine).
    fam = profile_homography_families(H, landmarks, mask_u, out_size=int(out_size))
    log_homography_inliers(
        cam_idx=cam_idx, profile=fam, ransac_thresh=float(HOMOGRAPHY_RANSAC_REPROJ)
    )
    diagnose_dead_homography_families(
        cam_idx=cam_idx,
        profile=fam,
        landmarks=landmarks,
        H=H,
        mask=mask_u,
        out_size=int(out_size),
    )
    if n_in < HOMOGRAPHY_MIN_INLIERS:
        return None
    if n_in / max(1, len(landmarks)) < HOMOGRAPHY_MIN_INLIER_FRAC:
        return None
    # Least-squares refine on inliers (more stable / repeatable than RANSAC H).
    in_src = src[mask_u != 0]
    in_dst = dst[mask_u != 0]
    H2, _ = cv2.findHomography(in_src, in_dst, method=0)
    if H2 is None:
        H2 = H
    stats = _reprojection_stats(H2, landmarks, mask_u)
    stats["n_landmarks"] = float(len(landmarks))
    stats["n_inliers"] = float(n_in)
    stats["n_outliers"] = float(len(landmarks) - n_in)
    stats["family"] = {k: fam[k] for k in _FAMILY_ORDER}
    if stats["rms"] > HOMOGRAPHY_MAX_RMS_PX or stats["max"] > HOMOGRAPHY_MAX_ERR_PX:
        return None
    return H2.astype(np.float64), mask_u, stats


def _apply_homography_to_cal(
    cal: bc.BoardCalibration,
    H: np.ndarray,
    out_size: int,
    ring_ellipses: Optional[Dict[str, bc.Ellipse]] = None,
) -> bc.BoardCalibration:
    size = int(out_size)
    _, _, board_r = bc._warp_radii(size)
    ell = ring_ellipses if ring_ellipses is not None else bc._nominal_final_ring_ellipses(size)
    fracs = bc._ring_fracs_from_ellipses(ell, board_r)
    if not bc._ring_fracs_sane(fracs, board_r=board_r):
        ell = bc._nominal_final_ring_ellipses(size)
        fracs = bc._ring_fracs_from_ellipses(ell, board_r)
    return replace(
        cal,
        warp_mode="homography_landmarks",
        landmark_H=_H_to_tuple(H),
        landmark_H_size=size,
        landmark_H_offset=int(cal.segment20_offset) % bc.SEGMENT_COUNT,
        measured_double_outer_frac=float(fracs[0]),
        measured_double_inner_frac=float(fracs[1]),
        measured_triple_outer_frac=float(fracs[2]),
        measured_triple_inner_frac=float(fracs[3]),
        warp_ring_size=size,
        warp_double_outer=ell["double_outer"],
        warp_double_inner=ell["double_inner"],
        warp_triple_outer=ell["triple_outer"],
        warp_triple_inner=ell["triple_inner"],
        # No Stage2 egg→circle in this mode.
        warp1_double_outer=None,
        warp1_triple_outer=None,
        warp1_center=None,
        stage2_double_inner_ratio=0.0,
        stage2_triple_inner_ratio=0.0,
    )


def _draw_canonical_geometry(
    img: np.ndarray,
    out_size: int,
    ring_ellipses: Optional[Dict[str, bc.Ellipse]] = None,
) -> np.ndarray:
    vis = img.copy()
    cx, _view_r, board_r = bc._warp_radii(int(out_size))
    cy = cx
    cxy = (int(round(cx)), int(round(cy)))
    if ring_ellipses is not None:
        for key, col in (
            ("triple_inner", (0, 255, 128)),
            ("triple_outer", (0, 255, 128)),
            ("double_inner", (0, 165, 255)),
            ("double_outer", (0, 165, 255)),
        ):
            ell = ring_ellipses.get(key)
            if ell is not None:
                bc._draw_cv_ellipse(vis, ell, col, 1)
    else:
        rings = {
            "triple_inner": (bc.RING_TRIPLE_INNER_FRAC, (0, 255, 128)),
            "triple_outer": (bc.RING_TRIPLE_OUTER_FRAC, (0, 255, 128)),
            "double_inner": (bc.RING_DOUBLE_INNER_FRAC, (0, 165, 255)),
            "double_outer": (bc.RING_DOUBLE_OUTER_FRAC, (0, 165, 255)),
        }
        for _name, (frac, col) in rings.items():
            rr = max(1, int(round(board_r * float(frac))))
            cv2.circle(vis, cxy, rr, col, 1, cv2.LINE_AA)
    for frac, col in (
        (bc.RING_BULL_INNER_FRAC, (255, 80, 80)),
        (bc.RING_BULL_OUTER_FRAC, (255, 80, 80)),
    ):
        rr = max(1, int(round(board_r * float(frac))))
        cv2.circle(vis, cxy, rr, col, 1, cv2.LINE_AA)
    r_wire = board_r
    for j in range(bc.SEGMENT_COUNT):
        ang = (bc.SEG20_LEFT_WIRE_DST_DEG + j * bc.SEGMENT_STEP_DEG) % 360.0
        st, ct = bc._angle_sin_cos(ang)
        x1 = int(round(cx + r_wire * st))
        y1 = int(round(cy + r_wire * ct))
        cv2.line(vis, cxy, (x1, y1), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.drawMarker(vis, cxy, (255, 0, 255), cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)
    return vis


def _ring_error_summary(warped: np.ndarray, out_size: int) -> Tuple[str, Dict[str, float]]:
    size = int(out_size)
    cx, _, board_r = bc._warp_radii(size)
    d1 = bc.detect_double_outer_ellipse_on_warp(warped, nominal_board_r=board_r)
    t1 = bc.detect_triple_outer_ellipse_on_warp(warped, d1) if d1 is not None else None
    parts: List[str] = []
    nums: Dict[str, float] = {}
    if d1 is not None:
        ecx, ecy = bc._ellipse_center(d1)
        semi = bc._mean_ellipse_semi(d1)
        dc = math.hypot(ecx - cx, ecy - cx)
        dr = abs(semi - board_r)
        nums["double_center"] = dc
        nums["double_r"] = dr
        parts.append(f"D_out(c={dc:.2f},r={dr:.2f})")
    else:
        parts.append("D_out=miss")
    if t1 is not None:
        ecx, ecy = bc._ellipse_center(t1)
        semi = bc._mean_ellipse_semi(t1)
        target = board_r * float(bc.RING_TRIPLE_OUTER_FRAC)
        dc = math.hypot(ecx - cx, ecy - cx)
        dr = abs(semi - target)
        nums["triple_center"] = dc
        nums["triple_r"] = dr
        parts.append(f"T_out(c={dc:.2f},r={dr:.2f})")
    else:
        parts.append("T_out=miss")
    return ",".join(parts), nums


def _wire_error_summary(
    H: np.ndarray,
    cal: bc.BoardCalibration,
    out_size: int,
) -> Tuple[str, Dict[str, float]]:
    cx, _, _board_r = bc._warp_radii(int(out_size))
    deltas: List[float] = []
    for _j, src_ang, dst_ang in _clockwise_wires(cal):
        pt = bc._ray_to_ellipse_edge_f(cal.center, src_ang, cal.double_ellipse)
        if pt is None:
            continue
        pred = _project_xy(H, pt[0], pt[1])
        if pred is None:
            continue
        th = math.degrees(math.atan2(pred[0] - cx, -(pred[1] - cx))) % 360.0
        deltas.append(bc._ang_abs_delta_deg(th, dst_ang))
    if not deltas:
        return "miss", {}
    arr = np.asarray(deltas, dtype=np.float64)
    mean_d = float(np.mean(arr))
    max_d = float(np.max(arr))
    return f"mean={mean_d:.3f}deg max={max_d:.3f}deg n={len(deltas)}", {
        "wire_mean_deg": mean_d,
        "wire_max_deg": max_d,
    }


def _center_error_px(warped: np.ndarray, H: np.ndarray, cal: bc.BoardCalibration, out_size: int) -> float:
    cx, _, _ = bc._warp_radii(int(out_size))
    bull = bc.detect_bull_center_on_warp(warped)
    if bull is not None:
        return math.hypot(bull[0] - cx, bull[1] - cx)
    pred = _project_xy(H, float(cal.center[0]), float(cal.center[1]))
    if pred is None:
        return 1e9
    return math.hypot(pred[0] - cx, pred[1] - cx)


def _save_homography_debug(
    cam_idx: Optional[int],
    warped: np.ndarray,
    out_size: int,
    ring_ellipses: Optional[Dict[str, bc.Ellipse]] = None,
) -> str:
    vis = _draw_canonical_geometry(warped, out_size, ring_ellipses=ring_ellipses)
    d = bc._geom_debug_dir()
    tag = int(cam_idx) if cam_idx is not None else 0
    path = os.path.join(d, f"cam{tag}_homography_debug.png")
    cv2.imwrite(path, vis)
    return path


def _crop_zone(img: np.ndarray, cx: float, cy: float, half: int = 48) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = max(0, int(round(cx)) - half)
    y0 = max(0, int(round(cy)) - half)
    x1 = min(w, x0 + 2 * half)
    y1 = min(h, y0 + 2 * half)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((2 * half, 2 * half, 3), dtype=np.uint8)
    return cv2.resize(crop, (2 * half, 2 * half), interpolation=cv2.INTER_NEAREST)


def save_homography_vs_legacy(
    cam_idx: Optional[int],
    frame: np.ndarray,
    hom_cal: bc.BoardCalibration,
    legacy_cal: bc.BoardCalibration,
    *,
    out_size: int = HOMOGRAPHY_OUT_SIZE,
) -> Optional[str]:
    size = int(out_size)
    hom_img = bc.warp_topdown_from_calibration(frame, hom_cal, out_size=size)
    leg_img = bc.warp_topdown_from_calibration(frame, legacy_cal, out_size=size)
    if hom_img is None or leg_img is None:
        return None
    hom_vis = _draw_canonical_geometry(
        hom_img, size, ring_ellipses=bc.scoring_ring_ellipses(size, cal=hom_cal)
    )
    leg_vis = _draw_canonical_geometry(leg_img, size)
    cx, _, board_r = bc._warp_radii(size)
    # T20 (12h) at ~outer double / triple.
    t20 = _canonical_xy(cx, cx, board_r * 0.92, 0.0)
    t20_t = _canonical_xy(cx, cx, board_r * bc.RING_TRIPLE_OUTER_FRAC, 0.0)
    # 16/8 wire ≈ 243° (right wire of 16).
    a_168 = (bc.SEG20_LEFT_WIRE_DST_DEG + 14 * bc.SEGMENT_STEP_DEG) % 360.0
    z168_d = _canonical_xy(cx, cx, board_r * 0.97, a_168)
    z168_t = _canonical_xy(cx, cx, board_r * bc.RING_TRIPLE_OUTER_FRAC, a_168)

    def _panel(vis: np.ndarray, title: str) -> np.ndarray:
        crop_size = 96
        crops = [
            cv2.resize(_crop_zone(vis, t20[0], t20[1], 52), (crop_size, crop_size), interpolation=cv2.INTER_NEAREST),
            cv2.resize(_crop_zone(vis, t20_t[0], t20_t[1], 40), (crop_size, crop_size), interpolation=cv2.INTER_NEAREST),
            cv2.resize(_crop_zone(vis, z168_d[0], z168_d[1], 52), (crop_size, crop_size), interpolation=cv2.INTER_NEAREST),
            cv2.resize(_crop_zone(vis, z168_t[0], z168_t[1], 40), (crop_size, crop_size), interpolation=cv2.INTER_NEAREST),
        ]
        row = np.hstack(crops)
        header = vis.copy()
        cv2.putText(
            header, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA
        )
        # labels under crops
        labeled = np.zeros((row.shape[0] + 18, row.shape[1], 3), dtype=np.uint8)
        labeled[18:] = row
        for i, lab in enumerate(("T20 D", "T20 T", "16/8 D", "16/8 T")):
            x = i * (row.shape[1] // 4) + 6
            cv2.putText(
                labeled, lab, (x, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA
            )
        scale = header.shape[1] / max(1, labeled.shape[1])
        new_h = max(1, int(round(labeled.shape[0] * scale)))
        labeled = cv2.resize(labeled, (header.shape[1], new_h), interpolation=cv2.INTER_NEAREST)
        return np.vstack([header, labeled])

    left = _panel(leg_vis, "LEGACY Stage1+Stage2")
    right = _panel(hom_vis, "HOMOGRAPHY landmarks")
    # equalize heights
    if left.shape[0] != right.shape[0]:
        hh = max(left.shape[0], right.shape[0])
        def _pad(im: np.ndarray) -> np.ndarray:
            if im.shape[0] == hh:
                return im
            pad = np.zeros((hh - im.shape[0], im.shape[1], 3), dtype=np.uint8)
            return np.vstack([im, pad])
        left, right = _pad(left), _pad(right)
    combo = np.hstack([left, right])
    d = bc._geom_debug_dir()
    tag = int(cam_idx) if cam_idx is not None else 0
    path = os.path.join(d, f"cam{tag}_homography_vs_legacy.png")
    cv2.imwrite(path, combo)
    return path


def _ring_measure_summary(
    ring_ellipses: Optional[Dict[str, bc.Ellipse]],
    out_size: int,
) -> str:
    _, _, br = bc._warp_radii(int(out_size))
    if ring_ellipses is None:
        return "nominal"
    fracs = bc._ring_fracs_from_ellipses(ring_ellipses, br)
    names = ("D", "Di", "To", "Ti")
    noms = (
        bc.RING_DOUBLE_OUTER_FRAC,
        bc.RING_DOUBLE_INNER_FRAC,
        bc.RING_TRIPLE_OUTER_FRAC,
        bc.RING_TRIPLE_INNER_FRAC,
    )
    parts = []
    for name, frac, nom in zip(names, fracs, noms):
        px = br * frac
        dpx = br * (frac - nom)
        parts.append(f"{name}={px:.1f}({dpx:+.2f})")
    return " ".join(parts)


def try_homography_landmark_calibration(
    frame: np.ndarray,
    cal: bc.BoardCalibration,
    *,
    cam_idx: Optional[int] = None,
    out_size: int = HOMOGRAPHY_OUT_SIZE,
    save_debug: bool = True,
) -> Tuple[Optional[bc.BoardCalibration], Dict[str, object]]:
    """Fit one global homography. Returns (None, stats) on reject — caller uses legacy."""
    t_total = time.perf_counter()
    stats: Dict[str, object] = {
        "ok": False,
        "reason": "",
        "cam": cam_idx,
        "landmarks": 0,
        "inliers": 0,
        "outliers": 0,
        "rms_px": None,
        "max_px": None,
        "center_error_px": None,
        "ring_error": "",
        "wire_error": "",
        "landmark_counts": {},
        "landmark_detection_ms": 0.0,
        "homography_ms": 0.0,
        "warp_ms": 0.0,
        "validation_ms": 0.0,
        "ring_measure": "",
        "ring_measure_ms": 0.0,
        "total_ms": 0.0,
    }
    if frame is None or not bc._is_calibration_complete(cal):
        stats["reason"] = "incomplete_stage1"
        stats["total_ms"] = _perf_ms(t_total)
        return None, stats

    size = int(out_size) if out_size > 0 else HOMOGRAPHY_OUT_SIZE
    t0 = time.perf_counter()
    landmarks, counts = detect_homography_landmarks(frame, cal, out_size=size)
    stats["landmark_detection_ms"] = _perf_ms(t0)
    stats["landmarks"] = len(landmarks)
    stats["landmark_counts"] = counts
    if len(landmarks) < HOMOGRAPHY_MIN_LANDMARKS:
        stats["reason"] = f"too_few_landmarks:{len(landmarks)}<{HOMOGRAPHY_MIN_LANDMARKS}"
        stats["total_ms"] = _perf_ms(t_total)
        return None, stats
    if int(counts.get("double_outer", 0)) < 12 or int(counts.get("triple_outer", 0)) < 12:
        stats["reason"] = (
            f"missing_primary_rings:D={counts.get('double_outer')} T={counts.get('triple_outer')}"
        )
        stats["total_ms"] = _perf_ms(t_total)
        return None, stats

    t0 = time.perf_counter()
    found = find_landmark_homography(landmarks, cam_idx=cam_idx, out_size=size)
    stats["homography_ms"] = _perf_ms(t0)
    if found is None:
        stats["reason"] = "ransac_reject"
        stats["total_ms"] = _perf_ms(t_total)
        return None, stats
    H, mask, hstats = found
    stats["inliers"] = int(hstats["n_inliers"])
    stats["outliers"] = int(hstats["n_outliers"])
    stats["rms_px"] = float(hstats["rms"])
    stats["max_px"] = float(hstats["max"])

    t0 = time.perf_counter()
    warped = cv2.warpPerspective(
        frame,
        H,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    stats["warp_ms"] = _perf_ms(t0)

    t0 = time.perf_counter()
    center_err = _center_error_px(warped, H, cal, size)
    ring_txt, _ring_nums = _ring_error_summary(warped, size)
    wire_txt, _wire_nums = _wire_error_summary(H, cal, size)
    stats["center_error_px"] = float(center_err)
    stats["ring_error"] = ring_txt
    stats["wire_error"] = wire_txt
    t_ring = time.perf_counter()
    _, _, board_r = bc._warp_radii(size)
    ring_ell = bc.detect_circular_rings_on_aligned_warp(warped, nominal_board_r=board_r)
    stats["ring_measure_ms"] = _perf_ms(t_ring)
    stats["ring_measure"] = _ring_measure_summary(ring_ell, size)
    stats["H"] = _H_to_tuple(H)
    stats["landmarks_xy"] = [
        (lm["source"], lm["target"], lm["ring"], float(lm["confidence"])) for lm in landmarks
    ]
    if save_debug:
        _save_homography_debug(cam_idx, warped, size, ring_ellipses=ring_ell)
    stats["validation_ms"] = _perf_ms(t0)
    stats["ok"] = True
    stats["total_ms"] = _perf_ms(t_total)

    new_cal = _apply_homography_to_cal(cal, H, size, ring_ellipses=ring_ell)
    print(
        f"[HOMOGRAPHY_RINGS] cam={cam_idx} {stats['ring_measure']} "
        f"ms={float(stats['ring_measure_ms']):.1f}",
        flush=True,
    )
    return new_cal, stats


def log_homography_cal(stats: Dict[str, object]) -> None:
    cam = stats.get("cam")
    print(
        f"[HOMOGRAPHY_CAL] cam={cam} landmarks={stats.get('landmarks')} "
        f"inliers={stats.get('inliers')} outliers={stats.get('outliers')} "
        f"rms_px={stats.get('rms_px')} max_px={stats.get('max_px')} "
        f"center_error_px={stats.get('center_error_px')} "
        f"ring_error={stats.get('ring_error')} wire_error={stats.get('wire_error')} "
        f"ring_measure={stats.get('ring_measure') or '-'} "
        f"ok={stats.get('ok')} reason={stats.get('reason') or '-'} "
        f"counts={stats.get('landmark_counts')} "
        f"landmark_detection_ms={float(stats.get('landmark_detection_ms') or 0):.1f} "
        f"homography_ms={float(stats.get('homography_ms') or 0):.1f} "
        f"warp_ms={float(stats.get('warp_ms') or 0):.1f} "
        f"validation_ms={float(stats.get('validation_ms') or 0):.1f} "
        f"total_ms={float(stats.get('total_ms') or 0):.1f}",
        flush=True,
    )


def apply_homography_mode_or_none(
    frame: np.ndarray,
    cal: bc.BoardCalibration,
    *,
    cam_idx: Optional[int] = None,
) -> Optional[bc.BoardCalibration]:
    """If CALIBRATION_MODE is homography_landmarks, try it. None => legacy fallback."""
    mode = str(getattr(bc, "CALIBRATION_MODE", "legacy")).strip().lower()
    if mode not in ("homography_landmarks", "homography", "landmarks"):
        return None
    new_cal, stats = try_homography_landmark_calibration(
        frame, cal, cam_idx=cam_idx, save_debug=bool(bc.DEBUG_SAVE_CANNY_EDGES)
    )
    log_homography_cal(stats)
    if new_cal is None:
        print(
            f"[HOMOGRAPHY_CAL] cam={cam_idx} fallback=legacy ({stats.get('reason')})",
            flush=True,
        )
        return None
    return new_cal


def _ang_delta_max(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    return max(bc._ang_abs_delta_deg(a[i], b[i]) for i in range(n))


def run_repeatability_and_comparison(
    frames: Dict[int, np.ndarray],
    *,
    repeats: int = 10,
) -> Dict[int, dict]:
    """Same saved frames, N calibrations. Compare H vs legacy; log max drift."""
    report: Dict[int, dict] = {}
    for cam_idx, frame in frames.items():
        if frame is None:
            continue
        print(f"[HOMOGRAPHY_REP] cam={cam_idx} start repeats={repeats}", flush=True)
        Hs: List[np.ndarray] = []
        centers: List[Tuple[float, float]] = []
        wires: List[Tuple[float, ...]] = []
        rings: List[Tuple[float, float]] = []
        last_stats: Optional[Dict[str, object]] = None
        last_hcal: Optional[bc.BoardCalibration] = None
        last_stage1: Optional[bc.BoardCalibration] = None
        times: List[float] = []
        t_stage1_list: List[float] = []

        for i in range(int(repeats)):
            t0 = time.perf_counter()
            stage1 = bc.calibrate_board(frame, cam_idx=int(cam_idx))
            t_stage1_list.append(_perf_ms(t0))
            if stage1 is None or not bc._is_calibration_complete(stage1):
                print(f"[HOMOGRAPHY_REP] cam={cam_idx} iter={i} stage1 incomplete", flush=True)
                continue
            last_stage1 = stage1
            hcal, stats = try_homography_landmark_calibration(
                frame, stage1, cam_idx=int(cam_idx), save_debug=(i == 0)
            )
            times.append(float(stats.get("total_ms") or 0.0))
            last_stats = stats
            if hcal is None or not hcal.has_landmark_homography():
                print(
                    f"[HOMOGRAPHY_REP] cam={cam_idx} iter={i} H failed: {stats.get('reason')}",
                    flush=True,
                )
                continue
            last_hcal = hcal
            Hs.append(_H_from_tuple(tuple(hcal.landmark_H)))  # type: ignore[arg-type]
            centers.append((float(stage1.center[0]), float(stage1.center[1])))
            wires.append(tuple(float(a) % 360.0 for a in stage1.wire_angles_deg[: bc.SEGMENT_COUNT]))
            if stage1.double_ellipse is not None:
                ecx, ecy = bc._ellipse_center(stage1.double_ellipse)
                rings.append((ecx, ecy))

        max_h = 0.0
        max_c = 0.0
        max_w = 0.0
        max_r = 0.0
        if len(Hs) >= 2:
            H0 = Hs[0]
            for H in Hs[1:]:
                max_h = max(max_h, float(np.max(np.abs(H - H0))))
            c0 = centers[0]
            for c in centers[1:]:
                max_c = max(max_c, math.hypot(c[0] - c0[0], c[1] - c0[1]))
            w0 = wires[0]
            for w in wires[1:]:
                max_w = max(max_w, _ang_delta_max(w0, w))
            if rings:
                r0 = rings[0]
                for r in rings[1:]:
                    max_r = max(max_r, math.hypot(r[0] - r0[0], r[1] - r0[1]))

        print(
            f"[HOMOGRAPHY_REP] cam={cam_idx} n_ok={len(Hs)}/{repeats} "
            f"max_dH={max_h:.6g} max_dcenter={max_c:.4f}px "
            f"max_dwire={max_w:.4f}deg max_dring_c={max_r:.4f}px "
            f"H_ms_mean={float(np.mean(times)) if times else 0.0:.1f} "
            f"stage1_ms_mean={float(np.mean(t_stage1_list)) if t_stage1_list else 0.0:.1f}",
            flush=True,
        )

        legacy_cal = None
        t_leg = 0.0
        if last_stage1 is not None:
            t0 = time.perf_counter()
            legacy_cal = bc.enrich_calibration_with_warp_rings(frame, last_stage1)
            t_leg = _perf_ms(t0)
            if not legacy_cal.has_ring_align_warp():
                legacy_cal = bc.ensure_ring_align_warp(legacy_cal)
            print(f"[HOMOGRAPHY_REP] cam={cam_idx} legacy_stage2_ms={t_leg:.1f}", flush=True)
            if last_hcal is not None:
                path = save_homography_vs_legacy(cam_idx, frame, last_hcal, legacy_cal)
                print(f"[HOMOGRAPHY_REP] cam={cam_idx} comparison={path}", flush=True)

        report[int(cam_idx)] = {
            "stats": last_stats,
            "n_ok": len(Hs),
            "max_dH": max_h,
            "max_dcenter": max_c,
            "max_dwire": max_w,
            "max_dring": max_r,
            "H_ms_mean": float(np.mean(times)) if times else 0.0,
            "stage1_ms_mean": float(np.mean(t_stage1_list)) if t_stage1_list else 0.0,
            "legacy_ms": t_leg,
            "hcal": last_hcal,
            "legacy_cal": legacy_cal,
        }
    return report


def load_fake_cam_frames(root: Optional[str] = None) -> Dict[int, np.ndarray]:
    base = root or os.path.dirname(os.path.abspath(__file__))
    fake = os.path.join(base, "fake_cams")
    out: Dict[int, np.ndarray] = {}
    for idx in (0, 2, 4):
        path = os.path.join(fake, f"cam_{idx}.png")
        if not os.path.isfile(path):
            print(f"[HOMOGRAPHY_REP] missing {path}", flush=True)
            continue
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[HOMOGRAPHY_REP] failed read {path}", flush=True)
            continue
        out[idx] = img
    return out


def main() -> None:
    print(
        f"[HOMOGRAPHY_REP] CALIBRATION_MODE={getattr(bc, 'CALIBRATION_MODE', '?')}",
        flush=True,
    )
    frames = load_fake_cam_frames()
    if not frames:
        print("[HOMOGRAPHY_REP] no fake_cams frames", flush=True)
        return
    report = run_repeatability_and_comparison(frames, repeats=10)
    print("========== HOMOGRAPHY LANDMARK REPORT ==========", flush=True)
    for cam, row in report.items():
        st = row.get("stats") or {}
        print(
            f"cam{cam}: landmarks={st.get('landmarks')} counts={st.get('landmark_counts')} "
            f"inliers={st.get('inliers')} rms_px={st.get('rms_px')} max_px={st.get('max_px')} "
            f"center_err={st.get('center_error_px')} ring={st.get('ring_error')} "
            f"wire={st.get('wire_error')} H_total_ms={st.get('total_ms')} "
            f"stage1_ms={row.get('stage1_ms_mean')} legacy_stage2_ms={row.get('legacy_ms')} "
            f"repeat n_ok={row.get('n_ok')} max_dH={row.get('max_dH')} "
            f"max_dcenter={row.get('max_dcenter')} max_dwire={row.get('max_dwire')} "
            f"ok={st.get('ok')} reason={st.get('reason') or '-'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
