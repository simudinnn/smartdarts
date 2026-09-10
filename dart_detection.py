"""Motion diff + intensity-weighted FitLine detekcija strijelice na top-down warp slici (3 kamere)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from board_calibration import (
    BOARD_SISAL_EDGE_FRAC,
    BoardCalibration,
    BoardCalibrator,
    warp_topdown_from_calibration,
    topdown_point_to_score,
    scoring_ring_radii,
    scoring_ring_ellipses,
    normalized_ellipse_radius,
    _board_ellipse_mask,
    _debug_dir_path,
    _ensure_debug_dir_exists,
    _scale_ellipse,
    _scale_ellipse_axes,
    _warp_radii,
)


MOTION_DIFF_THRESH = 14
MOTION_MIN_PIXELS = 12
MOTION_MAX_PIXELS = 12000
MOTION_BLUR_KSIZE = 2
MOTION_LOG_MIN_PIXELS = 6
# Idle LED/USB flicker is often ~8–20 px; a real dart is far above this.
MOTION_ARM_PIXELS = 28
MOTION_CLEAR_PIXELS = 12
SETTLE_FRAME_DELAY = 4
# FitLine (2-stage): soft grayscale motion gated by dilated morph ROI →
# fixed FITLINE_SIZE → mild Gaussian blur → radial flight downweight →
# intensity-weighted PCA (stage1) → multicam tip fuse → stage2 FitLine on
# soft pixels inside a circle around that tip → fuse again → map back.
# Morph is a soft pixel gate only (no CC / shaft / flight blob return).
# Detection warp + FitLine ds space (preview stays at DEBUG_WARP_SIZE=200).
FITLINE_SIZE = 300
FITLINE_DS_SIZE = FITLINE_SIZE
FITLINE_BLUR_KSIZE = 5
FITLINE_BLUR_SIGMA = 1.0
# Ignore near-black for numerical stability (weights stay continuous above this).
FITLINE_WEIGHT_EPS = 2.0
FITLINE_MIN_PX = 8
# Dilate morph gate so soft FitLine keeps a margin around the motion streak.
FITLINE_GATE_DILATE = 5


def _warp_frame(
    board_calibrator: Optional[BoardCalibrator],
    cam_idx: int,
    frame: np.ndarray,
    cal: BoardCalibration,
) -> Optional[np.ndarray]:
    """Preferiraj cached remap na BoardCalibrator; fallback na sporiji path.

    Detection / FitLine path warps at FITLINE_SIZE. Calibration live preview
    keeps DEBUG_WARP_SIZE via BoardCalibrator.render_topdown_frames.
    """
    if board_calibrator is not None:
        warped = board_calibrator.warp_topdown(cam_idx, frame, out_size=FITLINE_SIZE)
        if warped is not None:
            return warped
    return warp_topdown_from_calibration(frame, cal, out_size=FITLINE_SIZE)
# Flight isolation (lightweight radial prior): tip/shaft = inner (toward bull),
# bright wide flight = outer (toward rim). Keep soft FitLine/PCA; only
# downweight the outer intensity mass so fat flight blobs cannot swing the axis.
FITLINE_FLIGHT_INNER_MASS_FRAC = 0.55
FITLINE_FLIGHT_OUTER_WEIGHT = 0.08
FITLINE_FLIGHT_TAPER_FRAC = 0.12
# Stage-2 tip ROI: circular radius = this fraction of board radius (in FitLine /
# ds space). Includes shaft/tip near the tip; excludes outer flight blobs.
# Name: FITLINE_STAGE2_TIP_RADIUS_FRAC
FITLINE_STAGE2_TIP_RADIUS_FRAC = 0.50
# Stage-2 soft falloff: weight *= (1 - dist/r)^power so tip-local shaft
# constrains direction more than residual flight at the ROI rim.
FITLINE_STAGE2_DIST_WEIGHT_POWER = 1.25
# Legacy midline helpers (unused by default path; kept for experiments).
FITLINE_SLICE_STEP = 1.0
FITLINE_MIDLINE_MIN_SLICES = 4
# Debug viz: percentile stretch of soft motion (nonzero pixels).
FITLINE_DEBUG_LO_PCT = 5.0
FITLINE_DEBUG_HI_PCT = 99.5
# Corridor around final axis (legacy fuse-support helpers; not used for axis fit).
FITLINE_CORRIDOR_PX = 3.5
# Fuse: drop one cam if its tip-to-line distance is an outlier (kept for 3+ cam).
FITLINE_OUTLIER_CAM_DIST_PX = 12.0
FITLINE_OUTLIER_CAM_RATIO = 2.5
# Consensus pair tip: drop a third line that flees the best-pair tip by more
# than max(min_px, board_r * frac). Pair lines define the tip; fleeers cannot
# pull a least-squares average.
FITLINE_CONSENSUS_FLEE_FRAC = 0.045
FITLINE_CONSENSUS_FLEE_MIN_PX = 8.0
# Legacy flee/resnap constants (path disabled; kept for unused helpers).
FITLINE_RESNAP_MISS_PX = 7.0
FITLINE_RESNAP_NEAR_TIP_FRAC = 0.28
FITLINE_RESNAP_NEAR_TIP_MIN_PX = 28.0
FITLINE_RESNAP_MIN_SUPPORT = 0.35
FITLINE_OUTER_PRESERVE_FRAC = 0.90
TIP_RIM_OUTSIDE_MAX_FRAC = 1.08
TIP_RIM_DOUBLE_BAND_MIN_PX = 4
TIP_RIM_THIN_INWARD_FRAC = 0.80
# Alias used by leftover flee-resnap helpers.
FITLINE_SHAFT_MIN_PX = FITLINE_MIN_PX
HAND_STABILITY_FRAMES = 2
HAND_INTERFRAME_DIFF_MAX = 11.0
HAND_MASK_SHIFT_MAX = 0.32
HAND_CENTROID_SHIFT_MIN = 16.0
HAND_LARGE_BLOB_PIXELS = 420
HAND_BLOCK_CLEAR_FRAMES = 1
HAND_CAM_VOTES_REQUIRED = 2
HAND_SCORE_THRESHOLD = 2.0
HAND_STRONG_SINGLE_CAM_SCORE = 2.5
HAND_POST_ACTIVITY_COOLDOWN = 3
HAND_WEAK_SCORE = 1.0
RAW_SURROUND_OUTER_SCALE = BOARD_SISAL_EDGE_FRAC * 1.12
RAW_SURROUND_MIN_PIXELS = 28
REARM_CLEAR_FRAMES = 4
# Empty-board: Pi logovi na praznoj ploci su ~td 700–900 px / diff ~23,
# a dok se vade strijelice 2000–7000 / diff 50+. Stari prag 42/7.5 nikad ne prode.
EMPTY_BOARD_MAX_PIXELS = 1600
EMPTY_BOARD_MAX_DIFF_MEAN = 36.0
EMPTY_BOARD_DIFF_APPLY_MIN_PX = 400
EMPTY_BOARD_RAW_HAND_PIXELS = 2800
# Nakon hita: prikazi odmah, pa 0.5s da se ne upisu dva hita odjednom.
# (Ne rearm koji ceka px<12 — to se zaglavi na sumu.)
POST_HIT_COOLDOWN_SEC = 0.50
# Leftover shaft after a real hit is radial; 1-cam fuse projects to bull.
BULL_MIN_CAMS = 2
BULL_MOTION_MAX_BULL_OUTER = 2.2
SAVE_MOTION_DEBUG = False
# Hit debug: fused_motion_raw + fitline_smooth_intersect only (see save_hit_fusion_debug).
SAVE_HIT_FUSION_DEBUG = False
MOTION_REFS_DIRNAME = "dart_motion_refs"
# BGR boje po kameri za hit fusion debug overlay (jarke radi vidljivosti).
_HIT_DEBUG_CAM_COLORS = {
    0: (0, 0, 255),
    2: (0, 255, 0),
    4: (255, 200, 0),
}


def _safe_imwrite(path: str, img: Optional[np.ndarray]) -> bool:
    """Pi/OpenCV: imwrite na view/non-contiguous bufferu zna pasti u C (segfault)."""
    if img is None or getattr(img, "size", 0) <= 0:
        return False
    try:
        buf = np.ascontiguousarray(img)
        if buf.ndim not in (2, 3) or buf.size <= 0:
            return False
        return bool(cv2.imwrite(path, buf))
    except Exception as exc:
        print(f"[dart] imwrite fail {path}: {exc}", flush=True)
        return False


def motion_refs_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), MOTION_REFS_DIRNAME)


# Lazy cache: False = unavailable; callable = apply_ml_tip_pipeline.
_ML_TIP_FN: Optional[object] = None


def _apply_ml_tip_cached(fused: FusedDartResult, per_cam, board_calibrator) -> FusedDartResult:
    global _ML_TIP_FN
    if _ML_TIP_FN is False:
        return fused
    if _ML_TIP_FN is None:
        try:
            from ml_tip.runtime import apply_ml_tip_pipeline

            _ML_TIP_FN = apply_ml_tip_pipeline
        except Exception:
            _ML_TIP_FN = False
            return fused
    try:
        return _ML_TIP_FN(  # type: ignore[misc]
            fused,
            per_cam,
            out_size=FITLINE_SIZE,
            board_calibrator=board_calibrator,
        )
    except Exception as exc:
        print(f"[ml_tip] pipeline error (ignored): {exc}", flush=True)
        return fused


@dataclass
class DartDetectResult:
    cam_idx: int
    found: bool
    tip_xy: Tuple[float, float] = (0.0, 0.0)
    line_vx: float = 0.0
    line_vy: float = 0.0
    line_x0: float = 0.0
    line_y0: float = 0.0
    line_angle_deg: float = 0.0
    confidence: float = 0.0
    segment_number: int = 0
    zone_name: str = "miss"
    score: int = 0
    motion_pixels: int = 0
    diff_mean: float = 0.0
    reject_reason: str = ""
    debug_bgr: Optional[np.ndarray] = None
    # Grayscale motion diff (0=crno, 255=max promjena) — morph gating mask.
    motion_gray: Optional[np.ndarray] = None
    # Čisti absdiff u warp prostoru (prije threshold/morph), za debug.
    motion_raw: Optional[np.ndarray] = None
    # Soft motion used for FitLine, upsampled to original warp size (debug overlay).
    fit_gray: Optional[np.ndarray] = None
    # Blurred soft motion at FITLINE_SIZE — midline fit space.
    fit_ds_gray: Optional[np.ndarray] = None
    # Map FitLine/ds coords → original warp: original = ds * scale.
    fit_scale_x: float = 1.0
    fit_scale_y: float = 1.0
    # Per-cam tip: projekcija fused tipa na liniju (2+ cam) / board-center (1 cam).
    # line_* are in FITLINE_SIZE coordinates (same space as fit_ds_gray).
    radius_tip: Tuple[float, float] = (0.0, 0.0)


@dataclass
class FusedDartResult:
    found: bool
    tip_xy: Tuple[float, float] = (0.0, 0.0)
    confidence: float = 0.0
    segment_number: int = 0
    zone_name: str = "miss"
    score: int = 0
    cam_indices: List[int] = field(default_factory=list)
    per_cam: List[DartDetectResult] = field(default_factory=list)
    reject_reason: str = ""
    # Stage-2 ROI (stage1 fused tip + radius) in original warp space — debug only.
    stage2_roi_tip_xy: Optional[Tuple[float, float]] = None
    stage2_roi_radius: float = 0.0


def summarize_motion_log(results: List[DartDetectResult]) -> str:
    if not results:
        return "nema kalibriranih kamera"
    parts: List[str] = []
    for r in results:
        label = _zone_short_label(r.zone_name, r.segment_number) if r.found else "-"
        parts.append(
            f"cam{r.cam_idx}:px={r.motion_pixels} mean={r.diff_mean:.1f} "
            f"line={int(r.found)} {label}"
            + (f" ({r.reject_reason})" if r.reject_reason else "")
        )
    return " | ".join(parts)


def summarize_fused_log(fused: FusedDartResult) -> str:
    if not fused.found:
        return f"fused: FAIL ({fused.reject_reason})"
    label = _zone_short_label(fused.zone_name, fused.segment_number)
    cams = ",".join(str(c) for c in fused.cam_indices)
    return (
        f"fused:{label}={fused.score} tip=({fused.tip_xy[0]:.1f},{fused.tip_xy[1]:.1f}) "
        f"conf={fused.confidence:.2f} cams=[{cams}]"
    )


def _zone_short_label(zone: str, number: int) -> str:
    z = zone.lower()
    if z == "inner_bull":
        return "B50"
    if z == "outer_bull":
        return "B25"
    if z == "triple":
        return f"T{number}"
    if z == "double":
        return f"D{number}"
    if z == "miss":
        return "MISS"
    return f"S{number}"


def _motion_mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def _soft_motion_centroid(result: DartDetectResult) -> Optional[Tuple[float, float]]:
    img = result.fit_gray if result.fit_gray is not None else result.motion_gray
    if img is None or getattr(img, "size", 0) <= 0:
        return None
    w = img.astype(np.float64)
    s = float(w.sum())
    if s < 1e-6:
        return _motion_mask_centroid(img)
    ys, xs = np.indices(img.shape)
    return float((xs * w).sum() / s), float((ys * w).sum() / s)


def _hand_motion_score(
    ref_gray: np.ndarray,
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
) -> float:
    """
    Bodovi za ruku u kadru (0 = nema). Strijelica nakon uboda miruje -> nizak score.
    Pojava novog bloba (let strelice) -> 0, ne tretirati kao ruku.
    """
    if prev_gray.shape != cur_gray.shape:
        prev_gray = cv2.resize(
            prev_gray, (cur_gray.shape[1], cur_gray.shape[0]), interpolation=cv2.INTER_AREA
        )
    ref_g = ref_gray if ref_gray.shape == cur_gray.shape else cv2.resize(
        ref_gray, (cur_gray.shape[1], cur_gray.shape[0]), interpolation=cv2.INTER_AREA
    )
    mask_prev = _motion_mask(ref_g, prev_gray)
    mask_cur = _motion_mask(ref_g, cur_gray)
    bin_prev = _motion_binary(mask_prev)
    bin_cur = _motion_binary(mask_cur)
    n_prev = int(np.count_nonzero(bin_prev))
    n_cur = int(np.count_nonzero(bin_cur))
    roi = cv2.bitwise_or(bin_prev, bin_cur)
    n_roi = int(np.count_nonzero(roi))
    if n_roi < MOTION_MIN_PIXELS:
        return 0.0

    min_side = min(n_prev, n_cur)
    if min_side < MOTION_MIN_PIXELS:
        return 0.0

    live_diff = cv2.absdiff(prev_gray, cur_gray)
    roi_vals = live_diff[roi > 0]
    mean_live = float(np.mean(roi_vals)) if roi_vals.size else 0.0
    xor_px = int(np.count_nonzero(cv2.bitwise_xor(bin_prev, bin_cur)))
    shift_ratio = xor_px / max(n_roi, 1)

    c0 = _motion_mask_centroid(mask_prev)
    c1 = _motion_mask_centroid(mask_cur)
    centroid_shift = 0.0
    if c0 is not None and c1 is not None:
        centroid_shift = float(np.hypot(c1[0] - c0[0], c1[1] - c0[1]))

    score = 0.0
    if mean_live > HAND_INTERFRAME_DIFF_MAX:
        score += 1.0
    if shift_ratio > HAND_MASK_SHIFT_MAX:
        score += 1.0
    if centroid_shift > HAND_CENTROID_SHIFT_MIN:
        score += 1.0
    if (
        n_roi > HAND_LARGE_BLOB_PIXELS
        and mean_live > 14.0
        and shift_ratio > 0.28
        and centroid_shift > 14.0
    ):
        score += 1.0
    return score


def _hand_like_motion_pair(
    ref_gray: np.ndarray,
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
) -> bool:
    return _hand_motion_score(ref_gray, prev_gray, cur_gray) >= HAND_SCORE_THRESHOLD


def _fitline_angle_deg(vx: float, vy: float) -> float:
    return float(np.degrees(np.arctan2(vx, -vy)) % 360.0)


def _orient_fitline_tipward(
    vx: float,
    vy: float,
    x0: float,
    y0: float,
    board_center_ds: Tuple[float, float],
) -> Tuple[float, float]:
    """Force direction flight (outer) → tip (inner / toward board center)."""
    bcx, bcy = float(board_center_ds[0]), float(board_center_ds[1])
    r0 = float(np.hypot(x0 - bcx, y0 - bcy))
    r1 = float(np.hypot(x0 + vx - bcx, y0 + vy - bcy))
    if r1 > r0:
        return -float(vx), -float(vy)
    return float(vx), float(vy)


def _map_ds_xy_to_original(
    x: float, y: float, scale_x: float, scale_y: float
) -> Tuple[float, float]:
    return float(x) * float(scale_x), float(y) * float(scale_y)


def _map_original_xy_to_ds(
    x: float, y: float, scale_x: float, scale_y: float
) -> Tuple[float, float]:
    sx = float(scale_x) if float(scale_x) > 1e-9 else 1.0
    sy = float(scale_y) if float(scale_y) > 1e-9 else 1.0
    return float(x) / sx, float(y) / sy


def _soft_motion_gray_for_fit(
    raw_diff: np.ndarray,
    board_mask: np.ndarray,
) -> np.ndarray:
    """Continuous grayscale motion inside board ROI — no binarization."""
    return cv2.bitwise_and(raw_diff, raw_diff, mask=board_mask)


def _fit_from_points(
    px: np.ndarray,
    py: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float]:
    """PCA / fitLine through points. Equal weight when weights is None."""
    xs_f = px.astype(np.float64)
    ys_f = py.astype(np.float64)
    n = int(xs_f.size)
    if n < 2:
        x0 = float(xs_f[0]) if n else 0.0
        y0 = float(ys_f[0]) if n else 0.0
        return 1.0, 0.0, x0, y0

    if weights is not None and int(weights.size) == n:
        w = np.clip(weights.astype(np.float64), 0.0, None)
        s = float(w.sum())
        if s > 1e-9:
            w = w / s
            x0 = float((xs_f * w).sum())
            y0 = float((ys_f * w).sum())
            dx = xs_f - x0
            dy = ys_f - y0
            cxx = float((w * dx * dx).sum())
            cxy = float((w * dx * dy).sum())
            cyy = float((w * dy * dy).sum())
            cov = np.array([[cxx, cxy], [cxy, cyy]], dtype=np.float64)
            eigvals, eigvecs = np.linalg.eigh(cov)
            _ = eigvals
            vx = float(eigvecs[0, -1])
            vy = float(eigvecs[1, -1])
            nn = float(np.hypot(vx, vy)) or 1.0
            return vx / nn, vy / nn, x0, y0

    pts = np.column_stack([xs_f.astype(np.float32), ys_f.astype(np.float32)])
    if n >= 3:
        line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        vx_o, vy_o = float(line[0]), float(line[1])
        nn = float(np.hypot(vx_o, vy_o)) or 1.0
        return vx_o / nn, vy_o / nn, float(np.mean(xs_f)), float(np.mean(ys_f))
    x0 = float(np.mean(xs_f))
    y0 = float(np.mean(ys_f))
    vx = float(xs_f[-1] - xs_f[0])
    vy = float(ys_f[-1] - ys_f[0])
    nn = float(np.hypot(vx, vy)) or 1.0
    return vx / nn, vy / nn, x0, y0


def _soft_slice_midpoints(
    xs: np.ndarray,
    ys: np.ndarray,
    weights: np.ndarray,
    vx: float,
    vy: float,
    x0: float,
    y0: float,
    *,
    slice_step: float = FITLINE_SLICE_STEP,
    mass_eps: float = FITLINE_WEIGHT_EPS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Along seed axis, soft midline point of each perpendicular cross-section.

    Soft pixels in each axial slab vote with equal weight for the midpoint
    (geometric middle of the soft support). Intensity is used only to skip
    empty/near-empty slices (tiny mass_eps) — bright edges must not pull the
    midline off the streak center. No binary mask / CC / morph.
    """
    step = max(float(slice_step), 1e-6)
    t = (xs - float(x0)) * float(vx) + (ys - float(y0)) * float(vy)
    # Integer axial bins; slab width ≈ slice_step.
    bins = np.floor(t / step).astype(np.int32)
    if bins.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    order = np.argsort(bins, kind="mergesort")
    bins_s = bins[order]
    xs_s = xs[order]
    ys_s = ys[order]
    w_s = weights[order]

    mid_x: List[float] = []
    mid_y: List[float] = []
    i = 0
    n = int(bins_s.size)
    mass_floor = float(mass_eps)
    while i < n:
        j = i + 1
        b = int(bins_s[i])
        while j < n and int(bins_s[j]) == b:
            j += 1
        w_seg = w_s[i:j]
        wsum = float(w_seg.sum())
        if wsum > mass_floor:
            # Equal-weight mean of soft pixels = geometric middle of support.
            # (Intensity-weighted COG would sit on a bright edge.)
            mid_x.append(float(xs_s[i:j].mean()))
            mid_y.append(float(ys_s[i:j].mean()))
        i = j

    return (
        np.asarray(mid_x, dtype=np.float64),
        np.asarray(mid_y, dtype=np.float64),
    )


def _suppress_flight_radial_mass(
    soft_f32: np.ndarray,
    board_center_xy: Tuple[float, float],
    *,
    eps: float = FITLINE_WEIGHT_EPS,
    inner_mass_frac: float = FITLINE_FLIGHT_INNER_MASS_FRAC,
    outer_weight: float = FITLINE_FLIGHT_OUTER_WEIGHT,
    taper_frac: float = FITLINE_FLIGHT_TAPER_FRAC,
) -> np.ndarray:
    """Downweight outer (flight) soft intensity via radial mass prior.

    Sort soft pixels by distance to board center (inner = tip/shaft first).
    Keep ~inner_mass_frac of total intensity mass at full weight; taper the
    remainder (bright outer flight) down to outer_weight. No CC labeling.
    """
    ys, xs = np.where(soft_f32 > float(eps))
    n = int(xs.size)
    if n < FITLINE_MIN_PX:
        return soft_f32

    w = soft_f32[ys, xs].astype(np.float64)
    bcx, bcy = float(board_center_xy[0]), float(board_center_xy[1])
    r = np.hypot(xs.astype(np.float64) - bcx, ys.astype(np.float64) - bcy)
    order = np.argsort(r, kind="mergesort")
    w_ord = w[order]
    cum = np.cumsum(w_ord)
    total = float(cum[-1])
    if total < 1e-9:
        return soft_f32

    frac = float(np.clip(inner_mass_frac, 0.15, 0.95))
    target = frac * total
    cut_i = int(np.searchsorted(cum, target, side="left"))
    cut_i = min(max(cut_i, 0), n - 1)
    r_cut = float(r[order[cut_i]])
    r_min = float(r[order[0]])
    r_max = float(r[order[-1]])
    span = max(r_max - r_min, 1.0)
    taper = max(float(taper_frac) * span, 1.0)
    ow = float(np.clip(outer_weight, 0.0, 1.0))

    # scale: 1 inside r_cut → lerp to ow across taper → ow beyond
    t = (r - r_cut) / taper
    scale = np.where(
        r <= r_cut,
        1.0,
        np.where(r >= r_cut + taper, ow, 1.0 + t * (ow - 1.0)),
    )
    out = soft_f32.copy()
    out[ys, xs] = (soft_f32[ys, xs].astype(np.float64) * scale).astype(np.float32)
    return out


def _weighted_fitline_on_soft_motion(
    soft_gray: np.ndarray,
    *,
    board_center: Optional[Tuple[float, float]] = None,
    out_size: int = FITLINE_SIZE,
) -> Tuple[
    bool,
    Tuple[float, float, float, float],
    np.ndarray,
    float,
    float,
    np.ndarray,
    np.ndarray,
]:
    """Soft motion → blur → radial flight suppress → intensity-weighted FitLine.

    Simple correct path (no midline slice refinement):
      morph-gated soft pixels → blur → downweight outer flight mass →
      intensity-weighted PCA for direction (vx,vy) and point (x0,y0).

    Caller should gate soft_gray to the motion streak (dilated morph ROI) so
    board-wide absdiff noise cannot pull the centroid to image center.

    Returns:
      ok, (vx,vy,x0,y0), ds_blurred_u8, scale_x, scale_y, xs_ds, ys_ds
    where original_xy = ds_xy * scale.
    """
    h, w = soft_gray.shape[:2]
    nw = max(1, int(out_size))
    nh = max(1, int(out_size))
    scale_x = float(w) / float(nw)
    scale_y = float(h) / float(nh)

    if board_center is not None:
        bcx = float(board_center[0]) / scale_x
        bcy = float(board_center[1]) / scale_y
    else:
        bcx, bcy = 0.5 * float(nw), 0.5 * float(nh)
    board_center_ds = (bcx, bcy)

    if nw == w and nh == h:
        ds = soft_gray.astype(np.float32, copy=False)
    else:
        ds = cv2.resize(
            soft_gray.astype(np.float32),
            (nw, nh),
            interpolation=cv2.INTER_AREA,
        )

    k = int(FITLINE_BLUR_KSIZE)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    if k >= 3:
        ds_blur = cv2.GaussianBlur(ds, (k, k), float(FITLINE_BLUR_SIGMA))
    else:
        ds_blur = ds

    # Suppress bright outer flight before PCA so the axis tracks shaft/tip.
    ds_blur = _suppress_flight_radial_mass(ds_blur, board_center_ds)
    ds_u8 = np.clip(np.round(ds_blur), 0, 255).astype(np.uint8)

    eps = float(FITLINE_WEIGHT_EPS)
    ys, xs = np.where(ds_blur > eps)
    n = int(xs.size)
    empty_x = xs.astype(np.float64)
    empty_y = ys.astype(np.float64)
    if n < FITLINE_MIN_PX:
        tip = (
            float(np.mean(xs)) if n else 0.0,
            float(np.mean(ys)) if n else 0.0,
        )
        return (
            False,
            (1.0, 0.0, tip[0], tip[1]),
            ds_u8,
            scale_x,
            scale_y,
            empty_x,
            empty_y,
        )

    xs_f = xs.astype(np.float64)
    ys_f = ys.astype(np.float64)
    weights = ds_blur[ys, xs].astype(np.float64)

    # Intensity-weighted PCA: (x0,y0) is the intensity centroid of the motion,
    # never a stale image-center default.
    vx, vy, x0, y0 = _fit_from_points(xs_f, ys_f, weights)
    vx, vy = _orient_fitline_tipward(vx, vy, x0, y0, board_center_ds)
    span = float(
        np.hypot(float(xs_f.max()) - float(xs_f.min()), float(ys_f.max()) - float(ys_f.min()))
    )
    ok = span >= 2.0 and n >= FITLINE_MIN_PX
    return (
        ok,
        (vx, vy, x0, y0),
        ds_u8,
        scale_x,
        scale_y,
        xs_f,
        ys_f,
    )


def _refit_fitline_near_tip(
    ds_soft: np.ndarray,
    tip_xy_ds: Tuple[float, float],
    radius_ds: float,
    board_center_ds: Tuple[float, float],
) -> Optional[Tuple[float, float, float, float]]:
    """Stage-2 FitLine: intensity-weighted PCA on soft pixels inside tip circle.

    Uses only soft intensity inside ``radius_ds`` of the fused tip so outer
    flight blobs cannot pull the axis. Returns (vx,vy,x0,y0) or None if too few
    pixels remain.
    """
    if ds_soft is None or ds_soft.size == 0:
        return None
    soft = ds_soft.astype(np.float32, copy=False)
    eps = float(FITLINE_WEIGHT_EPS)
    ys, xs = np.where(soft > eps)
    n = int(xs.size)
    if n < FITLINE_MIN_PX:
        return None

    tx, ty = float(tip_xy_ds[0]), float(tip_xy_ds[1])
    r = max(float(radius_ds), 1.0)
    dist = np.hypot(xs.astype(np.float64) - tx, ys.astype(np.float64) - ty)
    keep = dist <= r
    n_keep = int(np.count_nonzero(keep))
    if n_keep < FITLINE_MIN_PX:
        return None

    xs_f = xs[keep].astype(np.float64)
    ys_f = ys[keep].astype(np.float64)
    dist_k = dist[keep]
    # Soft intensity × tip-proximity: rim flight at ROI edge cannot swing axis.
    inten = soft[ys[keep], xs[keep]].astype(np.float64)
    prox = np.clip(1.0 - dist_k / r, 0.05, 1.0) ** float(FITLINE_STAGE2_DIST_WEIGHT_POWER)
    weights = inten * prox
    vx, vy, x0, y0 = _fit_from_points(xs_f, ys_f, weights)
    vx, vy = _orient_fitline_tipward(vx, vy, x0, y0, board_center_ds)
    span = float(
        np.hypot(float(xs_f.max()) - float(xs_f.min()), float(ys_f.max()) - float(ys_f.min()))
    )
    if span < 2.0:
        return None
    return float(vx), float(vy), float(x0), float(y0)


def _gate_soft_motion_for_fit(soft: np.ndarray, morph_mask: np.ndarray) -> np.ndarray:
    """Keep soft intensities, but only inside a dilated morph motion ROI.

    Ungated board-wide absdiff noise pulls the intensity centroid toward the
    board/image center and wrecks FitLine orientation.
    """
    if morph_mask is None or morph_mask.size == 0:
        return soft
    gate = (morph_mask > 0).astype(np.uint8) * 255
    if soft.shape[:2] != gate.shape[:2]:
        gate = cv2.resize(
            gate, (soft.shape[1], soft.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    d = int(FITLINE_GATE_DILATE)
    if d >= 3:
        if d % 2 == 0:
            d += 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
        gate = cv2.dilate(gate, k, iterations=1)
    return cv2.bitwise_and(soft, soft, mask=gate)


def _motion_mask(ref_gray: np.ndarray, cur_gray: np.ndarray) -> np.ndarray:
    """Hard morph mask for motion gating / pixel counts only (not FitLine input)."""
    diff = cv2.absdiff(cur_gray, ref_gray)
    if MOTION_BLUR_KSIZE >= 3:
        diff = cv2.GaussianBlur(diff, (MOTION_BLUR_KSIZE, MOTION_BLUR_KSIZE), 0)
    _, binary = cv2.threshold(diff, MOTION_DIFF_THRESH, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    return cv2.bitwise_and(diff, diff, mask=binary)


def _motion_binary(motion_gray: np.ndarray) -> np.ndarray:
    """0/255 maska iz grayscale motion diffa (za bitwise OR/XOR)."""
    return (motion_gray > 0).astype(np.uint8) * 255


def _save_motion_debug(
    cam_idx: int,
    mask: np.ndarray,
    *,
    fit_img: Optional[np.ndarray] = None,
    motion_pixels: int = 0,
    diff_mean: float = 0.0,
    status: str = "",
) -> None:
    if not SAVE_MOTION_DEBUG or cam_idx < 0:
        return
    _ensure_debug_dir_exists()
    base = os.path.join(_debug_dir_path(), f"cam_{cam_idx}")
    if mask.ndim == 2:
        diff_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        diff_vis = mask.copy()
    txt = f"px={motion_pixels} mean={diff_mean:.1f}"
    if status:
        txt += f" {status}"
    cv2.putText(diff_vis, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.imwrite(base + "_motion_diff.png", diff_vis)
    if fit_img is not None:
        cv2.imwrite(base + "_motion_fit.png", fit_img)


def _hit_debug_cam_color(cam_idx: int) -> Tuple[int, int, int]:
    return _HIT_DEBUG_CAM_COLORS.get(int(cam_idx), (200, 200, 200))


def _overlay_grayscale_blob(
    canvas_bgr: np.ndarray,
    motion_gray: np.ndarray,
    color_bgr: Tuple[int, int, int],
) -> None:
    """Uboi blob jarko — fit pikselima punom bojom radi vidljivosti."""
    if motion_gray.shape[:2] != canvas_bgr.shape[:2]:
        motion_gray = cv2.resize(
            motion_gray,
            (canvas_bgr.shape[1], canvas_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    m = motion_gray > 0
    if not np.any(m):
        return
    color = np.array(color_bgr, dtype=np.uint8)
    # Puna boja na fit pikselima (bez prigušenja).
    canvas_bgr[m] = color


def _draw_fitline_on(
    img: np.ndarray,
    x0: float,
    y0: float,
    vx: float,
    vy: float,
    color: Tuple[int, int, int],
    *,
    thickness: int = 2,
) -> None:
    h, w = img.shape[:2]
    span = float(max(h, w) * 1.5)
    p1 = (int(round(x0 - vx * span)), int(round(y0 - vy * span)))
    p2 = (int(round(x0 + vx * span)), int(round(y0 + vy * span)))
    th = max(1, int(thickness))
    line_type = cv2.LINE_AA if th >= 2 else cv2.LINE_8
    cv2.line(img, p1, p2, color, th, line_type)


def _hit_motion_debug_enabled() -> bool:
    """Spremaj pure-motion debug kad je hit fusion ili motion debug uključen."""
    return bool(SAVE_HIT_FUSION_DEBUG or SAVE_MOTION_DEBUG)


def _contrast_stretch_gray(
    gray: np.ndarray,
    *,
    lo_pct: float = FITLINE_DEBUG_LO_PCT,
    hi_pct: float = FITLINE_DEBUG_HI_PCT,
    floor: float = FITLINE_WEIGHT_EPS,
) -> np.ndarray:
    """Percentile stretch soft motion to a readable 0..255 grayscale."""
    if gray.ndim == 3:
        g = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        g = gray.astype(np.float32, copy=False)
    nz = g[g > float(floor)]
    if nz.size < 8:
        nz = g.reshape(-1)
    lo = float(np.percentile(nz, float(lo_pct)))
    hi = float(np.percentile(nz, float(hi_pct)))
    if hi <= lo + 1e-3:
        hi = float(np.max(nz)) if nz.size else 1.0
        lo = float(np.min(nz)) if nz.size else 0.0
    if hi <= lo + 1e-3:
        return np.zeros(g.shape[:2], dtype=np.uint8)
    out = (g - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def _gray_to_bgr_vis(gray: np.ndarray, *, gain: float = 1.0) -> np.ndarray:
    """Grayscale → BGR for FitLine debug (optional gain for faint motion)."""
    g = gray
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    if gain != 1.0:
        g = np.clip(g.astype(np.float32) * float(gain), 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _soft_motion_bgr_vis(gray: np.ndarray) -> np.ndarray:
    """Contrast-stretched soft motion as readable BGR (not noisy RGB mess)."""
    stretched = _contrast_stretch_gray(gray)
    return cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)


def _draw_fitline_ds_on_original(
    img: np.ndarray,
    x0_ds: float,
    y0_ds: float,
    vx_ds: float,
    vy_ds: float,
    scale_x: float,
    scale_y: float,
    color: Tuple[int, int, int],
    *,
    thickness: int = 2,
) -> None:
    """Draw a FitLine defined in ds space onto an original-size BGR image."""
    x0, y0 = _map_ds_xy_to_original(x0_ds, y0_ds, scale_x, scale_y)
    vx = float(vx_ds) * float(scale_x)
    vy = float(vy_ds) * float(scale_y)
    n = float(np.hypot(vx, vy))
    if n < 1e-9:
        return
    vx /= n
    vy /= n
    _draw_fitline_on(img, x0, y0, vx, vy, color, thickness=thickness)


def _save_fitline_smooth_debug(
    stem: str,
    fused: FusedDartResult,
    line_cams: List[DartDetectResult],
    motion_dir: str,
    *,
    base_gray: np.ndarray,
    soft_acc: Optional[np.ndarray],
    draw_cams: Optional[List[DartDetectResult]] = None,
) -> List[str]:
    """Board underlay + soft blobs tinted by stage2 ROI + cam lines + tip.

    ``line_cams`` = consensus-used cams (full color). ``draw_cams`` may include
    rejected fleeers drawn dim so bad FitLines are visible but not dominant.
    """
    used = list(line_cams)
    draw = list(draw_cams) if draw_cams is not None else list(line_cams)
    if not _hit_motion_debug_enabled() or not draw:
        return []
    saved: List[str] = []
    used_ids = {int(r.cam_idx) for r in used}

    h, w = int(base_gray.shape[0]), int(base_gray.shape[1])
    under = cv2.cvtColor(
        (base_gray.astype(np.float32) * 0.40).astype(np.uint8), cv2.COLOR_GRAY2BGR
    )
    composite = under.copy()

    # Stage2 ROI center/radius used for re-FitLine (stage1 tip in warp space).
    if fused.stage2_roi_tip_xy is not None:
        roi_cx, roi_cy = float(fused.stage2_roi_tip_xy[0]), float(fused.stage2_roi_tip_xy[1])
        roi_r = float(fused.stage2_roi_radius)
    else:
        roi_cx, roi_cy = float(fused.tip_xy[0]), float(fused.tip_xy[1])
        _, board_r = _board_center_and_r(max(h, w))
        roi_r = float(FITLINE_STAGE2_TIP_RADIUS_FRAC) * float(board_r)

    if soft_acc is not None and soft_acc.shape[0] == h and soft_acc.shape[1] == w:
        stretched = _contrast_stretch_gray(soft_acc).astype(np.float32) / 255.0
        motion_m = soft_acc > float(FITLINE_WEIGHT_EPS)
        yy, xx = np.ogrid[:h, :w]
        inside = (xx - roi_cx) ** 2 + (yy - roi_cy) ** 2 <= (roi_r * roi_r)
        # Inside ROI (used by stage2 FitLine): cyan; excluded outside: red.
        in_color = np.array((0, 255, 220), dtype=np.float32)  # BGR cyan
        out_color = np.array((40, 40, 255), dtype=np.float32)  # BGR red
        for mask, color in ((motion_m & inside, in_color), (motion_m & ~inside, out_color)):
            if not np.any(mask):
                continue
            a = stretched[mask][:, None]
            a = np.clip(a * 0.85 + 0.15, 0.0, 1.0)
            base = composite[mask].astype(np.float32)
            composite[mask] = (base * (1.0 - a) + color * a).astype(np.uint8)

    # Rejected (flee) lines first, dim; consensus-used lines on top, full color.
    for r in draw:
        sx = float(r.fit_scale_x) if float(r.fit_scale_x) > 1e-9 else 1.0
        sy = float(r.fit_scale_y) if float(r.fit_scale_y) > 1e-9 else 1.0
        color = _hit_debug_cam_color(r.cam_idx)
        used_cam = int(r.cam_idx) in used_ids
        if not used_cam:
            color = tuple(int(round(c * 0.35)) for c in color)
        _draw_fitline_ds_on_original(
            composite,
            r.line_x0,
            r.line_y0,
            r.line_vx,
            r.line_vy,
            sx,
            sy,
            color,
            thickness=1 if not used_cam else 2,
        )

    # Stage2 ROI circle (matches re-FitLine radius).
    roi_pt = (int(round(roi_cx)), int(round(roi_cy)))
    cv2.circle(
        composite,
        roi_pt,
        max(1, int(round(roi_r))),
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    tip_pt = (int(round(fused.tip_xy[0])), int(round(fused.tip_xy[1])))
    cv2.circle(composite, tip_pt, 5, (0, 180, 180), 1, cv2.LINE_AA)
    cv2.circle(composite, tip_pt, 3, (0, 255, 255), -1, cv2.LINE_AA)
    n_rej = sum(1 for r in draw if int(r.cam_idx) not in used_ids)
    rej_note = f" dim={n_rej}rej" if n_rej else ""
    cv2.putText(
        composite,
        f"FitLine consensus + stage2 r={roi_r:.0f} (cyan=in red=out){rej_note}",
        (4, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cpath = os.path.join(motion_dir, f"{stem}_fitline_smooth_intersect.png")
    cv2.imwrite(cpath, composite)
    saved.append(cpath)
    return saved


def _warp_grays_for_debug(
    board_calibrator: Optional[BoardCalibrator],
    raw_frames: Optional[Dict[int, np.ndarray]],
) -> Dict[int, np.ndarray]:
    """Warp fire frames with current calibrator — same path as detect/score."""
    out: Dict[int, np.ndarray] = {}
    if board_calibrator is None or not raw_frames:
        return out
    for cam_idx, frame in raw_frames.items():
        if frame is None:
            continue
        cal = board_calibrator.get(int(cam_idx))
        if cal is None or not cal.is_valid() or cal.double_ellipse is None:
            continue
        warped = _warp_frame(board_calibrator, int(cam_idx), frame, cal)
        if warped is None:
            continue
        out[int(cam_idx)] = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return out


def save_hit_fusion_debug(
    fused: FusedDartResult,
    empty_refs: Dict[int, np.ndarray],
    *,
    raw_frames: Optional[Dict[int, np.ndarray]] = None,
    board_calibrator: Optional[BoardCalibrator] = None,
    warp_refs: Optional[Dict[int, np.ndarray]] = None,
    out_size: int = FITLINE_SIZE,
) -> Optional[str]:
    """Save only two hit debug images during normal detection:

    - ``{stem}_fused_motion_raw.png`` — fused soft-motion blobs on black (no lines)
    - ``motion/{stem}_fitline_smooth_intersect.png`` — board underlay + stage2 in/out
      blobs + cam intersection lines + tip + stage2 radius circle

    Per-cam fitline_smooth / motion_raw / motion_mask / raw cam frames are not saved.
    """
    if not _hit_motion_debug_enabled() or not fused.found:
        return None

    # All on-board FitLine cams (including consensus-rejected fleeers).
    all_line_cams = [
        r
        for r in fused.per_cam
        if r.found
        and r.motion_pixels >= MOTION_MIN_PIXELS
        and r.reject_reason != "off_board_raw"
    ]
    used_ids = {int(c) for c in fused.cam_indices}
    line_cams = [r for r in all_line_cams if int(r.cam_idx) in used_ids]
    if not line_cams:
        line_cams = list(all_line_cams)
    draw_cams = list(all_line_cams) if all_line_cams else list(line_cams)

    # Optional fire-warps only as last-resort fallback (never preferred over empty).
    fire_warps = dict(warp_refs) if warp_refs else {}
    if not fire_warps:
        fire_warps = _warp_grays_for_debug(board_calibrator, raw_frames)

    _ensure_debug_dir_exists()
    base_gray: Optional[np.ndarray] = None
    # Stable empty board from calibration / board-clear — not post-hit motion ref.
    for cam_idx in fused.cam_indices:
        ref = empty_refs.get(int(cam_idx))
        if ref is not None:
            base_gray = ref
            break
    if base_gray is None:
        for ref in empty_refs.values():
            base_gray = ref
            break
    if base_gray is None:
        for cam_idx in fused.cam_indices:
            ref = fire_warps.get(int(cam_idx))
            if ref is not None:
                base_gray = ref
                break
    if base_gray is None:
        for ref in fire_warps.values():
            base_gray = ref
            break
    if base_gray is None:
        for r in draw_cams:
            if r.debug_bgr is not None:
                base_gray = cv2.cvtColor(r.debug_bgr, cv2.COLOR_BGR2GRAY)
                break
    if base_gray is None:
        base_gray = np.zeros((out_size, out_size), dtype=np.uint8)

    if base_gray.shape[0] != out_size or base_gray.shape[1] != out_size:
        base_gray = cv2.resize(base_gray, (out_size, out_size), interpolation=cv2.INTER_AREA)

    # Fused soft motion blobs only — no FitLine strokes / tip overlays.
    soft_acc: Optional[np.ndarray] = None
    for r in draw_cams:
        blob = r.fit_gray
        if blob is None:
            blob = r.motion_raw
        if blob is None:
            continue
        if blob.ndim == 3:
            blob = cv2.cvtColor(blob, cv2.COLOR_BGR2GRAY)
        if blob.shape[0] != out_size or blob.shape[1] != out_size:
            blob = cv2.resize(blob, (out_size, out_size), interpolation=cv2.INTER_AREA)
        if soft_acc is None:
            soft_acc = blob.astype(np.uint8, copy=True)
        else:
            np.maximum(soft_acc, blob, out=soft_acc)

    # Blobs-only image: soft motion on black (no board underlay, no lines).
    if soft_acc is not None and int(np.count_nonzero(soft_acc)) > 0:
        canvas = _soft_motion_bgr_vis(soft_acc)
        motion_m = soft_acc > float(FITLINE_WEIGHT_EPS)
        canvas = np.where(motion_m[:, :, None], canvas, 0)
    else:
        canvas = np.zeros((out_size, out_size, 3), dtype=np.uint8)

    stamp = time.strftime("%H%M%S")
    ms = int((time.time() % 1) * 1000)
    dbg = _debug_dir_path()
    label = _zone_short_label(fused.zone_name, fused.segment_number)
    stem = f"hit_{stamp}_{ms:03d}_{label}"
    motion_dir = os.path.join(dbg, "motion")
    try:
        os.makedirs(motion_dir, exist_ok=True)
    except OSError:
        motion_dir = dbg

    # 1) Fused soft-motion blobs only on black (no FitLine strokes).
    fused_path = os.path.join(motion_dir, f"{stem}_fused_motion_raw.png")
    if SAVE_HIT_FUSION_DEBUG or SAVE_MOTION_DEBUG:
        cv2.imwrite(fused_path, canvas)
        print(f"[dart] hit debug: {fused_path}", flush=True)

    # 2) Board underlay + stage2 in/out tint + intersection lines + tip.
    intersect_paths: List[str] = []
    for p in _save_fitline_smooth_debug(
        stem,
        fused,
        line_cams,
        motion_dir,
        base_gray=base_gray,
        soft_acc=soft_acc,
        draw_cams=draw_cams,
    ):
        print(f"[dart] hit fitline smooth debug: {p}", flush=True)
        intersect_paths.append(p)

    return fused_path if (SAVE_HIT_FUSION_DEBUG or SAVE_MOTION_DEBUG) else (
        intersect_paths[0] if intersect_paths else None
    )


def _board_center_and_r(out_size: int = FITLINE_SIZE) -> Tuple[Tuple[float, float], float]:
    cx_out, _view_r, board_r = _warp_radii(out_size)
    return (cx_out, cx_out), board_r


def _point_on_line_closest_to(
    x0: float, y0: float, vx: float, vy: float, px: float, py: float
) -> Tuple[float, float]:
    norm = float(np.hypot(vx, vy))
    if norm < 1e-9:
        return x0, y0
    vx, vy = vx / norm, vy / norm
    dx, dy = px - x0, py - y0
    t = dx * vx + dy * vy
    return x0 + t * vx, y0 + t * vy


def _point_to_line_distance(
    x0: float, y0: float, vx: float, vy: float, px: float, py: float
) -> float:
    """Perpendicular distance from point to infinite line (unit-safe)."""
    norm = float(np.hypot(vx, vy))
    if norm < 1e-9:
        return float(np.hypot(px - x0, py - y0))
    vx, vy = vx / norm, vy / norm
    return float(abs((px - x0) * (-vy) + (py - y0) * vx))


def _line_circle_outer_intersection(
    x0: float,
    y0: float,
    vx: float,
    vy: float,
    cx: float,
    cy: float,
    radius: float,
    toward: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    """Intersection of line with circle closest to `toward` (prefer outer rim snap)."""
    norm = float(np.hypot(vx, vy))
    if norm < 1e-9 or radius <= 1e-6:
        return None
    vx, vy = vx / norm, vy / norm
    fx, fy = x0 - cx, y0 - cy
    b = 2.0 * (fx * vx + fy * vy)
    c = fx * fx + fy * fy - radius * radius
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return None
    s = float(np.sqrt(disc))
    t1 = (-b - s) * 0.5
    t2 = (-b + s) * 0.5
    p1 = (x0 + t1 * vx, y0 + t1 * vy)
    p2 = (x0 + t2 * vx, y0 + t2 * vy)
    d1 = (p1[0] - toward[0]) ** 2 + (p1[1] - toward[1]) ** 2
    d2 = (p2[0] - toward[0]) ** 2 + (p2[1] - toward[1]) ** 2
    return p1 if d1 <= d2 else p2


def _refine_tip_near_double_rim(
    tip: Tuple[float, float],
    line_cams: List[DartDetectResult],
    board_center: Tuple[float, float],
    rings: Dict[str, float],
    ring_ellipses: Optional[Dict] = None,
) -> Tuple[float, float]:
    """If fused/seed tip sits slightly outside double_outer but motion supports the
    double annulus, retract tip onto outermost on-board motion (or line∩rim).

    Detection-only: does not widen scoring radii. Far-outside tips without double-band
    motion stay outside (true MISS).
    """
    tx, ty = float(tip[0]), float(tip[1])
    cx, cy = float(board_center[0]), float(board_center[1])
    d_out = float(rings["double_outer"])
    d_in = float(rings["double_inner"])
    if d_out <= 1e-6:
        return tip

    if ring_ellipses is not None and "double_outer" in ring_ellipses:
        n_do = float(normalized_ellipse_radius(ring_ellipses["double_outer"], tx, ty))
        tip_outside = n_do > 1.0
        tip_r_frac = n_do
    else:
        r_tip = float(np.hypot(tx - cx, ty - cy))
        tip_outside = r_tip > d_out
        tip_r_frac = r_tip / d_out

    # Debug: tip near double band (inside or just outside).
    r_tip_c = float(np.hypot(tx - cx, ty - cy))
    if d_in * 0.96 <= r_tip_c <= d_out * float(TIP_RIM_OUTSIDE_MAX_FRAC):
        print(
            f"[dart] tip near double: r={r_tip_c:.1f} "
            f"d_in={d_in:.1f} d_out={d_out:.1f} outside={tip_outside}",
            flush=True,
        )

    if not tip_outside:
        return tip
    # Only a few px past the wire — not a distant surround miss.
    if tip_r_frac > float(TIP_RIM_OUTSIDE_MAX_FRAC):
        return tip

    xs_list: List[np.ndarray] = []
    ys_list: List[np.ndarray] = []
    for r in line_cams:
        xs, ys, _w = _cam_fit_pixels(r)
        if int(xs.size) > 0:
            xs_list.append(xs.astype(np.float64))
            ys_list.append(ys.astype(np.float64))
    if not xs_list:
        return tip
    xs = np.concatenate(xs_list)
    ys = np.concatenate(ys_list)
    rr = np.hypot(xs - cx, ys - cy)

    near_line = np.zeros(xs.shape, dtype=bool)
    corridor = float(FITLINE_CORRIDOR_PX) * 1.6
    for r in line_cams:
        dist = np.abs(
            (xs - r.line_x0) * (-r.line_vy) + (ys - r.line_y0) * r.line_vx
        ) / max(float(np.hypot(r.line_vx, r.line_vy)), 1e-9)
        near_line |= dist <= corridor

    if ring_ellipses is not None and "double_outer" in ring_ellipses:
        d_out_ell = ring_ellipses["double_outer"]
        d_in_ell = ring_ellipses.get("double_inner")
        n_do_px = np.array(
            [normalized_ellipse_radius(d_out_ell, float(x), float(y)) for x, y in zip(xs, ys)],
            dtype=np.float64,
        )
        if d_in_ell is not None:
            n_di_px = np.array(
                [normalized_ellipse_radius(d_in_ell, float(x), float(y)) for x, y in zip(xs, ys)],
                dtype=np.float64,
            )
            in_double = (n_di_px >= 0.98) & (n_do_px <= 1.02) & near_line
            on_board = near_line & (n_do_px <= 1.02)
            score_r = n_do_px  # larger = more outer on ellipse scale
        else:
            in_double = (rr >= d_in * 0.98) & (n_do_px <= 1.02) & near_line
            on_board = near_line & (n_do_px <= 1.02)
            score_r = n_do_px
    else:
        in_double = (rr >= d_in * 0.98) & (rr <= d_out + 0.75) & near_line
        on_board = near_line & (rr <= d_out + 0.75)
        score_r = rr

    n_band = int(np.count_nonzero(in_double))
    if n_band < int(TIP_RIM_DOUBLE_BAND_MIN_PX):
        if ring_ellipses is not None and "double_outer" in ring_ellipses:
            in_double = (score_r <= 1.02) & (rr >= d_in * 0.98)
        else:
            in_double = (rr >= d_in * 0.98) & (rr <= d_out + 0.75)
        n_band = int(np.count_nonzero(in_double))
    if n_band < int(TIP_RIM_DOUBLE_BAND_MIN_PX):
        return tip

    # Prefer outermost motion still on the scoring face (inside double_outer).
    if int(np.count_nonzero(on_board)) < 3:
        if ring_ellipses is not None and "double_outer" in ring_ellipses:
            on_board = score_r <= 1.02
        else:
            on_board = rr <= d_out + 0.75
    if not np.any(on_board):
        return tip
    idx = int(np.where(on_board)[0][int(np.argmax(score_r[on_board]))])
    cand = (float(xs[idx]), float(ys[idx]))
    # Project onto strongest cam line through tip for concurrence with fuse.
    best_line = line_cams[0]
    best_sup = -1.0
    for r in line_cams:
        sup = _fitline_support_conf(r)
        if sup > best_sup:
            best_sup = sup
            best_line = r
    proj = _point_on_line_closest_to(
        best_line.line_x0,
        best_line.line_y0,
        best_line.line_vx,
        best_line.line_vy,
        cand[0],
        cand[1],
    )
    # If projection slipped outside again, clamp to circular rim along the line.
    if ring_ellipses is not None and "double_outer" in ring_ellipses:
        still_out = float(normalized_ellipse_radius(ring_ellipses["double_outer"], proj[0], proj[1])) > 1.0
    else:
        still_out = float(np.hypot(proj[0] - cx, proj[1] - cy)) > d_out
    if still_out:
        hit = _line_circle_outer_intersection(
            best_line.line_x0,
            best_line.line_y0,
            best_line.line_vx,
            best_line.line_vy,
            cx,
            cy,
            d_out,
            toward=tip,
        )
        if hit is not None:
            proj = hit
    print(
        f"[dart] tip rim-retract: ({tx:.1f},{ty:.1f})->({proj[0]:.1f},{proj[1]:.1f}) "
        f"band_px={n_band}",
        flush=True,
    )
    return float(proj[0]), float(proj[1])


def _cam_fit_pixels(
    r: DartDetectResult,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """xs, ys, weights from fit_gray (preferred) or motion_gray."""
    src = r.fit_gray if r.fit_gray is not None and int(np.count_nonzero(r.fit_gray)) > 0 else r.motion_gray
    if src is None:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float64),
        )
    ys, xs = np.where(src > 0)
    if xs.size == 0:
        return (
            xs.astype(np.int32),
            ys.astype(np.int32),
            np.zeros((0,), dtype=np.float64),
        )
    w = src[ys, xs].astype(np.float64)
    if r.motion_raw is not None and r.motion_raw.shape[:2] == src.shape[:2]:
        w = np.maximum(w, r.motion_raw[ys, xs].astype(np.float64))
    return xs.astype(np.int32), ys.astype(np.int32), w


def _direction_through_point(
    xs: np.ndarray,
    ys: np.ndarray,
    px: float,
    py: float,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Principal direction of pixels about a fixed point (line forced through tip)."""
    dx = xs.astype(np.float64) - float(px)
    dy = ys.astype(np.float64) - float(py)
    n = int(dx.size)
    if n < 2:
        if n == 1:
            nn = float(np.hypot(dx[0], dy[0])) or 1.0
            return float(dx[0] / nn), float(dy[0] / nn)
        return 1.0, 0.0
    if weights is not None and int(weights.size) == n:
        w = np.clip(weights.astype(np.float64), 0.0, None)
        s = float(w.sum())
        if s > 1e-9:
            w = w / s
        else:
            w = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        w = np.full(n, 1.0 / n, dtype=np.float64)
    cxx = float((w * dx * dx).sum())
    cxy = float((w * dx * dy).sum())
    cyy = float((w * dy * dy).sum())
    cov = np.array([[cxx, cxy], [cxy, cyy]], dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(cov)
    _ = eigvals
    vx = float(eigvecs[0, -1])
    vy = float(eigvecs[1, -1])
    nn = float(np.hypot(vx, vy)) or 1.0
    return vx / nn, vy / nn


def _line_pixel_support(
    xs: np.ndarray,
    ys: np.ndarray,
    x0: float,
    y0: float,
    vx: float,
    vy: float,
    *,
    corridor_px: float = FITLINE_CORRIDOR_PX,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Fraction of (weighted) pixels within corridor of the line."""
    n = int(xs.size)
    if n <= 0:
        return 0.0
    dists = np.array(
        [
            _point_to_line_distance(x0, y0, vx, vy, float(xs[i]), float(ys[i]))
            for i in range(n)
        ],
        dtype=np.float64,
    )
    in_c = dists <= float(corridor_px)
    if weights is not None and int(weights.size) == n:
        w = np.clip(weights.astype(np.float64), 0.0, None)
        s = float(w.sum())
        if s > 1e-9:
            return float(w[in_c].sum() / s)
    return float(np.count_nonzero(in_c) / max(n, 1))


def _fuse_tip_from_best_pair(
    line_cams: List[DartDetectResult],
    board_center: Tuple[float, float],
) -> Tuple[Optional[Tuple[float, float]], float, List[DartDetectResult]]:
    """Tip from pairwise consensus — not blob confidence.

    With 2 lines: intersect them.
    With 3+: for each pair, take their intersection and score by sum of
    squared perpendicular distances from that tip to *all* lines (pair ~0,
    third line dominates). Pick the pair whose tip the others agree with.
    Drop a line that flees the consensus tip far beyond the others so one bad
    FitLine cannot pull a 3-line least-squares average.
    """
    n = len(line_cams)
    if n < 2:
        if n == 1:
            return line_cams[0].tip_xy, 0.0, list(line_cams)
        return None, 0.0, []

    if n == 2:
        lines = [
            (r.line_x0, r.line_y0, r.line_vx, r.line_vy) for r in line_cams
        ]
        tip, res = intersect_lines_least_squares(lines, board_center=board_center)
        return tip, float(res), list(line_cams)

    board_r = float(_board_center_and_r(FITLINE_SIZE)[1])
    flee_thresh = max(
        float(FITLINE_CONSENSUS_FLEE_MIN_PX),
        float(FITLINE_CONSENSUS_FLEE_FRAC) * board_r,
    )

    best_tip: Optional[Tuple[float, float]] = None
    best_cost = 1e18
    best_cross = -1.0
    best_pair_idx = (0, 1)
    best_dists: Optional[np.ndarray] = None

    for i in range(n):
        for j in range(i + 1, n):
            a, b = line_cams[i], line_cams[j]
            na = float(np.hypot(a.line_vx, a.line_vy)) or 1.0
            nb = float(np.hypot(b.line_vx, b.line_vy)) or 1.0
            cross = abs(
                a.line_vx / na * b.line_vy / nb - a.line_vy / na * b.line_vx / nb
            )
            # Reject near-parallel pairs (unstable intersection).
            if cross < 0.08:
                continue
            lines = [
                (a.line_x0, a.line_y0, a.line_vx, a.line_vy),
                (b.line_x0, b.line_y0, b.line_vx, b.line_vy),
            ]
            tip, _res = intersect_lines_least_squares(
                lines, board_center=board_center
            )
            if tip is None:
                continue
            dists = np.array(
                [
                    _point_to_line_distance(
                        r.line_x0, r.line_y0, r.line_vx, r.line_vy, tip[0], tip[1]
                    )
                    for r in line_cams
                ],
                dtype=np.float64,
            )
            # Consensus cost: how far ALL lines (esp. the third) miss this tip.
            cost = float(np.sum(dists * dists))
            # Mild geometric preference for well-conditioned pairs on ties.
            cost_adj = cost / (0.35 + cross)
            better = cost_adj < best_cost - 1e-9 or (
                abs(cost_adj - best_cost) < 1e-9 and cross > best_cross
            )
            if better:
                best_cost = cost_adj
                best_cross = cross
                best_tip = tip
                best_pair_idx = (i, j)
                best_dists = dists

    if best_tip is None or best_dists is None:
        lines = [(r.line_x0, r.line_y0, r.line_vx, r.line_vy) for r in line_cams]
        tip, res = intersect_lines_least_squares(lines, board_center=board_center)
        return tip, float(res), list(line_cams)

    pair_i, pair_j = best_pair_idx
    kept: List[DartDetectResult] = []
    for k, r in enumerate(line_cams):
        d = float(best_dists[k])
        if k in (pair_i, pair_j):
            kept.append(r)
            continue
        # Keep third (or further) only if it agrees with the consensus tip.
        if d <= flee_thresh:
            kept.append(r)
        else:
            print(
                f"[fuse] consensus drop cam{r.cam_idx}: "
                f"flee={d:.1f}px thresh={flee_thresh:.1f} "
                f"(pair cams {[line_cams[pair_i].cam_idx, line_cams[pair_j].cam_idx]})",
                flush=True,
            )

    if len(kept) < 2:
        kept = [line_cams[pair_i], line_cams[pair_j]]

    # Soft refine: if 3+ agree, LS among kept; else stick to best-pair tip.
    if len(kept) >= 3:
        lines = [(r.line_x0, r.line_y0, r.line_vx, r.line_vy) for r in kept]
        tip2, res2 = intersect_lines_least_squares(lines, board_center=board_center)
        if tip2 is not None:
            return tip2, float(res2), kept

    lines = [
        (line_cams[pair_i].line_x0, line_cams[pair_i].line_y0,
         line_cams[pair_i].line_vx, line_cams[pair_i].line_vy),
        (line_cams[pair_j].line_x0, line_cams[pair_j].line_y0,
         line_cams[pair_j].line_vx, line_cams[pair_j].line_vy),
    ]
    tip, res = intersect_lines_least_squares(lines, board_center=board_center)
    return tip if tip is not None else best_tip, float(res), kept


def _fitline_support_conf(r: DartDetectResult) -> float:
    """FitLine quality = weighted pixel corridor support (not blob size confidence)."""
    xs, ys, w = _cam_fit_pixels(r)
    if int(xs.size) <= 0:
        return 0.0
    return _line_pixel_support(
        xs,
        ys,
        r.line_x0,
        r.line_y0,
        r.line_vx,
        r.line_vy,
        weights=w if int(w.size) == int(xs.size) else None,
    )


def _pin_line_through_tip(r: DartDetectResult, tip: Tuple[float, float]) -> None:
    """Keep direction; force line origin onto tip so all lines concur."""
    r.line_x0 = float(tip[0])
    r.line_y0 = float(tip[1])


def _try_resnap_line_through_tip(
    r: DartDetectResult,
    tip: Tuple[float, float],
    *,
    board_r: float,
) -> Optional[Tuple[float, float, float, float, float]]:
    """Force line through tip; direction from motion near tip. None if no blob support."""
    xs, ys, w = _cam_fit_pixels(r)
    if int(xs.size) < FITLINE_SHAFT_MIN_PX:
        return None
    tx, ty = float(tip[0]), float(tip[1])
    near_r = max(float(FITLINE_RESNAP_NEAR_TIP_MIN_PX), float(board_r) * float(FITLINE_RESNAP_NEAR_TIP_FRAC))
    dist = np.hypot(xs.astype(np.float64) - tx, ys.astype(np.float64) - ty)
    near = dist <= near_r
    if int(np.count_nonzero(near)) < FITLINE_SHAFT_MIN_PX:
        # No usable motion near tip — do not invent a fleeing shaft.
        return None
    xs_n = xs[near]
    ys_n = ys[near]
    w_n = w[near] if int(w.size) == int(xs.size) else None
    vx, vy = _direction_through_point(xs_n, ys_n, tx, ty, w_n)
    # Prefer orientation matching the original thin-end / tip seed when possible.
    tip_seed = r.tip_xy
    if (tip_seed[0] - tx) * vx + (tip_seed[1] - ty) * vy < 0.0:
        vx, vy = -vx, -vy
    support = _line_pixel_support(xs_n, ys_n, tx, ty, vx, vy, weights=w_n)
    support_all = _line_pixel_support(
        xs, ys, tx, ty, vx, vy, weights=w if int(w.size) == int(xs.size) else None
    )
    conf = 0.65 * support + 0.35 * support_all
    if conf < float(FITLINE_RESNAP_MIN_SUPPORT):
        return None
    return vx, vy, tx, ty, float(conf)


def _fuse_resnap_outlier_lines(
    line_cams: List[DartDetectResult],
    tip: Tuple[float, float],
    *,
    board_r: float,
) -> List[DartDetectResult]:
    """On flee: re-snap lowest-fitLine-conf line; force all used lines through tip.

    Trigger = any line misses tip by FITLINE_RESNAP_MISS_PX (ignore blob confidence).
    Cam with no motion near tip is dropped; others are pinned so tip is concurrent.
    """
    if len(line_cams) < 2:
        return line_cams

    dists = [
        _point_to_line_distance(r.line_x0, r.line_y0, r.line_vx, r.line_vy, tip[0], tip[1])
        for r in line_cams
    ]
    fleers = [i for i, d in enumerate(dists) if d >= float(FITLINE_RESNAP_MISS_PX)]
    if not fleers:
        # Already close enough — still pin origins so debug lines meet the yellow tip.
        for r in line_cams:
            _pin_line_through_tip(r, tip)
        return line_cams

    supports = [_fitline_support_conf(r) for r in line_cams]
    # Flee is the trigger; choose which line to re-snap by lowest fitLine support
    # (not DartDetectResult.confidence / blob size).
    snap_i = int(np.argmin(np.asarray(supports, dtype=np.float64)))
    # If the weakest fitline is not fleeing, still prefer the worst fleer.
    if snap_i not in fleers:
        snap_i = min(fleers, key=lambda i: (supports[i], -dists[i]))

    snap_r = line_cams[snap_i]
    print(
        f"[fuse] flee->resnap cam{snap_r.cam_idx} "
        f"dist={dists[snap_i]:.1f}px fitline_sup={supports[snap_i]:.2f} "
        f"(blob_conf={snap_r.confidence:.2f})",
        flush=True,
    )
    snapped = _try_resnap_line_through_tip(snap_r, tip, board_r=board_r)

    kept: List[DartDetectResult] = []
    for i, r in enumerate(line_cams):
        if i == snap_i:
            if snapped is None:
                print(
                    f"[fuse] drop cam{r.cam_idx}: flee + no motion near tip",
                    flush=True,
                )
                continue
            vx, vy, x0, y0, _conf = snapped
            r.line_vx = float(vx)
            r.line_vy = float(vy)
            r.line_x0 = float(x0)
            r.line_y0 = float(y0)
            r.line_angle_deg = _fitline_angle_deg(vx, vy)
            kept.append(r)
            continue

        # Remaining cams: drop only if they flee badly AND have no tip-local motion.
        if dists[i] >= float(FITLINE_OUTLIER_CAM_DIST_PX):
            alt = _try_resnap_line_through_tip(r, tip, board_r=board_r)
            if alt is None:
                print(
                    f"[fuse] drop cam{r.cam_idx}: outlier flee + no tip support",
                    flush=True,
                )
                continue
            vx, vy, x0, y0, _conf = alt
            r.line_vx = float(vx)
            r.line_vy = float(vy)
            r.line_x0 = float(x0)
            r.line_y0 = float(y0)
            r.line_angle_deg = _fitline_angle_deg(vx, vy)
            kept.append(r)
            continue

        _pin_line_through_tip(r, tip)
        kept.append(r)

    if not kept:
        return list(line_cams)

    # Force geometric concurrence: every kept line passes through the same tip.
    for r in kept:
        _pin_line_through_tip(r, tip)
    return kept


def _fuse_drop_outlier_cam(
    line_cams: List[DartDetectResult],
    tip: Tuple[float, float],
    residual: float,
    board_center: Tuple[float, float],
) -> Tuple[List[DartDetectResult], Tuple[float, float], float]:
    """Ako jedna kamera vuče sjecište daleko, izbaci je i refuziraj (3+ cam)."""
    if len(line_cams) < 3 or tip is None:
        return line_cams, tip, residual

    dists = np.array(
        [
            _point_to_line_distance(r.line_x0, r.line_y0, r.line_vx, r.line_vy, tip[0], tip[1])
            for r in line_cams
        ],
        dtype=np.float64,
    )
    worst_i = int(np.argmax(dists))
    med = float(np.median(dists))
    worst = float(dists[worst_i])
    board_r = float(_board_center_and_r(FITLINE_SIZE)[1])
    residual_bad = residual >= max(board_r * 0.04, 6.0)
    drop_by_dist = worst >= float(FITLINE_OUTLIER_CAM_DIST_PX) and worst >= max(
        med * float(FITLINE_OUTLIER_CAM_RATIO), 4.0
    )
    if not drop_by_dist and not residual_bad:
        return line_cams, tip, residual

    # Prefer dropping the farthest cam; if residual is bad, try each leave-one-out.
    trial_idx = list(range(len(line_cams))) if residual_bad else [worst_i]
    best_kept = line_cams
    best_tip = tip
    best_res = residual
    for drop_i in trial_idx:
        kept = [r for i, r in enumerate(line_cams) if i != drop_i]
        lines = [(r.line_x0, r.line_y0, r.line_vx, r.line_vy) for r in kept]
        tip2, res2 = intersect_lines_least_squares(lines, board_center=board_center)
        if tip2 is None:
            continue
        if res2 < best_res:
            best_kept, best_tip, best_res = kept, tip2, res2

    if best_res < residual * 0.85 or (
        drop_by_dist and best_kept is not line_cams and best_res <= residual * 1.05
    ):
        return best_kept, best_tip, best_res
    return line_cams, tip, residual


def intersect_lines_least_squares(
    lines: List[Tuple[float, float, float, float]],
    *,
    board_center: Tuple[float, float],
) -> Tuple[Optional[Tuple[float, float]], float]:
    """Sjeciste 2+ pravca (x0,y0,vx,vy). Vraca (tip, residual)."""
    if not lines:
        return None, 0.0
    if len(lines) == 1:
        x0, y0, vx, vy = lines[0]
        cx, cy = board_center
        pt = _point_on_line_closest_to(x0, y0, vx, vy, cx, cy)
        return pt, 0.0

    A: List[List[float]] = []
    b: List[float] = []
    for x0, y0, vx, vy in lines:
        norm = float(np.hypot(vx, vy))
        if norm < 1e-9:
            continue
        vx, vy = vx / norm, vy / norm
        nx, ny = -vy, vx
        A.append([nx, ny])
        b.append(nx * x0 + ny * y0)
    if len(A) < 2:
        return intersect_lines_least_squares([lines[0]], board_center=board_center)
    A_arr = np.array(A, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    pt, residuals, _, _ = np.linalg.lstsq(A_arr, b_arr, rcond=None)
    residual = float(np.sqrt(residuals[0] / max(len(A), 1))) if len(residuals) else 0.0
    return (float(pt[0]), float(pt[1])), residual


def score_topdown_point(
    px: float,
    py: float,
    *,
    out_size: int = FITLINE_SIZE,
    cal: Optional[BoardCalibration] = None,
    cals: Optional[List[Optional[BoardCalibration]]] = None,
) -> Tuple[int, str, int]:
    """Score u aligned top-down prostoru (seg.20 gore, offset=0)."""
    center, board_r = _board_center_and_r(out_size)
    ring_ellipses = scoring_ring_ellipses(out_size, cal=cal, cals=cals)
    rings = scoring_ring_radii(out_size, cal=cal, cals=cals)
    return topdown_point_to_score(
        px,
        py,
        center,
        board_r,
        segment20_offset=0,
        rings=rings,
        ring_ellipses=ring_ellipses,
    )


def detect_dart_on_topdown(
    topdown_bgr: np.ndarray,
    ref_gray: np.ndarray,
    cal: BoardCalibration,
    *,
    cam_idx: int = -1,
    save_debug: bool = SAVE_MOTION_DEBUG,
    draw_fused_tip: Optional[Tuple[float, float]] = None,
) -> DartDetectResult:
    """Motion gating + intensity-weighted FitLine on soft grayscale (FITLINE_SIZE).

    Soft absdiff (board ROI) gated by dilated morph motion → resize to
    FITLINE_SIZE → mild Gaussian blur → radial flight downweight →
    intensity-weighted PCA (stage1 line). Multicam tip fuse then runs stage2
    FitLine inside a tip-radius ROI (see fuse_multicam_lines /
    FITLINE_STAGE2_TIP_RADIUS_FRAC). Tip = multicam intersect mapped back with
    scale_x/y (identity when warp is already FITLINE_SIZE).
    """
    h, w = topdown_bgr.shape[:2]
    cx_out, view_r, board_r = _warp_radii(w)
    board_center = (float(cx_out), float(cx_out))

    cur_gray = cv2.cvtColor(topdown_bgr, cv2.COLOR_BGR2GRAY)
    if ref_gray.shape != cur_gray.shape:
        ref_gray = cv2.resize(ref_gray, (w, h), interpolation=cv2.INTER_AREA)

    raw_diff = cv2.absdiff(cur_gray, ref_gray)
    diff_mean = float(np.mean(raw_diff))
    mask = _motion_mask(ref_gray, cur_gray)
    board_mask = np.zeros((h, w), dtype=np.uint8)
    # Detection ROI: full warp view (includes tip just outside double wire).
    # Scoring radii are unchanged — this is coverage only.
    det_r = max(float(view_r), float(board_r) * float(BOARD_SISAL_EDGE_FRAC) * 0.92)
    det_r = min(det_r, float(cx_out) - 1.0)
    cv2.circle(board_mask, (int(cx_out), int(cx_out)), int(round(det_r)), 255, -1)
    mask = cv2.bitwise_and(mask, mask, mask=board_mask)

    # Binary morph mask: motion count + soft FitLine pixel gate (dilated).
    ys_m, xs_m = np.where(mask > 0)
    n_px = len(xs_m)
    if n_px < MOTION_MIN_PIXELS:
        if save_debug:
            _save_motion_debug(
                cam_idx, mask, motion_pixels=n_px, diff_mean=diff_mean, status="low_motion"
            )
        return DartDetectResult(
            cam_idx=cam_idx,
            found=False,
            motion_pixels=n_px,
            diff_mean=diff_mean,
            reject_reason="low_motion",
            motion_gray=mask.copy(),
            motion_raw=raw_diff.copy(),
        )
    if n_px > MOTION_MAX_PIXELS:
        if save_debug:
            _save_motion_debug(
                cam_idx, mask, motion_pixels=n_px, diff_mean=diff_mean, status="high_motion"
            )
        return DartDetectResult(
            cam_idx=cam_idx,
            found=False,
            motion_pixels=n_px,
            diff_mean=diff_mean,
            reject_reason="high_motion",
            motion_gray=mask.copy(),
            motion_raw=raw_diff.copy(),
        )

    soft = _soft_motion_gray_for_fit(raw_diff, board_mask)
    soft = _gate_soft_motion_for_fit(soft, mask)
    axis_ok, (vx, vy, x0, y0), ds_blur, scale_x, scale_y, xs_ds, ys_ds = (
        _weighted_fitline_on_soft_motion(soft, board_center=board_center)
    )
    if int(xs_ds.size) < FITLINE_MIN_PX or not axis_ok:
        return DartDetectResult(
            cam_idx=cam_idx,
            found=False,
            motion_pixels=n_px,
            diff_mean=diff_mean,
            reject_reason="tip_radius_empty",
            motion_gray=mask,
            motion_raw=raw_diff.copy(),
            fit_gray=np.zeros_like(mask),
            fit_ds_gray=ds_blur,
            fit_scale_x=scale_x,
            fit_scale_y=scale_y,
            radius_tip=(0.0, 0.0),
        )

    line_angle = _fitline_angle_deg(vx, vy)
    # Provisional tip = board-center projection (fuse overwrites for multi-cam).
    bc_ds = _map_original_xy_to_ds(board_center[0], board_center[1], scale_x, scale_y)
    tip_ds = _point_on_line_closest_to(x0, y0, vx, vy, bc_ds[0], bc_ds[1])
    tip_x, tip_y = _map_ds_xy_to_original(tip_ds[0], tip_ds[1], scale_x, scale_y)

    # Original-size fit_gray = upsample of blurred soft motion (debug / support).
    fit_gray = cv2.resize(ds_blur, (w, h), interpolation=cv2.INTER_LINEAR)

    span_ds = float(
        np.hypot(
            float(xs_ds.max()) - float(xs_ds.min()),
            float(ys_ds.max()) - float(ys_ds.min()),
        )
    )
    span = span_ds * 0.5 * (scale_x + scale_y)
    conf = min(1.0, n_px / 180.0) * min(1.0, span / max(board_r * 0.35, 1.0))
    if not axis_ok:
        conf *= 0.7

    number, zone, score = 0, "miss", 0
    if draw_fused_tip is not None:
        number, zone, score = score_topdown_point(
            draw_fused_tip[0], draw_fused_tip[1], out_size=w, cal=cal
        )

    debug = None
    if save_debug:
        debug = topdown_bgr.copy()
        ox0, oy0 = _map_ds_xy_to_original(x0, y0, scale_x, scale_y)
        p1 = (int(ox0 - vx * 200), int(oy0 - vy * 200))
        p2 = (int(ox0 + vx * 200), int(oy0 + vy * 200))
        cv2.line(debug, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
        dbg_m = cv2.cvtColor(fit_gray, cv2.COLOR_GRAY2BGR)
        status = _zone_short_label(zone, number) if draw_fused_tip is not None else "line"
        if cam_idx >= 0:
            _save_motion_debug(
                cam_idx, dbg_m, fit_img=debug, motion_pixels=n_px, diff_mean=diff_mean, status=status
            )

    return DartDetectResult(
        cam_idx=cam_idx,
        found=True,
        tip_xy=(tip_x, tip_y),
        line_vx=vx,
        line_vy=vy,
        line_x0=x0,
        line_y0=y0,
        line_angle_deg=line_angle,
        confidence=conf,
        segment_number=number,
        zone_name=zone,
        score=score,
        motion_pixels=n_px,
        diff_mean=diff_mean,
        debug_bgr=debug,
        motion_gray=mask,
        motion_raw=raw_diff.copy(),
        fit_gray=fit_gray,
        fit_ds_gray=ds_blur,
        fit_scale_x=scale_x,
        fit_scale_y=scale_y,
        radius_tip=(tip_x, tip_y),
    )


def _raw_surround_vicinity_mask(shape: Tuple[int, int], ellipse) -> np.ndarray:
    outer = _scale_ellipse(ellipse, RAW_SURROUND_OUTER_SCALE)
    return _board_ellipse_mask(shape, outer, shrink=1.0)


def _count_raw_surround_motion(
    gray: np.ndarray,
    ref_gray: np.ndarray,
    cal: BoardCalibration,
) -> Tuple[int, float]:
    if cal.double_ellipse is None:
        return 0, 0.0
    h, w = gray.shape[:2]
    ref_g = ref_gray if ref_gray.shape == gray.shape else cv2.resize(
        ref_gray, (w, h), interpolation=cv2.INTER_AREA
    )
    mask = _motion_mask(ref_g, gray)
    inside = _board_ellipse_mask((h, w), cal.double_ellipse, shrink=0.99)
    vicinity = _raw_surround_vicinity_mask((h, w), cal.double_ellipse)
    surround_zone = cv2.bitwise_and(vicinity, cv2.bitwise_not(inside))
    mask = cv2.bitwise_and(mask, mask, mask=surround_zone)
    n_px = int(np.count_nonzero(mask))
    if n_px > 0:
        diff_mean = float(np.mean(cv2.absdiff(gray, ref_g)[surround_zone > 0]))
    else:
        diff_mean = 0.0
    return n_px, diff_mean


def detect_dart_offboard_raw(
    frame_bgr: np.ndarray,
    ref_gray: np.ndarray,
    cal: BoardCalibration,
    *,
    cam_idx: int = -1,
    save_debug: bool = SAVE_MOTION_DEBUG,
) -> DartDetectResult:
    """Motion na ne-warpanoj slici u surround zoni -> MISS izvan ploce."""
    if cal.double_ellipse is None:
        return DartDetectResult(cam_idx=cam_idx, found=False, reject_reason="no_ellipse")

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    ref_g = ref_gray if ref_gray.shape == gray.shape else cv2.resize(
        ref_gray, (w, h), interpolation=cv2.INTER_AREA
    )
    diff_mean = float(np.mean(cv2.absdiff(gray, ref_g)))
    mask = _motion_mask(ref_g, gray)
    ellipse = cal.double_ellipse
    inside = _board_ellipse_mask((h, w), ellipse, shrink=0.99)
    vicinity = _raw_surround_vicinity_mask((h, w), ellipse)
    surround_zone = cv2.bitwise_and(vicinity, cv2.bitwise_not(inside))
    mask = cv2.bitwise_and(mask, mask, mask=surround_zone)

    ys, xs = np.where(mask > 0)
    n_px = len(xs)
    if n_px < RAW_SURROUND_MIN_PIXELS:
        if save_debug and cam_idx >= 0:
            _save_motion_debug(
                cam_idx, mask, motion_pixels=n_px, diff_mean=diff_mean, status="raw_low"
            )
        return DartDetectResult(
            cam_idx=cam_idx,
            found=False,
            motion_pixels=n_px,
            diff_mean=diff_mean,
            reject_reason="raw_low_motion",
        )

    step = max(1, n_px // 200)
    sample_r = [
        normalized_ellipse_radius(ellipse, float(xs[i]), float(ys[i]))
        for i in range(0, n_px, step)
    ]
    outside_frac = sum(1 for r in sample_r if r > 1.02) / max(len(sample_r), 1)
    if outside_frac < 0.72:
        return DartDetectResult(
            cam_idx=cam_idx,
            found=False,
            motion_pixels=n_px,
            diff_mean=diff_mean,
            reject_reason="raw_inside_board",
        )

    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    line = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    vx, vy = float(line[0]), float(line[1])
    x0, y0 = float(line[2]), float(line[3])
    bx, by = cal.center
    d2 = (xs.astype(np.float64) - bx) ** 2 + (ys.astype(np.float64) - by) ** 2
    far_i = int(np.argmax(d2))
    tip_x, tip_y = _point_on_line_closest_to(
        x0, y0, vx, vy, float(xs[far_i]), float(ys[far_i])
    )
    tip_r = normalized_ellipse_radius(ellipse, tip_x, tip_y)
    if tip_r < 1.02:
        return DartDetectResult(
            cam_idx=cam_idx,
            found=False,
            motion_pixels=n_px,
            diff_mean=diff_mean,
            reject_reason="raw_tip_on_board",
        )
    conf = min(1.0, n_px / 140.0) * min(1.0, outside_frac + 0.15)

    debug = frame_bgr.copy()
    cv2.line(
        debug,
        (int(x0 - vx * 120), int(y0 - vy * 120)),
        (int(x0 + vx * 120), int(y0 + vy * 120)),
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.circle(debug, (int(tip_x), int(tip_y)), 5, (0, 80, 255), -1, cv2.LINE_AA)
    cv2.putText(
        debug, "RAW MISS", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1, cv2.LINE_AA
    )
    if save_debug and cam_idx >= 0:
        _save_motion_debug(
            cam_idx,
            mask,
            fit_img=debug,
            motion_pixels=n_px,
            diff_mean=diff_mean,
            status="raw_miss",
        )

    return DartDetectResult(
        cam_idx=cam_idx,
        found=True,
        tip_xy=(tip_x, tip_y),
        line_vx=vx,
        line_vy=vy,
        line_x0=x0,
        line_y0=y0,
        line_angle_deg=_fitline_angle_deg(vx, vy),
        confidence=conf,
        segment_number=0,
        zone_name="miss",
        score=0,
        motion_pixels=n_px,
        diff_mean=diff_mean,
        reject_reason="off_board_raw",
        debug_bgr=debug,
    )


def _merge_cam_detect_results(
    warped: DartDetectResult,
    raw: DartDetectResult,
) -> DartDetectResult:
    """Spoji top-down i raw rezultat jedne kamere."""
    merged_px = max(warped.motion_pixels, raw.motion_pixels)
    warped_has_line = warped.found and warped.motion_pixels >= MOTION_MIN_PIXELS
    if warped_has_line:
        warped.motion_pixels = merged_px
        return warped
    if raw.found and raw.reject_reason == "off_board_raw":
        raw.motion_pixels = merged_px
        return raw
    if warped.found:
        warped.motion_pixels = merged_px
        return warped
    best = warped if warped.motion_pixels >= raw.motion_pixels else raw
    best.motion_pixels = merged_px
    return best


def _false_bull_reject_reason(
    line_cams: List[DartDetectResult],
    board_center: Tuple[float, float],
    rings: Dict[str, float],
) -> str:
    """Leftover radial shaft after a real hit fuses to bull (esp. 1-cam → center)."""
    if len(line_cams) < BULL_MIN_CAMS:
        return "bull_need_2cam"
    cents: List[Tuple[float, float]] = []
    for r in line_cams:
        c = _soft_motion_centroid(r)
        if c is not None:
            cents.append(c)
    if not cents:
        return ""
    mx = float(sum(c[0] for c in cents) / len(cents))
    my = float(sum(c[1] for c in cents) / len(cents))
    dist = float(np.hypot(mx - board_center[0], my - board_center[1]))
    limit = float(rings.get("bull_outer", 0.0)) * BULL_MOTION_MAX_BULL_OUTER
    if limit > 1.0 and dist > limit:
        return "bull_motion_far"
    return ""


def fuse_multicam_lines(
    per_cam: List[DartDetectResult],
    *,
    board_calibrator: Optional[BoardCalibrator] = None,
) -> FusedDartResult:
    """Spoji fitline s vise kamera u jedan tip (sjeciste pravaca).

    2-stage FitLine:
      1) per-cam stage1 lines → pairwise consensus tip fuse
      2) re-FitLine all cams on soft pixels inside tip radius → fuse again

    Consensus (3+ cams): score each pair's intersection by sum of squared
    distances to all lines; keep the agreeing pair and drop a fleeing outlier.
    Tip = sjecište u FITLINE_SIZE, zatim map u original top-down scoring coords.
    """
    ring_cals: List[Optional[BoardCalibration]] = []
    if board_calibrator is not None:
        for r in per_cam:
            ring_cals.append(board_calibrator.get(int(r.cam_idx)))

    board_line_cams = [
        r
        for r in per_cam
        if r.found
        and r.motion_pixels >= MOTION_MIN_PIXELS
        and r.reject_reason != "off_board_raw"
    ]
    line_cams = board_line_cams
    raw_miss_cams = [
        r
        for r in per_cam
        if r.found
        and r.reject_reason == "off_board_raw"
    ]
    # Većina kamera: surround miss — jedna kriva on-board linija ne smije pobijediti.
    if len(raw_miss_cams) >= 2 and len(raw_miss_cams) > len(line_cams):
        return FusedDartResult(
            found=True,
            tip_xy=raw_miss_cams[0].tip_xy,
            zone_name="miss",
            score=0,
            segment_number=0,
            confidence=max(0.40, float(np.mean([r.confidence for r in raw_miss_cams]))),
            cam_indices=[r.cam_idx for r in raw_miss_cams],
            per_cam=per_cam,
            reject_reason="miss_majority",
        )
    if not line_cams:
        # Jedna kamera u surroundu je često flicker/sjena — miss samo uz 2+ kamere.
        if len(raw_miss_cams) >= 2:
            return FusedDartResult(
                found=True,
                tip_xy=raw_miss_cams[0].tip_xy,
                zone_name="miss",
                score=0,
                segment_number=0,
                confidence=max(0.38, float(np.mean([r.confidence for r in raw_miss_cams]))),
                cam_indices=[r.cam_idx for r in raw_miss_cams],
                per_cam=per_cam,
                reject_reason="off_board_raw",
            )
        return FusedDartResult(found=False, per_cam=per_cam, reject_reason="no_lines")

    # Scoring / board geometry match detection warp (= FITLINE_SIZE), not preview.
    board_center, board_r = _board_center_and_r(FITLINE_SIZE)
    # Uniform resize of full frame — same scale for all cams; use first cam's actual dims.
    scale_x = float(line_cams[0].fit_scale_x) if float(line_cams[0].fit_scale_x) > 1e-9 else 1.0
    scale_y = float(line_cams[0].fit_scale_y) if float(line_cams[0].fit_scale_y) > 1e-9 else 1.0
    board_center_ds = _map_original_xy_to_ds(
        board_center[0], board_center[1], scale_x, scale_y
    )

    # --- Stage 1 tip fuse from preliminary per-cam FitLines ---
    # Keep all cams for stage2 refit; consensus may drop a fleeer only for tip.
    all_line_cams = list(line_cams)
    tip_ds, residual, line_cams = _estimate_fused_tip_ds(all_line_cams, board_center_ds)
    if tip_ds is None:
        return FusedDartResult(found=False, per_cam=per_cam, reject_reason="no_intersect")

    # --- Stage 2: re-FitLine on soft pixels inside tip ROI, then fuse again ---
    # Refit every cam near the consensus tip — a bad stage1 direction may recover.
    mean_scale = max(0.5 * (scale_x + scale_y), 1e-9)
    board_r_ds = float(board_r) / mean_scale
    tip_roi_r_ds = float(FITLINE_STAGE2_TIP_RADIUS_FRAC) * board_r_ds
    # Persist ROI in original warp space for hit debug overlays.
    stage2_roi_tip_xy = _map_ds_xy_to_original(tip_ds[0], tip_ds[1], scale_x, scale_y)
    stage2_roi_radius = float(FITLINE_STAGE2_TIP_RADIUS_FRAC) * float(board_r)
    _stage2_refit_lines_near_tip(
        all_line_cams,
        tip_ds,
        tip_roi_r_ds,
        board_center,
        scale_x=scale_x,
        scale_y=scale_y,
    )
    tip_ds, residual, line_cams = _estimate_fused_tip_ds(all_line_cams, board_center_ds)
    if tip_ds is None:
        return FusedDartResult(found=False, per_cam=per_cam, reject_reason="no_intersect")

    tip = _map_ds_xy_to_original(tip_ds[0], tip_ds[1], scale_x, scale_y)
    tx_ds, ty_ds = float(tip_ds[0]), float(tip_ds[1])
    for r in line_cams:
        proj_ds = _point_on_line_closest_to(
            r.line_x0, r.line_y0, r.line_vx, r.line_vy, tx_ds, ty_ds
        )
        sx = float(r.fit_scale_x) if float(r.fit_scale_x) > 1e-9 else scale_x
        sy = float(r.fit_scale_y) if float(r.fit_scale_y) > 1e-9 else scale_y
        proj = _map_ds_xy_to_original(proj_ds[0], proj_ds[1], sx, sy)
        r.tip_xy = proj
        r.radius_tip = proj

    tx, ty = tip
    cx_out, _ = board_center
    r_tip = float(np.hypot(tx - cx_out, ty - cx_out))
    ring_ellipses = scoring_ring_ellipses(FITLINE_SIZE, cals=ring_cals)
    rings = scoring_ring_radii(FITLINE_SIZE, cals=ring_cals)
    # Rim-retract intentionally disabled for plain FitLine path.

    if ring_ellipses is not None:
        sisal_ell = _scale_ellipse_axes(ring_ellipses["double_outer"], BOARD_SISAL_EDGE_FRAC)
        off_board = normalized_ellipse_radius(sisal_ell, tx, ty) > 1.10
    else:
        off_board = r_tip > rings["sisal_edge"] * 1.10
    if off_board:
        return FusedDartResult(
            found=True,
            tip_xy=tip,
            zone_name="miss",
            score=0,
            segment_number=0,
            confidence=max(0.35, float(np.mean([r.confidence for r in line_cams]))),
            cam_indices=[r.cam_idx for r in line_cams],
            per_cam=per_cam,
            reject_reason="off_board",
            stage2_roi_tip_xy=stage2_roi_tip_xy,
            stage2_roi_radius=stage2_roi_radius,
        )

    number, zone, score = score_topdown_point(
        tx, ty, out_size=FITLINE_SIZE, cals=ring_cals
    )
    if score <= 0:
        return FusedDartResult(
            found=True,
            tip_xy=tip,
            zone_name="miss",
            score=0,
            segment_number=0,
            confidence=0.35,
            cam_indices=[r.cam_idx for r in line_cams],
            per_cam=per_cam,
            reject_reason="miss",
            stage2_roi_tip_xy=stage2_roi_tip_xy,
            stage2_roi_radius=stage2_roi_radius,
        )

    if zone in ("inner_bull", "outer_bull"):
        bull_reject = _false_bull_reject_reason(
            line_cams, (cx_out, board_center[1]), rings
        )
        if bull_reject:
            return FusedDartResult(
                found=False,
                tip_xy=tip,
                per_cam=per_cam,
                reject_reason=bull_reject,
                cam_indices=[r.cam_idx for r in line_cams],
                stage2_roi_tip_xy=stage2_roi_tip_xy,
                stage2_roi_radius=stage2_roi_radius,
            )

    mean_conf = float(np.mean([r.confidence for r in line_cams]))
    residual_orig = float(residual) * 0.5 * (scale_x + scale_y)
    residual_pen = min(1.0, residual_orig / max(board_r * 0.08, 1.0))
    cam_bonus = min(1.0, len(line_cams) / 3.0)
    confidence = mean_conf * cam_bonus * (1.0 - 0.35 * residual_pen)

    return FusedDartResult(
        found=True,
        tip_xy=tip,
        confidence=confidence,
        segment_number=number,
        zone_name=zone,
        score=score,
        cam_indices=[r.cam_idx for r in line_cams],
        per_cam=per_cam,
        stage2_roi_tip_xy=stage2_roi_tip_xy,
        stage2_roi_radius=stage2_roi_radius,
    )


def _estimate_fused_tip_ds(
    line_cams: List[DartDetectResult],
    board_center_ds: Tuple[float, float],
) -> Tuple[Optional[Tuple[float, float]], float, List[DartDetectResult]]:
    """Fuse tip from current cam lines in ds space via pairwise consensus.

    With 3+ cams, prefers the pair whose intersection the third line agrees
    with; drops a fleeing outlier so one bad FitLine cannot dominate.
    """
    cams = list(line_cams)
    tip_ds: Optional[Tuple[float, float]] = None
    residual = 0.0
    if len(cams) == 1:
        r0 = cams[0]
        tip_ds = _point_on_line_closest_to(
            r0.line_x0,
            r0.line_y0,
            r0.line_vx,
            r0.line_vy,
            board_center_ds[0],
            board_center_ds[1],
        )
        residual = 0.0
    else:
        tip_ds, residual, cams = _fuse_tip_from_best_pair(cams, board_center_ds)
        if tip_ds is None:
            cams = list(line_cams)
            lines = [(r.line_x0, r.line_y0, r.line_vx, r.line_vy) for r in cams]
            tip_ds, residual = intersect_lines_least_squares(
                lines, board_center=board_center_ds
            )
        # Safety net: if consensus kept 3 but one still flees badly, drop it.
        if tip_ds is not None and len(cams) >= 3:
            cams, tip_ds, residual = _fuse_drop_outlier_cam(
                cams, tip_ds, residual, board_center_ds
            )
    return tip_ds, float(residual), cams


def _stage2_refit_lines_near_tip(
    line_cams: List[DartDetectResult],
    tip_ds: Tuple[float, float],
    radius_ds: float,
    board_center_orig: Tuple[float, float],
    *,
    scale_x: float,
    scale_y: float,
) -> None:
    """In-place stage-2 FitLine: soft intensity inside tip circle only (no CC)."""
    tip_orig = _map_ds_xy_to_original(tip_ds[0], tip_ds[1], scale_x, scale_y)
    radius_orig = float(radius_ds) * max(0.5 * (scale_x + scale_y), 1e-9)
    for r in line_cams:
        ds = r.fit_ds_gray
        if ds is None:
            continue
        sx = float(r.fit_scale_x) if float(r.fit_scale_x) > 1e-9 else scale_x
        sy = float(r.fit_scale_y) if float(r.fit_scale_y) > 1e-9 else scale_y
        tip_cam = _map_original_xy_to_ds(tip_orig[0], tip_orig[1], sx, sy)
        r_cam = float(radius_orig) / max(0.5 * (sx + sy), 1e-9)
        bc_cam = _map_original_xy_to_ds(
            board_center_orig[0], board_center_orig[1], sx, sy
        )
        refined = _refit_fitline_near_tip(ds, tip_cam, r_cam, bc_cam)
        if refined is None:
            continue
        vx, vy, x0, y0 = refined
        r.line_vx = float(vx)
        r.line_vy = float(vy)
        r.line_x0 = float(x0)
        r.line_y0 = float(y0)
        r.line_angle_deg = _fitline_angle_deg(vx, vy)


class DartMotionDetector:
    """Referentni top-down kadrovi + motion state machine + 3-kamera fusion."""

    def __init__(self) -> None:
        self._refs: Dict[int, np.ndarray] = {}
        self._raw_refs: Dict[int, np.ndarray] = {}
        self._empty_refs: Dict[int, np.ndarray] = {}
        self._empty_raw_refs: Dict[int, np.ndarray] = {}
        self._motion_state: str = "idle"
        self._armed_frames: int = 0
        self._clear_frames: int = 0
        self._defer_snapshot: bool = False
        self._settle_prev_gray: Dict[int, np.ndarray] = {}
        self._settle_prev_warped: Dict[int, np.ndarray] = {}
        # Zadnji stabilni BGR kadrovi za fire (ne loviti USB lag na trenutku fuse).
        self._fire_frames: Dict[int, np.ndarray] = {}
        # Warp gray iz ovog ticka — probe/settle/hand dijele isti warp (bez 2–3x remap).
        self._tick_warped_gray: Dict[int, np.ndarray] = {}
        self._hand_stable_streak: int = 0
        self._hand_cooldown_frames: int = 0
        self._post_hit_until: float = 0.0
        self._post_hit_need_clear: bool = False
        self._pending_ui_commit: bool = False
        self._last_empty_td_px: int = 0
        self._last_empty_raw_px: int = 0
        self._last_empty_td_diff: float = 0.0
        n = self.load_refs_persistent()
        nr = self.load_raw_refs_persistent()
        ne = self.load_empty_refs_persistent()
        if n > 0:
            print(f"[dart] ucitano {n} motion ref s diska", flush=True)
        if nr > 0:
            print(f"[dart] ucitano {nr} raw ref s diska", flush=True)
        if ne > 0:
            print(f"[dart] ucitano {ne} empty ref s diska", flush=True)
        ner = len(self._empty_raw_refs)
        if ner > 0:
            print(f"[dart] ucitano {ner} empty raw ref s diska", flush=True)

    def clear(self) -> None:
        self._refs.clear()
        self._raw_refs.clear()
        self._empty_refs.clear()
        self._empty_raw_refs.clear()
        self.reset_motion_state()
        d = motion_refs_dir()
        if os.path.isdir(d):
            for name in os.listdir(d):
                if (
                    name.startswith("cam_")
                    or name.startswith("raw_cam_")
                    or name.startswith("empty_cam_")
                    or name.startswith("empty_raw_cam_")
                ):
                    try:
                        os.remove(os.path.join(d, name))
                    except OSError:
                        pass

    def reset_motion_state(self) -> None:
        self._motion_state = "idle"
        self._armed_frames = 0
        self._clear_frames = 0
        self._defer_snapshot = False
        self._settle_prev_gray.clear()
        self._settle_prev_warped.clear()
        self._fire_frames.clear()
        self._hand_stable_streak = 0
        self._hand_cooldown_frames = 0
        self._post_hit_until = 0.0
        self._post_hit_need_clear = False
        self._pending_ui_commit = False

    def defer_board_snapshot(self) -> None:
        """Odgodi motion ref do mirne ploce (npr. nakon MISS izvan ploce)."""
        self._defer_snapshot = True

    def try_commit_deferred_snapshot(
        self,
        board_calibrator: BoardCalibrator,
        frames: Dict[int, Optional[np.ndarray]],
    ) -> bool:
        if not self._defer_snapshot or self._motion_state != "idle":
            return False
        probe, _max_px = self._probe_motion_vs_ref_map(board_calibrator, frames, self._refs)
        if self._max_motion_pixels(probe) >= MOTION_CLEAR_PIXELS:
            return False
        saved = self.commit_board_snapshot(board_calibrator, frames)
        self._defer_snapshot = False
        return saved > 0

    def finalize_shot(self, *, enter_rearm: bool) -> None:
        """Nakon upisa hita: idle + kratki cooldown. Rearm je ugašen (zaglavljivao se na šumu)."""
        _ = enter_rearm
        self._armed_frames = 0
        self._clear_frames = 0
        self._motion_state = "idle"
        self._post_hit_until = time.perf_counter() + POST_HIT_COOLDOWN_SEC
        self._post_hit_need_clear = True

    def abort_shot(self) -> None:
        """Odbijen hit (npr. niska conf) — vrati detekciju u idle."""
        self._armed_frames = 0
        self._clear_frames = 0
        self._motion_state = "idle"
        self._post_hit_until = time.perf_counter() + POST_HIT_COOLDOWN_SEC
        self._post_hit_need_clear = True

    def save_refs_persistent(self) -> int:
        if not self._refs:
            return 0
        d = motion_refs_dir()
        os.makedirs(d, exist_ok=True)
        for cam_idx, gray in self._refs.items():
            _safe_imwrite(os.path.join(d, f"cam_{cam_idx}.png"), gray)
        for cam_idx, gray in self._raw_refs.items():
            _safe_imwrite(os.path.join(d, f"raw_cam_{cam_idx}.png"), gray)
        return len(self._refs)

    def load_raw_refs_persistent(self) -> int:
        d = motion_refs_dir()
        if not os.path.isdir(d):
            return 0
        loaded = 0
        for name in os.listdir(d):
            if not name.startswith("raw_cam_") or not name.endswith(".png"):
                continue
            try:
                cam_idx = int(name[8:-4])
            except ValueError:
                continue
            gray = cv2.imread(os.path.join(d, name), cv2.IMREAD_GRAYSCALE)
            if gray is not None and gray.size > 0:
                self._raw_refs[cam_idx] = gray
                loaded += 1
        return loaded

    def save_empty_refs_persistent(self, *, include_raw: bool = False) -> int:
        if not self._empty_refs:
            return 0
        d = motion_refs_dir()
        os.makedirs(d, exist_ok=True)
        for cam_idx, gray in self._empty_refs.items():
            _safe_imwrite(os.path.join(d, f"empty_cam_{cam_idx}.png"), gray)
        # Full-res raw PNG na svakom prebacivanju igrača na Pi-ju zna srušiti proces.
        if include_raw:
            for cam_idx, gray in self._empty_raw_refs.items():
                _safe_imwrite(os.path.join(d, f"empty_raw_cam_{cam_idx}.png"), gray)
        return len(self._empty_refs)

    def load_refs_persistent(self) -> int:
        d = motion_refs_dir()
        if not os.path.isdir(d):
            return 0
        loaded = 0
        for name in os.listdir(d):
            if name.startswith("empty_") or not name.startswith("cam_") or not name.endswith(".png"):
                continue
            try:
                cam_idx = int(name[4:-4])
            except ValueError:
                continue
            gray = cv2.imread(os.path.join(d, name), cv2.IMREAD_GRAYSCALE)
            if gray is not None and gray.size > 0:
                self._refs[cam_idx] = gray
                loaded += 1
        return loaded

    def load_empty_refs_persistent(self) -> int:
        d = motion_refs_dir()
        if not os.path.isdir(d):
            return 0
        loaded = 0
        for name in os.listdir(d):
            if name.startswith("empty_raw_cam_") and name.endswith(".png"):
                try:
                    cam_idx = int(name[14:-4])
                except ValueError:
                    continue
                gray = cv2.imread(os.path.join(d, name), cv2.IMREAD_GRAYSCALE)
                if gray is not None and gray.size > 0:
                    self._empty_raw_refs[cam_idx] = gray
                continue
            if not name.startswith("empty_cam_") or not name.endswith(".png"):
                continue
            try:
                cam_idx = int(name[10:-4])
            except ValueError:
                continue
            gray = cv2.imread(os.path.join(d, name), cv2.IMREAD_GRAYSCALE)
            if gray is not None and gray.size > 0:
                self._empty_refs[cam_idx] = gray
                loaded += 1
        return loaded

    def has_references(self) -> bool:
        return bool(self._refs)

    def has_empty_references(self) -> bool:
        return bool(self._empty_refs)

    def set_reference_topdown(self, cam_idx: int, topdown_bgr: np.ndarray) -> None:
        """Ažurira samo motion referencu (nikad praznu kalibracijsku)."""
        self._refs[cam_idx] = cv2.cvtColor(topdown_bgr, cv2.COLOR_BGR2GRAY).copy()
        if SAVE_MOTION_DEBUG and cam_idx >= 0:
            _ensure_debug_dir_exists()
            path = os.path.join(_debug_dir_path(), f"cam_{cam_idx}_motion_ref.png")
            cv2.imwrite(path, topdown_bgr)

    def set_raw_reference(self, cam_idx: int, frame_bgr: np.ndarray) -> None:
        self._raw_refs[cam_idx] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).copy()

    def capture_references(self, board_calibrator: BoardCalibrator, frames: Dict[int, Optional[np.ndarray]]) -> int:
        saved = 0
        for cam_idx, frame in frames.items():
            if frame is None:
                continue
            cal = board_calibrator.get(cam_idx)
            if cal is None or not cal.is_valid() or cal.double_ellipse is None:
                continue
            warped = _warp_frame(board_calibrator, cam_idx, frame, cal)
            if warped is None:
                continue
            self.set_reference_topdown(cam_idx, warped)
            self.set_raw_reference(cam_idx, frame)
            self._empty_refs[cam_idx] = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).copy()
            self._empty_raw_refs[cam_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).copy()
            saved += 1
        self.reset_motion_state()
        if saved > 0:
            self.save_refs_persistent()
            self.save_empty_refs_persistent(include_raw=True)
        return saved

    def update_empty_board_references(
        self,
        board_calibrator: BoardCalibrator,
        frames: Dict[int, Optional[np.ndarray]],
        *,
        include_raw: bool = False,
    ) -> int:
        """Azurira samo praznu referencu ploce (empty_cam_*.png), bez motion ref."""
        saved = 0
        for cam_idx, frame in frames.items():
            if frame is None:
                continue
            cal = board_calibrator.get(cam_idx)
            if cal is None or not cal.is_valid() or cal.double_ellipse is None:
                continue
            warped = _warp_frame(board_calibrator, cam_idx, frame, cal)
            if warped is None:
                continue
            self._empty_refs[cam_idx] = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).copy()
            self._empty_raw_refs[cam_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).copy()
            saved += 1
        if saved > 0:
            self.save_empty_refs_persistent(include_raw=include_raw)
        return saved

    def commit_board_snapshot(
        self,
        board_calibrator: BoardCalibrator,
        frames: Dict[int, Optional[np.ndarray]],
        *,
        persist: bool = True,
    ) -> int:
        """Samo motion/_refs: trenutna ploča (sa strijelama) za sljedeći diff.

        Nikad ne dira _empty_refs — prazna ploča ostaje iz kalibracije / clear.
        persist=False: samo RAM (hit path) — disk PNG usporava prikaz hita.
        """
        saved = 0
        for cam_idx, frame in frames.items():
            if frame is None:
                continue
            cal = board_calibrator.get(cam_idx)
            if cal is None or not cal.is_valid() or cal.double_ellipse is None:
                continue
            warped = _warp_frame(board_calibrator, cam_idx, frame, cal)
            if warped is None:
                continue
            self.set_reference_topdown(cam_idx, warped)
            self.set_raw_reference(cam_idx, frame)
            saved += 1
        if persist and saved > 0:
            self.save_refs_persistent()
        return saved

    def absorb_live_board_and_idle(
        self,
        board_calibrator: BoardCalibrator,
        frames: Dict[int, Optional[np.ndarray]],
    ) -> None:
        """Trenutna ploča = nova motion nula. Ništa od starog diffa ne smije postati hit."""
        self.reset_motion_state()
        self.commit_board_snapshot(board_calibrator, frames, persist=False)
        self.reset_motion_state()
        self._post_hit_until = time.perf_counter() + POST_HIT_COOLDOWN_SEC

    def commit_last_fire_snapshot(self, *, persist: bool = False) -> int:
        """Motion ref iz settle/fire cachea — bez novog warpa i (default) bez diska."""
        saved = 0
        for cam_idx, gray in self._settle_prev_warped.items():
            if gray is None or getattr(gray, "size", 0) <= 0:
                continue
            self._refs[cam_idx] = gray.copy()
            saved += 1
        for cam_idx, frame in self._fire_frames.items():
            if frame is None:
                continue
            self.set_raw_reference(cam_idx, frame)
            if saved == 0:
                saved += 1
        if persist and saved > 0:
            self.save_refs_persistent()
        return saved

    def mark_commit_after_ui(self) -> None:
        """Hit je već upisan — motion ref tek nakon što UI nacrta rezultat."""
        self._pending_ui_commit = True

    def flush_commit_after_ui(
        self,
        board_calibrator: Optional[BoardCalibrator] = None,
        frames: Optional[Dict[int, Optional[np.ndarray]]] = None,
    ) -> int:
        if not self._pending_ui_commit:
            return 0
        self._pending_ui_commit = False
        n = self.commit_last_fire_snapshot(persist=False)
        if n > 0 or board_calibrator is None:
            return n
        fallback = frames if frames else self.last_fire_frames()
        if not fallback:
            return 0
        return self.commit_board_snapshot(board_calibrator, fallback, persist=False)

    def sync_references_all(
        self,
        board_calibrator: BoardCalibrator,
        frames: Dict[int, Optional[np.ndarray]],
    ) -> int:
        saved = self.commit_board_snapshot(board_calibrator, frames)
        if saved > 0:
            self.reset_motion_state()
        return saved

    def update_references_after_hit(
        self,
        board_calibrator: BoardCalibrator,
        frames: Dict[int, Optional[np.ndarray]],
        cam_indices: List[int],
    ) -> None:
        """Nakon pogotka: samo motion/_refs (nova strijela u pozadini). Ne dira empty_refs."""
        for cam_idx in cam_indices:
            frame = frames.get(cam_idx)
            if frame is None:
                continue
            cal = board_calibrator.get(cam_idx)
            if cal is None or not cal.is_valid() or cal.double_ellipse is None:
                continue
            warped = _warp_frame(board_calibrator, cam_idx, frame, cal)
            if warped is not None:
                self.set_reference_topdown(cam_idx, warped)
            self.set_raw_reference(cam_idx, frame)

    def absorb_motion_from_results(
        self,
        board_calibrator: BoardCalibrator,
        frames: Dict[int, Optional[np.ndarray]],
        per_cam: List[DartDetectResult],
    ) -> int:
        """Strijelica (ili sum) postaje pozadina — sve kamere s motion px."""
        cam_indices = [r.cam_idx for r in per_cam if r.motion_pixels >= MOTION_MIN_PIXELS]
        if not cam_indices:
            cam_indices = [r.cam_idx for r in per_cam if r.found]
        if cam_indices:
            self.update_references_after_hit(board_calibrator, frames, cam_indices)
        return len(cam_indices)

    def _max_motion_pixels(self, per_cam: List[DartDetectResult]) -> int:
        return max((r.motion_pixels for r in per_cam), default=0)

    def detect_cam(
        self,
        cam_idx: int,
        frame: np.ndarray,
        cal: BoardCalibration,
        *,
        board_calibrator: Optional[BoardCalibrator] = None,
        save_debug: bool = SAVE_MOTION_DEBUG,
        fused_tip: Optional[Tuple[float, float]] = None,
    ) -> DartDetectResult:
        ref = self._refs.get(cam_idx)
        raw_ref = self._raw_refs.get(cam_idx)
        warped = _warp_frame(board_calibrator, cam_idx, frame, cal)
        warped_r = DartDetectResult(cam_idx=cam_idx, found=False, reject_reason="warp_fail")
        if warped is None:
            warped_r = DartDetectResult(cam_idx=cam_idx, found=False, reject_reason="warp_fail")
        elif ref is None:
            warped_r = DartDetectResult(cam_idx=cam_idx, found=False, reject_reason="no_ref")
        else:
            warped_r = detect_dart_on_topdown(
                warped,
                ref,
                cal,
                cam_idx=cam_idx,
                save_debug=save_debug,
                draw_fused_tip=fused_tip,
            )

        raw_r = DartDetectResult(cam_idx=cam_idx, found=False, reject_reason="no_raw_ref")
        warped_has_line = warped_r.found and warped_r.motion_pixels >= MOTION_MIN_PIXELS
        if raw_ref is not None and not warped_has_line:
            raw_r = detect_dart_offboard_raw(
                frame,
                raw_ref,
                cal,
                cam_idx=cam_idx,
                save_debug=save_debug,
            )
        return _merge_cam_detect_results(warped_r, raw_r)

    def detect_all(
        self,
        board_calibrator: BoardCalibrator,
        frames: Dict[int, Optional[np.ndarray]],
        *,
        save_debug: bool = SAVE_MOTION_DEBUG,
        fused_tip: Optional[Tuple[float, float]] = None,
    ) -> List[DartDetectResult]:
        out: List[DartDetectResult] = []
        for cam_idx, frame in frames.items():
            if frame is None:
                continue
            cal = board_calibrator.get(cam_idx)
            if cal is None or not cal.is_valid() or cal.double_ellipse is None:
                continue
            out.append(
                self.detect_cam(
                    cam_idx,
                    frame,
                    cal,
                    board_calibrator=board_calibrator,
                    save_debug=save_debug,
                    fused_tip=fused_tip,
                )
            )
        return out

    def probe_motion(
        self,
        board_calibrator: BoardCalibrator,
        frames: Dict[int, Optional[np.ndarray]],
        *,
        save_debug: bool = False,
    ) -> List[DartDetectResult]:
        """Samo motion px (bez fitline) za state machine."""
        self._tick_warped_gray.clear()
        out: List[DartDetectResult] = []
        for cam_idx, frame in frames.items():
            if frame is None:
                continue
            cal = board_calibrator.get(cam_idx)
            if cal is None or not cal.is_valid() or cal.double_ellipse is None:
                continue
            ref = self._refs.get(cam_idx)
            warped = _warp_frame(board_calibrator, cam_idx, frame, cal)
            if warped is None or ref is None:
                out.append(
                    DartDetectResult(
                        cam_idx=cam_idx,
                        found=False,
                        reject_reason="no_ref" if ref is None else "warp_fail",
                    )
                )
                continue
            h, w = warped.shape[:2]
            cx_out, view_r, _board_r = _warp_radii(w)
            cur_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            self._tick_warped_gray[cam_idx] = cur_gray
            ref_g = ref if ref.shape == cur_gray.shape else cv2.resize(
                ref, (w, h), interpolation=cv2.INTER_AREA
            )
            diff_mean = float(np.mean(cv2.absdiff(cur_gray, ref_g)))
            mask = _motion_mask(ref_g, cur_gray)
            board_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(board_mask, (int(cx_out), int(cx_out)), int(view_r), 255, -1)
            mask = cv2.bitwise_and(mask, mask, mask=board_mask)
            n_px = int(np.count_nonzero(mask))
            raw_px = 0
            raw_ref = self._raw_refs.get(cam_idx)
            if raw_ref is not None:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                raw_px, raw_diff = _count_raw_surround_motion(frame_gray, raw_ref, cal)
                diff_mean = max(diff_mean, raw_diff)
            n_px = max(n_px, raw_px)
            if save_debug:
                _save_motion_debug(cam_idx, mask, motion_pixels=n_px, diff_mean=diff_mean, status="probe")
            out.append(
                DartDetectResult(
                    cam_idx=cam_idx,
                    found=False,
                    motion_pixels=n_px,
                    diff_mean=diff_mean,
                )
            )
        return out

    def _probe_motion_vs_ref_map(
        self,
        board_calibrator: BoardCalibrator,
        frames: Dict[int, Optional[np.ndarray]],
        ref_map: Dict[int, np.ndarray],
    ) -> Tuple[List[DartDetectResult], int]:
        out: List[DartDetectResult] = []
        max_px = 0
        for cam_idx, frame in frames.items():
            if frame is None:
                continue
            cal = board_calibrator.get(cam_idx)
            if cal is None or not cal.is_valid() or cal.double_ellipse is None:
                continue
            ref = ref_map.get(cam_idx)
            warped = _warp_frame(board_calibrator, cam_idx, frame, cal)
            if warped is None or ref is None:
                continue
            h, w = warped.shape[:2]
            cx_out, view_r, _board_r = _warp_radii(w)
            cur_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            ref_g = ref if ref.shape == cur_gray.shape else cv2.resize(
                ref, (w, h), interpolation=cv2.INTER_AREA
            )
            mask = _motion_mask(ref_g, cur_gray)
            board_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(board_mask, (int(cx_out), int(cx_out)), int(view_r), 255, -1)
            mask = cv2.bitwise_and(mask, mask, mask=board_mask)
            n_px = int(np.count_nonzero(mask))
            if n_px > 0:
                diff_mean = float(np.mean(cv2.absdiff(cur_gray, ref_g)[mask > 0]))
            else:
                diff_mean = 0.0
            max_px = max(max_px, n_px)
            out.append(
                DartDetectResult(
                    cam_idx=cam_idx,
                    found=False,
                    motion_pixels=n_px,
                    diff_mean=diff_mean,
                )
            )
        return out, max_px

    def _probe_empty_raw_surround(
        self,
        board_calibrator: BoardCalibrator,
        frames: Dict[int, Optional[np.ndarray]],
    ) -> Tuple[int, float]:
        """Motion u surround zoni na ne-warpanoj slici vs prazna raw ref."""
        max_px = 0
        max_diff = 0.0
        for cam_idx, frame in frames.items():
            if frame is None:
                continue
            cal = board_calibrator.get(cam_idx)
            empty_raw = self._empty_raw_refs.get(cam_idx)
            if cal is None or not cal.is_valid() or cal.double_ellipse is None or empty_raw is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            n_px, diff_mean = _count_raw_surround_motion(gray, empty_raw, cal)
            max_px = max(max_px, n_px)
            max_diff = max(max_diff, diff_mean)
        return max_px, max_diff

    def probe_empty_board(
        self,
        board_calibrator: BoardCalibrator,
        frames: Dict[int, Optional[np.ndarray]],
    ) -> Tuple[int, float]:
        """Motion px i diff_mean vs prazna ref (top-down + raw surround)."""
        if not self._empty_refs:
            self._last_empty_td_px = 999999
            self._last_empty_raw_px = 999999
            self._last_empty_td_diff = 999999.0
            return 999999, 999999.0
        _results, td_px = self._probe_motion_vs_ref_map(
            board_calibrator, frames, self._empty_refs
        )
        topdown_diff = max((r.diff_mean for r in _results), default=0.0)
        raw_px, raw_diff = self._probe_empty_raw_surround(board_calibrator, frames)
        self._last_empty_td_px = int(td_px)
        self._last_empty_raw_px = int(raw_px)
        self._last_empty_td_diff = float(topdown_diff)
        # diff prag samo top-down; raw oko ploce samo motion px (ruka u kadru)
        return max(td_px, raw_px), max(topdown_diff, raw_diff if raw_px > 0 else 0.0)

    def is_board_like_empty(
        self,
        board_calibrator: BoardCalibrator,
        frames: Dict[int, Optional[np.ndarray]],
    ) -> Tuple[bool, int, float]:
        max_px, max_diff = self.probe_empty_board(board_calibrator, frames)
        td_px = int(self._last_empty_td_px)
        raw_px = int(self._last_empty_raw_px)
        td_diff = float(self._last_empty_td_diff)
        ok = td_px <= EMPTY_BOARD_MAX_PIXELS
        if td_px >= EMPTY_BOARD_DIFF_APPLY_MIN_PX:
            ok = ok and td_diff <= EMPTY_BOARD_MAX_DIFF_MEAN
        # Ruka koja vadi strijelice: raw surround skoči na tisuće (LED šum je ~800).
        if raw_px >= EMPTY_BOARD_RAW_HAND_PIXELS:
            ok = False
        return ok, max_px, max_diff

    def _cache_settle_frame(
        self,
        board_calibrator: BoardCalibrator,
        frames: Dict[int, Optional[np.ndarray]],
        *,
        keep_bgr: bool = False,
    ) -> None:
        for cam_idx, frame in frames.items():
            if frame is None:
                continue
            cal = board_calibrator.get(cam_idx)
            if cal is None or not cal.is_valid():
                continue
            self._settle_prev_gray[cam_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if keep_bgr:
                self._fire_frames[cam_idx] = frame.copy()
            if cal.double_ellipse is not None:
                tick_g = self._tick_warped_gray.get(cam_idx)
                if tick_g is not None:
                    self._settle_prev_warped[cam_idx] = tick_g
                else:
                    warped = _warp_frame(board_calibrator, cam_idx, frame, cal)
                    if warped is not None:
                        self._settle_prev_warped[cam_idx] = cv2.cvtColor(
                            warped, cv2.COLOR_BGR2GRAY
                        )

    def _note_hand_activity(self, strength: float = 1.0) -> None:
        extra = int(HAND_POST_ACTIVITY_COOLDOWN * max(0.5, min(2.0, strength)))
        self._hand_cooldown_frames = max(self._hand_cooldown_frames, extra)

    def is_hand_cooldown_active(self) -> bool:
        return self._hand_cooldown_frames > 0

    def _hand_votes_and_max(
        self,
        board_calibrator: BoardCalibrator,
        frames: Dict[int, Optional[np.ndarray]],
    ) -> Tuple[int, float]:
        if not self._settle_prev_gray:
            return 0, 0.0
        votes = 0
        max_score = 0.0
        for cam_idx, frame in frames.items():
            if frame is None:
                continue
            cal = board_calibrator.get(cam_idx)
            if cal is None or not cal.is_valid():
                continue
            prev_gray = self._settle_prev_gray.get(cam_idx)
            if prev_gray is None:
                continue
            cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cam_score = 0.0

            raw_ref = self._raw_refs.get(cam_idx)
            if raw_ref is not None:
                cam_score = max(cam_score, _hand_motion_score(raw_ref, prev_gray, cur_gray))

            # Warp path samo ako raw nije već jasan — inače dupli motion_mask + remap.
            if cam_score < HAND_STRONG_SINGLE_CAM_SCORE:
                prev_warped = self._settle_prev_warped.get(cam_idx)
                warp_ref = self._refs.get(cam_idx)
                if prev_warped is not None and warp_ref is not None:
                    cur_warped = self._tick_warped_gray.get(cam_idx)
                    if cur_warped is None:
                        warped = _warp_frame(board_calibrator, cam_idx, frame, cal)
                        if warped is not None:
                            cur_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                    if cur_warped is not None:
                        cam_score = max(
                            cam_score,
                            _hand_motion_score(warp_ref, prev_warped, cur_warped),
                        )

            max_score = max(max_score, cam_score)
            if cam_score >= 1.5:
                votes += 1
        return votes, max_score

    def _any_hand_in_frames(
        self,
        board_calibrator: BoardCalibrator,
        frames: Dict[int, Optional[np.ndarray]],
        *,
        strict: bool = False,
    ) -> bool:
        votes, max_score = self._hand_votes_and_max(board_calibrator, frames)
        if max_score >= HAND_WEAK_SCORE:
            self._note_hand_activity(max(0.6, max_score / HAND_SCORE_THRESHOLD))
        if strict and max_score >= HAND_WEAK_SCORE:
            return True
        if max_score >= HAND_STRONG_SINGLE_CAM_SCORE:
            return True
        if votes >= HAND_CAM_VOTES_REQUIRED and max_score >= HAND_SCORE_THRESHOLD:
            return True
        return False

    def _enter_hand_block(self) -> None:
        self._motion_state = "hand_block"
        self._armed_frames = 0
        self._hand_stable_streak = 0
        self._clear_frames = 0

    def _exit_hand_block(self) -> None:
        self._motion_state = "idle"
        self._clear_frames = 0
        self._hand_cooldown_frames = 0
        self._armed_frames = 0
        self._hand_stable_streak = 0

    def last_fire_frames(self) -> Dict[int, np.ndarray]:
        """BGR kadrovi s kojima je napravljen zadnji fuse (za commit ref)."""
        return {k: v for k, v in self._fire_frames.items()}

    def detect_hand_activity(
        self,
        board_calibrator: BoardCalibrator,
        frames: Dict[int, Optional[np.ndarray]],
        *,
        strict: bool = False,
    ) -> bool:
        return self._any_hand_in_frames(board_calibrator, frames, strict=strict)

    def force_hand_block(self) -> None:
        self._enter_hand_block()

    def tick_and_detect(
        self,
        board_calibrator: BoardCalibrator,
        frames: Dict[int, Optional[np.ndarray]],
        *,
        save_debug: bool = SAVE_MOTION_DEBUG,
        accept_fire: bool = True,
    ) -> Tuple[List[DartDetectResult], Optional[FusedDartResult], str]:
        """
        State machine:
        idle -> armed -> +delay -> fuse -> (main finalize_shot)
          inace: idle (nastavi detekciju, i nakon MISS/surround)
          3/3 + turn submit: rearm -> idle (cekaj praznu plocu)

        accept_fire=False: motion se prati, ali se ne izvrsava fuse (npr. 3/3 puna).
        """
        probe = self.probe_motion(board_calibrator, frames, save_debug=False)
        max_px = self._max_motion_pixels(probe)
        state = self._motion_state
        if self._hand_cooldown_frames > 0:
            self._hand_cooldown_frames -= 1

        if state == "post_fire":
            return probe, None, "post_fire"

        if state == "hand_block":
            hand_now = self._any_hand_in_frames(board_calibrator, frames)
            if hand_now:
                self._clear_frames = 0
            else:
                self._clear_frames += 1
                if self._clear_frames >= HAND_BLOCK_CLEAR_FRAMES:
                    # Ako strijela i dalje stoji (motion), odmah natrag u armed —
                    # inace idle+cooldown petlja propušta bacanja i gomila blobove.
                    if max_px >= MOTION_ARM_PIXELS:
                        self._motion_state = "armed"
                        self._armed_frames = 1
                        self._hand_stable_streak = 0
                        self._clear_frames = 0
                        self._hand_cooldown_frames = 0
                        self._cache_settle_frame(board_calibrator, frames, keep_bgr=True)
                        return probe, None, "armed_after_hand"
                    self._exit_hand_block()
            self._cache_settle_frame(board_calibrator, frames, keep_bgr=False)
            return probe, None, "hand_block"

        if state == "rearm":
            # Rearm ugašen — odmah idle, inače ostane zaglavljen dok px ne padne ispod 12.
            self._motion_state = "idle"
            self._clear_frames = 0
            state = "idle"

        if state == "idle":
            if time.perf_counter() < self._post_hit_until:
                self._cache_settle_frame(board_calibrator, frames, keep_bgr=False)
                return probe, None, "idle"
            if self._post_hit_need_clear:
                # Leftover dart/vibration after a scored hit: eat it, do not fuse
                # (radial leftover on 1 cam projects onto bull).
                if max_px >= MOTION_ARM_PIXELS:
                    self.commit_board_snapshot(board_calibrator, frames, persist=False)
                self._post_hit_need_clear = False
                self._cache_settle_frame(board_calibrator, frames, keep_bgr=False)
                return probe, None, "idle"
            if (
                not self.is_hand_cooldown_active()
                and max_px >= MOTION_ARM_PIXELS
            ):
                self._motion_state = "armed"
                self._armed_frames = 1
                self._hand_stable_streak = 0
            self._cache_settle_frame(board_calibrator, frames, keep_bgr=False)
            return probe, None, "idle"

        # armed — settle + stabilnost (ignoriraj ruku)
        if max_px < MOTION_ARM_PIXELS:
            self._motion_state = "idle"
            self._armed_frames = 0
            self._hand_stable_streak = 0
            self._settle_prev_gray.clear()
            self._settle_prev_warped.clear()
            self._fire_frames.clear()
            return probe, None, "idle"

        self._armed_frames += 1

        if self._armed_frames <= SETTLE_FRAME_DELAY:
            self._cache_settle_frame(board_calibrator, frames, keep_bgr=True)
            return probe, None, f"armed_{self._armed_frames}/{SETTLE_FRAME_DELAY}"

        hand_now = self._any_hand_in_frames(board_calibrator, frames)
        if hand_now:
            self._enter_hand_block()
            self._cache_settle_frame(board_calibrator, frames, keep_bgr=False)
            return probe, None, "armed_hand_reject"

        self._hand_stable_streak += 1
        self._cache_settle_frame(board_calibrator, frames, keep_bgr=True)
        if self._hand_stable_streak < HAND_STABILITY_FRAMES:
            return probe, None, f"armed_stable_{self._hand_stable_streak}/{HAND_STABILITY_FRAMES}"

        if not accept_fire:
            if max_px < MOTION_ARM_PIXELS:
                self._motion_state = "idle"
                self._armed_frames = 0
                return probe, None, "armed_abort"
            return probe, None, "armed_hold"

        if self.is_hand_cooldown_active():
            self._cache_settle_frame(board_calibrator, frames, keep_bgr=False)
            return probe, None, "hand_cooldown"

        # Detekcija na zadnjem stabilnom kadru (settle cache), ne na svjezem USB bufferu.
        fire_frames = self._fire_frames if len(self._fire_frames) >= 2 else frames
        # Soft-threshold FitLine po kameri (downscaled) → tip = sjecište u ds, map natrag.
        per_cam = self.detect_all(board_calibrator, fire_frames, save_debug=False)
        fused = fuse_multicam_lines(per_cam, board_calibrator=board_calibrator)
        fused.per_cam = per_cam
        fused = _apply_ml_tip_cached(fused, per_cam, board_calibrator)
        # Debug / raw-shaft shadow nakon UI-ja — ne smiju blokirati prikaz hita.
        if save_debug and fused.found and _hit_motion_debug_enabled():
            save_hit_fusion_debug(
                fused,
                self._empty_refs,
                raw_frames=fire_frames,
                board_calibrator=board_calibrator,
            )

        self._motion_state = "post_fire"
        self._armed_frames = 0
        self._clear_frames = 0
        if fused.found:
            tag = "fired_miss" if fused.zone_name == "miss" or fused.score <= 0 else "fired"
            return probe, fused, tag
        if fused.reject_reason:
            print(f"[dart] fusion reject: {fused.reject_reason}", flush=True)
        return probe, None, "fired_fail"

    def best_hit(self, results: List[DartDetectResult]) -> Optional[DartDetectResult]:
        """Legacy: pojedinacna kamera."""
        hits = [r for r in results if r.found and r.score > 0]
        if not hits:
            return None
        return max(hits, key=lambda r: r.confidence)


def _fitline_smoke_test() -> None:
    """Synthetic diagonal streak must get ~45° FitLine through streak center (not image center)."""
    h = w = 400
    img = np.zeros((h, w), dtype=np.uint8)
    # Offset 45° streak: (30,150)→(150,270) — bright middle far from board center.
    for t in range(0, 120):
        cv2.circle(img, (30 + t, 150 + t), 4, 220, -1)
    # Board-wide faint noise (the bug that pulled old fits to ~100,100).
    yy, xx = np.ogrid[:h, :w]
    board = (xx - 200) ** 2 + (yy - 200) ** 2 <= 180 ** 2
    rng = np.random.default_rng(0)
    noise = rng.integers(3, 10, size=(h, w), dtype=np.uint8)
    noisy = img.copy()
    noisy[board] = np.maximum(noisy[board], noise[board])

    # Gate like production: morph-ish ROI around the bright streak.
    _, gate = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gate = cv2.dilate(gate, k, iterations=1)
    gated = cv2.bitwise_and(noisy, noisy, mask=gate)

    ok, (vx, vy, x0, y0), ds, _sx, _sy, xs, ys = _weighted_fitline_on_soft_motion(
        gated, board_center=(200.0, 200.0)
    )
    assert ok, "FitLine smoke: fit failed"
    assert abs(abs(vx) - abs(vy)) < 0.15, f"FitLine smoke: not ~45° dir vx={vx} vy={vy}"
    wts = ds[ys.astype(np.intp), xs.astype(np.intp)].astype(np.float64)
    cx = float((xs * wts).sum() / max(float(wts.sum()), 1e-9))
    cy = float((ys * wts).sum() / max(float(wts.sum()), 1e-9))
    # Origin must sit on the streak centroid, not FitLine image center.
    ds_cx = 0.5 * float(FITLINE_SIZE)
    assert np.hypot(x0 - cx, y0 - cy) < 4.0, (
        f"FitLine smoke: origin ({x0:.1f},{y0:.1f}) != centroid ({cx:.1f},{cy:.1f})"
    )
    assert np.hypot(x0 - ds_cx, y0 - ds_cx) > 20.0, (
        f"FitLine smoke: origin stuck at image center ({x0:.1f},{y0:.1f})"
    )
    perp = abs((cx - x0) * (-vy) + (cy - y0) * vx)
    assert perp < 2.0, f"FitLine smoke: line misses centroid perp={perp:.2f}"
    print(
        f"[fitline smoke] OK vx={vx:.3f} vy={vy:.3f} origin=({x0:.1f},{y0:.1f}) "
        f"centroid=({cx:.1f},{cy:.1f}) angle={_fitline_angle_deg(vx, vy):.1f}",
        flush=True,
    )


def _consensus_fuse_smoke_test() -> None:
    """Two good lines + one fleeing line → tip must be the good-pair intersection."""
    # True tip at (150, 150). Cams 0/2 pass through it; cam 4 is rotated away.
    true_tip = (150.0, 150.0)
    cams = [
        DartDetectResult(
            cam_idx=0,
            found=True,
            tip_xy=true_tip,
            line_x0=150.0,
            line_y0=150.0,
            line_vx=1.0,
            line_vy=0.0,
            confidence=0.9,
            motion_pixels=80,
        ),
        DartDetectResult(
            cam_idx=2,
            found=True,
            tip_xy=true_tip,
            line_x0=150.0,
            line_y0=150.0,
            line_vx=0.0,
            line_vy=1.0,
            confidence=0.5,  # lower conf than the bad cam
            motion_pixels=40,
        ),
        DartDetectResult(
            cam_idx=4,
            found=True,
            tip_xy=(200.0, 100.0),
            # Parallel-ish miss: vertical line shifted 40px — flees true tip.
            line_x0=190.0,
            line_y0=100.0,
            line_vx=0.0,
            line_vy=1.0,
            confidence=0.95,  # high blob conf must not dominate
            motion_pixels=200,
        ),
    ]
    tip, _res, kept = _fuse_tip_from_best_pair(cams, board_center=(150.0, 150.0))
    assert tip is not None, "consensus smoke: no tip"
    err = float(np.hypot(tip[0] - true_tip[0], tip[1] - true_tip[1]))
    assert err < 2.0, f"consensus smoke: tip={tip} err={err:.2f} (bad line dominated)"
    kept_ids = {int(r.cam_idx) for r in kept}
    assert 4 not in kept_ids, f"consensus smoke: flee cam kept {kept_ids}"
    assert kept_ids == {0, 2}, f"consensus smoke: unexpected kept {kept_ids}"
    print(
        f"[consensus smoke] OK tip=({tip[0]:.1f},{tip[1]:.1f}) kept={sorted(kept_ids)}",
        flush=True,
    )


if __name__ == "__main__":
    _fitline_smoke_test()
    _consensus_fuse_smoke_test()
