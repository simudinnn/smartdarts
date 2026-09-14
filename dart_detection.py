"""Motion diff + intensity-weighted FitLine detekcija strijelice na top-down warp slici (3 kamere)."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import distance_yfit as _dyfit

from board_calibration import (
    BOARD_SEGMENT_NUMBERS,
    BOARD_SISAL_EDGE_FRAC,
    RING_BULL_INNER_FRAC,
    RING_BULL_OUTER_FRAC,
    SEGMENT_COUNT,
    BoardCalibration,
    BoardCalibrator,
    warp_topdown_from_calibration,
    topdown_point_to_score,
    scoring_ring_radii,
    scoring_ring_ellipses,
    calibrated_topdown_wire_thetas,
    canonical_topdown_wire_thetas,
    normalized_ellipse_radius,
    _ang_abs_delta_deg,
    _angle_sin_cos,
    _board_ellipse_mask,
    _debug_dir_path,
    _ensure_debug_dir_exists,
    _mean_ellipse_semi,
    _ray_to_ellipse_edge_f,
    _scale_ellipse,
    _scale_ellipse_axes,
    _segment_slot_from_theta,
    _segment_slot_from_wires,
    _warp_radii,
)


MOTION_DIFF_THRESH = 25
MOTION_MIN_PIXELS = 20
MOTION_MAX_PIXELS = 12000
# Scattered LED/empty-ref specks can sum past MOTION_MIN_PIXELS. Reject
# only a *field* of tiny CCs — a thin dart shaft may be 12–40 px after morph.
MOTION_MIN_BLOB_AREA = 12
MOTION_SPECKLE_MIN_CCS = 8
MOTION_BLUR_KSIZE = 3
MOTION_LOG_MIN_PIXELS = 6
# Idle LED/USB flicker is often ~8–20 px; a real dart is far above this.
MOTION_ARM_PIXELS = 28
MOTION_CLEAR_PIXELS = 12
SETTLE_FRAME_DELAY = 2
# FitLine (2-stage): soft grayscale motion gated by dilated morph ROI →
# fixed FITLINE_SIZE → mild Gaussian blur → radial flight downweight →
# intensity-weighted PCA (stage1) → multicam tip fuse → stage2 FitLine on
# soft pixels inside a circle around that tip → fuse again → map back.
# Morph is a soft pixel gate only (no CC / shaft / flight blob return).
# Detection warp + FitLine ds space (preview stays at DEBUG_WARP_SIZE=200).
FITLINE_SIZE = 300
FITLINE_DS_SIZE = FITLINE_SIZE
FITLINE_BLUR_KSIZE = 5
FITLINE_BLUR_SIGMA = 2.0
# Ignore near-black for numerical stability (weights stay continuous above this).
FITLINE_WEIGHT_EPS = 2.0
FITLINE_MIN_PX = 8
# Dilate morph gate so soft FitLine keeps a margin around the motion streak.
FITLINE_GATE_DILATE = 5
# Experimental joint Y detector (no FitLine seed). "fitline" keeps production path.
# "hybrid" = FitLine seed + local Y-refine.
DART_DETECTOR_MODE = os.environ.get("SMARTDARTS_DETECTOR", "hybrid").strip().lower()
if DART_DETECTOR_MODE not in ("fitline", "yfit", "hybrid"):
    DART_DETECTOR_MODE = "hybrid"
# Experimental warpPolar distance-Y (distance_yfit_warppolar_final_v2).
# Default off: hybrid FitLine/Y-refine stays production.
# SMARTDARTS_TIP_ESTIMATOR=distance_yfit  (alias: SMARTDARTS_TIP_MODE)
_raw_tip = (
    os.environ.get("SMARTDARTS_TIP_ESTIMATOR")
    or os.environ.get("SMARTDARTS_TIP_MODE")
    or "off"
).strip().lower()
if _raw_tip in ("", "none", "hybrid", "fitline", "yfit", "off"):
    DART_TIP_ESTIMATOR = "off"
elif _raw_tip in ("distance_yfit", "distance_yfit_standalone"):
    DART_TIP_ESTIMATOR = "distance_yfit"
else:
    DART_TIP_ESTIMATOR = "off"
DART_TIP_MODE = DART_TIP_ESTIMATOR
DYFIT_2CAM_CONF_SCALE = 0.72
# Y-FIT knobs (board px @ FITLINE_SIZE). Easy to retune.
YFIT_COARSE_XY_STEP = 4.0
YFIT_FINE_XY_STEP = 1.0
YFIT_COARSE_ANGLE_DEG = 6.0
YFIT_FINE_ANGLE_DEG = 1.0
YFIT_SEARCH_MARGIN_PX = 15.0  # P search bbox = fused motion + this margin
YFIT_RAY_LENGTH_FRAC = 0.50  # of board_r; matches Stage2 ROI radius
YFIT_CORRIDOR_PX = 3.0  # Gaussian sigma for distance-weighted support
YFIT_DIST_CUTOFF_SIGMA = 3.0  # ignore |v| > this * sigma (not a binary corridor)
YFIT_BALANCE_PENALTY = 0.15  # mild left/right mass imbalance penalty
YFIT_BIN_TAU = 0.80  # per-bin mass saturation (coverage, not raw intensity)
YFIT_FINE_XY_RADIUS = 4.0  # around best coarse P (covers 4px coarse cell)
YFIT_FINE_ANGLE_SPAN_DEG = 6.0  # around best coarse theta (covers 6° coarse cell)
YFIT_U_BINS = 12
YFIT_MIN_TOTAL = 0.35
YFIT_P_CHUNK = 8192
# 1° ray/corridor templates, keyed by (ray_len, corridor, n_bins, cutoff).
_YFIT_RAY_BANK: Dict[Tuple, dict] = {}
# Local Y-refine around FitLine seed (hybrid). Search windows in FITLINE_SIZE px / deg.
# Adaptive P/angle from pair_spread (tight hits stay small/fast).
YREFINE_SPREAD_TIGHT = 3.0
YREFINE_SPREAD_WIDE = 8.0
YREFINE_P_RADIUS_TIGHT = 5.0
YREFINE_P_RADIUS_MID = 9.0
YREFINE_P_RADIUS_WIDE = 18.0
YREFINE_P_RADIUS_EXPAND_CAP = 20.0
YREFINE_MAX_DP_TIGHT = 6.0
YREFINE_MAX_DP_MID = 12.0
YREFINE_MAX_DP_WIDE = 20.0
YREFINE_ANG_TIGHT = 5.0
YREFINE_ANG_MID_GOOD = 7.0
YREFINE_ANG_MID_BAD = 10.0
YREFINE_ANG_WIDE_GOOD = 8.0
YREFINE_ANG_WIDE_BAD = 15.0
YREFINE_COARSE_XY_STEP = 2.0
YREFINE_FINE_XY_STEP = 1.0
YREFINE_FINE_XY_RADIUS = 2.0
YREFINE_COARSE_ANG_STEP = 2.0
YREFINE_FINE_ANG_STEP = 1.0
YREFINE_FINE_ANG_SPAN = 2.0
YREFINE_LOW_Q = 0.34
YREFINE_S1S2_JUMP_DEG = 12.0
YREFINE_MIN_CAM_NORMAL = 0.28
YREFINE_MIN_CAM_SUSP = 0.18
YREFINE_SCORE_EPS_NORMAL = 0.005
YREFINE_SCORE_EPS_SUSP = 0.002
YREFINE_ABSURD_ANG_DEG = 22.0
# Per-cam ray: keep Gaussian corridor; weight tightness (center) over coverage.
YREFINE_COV_W = 0.40
YREFINE_TIGHT_W = 0.60
YREFINE_OBJECTIVE = "consensus"  # 0.5*mean + 0.5*min; see _yrefine_objectives
YREFINE_ALIVE_SUPPORT = 0.08  # ignore dead cams in min/geom/consensus
# Hybrid: Y-refine only when a FitLine misses the fused tip (poor concurrence).
YREFINE_FLEE_PX = 2.0
# Motion pixels Y-refine may use: bright dart core + small halo.
# Dim specks must be zeroed *before* CC or they 8-connect onto the dart
# and ray coverage walks the tip along board wires.
YREFINE_NOISE_PCTL = 90.0
YREFINE_NOISE_CORE_FRAC = 0.32
YREFINE_NOISE_CORE_MIN_AREA = 6
YREFINE_NOISE_DILATE_PX = 2
# Bright satellite beside the dart (not along the shaft) rotates the Y ray.
YREFINE_SATELLITE_PERP_PX = 4.5
YREFINE_SATELLITE_AREA_FRAC = 0.22
YREFINE_SATELLITE_ELONG_RATIO = 1.6


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
# Centerline bin-refit is debug-only and must not change the production line.
DEBUG_CENTERLINE_REFIT = False
CENTERLINE_N_BINS = 14
CENTERLINE_MIN_BINS = 5
CENTERLINE_MIN_BIN_PX = 3
CENTERLINE_MIN_LENGTH_PX = 10.0
CENTERLINE_RES_WORSE_FRAC = 1.35
CENTERLINE_RES_WORSE_PX = 0.35
# Stage2 shaft component selection (CC). Not ranked by area.
SHAFT_CC_MIN_AREA = 8
SHAFT_MIN_ELONG = 0.28
SHAFT_MIN_LENGTH_PX = 8.0
SHAFT_MAX_THICK_PX = 12.0
SHAFT_TIP_DIST_FRAC = 0.62
SHAFT_MIN_SCORE = 0.22
SHAFT_MERGE_ANGLE_DEG = 18.0
SHAFT_MERGE_COLLINEAR_PX = 4.5
SHAFT_MERGE_GAP_PX = 16.0
SHAFT_MERGE_TIP_PERP_PX = 10.0
# FitLine quality (always) + conservative recovery (suspicious hits only).
# Quality describes geometric line shape, not blob size / motion energy.
FITQ_CORRIDOR_PX = 3.0
FITQ_RESIDUAL_REF_PX = 4.0
FITQ_THICK_REF_PX = 5.5
FITQ_LENGTH_REF_FRAC = 0.40
FITQ_WEAK_PIXEL_REF = 400.0
FITQ_ANGLE_PEN_DEG = 28.0
# Normal 3-cam hits usually have pair_spread of a few px; keep this high so
# ordinary darts never enter recovery.
FITQ_PAIR_SPREAD_MIN_PX = 18.0
FITQ_PAIR_SPREAD_FRAC = 0.12
FITQ_WIDER_ROI_SCALE = 1.40
FITQ_IRLS_SIGMA_PX = 1.8
FITQ_MIN_SPREAD_IMPROVE = 0.85
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
POST_HIT_COOLDOWN_SEC = 0.28
# Leftover shaft after a real hit is radial; 1-cam fuse used to project to bull.
BULL_MIN_CAMS = 2
# A 2D tip needs two lines with a real crossing. One FitLine through the
# board center turns a miss-flight into a fake on-board hit; two nearly
# parallel lines slide the intersection along the dart (S4 vs S13).
FUSE_MIN_CAMS = 2
FUSE_MIN_CROSS_2CAM = 0.18  # |sin θ| ≈ 10.4°
FITLINE_PAIR_CROSS_MIN = 0.08  # 3-cam pairwise skip (unchanged)
BULL_MOTION_MAX_BULL_OUTER = 2.2
SAVE_MOTION_DEBUG = False
# Hit debug: fused_motion_raw (blobs) + scoring-canvas FitLine / Y-refine overlays.
SAVE_HIT_FUSION_DEBUG = True
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
    # Per-cam tip: projekcija fused tipa na liniju (2+ cam).
    # line_* are in FITLINE_SIZE coordinates (same space as fit_ds_gray).
    radius_tip: Tuple[float, float] = (0.0, 0.0)
    # Stage-1 line snapshot (before Stage2 ROI overwrite).
    stage1_vx: float = 0.0
    stage1_vy: float = 0.0
    stage1_x0: float = 0.0
    stage1_y0: float = 0.0
    # Geometric FitLine quality (not blob-size confidence).
    fitq_pixels: int = 0
    fitq_support: float = 0.0
    fitq_residual: float = 0.0
    fitq_linearity: float = 0.0
    fitq_length: float = 0.0
    fitq_thickness: float = 0.0
    fitq_s1s2_angle: float = 0.0
    fitq_s1s2_tip: float = 0.0
    fitq_quality: float = 0.0
    # Stage2 centerline-bin points (DEBUG_CENTERLINE_REFIT only; not used for line).
    centerline_pts_ds: List[Tuple[float, float]] = field(default_factory=list)
    centerline_old_ds: Optional[Tuple[float, float, float, float]] = None
    centerline_cand_ds: Optional[Tuple[float, float, float, float]] = None
    centerline_accepted: bool = False
    centerline_reject_reason: str = ""
    # Stage2 shaft CC debug (ds coords).
    shaft_sel_xs: Optional[np.ndarray] = None
    shaft_sel_ys: Optional[np.ndarray] = None
    shaft_rej_xs: Optional[np.ndarray] = None
    shaft_rej_ys: Optional[np.ndarray] = None
    yfit_support: float = 0.0


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
    fitq_pair_spread: float = 0.0
    fitq_suspicious: bool = False
    fitq_recovery_cam: int = -1
    yfit_total: float = 0.0
    yfit_ms: float = 0.0
    yrefine_p0: Tuple[float, float] = (0.0, 0.0)
    yrefine_p: Tuple[float, float] = (0.0, 0.0)
    yrefine_accepted: bool = False
    yrefine_mode: str = ""
    yrefine_seed_dir: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    yrefine_dir: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    yrefine_scores: Dict[int, float] = field(default_factory=dict)
    yrefine_search_radius: float = 0.0
    yrefine_on_edge: bool = False
    yrefine_expanded: bool = False


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


def _pca_geom_xy(
    xs: np.ndarray, ys: np.ndarray
) -> Tuple[float, float, float, float, float, float, float, float]:
    """Equal-weight PCA geometry: vx,vy,cx,cy,elong,length,thickness,n."""
    xs_f = xs.astype(np.float64)
    ys_f = ys.astype(np.float64)
    n = int(xs_f.size)
    if n < 2:
        cx = float(xs_f[0]) if n else 0.0
        cy = float(ys_f[0]) if n else 0.0
        return 1.0, 0.0, cx, cy, 0.0, 0.0, 0.0, float(n)
    cx = float(xs_f.mean())
    cy = float(ys_f.mean())
    dx = xs_f - cx
    dy = ys_f - cy
    cov = np.array(
        [
            [float((dx * dx).mean()), float((dx * dy).mean())],
            [float((dx * dy).mean()), float((dy * dy).mean())],
        ],
        dtype=np.float64,
    )
    eigvals, eigvecs = np.linalg.eigh(cov)
    l2, l1 = float(eigvals[0]), float(eigvals[1])
    vx = float(eigvecs[0, -1])
    vy = float(eigvecs[1, -1])
    nn = float(np.hypot(vx, vy)) or 1.0
    vx, vy = vx / nn, vy / nn
    elong = float((l1 - l2) / (l1 + l2 + 1e-9))
    par = dx * vx + dy * vy
    perp = dx * (-vy) + dy * vx
    length = float(np.max(par) - np.min(par)) if n else 0.0
    thickness = float(2.0 * np.mean(np.abs(perp))) if n else 0.0
    return vx, vy, cx, cy, elong, length, thickness, float(n)


def _shaft_component_score(
    *,
    dmin: float,
    radius: float,
    elong: float,
    length: float,
    thickness: float,
) -> float:
    """Prefer near-tip, long, thin structure. Area/pixel count is not the rank."""
    r = max(float(radius), 1.0)
    tip_prox = float(np.clip(1.0 - float(dmin) / r, 0.0, 1.0))
    length_n = float(np.clip(float(length) / max(0.35 * r, 8.0), 0.0, 1.0))
    thick_n = float(np.clip(float(thickness) / float(SHAFT_MAX_THICK_PX), 0.0, 1.0))
    elong_n = float(np.clip(float(elong), 0.0, 1.0))
    return 0.40 * tip_prox + 0.30 * elong_n + 0.25 * length_n - 0.25 * thick_n


def _comps_mergeable(
    a: dict,
    b: dict,
    tip_xy: Tuple[float, float],
) -> bool:
    ang = _dir_angle_delta_deg(a["vx"], a["vy"], b["vx"], b["vy"])
    if ang > float(SHAFT_MERGE_ANGLE_DEG):
        return False
    d_cent = _point_to_line_distance(
        a["cx"], a["cy"], a["vx"], a["vy"], b["cx"], b["cy"]
    )
    if d_cent > float(SHAFT_MERGE_COLLINEAR_PX):
        return False
    ua = (a["xs"] - a["cx"]) * a["vx"] + (a["ys"] - a["cy"]) * a["vy"]
    ub = (b["xs"] - a["cx"]) * a["vx"] + (b["ys"] - a["cy"]) * a["vy"]
    a0, a1 = float(np.min(ua)), float(np.max(ua))
    b0, b1 = float(np.min(ub)), float(np.max(ub))
    if a1 < b0:
        gap = b0 - a1
    elif b1 < a0:
        gap = a0 - b1
    else:
        gap = 0.0
    if gap > float(SHAFT_MERGE_GAP_PX):
        return False
    tip_d = _point_to_line_distance(
        a["cx"], a["cy"], a["vx"], a["vy"], float(tip_xy[0]), float(tip_xy[1])
    )
    tip_db = _point_to_line_distance(
        b["cx"], b["cy"], b["vx"], b["vy"], float(tip_xy[0]), float(tip_xy[1])
    )
    if tip_d > float(SHAFT_MERGE_TIP_PERP_PX) and tip_db > float(SHAFT_MERGE_TIP_PERP_PX):
        return False
    return True


def _select_shaft_components_stage2(
    ds_soft: np.ndarray,
    tip_xy_ds: Tuple[float, float],
    radius_ds: float,
    board_center_ds: Tuple[float, float],
    *,
    cam_idx: int = -1,
) -> Tuple[
    Optional[Tuple[float, float, float, float]],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """Pick elongated near-tip CC in Stage2 ROI; fallback None → original Stage2."""
    empty = (None, None, None, None, None)
    if ds_soft is None or ds_soft.size == 0:
        print(
            f"[SHAFT_SELECT] cam{int(cam_idx)} comps=0 selected=[] pixels=0 "
            f"elong=- length=- rejected_reason=no_soft",
            flush=True,
        )
        return empty
    soft = ds_soft.astype(np.float32, copy=False)
    h, w = soft.shape[:2]
    tx, ty = float(tip_xy_ds[0]), float(tip_xy_ds[1])
    r = max(float(radius_ds), 1.0)
    ys, xs = np.where(soft > float(FITLINE_WEIGHT_EPS))
    if int(xs.size) < FITLINE_MIN_PX:
        print(
            f"[SHAFT_SELECT] cam{int(cam_idx)} comps=0 selected=[] pixels=0 "
            f"elong=- length=- rejected_reason=no_pixels",
            flush=True,
        )
        return empty
    dist = np.hypot(xs.astype(np.float64) - tx, ys.astype(np.float64) - ty)
    in_roi = dist <= r
    if int(np.count_nonzero(in_roi)) < FITLINE_MIN_PX:
        print(
            f"[SHAFT_SELECT] cam{int(cam_idx)} comps=0 selected=[] pixels=0 "
            f"elong=- length=- rejected_reason=roi_empty",
            flush=True,
        )
        return empty

    binary = np.zeros((h, w), dtype=np.uint8)
    binary[ys[in_roi], xs[in_roi]] = 255
    n_lab, labels, stats, _cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    comps: List[dict] = []
    rej_xs: List[np.ndarray] = []
    rej_ys: List[np.ndarray] = []
    for lab in range(1, int(n_lab)):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < int(SHAFT_CC_MIN_AREA):
            continue
        ys_c, xs_c = np.where(labels == lab)
        xs_f = xs_c.astype(np.float64)
        ys_f = ys_c.astype(np.float64)
        vx, vy, cx, cy, elong, length, thick, _n = _pca_geom_xy(xs_f, ys_f)
        dmin = float(
            np.min(np.hypot(xs_f - tx, ys_f - ty))
        )
        rec = {
            "lab": int(lab),
            "area": area,
            "xs": xs_f,
            "ys": ys_f,
            "vx": vx,
            "vy": vy,
            "cx": cx,
            "cy": cy,
            "elong": elong,
            "length": length,
            "thick": thick,
            "dmin": dmin,
        }
        shaft_like = (
            elong >= float(SHAFT_MIN_ELONG)
            and length >= float(SHAFT_MIN_LENGTH_PX)
            and thick <= float(SHAFT_MAX_THICK_PX)
            and dmin <= float(SHAFT_TIP_DIST_FRAC) * r
        )
        rec["score"] = _shaft_component_score(
            dmin=dmin, radius=r, elong=elong, length=length, thickness=thick
        )
        rec["shaft_like"] = bool(shaft_like)
        if shaft_like:
            comps.append(rec)
        else:
            rej_xs.append(xs_f)
            rej_ys.append(ys_f)

    n_cc = max(int(n_lab) - 1, 0)
    if not comps:
        rx = np.concatenate(rej_xs) if rej_xs else None
        ry = np.concatenate(rej_ys) if rej_ys else None
        print(
            f"[SHAFT_SELECT] cam{int(cam_idx)} comps={n_cc} selected=[] pixels=0 "
            f"elong=- length=- rejected_reason=no_shaft_component",
            flush=True,
        )
        return None, None, None, rx, ry

    comps.sort(key=lambda c: float(c["score"]), reverse=True)
    best = comps[0]
    if float(best["score"]) < float(SHAFT_MIN_SCORE):
        rx = np.concatenate([c["xs"] for c in comps] + rej_xs)
        ry = np.concatenate([c["ys"] for c in comps] + rej_ys)
        print(
            f"[SHAFT_SELECT] cam{int(cam_idx)} comps={n_cc} selected=[] pixels=0 "
            f"elong={best['elong']:.2f} length={best['length']:.1f} "
            f"rejected_reason=low_score",
            flush=True,
        )
        return None, None, None, rx, ry

    selected = [best]
    selected_labs = {int(best["lab"])}
    for other in comps[1:]:
        if any(_comps_mergeable(s, other, (tx, ty)) for s in selected):
            selected.append(other)
            selected_labs.add(int(other["lab"]))
        else:
            rej_xs.append(other["xs"])
            rej_ys.append(other["ys"])

    sel_xs = np.concatenate([c["xs"] for c in selected])
    sel_ys = np.concatenate([c["ys"] for c in selected])
    if int(sel_xs.size) < FITLINE_MIN_PX:
        rx = np.concatenate(rej_xs + [sel_xs])
        ry = np.concatenate(rej_ys + [sel_ys])
        print(
            f"[SHAFT_SELECT] cam{int(cam_idx)} comps={n_cc} selected=[] pixels=0 "
            f"elong=- length=- rejected_reason=few_pixels",
            flush=True,
        )
        return None, None, None, rx, ry

    inten = soft[sel_ys.astype(np.int32), sel_xs.astype(np.int32)].astype(np.float64)
    dist_s = np.hypot(sel_xs - tx, sel_ys - ty)
    prox = np.clip(1.0 - dist_s / r, 0.05, 1.0) ** float(FITLINE_STAGE2_DIST_WEIGHT_POWER)
    vx, vy, x0, y0 = _fit_from_points(sel_xs, sel_ys, inten * prox)
    vx, vy = _orient_fitline_tipward(vx, vy, x0, y0, board_center_ds)
    span = float(
        np.hypot(float(sel_xs.max()) - float(sel_xs.min()), float(sel_ys.max()) - float(sel_ys.min()))
    )
    if span < 2.0:
        rx = np.concatenate(rej_xs + [sel_xs])
        ry = np.concatenate(rej_ys + [sel_ys])
        print(
            f"[SHAFT_SELECT] cam{int(cam_idx)} comps={n_cc} selected=[] pixels={int(sel_xs.size)} "
            f"elong=- length=- rejected_reason=short_span",
            flush=True,
        )
        return None, None, None, rx, ry

    elong_s = float(np.mean([c["elong"] for c in selected]))
    length_s = float(sum(c["length"] for c in selected))
    labs = ",".join(str(int(c["lab"])) for c in selected)
    print(
        f"[SHAFT_SELECT] cam{int(cam_idx)} comps={n_cc} selected=[{labs}] "
        f"pixels={int(sel_xs.size)} elong={elong_s:.2f} length={length_s:.1f} "
        f"rejected_reason=-",
        flush=True,
    )
    rx = np.concatenate(rej_xs) if rej_xs else None
    ry = np.concatenate(rej_ys) if rej_ys else None
    return (
        (float(vx), float(vy), float(x0), float(y0)),
        sel_xs,
        sel_ys,
        rx,
        ry,
    )


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median; equal-weight median when weights are degenerate."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.clip(np.asarray(weights, dtype=np.float64).reshape(-1), 0.0, None)
    n = int(v.size)
    if n <= 0:
        return 0.0
    if n == 1:
        return float(v[0])
    order = np.argsort(v, kind="mergesort")
    v = v[order]
    w = w[order]
    wsum = float(w.sum())
    if wsum <= 1e-12:
        return float(v[n // 2])
    cdf = np.cumsum(w)
    return float(v[int(np.searchsorted(cdf, 0.5 * wsum, side="left"))])


def _equal_weight_rms_perp(
    xs: np.ndarray,
    ys: np.ndarray,
    vx: float,
    vy: float,
    x0: float,
    y0: float,
) -> float:
    """RMS perpendicular distance, equal weight per point (not blob mass)."""
    n = int(xs.size)
    if n <= 0:
        return 0.0
    nn = float(np.hypot(vx, vy)) or 1.0
    ux, uy = float(vx) / nn, float(vy) / nn
    dx = xs.astype(np.float64) - float(x0)
    dy = ys.astype(np.float64) - float(y0)
    perp = dx * (-uy) + dy * ux
    return float(np.sqrt(np.mean(perp * perp)))


def _log_center_refit(
    *,
    cam_idx: int,
    bins: int,
    old_angle: float,
    candidate_angle: float,
    old_res: float,
    candidate_res: float,
    accepted: bool,
    reject_reason: str,
    scale_x: float,
    scale_y: float,
) -> None:
    print(
        "[CENTER_REFIT]\n"
        f"cam={int(cam_idx)}\n"
        f"bins={int(bins)}\n"
        f"old_angle={old_angle:.1f}\n"
        f"candidate_angle={candidate_angle:.1f}\n"
        f"old_res={old_res:.3f}\n"
        f"candidate_res={candidate_res:.3f}\n"
        f"accepted={bool(accepted)}\n"
        f"reject_reason={reject_reason}\n"
        f"coordinate_space=fit_ds FITLINE_SIZE={int(FITLINE_SIZE)}\n"
        f"scale_x={float(scale_x):.6f}\n"
        f"scale_y={float(scale_y):.6f}",
        flush=True,
    )


def _centerline_refit_stage2(
    ds_soft: np.ndarray,
    tip_xy_ds: Tuple[float, float],
    radius_ds: float,
    line: Tuple[float, float, float, float],
    board_center_ds: Tuple[float, float],
    *,
    cam_idx: int = -1,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> Tuple[
    Tuple[float, float, float, float],
    List[Tuple[float, float]],
    bool,
    str,
    Optional[Tuple[float, float, float, float]],
]:
    """Robust midline after Stage2 PCA: one median point per longitudinal bin.

    Fit is equal-weight PCA through the SAME xy points that are drawn yellow.
    Guard compares RMS perp of that same point set to old vs candidate.
    Returns (chosen_line, bin_xy, accepted, reject_reason, candidate_line).
    """
    vx0, vy0, x00, y00 = (float(line[0]), float(line[1]), float(line[2]), float(line[3]))
    orig = (vx0, vy0, x00, y00)
    old_ang = _fitline_angle_deg(vx0, vy0)

    def _fail(reason: str, pts: List[Tuple[float, float]], bins: int) -> tuple:
        _log_center_refit(
            cam_idx=cam_idx,
            bins=bins,
            old_angle=old_ang,
            candidate_angle=old_ang,
            old_res=-1.0,
            candidate_res=-1.0,
            accepted=False,
            reject_reason=reason,
            scale_x=scale_x,
            scale_y=scale_y,
        )
        return orig, pts, False, reason, None

    xs, ys, inten, prox_w = _soft_roi_xyw(ds_soft, tip_xy_ds, radius_ds)
    n = int(xs.size)
    if n < FITLINE_MIN_PX:
        return _fail("low_pixels", [], 0)
    w = np.clip(prox_w.astype(np.float64), 0.0, None)
    if float(w.sum()) <= 1e-12:
        w = np.clip(inten.astype(np.float64), 0.0, None)
    nn = float(np.hypot(vx0, vy0)) or 1.0
    ux, uy = vx0 / nn, vy0 / nn
    nx, ny = -uy, ux
    dx = xs - x00
    dy = ys - y00
    u = dx * ux + dy * uy
    v = dx * nx + dy * ny
    u_lo = float(np.min(u))
    u_hi = float(np.max(u))
    span = u_hi - u_lo
    if span < float(CENTERLINE_MIN_LENGTH_PX):
        return _fail("short_span", [], 0)

    n_bins = int(CENTERLINE_N_BINS)
    edges = np.linspace(u_lo, u_hi, n_bins + 1)
    edges[-1] = u_hi + 1e-9
    bin_id = np.digitize(u, edges[1:-1], right=False)
    bin_id = np.clip(bin_id, 0, n_bins - 1)

    bin_x: List[float] = []
    bin_y: List[float] = []
    bin_u: List[float] = []
    min_px = int(CENTERLINE_MIN_BIN_PX)
    for b in range(n_bins):
        m = bin_id == b
        n_b = int(np.count_nonzero(m))
        if n_b < min_px:
            continue
        wb = w[m]
        if float(wb.sum()) <= 1e-12:
            continue
        u_m = float(_weighted_median(u[m], wb))
        v_m = float(_weighted_median(v[m], wb))
        bin_u.append(u_m)
        bin_x.append(float(x00 + u_m * ux + v_m * nx))
        bin_y.append(float(y00 + u_m * uy + v_m * ny))

    pts: List[Tuple[float, float]] = list(zip(bin_x, bin_y))
    n_valid = len(bin_x)
    if n_valid < int(CENTERLINE_MIN_BINS):
        return _fail("few_bins", pts, n_valid)
    length = float(max(bin_u) - min(bin_u))
    if length < float(CENTERLINE_MIN_LENGTH_PX):
        return _fail("short_length", pts, n_valid)

    px = np.asarray(bin_x, dtype=np.float64)
    py = np.asarray(bin_y, dtype=np.float64)
    # Equal weight per yellow point — same set used for the candidate fit.
    eq = np.ones(n_valid, dtype=np.float64)
    vx, vy, x0, y0 = _fit_from_points(px, py, eq)
    vx, vy = _orient_fitline_tipward(vx, vy, x0, y0, board_center_ds)
    cand = (float(vx), float(vy), float(x0), float(y0))
    cand_ang = _fitline_angle_deg(vx, vy)

    old_res = _equal_weight_rms_perp(px, py, vx0, vy0, x00, y00)
    cand_res = _equal_weight_rms_perp(px, py, vx, vy, x0, y0)
    worse = cand_res > (
        old_res * float(CENTERLINE_RES_WORSE_FRAC) + float(CENTERLINE_RES_WORSE_PX)
    )
    if worse:
        _log_center_refit(
            cam_idx=cam_idx,
            bins=n_valid,
            old_angle=old_ang,
            candidate_angle=cand_ang,
            old_res=old_res,
            candidate_res=cand_res,
            accepted=False,
            reject_reason="residual_worse",
            scale_x=scale_x,
            scale_y=scale_y,
        )
        return orig, pts, False, "residual_worse", cand

    _log_center_refit(
        cam_idx=cam_idx,
        bins=n_valid,
        old_angle=old_ang,
        candidate_angle=cand_ang,
        old_res=old_res,
        candidate_res=cand_res,
        accepted=True,
        reject_reason="ok",
        scale_x=scale_x,
        scale_y=scale_y,
    )
    return cand, pts, True, "ok", cand


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


def _largest_motion_cc_area(mask: np.ndarray) -> int:
    """Area of the largest connected component (0 if none)."""
    largest, _n_cc = _motion_cc_stats(mask)
    return largest


def _motion_cc_stats(mask: np.ndarray) -> Tuple[int, int]:
    """(largest CC area, number of CCs)."""
    if mask is None or mask.size == 0:
        return 0, 0
    binary = (mask > 0).astype(np.uint8)
    n_lab, _labels, stats, _cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if int(n_lab) <= 1:
        return 0, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return int(np.max(areas)), int(areas.size)


def _motion_is_speckle(mask: np.ndarray) -> bool:
    """True when motion is many tiny dots, not a dart (even a thin one)."""
    largest, n_cc = _motion_cc_stats(mask)
    if largest >= int(MOTION_MIN_BLOB_AREA):
        return False
    return n_cc >= int(MOTION_SPECKLE_MIN_CCS)


def _keep_large_motion_ccs(gray: np.ndarray, *, min_area: int) -> np.ndarray:
    """Zero CCs smaller than min_area (speckle). Gray/mask same HxW."""
    if gray is None or gray.size == 0:
        return gray
    binary = (gray > 0).astype(np.uint8)
    n_lab, labels, stats, _cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(gray.shape[:2], dtype=np.uint8)
    min_a = max(1, int(min_area))
    for lab in range(1, int(n_lab)):
        if int(stats[lab, cv2.CC_STAT_AREA]) >= min_a:
            keep[labels == lab] = 255
    if gray.ndim == 2:
        return np.where(keep > 0, gray, 0).astype(gray.dtype, copy=False)
    return gray


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


def _motion_blob_gray_u8(
    r: DartDetectResult, out_size: int
) -> Optional[np.ndarray]:
    """Gated dart motion only — never ungated absdiff speckle."""
    blob = r.fit_gray
    if blob is None or int(np.count_nonzero(blob)) <= 0:
        blob = r.motion_gray
    if blob is None or int(np.count_nonzero(blob)) <= 0:
        return None
    if blob.ndim == 3:
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2GRAY)
    if blob.shape[0] != out_size or blob.shape[1] != out_size:
        blob = cv2.resize(blob, (out_size, out_size), interpolation=cv2.INTER_AREA)
    blob = _keep_large_motion_ccs(blob, min_area=int(MOTION_MIN_BLOB_AREA))
    if int(np.count_nonzero(blob)) <= 0:
        return None
    return blob.astype(np.uint8, copy=False)


def _compose_cam_colored_motion(
    cams: List[DartDetectResult],
    out_size: int,
) -> np.ndarray:
    """One black frame: each camera's motion blob in that camera's debug color."""
    acc = np.zeros((out_size, out_size, 3), dtype=np.float32)
    seen: set[int] = set()
    ordered = sorted(cams, key=lambda r: int(r.cam_idx))
    for r in ordered:
        cam = int(r.cam_idx)
        if cam in seen:
            continue
        gray = _motion_blob_gray_u8(r, out_size)
        if gray is None or int(np.count_nonzero(gray)) <= 0:
            continue
        seen.add(cam)
        stretched = _contrast_stretch_gray(gray)
        w = stretched.astype(np.float32) / 255.0
        col = np.array(_hit_debug_cam_color(cam), dtype=np.float32)
        acc += w[:, :, None] * col
    canvas = np.clip(acc, 0, 255).astype(np.uint8)
    y = 10
    for cam in sorted(seen):
        col = _hit_debug_cam_color(cam)
        cv2.rectangle(canvas, (8, y), (20, y + 12), col, -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"cam{cam}",
            (26, y + 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            col,
            1,
            cv2.LINE_AA,
        )
        y += 16
    return canvas


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


_SCORE_CANVAS_BG = (22, 22, 22)
_WIRE_NEAR_DEG = 3.0


def _score_cals_for_debug(
    board_calibrator: Optional[BoardCalibrator],
) -> Optional[List[Optional[BoardCalibration]]]:
    if board_calibrator is None:
        return None
    return [board_calibrator.get(i) for i in (0, 2, 4)]


def _fitline_original_xyv(r: DartDetectResult) -> Tuple[float, float, float, float]:
    """FitLine (x0,y0,vx,vy) in original FITLINE_SIZE warp pixels (unit direction)."""
    sx = float(r.fit_scale_x) if float(r.fit_scale_x) > 1e-9 else 1.0
    sy = float(r.fit_scale_y) if float(r.fit_scale_y) > 1e-9 else 1.0
    x0, y0 = _map_ds_xy_to_original(float(r.line_x0), float(r.line_y0), sx, sy)
    vx = float(r.line_vx) * sx
    vy = float(r.line_vy) * sy
    n = float(np.hypot(vx, vy))
    if n < 1e-9:
        return x0, y0, 0.0, 0.0
    return x0, y0, vx / n, vy / n


def _dir_original_unit(vx: float, vy: float, r: Optional[DartDetectResult] = None) -> Tuple[float, float]:
    sx = float(r.fit_scale_x) if r is not None and float(r.fit_scale_x) > 1e-9 else 1.0
    sy = float(r.fit_scale_y) if r is not None and float(r.fit_scale_y) > 1e-9 else 1.0
    ox, oy = float(vx) * sx, float(vy) * sy
    n = float(np.hypot(ox, oy))
    if n < 1e-9:
        return 0.0, 0.0
    return ox / n, oy / n


def _hit_score_label(
    px: float,
    py: float,
    *,
    out_size: int,
    cals: Optional[List[Optional[BoardCalibration]]],
) -> str:
    n, z, _s = score_topdown_point(
        float(px), float(py), out_size=out_size, cals=cals, log_geom=False
    )
    return _zone_short_label(z, int(n))


def _theta_from_center(px: float, py: float, center: Tuple[float, float]) -> float:
    return math.degrees(math.atan2(float(px) - center[0], -(float(py) - center[1]))) % 360.0


def _nearest_scoring_wire(
    px: float,
    py: float,
    center: Tuple[float, float],
    wires: Tuple[float, ...],
) -> Tuple[int, float]:
    theta = _theta_from_center(px, py, center)
    best_i = 0
    best_d = 999.0
    for i, ang in enumerate(wires):
        d = float(_ang_abs_delta_deg(theta, float(ang)))
        if d < best_d:
            best_i = i
            best_d = d
    return best_i, best_d


def _wire_pair_label(wire_idx: int) -> str:
    prev_n = BOARD_SEGMENT_NUMBERS[(int(wire_idx) - 1) % SEGMENT_COUNT]
    next_n = BOARD_SEGMENT_NUMBERS[int(wire_idx) % SEGMENT_COUNT]
    return f"{prev_n}/{next_n}"


def _debug_draw_ellipse(
    img: np.ndarray,
    ell: Tuple,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    (ecx, ecy), (ew, eh), ang = ell
    ax = max(1, int(round(float(ew) * 0.5)))
    ay = max(1, int(round(float(eh) * 0.5)))
    cv2.ellipse(
        img,
        (int(round(float(ecx))), int(round(float(ecy)))),
        (ax, ay),
        float(ang),
        0,
        360,
        color,
        int(thickness),
        cv2.LINE_AA,
    )


def _wire_end_xy(
    cx: float,
    cy: float,
    ang: float,
    r_out: float,
    double_ell: Optional[Tuple],
) -> Tuple[int, int]:
    if double_ell is not None:
        pt = _ray_to_ellipse_edge_f((cx, cy), float(ang), double_ell)
        if pt is not None:
            return int(round(pt[0])), int(round(pt[1]))
    st, ct = _angle_sin_cos(float(ang))
    return int(round(cx + r_out * st)), int(round(cy + r_out * ct))


def make_scoring_board_canvas(
    *,
    out_size: int = FITLINE_SIZE,
    cal: Optional[BoardCalibration] = None,
    cals: Optional[List[Optional[BoardCalibration]]] = None,
    near_points: Optional[List[Tuple[float, float]]] = None,
    wire_near_deg: float = _WIRE_NEAR_DEG,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Clean 300x300 board-coordinate canvas using score_topdown_point geometry.

    No camera photograph. Rings/wires/center match scoring, including ellipses
    when score_topdown_point uses them.
    """
    size = int(out_size) if out_size > 0 else FITLINE_SIZE
    img = np.full((size, size, 3), _SCORE_CANVAS_BG, dtype=np.uint8)
    center, board_r = _board_center_and_r(size)
    rings = scoring_ring_radii(size, cal=cal, cals=cals)
    ellipses = scoring_ring_ellipses(size, cal=cal, cals=cals)
    wires = _score_wire_thetas(size, cal=cal, cals=cals)
    cx, cy = float(center[0]), float(center[1])
    icx, icy = int(round(cx)), int(round(cy))
    r_out = float(rings.get("double_outer", board_r))
    if ellipses is not None:
        mean_d = float(_mean_ellipse_semi(ellipses["double_outer"]))
        r_bull_in = mean_d * float(RING_BULL_INNER_FRAC)
        r_bull_out = mean_d * float(RING_BULL_OUTER_FRAC)
    else:
        r_bull_in = float(rings["bull_inner"])
        r_bull_out = float(rings["bull_outer"])

    col_d = (0, 165, 255)
    col_t = (0, 220, 120)
    col_b = (70, 70, 255)
    col_wire = (190, 190, 190)
    col_hi = (0, 255, 255)
    col_num = (170, 170, 170)
    col_cen = (255, 0, 255)

    if ellipses is not None:
        _debug_draw_ellipse(img, ellipses["double_outer"], col_d, 1)
        _debug_draw_ellipse(img, ellipses["double_inner"], col_d, 1)
        _debug_draw_ellipse(img, ellipses["triple_outer"], col_t, 1)
        _debug_draw_ellipse(img, ellipses["triple_inner"], col_t, 1)
        d_ell = ellipses["double_outer"]
    else:
        for key, col in (
            ("double_outer", col_d),
            ("double_inner", col_d),
            ("triple_outer", col_t),
            ("triple_inner", col_t),
        ):
            rr = max(1, int(round(float(rings[key]))))
            cv2.circle(img, (icx, icy), rr, col, 1, cv2.LINE_AA)
        d_ell = None

    cv2.circle(img, (icx, icy), max(1, int(round(r_bull_out))), col_b, 1, cv2.LINE_AA)
    cv2.circle(img, (icx, icy), max(1, int(round(r_bull_in))), col_b, 1, cv2.LINE_AA)

    near_idx: Dict[int, float] = {}
    for pt in near_points or []:
        wi, dist = _nearest_scoring_wire(float(pt[0]), float(pt[1]), center, wires)
        if dist <= float(wire_near_deg):
            prev = near_idx.get(wi)
            if prev is None or dist < prev:
                near_idx[wi] = dist

    for i, ang in enumerate(wires):
        x1, y1 = _wire_end_xy(cx, cy, float(ang), r_out, d_ell)
        hi = i in near_idx
        cv2.line(
            img,
            (icx, icy),
            (x1, y1),
            col_hi if hi else col_wire,
            2 if hi else 1,
            cv2.LINE_AA,
        )

    r_num = 0.5 * (float(rings["triple_outer"]) + float(rings["double_inner"]))
    for k in range(min(len(wires), SEGMENT_COUNT)):
        a0 = float(wires[k])
        a1 = float(wires[(k + 1) % len(wires)])
        span = (a1 - a0) % 360.0
        mid = (a0 + 0.5 * span) % 360.0
        st, ct = _angle_sin_cos(mid)
        tx = int(round(cx + r_num * st))
        ty = int(round(cy + r_num * ct))
        lab = str(int(BOARD_SEGMENT_NUMBERS[k % SEGMENT_COUNT]))
        cv2.putText(
            img,
            lab,
            (tx - 6, ty + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            col_num,
            1,
            cv2.LINE_AA,
        )

    cv2.drawMarker(img, (icx, icy), col_cen, cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)
    cv2.circle(img, (icx, icy), 3, col_cen, 1, cv2.LINE_AA)

    meta: Dict[str, object] = {
        "center": center,
        "wires": wires,
        "rings": rings,
        "near_wires": near_idx,
    }
    return img, meta


def _draw_marked_point(
    img: np.ndarray,
    xy: Tuple[float, float],
    color: Tuple[int, int, int],
    *,
    radius: int = 3,
    cross: int = 7,
) -> None:
    pt = (int(round(xy[0])), int(round(xy[1])))
    cv2.drawMarker(img, pt, color, cv2.MARKER_CROSS, cross, 1, cv2.LINE_AA)
    cv2.circle(img, pt, max(1, int(radius)), color, -1, cv2.LINE_AA)
    cv2.circle(img, pt, max(2, int(radius) + 1), (255, 255, 255), 1, cv2.LINE_AA)


def _draw_hud_lines(
    img: np.ndarray,
    lines: List[str],
    *,
    color: Tuple[int, int, int] = (240, 240, 240),
    x: int = 4,
    y0: int = 14,
    step: int = 14,
    scale: float = 0.40,
) -> None:
    if not lines:
        return
    widths = [
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for line in lines
    ]
    box_w = int(max(widths)) + 10
    box_h = int(step * len(lines)) + 8
    x0, y_top = 2, 2
    cv2.rectangle(img, (x0, y_top), (x0 + box_w, y_top + box_h), (8, 8, 8), -1)
    cv2.rectangle(img, (x0, y_top), (x0 + box_w, y_top + box_h), (60, 60, 60), 1)
    for i, line in enumerate(lines):
        cv2.putText(
            img,
            line,
            (x, y0 + i * step),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )


def _label_xy(
    img: np.ndarray,
    xy: Tuple[float, float],
    text: str,
    color: Tuple[int, int, int],
    *,
    dx: int = 8,
    dy: int = -8,
) -> None:
    cv2.putText(
        img,
        text,
        (int(round(xy[0])) + int(dx), int(round(xy[1])) + int(dy)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        color,
        1,
        cv2.LINE_AA,
    )


def _pairwise_intersections_original(
    line_cams: List[DartDetectResult],
    board_center: Tuple[float, float],
) -> List[Tuple[int, int, Tuple[float, float]]]:
    out: List[Tuple[int, int, Tuple[float, float]]] = []
    n = len(line_cams)
    orig = [_fitline_original_xyv(r) for r in line_cams]
    for i in range(n):
        for j in range(i + 1, n):
            ax0, ay0, avx, avy = orig[i]
            bx0, by0, bvx, bvy = orig[j]
            cross = _line_direction_cross(avx, avy, bvx, bvy)
            if cross < float(FITLINE_PAIR_CROSS_MIN):
                continue
            tip, _res = intersect_lines_least_squares(
                [(ax0, ay0, avx, avy), (bx0, by0, bvx, bvy)],
                board_center=board_center,
            )
            if tip is None:
                continue
            out.append((i, j, (float(tip[0]), float(tip[1]))))
    return out


def _save_fitline_smooth_debug(
    stem: str,
    fused: FusedDartResult,
    line_cams: List[DartDetectResult],
    motion_dir: str,
    *,
    base_gray: np.ndarray,
    soft_acc: Optional[np.ndarray],
    draw_cams: Optional[List[DartDetectResult]] = None,
    board_calibrator: Optional[BoardCalibrator] = None,
) -> List[str]:
    """Scoring-coordinate canvas + FitLine rays + pair intersections + FIT P.

    ``line_cams`` = consensus-used cams (full color). ``draw_cams`` may include
    rejected fleeers drawn dim. Photo / motion underlay is not used.
    """
    used = list(line_cams)
    draw = list(draw_cams) if draw_cams is not None else list(line_cams)
    if not _hit_motion_debug_enabled() or not draw:
        return []
    _ = base_gray
    _ = soft_acc
    saved: List[str] = []
    used_ids = {int(r.cam_idx) for r in used}
    out_size = FITLINE_SIZE
    cals = _score_cals_for_debug(board_calibrator)
    fit_xy = (float(fused.tip_xy[0]), float(fused.tip_xy[1]))
    canvas, meta = make_scoring_board_canvas(
        out_size=out_size,
        cals=cals,
        near_points=[fit_xy],
    )
    center = meta["center"]  # type: ignore[assignment]
    assert isinstance(center, tuple)

    for r in draw:
        x0, y0, vx, vy = _fitline_original_xyv(r)
        color = _hit_debug_cam_color(r.cam_idx)
        used_cam = int(r.cam_idx) in used_ids
        if not used_cam:
            color = tuple(int(round(c * 0.35)) for c in color)
        _draw_fitline_on(
            canvas, x0, y0, vx, vy, color, thickness=1 if not used_cam else 2
        )

    for _i, _j, (px, py) in _pairwise_intersections_original(used, center):
        cv2.circle(
            canvas,
            (int(round(px)), int(round(py))),
            3,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )

    _draw_marked_point(canvas, fit_xy, (0, 255, 255), radius=3, cross=9)
    _label_xy(canvas, fit_xy, "FIT", (0, 255, 255), dx=8, dy=-10)
    fit_lab = _hit_score_label(fit_xy[0], fit_xy[1], out_size=out_size, cals=cals)
    rec_note = ""
    if int(fused.fitq_recovery_cam) >= 0:
        rec_note = f" recovery=cam{int(fused.fitq_recovery_cam)}"
    hud = [
        f"FIT ({fit_xy[0]:.1f},{fit_xy[1]:.1f}) {fit_lab}",
        f"spread={fused.fitq_pair_spread:.1f}{rec_note}",
    ]
    near_wires: Dict[int, float] = meta["near_wires"]  # type: ignore[assignment]
    if near_wires:
        wi, dist = min(near_wires.items(), key=lambda kv: kv[1])
        hud.append(f"wire {_wire_pair_label(wi)}  {dist:.2f}deg")
    for r in draw:
        x0, y0, vx, vy = _fitline_original_xyv(r)
        qx = int(round(x0 + 28.0 * vx))
        qy = int(round(y0 + 28.0 * vy))
        color = _hit_debug_cam_color(r.cam_idx)
        cv2.putText(
            canvas,
            f"cam{int(r.cam_idx)} q={r.fitq_quality:.2f}",
            (qx, qy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            color,
            1,
            cv2.LINE_AA,
        )
    _draw_hud_lines(canvas, hud)
    cpath = os.path.join(motion_dir, f"{stem}_fitline_smooth_intersect.png")
    cv2.imwrite(cpath, canvas)
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
    """Save hit debug images during normal detection:

    - ``{stem}_fused_motion_raw.png`` — fused motion blobs on black, color per camera
    - ``motion/{stem}_fitline_smooth_intersect.png`` — scoring-coordinate canvas
      (no board photo) + FitLine rays + pair intersections + FIT P
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
    blob_cams: List[DartDetectResult] = []
    seen_blob: set[int] = set()
    for r in list(draw_cams) + list(fused.per_cam):
        cam = int(r.cam_idx)
        if cam in seen_blob:
            continue
        gray = _motion_blob_gray_u8(r, out_size)
        if gray is None or int(np.count_nonzero(gray)) <= 0:
            continue
        seen_blob.add(cam)
        blob_cams.append(r)
        if soft_acc is None:
            soft_acc = gray.astype(np.uint8, copy=True)
        else:
            np.maximum(soft_acc, gray, out=soft_acc)

    # One frame, blobs colored by camera (cam0 red / cam2 green / cam4 cyan).
    if blob_cams:
        canvas = _compose_cam_colored_motion(blob_cams, out_size)
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

    # 2) Scoring-coordinate canvas + FitLine rays + intersections + FIT P.
    intersect_paths: List[str] = []
    for p in _save_fitline_smooth_debug(
        stem,
        fused,
        line_cams,
        motion_dir,
        base_gray=base_gray,
        soft_acc=soft_acc,
        draw_cams=draw_cams,
        board_calibrator=board_calibrator,
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


def _dir_angle_delta_deg(vx0: float, vy0: float, vx1: float, vy1: float) -> float:
    """Smallest angle between two directions, 0..90 deg (unsigned axis)."""
    n0 = float(np.hypot(vx0, vy0)) or 1.0
    n1 = float(np.hypot(vx1, vy1)) or 1.0
    c = abs((vx0 / n0) * (vx1 / n1) + (vy0 / n0) * (vy1 / n1))
    c = float(np.clip(c, 0.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _soft_roi_xyw(
    ds_soft: np.ndarray,
    tip_xy_ds: Tuple[float, float],
    radius_ds: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """xs, ys, intensity, tip-proximity weights inside Stage2 circle."""
    empty = (
        np.zeros((0,), dtype=np.float64),
        np.zeros((0,), dtype=np.float64),
        np.zeros((0,), dtype=np.float64),
        np.zeros((0,), dtype=np.float64),
    )
    if ds_soft is None or ds_soft.size == 0:
        return empty
    soft = ds_soft.astype(np.float32, copy=False)
    ys, xs = np.where(soft > float(FITLINE_WEIGHT_EPS))
    if int(xs.size) < 1:
        return empty
    tx, ty = float(tip_xy_ds[0]), float(tip_xy_ds[1])
    r = max(float(radius_ds), 1.0)
    dist = np.hypot(xs.astype(np.float64) - tx, ys.astype(np.float64) - ty)
    keep = dist <= r
    if int(np.count_nonzero(keep)) < 1:
        return empty
    xs_f = xs[keep].astype(np.float64)
    ys_f = ys[keep].astype(np.float64)
    dist_k = dist[keep]
    inten = soft[ys[keep], xs[keep]].astype(np.float64)
    prox = np.clip(1.0 - dist_k / r, 0.05, 1.0) ** float(FITLINE_STAGE2_DIST_WEIGHT_POWER)
    return xs_f, ys_f, inten, inten * prox


def _perp_and_par(
    xs: np.ndarray,
    ys: np.ndarray,
    vx: float,
    vy: float,
    x0: float,
    y0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    nn = float(np.hypot(vx, vy)) or 1.0
    ux, uy = float(vx) / nn, float(vy) / nn
    dx = xs.astype(np.float64) - float(x0)
    dy = ys.astype(np.float64) - float(y0)
    par = dx * ux + dy * uy
    perp = np.abs(dx * (-uy) + dy * ux)
    return perp, par


def _compute_fitline_quality(
    xs: np.ndarray,
    ys: np.ndarray,
    weights: np.ndarray,
    vx: float,
    vy: float,
    x0: float,
    y0: float,
    *,
    board_r: float,
    s1s2_angle: float = 0.0,
    s1s2_tip: float = 0.0,
) -> Dict[str, float]:
    """Geometric line quality. Pixel count is only a weak extra term."""
    n = int(xs.size)
    z = {
        "pixels": float(n),
        "support": 0.0,
        "residual": 0.0,
        "linearity": 0.0,
        "length": 0.0,
        "thickness": 0.0,
        "s1s2_angle": float(s1s2_angle),
        "s1s2_tip": float(s1s2_tip),
        "quality": 0.0,
    }
    if n < 2:
        return z
    w = np.clip(weights.astype(np.float64), 0.0, None) if weights is not None else np.ones(n, dtype=np.float64)
    if int(w.size) != n:
        w = np.ones(n, dtype=np.float64)
    wsum = float(w.sum())
    if wsum <= 1e-9:
        w = np.ones(n, dtype=np.float64)
        wsum = float(n)
    wn = w / wsum
    perp, par = _perp_and_par(xs, ys, vx, vy, x0, y0)
    corridor = float(FITQ_CORRIDOR_PX)
    in_c = perp <= corridor
    support = float(w[in_c].sum() / wsum)
    residual = float(np.sqrt(np.sum(wn * perp * perp)))
    thickness = float(np.sum(wn * perp))
    if np.any(in_c):
        length = float(np.max(par[in_c]) - np.min(par[in_c]))
    else:
        length = float(np.max(par) - np.min(par)) if n else 0.0
    mx = float(np.sum(wn * xs))
    my = float(np.sum(wn * ys))
    dx = xs.astype(np.float64) - mx
    dy = ys.astype(np.float64) - my
    cxx = float(np.sum(wn * dx * dx))
    cxy = float(np.sum(wn * dx * dy))
    cyy = float(np.sum(wn * dy * dy))
    tr = cxx + cyy
    det = cxx * cyy - cxy * cxy
    disc = max(tr * tr - 4.0 * det, 0.0)
    l1 = 0.5 * (tr + np.sqrt(disc))
    l2 = 0.5 * (tr - np.sqrt(disc))
    linearity = float((l1 - l2) / (l1 + l2 + 1e-9))
    br = max(float(board_r), 1.0)
    length_n = float(np.clip(length / max(FITQ_LENGTH_REF_FRAC * br, 1.0), 0.0, 1.0))
    res_n = float(np.clip(residual / FITQ_RESIDUAL_REF_PX, 0.0, 1.0))
    th_n = float(np.clip(thickness / FITQ_THICK_REF_PX, 0.0, 1.0))
    ang_n = float(np.clip(float(s1s2_angle) / FITQ_ANGLE_PEN_DEG, 0.0, 1.0))
    pix_n = float(
        np.clip(np.log1p(float(n)) / np.log1p(FITQ_WEAK_PIXEL_REF), 0.0, 1.0)
    )
    # Shape first: a 15px thin shaft outranks an 800px round flight blob.
    quality = (
        0.32 * linearity
        + 0.24 * support
        + 0.20 * length_n
        + 0.04 * pix_n
        - 0.22 * res_n
        - 0.18 * th_n
        - 0.12 * ang_n
    )
    z.update(
        {
            "support": support,
            "residual": residual,
            "linearity": float(np.clip(linearity, 0.0, 1.0)),
            "length": length,
            "thickness": thickness,
            "quality": float(np.clip(quality, 0.0, 1.0)),
        }
    )
    return z


def _apply_fitq_to_result(r: DartDetectResult, q: Dict[str, float]) -> None:
    r.fitq_pixels = int(q.get("pixels", 0.0))
    r.fitq_support = float(q.get("support", 0.0))
    r.fitq_residual = float(q.get("residual", 0.0))
    r.fitq_linearity = float(q.get("linearity", 0.0))
    r.fitq_length = float(q.get("length", 0.0))
    r.fitq_thickness = float(q.get("thickness", 0.0))
    r.fitq_s1s2_angle = float(q.get("s1s2_angle", 0.0))
    r.fitq_s1s2_tip = float(q.get("s1s2_tip", 0.0))
    r.fitq_quality = float(q.get("quality", 0.0))


def _cam_stage2_roi_ds(
    r: DartDetectResult,
    tip_orig: Tuple[float, float],
    radius_orig: float,
    scale_x: float,
    scale_y: float,
) -> Tuple[Tuple[float, float], float]:
    sx = float(r.fit_scale_x) if float(r.fit_scale_x) > 1e-9 else scale_x
    sy = float(r.fit_scale_y) if float(r.fit_scale_y) > 1e-9 else scale_y
    tip_cam = _map_original_xy_to_ds(tip_orig[0], tip_orig[1], sx, sy)
    r_cam = float(radius_orig) / max(0.5 * (sx + sy), 1e-9)
    return tip_cam, r_cam


def _measure_cam_fitq(
    r: DartDetectResult,
    *,
    tip_orig: Tuple[float, float],
    radius_orig: float,
    scale_x: float,
    scale_y: float,
    board_r_ds: float,
    stage1_tip_ds: Optional[Tuple[float, float]] = None,
    roi_scale: float = 1.0,
) -> Dict[str, float]:
    tip_cam, r_cam = _cam_stage2_roi_ds(
        r, tip_orig, float(radius_orig) * float(roi_scale), scale_x, scale_y
    )
    xs, ys, inten, _prox = _soft_roi_xyw(r.fit_ds_gray, tip_cam, r_cam)
    ang = 0.0
    tip_d = 0.0
    if abs(r.stage1_vx) + abs(r.stage1_vy) > 1e-9:
        ang = _dir_angle_delta_deg(r.stage1_vx, r.stage1_vy, r.line_vx, r.line_vy)
        if stage1_tip_ds is not None:
            tip_d = _point_to_line_distance(
                r.line_x0,
                r.line_y0,
                r.line_vx,
                r.line_vy,
                float(stage1_tip_ds[0]),
                float(stage1_tip_ds[1]),
            )
    w = inten if int(inten.size) else np.ones((0,), dtype=np.float64)
    return _compute_fitline_quality(
        xs,
        ys,
        w,
        r.line_vx,
        r.line_vy,
        r.line_x0,
        r.line_y0,
        board_r=board_r_ds,
        s1s2_angle=ang,
        s1s2_tip=tip_d,
    )


def _line_direction_cross(
    vx0: float, vy0: float, vx1: float, vy1: float
) -> float:
    """|sin θ| between two direction vectors."""
    na = float(np.hypot(vx0, vy0)) or 1.0
    nb = float(np.hypot(vx1, vy1)) or 1.0
    return abs((vx0 / na) * (vy1 / nb) - (vy0 / na) * (vx1 / nb))


def _max_line_cross(line_cams: List[DartDetectResult]) -> float:
    best = 0.0
    n = len(line_cams)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = line_cams[i], line_cams[j]
            best = max(
                best,
                _line_direction_cross(a.line_vx, a.line_vy, b.line_vx, b.line_vy),
            )
    return best


def _pairwise_intersections(
    line_cams: List[DartDetectResult],
    board_center_ds: Tuple[float, float],
) -> List[Tuple[int, int, Tuple[float, float]]]:
    out: List[Tuple[int, int, Tuple[float, float]]] = []
    n = len(line_cams)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = line_cams[i], line_cams[j]
            cross = _line_direction_cross(a.line_vx, a.line_vy, b.line_vx, b.line_vy)
            if cross < float(FITLINE_PAIR_CROSS_MIN):
                continue
            tip, _res = intersect_lines_least_squares(
                [
                    (a.line_x0, a.line_y0, a.line_vx, a.line_vy),
                    (b.line_x0, b.line_y0, b.line_vx, b.line_vy),
                ],
                board_center=board_center_ds,
            )
            if tip is None:
                continue
            out.append((i, j, (float(tip[0]), float(tip[1]))))
    return out


def _pair_spread_px(
    line_cams: List[DartDetectResult],
    board_center_ds: Tuple[float, float],
) -> float:
    """RMS distance of pairwise intersections from their centroid (ds px)."""
    pairs = _pairwise_intersections(line_cams, board_center_ds)
    if len(pairs) < 2:
        return 0.0
    pts = np.array([p[2] for p in pairs], dtype=np.float64)
    c = pts.mean(axis=0)
    d = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
    return float(np.sqrt(np.mean(d * d)))


def _identify_suspect_cam(
    line_cams: List[DartDetectResult],
    board_center_ds: Tuple[float, float],
) -> int:
    """Index into line_cams. Geometry + quality, never min(pixel count)."""
    n = len(line_cams)
    if n < 3:
        if n <= 0:
            return -1
        return int(np.argmin([r.fitq_quality for r in line_cams]))
    pairs = _pairwise_intersections(line_cams, board_center_ds)
    by_pair = {(i, j): xy for i, j, xy in pairs}
    geom = np.zeros(n, dtype=np.float64)
    for k in range(n):
        others = [i for i in range(n) if i != k]
        if len(others) != 2:
            continue
        a, b = others[0], others[1]
        key = (a, b) if a < b else (b, a)
        trusted = by_pair.get(key)
        if trusted is None:
            continue
        dists: List[float] = []
        for j in others:
            key2 = (k, j) if k < j else (j, k)
            pt = by_pair.get(key2)
            if pt is None:
                continue
            dists.append(float(np.hypot(pt[0] - trusted[0], pt[1] - trusted[1])))
        if dists:
            geom[k] = float(np.mean(dists))
    gmax = float(np.max(geom)) + 1e-6
    scores = []
    for k, r in enumerate(line_cams):
        scores.append(
            0.55 * float(geom[k] / gmax)
            + 0.35 * (1.0 - float(r.fitq_quality))
            + 0.10 * float(np.clip(r.fitq_s1s2_angle / 40.0, 0.0, 1.0))
        )
    return int(np.argmax(np.asarray(scores, dtype=np.float64)))


def _orient_line_tipward(
    vx: float,
    vy: float,
    x0: float,
    y0: float,
    board_center_ds: Tuple[float, float],
) -> Tuple[float, float, float, float]:
    vx, vy = _orient_fitline_tipward(vx, vy, x0, y0, board_center_ds)
    return float(vx), float(vy), float(x0), float(y0)


def _shaft_focused_candidates(
    xs: np.ndarray,
    ys: np.ndarray,
    inten: np.ndarray,
    prox_w: np.ndarray,
    tip_xy: Tuple[float, float],
    board_center_ds: Tuple[float, float],
) -> List[Tuple[float, float, float, float]]:
    """Recovery-only line candidates. No connected-component size ranking."""
    n = int(xs.size)
    if n < FITLINE_MIN_PX:
        return []
    tx, ty = float(tip_xy[0]), float(tip_xy[1])
    cands: List[Tuple[float, float, float, float]] = []
    sqrt_w = np.sqrt(np.clip(inten, 0.0, None)) * np.clip(prox_w, 1e-6, None)

    def _add(vx: float, vy: float, x0: float, y0: float) -> None:
        span = float(
            np.hypot(float(np.max(xs) - np.min(xs)), float(np.max(ys) - np.min(ys)))
        )
        if span < 2.0:
            return
        cands.append(_orient_line_tipward(vx, vy, x0, y0, board_center_ds))

    # Equal-weight PCA: spatial extent, not brightness (shaft can beat fat flight).
    vx, vy, x0, y0 = _fit_from_points(xs, ys, None)
    _add(vx, vy, x0, y0)
    vx, vy, x0, y0 = _fit_from_points(xs, ys, sqrt_w)
    _add(vx, vy, x0, y0)
    vx, vy = _direction_through_point(xs, ys, tx, ty, sqrt_w)
    perp, _par = _perp_and_par(xs, ys, vx, vy, tx, ty)
    sigma = float(FITQ_IRLS_SIGMA_PX)
    irls_w = sqrt_w * np.exp(-0.5 * (perp / max(sigma, 0.4)) ** 2)
    vx2, vy2, x02, y02 = _fit_from_points(xs, ys, irls_w)
    _add(vx2, vy2, x02, y02)
    return cands


def _score_recovery_candidate(
    xs: np.ndarray,
    ys: np.ndarray,
    inten: np.ndarray,
    line: Tuple[float, float, float, float],
    *,
    board_r_ds: float,
    other_cams: List[DartDetectResult],
    self_r: DartDetectResult,
    board_center_ds: Tuple[float, float],
) -> Tuple[float, float, Dict[str, float]]:
    vx, vy, x0, y0 = line
    q = _compute_fitline_quality(
        xs, ys, inten, vx, vy, x0, y0, board_r=board_r_ds
    )
    saved = (self_r.line_vx, self_r.line_vy, self_r.line_x0, self_r.line_y0)
    self_r.line_vx, self_r.line_vy, self_r.line_x0, self_r.line_y0 = vx, vy, x0, y0
    trial = [self_r] + list(other_cams)
    spread = _pair_spread_px(trial, board_center_ds)
    self_r.line_vx, self_r.line_vy, self_r.line_x0, self_r.line_y0 = saved
    score = float(q["quality"]) + 0.30 * float(
        np.clip(1.0 - spread / max(FITQ_PAIR_SPREAD_MIN_PX, 1.0), 0.0, 1.0)
    )
    return score, spread, q


def _recover_suspicious_cam(
    line_cams: List[DartDetectResult],
    suspect_i: int,
    *,
    tip_orig: Tuple[float, float],
    radius_orig: float,
    scale_x: float,
    scale_y: float,
    board_center_orig: Tuple[float, float],
    board_center_ds: Tuple[float, float],
    board_r_ds: float,
    stage1_tip_ds: Tuple[float, float],
) -> bool:
    """Refit only the suspect cam. Keep original unless spread+quality improve."""
    if suspect_i < 0 or suspect_i >= len(line_cams):
        return False
    r = line_cams[suspect_i]
    others = [c for i, c in enumerate(line_cams) if i != suspect_i]
    if r.fit_ds_gray is None or len(others) < 1:
        return False
    sx = float(r.fit_scale_x) if float(r.fit_scale_x) > 1e-9 else scale_x
    sy = float(r.fit_scale_y) if float(r.fit_scale_y) > 1e-9 else scale_y
    bc_cam = _map_original_xy_to_ds(
        board_center_orig[0], board_center_orig[1], sx, sy
    )
    old_spread = _pair_spread_px(line_cams, board_center_ds)
    old_q = float(r.fitq_quality)
    old_line = (float(r.line_vx), float(r.line_vy), float(r.line_x0), float(r.line_y0))
    best_line: Optional[Tuple[float, float, float, float]] = None
    best_score = -1e9
    best_spread = old_spread
    best_q: Optional[Dict[str, float]] = None
    best_roi_scale = 1.0

    for roi_scale in (1.0, float(FITQ_WIDER_ROI_SCALE)):
        tip_cam, r_cam = _cam_stage2_roi_ds(
            r, tip_orig, float(radius_orig) * roi_scale, scale_x, scale_y
        )
        xs, ys, inten, prox_w = _soft_roi_xyw(r.fit_ds_gray, tip_cam, r_cam)
        if int(xs.size) < FITLINE_MIN_PX:
            continue
        cands = _shaft_focused_candidates(xs, ys, inten, prox_w, tip_cam, bc_cam)
        cands.append((r.line_vx, r.line_vy, r.line_x0, r.line_y0))
        for line in cands:
            score, spread, q = _score_recovery_candidate(
                xs,
                ys,
                inten,
                line,
                board_r_ds=board_r_ds,
                other_cams=others,
                self_r=r,
                board_center_ds=board_center_ds,
            )
            if roi_scale > 1.0 + 1e-6:
                if (
                    float(q["linearity"]) + 0.05 < r.fitq_linearity
                    and float(q["residual"]) > r.fitq_residual
                    and float(q["length"]) < r.fitq_length
                ):
                    continue
            if score > best_score + 1e-9:
                best_score = score
                best_line = line
                best_spread = spread
                best_q = q
                best_roi_scale = roi_scale

    if best_line is None or best_q is None:
        return False
    improved_spread = best_spread <= old_spread * float(FITQ_MIN_SPREAD_IMPROVE)
    similar_spread_better_q = (
        best_spread <= old_spread * 1.02 and float(best_q["quality"]) >= old_q + 0.08
    )
    if not improved_spread and not similar_spread_better_q:
        return False
    if float(best_q["quality"]) < old_q - 0.12:
        return False
    vx, vy, x0, y0 = best_line
    ang_chg = _dir_angle_delta_deg(old_line[0], old_line[1], vx, vy)
    orig_chg = float(np.hypot(x0 - old_line[2], y0 - old_line[3]))
    if ang_chg < 2.0 and orig_chg < 2.5:
        return False
    r.line_vx, r.line_vy, r.line_x0, r.line_y0 = float(vx), float(vy), float(x0), float(y0)
    r.line_angle_deg = _fitline_angle_deg(vx, vy)
    q_final = _measure_cam_fitq(
        r,
        tip_orig=tip_orig,
        radius_orig=radius_orig,
        scale_x=scale_x,
        scale_y=scale_y,
        board_r_ds=board_r_ds,
        stage1_tip_ds=stage1_tip_ds,
        roi_scale=best_roi_scale,
    )
    _apply_fitq_to_result(r, q_final)
    print(
        f"[FITQ] recovery cam{r.cam_idx} roi_scale={best_roi_scale:.2f} "
        f"spread {old_spread:.1f}->{best_spread:.1f} "
        f"q {old_q:.2f}->{r.fitq_quality:.2f}",
        flush=True,
    )
    return True


def _log_fitq_block(r: DartDetectResult) -> None:
    print(
        f"[FITQ] cam{int(r.cam_idx)}\n"
        f"pixels={int(r.fitq_pixels)}\n"
        f"support={r.fitq_support:.3f}\n"
        f"residual={r.fitq_residual:.2f}\n"
        f"linearity={r.fitq_linearity:.3f}\n"
        f"length={r.fitq_length:.1f}\n"
        f"thickness={r.fitq_thickness:.2f}\n"
        f"s1s2_angle={r.fitq_s1s2_angle:.1f}\n"
        f"s1s2_tip={r.fitq_s1s2_tip:.1f}\n"
        f"quality={r.fitq_quality:.3f}",
        flush=True,
    )


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
        return None, 0.0, list(line_cams)

    if n == 2:
        a, b = line_cams[0], line_cams[1]
        cross = _line_direction_cross(a.line_vx, a.line_vy, b.line_vx, b.line_vy)
        if cross < float(FUSE_MIN_CROSS_2CAM):
            print(
                f"[fuse] parallel_2cam cross={cross:.3f} "
                f"min={float(FUSE_MIN_CROSS_2CAM):.3f}",
                flush=True,
            )
            return None, 0.0, list(line_cams)
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
            cross = _line_direction_cross(a.line_vx, a.line_vy, b.line_vx, b.line_vy)
            # Reject near-parallel pairs (unstable intersection).
            if cross < float(FITLINE_PAIR_CROSS_MIN):
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
        return None, 0.0, list(line_cams)

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
        return None, 0.0

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
        return None, 0.0
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
    log_geom: bool = True,
) -> Tuple[int, str, int]:
    """Score u aligned top-down prostoru (seg.20 gore, offset=0)."""
    center, board_r = _board_center_and_r(out_size)
    ring_ellipses = scoring_ring_ellipses(out_size, cal=cal, cals=cals)
    rings = scoring_ring_radii(out_size, cal=cal, cals=cals)
    wires = calibrated_topdown_wire_thetas(out_size, cal=cal, cals=cals)
    if log_geom:
        _log_score_geom_sanity(out_size, center, wires)
    number, zone, score = topdown_point_to_score(
        px,
        py,
        center,
        board_r,
        segment20_offset=0,
        rings=rings,
        ring_ellipses=ring_ellipses,
        wire_thetas_deg=wires,
        log_geom=log_geom,
        center_source="warp_canvas_center:_board_center_and_r",
    )
    if log_geom:
        _log_score_debug(
            px, py, center, wires, number=number, zone=zone, score=score
        )
    return number, zone, score


def _score_wire_thetas(
    out_size: int,
    *,
    cal: Optional[BoardCalibration] = None,
    cals: Optional[List[Optional[BoardCalibration]]] = None,
) -> Tuple[float, ...]:
    """Exactly the wire set score_topdown_point uses (calibrated, else nominal 18°)."""
    wires = calibrated_topdown_wire_thetas(out_size, cal=cal, cals=cals)
    if wires is not None and len(wires) >= SEGMENT_COUNT - 2:
        return tuple(float(w) % 360.0 for w in wires[:SEGMENT_COUNT])
    return canonical_topdown_wire_thetas()


def _zone_multiplier(zone: str, score: int) -> int:
    if zone == "triple":
        return 3
    if zone == "double":
        return 2
    if zone in ("inner_bull", "outer_bull"):
        return int(score)
    if zone == "single":
        return 1
    return 0


def _log_score_debug(
    px: float,
    py: float,
    center: Tuple[float, float],
    wires: Optional[Tuple[float, ...]],
    *,
    number: int,
    zone: str,
    score: int,
) -> None:
    cx, cy = float(center[0]), float(center[1])
    dx, dy = float(px) - cx, float(py) - cy
    radius = math.hypot(dx, dy)
    raw_angle = math.atan2(dx, -dy)
    theta = math.degrees(raw_angle) % 360.0
    if wires is not None and len(wires) >= SEGMENT_COUNT - 2:
        slot, _a0, _a1, _n = _segment_slot_from_wires(theta, wires)
    else:
        slot = _segment_slot_from_theta(theta, 0)
    print(
        f"[SCORE_DEBUG] point=({float(px):.3f},{float(py):.3f}) "
        f"center=({cx:.3f},{cy:.3f}) dx={dx:.3f} dy={dy:.3f} radius={radius:.3f} "
        f"raw_angle={raw_angle:.6f} normalized_angle={theta:.3f} "
        f"segment_index={int(slot)} segment={int(number)} "
        f"multiplier={_zone_multiplier(zone, score)} zone={zone} score={int(score)}",
        flush=True,
    )


def draw_score_function_boundaries(
    img: np.ndarray,
    *,
    out_size: int = FITLINE_SIZE,
    cal: Optional[BoardCalibration] = None,
    cals: Optional[List[Optional[BoardCalibration]]] = None,
    color: Tuple[int, int, int] = (0, 220, 255),
    thickness: int = 1,
    label: bool = True,
) -> None:
    """Radial wires of the live score_topdown_point function (not an ideal overlay)."""
    if img is None or img.size == 0:
        return
    center, board_r = _board_center_and_r(int(out_size))
    rings = scoring_ring_radii(int(out_size), cal=cal, cals=cals)
    r_out = float(rings.get("double_outer", board_r))
    wires = _score_wire_thetas(int(out_size), cal=cal, cals=cals)
    cx, cy = float(center[0]), float(center[1])
    icx, icy = int(round(cx)), int(round(cy))
    for i, ang in enumerate(wires):
        st, ct = _angle_sin_cos(float(ang))
        x1 = int(round(cx + r_out * st))
        y1 = int(round(cy + r_out * ct))
        cv2.line(img, (icx, icy), (x1, y1), color, thickness, cv2.LINE_AA)
        if label:
            lx = int(round(cx + r_out * 1.04 * st))
            ly = int(round(cy + r_out * 1.04 * ct))
            cv2.putText(
                img,
                f"W{i}",
                (lx - 6, ly + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                color,
                1,
                cv2.LINE_AA,
            )
    cv2.circle(img, (icx, icy), 2, color, 1, cv2.LINE_AA)


def _log_score_geom_sanity(
    out_size: int,
    center: Tuple[float, float],
    wires: Optional[Tuple[float, ...]],
) -> None:
    """Confirm scoring uses warp canvas center + calibrated topdown wires. No math change."""
    exp = (int(out_size) - 1) * 0.5
    cx, cy = float(center[0]), float(center[1])
    if abs(cx - exp) > 1e-3 or abs(cy - exp) > 1e-3:
        print(
            f"[SCORE_GEOM] BUG scoring center=({cx:.3f},{cy:.3f}) "
            f"!= warp_canvas_center=({exp:.3f},{exp:.3f}) "
            f"out_size={int(out_size)}",
            flush=True,
        )
    if wires is None:
        print(
            "[SCORE_GEOM] NOTE wire_mode=nominal — calibrated_topdown_wire_thetas unavailable",
            flush=True,
        )
        return
    n = len(wires)
    if n < SEGMENT_COUNT:
        print(
            f"[SCORE_GEOM] NOTE calibrated wires n={n} expected={SEGMENT_COUNT}",
            flush=True,
        )
    gaps: List[float] = []
    for i in range(n):
        a0 = float(wires[i]) % 360.0
        a1 = float(wires[(i + 1) % n]) % 360.0
        gaps.append((a1 - a0) % 360.0)
    if gaps:
        mn = float(min(gaps))
        mx = float(max(gaps))
        mean = float(sum(gaps) / len(gaps))
        print(
            f"[SCORE_GEOM] wires n={n} clockwise_gaps_deg "
            f"mean={mean:.2f} min={mn:.2f} max={mx:.2f}",
            flush=True,
        )
        if mn < 8.0 or mx > 30.0 or abs(mean - 18.0) > 2.0:
            print(
                "[SCORE_GEOM] NOTE wire spacing irregular (still using these calibrated bounds)",
                flush=True,
            )


def extract_yfit_motion_layer(
    topdown_bgr: np.ndarray,
    ref_gray: np.ndarray,
    *,
    cam_idx: int = -1,
) -> DartDetectResult:
    """Board-gated soft motion in FITLINE_SIZE space — no FitLine."""
    h, w = topdown_bgr.shape[:2]
    cx_out, view_r, board_r = _warp_radii(w)
    cur_gray = cv2.cvtColor(topdown_bgr, cv2.COLOR_BGR2GRAY)
    if ref_gray.shape != cur_gray.shape:
        ref_gray = cv2.resize(ref_gray, (w, h), interpolation=cv2.INTER_AREA)
    raw_diff = cv2.absdiff(cur_gray, ref_gray)
    diff_mean = float(np.mean(raw_diff))
    mask = _motion_mask(ref_gray, cur_gray)
    board_mask = np.zeros((h, w), dtype=np.uint8)
    det_r = max(float(view_r), float(board_r) * float(BOARD_SISAL_EDGE_FRAC) * 0.92)
    det_r = min(det_r, float(cx_out) - 1.0)
    cv2.circle(board_mask, (int(cx_out), int(cx_out)), int(round(det_r)), 255, -1)
    mask = cv2.bitwise_and(mask, mask, mask=board_mask)
    n_px = int(np.count_nonzero(mask))
    soft = _soft_motion_gray_for_fit(raw_diff, board_mask)
    soft = _gate_soft_motion_for_fit(soft, mask)
    k = int(FITLINE_BLUR_KSIZE)
    if k >= 3:
        if k % 2 == 0:
            k += 1
        soft = cv2.GaussianBlur(soft.astype(np.float32), (k, k), float(FITLINE_BLUR_SIGMA))
        soft = np.clip(np.round(soft), 0, 255).astype(np.uint8)
    reason = ""
    if n_px < MOTION_MIN_PIXELS:
        reason = "low_motion"
    elif n_px > MOTION_MAX_PIXELS:
        reason = "high_motion"
    elif _motion_is_speckle(mask):
        reason = "no_blob"
    return DartDetectResult(
        cam_idx=cam_idx,
        found=False,
        motion_pixels=n_px,
        diff_mean=diff_mean,
        reject_reason=reason,
        motion_gray=mask,
        motion_raw=raw_diff.copy(),
        fit_gray=soft,
        fit_ds_gray=soft,
        fit_scale_x=1.0,
        fit_scale_y=1.0,
    )


def _yfit_motion_image(soft: np.ndarray) -> np.ndarray:
    """float32 motion weights in [0,1]; zeros below FITLINE_WEIGHT_EPS."""
    out = np.zeros(soft.shape, dtype=np.float32)
    if soft is None or soft.size == 0:
        return out
    m = soft > float(FITLINE_WEIGHT_EPS)
    if np.any(m):
        out[m] = soft[m].astype(np.float32) * (1.0 / 255.0)
    return out


def _yfit_angle_bank() -> dict:
    """Cached 1° sin/cos tables (invariant across hits)."""
    key = ("deg1",)
    bank = _YFIT_RAY_BANK.get(key)
    if bank is None:
        rad = np.deg2rad(np.arange(0.0, 360.0, 1.0, dtype=np.float64))
        bank = {
            "rad": rad.astype(np.float32),
            "ct": np.cos(rad).astype(np.float32),
            "st": np.sin(rad).astype(np.float32),
        }
        _YFIT_RAY_BANK[key] = bank
    return bank


def _yfit_fused_bbox(
    union: np.ndarray,
    *,
    margin: int,
    cx: float,
    cy: float,
    board_lim: float,
) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(union > 0)
    if int(xs.size) <= 0:
        return None
    h, w = union.shape[:2]
    m = int(max(0, margin))
    x0 = int(max(0, int(xs.min()) - m))
    y0 = int(max(0, int(ys.min()) - m))
    x1 = int(min(w - 1, int(xs.max()) + m))
    y1 = int(min(h - 1, int(ys.max()) + m))
    r = float(board_lim)
    x0 = int(max(x0, int(np.floor(cx - r))))
    y0 = int(max(y0, int(np.floor(cy - r))))
    x1 = int(min(x1, int(np.ceil(cx + r))))
    y1 = int(min(y1, int(np.ceil(cy + r))))
    if x1 < x0 or y1 < y0:
        return None
    return x0, y0, x1, y1


def _yfit_grid_xy(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    step: float,
    cx: float,
    cy: float,
    board_lim: float,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    st = max(int(round(float(step))), 1)
    xs = np.arange(int(x0), int(x1) + 1, st, dtype=np.int32)
    ys = np.arange(int(y0), int(y1) + 1, st, dtype=np.int32)
    if xs.size <= 0 or ys.size <= 0:
        z = np.zeros((0,), dtype=np.int32)
        return z, z
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    rr = (xx.astype(np.float32) - float(cx)) ** 2 + (yy.astype(np.float32) - float(cy)) ** 2
    keep = rr <= (float(board_lim) * float(board_lim))
    if mask is not None:
        ix = np.clip(xx, 0, mask.shape[1] - 1)
        iy = np.clip(yy, 0, mask.shape[0] - 1)
        keep = keep & (mask[iy, ix] > 0)
    return xx[keep].astype(np.int32), yy[keep].astype(np.int32)


def _yfit_argmax_theta_sparse(
    wn: np.ndarray,
    u0: np.ndarray,
    v0: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    ct: np.ndarray,
    st: np.ndarray,
    *,
    ray_len: float,
    corridor: float,
    n_bins: int,
    coverage_w: float = 0.65,
    tight_w: float = 0.35,
    balance_pen: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """best_theta(P)=argmax support(P,theta).

    u0/v0 are (N, A) pixel projections, built once per camera/angle-set.
    Inner work is NumPy on (K, N); angles with no v-overlap are skipped.
    Distance-weighted corridor (Gaussian on perpendicular v) + mild L/R
    imbalance penalty. Defaults match experimental full Y-FIT.
    """
    k = int(px.size)
    a = int(ct.size)
    n = int(wn.size)
    best = np.zeros(k, dtype=np.float32)
    best_i = np.zeros(k, dtype=np.int32)
    if k <= 0 or a <= 0 or n <= 0:
        return best, best_i
    sigma = np.float32(max(float(corridor), 1e-6))
    cutoff = np.float32(float(YFIT_DIST_CUTOFF_SIGMA) * float(sigma))
    tau = np.float32(max(float(YFIT_BIN_TAU), 1e-6))
    pen = np.float32(
        float(YFIT_BALANCE_PENALTY) if balance_pen is None else float(balance_pen)
    )
    cov_w = np.float32(float(coverage_w))
    t_w = np.float32(float(tight_w))
    n_bins = max(int(n_bins), 4)
    inv_l = np.float32(float(n_bins) / max(float(ray_len), 1e-6))
    ray_len_f = np.float32(ray_len)
    wn_f = wn.astype(np.float32, copy=False)
    px_f = px.astype(np.float32, copy=False)
    py_f = py.astype(np.float32, copy=False)
    pu = px_f[:, None] * ct[None, :] + py_f[:, None] * st[None, :]
    pv = -px_f[:, None] * st[None, :] + py_f[:, None] * ct[None, :]
    u0_c = u0.astype(np.float32, copy=False)
    v0_c = v0.astype(np.float32, copy=False)
    for ai in range(a):
        vn = v0_c[:, ai]
        pvn = pv[:, ai]
        if float(pvn.max()) < float(vn.min()) - float(cutoff) or float(pvn.min()) > float(vn.max()) + float(cutoff):
            continue
        u = u0_c[None, :, ai] - pu[:, ai][:, None]
        v = vn[None, :] - pvn[:, None]
        near = (u >= 0.0) & (u <= ray_len_f) & (np.abs(v) <= cutoff)
        kk, jj = np.nonzero(near)
        if int(kk.size) <= 0:
            continue
        vv = v[kk, jj]
        cc = wn_f[jj] * np.exp(np.float32(-0.5) * np.square(vv / sigma))
        ww = wn_f[jj]
        num = np.bincount(kk, weights=cc, minlength=k).astype(np.float32)
        den = np.bincount(kk, weights=ww, minlength=k).astype(np.float32)
        tight = num / (den + np.float32(1e-9))
        left_m = vv < 0.0
        right_m = vv > 0.0
        left = np.bincount(kk[left_m], weights=cc[left_m], minlength=k).astype(np.float32)
        right = np.bincount(kk[right_m], weights=cc[right_m], minlength=k).astype(np.float32)
        imb = np.abs(left - right) / (left + right + np.float32(1e-9))
        bb = np.clip((u[kk, jj] * inv_l).astype(np.int32), 0, n_bins - 1)
        mass = np.bincount(kk * n_bins + bb, weights=cc, minlength=k * n_bins)
        acc = mass.reshape(k, n_bins).astype(np.float32, copy=False)
        coverage = (acc / (acc + tau)).mean(axis=1)
        scores = cov_w * coverage + t_w * tight - pen * imb
        np.maximum(scores, 0.0, out=scores)
        better = scores > best
        best[better] = scores[better]
        best_i[better] = ai
    return best, best_i


def _yfit_empty_stats() -> Dict[str, float]:
    return {
        "bbox_x0": -1.0,
        "bbox_y0": -1.0,
        "bbox_x1": -1.0,
        "bbox_y1": -1.0,
        "coarse_P_count": 0.0,
        "coarse_angle_evals": 0.0,
        "coarse_ms": 0.0,
        "fine_P_count": 0.0,
        "fine_angle_evals": 0.0,
        "fine_ms": 0.0,
        "total_yfit_ms": 0.0,
    }


def run_yfit_search(
    layers: List[DartDetectResult],
    *,
    board_center: Tuple[float, float],
    board_r: float,
) -> Tuple[
    Optional[Tuple[float, float]],
    Dict[int, Tuple[float, float, float]],
    float,
    Dict[str, float],
]:
    """Joint P + independent per-cam short-ray search.

    Returns (P, {cam: (vx,vy,support)}, total, profile_stats).
    score(P) = sum_cam argmax_theta support(P, theta). No theta product search.
    """
    t_all = time.perf_counter()
    stats = _yfit_empty_stats()
    ray_len = float(YFIT_RAY_LENGTH_FRAC) * float(board_r)
    corridor = float(YFIT_CORRIDOR_PX)
    n_bins = int(YFIT_U_BINS)
    cx, cy = float(board_center[0]), float(board_center[1])
    size = FITLINE_SIZE
    imgs: Dict[int, np.ndarray] = {}
    union = None
    for r in layers:
        soft = r.fit_ds_gray if r.fit_ds_gray is not None else r.fit_gray
        if soft is None:
            continue
        if soft.shape[0] != size or soft.shape[1] != size:
            soft = cv2.resize(soft, (size, size), interpolation=cv2.INTER_AREA)
        img = _yfit_motion_image(soft)
        imgs[int(r.cam_idx)] = img
        m = (soft > float(FITLINE_WEIGHT_EPS)).astype(np.uint8)
        union = m if union is None else cv2.bitwise_or(union, m)
    if union is None or int(np.count_nonzero(union)) <= 0 or not imgs:
        stats["total_yfit_ms"] = (time.perf_counter() - t_all) * 1000.0
        return None, {}, 0.0, stats

    board_lim = float(board_r) * 1.08
    bbox = _yfit_fused_bbox(
        union,
        margin=int(round(float(YFIT_SEARCH_MARGIN_PX))),
        cx=cx,
        cy=cy,
        board_lim=board_lim,
    )
    if bbox is None:
        stats["total_yfit_ms"] = (time.perf_counter() - t_all) * 1000.0
        return None, {}, 0.0, stats
    x0, y0, x1, y1 = bbox
    stats["bbox_x0"] = float(x0)
    stats["bbox_y0"] = float(y0)
    stats["bbox_x1"] = float(x1)
    stats["bbox_y1"] = float(y1)
    margin = int(round(float(YFIT_SEARCH_MARGIN_PX)))
    kdil = max(3, (margin * 2 + 1) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kdil, kdil))
    near_motion = cv2.dilate(union, kernel, iterations=1)

    bank = _yfit_angle_bank()
    ct_all = bank["ct"]
    st_all = bank["st"]
    rad_all = bank["rad"]
    pix: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    coarse_step_deg = max(int(round(float(YFIT_COARSE_ANGLE_DEG))), 1)
    coarse_idx = np.arange(0, 360, coarse_step_deg, dtype=np.int32)
    n_coarse_ang = int(coarse_idx.size)
    ct_c = ct_all[coarse_idx]
    st_c = st_all[coarse_idx]
    for cam, img in imgs.items():
        ys, xs = np.nonzero(img > 0)
        if int(xs.size) <= 0:
            continue
        xs_f = xs.astype(np.float32)
        ys_f = ys.astype(np.float32)
        wn = img[ys, xs].astype(np.float32, copy=False)
        # One (N, A_coarse) projection table per camera per hit.
        u0c = xs_f[:, None] * ct_c[None, :] + ys_f[:, None] * st_c[None, :]
        v0c = -xs_f[:, None] * st_c[None, :] + ys_f[:, None] * ct_c[None, :]
        pix[int(cam)] = (xs_f, ys_f, wn, u0c, v0c)
    cam_ids = [int(c) for c in sorted(pix.keys())]
    if not cam_ids:
        stats["total_yfit_ms"] = (time.perf_counter() - t_all) * 1000.0
        return None, {}, 0.0, stats

    def _joint(px, py, ang_idx: np.ndarray, *, coarse_table: bool):
        tot = np.zeros(int(px.size), dtype=np.float32)
        loc: Dict[int, np.ndarray] = {}
        sc: Dict[int, np.ndarray] = {}
        ct = ct_all[ang_idx]
        st = st_all[ang_idx]
        col = None
        if coarse_table:
            step = int(ang_idx[1] - ang_idx[0]) if int(ang_idx.size) > 1 else 1
            if step % coarse_step_deg == 0 and int(ang_idx[0]) % coarse_step_deg == 0:
                col = (ang_idx // coarse_step_deg).astype(np.int32)
        for cam in cam_ids:
            xs_f, ys_f, wn, u0c, v0c = pix[cam]
            if col is not None:
                u0 = u0c[:, col]
                v0 = v0c[:, col]
            else:
                u0 = xs_f[:, None] * ct[None, :] + ys_f[:, None] * st[None, :]
                v0 = -xs_f[:, None] * st[None, :] + ys_f[:, None] * ct[None, :]
            s, li = _yfit_argmax_theta_sparse(
                wn, u0, v0, px, py, ct, st,
                ray_len=ray_len, corridor=corridor, n_bins=n_bins,
            )
            sc[cam] = s
            loc[cam] = li
            tot += s
        return tot, loc, sc

    def _set_best(px, py, tot, loc, sc, ang_idx):
        wi = int(np.argmax(tot))
        p = (float(px[wi]), float(py[wi]))
        total = float(tot[wi])
        abs_i: Dict[int, int] = {}
        dirs: Dict[int, Tuple[float, float, float, float]] = {}
        for cam in cam_ids:
            ai = int(ang_idx[int(loc[cam][wi])])
            abs_i[cam] = ai
            dirs[cam] = (
                float(ct_all[ai]),
                float(st_all[ai]),
                float(sc[cam][wi]),
                float(rad_all[ai]),
            )
        return p, total, abs_i, dirs

    t_coarse = time.perf_counter()
    step_c = float(YFIT_COARSE_XY_STEP)
    step_0 = float(max(step_c * 2.0, step_c))
    ang_0 = max(int(coarse_step_deg) * 2, int(coarse_step_deg))
    idx_0 = np.arange(0, 360, ang_0, dtype=np.int32)
    px_0, py_0 = _yfit_grid_xy(
        x0, y0, x1, y1, step_0, cx, cy, board_lim, mask=near_motion,
    )
    n_eval_p = 0
    n_eval_ang = 0
    if int(px_0.size) <= 0:
        stats["coarse_ms"] = (time.perf_counter() - t_coarse) * 1000.0
        stats["total_yfit_ms"] = (time.perf_counter() - t_all) * 1000.0
        return None, {}, 0.0, stats
    tot0, loc0, sc0 = _joint(px_0, py_0, idx_0, coarse_table=True)
    n_eval_p += int(px_0.size)
    n_eval_ang += int(px_0.size) * int(idx_0.size) * len(cam_ids)
    best_p, best_total, best_abs, best_dir = _set_best(px_0, py_0, tot0, loc0, sc0, idx_0)

    win = int(max(round(step_0), round(step_c)))
    cx0 = int(round(best_p[0]))
    cy0 = int(round(best_p[1]))
    px_c, py_c = _yfit_grid_xy(
        max(x0, cx0 - win), max(y0, cy0 - win),
        min(x1, cx0 + win), min(y1, cy0 + win),
        step_c, cx, cy, board_lim, mask=near_motion,
    )
    if int(px_c.size) > 0:
        totc, locc, scc = _joint(px_c, py_c, coarse_idx, coarse_table=True)
        n_eval_p += int(px_c.size)
        n_eval_ang += int(px_c.size) * n_coarse_ang * len(cam_ids)
        p1, t1, a1, d1 = _set_best(px_c, py_c, totc, locc, scc, coarse_idx)
        if t1 >= best_total:
            best_p, best_total, best_abs, best_dir = p1, t1, a1, d1
    stats["coarse_P_count"] = float(n_eval_p)
    stats["coarse_angle_evals"] = float(n_eval_ang)
    stats["coarse_ms"] = (time.perf_counter() - t_coarse) * 1000.0

    fine_span = int(round(float(YFIT_FINE_XY_RADIUS)))
    fx0 = max(x0, int(round(best_p[0])) - fine_span)
    fy0 = max(y0, int(round(best_p[1])) - fine_span)
    fx1 = min(x1, int(round(best_p[0])) + fine_span)
    fy1 = min(y1, int(round(best_p[1])) + fine_span)
    px_f, py_f = _yfit_grid_xy(
        fx0, fy0, fx1, fy1, float(YFIT_FINE_XY_STEP), cx, cy, board_lim
    )
    span_deg = int(round(float(YFIT_FINE_ANGLE_SPAN_DEG)))
    fine_idx_cam: Dict[int, np.ndarray] = {}
    n_fine_ang = 0
    for cam in cam_ids:
        cdeg = int(best_abs[cam]) % 360
        idxs = np.array([(cdeg + d) % 360 for d in range(-span_deg, span_deg + 1)], dtype=np.int32)
        fine_idx_cam[cam] = idxs
        n_fine_ang += int(idxs.size)
    stats["fine_P_count"] = float(px_f.size)
    stats["fine_angle_evals"] = float(int(px_f.size) * n_fine_ang)

    t_fine = time.perf_counter()
    if int(px_f.size) > 0:
        fine_total = np.zeros(int(px_f.size), dtype=np.float32)
        fine_local: Dict[int, np.ndarray] = {}
        fine_scores: Dict[int, np.ndarray] = {}
        for cam in cam_ids:
            idx = fine_idx_cam[cam]
            xs_f, ys_f, wn, _u0c, _v0c = pix[cam]
            ct = ct_all[idx]
            st = st_all[idx]
            u0 = xs_f[:, None] * ct[None, :] + ys_f[:, None] * st[None, :]
            v0 = -xs_f[:, None] * st[None, :] + ys_f[:, None] * ct[None, :]
            s, li = _yfit_argmax_theta_sparse(
                wn, u0, v0, px_f, py_f, ct, st,
                ray_len=ray_len, corridor=corridor, n_bins=n_bins,
            )
            fine_scores[cam] = s
            fine_local[cam] = li
            fine_total += s
        wif = int(np.argmax(fine_total))
        fine_best = float(fine_total[wif])
        if fine_best >= best_total:
            best_total = fine_best
            best_p = (float(px_f[wif]), float(py_f[wif]))
            for cam in cam_ids:
                ai = int(fine_idx_cam[cam][int(fine_local[cam][wif])])
                best_abs[cam] = ai
                best_dir[cam] = (
                    float(ct_all[ai]),
                    float(st_all[ai]),
                    float(fine_scores[cam][wif]),
                    float(rad_all[ai]),
                )
    stats["fine_ms"] = (time.perf_counter() - t_fine) * 1000.0
    stats["total_yfit_ms"] = (time.perf_counter() - t_all) * 1000.0

    out_dir = {
        cam: (float(v[0]), float(v[1]), float(v[2])) for cam, v in best_dir.items()
    }
    return best_p, out_dir, float(best_total), stats


def yfit_to_fused(
    layers: List[DartDetectResult],
    p_xy: Tuple[float, float],
    dirs: Dict[int, Tuple[float, float, float]],
    total: float,
    *,
    board_calibrator: Optional[BoardCalibrator],
    yfit_ms: float,
) -> FusedDartResult:
    px, py = float(p_xy[0]), float(p_xy[1])
    ray_len = float(YFIT_RAY_LENGTH_FRAC) * float(_board_center_and_r(FITLINE_SIZE)[1])
    ring_cals: List[Optional[BoardCalibration]] = []
    per_cam: List[DartDetectResult] = []
    used: List[int] = []
    for r in layers:
        cam = int(r.cam_idx)
        d = dirs.get(cam)
        if board_calibrator is not None:
            ring_cals.append(board_calibrator.get(cam))
        if d is None:
            per_cam.append(r)
            continue
        vx, vy, sup = d
        nn = float(np.hypot(vx, vy)) or 1.0
        vx, vy = vx / nn, vy / nn
        r.found = True
        r.tip_xy = (px, py)
        r.radius_tip = (px, py)
        r.line_vx = float(vx)
        r.line_vy = float(vy)
        r.line_x0 = px
        r.line_y0 = py
        r.line_angle_deg = _fitline_angle_deg(vx, vy)
        r.yfit_support = float(sup)
        r.confidence = float(np.clip(sup, 0.0, 1.0))
        r.reject_reason = ""
        per_cam.append(r)
        used.append(cam)

    number, zone, score = score_topdown_point(
        px, py, out_size=FITLINE_SIZE, cals=ring_cals
    )
    conf = float(np.clip(total / max(len(used), 1), 0.0, 1.0))
    reject = ""
    if score <= 0:
        zone, score, number = "miss", 0, 0
        reject = "miss"
    return FusedDartResult(
        found=True,
        tip_xy=(px, py),
        confidence=conf,
        segment_number=number,
        zone_name=zone,
        score=score,
        cam_indices=used,
        per_cam=per_cam,
        reject_reason=reject,
        stage2_roi_tip_xy=(px, py),
        stage2_roi_radius=ray_len,
        yfit_total=float(total),
        yfit_ms=float(yfit_ms),
    )


def _distance_yfit_enabled() -> bool:
    return DART_TIP_ESTIMATOR == "distance_yfit"


def dyfit_estimate_to_fused(
    layers: List[DartDetectResult],
    est: _dyfit.DistanceYFitEstimate,
    *,
    board_calibrator: Optional[BoardCalibrator],
    mode: str = "distance_yfit",
) -> FusedDartResult:
    dirs: Dict[int, Tuple[float, float, float]] = {}
    for cam, ang in est.per_cam_angle.items():
        vx, vy = _dyfit.ray_direction(ang)
        dirs[int(cam)] = (float(vx), float(vy), float(est.per_cam_score.get(int(cam), 0.0)))
    fused = yfit_to_fused(
        layers,
        (float(est.x), float(est.y)),
        dirs,
        float(est.score),
        board_calibrator=board_calibrator,
        yfit_ms=float(est.runtime_ms),
    )
    conf = float(np.clip(est.score, 0.0, 1.0))
    if len(est.usable_cams) == 2:
        conf *= float(DYFIT_2CAM_CONF_SCALE)
    fused.confidence = conf
    fused.yfit_total = float(est.score)
    fused.yrefine_mode = str(mode)
    fused.yrefine_accepted = True
    fused.yrefine_p = (float(est.x), float(est.y))
    fused.yrefine_p0 = (
        (float(est.coarse_seed[0]), float(est.coarse_seed[1]))
        if est.coarse_seed is not None
        else fused.yrefine_p
    )
    fused.yrefine_dir = {int(c): (float(v[0]), float(v[1])) for c, v in dirs.items()}
    fused.yrefine_scores = {int(c): float(v) for c, v in est.per_cam_score.items()}
    fused.stage2_roi_radius = float(_dyfit.RAY_LENGTH)
    return fused


def _yrefine_wrap_delta_deg(a: float, b: float) -> float:
    return float((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _yrefine_outward_theta_deg(vx: float, vy: float) -> float:
    """Ray from tip along shaft (opposite FitLine tipward orientation)."""
    return float(np.degrees(np.arctan2(-float(vy), -float(vx))) % 360.0)


def _yrefine_cam_seed_theta(r: DartDetectResult) -> float:
    if abs(float(r.stage1_vx)) + abs(float(r.stage1_vy)) > 1e-6:
        return _yrefine_outward_theta_deg(r.stage1_vx, r.stage1_vy)
    return _yrefine_outward_theta_deg(r.line_vx, r.line_vy)


def _yrefine_deg_list(center: float, span: float, step: float) -> np.ndarray:
    step = max(float(step), 0.25)
    span = max(float(span), 0.0)
    n = int(round(2.0 * span / step)) + 1
    n = max(n, 1)
    return (float(center) + np.linspace(-span, span, n, dtype=np.float64)) % 360.0


def _yrefine_local_xy(
    p0: Tuple[float, float],
    radius: float,
    step: float,
    cx: float,
    cy: float,
    board_lim: float,
) -> Tuple[np.ndarray, np.ndarray]:
    r = int(round(float(radius)))
    st = max(int(round(float(step))), 1)
    x0 = int(round(float(p0[0])))
    y0 = int(round(float(p0[1])))
    xs = np.arange(x0 - r, x0 + r + 1, st, dtype=np.int32)
    ys = np.arange(y0 - r, y0 + r + 1, st, dtype=np.int32)
    if xs.size <= 0 or ys.size <= 0:
        z = np.zeros((0,), dtype=np.int32)
        return z, z
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    rr = (xx.astype(np.float32) - float(cx)) ** 2 + (yy.astype(np.float32) - float(cy)) ** 2
    keep = rr <= (float(board_lim) * float(board_lim))
    return xx[keep].astype(np.int32), yy[keep].astype(np.int32)


def _yrefine_spread_tier(spread: float, suspicious: bool) -> str:
    """tight = fast normal hit; mid/wide only when FitLine disagrees."""
    sp = float(spread)
    if sp >= float(YREFINE_SPREAD_WIDE):
        return "wide"
    if sp >= float(YREFINE_SPREAD_TIGHT) or bool(suspicious):
        return "mid"
    return "tight"


def _yrefine_p_radius(tier: str) -> float:
    if tier == "wide":
        return float(YREFINE_P_RADIUS_WIDE)
    if tier == "mid":
        return float(YREFINE_P_RADIUS_MID)
    return float(YREFINE_P_RADIUS_TIGHT)


def _yrefine_max_dp(tier: str) -> float:
    if tier == "wide":
        return float(YREFINE_MAX_DP_WIDE)
    if tier == "mid":
        return float(YREFINE_MAX_DP_MID)
    return float(YREFINE_MAX_DP_TIGHT)


def _yrefine_ang_span(tier: str, problem: bool) -> float:
    if tier == "wide":
        return float(YREFINE_ANG_WIDE_BAD if problem else YREFINE_ANG_WIDE_GOOD)
    if tier == "mid":
        return float(YREFINE_ANG_MID_BAD if problem else YREFINE_ANG_MID_GOOD)
    return float(YREFINE_ANG_TIGHT)


def _yrefine_p_on_edge(
    p: Tuple[float, float],
    origin: Tuple[float, float],
    radius: float,
) -> bool:
    return (abs(float(p[0]) - float(origin[0])) >= float(radius) - 0.51) or (
        abs(float(p[1]) - float(origin[1])) >= float(radius) - 0.51
    )


def _yrefine_expand_radius(old_radius: float, max_dp: float) -> float:
    bumped = float(old_radius) + 8.0
    return min(float(YREFINE_P_RADIUS_EXPAND_CAP), float(max_dp), bumped)


def _yrefine_problem_cams(fused: FusedDartResult) -> set:
    cams = [r for r in fused.per_cam if r.found]
    bad: set = set()
    qs = [float(r.fitq_quality) for r in cams]
    med = float(np.median(qs)) if qs else 0.0
    used = {int(c) for c in fused.cam_indices}
    for r in cams:
        if float(r.fitq_quality) < float(YREFINE_LOW_Q):
            bad.add(int(r.cam_idx))
        if float(r.fitq_s1s2_angle) > float(YREFINE_S1S2_JUMP_DEG):
            bad.add(int(r.cam_idx))
        if qs and float(r.fitq_quality) < med - 0.18:
            bad.add(int(r.cam_idx))
        if used and int(r.cam_idx) not in used:
            bad.add(int(r.cam_idx))
    if int(fused.fitq_recovery_cam) >= 0:
        bad.add(int(fused.fitq_recovery_cam))
    if fused.fitq_suspicious and not bad and cams:
        bad.add(int(min(cams, key=lambda c: float(c.fitq_quality)).cam_idx))
    return bad


def _yrefine_cc_axis(
    xs: np.ndarray, ys: np.ndarray
) -> Tuple[Optional[Tuple[float, float]], Tuple[float, float]]:
    """Principal axis of a CC, or None if the blob is too round/small."""
    if int(xs.size) < 8:
        mx = float(np.mean(xs)) if int(xs.size) else 0.0
        my = float(np.mean(ys)) if int(ys.size) else 0.0
        return None, (mx, my)
    mx = float(np.mean(xs))
    my = float(np.mean(ys))
    x = np.stack([xs.astype(np.float64) - mx, ys.astype(np.float64) - my], axis=1)
    _u, s, vt = np.linalg.svd(x, full_matrices=False)
    if s.size < 2 or float(s[0]) < float(YREFINE_SATELLITE_ELONG_RATIO) * max(
        float(s[1]), 1e-6
    ):
        return None, (mx, my)
    vx, vy = float(vt[0, 0]), float(vt[0, 1])
    n = float(np.hypot(vx, vy)) or 1.0
    return (vx / n, vy / n), (mx, my)


def _yrefine_keep_dart_motion(img: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Keep pixels near the bright dart; drop faint board-wire specks.

    Floor the image *before* CC so dim dots cannot 8-connect onto the shaft.
    Dilate the bright core a few px so dart edges stay. Ray objective unchanged.
    Small bright satellites *beside* the dart (not along the shaft) are punched
    out so they cannot rotate the Y ray parallel to another camera.
    Returns (filtered_img, n_kept, n_all).
    """
    if img is None or img.size == 0:
        z = img if img is not None else np.zeros((0, 0), dtype=np.float32)
        return z, 0, 0
    nz = img > 1e-6
    n_all = int(np.count_nonzero(nz))
    if n_all < 16:
        return img, n_all, n_all
    hi = float(np.percentile(img[nz], float(YREFINE_NOISE_PCTL)))
    core_thr = max(1e-4, float(YREFINE_NOISE_CORE_FRAC) * hi)
    core = (img >= core_thr).astype(np.uint8)
    n_lab, labels, stats, cents = cv2.connectedComponentsWithStats(core, connectivity=8)
    labs = [
        lab
        for lab in range(1, int(n_lab))
        if int(stats[lab, cv2.CC_STAT_AREA]) >= int(YREFINE_NOISE_CORE_MIN_AREA)
    ]
    if not labs:
        return img, n_all, n_all
    largest = max(labs, key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))
    max_area = float(stats[largest, cv2.CC_STAT_AREA])
    ys_l, xs_l = np.where(labels == largest)
    axis, origin = _yrefine_cc_axis(xs_l, ys_l)
    keep_labs = set()
    sat = np.zeros(img.shape[:2], dtype=np.uint8)
    for lab in labs:
        area = float(stats[lab, cv2.CC_STAT_AREA])
        if lab == largest or axis is None or area >= float(YREFINE_SATELLITE_AREA_FRAC) * max_area:
            keep_labs.add(lab)
            continue
        cx, cy = float(cents[lab, 0]), float(cents[lab, 1])
        dx, dy = cx - origin[0], cy - origin[1]
        vx, vy = axis
        perp = abs(dx * (-vy) + dy * vx)
        if perp <= float(YREFINE_SATELLITE_PERP_PX):
            keep_labs.add(lab)
        else:
            sat[labels == lab] = 255
    core_keep = np.zeros(img.shape[:2], dtype=np.uint8)
    for lab in keep_labs:
        core_keep[labels == lab] = 255
    if int(np.count_nonzero(core_keep)) < 6:
        return img, n_all, n_all
    rad = max(0, int(YREFINE_NOISE_DILATE_PX))
    if rad > 0:
        k = 2 * rad + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        core_keep = cv2.dilate(core_keep, kernel, iterations=1)
    if int(np.count_nonzero(sat)) > 0:
        core_keep[sat > 0] = 0
    keep = np.where(core_keep > 0, img, 0).astype(np.float32, copy=False)
    n_keep = int(np.count_nonzero(keep > 1e-6))
    if n_keep < max(12, n_all // 8):
        return img, n_all, n_all
    return keep, n_keep, n_all


def _yrefine_pix_from_cam(r: DartDetectResult) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    soft = r.fit_ds_gray if r.fit_ds_gray is not None else r.fit_gray
    if soft is None:
        return None
    if soft.shape[0] != FITLINE_SIZE or soft.shape[1] != FITLINE_SIZE:
        soft = cv2.resize(soft, (FITLINE_SIZE, FITLINE_SIZE), interpolation=cv2.INTER_AREA)
    img = _yfit_motion_image(soft)
    img, n_keep, n_all = _yrefine_keep_dart_motion(img)
    if n_keep < n_all:
        print(
            f"[YREFINE] denoise cam{int(r.cam_idx)} kept={n_keep}/{n_all}",
            flush=True,
        )
    ys, xs = np.nonzero(img > 0)
    if int(xs.size) <= 0:
        return None
    return (
        xs.astype(np.float32),
        ys.astype(np.float32),
        img[ys, xs].astype(np.float32, copy=False),
    )


def _yrefine_score_grid(
    pix: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    px: np.ndarray,
    py: np.ndarray,
    ang: Dict[int, np.ndarray],
    *,
    ray_len: float,
    corridor: float,
    n_bins: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    loc: Dict[int, np.ndarray] = {}
    sc: Dict[int, np.ndarray] = {}
    for cam, (xs, ys, wn) in pix.items():
        deg = ang[cam]
        rad = np.deg2rad(deg)
        ct = np.cos(rad).astype(np.float32)
        st = np.sin(rad).astype(np.float32)
        u0 = xs[:, None] * ct[None, :] + ys[:, None] * st[None, :]
        v0 = -xs[:, None] * st[None, :] + ys[:, None] * ct[None, :]
        s, li = _yfit_argmax_theta_sparse(
            wn, u0, v0, px, py, ct, st,
            ray_len=ray_len, corridor=corridor, n_bins=n_bins,
            coverage_w=float(YREFINE_COV_W),
            tight_w=float(YREFINE_TIGHT_W),
        )
        sc[cam] = s
        loc[cam] = li
    return loc, sc


def _yrefine_alive_cams(sc: Dict[int, np.ndarray]) -> List[int]:
    cams = sorted(sc.keys())
    alive = [
        c for c in cams
        if sc[c].size > 0 and float(np.max(sc[c])) >= float(YREFINE_ALIVE_SUPPORT)
    ]
    if len(alive) < 2:
        return cams
    return alive


def _yrefine_objectives(sc: Dict[int, np.ndarray]) -> Dict[str, np.ndarray]:
    """Global P scores from per-camera supports. No FitLine distance term.

    sum: old behaviour (one strong cam can dominate).
    mean / min / geom: comparison objectives on live cameras.
    consensus: 0.5*mean + 0.5*min. A cam with no support anywhere in this
    grid is dropped from min/mean/geom/consensus so it cannot flatten min to 0.
    """
    z = np.zeros((0,), dtype=np.float32)
    empty = {k: z for k in ("sum", "mean", "min", "geom", "consensus")}
    if not sc:
        return empty
    cams = sorted(sc.keys())
    stack = np.stack([sc[c] for c in cams], axis=0).astype(np.float32, copy=False)
    alive = _yrefine_alive_cams(sc)
    live = np.stack([sc[c] for c in alive], axis=0).astype(np.float32, copy=False)
    mean = live.mean(axis=0)
    mn = live.min(axis=0)
    geom = np.exp(np.mean(np.log(np.clip(live, 1e-6, None)), axis=0)).astype(np.float32)
    return {
        "sum": stack.sum(axis=0),
        "mean": mean.astype(np.float32, copy=False),
        "min": mn.astype(np.float32, copy=False),
        "geom": geom,
        "consensus": (0.5 * mean + 0.5 * mn).astype(np.float32),
    }


def _yrefine_scores_at(sc: Dict[int, np.ndarray], i: int, cams: Optional[List[int]] = None) -> List[float]:
    keys = cams if cams is not None else sorted(sc.keys())
    return [float(sc[c][i]) for c in keys]


def _yrefine_obj_scalars(scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {k: 0.0 for k in ("sum", "mean", "min", "geom", "consensus")}
    a = np.asarray(scores, dtype=np.float64)
    mean = float(a.mean())
    mn = float(a.min())
    geom = float(np.exp(np.mean(np.log(np.clip(a, 1e-6, None)))))
    return {
        "sum": float(a.sum()),
        "mean": mean,
        "min": mn,
        "geom": geom,
        "consensus": 0.5 * mean + 0.5 * mn,
    }


def _yrefine_pick(
    obj_name: str,
    agg: Dict[str, np.ndarray],
    loc: Dict[int, np.ndarray],
    sc: Dict[int, np.ndarray],
    ang: Dict[int, np.ndarray],
    px: np.ndarray,
    py: np.ndarray,
) -> Tuple[Tuple[float, float], Dict[int, float], Dict[int, float], Dict[str, float]]:
    key = obj_name if obj_name in agg and int(agg[obj_name].size) > 0 else "consensus"
    arr = agg.get(key)
    if arr is None or int(arr.size) <= 0:
        return (0.0, 0.0), {}, {}, _yrefine_obj_scalars([])
    wi = int(np.argmax(arr))
    th: Dict[int, float] = {}
    sc_p: Dict[int, float] = {}
    for cam in sc:
        th[cam] = float(ang[cam][int(loc[cam][wi])])
        sc_p[cam] = float(sc[cam][wi])
    return (
        (float(px[wi]), float(py[wi])),
        th,
        sc_p,
        _yrefine_obj_scalars(_yrefine_scores_at(sc, wi, _yrefine_alive_cams(sc))),
    )


def _yrefine_fmt_list(vals: List[float]) -> str:
    return "[" + ", ".join(f"{v:.2f}" for v in vals) + "]"


def _yrefine_log_obj_picks(
    tag: str,
    agg: Dict[str, np.ndarray],
    sc: Dict[int, np.ndarray],
    px: np.ndarray,
    py: np.ndarray,
) -> None:
    for name in ("sum", "mean", "min", "geom", "consensus"):
        arr = agg.get(name)
        if arr is None or int(arr.size) <= 0:
            continue
        i = int(np.argmax(arr))
        scores = _yrefine_scores_at(sc, i)
        print(
            f"[YREFINE][obj] {tag} {name} "
            f"P=({float(px[i]):.1f},{float(py[i]):.1f}) "
            f"scores={_yrefine_fmt_list(scores)} "
            f"mean={float(agg['mean'][i]):.3f} min={float(agg['min'][i]):.3f} "
            f"geom={float(agg['geom'][i]):.3f} consensus={float(agg['consensus'][i]):.3f} "
            f"sum={float(agg['sum'][i]):.3f}",
            flush=True,
        )


def _yrefine_coarse_fine(
    pix: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    p_center: Tuple[float, float],
    p_rad: float,
    coarse_ang: Dict[int, np.ndarray],
    *,
    ray_len: float,
    corridor: float,
    n_bins: int,
    cx: float,
    cy: float,
    board_lim: float,
    obj_name: str,
    log_tag: str,
    origin: Optional[Tuple[float, float]] = None,
    max_dp: Optional[float] = None,
) -> Optional[dict]:
    px_c, py_c = _yrefine_local_xy(
        p_center, p_rad, float(YREFINE_COARSE_XY_STEP), cx, cy, board_lim
    )
    if origin is not None and max_dp is not None and int(px_c.size) > 0:
        dist = np.hypot(
            px_c.astype(np.float64) - float(origin[0]),
            py_c.astype(np.float64) - float(origin[1]),
        )
        keep = dist <= float(max_dp) + 0.51
        px_c = px_c[keep]
        py_c = py_c[keep]
    coarse_n = int(px_c.size)
    if coarse_n <= 0:
        return None
    loc_c, sc_c = _yrefine_score_grid(
        pix, px_c, py_c, coarse_ang,
        ray_len=ray_len, corridor=corridor, n_bins=n_bins,
    )
    agg_c = _yrefine_objectives(sc_c)
    if log_tag:
        _yrefine_log_obj_picks(log_tag, agg_c, sc_c, px_c, py_c)
    p_sum_c, th_sum_c, sc_sum_c, obj_sum_c = _yrefine_pick(
        "sum", agg_c, loc_c, sc_c, coarse_ang, px_c, py_c
    )
    best_p, best_th, best_sc, best_obj = _yrefine_pick(
        obj_name, agg_c, loc_c, sc_c, coarse_ang, px_c, py_c
    )
    best_total = float(best_obj.get(obj_name, best_obj["consensus"]))
    px_f, py_f = _yrefine_local_xy(
        best_p, float(YREFINE_FINE_XY_RADIUS), float(YREFINE_FINE_XY_STEP),
        cx, cy, board_lim,
    )
    fine_n = int(px_f.size)
    fine_ang: Dict[int, np.ndarray] = {}
    n_fine_ang = 0
    for cam in pix:
        fine_ang[cam] = _yrefine_deg_list(
            best_th[cam], float(YREFINE_FINE_ANG_SPAN), float(YREFINE_FINE_ANG_STEP)
        )
        n_fine_ang += int(fine_ang[cam].size)
    if fine_n > 0:
        loc_f, sc_f = _yrefine_score_grid(
            pix, px_f, py_f, fine_ang,
            ray_len=ray_len, corridor=corridor, n_bins=n_bins,
        )
        agg_f = _yrefine_objectives(sc_f)
        if log_tag:
            _yrefine_log_obj_picks(log_tag + "_fine", agg_f, sc_f, px_f, py_f)
        p_f, th_f, sc_f_p, obj_f = _yrefine_pick(
            obj_name, agg_f, loc_f, sc_f, fine_ang, px_f, py_f
        )
        if float(obj_f.get(obj_name, obj_f["consensus"])) >= best_total:
            best_total = float(obj_f.get(obj_name, obj_f["consensus"]))
            best_p, best_th, best_sc, best_obj = p_f, th_f, sc_f_p, obj_f
    n_ang = sum(int(a.size) for a in coarse_ang.values())
    return {
        "best_p": best_p,
        "best_th": best_th,
        "best_sc": best_sc,
        "best_obj": best_obj,
        "p_sum": p_sum_c,
        "sc_sum": sc_sum_c,
        "obj_sum": obj_sum_c,
        "th_sum": th_sum_c,
        "coarse_n": coarse_n,
        "fine_n": fine_n,
        "n_ang": n_ang,
        "sc_c": sc_c,
        "alive": _yrefine_alive_cams(sc_c),
    }


def _fitline_cam_flee_px(
    fused: FusedDartResult,
) -> List[Tuple[int, float]]:
    """Perpendicular distance of each used FitLine from the fused tip."""
    tx, ty = float(fused.tip_xy[0]), float(fused.tip_xy[1])
    used = {int(c) for c in fused.cam_indices} if fused.cam_indices else None
    out: List[Tuple[int, float]] = []
    for r in fused.per_cam:
        if not r.found:
            continue
        if used is not None and int(r.cam_idx) not in used:
            continue
        x0, y0, vx, vy = _fitline_original_xyv(r)
        d = _point_to_line_distance(x0, y0, vx, vy, tx, ty)
        out.append((int(r.cam_idx), float(d)))
    return out


def _fitline_needs_yrefine(fused: FusedDartResult) -> Tuple[bool, float, str]:
    """True when FitLine concurrence is untrustworthy.

    Two lines always meet at a point, so 2-cam cannot measure flee — always
    refine. 3+ cams: only if a used FitLine misses the fused tip.
    """
    flees = _fitline_cam_flee_px(fused)
    if not flees:
        return False, 0.0, ""
    max_flee = max(d for _c, d in flees)
    detail = "[" + ", ".join(f"{c}:{d:.2f}" for c, d in flees) + "]"
    if len(flees) < 2:
        return False, float(max_flee), detail
    if len(flees) < 3:
        return True, float(max_flee), detail
    need = max_flee >= float(YREFINE_FLEE_PX)
    return need, float(max_flee), detail


def run_yrefine_local(
    fused: FusedDartResult,
    *,
    board_calibrator: Optional[BoardCalibrator],
) -> FusedDartResult:
    """Local Y-refine around FitLine P0. Y P is final when valid; else FitLine."""
    t0 = time.perf_counter()
    p0 = (float(fused.tip_xy[0]), float(fused.tip_xy[1]))
    fused.yrefine_p0 = p0
    fused.yrefine_p = p0
    fused.yrefine_seed_dir = {
        int(r.cam_idx): (float(r.line_vx), float(r.line_vy))
        for r in fused.per_cam
        if r.found
    }
    need, max_flee, detail = _fitline_needs_yrefine(fused)
    n_lines = len(_fitline_cam_flee_px(fused))
    if not need:
        fused.yrefine_mode = "skip"
        fused.yfit_ms = (time.perf_counter() - t0) * 1000.0
        print(
            f"[YREFINE] skip good_intersect n={n_lines} max_flee={max_flee:.2f} "
            f"cams={detail or '-'} thresh={float(YREFINE_FLEE_PX):.1f} — keep FitLine",
            flush=True,
        )
        return fused
    why = "2cam" if n_lines < 3 else "flee"
    print(
        f"[YREFINE] run {why} n={n_lines} max_flee={max_flee:.2f} cams={detail} "
        f"thresh={float(YREFINE_FLEE_PX):.1f}",
        flush=True,
    )
    try:
        return _run_yrefine_local_body(fused, board_calibrator=board_calibrator, t0=t0, p0=p0)
    except Exception as exc:
        fused.yrefine_mode = "fail"
        fused.yfit_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[YREFINE] fail {exc!r} — keep FitLine", flush=True)
        return fused


def _run_yrefine_local_body(
    fused: FusedDartResult,
    *,
    board_calibrator: Optional[BoardCalibrator],
    t0: float,
    p0: Tuple[float, float],
) -> FusedDartResult:
    line_cams = [r for r in fused.per_cam if r.found]
    if len(line_cams) < 2:
        fused.yfit_ms = (time.perf_counter() - t0) * 1000.0
        fused.yrefine_mode = "skip"
        print("[YREFINE] skip cams<2 — keep FitLine", flush=True)
        return fused

    pix: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for r in line_cams:
        got = _yrefine_pix_from_cam(r)
        if got is not None:
            pix[int(r.cam_idx)] = got
    if len(pix) < 2:
        fused.yfit_ms = (time.perf_counter() - t0) * 1000.0
        fused.yrefine_mode = "skip"
        print("[YREFINE] skip no_motion — keep FitLine", flush=True)
        return fused

    problem = _yrefine_problem_cams(fused)
    suspicious = bool(fused.fitq_suspicious) or bool(problem)
    spread = float(fused.fitq_pair_spread)
    tier = _yrefine_spread_tier(spread, suspicious)
    mode = "suspicious" if (suspicious or tier != "tight") else "normal"
    fused.yrefine_mode = mode
    p_rad = _yrefine_p_radius(tier)
    max_dp = _yrefine_max_dp(tier)
    fused.yrefine_search_radius = float(p_rad)
    cam_span: Dict[int, float] = {}
    for r in line_cams:
        cam = int(r.cam_idx)
        if cam not in pix:
            continue
        cam_span[cam] = _yrefine_ang_span(tier, cam in problem)

    ray_len = float(YFIT_RAY_LENGTH_FRAC) * float(_board_center_and_r(FITLINE_SIZE)[1])
    corridor = float(YFIT_CORRIDOR_PX)
    n_bins = int(YFIT_U_BINS)
    (cx, cy), board_r = _board_center_and_r(FITLINE_SIZE)
    board_lim = float(board_r) * 1.08

    seed_th: Dict[int, float] = {}
    p0x = np.array([p0[0]], dtype=np.float32)
    p0y = np.array([p0[1]], dtype=np.float32)
    for r in line_cams:
        cam = int(r.cam_idx)
        if cam not in pix:
            continue
        th_a = _yrefine_cam_seed_theta(r)
        th_b = (th_a + 180.0) % 360.0
        flip = np.array([th_a, th_b], dtype=np.float64)
        loc, sc = _yrefine_score_grid(
            {cam: pix[cam]}, p0x, p0y, {cam: flip},
            ray_len=ray_len, corridor=corridor, n_bins=n_bins,
        )
        seed_th[cam] = float(flip[int(loc[cam][0])])

    seed_ang = {cam: np.array([seed_th[cam]], dtype=np.float64) for cam in seed_th}
    _seed_loc, seed_sc_grid = _yrefine_score_grid(
        pix, p0x, p0y, seed_ang,
        ray_len=ray_len, corridor=corridor, n_bins=n_bins,
    )
    seed_scores = _yrefine_scores_at(seed_sc_grid, 0) if seed_sc_grid else []
    seed_alive = (
        _yrefine_scores_at(seed_sc_grid, 0, _yrefine_alive_cams(seed_sc_grid))
        if seed_sc_grid
        else []
    )
    seed_obj = _yrefine_obj_scalars(seed_alive)

    coarse_ang: Dict[int, np.ndarray] = {}
    n_ang_eval = 0
    for cam, span in cam_span.items():
        if cam not in seed_th:
            continue
        coarse_ang[cam] = _yrefine_deg_list(
            seed_th[cam], span, float(YREFINE_COARSE_ANG_STEP)
        )
        n_ang_eval += int(coarse_ang[cam].size)
    pix = {c: pix[c] for c in coarse_ang if c in pix}
    if len(pix) < 2 or not coarse_ang:
        fused.yfit_ms = (time.perf_counter() - t0) * 1000.0
        fused.yrefine_mode = "skip"
        print("[YREFINE] skip no_angles — keep FitLine", flush=True)
        return fused

    obj_name = str(YREFINE_OBJECTIVE)
    t_c = time.perf_counter()
    got = _yrefine_coarse_fine(
        pix, p0, p_rad, coarse_ang,
        ray_len=ray_len, corridor=corridor, n_bins=n_bins,
        cx=cx, cy=cy, board_lim=board_lim, obj_name=obj_name, log_tag="coarse",
        origin=p0, max_dp=max_dp,
    )
    if got is None:
        fused.yfit_ms = (time.perf_counter() - t0) * 1000.0
        print("[YREFINE] skip empty_grid — keep FitLine", flush=True)
        return fused
    sc_c0 = got["sc_c"]
    print(
        f"[YREFINE] alive_cams={got['alive']} "
        f"max_support=[{', '.join(f'{c}:{float(np.max(sc_c0[c])):.2f}' for c in sorted(sc_c0))}]",
        flush=True,
    )
    coarse_ms = (time.perf_counter() - t_c) * 1000.0
    best_p = got["best_p"]
    best_th = got["best_th"]
    best_sc = got["best_sc"]
    best_obj = got["best_obj"]
    p_sum = got["p_sum"]
    sc_sum_p = got["sc_sum"]
    obj_sum = got["obj_sum"]
    coarse_n = int(got["coarse_n"])
    fine_n = int(got["fine_n"])
    coarse_ang_evals = coarse_n * n_ang_eval
    fine_ang_evals = fine_n * int(got["n_ang"])
    expanded = False
    edge_before = _yrefine_p_on_edge(best_p, p0, p_rad)
    edge_after = edge_before
    used_radius = float(p_rad)

    # One expansion pass: edge + (suspicious or large spread). Never on tight/fast hits.
    if edge_before and tier != "tight":
        new_rad = _yrefine_expand_radius(p_rad, max_dp)
        if new_rad > p_rad + 0.51:
            old_p = best_p
            old_score = float(best_obj["consensus"])
            t_e = time.perf_counter()
            got_e = _yrefine_coarse_fine(
                pix, p0, new_rad, coarse_ang,
                ray_len=ray_len, corridor=corridor, n_bins=n_bins,
                cx=cx, cy=cy, board_lim=board_lim, obj_name=obj_name,
                log_tag="expand",
                origin=p0, max_dp=max_dp,
            )
            expand_ms = (time.perf_counter() - t_e) * 1000.0
            if got_e is not None:
                new_p = got_e["best_p"]
                new_score = float(got_e["best_obj"]["consensus"])
                edge_after = _yrefine_p_on_edge(new_p, p0, new_rad)
                print(
                    f"[YREFINE_EXPAND] old_radius={p_rad:.1f} new_radius={new_rad:.1f} "
                    f"old_best_P=({old_p[0]:.1f},{old_p[1]:.1f}) "
                    f"new_best_P=({new_p[0]:.1f},{new_p[1]:.1f}) "
                    f"old_score={old_score:.3f} new_score={new_score:.3f} "
                    f"edge_before={int(edge_before)} edge_after={int(edge_after)} "
                    f"expand_ms={expand_ms:.1f}",
                    flush=True,
                )
                if new_score + 1e-9 >= old_score:
                    best_p = new_p
                    best_th = got_e["best_th"]
                    best_sc = got_e["best_sc"]
                    best_obj = got_e["best_obj"]
                    p_sum = got_e["p_sum"]
                    sc_sum_p = got_e["sc_sum"]
                    obj_sum = got_e["obj_sum"]
                    coarse_n = int(got_e["coarse_n"])
                    fine_n = int(got_e["fine_n"])
                    coarse_ang_evals += coarse_n * n_ang_eval
                    fine_ang_evals += fine_n * int(got_e["n_ang"])
                    used_radius = float(new_rad)
                    expanded = True
                    fused.yrefine_search_radius = float(new_rad)
            else:
                print(
                    f"[YREFINE_EXPAND] old_radius={p_rad:.1f} new_radius={new_rad:.1f} "
                    f"empty_grid expand_ms={expand_ms:.1f}",
                    flush=True,
                )

    fine_ms = (time.perf_counter() - t_c) * 1000.0 - coarse_ms
    total_ms = (time.perf_counter() - t0) * 1000.0
    fused.yfit_ms = float(total_ms)
    fused.yfit_total = float(best_obj["consensus"])
    fused.yrefine_p = best_p
    fused.yrefine_expanded = bool(expanded)
    fused.yrefine_dir = {}
    fused.yrefine_scores = {}
    for cam, th in best_th.items():
        rad = np.deg2rad(th)
        vx, vy = float(np.cos(rad)), float(np.sin(rad))
        nn = float(np.hypot(vx, vy)) or 1.0
        fused.yrefine_dir[int(cam)] = (vx / nn, vy / nn)
        fused.yrefine_scores[int(cam)] = float(best_sc.get(cam, 0.0))

    dpx = best_p[0] - p0[0]
    dpy = best_p[1] - p0[1]
    delta_p = float(np.hypot(dpx, dpy))
    ang_deltas = []
    absurd = False
    for cam in sorted(best_th.keys()):
        dth = abs(_yrefine_wrap_delta_deg(best_th[cam], seed_th[cam]))
        ang_deltas.append(dth)
        if dth > float(YREFINE_ABSURD_ANG_DEG):
            absurd = True
    min_cam = float(YREFINE_MIN_CAM_SUSP if (suspicious or tier != "tight") else YREFINE_MIN_CAM_NORMAL)
    scores = [float(best_sc.get(c, 0.0)) for c in sorted(best_th.keys())]
    on_edge = _yrefine_p_on_edge(best_p, p0, used_radius)
    fused.yrefine_on_edge = bool(on_edge)
    n_ok = sum(1 for s in scores if s >= min_cam)
    weak = n_ok < 2
    new_cons = float(best_obj["consensus"])
    seed_cons = float(seed_obj["consensus"])
    improved = new_cons > seed_cons
    too_far = delta_p > float(max_dp) + 1e-6
    # FitLine is seed only. Y P is final whenever the refine is valid.
    # Fallback to FitLine only if the result is unusable (too few cams / no support).
    valid = (not weak) and math.isfinite(float(best_p[0])) and math.isfinite(float(best_p[1]))
    accept = bool(valid)
    print(
        f"[YREFINE] mode={mode} tier={tier} spread={spread:.1f} "
        f"P_radius={used_radius:.1f} max_dP={max_dp:.1f} "
        f"P0=({p0[0]:.1f},{p0[1]:.1f}) P=({best_p[0]:.1f},{best_p[1]:.1f}) "
        f"coarse_P_count={coarse_n} fine_P_count={fine_n} "
        f"angle_evals={coarse_ang_evals + fine_ang_evals} "
        f"coarse_ms={coarse_ms:.1f} fine_ms={fine_ms:.1f} total_ms={total_ms:.1f} "
        f"delta_P={delta_p:.2f} "
        f"angle_delta=[{', '.join(f'{d:.1f}' for d in ang_deltas)}] "
        f"scores={_yrefine_fmt_list(scores)} n_ok={n_ok} "
        f"valid={int(valid)} accept={int(accept)} edge={int(on_edge)} expanded={int(expanded)} "
        f"too_far={int(too_far)} weak={int(weak)} improved={int(improved)} "
        f"absurd={int(absurd)}",
        flush=True,
    )
    print(
        f"[YREFINE] "
        f"seed_scores={_yrefine_fmt_list(seed_scores)} "
        f"new_scores={_yrefine_fmt_list(scores)} "
        f"seed_mean={seed_obj['mean']:.3f} new_mean={best_obj['mean']:.3f} "
        f"seed_min={seed_obj['min']:.3f} new_min={best_obj['min']:.3f} "
        f"seed_geom={seed_obj['geom']:.3f} new_geom={best_obj['geom']:.3f} "
        f"seed_consensus={seed_cons:.3f} new_consensus={new_cons:.3f} "
        f"dP={delta_p:.2f} valid={int(valid)} accept={int(accept)}",
        flush=True,
    )
    print(
        f"[YREFINE][cmp] OLD sum P=({p_sum[0]:.1f},{p_sum[1]:.1f}) "
        f"scores={_yrefine_fmt_list([sc_sum_p[c] for c in sorted(sc_sum_p)])} "
        f"consensus={obj_sum['consensus']:.3f} sum={obj_sum['sum']:.3f}",
        flush=True,
    )
    print(
        f"[YREFINE][cmp] NEW consensus P=({best_p[0]:.1f},{best_p[1]:.1f}) "
        f"scores={_yrefine_fmt_list(scores)} "
        f"consensus={new_cons:.3f} sum={best_obj['sum']:.3f}",
        flush=True,
    )
    print(
        f"[YREFINE] old_ang=[{', '.join(f'{seed_th[c]:.1f}' for c in sorted(seed_th))}] "
        f"new_ang=[{', '.join(f'{best_th[c]:.1f}' for c in sorted(best_th))}]",
        flush=True,
    )

    if not accept:
        fused.yrefine_accepted = False
        print("[YREFINE] invalid — keep FitLine", flush=True)
        return fused

    px, py = best_p
    fused.yrefine_accepted = True
    fused.tip_xy = (px, py)
    fused.stage2_roi_tip_xy = (px, py)
    ring_cals: List[Optional[BoardCalibration]] = []
    for r in fused.per_cam:
        cam = int(r.cam_idx)
        if board_calibrator is not None:
            ring_cals.append(board_calibrator.get(cam))
        if cam not in best_th:
            continue
        th = best_th[cam]
        rad = np.deg2rad(th)
        vx, vy = float(np.cos(rad)), float(np.sin(rad))
        nn = float(np.hypot(vx, vy)) or 1.0
        vx, vy = vx / nn, vy / nn
        r.tip_xy = (px, py)
        r.radius_tip = (px, py)
        r.line_x0 = px
        r.line_y0 = py
        r.line_vx = vx
        r.line_vy = vy
        r.line_angle_deg = _fitline_angle_deg(vx, vy)
        r.yfit_support = float(best_sc.get(cam, 0.0))
    number, zone, score = score_topdown_point(
        px, py, out_size=FITLINE_SIZE, cals=ring_cals
    )
    fused.segment_number = int(number)
    fused.zone_name = str(zone)
    fused.score = int(score)
    if score <= 0:
        fused.zone_name, fused.score, fused.segment_number = "miss", 0, 0
        fused.reject_reason = "miss"
    return fused


def save_yrefine_debug(
    fused: FusedDartResult,
    empty_refs: Dict[int, np.ndarray],
    *,
    raw_frames: Optional[Dict[int, np.ndarray]] = None,
    board_calibrator: Optional[BoardCalibrator] = None,
    out_size: int = FITLINE_SIZE,
) -> Optional[str]:
    """Scoring-coordinate canvas + dim FitLine + Y rays + FIT/Y/FINAL points."""
    if not _hit_motion_debug_enabled() or not fused.found:
        return None
    _ = empty_refs
    _ = raw_frames
    per_cam = {int(r.cam_idx): r for r in fused.per_cam if r.found}
    size = int(out_size) if out_size > 0 else FITLINE_SIZE
    cals = _score_cals_for_debug(board_calibrator)
    p0x, p0y = float(fused.yrefine_p0[0]), float(fused.yrefine_p0[1])
    if p0x == 0.0 and p0y == 0.0:
        p0x, p0y = float(fused.tip_xy[0]), float(fused.tip_xy[1])
    ypx, ypy = float(fused.yrefine_p[0]), float(fused.yrefine_p[1])
    if ypx == 0.0 and ypy == 0.0:
        ypx, ypy = float(fused.tip_xy[0]), float(fused.tip_xy[1])
    fx, fy = float(fused.tip_xy[0]), float(fused.tip_xy[1])
    canvas, meta = make_scoring_board_canvas(
        out_size=size,
        cals=cals,
        near_points=[(p0x, p0y), (ypx, ypy), (fx, fy)],
    )
    rings: Dict[str, float] = meta["rings"]  # type: ignore[assignment]
    ray = float(rings.get("double_outer", _board_center_and_r(size)[1]))

    by_cam = per_cam
    for cam, (vx, vy) in fused.yrefine_seed_dir.items():
        r = by_cam.get(int(cam))
        ox, oy = _dir_original_unit(float(vx), float(vy), r)
        color = _hit_debug_cam_color(int(cam))
        dim = (int(color[0] * 0.35), int(color[1] * 0.35), int(color[2] * 0.35))
        _draw_fitline_on(canvas, p0x, p0y, ox, oy, dim, thickness=1)

    dirs = fused.yrefine_dir if fused.yrefine_dir else {
        int(r.cam_idx): (float(r.line_vx), float(r.line_vy))
        for r in fused.per_cam
        if r.found
    }
    for cam, (vx, vy) in dirs.items():
        r = by_cam.get(int(cam))
        ox, oy = _dir_original_unit(float(vx), float(vy), r)
        color = _hit_debug_cam_color(int(cam))
        x1 = int(round(ypx + ox * ray))
        y1 = int(round(ypy + oy * ray))
        x0 = int(round(ypx - ox * ray * 0.12))
        y0 = int(round(ypy - oy * ray * 0.12))
        cv2.line(canvas, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
        sc = float(fused.yrefine_scores.get(int(cam), 0.0))
        lx = int(round(ypx + ox * 36.0))
        ly = int(round(ypy + oy * 36.0))
        cv2.putText(
            canvas,
            f"cam{int(cam)} s={sc:.2f}",
            (lx, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            color,
            1,
            cv2.LINE_AA,
        )

    _draw_marked_point(canvas, (p0x, p0y), (200, 200, 200), radius=3, cross=8)
    _label_xy(canvas, (p0x, p0y), "FIT", (200, 200, 200), dx=8, dy=-12)
    ycol = (0, 255, 255) if fused.yrefine_accepted else (0, 140, 255)
    _draw_marked_point(canvas, (ypx, ypy), ycol, radius=3, cross=8)
    _label_xy(canvas, (ypx, ypy), "Y", ycol, dx=8, dy=14)
    cv2.circle(
        canvas,
        (int(round(fx)), int(round(fy))),
        7,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    if abs(fx - ypx) + abs(fy - ypy) > 1.5 or abs(fx - p0x) + abs(fy - p0y) > 1.5:
        _label_xy(canvas, (fx, fy), "FINAL", (0, 255, 0), dx=-36, dy=-12)

    fit_lab = _hit_score_label(p0x, p0y, out_size=size, cals=cals)
    y_lab = _hit_score_label(ypx, ypy, out_size=size, cals=cals)
    final_lab = _zone_short_label(fused.zone_name, int(fused.segment_number))
    hud = [
        f"FIT ({p0x:.1f},{p0y:.1f}) {fit_lab}",
        f"Y   ({ypx:.1f},{ypy:.1f}) {y_lab}",
        f"FINAL {final_lab}",
        f"dP={float(np.hypot(ypx - p0x, ypy - p0y)):.2f} "
        f"accept={int(fused.yrefine_accepted)} {fused.yrefine_mode}",
    ]
    near_wires: Dict[int, float] = meta["near_wires"]  # type: ignore[assignment]
    if near_wires:
        wi, dist = min(near_wires.items(), key=lambda kv: kv[1])
        hud.append(f"wire {_wire_pair_label(wi)}  {dist:.2f}deg")
    _draw_hud_lines(canvas, hud)

    print(
        f"FIT ({p0x:.1f},{p0y:.1f}) {fit_lab}  "
        f"Y ({ypx:.1f},{ypy:.1f}) {y_lab}  FINAL {final_lab}",
        flush=True,
    )
    stamp = time.strftime("%H%M%S")
    ms = int((time.time() % 1) * 1000)
    label = _zone_short_label(fused.zone_name, fused.segment_number)
    stem = f"hit_{stamp}_{ms:03d}_{label}"
    motion_dir = os.path.join(_debug_dir_path(), "motion")
    try:
        os.makedirs(motion_dir, exist_ok=True)
    except OSError:
        motion_dir = _debug_dir_path()
    path = os.path.join(motion_dir, f"{stem}_yrefine.png")
    _safe_imwrite(path, canvas)
    print(f"[yrefine] debug {path}", flush=True)
    return path


def save_yfit_debug(
    fused: FusedDartResult,
    empty_refs: Dict[int, np.ndarray],
    *,
    raw_frames: Optional[Dict[int, np.ndarray]] = None,
    board_calibrator: Optional[BoardCalibrator] = None,
    out_size: int = FITLINE_SIZE,
) -> Optional[str]:
    """Scoring-coordinate canvas + shared P + short per-cam rays."""
    if not _hit_motion_debug_enabled() or not fused.found:
        return None
    _ = empty_refs
    _ = raw_frames
    per_cam = list(fused.per_cam)
    size = int(out_size) if out_size > 0 else FITLINE_SIZE
    cals = _score_cals_for_debug(board_calibrator)
    px, py = float(fused.tip_xy[0]), float(fused.tip_xy[1])
    canvas, meta = make_scoring_board_canvas(
        out_size=size,
        cals=cals,
        near_points=[(px, py)],
    )
    rings: Dict[str, float] = meta["rings"]  # type: ignore[assignment]
    ray = float(rings.get("double_outer", 120.0))
    hud = [
        f"YFIT ({px:.1f},{py:.1f}) {_hit_score_label(px, py, out_size=size, cals=cals)}",
        f"Y={fused.yfit_total:.2f}  {fused.yfit_ms:.0f}ms",
    ]
    near_wires: Dict[int, float] = meta["near_wires"]  # type: ignore[assignment]
    if near_wires:
        wi, dist = min(near_wires.items(), key=lambda kv: kv[1])
        hud.append(f"wire {_wire_pair_label(wi)}  {dist:.2f}deg")
    for r in per_cam:
        if not r.found:
            continue
        x0, y0, vx, vy = _fitline_original_xyv(r)
        color = _hit_debug_cam_color(r.cam_idx)
        x1 = int(round(px + vx * ray))
        y1 = int(round(py + vy * ray))
        cv2.line(
            canvas,
            (int(round(px)), int(round(py))),
            (x1, y1),
            color,
            2,
            cv2.LINE_AA,
        )
        hud.append(f"cam{int(r.cam_idx)} s={r.yfit_support:.2f}")
    _draw_marked_point(canvas, (px, py), (0, 255, 255), radius=3, cross=8)
    _draw_hud_lines(canvas, hud)
    stamp = time.strftime("%H%M%S")
    ms = int((time.time() % 1) * 1000)
    label = _zone_short_label(fused.zone_name, fused.segment_number)
    stem = f"hit_{stamp}_{ms:03d}_{label}"
    motion_dir = os.path.join(_debug_dir_path(), "motion")
    try:
        os.makedirs(motion_dir, exist_ok=True)
    except OSError:
        motion_dir = _debug_dir_path()
    path = os.path.join(motion_dir, f"{stem}_yfit.png")
    cv2.imwrite(path, canvas)
    print(f"[yfit] debug {path}", flush=True)
    return path


def save_distance_yfit_debug(
    fused: FusedDartResult,
    empty_refs: Dict[int, np.ndarray],
    *,
    raw_frames: Optional[Dict[int, np.ndarray]] = None,
    board_calibrator: Optional[BoardCalibrator] = None,
    out_size: int = FITLINE_SIZE,
    missing_cams: Optional[List[int]] = None,
) -> Optional[str]:
    """Scoring-board canvas + DYFIT P + usable rays + consensus HUD."""
    if not _hit_motion_debug_enabled() or not fused.found:
        return None
    _ = empty_refs
    _ = raw_frames
    per_cam = list(fused.per_cam)
    size = int(out_size) if out_size > 0 else FITLINE_SIZE
    cals = _score_cals_for_debug(board_calibrator)
    px, py = float(fused.tip_xy[0]), float(fused.tip_xy[1])
    canvas, meta = make_scoring_board_canvas(
        out_size=size,
        cals=cals,
        near_points=[(px, py)],
    )
    rings: Dict[str, float] = meta["rings"]  # type: ignore[assignment]
    ray = float(_dyfit.RAY_LENGTH)
    _ = rings
    usable = [int(c) for c in fused.cam_indices]
    miss = [int(c) for c in (missing_cams or [])]
    if not miss:
        seen = {int(r.cam_idx) for r in per_cam}
        miss = sorted(int(c) for c in seen if c not in set(usable))
    hud = [
        f"DYFIT ({px:.1f},{py:.1f}) {_hit_score_label(px, py, out_size=size, cals=cals)}",
        f"usable_cams={usable}",
        f"consensus={fused.yfit_total:.3f}  {fused.yfit_ms:.1f}ms",
    ]
    used_set = set(usable)
    drawn = set()
    for r in per_cam:
        cam = int(r.cam_idx)
        if cam in used_set and r.found:
            _x0, _y0, vx, vy = _fitline_original_xyv(r)
            color = _hit_debug_cam_color(cam)
            x1 = int(round(px + vx * ray))
            y1 = int(round(py + vy * ray))
            cv2.line(
                canvas,
                (int(round(px)), int(round(py))),
                (x1, y1),
                color,
                2,
                cv2.LINE_AA,
            )
            hud.append(f"cam{cam} s={float(r.yfit_support):.2f}")
            drawn.add(cam)
        else:
            hud.append(f"cam{cam} = missing")
            print(f"[DYFIT] cam{cam} = missing", flush=True)
            drawn.add(cam)
    for cam in miss:
        if cam not in drawn:
            hud.append(f"cam{cam} = missing")
            print(f"[DYFIT] cam{cam} = missing", flush=True)
    _draw_marked_point(canvas, (px, py), (0, 255, 255), radius=3, cross=8)
    _draw_hud_lines(canvas, hud)
    stamp = time.strftime("%H%M%S")
    ms = int((time.time() % 1) * 1000)
    label = _zone_short_label(fused.zone_name, fused.segment_number)
    stem = f"hit_{stamp}_{ms:03d}_{label}"
    motion_dir = os.path.join(_debug_dir_path(), "motion")
    try:
        os.makedirs(motion_dir, exist_ok=True)
    except OSError:
        motion_dir = _debug_dir_path()
    path = os.path.join(motion_dir, f"{stem}_distance_yfit.png")
    cv2.imwrite(path, canvas)
    print(f"[dyfit] debug {path}", flush=True)
    return path


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
    if _motion_is_speckle(mask):
        if save_debug:
            _save_motion_debug(
                cam_idx, mask, motion_pixels=n_px, diff_mean=diff_mean, status="no_blob"
            )
        return DartDetectResult(
            cam_idx=cam_idx,
            found=False,
            motion_pixels=n_px,
            diff_mean=diff_mean,
            reject_reason="no_blob",
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
            draw_fused_tip[0], draw_fused_tip[1], out_size=w, cal=cal, log_geom=False
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
    perf: Optional[Dict[str, float]] = None,
) -> FusedDartResult:
    """Spoji fitline s vise kamera u jedan tip (sjeciste pravaca).

    2-stage FitLine:
      1) per-cam stage1 lines → pairwise consensus tip fuse
      2) re-FitLine all cams on soft pixels inside tip radius → fuse again

    Consensus (3+ cams): score each pair's intersection by sum of squared
    distances to all lines; keep the agreeing pair and drop a fleeing outlier.
    Tip = sjecište u FITLINE_SIZE, zatim map u original top-down scoring coords.
    """
    def _lap(name: str, t0: float) -> None:
        if perf is not None:
            perf[name] = (time.perf_counter() - t0) * 1000.0

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

    if len(line_cams) < int(FUSE_MIN_CAMS):
        print(
            f"[fuse] need_2cam n={len(line_cams)} — 1 line cannot locate a tip",
            flush=True,
        )
        return FusedDartResult(found=False, per_cam=per_cam, reject_reason="need_2cam")

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
    t_fuse1 = time.perf_counter()
    tip_ds, residual, line_cams = _estimate_fused_tip_ds(all_line_cams, board_center_ds)
    fuse1_ms = (time.perf_counter() - t_fuse1) * 1000.0
    if perf is not None:
        perf["fuse1"] = fuse1_ms
        perf["stage1"] = fuse1_ms
    if tip_ds is None:
        reason = "no_intersect"
        if len(all_line_cams) >= 2 and _max_line_cross(all_line_cams) < float(
            FUSE_MIN_CROSS_2CAM
        ):
            reason = "parallel_lines"
        return FusedDartResult(found=False, per_cam=per_cam, reject_reason=reason)

    stage1_tip_ds = (float(tip_ds[0]), float(tip_ds[1]))
    for r in all_line_cams:
        r.stage1_vx = float(r.line_vx)
        r.stage1_vy = float(r.line_vy)
        r.stage1_x0 = float(r.line_x0)
        r.stage1_y0 = float(r.line_y0)

    # --- Stage 2: re-FitLine on soft pixels inside tip ROI, then fuse again ---
    # Refit every cam near the consensus tip — a bad stage1 direction may recover.
    mean_scale = max(0.5 * (scale_x + scale_y), 1e-9)
    board_r_ds = float(board_r) / mean_scale
    tip_roi_r_ds = float(FITLINE_STAGE2_TIP_RADIUS_FRAC) * board_r_ds
    # Persist ROI in original warp space for hit debug overlays.
    stage2_roi_tip_xy = _map_ds_xy_to_original(tip_ds[0], tip_ds[1], scale_x, scale_y)
    stage2_roi_radius = float(FITLINE_STAGE2_TIP_RADIUS_FRAC) * float(board_r)
    t_s2 = time.perf_counter()
    _stage2_refit_lines_near_tip(
        all_line_cams,
        tip_ds,
        tip_roi_r_ds,
        board_center,
        scale_x=scale_x,
        scale_y=scale_y,
    )
    _lap("stage2", t_s2)

    fitq_pair_spread = 0.0
    fitq_suspicious = False
    fitq_recovery_cam = -1
    t_fitq = time.perf_counter()
    for r in all_line_cams:
        q = _measure_cam_fitq(
            r,
            tip_orig=stage2_roi_tip_xy,
            radius_orig=stage2_roi_radius,
            scale_x=scale_x,
            scale_y=scale_y,
            board_r_ds=board_r_ds,
            stage1_tip_ds=stage1_tip_ds,
        )
        _apply_fitq_to_result(r, q)
        _log_fitq_block(r)
    if len(all_line_cams) >= 3:
        fitq_pair_spread = _pair_spread_px(all_line_cams, board_center_ds)
        spread_thresh = max(
            float(FITQ_PAIR_SPREAD_MIN_PX),
            float(FITQ_PAIR_SPREAD_FRAC) * float(board_r_ds),
        )
        fitq_suspicious = fitq_pair_spread > spread_thresh
    print(
        f"[FITQ] pair_spread={fitq_pair_spread:.1f} px "
        f"suspicious={int(fitq_suspicious)}",
        flush=True,
    )
    rec_ms = 0.0
    if fitq_suspicious and len(all_line_cams) >= 3:
        t_rec = time.perf_counter()
        suspect_i = _identify_suspect_cam(all_line_cams, board_center_ds)
        if suspect_i >= 0:
            fitq_recovery_cam = int(all_line_cams[suspect_i].cam_idx)
            did = _recover_suspicious_cam(
                all_line_cams,
                suspect_i,
                tip_orig=stage2_roi_tip_xy,
                radius_orig=stage2_roi_radius,
                scale_x=scale_x,
                scale_y=scale_y,
                board_center_orig=board_center,
                board_center_ds=board_center_ds,
                board_r_ds=board_r_ds,
                stage1_tip_ds=stage1_tip_ds,
            )
            if did:
                fitq_pair_spread = _pair_spread_px(all_line_cams, board_center_ds)
                for r in all_line_cams:
                    _log_fitq_block(r)
                print(
                    f"[FITQ] pair_spread={fitq_pair_spread:.1f} px after recovery",
                    flush=True,
                )
            else:
                fitq_recovery_cam = -1
                print(
                    f"[FITQ] recovery cam{all_line_cams[suspect_i].cam_idx} rejected",
                    flush=True,
                )
        rec_ms = (time.perf_counter() - t_rec) * 1000.0
    fitq_ms = (time.perf_counter() - t_fitq) * 1000.0
    print(
        f"[FITQ] overhead={fitq_ms:.1f}ms recovery={rec_ms:.1f}ms",
        flush=True,
    )

    t_fuse2 = time.perf_counter()
    tip_ds, residual, line_cams = _estimate_fused_tip_ds(all_line_cams, board_center_ds)
    _lap("fuse2", t_fuse2)
    if tip_ds is None:
        reason = "no_intersect"
        if len(all_line_cams) >= 2 and _max_line_cross(all_line_cams) < float(
            FUSE_MIN_CROSS_2CAM
        ):
            reason = "parallel_lines"
        return FusedDartResult(
            found=False,
            per_cam=per_cam,
            reject_reason=reason,
            fitq_pair_spread=fitq_pair_spread,
            fitq_suspicious=fitq_suspicious,
            fitq_recovery_cam=fitq_recovery_cam,
        )

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
            fitq_pair_spread=fitq_pair_spread,
            fitq_suspicious=fitq_suspicious,
            fitq_recovery_cam=fitq_recovery_cam,
        )

    t_score = time.perf_counter()
    number, zone, score = score_topdown_point(
        tx, ty, out_size=FITLINE_SIZE, cals=ring_cals
    )
    _lap("score", t_score)
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
            fitq_pair_spread=fitq_pair_spread,
            fitq_suspicious=fitq_suspicious,
            fitq_recovery_cam=fitq_recovery_cam,
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
                fitq_pair_spread=fitq_pair_spread,
                fitq_suspicious=fitq_suspicious,
                fitq_recovery_cam=fitq_recovery_cam,
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
        fitq_pair_spread=fitq_pair_spread,
        fitq_suspicious=fitq_suspicious,
        fitq_recovery_cam=fitq_recovery_cam,
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
    if len(cams) < 2:
        return None, 0.0, cams
    tip_ds, residual, cams = _fuse_tip_from_best_pair(cams, board_center_ds)
    if tip_ds is None:
        return None, 0.0, list(line_cams)
    if len(cams) >= 3:
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
    """In-place stage-2 FitLine: shaft CC in tip ROI, else original ROI PCA."""
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
        fallback = _refit_fitline_near_tip(ds, tip_cam, r_cam, bc_cam)
        shaft_line, sel_xs, sel_ys, rej_xs, rej_ys = _select_shaft_components_stage2(
            ds, tip_cam, r_cam, bc_cam, cam_idx=int(r.cam_idx)
        )
        r.shaft_sel_xs = sel_xs
        r.shaft_sel_ys = sel_ys
        r.shaft_rej_xs = rej_xs
        r.shaft_rej_ys = rej_ys
        chosen = shaft_line if shaft_line is not None else fallback
        if chosen is None:
            continue
        if DEBUG_CENTERLINE_REFIT:
            r.centerline_old_ds = (
                float(chosen[0]),
                float(chosen[1]),
                float(chosen[2]),
                float(chosen[3]),
            )
            _c, bin_pts, _acc, _reason, cand = _centerline_refit_stage2(
                ds,
                tip_cam,
                r_cam,
                chosen,
                bc_cam,
                cam_idx=int(r.cam_idx),
                scale_x=sx,
                scale_y=sy,
            )
            r.centerline_pts_ds = list(bin_pts)
            r.centerline_cand_ds = cand
            r.centerline_accepted = False
            r.centerline_reject_reason = "debug_only_not_applied"
        vx, vy, x0, y0 = chosen
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
        self._arm_t0: float = 0.0
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
        self._arm_t0 = 0.0

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

    def detect_yfit(
        self,
        board_calibrator: BoardCalibrator,
        frames: Dict[int, Optional[np.ndarray]],
    ) -> FusedDartResult:
        """Joint Y-FIT on 3 warped motion layers. Does not call FitLine."""
        t0 = time.perf_counter()
        layers: List[DartDetectResult] = []
        for cam_idx, frame in frames.items():
            if frame is None:
                continue
            cal = board_calibrator.get(cam_idx)
            if cal is None or not cal.is_valid() or cal.double_ellipse is None:
                continue
            warped = _warp_frame(board_calibrator, int(cam_idx), frame, cal)
            ref = self._refs.get(int(cam_idx))
            if warped is None or ref is None:
                layers.append(
                    DartDetectResult(
                        cam_idx=int(cam_idx),
                        found=False,
                        reject_reason="no_ref" if ref is None else "warp_fail",
                    )
                )
                continue
            layers.append(
                extract_yfit_motion_layer(warped, ref, cam_idx=int(cam_idx))
            )
        board_center, board_r = _board_center_and_r(FITLINE_SIZE)
        usable = [r for r in layers if r.fit_ds_gray is not None and r.motion_pixels >= MOTION_MIN_PIXELS]
        if len(usable) < 2:
            return FusedDartResult(
                found=False, per_cam=layers, reject_reason="no_motion", yfit_ms=(time.perf_counter() - t0) * 1000.0
            )
        p_xy, dirs, total, yfit_prof = run_yfit_search(
            usable, board_center=board_center, board_r=board_r
        )
        yfit_ms = (time.perf_counter() - t0) * 1000.0
        print(
            f"[YFIT][prof] bbox=({yfit_prof['bbox_x0']:.0f},{yfit_prof['bbox_y0']:.0f})-"
            f"({yfit_prof['bbox_x1']:.0f},{yfit_prof['bbox_y1']:.0f}) "
            f"coarse_P={yfit_prof['coarse_P_count']:.0f} "
            f"coarse_angle_evals={yfit_prof['coarse_angle_evals']:.0f} "
            f"coarse_ms={yfit_prof['coarse_ms']:.1f} "
            f"fine_P={yfit_prof['fine_P_count']:.0f} "
            f"fine_angle_evals={yfit_prof['fine_angle_evals']:.0f} "
            f"fine_ms={yfit_prof['fine_ms']:.1f} "
            f"total_yfit_ms={yfit_prof['total_yfit_ms']:.1f}",
            flush=True,
        )
        if p_xy is None or total < float(YFIT_MIN_TOTAL):
            print(
                f"[YFIT] fail total={total:.3f} ms={yfit_ms:.1f} cams={len(usable)}",
                flush=True,
            )
            return FusedDartResult(
                found=False,
                per_cam=layers,
                reject_reason="yfit_low_score",
                yfit_total=float(total),
                yfit_ms=yfit_ms,
            )
        fused = yfit_to_fused(
            layers,
            p_xy,
            dirs,
            total,
            board_calibrator=board_calibrator,
            yfit_ms=yfit_ms,
        )
        print(
            f"[YFIT] P=({p_xy[0]:.1f},{p_xy[1]:.1f}) total={total:.3f} "
            f"ms={yfit_ms:.1f} "
            + " ".join(
                f"cam{c}={dirs[c][2]:.2f}" for c in sorted(dirs)
            ),
            flush=True,
        )
        return fused

    def detect_distance_yfit(
        self,
        board_calibrator: BoardCalibrator,
        frames: Dict[int, Optional[np.ndarray]],
    ) -> FusedDartResult:
        """warpPolar distance-Y on separate warped motion layers (final_v2).

        FitLine is not used for P. Caller falls back to FitLine/Y-refine if invalid.
        """
        t0 = time.perf_counter()
        mode = "distance_yfit"
        layers: List[DartDetectResult] = []
        motion: Dict[int, np.ndarray] = {}
        for cam_idx, frame in frames.items():
            if frame is None:
                continue
            cal = board_calibrator.get(cam_idx)
            if cal is None or not cal.is_valid() or cal.double_ellipse is None:
                continue
            warped = _warp_frame(board_calibrator, int(cam_idx), frame, cal)
            ref = self._refs.get(int(cam_idx))
            if warped is None or ref is None:
                layers.append(
                    DartDetectResult(
                        cam_idx=int(cam_idx),
                        found=False,
                        reject_reason="no_ref" if ref is None else "warp_fail",
                    )
                )
                continue
            layer = extract_yfit_motion_layer(warped, ref, cam_idx=int(cam_idx))
            layers.append(layer)
            soft = layer.fit_ds_gray if layer.fit_ds_gray is not None else layer.fit_gray
            if soft is not None:
                motion[int(cam_idx)] = soft

        board_center, _board_r = _board_center_and_r(FITLINE_SIZE)
        est = _dyfit.estimate_dart_tip_distance_yfit(
            motion,
            board_center=board_center,
            board_radius=float(_dyfit.MAX_IMPACT_RADIUS),
        )
        dyfit_ms = (time.perf_counter() - t0) * 1000.0
        seen = {int(r.cam_idx) for r in layers}
        missing = sorted(c for c in seen if c not in set(est.usable_cams))
        for cam in missing:
            print(f"[DYFIT] cam{int(cam)} = missing", flush=True)
        print(
            f"[DYFIT][prof] estimator={mode} usable={est.usable_cams} missing={missing} "
            f"mask={est.mask_ms:.1f}ms dist={est.distance_transform_ms:.1f}ms "
            f"seed={est.seed_ms:.1f}ms coarse={est.coarse_ms:.1f}ms "
            f"fine={est.fine_ms:.1f}ms total={est.runtime_ms:.1f}ms wall={dyfit_ms:.1f}ms",
            flush=True,
        )
        if not est.valid:
            print(
                f"[DYFIT] fail {est.reject_reason or 'invalid'} "
                f"score={est.score:.3f} ms={dyfit_ms:.1f} usable={est.usable_cams}",
                flush=True,
            )
            return FusedDartResult(
                found=False,
                per_cam=layers,
                reject_reason=est.reject_reason or "dyfit_fail",
                yfit_total=float(est.score),
                yfit_ms=dyfit_ms,
                yrefine_mode=mode,
            )
        fused = dyfit_estimate_to_fused(
            layers, est, board_calibrator=board_calibrator, mode=mode
        )
        fused.yfit_ms = dyfit_ms
        print(
            f"[DYFIT] P=({est.x:.1f},{est.y:.1f}) consensus={est.score:.3f} "
            f"ms={dyfit_ms:.1f} usable_cams={est.usable_cams} "
            + " ".join(
                f"cam{c}={est.per_cam_score.get(c, 0.0):.2f}"
                for c in sorted(est.usable_cams)
            ),
            flush=True,
        )
        return fused

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
                        self._arm_t0 = time.perf_counter()
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
                self._armed_frames = 0
                self._arm_t0 = time.perf_counter()
                self._hand_stable_streak = 0
                state = "armed"
            else:
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
        t_proc = time.perf_counter()
        wait_since_arm_ms = 0.0
        if self._arm_t0 > 0.0:
            wait_since_arm_ms = (t_proc - self._arm_t0) * 1000.0
        # Soft-threshold FitLine po kameri (downscaled) → tip = sjecište u ds, map natrag.
        t_detect = time.perf_counter()
        fuse_perf: Dict[str, float] = {}
        fitline_debug_saved = False
        dyfit_used = False
        if _distance_yfit_enabled():
            fused = self.detect_distance_yfit(board_calibrator, fire_frames)
            per_cam = list(fused.per_cam)
            detect_ms = (time.perf_counter() - t_detect) * 1000.0
            fuse_perf["stage1"] = 0.0
            fuse_perf["fuse1"] = 0.0
            fuse_perf["stage2"] = float(fused.yfit_ms)
            fuse_perf["fuse2"] = 0.0
            fuse_perf["score"] = 0.0
            if fused.found:
                dyfit_used = True
            else:
                print(
                    f"[DYFIT] fail {fused.reject_reason or 'unknown'} "
                    f"— fallback {DART_DETECTOR_MODE}",
                    flush=True,
                )
                t_detect = time.perf_counter()
                fuse_perf = {}
        if not dyfit_used:
            if DART_DETECTOR_MODE == "yfit":
                fused = self.detect_yfit(board_calibrator, fire_frames)
                per_cam = list(fused.per_cam)
                detect_ms = (time.perf_counter() - t_detect) * 1000.0
                fuse_perf["stage1"] = 0.0
                fuse_perf["fuse1"] = 0.0
                fuse_perf["stage2"] = float(fused.yfit_ms)
                fuse_perf["fuse2"] = 0.0
                fuse_perf["score"] = 0.0
            else:
                per_cam = self.detect_all(board_calibrator, fire_frames, save_debug=False)
                detect_ms = (time.perf_counter() - t_detect) * 1000.0
                fused = fuse_multicam_lines(
                    per_cam, board_calibrator=board_calibrator, perf=fuse_perf
                )
                fused.per_cam = per_cam
                fused = _apply_ml_tip_cached(fused, per_cam, board_calibrator)
                skip_refine = (not fused.found) or fused.reject_reason in (
                    "miss_majority",
                    "off_board_raw",
                )
                if DART_DETECTOR_MODE == "hybrid" and fused.found and not skip_refine:
                    if save_debug and _hit_motion_debug_enabled():
                        save_hit_fusion_debug(
                            fused,
                            self._empty_refs,
                            raw_frames=fire_frames,
                            board_calibrator=board_calibrator,
                        )
                        fitline_debug_saved = True
                    t_yr = time.perf_counter()
                    try:
                        fused = run_yrefine_local(
                            fused, board_calibrator=board_calibrator
                        )
                    except Exception as exc:
                        print(f"[YREFINE] fail {exc!r} — keep FitLine", flush=True)
                    fuse_perf["yrefine"] = (time.perf_counter() - t_yr) * 1000.0
        total_processing_ms = (time.perf_counter() - t_proc) * 1000.0
        total_arm_to_result_ms = (
            (time.perf_counter() - self._arm_t0) * 1000.0 if self._arm_t0 > 0.0 else total_processing_ms
        )
        # Debug / raw-shaft shadow nakon UI-ja — ne smiju blokirati prikaz hita.
        if save_debug and fused.found and _hit_motion_debug_enabled():
            if dyfit_used:
                save_distance_yfit_debug(
                    fused,
                    self._empty_refs,
                    raw_frames=fire_frames,
                    board_calibrator=board_calibrator,
                )
            elif DART_DETECTOR_MODE == "yfit":
                save_yfit_debug(
                    fused,
                    self._empty_refs,
                    raw_frames=fire_frames,
                    board_calibrator=board_calibrator,
                )
            elif DART_DETECTOR_MODE == "hybrid":
                save_yrefine_debug(
                    fused,
                    self._empty_refs,
                    raw_frames=fire_frames,
                    board_calibrator=board_calibrator,
                )
                if not fitline_debug_saved:
                    save_hit_fusion_debug(
                        fused,
                        self._empty_refs,
                        raw_frames=fire_frames,
                        board_calibrator=board_calibrator,
                    )
            else:
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
            print(
                "[PERF][HIT]\n"
                f"wait_since_arm={wait_since_arm_ms:.1f}ms\n"
                f"detect_all={detect_ms:.1f}ms\n"
                f"stage1={float(fuse_perf.get('stage1', 0.0)):.1f}ms\n"
                f"fuse1={float(fuse_perf.get('fuse1', 0.0)):.1f}ms\n"
                f"stage2={float(fuse_perf.get('stage2', 0.0)):.1f}ms\n"
                f"fuse2={float(fuse_perf.get('fuse2', 0.0)):.1f}ms\n"
                f"score={float(fuse_perf.get('score', 0.0)):.1f}ms\n"
                + (
                    f"yrefine={float(fuse_perf.get('yrefine', 0.0)):.1f}ms\n"
                    if "yrefine" in fuse_perf
                    else ""
                )
                + (
                    f"dyfit={float(fused.yfit_ms):.1f}ms\n"
                    if dyfit_used
                    else ""
                )
                + f"total_processing={total_processing_ms:.1f}ms\n"
                f"total_arm_to_result={total_arm_to_result_ms:.1f}ms",
                flush=True,
            )
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


def _yrefine_noise_smoke_test() -> None:
    """Bright dart survives; faint specks — even 8-connected trails — are dropped."""
    img = np.zeros((64, 64), dtype=np.float32)
    img[20:28, 10:40] = 0.85
    for i in range(8):
        img[40 + (i % 2), 8 + i * 6] = 0.12
    out, n_keep, n_all = _yrefine_keep_dart_motion(img)
    dart = int(np.count_nonzero(out[20:28, 10:40] > 0))
    specks = int(np.count_nonzero(out[38:44, :] > 0))
    assert dart >= 20, f"denoise smoke: dart pixels {dart}"
    assert specks == 0, f"denoise smoke: distant specks kept {specks}"

    trail = np.zeros((64, 64), dtype=np.float32)
    trail[20:28, 10:40] = 0.85
    for i in range(14):
        trail[28 + i, 39] = 0.15
    out2, k2, a2 = _yrefine_keep_dart_motion(trail)
    far = int(np.count_nonzero(out2[32:48, 37:42] > 0))
    near_dart = int(np.count_nonzero(out2[20:28, 10:40] > 0))
    assert near_dart >= 20, f"denoise smoke: attached dart lost {near_dart}"
    assert far == 0, f"denoise smoke: attached trail kept {far} (kept={k2}/{a2})"

    beside = np.zeros((64, 64), dtype=np.float32)
    beside[20:28, 10:40] = 0.85
    beside[12:16, 22:26] = 0.90
    out3, k3, a3 = _yrefine_keep_dart_motion(beside)
    dart3 = int(np.count_nonzero(out3[20:28, 10:40] > 0))
    sat = int(np.count_nonzero(out3[12:16, 22:26] > 0))
    assert dart3 >= 20, f"denoise smoke: dart lost next to satellite {dart3}"
    assert sat == 0, f"denoise smoke: bright satellite kept {sat} (kept={k3}/{a3})"
    print(
        f"[yrefine noise smoke] OK kept={n_keep}/{n_all} trail_kept={k2}/{a2} "
        f"sat_kept={k3}/{a3}",
        flush=True,
    )


def _fuse_geometry_smoke_test() -> None:
    """1-cam and near-parallel 2-cam must not invent a tip."""
    one = [
        DartDetectResult(
            cam_idx=0,
            found=True,
            line_x0=160.0,
            line_y0=40.0,
            line_vx=0.2,
            line_vy=1.0,
            motion_pixels=80,
        )
    ]
    tip, _res, _kept = _estimate_fused_tip_ds(one, board_center_ds=(150.0, 150.0))
    assert tip is None, f"1-cam fuse must not project to center, got {tip}"

    para = [
        DartDetectResult(
            cam_idx=0,
            found=True,
            line_x0=200.0,
            line_y0=100.0,
            line_vx=1.0,
            line_vy=0.02,
            motion_pixels=80,
        ),
        DartDetectResult(
            cam_idx=2,
            found=True,
            line_x0=200.0,
            line_y0=108.0,
            line_vx=1.0,
            line_vy=0.01,
            motion_pixels=80,
        ),
    ]
    tip2, _r2, _k2 = _fuse_tip_from_best_pair(para, board_center=(150.0, 150.0))
    assert tip2 is None, f"parallel 2-cam must not intersect, got {tip2}"

    good = [
        DartDetectResult(
            cam_idx=0,
            found=True,
            line_x0=150.0,
            line_y0=150.0,
            line_vx=1.0,
            line_vy=0.0,
            motion_pixels=80,
        ),
        DartDetectResult(
            cam_idx=2,
            found=True,
            line_x0=150.0,
            line_y0=150.0,
            line_vx=0.0,
            line_vy=1.0,
            motion_pixels=80,
        ),
    ]
    tip3, _r3, _k3 = _fuse_tip_from_best_pair(good, board_center=(150.0, 150.0))
    assert tip3 is not None, "crossing 2-cam should intersect"
    err = float(np.hypot(tip3[0] - 150.0, tip3[1] - 150.0))
    assert err < 1.0, f"crossing 2-cam tip={tip3} err={err:.2f}"
    print("[fuse geometry smoke] OK need_2cam + parallel_2cam + crossing_2cam", flush=True)


def _yrefine_flee_gate_smoke_test() -> None:
    """Y-refine only when a used FitLine misses the fused tip."""
    concurrent = FusedDartResult(
        found=True,
        tip_xy=(150.0, 150.0),
        cam_indices=[0, 2, 4],
        per_cam=[
            DartDetectResult(
                cam_idx=0, found=True, line_x0=150.0, line_y0=150.0, line_vx=1.0, line_vy=0.0
            ),
            DartDetectResult(
                cam_idx=2, found=True, line_x0=150.0, line_y0=150.0, line_vx=0.0, line_vy=1.0
            ),
            DartDetectResult(
                cam_idx=4, found=True, line_x0=150.0, line_y0=150.0, line_vx=0.7, line_vy=0.7
            ),
        ],
    )
    need, max_flee, _d = _fitline_needs_yrefine(concurrent)
    assert not need, f"concurrent 3-cam should skip Y, max_flee={max_flee:.2f}"

    two = FusedDartResult(
        found=True,
        tip_xy=(150.0, 150.0),
        cam_indices=[0, 2],
        per_cam=[
            DartDetectResult(
                cam_idx=0, found=True, line_x0=150.0, line_y0=150.0, line_vx=1.0, line_vy=0.0
            ),
            DartDetectResult(
                cam_idx=2, found=True, line_x0=150.0, line_y0=150.0, line_vx=0.0, line_vy=1.0
            ),
        ],
    )
    need_two, max_two, _dt = _fitline_needs_yrefine(two)
    assert need_two, f"2-cam must run Y even if concurrent, max_flee={max_two:.2f}"

    fleer = FusedDartResult(
        found=True,
        tip_xy=(150.0, 150.0),
        cam_indices=[0, 2, 4],
        per_cam=[
            DartDetectResult(
                cam_idx=0, found=True, line_x0=150.0, line_y0=150.0, line_vx=1.0, line_vy=0.0
            ),
            DartDetectResult(
                cam_idx=2, found=True, line_x0=150.0, line_y0=150.0, line_vx=0.0, line_vy=1.0
            ),
            DartDetectResult(
                cam_idx=4, found=True, line_x0=156.0, line_y0=150.0, line_vx=0.0, line_vy=1.0
            ),
        ],
    )
    need2, max_flee2, _d2 = _fitline_needs_yrefine(fleer)
    assert need2, f"fleeing line should run Y, max_flee={max_flee2:.2f}"
    assert max_flee2 >= float(YREFINE_FLEE_PX)
    print(
        f"[yrefine flee gate smoke] OK skip3={max_flee:.2f} run2cam={max_two:.2f} "
        f"run_flee={max_flee2:.2f}",
        flush=True,
    )


def _blob_gate_smoke_test() -> None:
    speckle = np.zeros((80, 80), dtype=np.uint8)
    for i in range(50):
        speckle[4 + (i * 3) % 70, 5 + (i * 7) % 70] = 255
    dart = np.zeros((80, 80), dtype=np.uint8)
    dart[24:34, 12:52] = 255
    thin = np.zeros((80, 80), dtype=np.uint8)
    thin[40, 20:38] = 255  # 18 px shaft — must still count as a hit cam
    assert _motion_is_speckle(speckle), "speckle field should reject"
    assert not _motion_is_speckle(dart), "dart blob should pass"
    assert not _motion_is_speckle(thin), "thin shaft should pass"
    kept = _keep_large_motion_ccs(speckle, min_area=int(MOTION_MIN_BLOB_AREA))
    assert int(np.count_nonzero(kept)) == 0, "speckle must be stripped from debug"
    print(
        f"[blob gate smoke] OK speckle={_motion_cc_stats(speckle)} "
        f"dart={_motion_cc_stats(dart)} thin={_motion_cc_stats(thin)}",
        flush=True,
    )


def _dyfit_blank() -> np.ndarray:
    return np.zeros((FITLINE_SIZE, FITLINE_SIZE), dtype=np.uint8)


def _dyfit_stroke_ray(
    img: np.ndarray,
    p: Tuple[float, float],
    theta_deg: float,
    length: float,
    *,
    thickness: int = 2,
    value: int = 220,
    skip: float = 0.0,
    dash_on: float = 0.0,
    dash_off: float = 0.0,
    blob_u: float = 0.0,
    blob_r: int = 0,
) -> None:
    px, py = float(p[0]), float(p[1])
    rad = float(np.deg2rad(theta_deg))
    vx, vy = float(np.cos(rad)), float(np.sin(rad))
    if dash_on > 0.0:
        t = float(skip)
        while t < float(length):
            t1 = min(t + float(dash_on), float(length))
            cv2.line(
                img,
                (int(round(px + vx * t)), int(round(py + vy * t))),
                (int(round(px + vx * t1)), int(round(py + vy * t1))),
                int(value),
                int(thickness),
            )
            t = t1 + float(dash_off if dash_off > 0.0 else dash_on)
    else:
        cv2.line(
            img,
            (int(round(px + vx * skip)), int(round(py + vy * skip))),
            (int(round(px + vx * length)), int(round(py + vy * length))),
            int(value),
            int(thickness),
        )
    if blob_r > 0 and blob_u > 0.0:
        cv2.circle(
            img,
            (int(round(px + vx * blob_u)), int(round(py + vy * blob_u))),
            int(blob_r),
            int(value),
            -1,
        )


def _dyfit_test_layer(cam: int, img: np.ndarray, *, usable: bool = True) -> DartDetectResult:
    n = int(np.count_nonzero(img))
    return DartDetectResult(
        cam_idx=int(cam),
        found=False,
        motion_pixels=n if usable else 0,
        reject_reason="" if usable else "low_motion",
        fit_gray=img,
        fit_ds_gray=img,
    )


def _dyfit_search_p(
    layers: List[DartDetectResult],
) -> Tuple[Optional[Tuple[float, float]], float, List[int], float]:
    motion: Dict[int, np.ndarray] = {}
    for r in layers:
        if r.reject_reason == "low_motion":
            continue
        g = r.fit_ds_gray if r.fit_ds_gray is not None else r.fit_gray
        if g is None:
            continue
        motion[int(r.cam_idx)] = g
    board_center, _br = _board_center_and_r(FITLINE_SIZE)
    est = _dyfit.estimate_dart_tip_distance_yfit(
        motion,
        board_center=board_center,
        board_radius=float(_dyfit.MAX_IMPACT_RADIUS),
    )
    return est.point, float(est.score), list(est.usable_cams), float(est.runtime_ms)


def _distance_yfit_smoke_test() -> None:
    """Synthetic layers + standalone-vs-production P match on fused_motion_raw."""
    p_true = (150.0, 150.0)
    th = {0: 0.0, 2: 90.0, 4: 225.0}
    length = 52.0

    def _three(*, skip: float = 0.0, dash: bool = False, flight: bool = False, drop: Optional[int] = None):
        layers: List[DartDetectResult] = []
        for cam, theta in th.items():
            img = _dyfit_blank()
            if drop is not None and int(cam) == int(drop):
                layers.append(_dyfit_test_layer(cam, img, usable=False))
                continue
            _dyfit_stroke_ray(
                img,
                p_true,
                theta,
                length,
                skip=skip,
                dash_on=6.0 if dash else 0.0,
                dash_off=4.0 if dash else 0.0,
                blob_u=48.0 if flight and cam == 0 else 0.0,
                blob_r=11 if flight and cam == 0 else 0,
            )
            layers.append(_dyfit_test_layer(cam, img))
        return layers

    p3, tot3, u3, ms3 = _dyfit_search_p(_three())
    assert p3 is not None, "3-cam DYFIT should find P"
    err3 = float(np.hypot(p3[0] - p_true[0], p3[1] - p_true[1]))
    assert err3 <= 4.0, f"3-cam P={p3} err={err3:.2f}"
    assert sorted(u3) == [0, 2, 4], f"usable {u3}"

    p24, tot24, u24, _ms = _dyfit_search_p(_three(drop=0))
    assert p24 is not None, "cam2+cam4 should locate P"
    err24 = float(np.hypot(p24[0] - p_true[0], p24[1] - p_true[1]))
    assert err24 <= 6.0, f"cam2+cam4 P={p24} err={err24:.2f}"
    assert sorted(u24) == [2, 4]
    drift24 = float(np.hypot(p24[0] - p3[0], p24[1] - p3[1]))
    assert drift24 <= 6.0, f"P drifted {drift24:.2f}px when cam0 vanished"

    p02, tot02, u02, _ms = _dyfit_search_p(_three(drop=4))
    assert p02 is not None, "cam0+cam2 should locate P"
    err02 = float(np.hypot(p02[0] - p_true[0], p02[1] - p_true[1]))
    assert err02 <= 6.0, f"cam0+cam2 P={p02} err={err02:.2f}"
    assert sorted(u02) == [0, 2]

    p_fl, tot_fl, _u, _ms = _dyfit_search_p(_three(flight=True))
    assert p_fl is not None, "flight-blob case should find P"
    err_fl = float(np.hypot(p_fl[0] - p_true[0], p_fl[1] - p_true[1]))
    assert err_fl <= 6.0, f"flight blob pulled P to {p_fl} err={err_fl:.2f}"

    p_fr, tot_fr, _u, _ms = _dyfit_search_p(_three(dash=True))
    assert p_fr is not None, "fragmented shaft should find P"
    err_fr = float(np.hypot(p_fr[0] - p_true[0], p_fr[1] - p_true[1]))
    assert err_fr <= 6.0, f"fragmented P={p_fr} err={err_fr:.2f}"

    p_h, tot_h, _u, _ms = _dyfit_search_p(_three(skip=8.0))
    assert p_h is not None, "tip hole should still find P"
    err_h = float(np.hypot(p_h[0] - p_true[0], p_h[1] - p_true[1]))
    assert err_h <= 8.0, f"tip-hole P={p_h} err={err_h:.2f}"

    only0 = [
        _three()[0],
        _dyfit_test_layer(2, _dyfit_blank(), usable=False),
        _dyfit_test_layer(4, _dyfit_blank(), usable=False),
    ]
    p1, tot1, u1, _ms = _dyfit_search_p(only0)
    assert p1 is None, f"1-cam must fail, got P={p1}"
    assert len(u1) < 2

    print(
        f"[distance_yfit smoke] OK 3cam={err3:.2f}px 2+4={err24:.2f} 0+2={err02:.2f} "
        f"flight={err_fl:.2f} frag={err_fr:.2f} hole={err_h:.2f} "
        f"ms3={ms3:.1f} c3={tot3:.3f} c24={tot24:.3f} c02={tot02:.3f} cfl={tot_fl:.3f}",
        flush=True,
    )

    import importlib.util

    v2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "distance_yfit_warppolar_final_v2.py")
    spec = importlib.util.spec_from_file_location("distance_yfit_warppolar_final_v2", v2_path)
    assert spec is not None and spec.loader is not None
    v2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v2)

    def _paint_fused(drop=None, flight=False, dash=False, skip=0.0):
        canvas = np.zeros((FITLINE_SIZE, FITLINE_SIZE, 3), dtype=np.uint8)
        colors = {0: (0, 0, 255), 2: (0, 255, 0), 4: (255, 255, 0)}
        for cam, theta in th.items():
            if drop is not None and int(cam) == int(drop):
                continue
            tmp = _dyfit_blank()
            _dyfit_stroke_ray(
                tmp,
                p_true,
                theta,
                length,
                skip=skip,
                dash_on=6.0 if dash else 0.0,
                dash_off=4.0 if dash else 0.0,
                blob_u=48.0 if flight and cam == 0 else 0.0,
                blob_r=11 if flight and cam == 0 else 0,
            )
            col = np.array(colors[cam], dtype=np.uint8)
            canvas[tmp > 0] = col
        return canvas

    cases = {
        "3cam": _paint_fused(),
        "cam2+cam4": _paint_fused(drop=0),
        "cam0+cam2": _paint_fused(drop=4),
        "flight": _paint_fused(flight=True),
        "frag": _paint_fused(dash=True),
        "hole": _paint_fused(skip=8.0),
    }
    compare_errs = []
    compare_ms = []
    for label, bgr in cases.items():
        st, *_ = v2.run_yfit(bgr)
        prod = _dyfit.estimate_from_fused_debug_png(bgr)
        assert prod.valid, f"{label} production invalid, standalone P={st.p}"
        err = float(np.hypot(prod.x - float(st.p[0]), prod.y - float(st.p[1])))
        compare_errs.append(err)
        compare_ms.append(float(prod.runtime_ms))
        print(
            f"[DYFIT][compare] {label} standalone=({st.p[0]:.1f},{st.p[1]:.1f}) "
            f"prod=({prod.x:.1f},{prod.y:.1f}) err={err:.2f}px "
            f"usable={prod.usable_cams} score={prod.score:.3f}/{st.score:.3f} "
            f"mask={prod.mask_ms:.1f} seed={prod.seed_ms:.1f} "
            f"coarse={prod.coarse_ms:.1f} fine={prod.fine_ms:.1f} total={prod.runtime_ms:.1f}",
            flush=True,
        )
        assert err <= 0.51, f"{label} production P drifted {err:.2f}px from standalone"
        assert abs(float(prod.score) - float(st.score)) < 1e-4, f"{label} score mismatch"

    motion_dir = os.path.join(_debug_dir_path(), "motion")
    n_png = 0
    if os.path.isdir(motion_dir):
        raws = sorted(f for f in os.listdir(motion_dir) if f.endswith("_fused_motion_raw.png"))
        for name in raws:
            path = os.path.join(motion_dir, name)
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            try:
                st, *_ = v2.run_yfit(bgr)
            except Exception as exc:
                print(f"[DYFIT][compare] {name} standalone fail {exc!r}", flush=True)
                continue
            prod = _dyfit.estimate_from_fused_debug_png(bgr)
            assert prod.valid, f"{name} production invalid, standalone P={st.p}"
            err = float(np.hypot(prod.x - float(st.p[0]), prod.y - float(st.p[1])))
            compare_errs.append(err)
            compare_ms.append(float(prod.runtime_ms))
            n_png += 1
            print(
                f"[DYFIT][compare] {name} standalone=({st.p[0]:.1f},{st.p[1]:.1f}) "
                f"prod=({prod.x:.1f},{prod.y:.1f}) err={err:.2f}px "
                f"usable={prod.usable_cams} score={prod.score:.3f} "
                f"mask={prod.mask_ms:.1f} seed={prod.seed_ms:.1f} "
                f"coarse={prod.coarse_ms:.1f} fine={prod.fine_ms:.1f} total={prod.runtime_ms:.1f}",
                flush=True,
            )
            assert err <= 0.51, f"{name} production P drifted {err:.2f}px from standalone"
    print(
        f"[distance_yfit compare] OK synthetic={len(cases)} png={n_png} "
        f"max_err={max(compare_errs):.2f}px mean_ms={float(np.mean(compare_ms)):.1f} "
        f"max_ms={max(compare_ms):.1f}",
        flush=True,
    )


if __name__ == "__main__":
    _fitline_smoke_test()
    _consensus_fuse_smoke_test()
    _fuse_geometry_smoke_test()
    _yrefine_flee_gate_smoke_test()
    _blob_gate_smoke_test()
    _yrefine_noise_smoke_test()
    _distance_yfit_smoke_test()
