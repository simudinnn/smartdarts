"""Bull + double-ring ellipse calibration for angled dart-board cameras."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# OpenCV ellipse: ((cx, cy), (width, height), angle_deg)
Ellipse = Tuple[Tuple[float, float], Tuple[float, float], float]

# Nakon "PLOČA JE PRAZNA" na clear_board ekranu: ponovo detektiraj ploču.
RECALIBRATE_BOARD_ON_CLEAR_CONFIRM = True

SEGMENT_COUNT = 20
DIAMETER_COUNT = SEGMENT_COUNT // 2
# Stage1: elipsa = VANJSKI rub doublea (prava geometrija ploče).
# Mali polar-margin samo za view (strijelice malo izvan doublea) — NE skalirati
# elipsu u 2D oko njezinog centra (to lomi planarnost i pravi jajasti topdown).
STAGE1_OUTSIDE_SCALE = 1.0
# Alias — stari naziv u refine/fit putanjama.
DOUBLE_OUTER_SCALE = STAGE1_OUTSIDE_SCALE
# Ako Stage1 double još nije dovoljno krug, Stage2 radi egg→circle (+ multi-ring).
STAGE2_OVALITY_MIN = 0.028
STAGE2_RADIUS_MAX_DEV = 0.035
STAGE2_CENTER_MAX_FRAC = 0.04
# Fit prstenova na warpu; debljina = razmak inner/outer, ne stroke.
WARP_RING_ELLIPSE_SCALE = 1.0
WARP_RING_DRAW_THICKNESS = 1
WARP_RING_INNER_TIGHTEN = 1.0
# Overlay = nominalni krugovi; Stage2 dovodi warp na njih (bez pad fudge-a).
# Scoring rings match calibrated/drawn geometry exactly — no radial pad expansion.
WIRE_DETECT_SCALE = 1.0
FITLINE_MIN_POINTS = 5
DIAMETER_LINE_DIST_PX = 4.5
FITLINE_REFINE_LINE_DIST_PX = 2.5
DIAMETER_VOTE_STEP_DEG = 0.25
DIAMETER_MIN_SEP_DEG = 9.0
DIAMETER_MERGE_SEP_DEG = 8.5
DIAMETER_VOTE_SCORE_RATIO = 0.22
DIAMETER_VOTE_SMOOTH_DEG = 4.0
# Popuna samo kad fali tocno 1 promjer (npr. vertikal 12h-6h); normalni razmak = 18°.
DIAMETER_MISSING_GAP_MIN_DEG = 26.0
WARP_SIZE = 320
# Top-down: double rub na board_radius, oko ploče ostaje margina za strijelice u doubleu.
WARP_BOARD_RADIUS_FRAC = 0.82
WARP_VIEW_RADIUS_FRAC = 0.98
WARP_OUTSIDE_EXTRAPOL = 0.28
# Standardni omjeri prstenova (udaljenost / vanjski double rub = 170 mm).
RING_DOUBLE_OUTER_FRAC = 1.0
RING_DOUBLE_INNER_FRAC = 161.0 / 170.0
RING_TRIPLE_OUTER_FRAC = 108.0 / 170.0
RING_TRIPLE_INNER_FRAC = 98.0 / 170.0
RING_BULL_OUTER_FRAC = 16.9 / 170.0
RING_BULL_INNER_FRAC = 6.35 / 170.0
BOARD_SISAL_EDGE_FRAC = 225.0 / 170.0
DOUBLE_RING_SAMPLE_INNER = 1
DOUBLE_RING_SAMPLE_OUTER = 1
# Kalibracija: privremeni upscale kad je capture mali (tanki double bolje vidljiv).
CALIBRATE_UPSCALE_IF_MIN_DIM_LT = 400
CALIBRATE_UPSCALE = 2
CALIBRATION_BLEND_ALPHA = 0.28
# Max bull shift (px / frame-diag fraction) for soft recover / stabilize.
CAL_BULL_STABLE_SHIFT_PX = 22.0
CAL_BULL_STABLE_SHIFT_FRAC = 0.07
CAL_BULL_REJECT_SHIFT_FRAC = 0.14
SEGMENT_STEP_DEG = 360.0 / SEGMENT_COUNT
BOARD_SEGMENT_NUMBERS: Tuple[int, ...] = (
    20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5,
)
SEG20_OVERLAY_BGR = (255, 60, 0)
SEG20_OVERLAY_ALPHA = 0.52
# Segment 20 na vrhu: lijeva zica 351°, desna 9° (12h=0°, clockwise).
SEG20_LEFT_WIRE_DST_DEG = (360.0 - SEGMENT_STEP_DEG * 0.5) % 360.0
DEBUG_WARP_SIZE = 200
EDGE_REFINE_RANGE_DEG = 6.0
EDGE_REFINE_STEP_DEG = 0.5
CANNY_LOW = 40
CANNY_HIGH = 120
CALIBRATION_FILENAME = "board_calibration.json"
DEBUG_SAVE_CANNY_EDGES = True


@dataclass
class BoardCalibration:
    center: Tuple[float, float]
    bull_radius: float
    confidence: float
    double_ellipse: Optional[Ellipse] = None
    board_confidence: float = 0.0
    wire_angles_deg: Tuple[float, ...] = ()
    homography_shift: int = 0
    align_rotation_deg: float = 0.0
    segment20_offset: int = 0
    segment20_wire_sector: int = -1
    segment20_wire_end: int = -1
    # Izmjereni prstenovi na top-down warpu (udio board_r) — fallback / miss.
    measured_double_outer_frac: float = 0.0
    measured_double_inner_frac: float = 0.0
    measured_triple_outer_frac: float = 0.0
    measured_triple_inner_frac: float = 0.0
    # Jajaste/kružne elipse prstenova u FINALNOM warp prostoru (px @ warp_ring_size).
    warp_ring_size: int = 0
    warp_double_outer: Optional[Ellipse] = None
    warp_double_inner: Optional[Ellipse] = None
    warp_triple_outer: Optional[Ellipse] = None
    warp_triple_inner: Optional[Ellipse] = None
    # Double elipsa na 1. topdown warpu (točke na VANJSKOM rubu doublea) — Stage2 warp.
    warp1_double_outer: Optional[Ellipse] = None
    warp1_triple_outer: Optional[Ellipse] = None  # Stage1 triple outer (multi-ring Stage2)
    # Ray-origin Stage2 (bull na Stage1); elipsa = fit vanjskog doublea.
    warp1_center: Optional[Tuple[float, float]] = None
    # Stage1 omjeri pojasa (inner/outer) za 4-ring Stage2; 0 = koristi RING_* default.
    stage2_double_inner_ratio: float = 0.0
    stage2_triple_inner_ratio: float = 0.0

    def is_valid(self) -> bool:
        return self.confidence >= 0.42 and self.bull_radius > 1.5

    def has_board_ellipse(self) -> bool:
        return self.double_ellipse is not None and self.board_confidence >= 0.35

    def has_wires(self) -> bool:
        return len(self.wire_angles_deg) >= SEGMENT_COUNT - 2

    def has_measured_rings(self) -> bool:
        return self.warp_double_outer is not None or self.measured_double_outer_frac >= 0.45

    def has_warp_ring_ellipses(self) -> bool:
        return (
            self.warp_double_outer is not None
            and self.warp_triple_outer is not None
            and self.warp_ring_size > 0
        )

    def has_ring_align_warp(self) -> bool:
        return self.warp1_double_outer is not None and self.warp_ring_size > 0


def _is_calibration_complete(cal: Optional[BoardCalibration]) -> bool:
    return (
        cal is not None
        and cal.is_valid()
        and cal.has_board_ellipse()
        and cal.has_wires()
    )


def _calibration_quality(cal: Optional[BoardCalibration]) -> float:
    """Gruba ocjena za usporedbu kandidata (veće = bolje)."""
    if cal is None or not cal.is_valid():
        return -1.0
    q = float(cal.confidence) * 2.0 + float(cal.board_confidence)
    if cal.has_wires():
        q += 0.05 * float(len(cal.wire_angles_deg))
    if cal.has_board_ellipse():
        q += 0.8
    if cal.has_ring_align_warp():
        q += 0.4
    return q


def _bull_shift_px(a: BoardCalibration, b: BoardCalibration) -> float:
    try:
        return float(
            math.hypot(
                float(a.center[0]) - float(b.center[0]),
                float(a.center[1]) - float(b.center[1]),
            )
        )
    except Exception:
        return 1e9


def _frame_diag(frame: np.ndarray) -> float:
    h, w = frame.shape[:2]
    return float(math.hypot(float(w), float(h)))


def calibration_file_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), CALIBRATION_FILENAME)


def normalized_ellipse_radius(ellipse: Ellipse, px: float, py: float) -> float:
    """<1 unutar elipse, 1 na rubu, >1 izvan (double outer skala)."""
    (ecx, ecy), (width, height), angle_deg = ellipse
    a = max(2.0, float(width) * 0.5)
    b = max(2.0, float(height) * 0.5)
    rad = math.radians(float(angle_deg))
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    dx, dy = float(px) - ecx, float(py) - ecy
    xr = cos_a * dx + sin_a * dy
    yr = -sin_a * dx + cos_a * dy
    return float(math.sqrt((xr / a) ** 2 + (yr / b) ** 2))


def _resample_polyline(pts: np.ndarray, n: int) -> np.ndarray:
    """pts (N,2) → n točaka duž zatvorenog oboda."""
    if pts is None or len(pts) < 2:
        return pts
    closed = np.vstack([pts, pts[0:1]])
    segs = np.sqrt(((closed[1:] - closed[:-1]) ** 2).sum(axis=1))
    total = float(segs.sum())
    if total < 1e-3:
        return pts.astype(np.float32, copy=False)
    d = np.concatenate([[0.0], np.cumsum(segs)])
    out: List[np.ndarray] = []
    for i in range(max(5, int(n))):
        t = (float(i) / float(n)) * total
        j = int(np.searchsorted(d, t, side="right") - 1)
        j = max(0, min(len(segs) - 1, j))
        span = float(segs[j]) if segs[j] > 1e-6 else 1.0
        u = (t - float(d[j])) / span
        out.append(closed[j] * (1.0 - u) + closed[j + 1] * u)
    return np.asarray(out, dtype=np.float32)


def _ellipse_from_hint_points(
    points: List[Tuple[float, float]],
    min_dim: float,
    *,
    strict: bool = True,
) -> Optional[Ellipse]:
    """Elipsa kroz 3–4 ručne točke na vanjskom rubu (ne kroz tetive)."""
    pts = [
        (float(p[0]), float(p[1]))
        for p in points
        if p is not None and len(p) >= 2
    ]
    if len(pts) < 3:
        return None
    arr = np.array(pts, dtype=np.float32)
    try:
        hull = cv2.convexHull(arr.reshape(-1, 1, 2)).reshape(-1, 2)
    except cv2.error:
        hull = arr
    if hull is None or len(hull) < 3:
        return None
    center = hull.mean(axis=0)
    extras: List[np.ndarray] = []
    n = len(hull)
    for i in range(n):
        a = hull[i]
        b = hull[(i + 1) % n]
        ra = float(np.linalg.norm(a - center))
        rb = float(np.linalg.norm(b - center))
        mid = (a + b) * 0.5
        vm = mid - center
        rm = float(np.linalg.norm(vm))
        if rm < 1e-3:
            continue
        # Točka na luku (isti radijus kao vrhovi), ne na tetivi unutar elipse.
        extras.append(center + vm * ((0.5 * (ra + rb)) / rm))
    stacked = np.repeat(hull, 12, axis=0)
    if extras:
        stacked = np.vstack([stacked, np.asarray(extras, dtype=np.float32)])
    if len(stacked) < 5:
        stacked = np.repeat(hull, 5, axis=0)
    try:
        ellipse = cv2.fitEllipse(stacked.reshape(-1, 1, 2))
    except cv2.error:
        return None
    ners = [normalized_ellipse_radius(ellipse, float(p[0]), float(p[1])) for p in pts]
    mean_n = float(sum(ners) / max(1, len(ners)))
    if mean_n > 1e-4:
        ellipse = _scale_ellipse_axes(ellipse, mean_n)
    (_ecx, _ecy), (ew, eh), _ang = ellipse
    semi_max = 0.5 * max(float(ew), float(eh))
    semi_min = 0.5 * min(float(ew), float(eh))
    if strict:
        if semi_max < min_dim * 0.10 or semi_max > min_dim * 1.25:
            return None
        if semi_min < min_dim * 0.06:
            return None
    elif semi_max < 4.0 or semi_min < 2.0:
        return None
    return ellipse


def standard_ring_radii(double_outer_r: float) -> Dict[str, float]:
    """Fiksni radijusi prstenova na top-down slici (relativno na vanjski double rub)."""
    d = float(double_outer_r)
    return {
        "double_outer": d * RING_DOUBLE_OUTER_FRAC,
        "double_inner": d * RING_DOUBLE_INNER_FRAC,
        "triple_outer": d * RING_TRIPLE_OUTER_FRAC,
        "triple_inner": d * RING_TRIPLE_INNER_FRAC,
        "bull_outer": d * RING_BULL_OUTER_FRAC,
        "bull_inner": d * RING_BULL_INNER_FRAC,
        "sisal_edge": d * BOARD_SISAL_EDGE_FRAC,
    }


def scoring_ring_radii(
    out_size: int,
    cal: Optional["BoardCalibration"] = None,
    cals: Optional[List[Optional["BoardCalibration"]]] = None,
) -> Dict[str, float]:
    """Radijusi za scoring/overlay: izmjereni na warpu ako postoje, inače standardni omjeri."""
    _, _, board_r = _warp_radii(int(out_size))
    fracs: List[Tuple[float, float, float, float]] = []
    candidates: List[Optional[BoardCalibration]] = []
    if cal is not None:
        candidates.append(cal)
    if cals:
        candidates.extend(cals)
    for c in candidates:
        if c is not None and c.measured_double_outer_frac >= 0.45:
            fracs.append(
                (
                    float(c.measured_double_outer_frac),
                    float(c.measured_double_inner_frac),
                    float(c.measured_triple_outer_frac),
                    float(c.measured_triple_inner_frac),
                )
            )
    if fracs:
        arr = np.median(np.asarray(fracs, dtype=np.float64), axis=0)
        d_out, d_in, t_out, t_in = (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
        rings = {
            "double_outer": board_r * d_out,
            "double_inner": board_r * d_in,
            "triple_outer": board_r * t_out,
            "triple_inner": board_r * t_in,
            "bull_outer": board_r * RING_BULL_OUTER_FRAC,
            "bull_inner": board_r * RING_BULL_INNER_FRAC,
            "sisal_edge": board_r * BOARD_SISAL_EDGE_FRAC,
        }
    else:
        rings = standard_ring_radii(board_r)
    return rings


def _ellipse_to_dict(ellipse: Ellipse) -> dict:
    (cx, cy), (w, h), ang = ellipse
    return {"cx": float(cx), "cy": float(cy), "w": float(w), "h": float(h), "angle": float(ang)}


def _ellipse_from_dict(data: dict) -> Optional[Ellipse]:
    try:
        return (
            (float(data["cx"]), float(data["cy"])),
            (float(data["w"]), float(data["h"])),
            float(data["angle"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _scale_ellipse_axes(ellipse: Ellipse, frac: float) -> Ellipse:
    (cx, cy), (w, h), ang = ellipse
    f = float(frac)
    return ((cx, cy), (w * f, h * f), ang)


def _resize_ellipse(ellipse: Ellipse, scale: float) -> Ellipse:
    (cx, cy), (w, h), ang = ellipse
    s = float(scale)
    return ((cx * s, cy * s), (w * s, h * s), ang)


def _mean_ellipse_semi(ellipse: Ellipse) -> float:
    (_, (w, h), _) = ellipse
    return 0.25 * (float(w) + float(h))


def _average_ellipses(ellipses: List[Ellipse]) -> Optional[Ellipse]:
    if not ellipses:
        return None
    if len(ellipses) == 1:
        return ellipses[0]
    cxs = [e[0][0] for e in ellipses]
    cys = [e[0][1] for e in ellipses]
    ws = [e[1][0] for e in ellipses]
    hs = [e[1][1] for e in ellipses]
    # Kut elipse je 0..180; prosjek preko unit vektora *2.
    angs = [math.radians(float(e[2]) * 2.0) for e in ellipses]
    ax = float(np.mean([math.cos(a) for a in angs]))
    ay = float(np.mean([math.sin(a) for a in angs]))
    ang = math.degrees(0.5 * math.atan2(ay, ax)) % 180.0
    return (
        (float(np.median(cxs)), float(np.median(cys))),
        (float(np.median(ws)), float(np.median(hs))),
        float(ang),
    )


def scoring_ring_ellipses(
    out_size: int,
    cal: Optional["BoardCalibration"] = None,
    cals: Optional[List[Optional["BoardCalibration"]]] = None,
) -> Optional[Dict[str, Ellipse]]:
    """Jajaste elipse double/triple na warpu (skalirane na out_size)."""
    size = int(out_size)
    candidates: List[BoardCalibration] = []
    if cal is not None and cal.has_warp_ring_ellipses():
        candidates.append(cal)
    if cals:
        for c in cals:
            if c is not None and c.has_warp_ring_ellipses():
                candidates.append(c)
    if not candidates:
        return None

    d_outs: List[Ellipse] = []
    d_ins: List[Ellipse] = []
    t_outs: List[Ellipse] = []
    t_ins: List[Ellipse] = []
    for c in candidates:
        s = float(size) / float(max(1, c.warp_ring_size))
        assert c.warp_double_outer is not None and c.warp_triple_outer is not None
        d_out = _resize_ellipse(c.warp_double_outer, s)
        t_out = _resize_ellipse(c.warp_triple_outer, s)
        if c.warp_double_inner is not None:
            d_in = _resize_ellipse(c.warp_double_inner, s)
        else:
            d_in = _scale_ellipse_axes(d_out, RING_DOUBLE_INNER_FRAC * WARP_RING_INNER_TIGHTEN)
        if c.warp_triple_inner is not None:
            t_in = _resize_ellipse(c.warp_triple_inner, s)
        else:
            t_in = _scale_ellipse_axes(
                t_out,
                (RING_TRIPLE_INNER_FRAC / max(RING_TRIPLE_OUTER_FRAC, 1e-6))
                * WARP_RING_INNER_TIGHTEN,
            )
        d_outs.append(d_out)
        d_ins.append(d_in)
        t_outs.append(t_out)
        t_ins.append(t_in)

    d_out_a = _average_ellipses(d_outs)
    d_in_a = _average_ellipses(d_ins)
    t_out_a = _average_ellipses(t_outs)
    t_in_a = _average_ellipses(t_ins)
    if d_out_a is None or d_in_a is None or t_out_a is None or t_in_a is None:
        return None
    return {
        "double_outer": d_out_a,
        "double_inner": d_in_a,
        "triple_outer": t_out_a,
        "triple_inner": t_in_a,
    }


def topdown_point_to_score(
    px: float,
    py: float,
    center_xy: Tuple[float, float],
    double_outer_r: float,
    segment20_offset: int = 0,
    rings: Optional[Dict[str, float]] = None,
    ring_ellipses: Optional[Dict[str, Ellipse]] = None,
) -> Tuple[int, str, int]:
    """Top-down tocka -> (segment_number, zone_name, score)."""
    cx, cy = center_xy
    dx, dy = float(px) - cx, float(py) - cy
    r = math.hypot(dx, dy)

    if ring_ellipses is not None:
        d_out = ring_ellipses["double_outer"]
        d_in = ring_ellipses["double_inner"]
        t_out = ring_ellipses["triple_outer"]
        t_in = ring_ellipses["triple_inner"]
        n_do = normalized_ellipse_radius(d_out, px, py)
        n_di = normalized_ellipse_radius(d_in, px, py)
        n_to = normalized_ellipse_radius(t_out, px, py)
        n_ti = normalized_ellipse_radius(t_in, px, py)
        sisal = _scale_ellipse_axes(d_out, BOARD_SISAL_EDGE_FRAC)
        if normalized_ellipse_radius(sisal, px, py) > 1.10:
            return 0, "miss", 0

        mean_d = _mean_ellipse_semi(d_out)
        bull_inner = mean_d * RING_BULL_INNER_FRAC
        bull_outer = mean_d * RING_BULL_OUTER_FRAC
        # Segment kut od centra warpa (seg.20 gore).
        theta = math.degrees(math.atan2(dx, -dy)) % 360.0
        seg_slot = int(round(theta / SEGMENT_STEP_DEG)) % SEGMENT_COUNT
        number = BOARD_SEGMENT_NUMBERS[(seg_slot - int(segment20_offset)) % SEGMENT_COUNT]

        if r <= bull_inner:
            return 50, "inner_bull", 50
        if r <= bull_outer:
            return 25, "outer_bull", 25
        if n_do > 1.0:
            return 0, "miss", 0
        if n_ti >= 1.0 and n_to <= 1.0:
            return number, "triple", number * 3
        if n_di >= 1.0 and n_do <= 1.0:
            return number, "double", number * 2
        if n_to > 1.0 and n_di < 1.0:
            return number, "single", number
        if n_ti < 1.0:
            return number, "single", number
        return 0, "miss", 0

    if double_outer_r <= 1e-6 and rings is None:
        return 0, "miss", 0

    use_rings = rings if rings is not None else standard_ring_radii(double_outer_r)
    if r > use_rings["sisal_edge"] * 1.10:
        return 0, "miss", 0

    if r <= use_rings["bull_inner"]:
        return 50, "inner_bull", 50
    if r <= use_rings["bull_outer"]:
        return 25, "outer_bull", 25

    theta = math.degrees(math.atan2(dx, -dy)) % 360.0
    seg_slot = int(round(theta / SEGMENT_STEP_DEG)) % SEGMENT_COUNT
    number = BOARD_SEGMENT_NUMBERS[(seg_slot - int(segment20_offset)) % SEGMENT_COUNT]

    if use_rings["triple_inner"] <= r <= use_rings["triple_outer"]:
        return number, "triple", number * 3
    if use_rings["double_inner"] <= r <= use_rings["double_outer"]:
        return number, "double", number * 2
    if use_rings["bull_outer"] < r < use_rings["triple_inner"]:
        return number, "single", number
    if use_rings["triple_outer"] < r < use_rings["double_inner"]:
        return number, "single", number
    return 0, "miss", 0


def _warp_radii(out_size: int) -> Tuple[float, float, float]:
    """(centar, view_radius, board_radius) za top-down warp."""
    cx_out = (out_size - 1) * 0.5
    view_r = cx_out * WARP_VIEW_RADIUS_FRAC
    board_r = view_r * WARP_BOARD_RADIUS_FRAC
    return cx_out, view_r, board_r


def _debug_dir_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_edges")


def _ensure_debug_dir_exists() -> None:
    d = _debug_dir_path()
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        pass


def _paint_board_wire_overlay(img: np.ndarray, cal: BoardCalibration) -> None:
    """In-place: plava elipsa doublea, žice bez točaka, bull točka, seg.20 fill."""
    cx, cy = int(round(cal.center[0])), int(round(cal.center[1]))
    br = max(2, int(round(cal.bull_radius)))
    color = _WARP_OVERLAY_BGR

    if cal.has_board_ellipse() and cal.double_ellipse is not None:
        ellipse = cal.double_ellipse
        (ecx, ecy), (ew, eh), angle = ellipse
        ax = max(1, int(round(ew * 0.5)))
        ay = max(1, int(round(eh * 0.5)))
        center_pt = (int(round(ecx)), int(round(ecy)))
        cv2.ellipse(img, center_pt, (ax, ay), angle, 0, 360, color, 2, cv2.LINE_AA)
        if cal.has_wires():
            if cal.segment20_wire_sector >= 0:
                _draw_wire_sector_wedge(
                    img,
                    cal.center,
                    ellipse,
                    cal.wire_angles_deg,
                    cal.segment20_wire_sector,
                    color,
                    wire_end=cal.segment20_wire_end if cal.segment20_wire_end >= 0 else None,
                    draw_edges=False,
                )
            _draw_wires_to_ellipse(
                img,
                cal.center,
                ellipse,
                cal.wire_angles_deg,
                line_color=color,
                dot_color=color,
                line_thickness=1,
                draw_dots=False,
            )

    cv2.circle(img, (cx, cy), max(2, br // 3 + 1), color, -1, cv2.LINE_AA)


def _save_debug_calibration_pack(
    *,
    cam_idx: Optional[int],
    frame_bgr: np.ndarray,
    cal: BoardCalibration,
) -> None:
    """Sprema: canny_edges, wires, masked (boja), topdown_warp (prepisuje stare)."""
    if not DEBUG_SAVE_CANNY_EDGES or cal.double_ellipse is None:
        return
    _ensure_debug_dir_exists()

    prefix = f"cam_{cam_idx}" if cam_idx is not None else "cam_unknown"
    base = os.path.join(_debug_dir_path(), prefix)
    ellipse = cal.double_ellipse

    try:
        gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edges_masked = _canny_edge_map(gray_full, ellipse)
        edges_u8 = edges_masked.astype(np.uint8)
        cv2.imwrite(base + "_canny_edges.png", cv2.cvtColor(edges_u8, cv2.COLOR_GRAY2BGR))

        mask = _board_ellipse_mask(frame_bgr.shape[:2], ellipse, shrink=0.99)
        masked_color = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)
        cv2.imwrite(base + "_masked.png", masked_color)

        img_wires = frame_bgr.copy()
        _paint_board_wire_overlay(img_wires, cal)
        cv2.imwrite(base + "_wires.png", img_wires)

        save_topdown_warp_image(
            cam_idx,
            frame_bgr,
            bull=cal.center,
            bull_r=cal.bull_radius,
            ellipse=ellipse,
            wire_angles_deg=cal.wire_angles_deg,
            segment20_offset=cal.segment20_offset,
            segment20_wire_sector=cal.segment20_wire_sector,
            cal=cal,
        )
    except Exception:
        pass


def save_topdown_warp_image(
    cam_idx: Optional[int],
    frame_bgr: np.ndarray,
    *,
    bull: Tuple[float, float],
    bull_r: float,
    ellipse: Ellipse,
    wire_angles_deg: Tuple[float, ...] = (),
    segment20_offset: int = 0,
    segment20_wire_sector: int = -1,
    out_size: int = DEBUG_WARP_SIZE,
    draw_segment20: bool = True,
    cal: Optional[BoardCalibration] = None,
) -> bool:
    """Sprema top-down pregled (seg.20 gore) kao cam_{N}_topdown_warp.png (prepis)."""
    _ensure_debug_dir_exists()
    prefix = f"cam_{cam_idx}" if cam_idx is not None else "cam_unknown"
    path = os.path.join(_debug_dir_path(), prefix + "_topdown_warp.png")
    try:
        warped: Optional[np.ndarray] = None
        if cal is not None and cal.has_board_ellipse():
            warped = warp_topdown_from_calibration(frame_bgr, cal, out_size=out_size)
        if warped is None:
            warped, _bull_w, _bull_r_w, _double_r_w, _shift = warp_board_topdown(
                frame_bgr,
                bull,
                bull_r,
                ellipse,
                out_size,
                wire_angles_deg=wire_angles_deg,
                segment20_offset=segment20_offset,
            )
        # Detektiraj bull PRIJE overlaya (inače plava/ boje poremete HSV).
        bull_pos = detect_bull_center_on_warp(warped)
        clean = warped.copy()
        if draw_segment20 and segment20_wire_sector >= 0:
            _draw_segment20_on_aligned_warp(warped)
        _draw_scoring_rings_on_aligned_warp(
            warped, cal=cal, bull_center=bull_pos, source_bgr=clean
        )
        cv2.imwrite(path, warped)
        return True
    except Exception:
        return False


def save_topdown_warp_from_calibration(
    cam_idx: Optional[int],
    frame_bgr: np.ndarray,
    cal: BoardCalibration,
    *,
    out_size: int = DEBUG_WARP_SIZE,
    draw_segment20: bool = True,
) -> bool:
    if not cal.is_valid() or cal.double_ellipse is None:
        return False
    return save_topdown_warp_image(
        cam_idx,
        frame_bgr,
        bull=cal.center,
        bull_r=cal.bull_radius,
        ellipse=cal.double_ellipse,
        wire_angles_deg=cal.wire_angles_deg if cal.has_wires() else (),
        segment20_offset=cal.segment20_offset,
        segment20_wire_sector=cal.segment20_wire_sector,
        out_size=out_size,
        draw_segment20=draw_segment20,
        cal=cal,
    )


def _calibration_to_dict(cam_idx: int, cal: BoardCalibration) -> dict:
    data: dict = {
        "cam_idx": cam_idx,
        "center": [cal.center[0], cal.center[1]],
        "bull_radius": cal.bull_radius,
        "confidence": cal.confidence,
        "board_confidence": cal.board_confidence,
        "wire_angles_deg": list(cal.wire_angles_deg),
    }
    if cal.double_ellipse is not None:
        (ecx, ecy), (ew, eh), ang = cal.double_ellipse
        data["ellipse"] = {"cx": ecx, "cy": ecy, "w": ew, "h": eh, "angle": ang}
    data["homography_shift"] = cal.homography_shift
    data["align_rotation_deg"] = cal.align_rotation_deg
    data["segment20_offset"] = cal.segment20_offset
    data["segment20_wire_sector"] = cal.segment20_wire_sector
    data["segment20_wire_end"] = cal.segment20_wire_end
    if cal.measured_double_outer_frac >= 0.45 or cal.has_warp_ring_ellipses():
        data["measured_rings"] = {
            "double_outer_frac": cal.measured_double_outer_frac,
            "double_inner_frac": cal.measured_double_inner_frac,
            "triple_outer_frac": cal.measured_triple_outer_frac,
            "triple_inner_frac": cal.measured_triple_inner_frac,
        }
    if cal.has_warp_ring_ellipses() or cal.has_ring_align_warp():
        wr: dict = {"size": int(cal.warp_ring_size)}
        if cal.warp_double_outer is not None:
            wr["double_outer"] = _ellipse_to_dict(cal.warp_double_outer)
        if cal.warp_double_inner is not None:
            wr["double_inner"] = _ellipse_to_dict(cal.warp_double_inner)
        if cal.warp_triple_outer is not None:
            wr["triple_outer"] = _ellipse_to_dict(cal.warp_triple_outer)
        if cal.warp_triple_inner is not None:
            wr["triple_inner"] = _ellipse_to_dict(cal.warp_triple_inner)
        if cal.warp1_double_outer is not None:
            wr["warp1_double_outer"] = _ellipse_to_dict(cal.warp1_double_outer)
        if cal.warp1_triple_outer is not None:
            wr["warp1_triple_outer"] = _ellipse_to_dict(cal.warp1_triple_outer)
        if cal.warp1_center is not None:
            wr["warp1_center"] = [float(cal.warp1_center[0]), float(cal.warp1_center[1])]
        if cal.stage2_double_inner_ratio > 0.5:
            wr["stage2_double_inner_ratio"] = float(cal.stage2_double_inner_ratio)
        if cal.stage2_triple_inner_ratio > 0.5:
            wr["stage2_triple_inner_ratio"] = float(cal.stage2_triple_inner_ratio)
        data["warp_rings"] = wr
    return data


def _calibration_from_dict(data: dict) -> Optional[BoardCalibration]:
    try:
        ellipse: Optional[Ellipse] = None
        if "ellipse" in data:
            e = data["ellipse"]
            ellipse = ((float(e["cx"]), float(e["cy"])), (float(e["w"]), float(e["h"])), float(e["angle"]))
        wires = tuple(float(x) for x in data.get("wire_angles_deg", ()))
        c = data["center"]
        mr = data.get("measured_rings") or {}
        wr = data.get("warp_rings") or {}
        return BoardCalibration(
            center=(float(c[0]), float(c[1])),
            bull_radius=float(data["bull_radius"]),
            confidence=float(data.get("confidence", 0.0)),
            double_ellipse=ellipse,
            board_confidence=float(data.get("board_confidence", 0.0)),
            wire_angles_deg=wires,
            homography_shift=int(data.get("homography_shift", 0)),
            align_rotation_deg=float(data.get("align_rotation_deg", 0.0)),
            segment20_offset=int(data.get("segment20_offset", 0)),
            segment20_wire_sector=int(data.get("segment20_wire_sector", -1)),
            segment20_wire_end=int(data.get("segment20_wire_end", -1)),
            measured_double_outer_frac=float(mr.get("double_outer_frac", 0.0)),
            measured_double_inner_frac=float(mr.get("double_inner_frac", 0.0)),
            measured_triple_outer_frac=float(mr.get("triple_outer_frac", 0.0)),
            measured_triple_inner_frac=float(mr.get("triple_inner_frac", 0.0)),
            warp_ring_size=int(wr.get("size", 0) or 0),
            warp_double_outer=_ellipse_from_dict(wr["double_outer"]) if wr.get("double_outer") else None,
            warp_double_inner=_ellipse_from_dict(wr["double_inner"]) if wr.get("double_inner") else None,
            warp_triple_outer=_ellipse_from_dict(wr["triple_outer"]) if wr.get("triple_outer") else None,
            warp_triple_inner=_ellipse_from_dict(wr["triple_inner"]) if wr.get("triple_inner") else None,
            warp1_double_outer=(
                _ellipse_from_dict(wr["warp1_double_outer"]) if wr.get("warp1_double_outer") else None
            ),
            warp1_triple_outer=(
                _ellipse_from_dict(wr["warp1_triple_outer"]) if wr.get("warp1_triple_outer") else None
            ),
            warp1_center=(
                (float(wr["warp1_center"][0]), float(wr["warp1_center"][1]))
                if isinstance(wr.get("warp1_center"), (list, tuple)) and len(wr["warp1_center"]) >= 2
                else None
            ),
            stage2_double_inner_ratio=float(wr.get("stage2_double_inner_ratio", 0.0) or 0.0),
            stage2_triple_inner_ratio=float(wr.get("stage2_triple_inner_ratio", 0.0) or 0.0),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _board_color_masks(hsv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Red, green, black sisal, white segment masks."""
    red = cv2.inRange(hsv, (0, 95, 70), (10, 255, 255)) | cv2.inRange(hsv, (168, 95, 70), (180, 255, 255))
    green = cv2.inRange(hsv, (40, 50, 40), (88, 255, 255))
    white = cv2.inRange(hsv, (0, 0, 165), (180, 55, 255))
    black = cv2.inRange(hsv, (0, 0, 0), (180, 255, 70))
    return red, green, black, white


def _bull_color_masks(hsv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    red, green, _, _ = _board_color_masks(hsv)
    return red, green


def _ring_fraction(mask: np.ndarray, cx: float, cy: float, r_in: float, r_out: float) -> float:
    h, w = mask.shape[:2]
    cx_i, cy_i = int(round(cx)), int(round(cy))
    ri, ro = max(0, int(r_in)), max(2, int(r_out))
    if ro <= ri:
        ro = ri + 2
    ring = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(ring, (cx_i, cy_i), ro, 255, -1)
    if ri > 0:
        cv2.circle(ring, (cx_i, cy_i), ri, 0, -1)
    total = int(cv2.countNonZero(ring))
    if total < 6:
        return 0.0
    return cv2.countNonZero(cv2.bitwise_and(mask, ring)) / float(total)


def _refine_red_centroid(red: np.ndarray, cx: float, cy: float, radius: float) -> Tuple[float, float]:
    h, w = red.shape[:2]
    r = max(3, int(radius * 2.0))
    cx_i, cy_i = int(round(cx)), int(round(cy))
    x1, y1 = max(0, cx_i - r), max(0, cy_i - r)
    x2, y2 = min(w, cx_i + r), min(h, cy_i + r)
    roi = red[y1:y2, x1:x2]
    if roi.size == 0:
        return cx, cy
    moments = cv2.moments(roi)
    if moments["m00"] < 10:
        return cx, cy
    return (moments["m10"] / moments["m00"] + x1, moments["m01"] / moments["m00"] + y1)


def _red_spill(red: np.ndarray, cx: float, cy: float, core_r: float) -> float:
    return _ring_fraction(red, cx, cy, core_r * 1.6, core_r * 7.0)


def _bull_signature(
    red: np.ndarray,
    green: np.ndarray,
    cx: float,
    cy: float,
    core_r: float,
    *,
    relaxed: bool = False,
) -> Optional[float]:
    core_r = max(2.0, core_r)
    red_core = _ring_fraction(red, cx, cy, 0.0, core_r * 1.1)
    green_ring = _ring_fraction(green, cx, cy, core_r * 0.55, core_r * 2.8)
    spill = _red_spill(red, cx, cy, core_r)
    if relaxed:
        # Click-assist: weaker lighting / partial green ring still OK.
        if red_core < 0.10 and green_ring < 0.10:
            return None
        if spill > 0.28:
            return None
        if red_core < 0.08:
            return None
    elif red_core < 0.22 or green_ring < 0.14 or spill > 0.10:
        return None
    return (
        0.40 * min(red_core * 2.5, 1.0)
        + 0.40 * min(green_ring * 3.0, 1.0)
        + 0.20 * (1.0 - min(spill * 8.0, 1.0))
    )


def _bull_color_masks_relaxed(hsv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Looser R/G for click-assist under poor side lighting (cams 0/2)."""
    red = cv2.inRange(hsv, (0, 55, 40), (14, 255, 255)) | cv2.inRange(
        hsv, (162, 55, 40), (180, 255, 255)
    )
    green = cv2.inRange(hsv, (32, 25, 25), (98, 255, 255))
    return red, green


def _find_bull(
    red: np.ndarray,
    green: np.ndarray,
    min_dim: float,
    *,
    search_ellipse: Optional[Ellipse] = None,
    hint_xy: Optional[Tuple[float, float]] = None,
    relaxed: bool = False,
) -> Optional[Tuple[Tuple[float, float], float, float]]:
    max_area = max(80.0, (min_dim * (0.055 if relaxed else 0.045)) ** 2)
    candidates: List[Tuple[Tuple[float, float], float, float]] = []
    # Hint: prefer R/G bull candidates near click (seed), never force click as bull.
    hint_r = max(16.0, min_dim * (0.35 if relaxed else 0.20)) if hint_xy is not None else 0.0
    circ_min = 0.32 if relaxed else 0.45
    aspect_max = 2.8 if relaxed else 2.2
    inside_cut = 0.72 if relaxed else 0.55

    def _inside_board(px: float, py: float) -> bool:
        if search_ellipse is None:
            return True
        # Bull mora biti unutar RG elipse ploce (ne na rubu / vani).
        return normalized_ellipse_radius(search_ellipse, px, py) < inside_cut

    def _hint_boost(px: float, py: float) -> float:
        if hint_xy is None or hint_r <= 0.0:
            return 1.0
        dist = math.hypot(px - hint_xy[0], py - hint_xy[1])
        # 1.0 at hint → ~0.35 at ROI edge; far candidates stay usable but lose.
        return 0.35 + 0.65 * max(0.0, 1.0 - dist / hint_r)

    for cnt in cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(cnt)
        if area < (5.0 if relaxed else 8.0) or area > max_area:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim < 4.0:
            continue
        if 4.0 * math.pi * area / (perim * perim) < circ_min:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if max(bw, bh) / max(1, min(bw, bh)) > aspect_max:
            continue
        moments = cv2.moments(cnt)
        if moments["m00"] < 1.0:
            continue
        rcx, rcy = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
        if not _inside_board(rcx, rcy):
            continue
        core_r = max(2.0, math.sqrt(area / math.pi))
        if _bull_signature(red, green, rcx, rcy, core_r, relaxed=relaxed) is None:
            continue
        cx, cy = _refine_red_centroid(red, rcx, rcy, core_r)
        if not _inside_board(cx, cy):
            continue
        sig = _bull_signature(red, green, cx, cy, core_r, relaxed=relaxed)
        if sig is None:
            continue
        score = (sig * 0.75 + (1.0 - min(area / max_area, 1.0)) * 0.25) * _hint_boost(cx, cy)
        candidates.append(((cx, cy), core_r, float(score)))

    max_green = (min_dim * (0.11 if relaxed else 0.09)) ** 2
    red_core_min = 0.10 if relaxed else 0.18
    for cnt in cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(cnt)
        if area < (20.0 if relaxed else 35.0) or area > max_green or len(cnt) < 5:
            continue
        (ecx, ecy), (ma, mi), _ = cv2.fitEllipse(cnt)
        gcx, gcy = float(ecx), float(ecy)
        if not _inside_board(gcx, gcy):
            continue
        g_r = max(3.0, 0.25 * (ma + mi))
        if _ring_fraction(red, gcx, gcy, 0.0, g_r * 0.42) < red_core_min:
            continue
        cx, cy = _refine_red_centroid(red, gcx, gcy, g_r * 0.35)
        if not _inside_board(cx, cy):
            continue
        core_r = max(2.0, g_r * 0.22)
        sig = _bull_signature(red, green, cx, cy, core_r, relaxed=relaxed)
        if sig is None:
            continue
        score = (sig * 0.85 + (1.0 - min(area / max_green, 1.0)) * 0.15) * _hint_boost(cx, cy)
        candidates.append(((cx, cy), core_r, float(score)))

    if not candidates:
        return None

    merged: List[Tuple[Tuple[float, float], float, float]] = []
    for hit in sorted(candidates, key=lambda x: x[2], reverse=True):
        if any(math.hypot(hit[0][0] - m[0][0], hit[0][1] - m[0][1]) < min_dim * 0.04 for m in merged):
            continue
        merged.append(hit)
    return max(merged, key=lambda x: x[2])


def _hint_roi_mask(
    shape_hw: Tuple[int, int],
    hint_xy: Tuple[float, float],
    radius: float,
) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    cx = int(round(hint_xy[0]))
    cy = int(round(hint_xy[1]))
    r = max(8, int(round(radius)))
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask


def _max_ray_radius(h: int, w: int, cx: float, cy: float) -> float:
    """Max radijus zrake do najdaljeg kuta kadra (ploča smije biti pomaknuta)."""
    return float(math.hypot(max(cx, w - 1.0 - cx), max(cy, h - 1.0 - cy)))


def _find_largest_rg_board_ellipse(
    red: np.ndarray,
    green: np.ndarray,
    min_dim: float,
) -> Optional[Ellipse]:
    """Najveća R/G kontura ≈ vanjski rub ploce (ne pretpostavlja centar kadra)."""
    board = cv2.bitwise_or(red, green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    board = cv2.morphologyEx(board, cv2.MORPH_CLOSE, kernel)
    board = cv2.morphologyEx(board, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    contours = cv2.findContours(board, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if not contours:
        return None

    best: Optional[Ellipse] = None
    best_area = 0.0
    min_area = (min_dim * 0.12) ** 2
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or len(cnt) < 24:
            continue
        try:
            ellipse = cv2.fitEllipse(cnt)
        except cv2.error:
            continue
        (_ecx, _ecy), (ew, eh), _ang = ellipse
        semi_max = 0.5 * max(float(ew), float(eh))
        semi_min = 0.5 * min(float(ew), float(eh))
        # Gornji prag 0.95: ploča smije puniti kadar (prije 0.78 odbacivalo krupni kadar
        # → bez board_guess maske bull traži po cijelom frameu i češće padne).
        if semi_max < min_dim * 0.14 or semi_max > min_dim * 0.95:
            continue
        if semi_min < min_dim * 0.08:
            continue
        # Mora biti "ploča-like": dovoljno popunjena R/G unutar elipse.
        fill = _ring_fraction(board, float(ellipse[0][0]), float(ellipse[0][1]), 0.0, semi_min * 0.85)
        if fill < 0.20:
            continue
        if area > best_area:
            best_area = area
            best = ellipse
    return best


def _find_largest_board_ellipse_near_hint(
    red: np.ndarray,
    green: np.ndarray,
    black: np.ndarray,
    hint_xy: Tuple[float, float],
    min_dim: float,
    *,
    max_center_dist: Optional[float] = None,
) -> Optional[Ellipse]:
    """Largest dark/RG board-like ellipse whose interior covers (or is near) the click."""
    board = cv2.bitwise_or(cv2.bitwise_or(red, green), black)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    board = cv2.morphologyEx(board, cv2.MORPH_CLOSE, kernel)
    board = cv2.morphologyEx(board, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours = cv2.findContours(board, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if not contours:
        return None

    hx, hy = float(hint_xy[0]), float(hint_xy[1])
    dist_lim = float(max_center_dist) if max_center_dist is not None else max(40.0, min_dim * 0.40)
    best: Optional[Ellipse] = None
    best_score = -1.0
    min_area = (min_dim * 0.10) ** 2
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or len(cnt) < 20:
            continue
        try:
            ellipse = cv2.fitEllipse(cnt)
        except cv2.error:
            continue
        (ecx, ecy), (ew, eh), _ang = ellipse
        semi_max = 0.5 * max(float(ew), float(eh))
        semi_min = 0.5 * min(float(ew), float(eh))
        if semi_max < min_dim * 0.12 or semi_max > min_dim * 0.98:
            continue
        if semi_min < min_dim * 0.07:
            continue
        ner = normalized_ellipse_radius(ellipse, hx, hy)
        # Prefer ellipses that contain the click; allow near-miss centers.
        center_dist = math.hypot(float(ecx) - hx, float(ecy) - hy)
        if ner > 1.15 and center_dist > dist_lim:
            continue
        fill = _ring_fraction(board, float(ecx), float(ecy), 0.0, semi_min * 0.85)
        if fill < 0.12:
            continue
        contain = max(0.0, 1.15 - ner)
        score = area * (0.55 + 0.45 * contain) / (1.0 + center_dist / max(1.0, min_dim * 0.25))
        if score > best_score:
            best_score = score
            best = ellipse
    return best


def _estimate_board_radius_from_mask(
    red: np.ndarray,
    green: np.ndarray,
    bull: Tuple[float, float],
    bull_r: float,
) -> float:
    """Gruba procjena vanjskog ruba ploce (RG maska) za suzenje double uzorkovanja.

    Ne uzima 'najudaljeniji RG po zraci' — kad je double nevidljiv, to pada na triple
    i elipsa postane premala.
    """
    h, w = red.shape[:2]
    board = cv2.bitwise_or(red, green)
    cx, cy = bull
    max_r = _max_ray_radius(h, w, cx, cy) * 0.98
    radii = np.arange(bull_r * 8.0, max_r, 1.0, dtype=np.float32)
    if len(radii) < 8:
        return bull_r * 24.0
    angs = np.deg2rad(np.arange(0, 360, 8, dtype=np.float32))
    sin_a = np.sin(angs)
    cos_a = -np.cos(angs)
    counts = np.zeros(len(radii), dtype=np.float32)
    for i, r in enumerate(radii):
        xs = (cx + r * sin_a).astype(np.int32)
        ys = (cy + r * cos_a).astype(np.int32)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if not np.any(valid):
            counts[i] = 0.0
            continue
        counts[i] = float(np.mean(board[ys[valid], xs[valid]] > 0))
    occupied = counts > 0.22
    if not np.any(occupied):
        return bull_r * 24.0
    last_idx = int(np.where(occupied)[0][-1])
    return float(radii[last_idx])


def _double_ring_sample_radii(
    red: np.ndarray,
    green: np.ndarray,
    bull: Tuple[float, float],
    bull_r: float,
) -> Tuple[float, float]:
    """Unutarnji/vanjski radijus uzorkovanja za double prsten."""
    board_r = _estimate_board_radius_from_mask(red, green, bull, bull_r)
    double_r = board_r * (170.0 / 225.0)
    return double_r * DOUBLE_RING_SAMPLE_INNER, double_r * DOUBLE_RING_SAMPLE_OUTER


def _sample_double_ring_outer_points_legacy(
    red: np.ndarray,
    green: np.ndarray,
    cx: float,
    cy: float,
    bull_r: float,
) -> List[Tuple[float, float]]:
    """Vanjski RG rub po kutu — double prsten, ne brojčana zona / outlieri."""
    h, w = red.shape[:2]
    board = cv2.bitwise_or(red, green)
    cv2.circle(board, (int(round(cx)), int(round(cy))), max(2, int(bull_r * 3.5)), 0, -1)

    r_lo = bull_r * 7.0
    r_hi = _max_ray_radius(h, w, cx, cy) * 0.98
    per_angle: List[List[Tuple[float, float, float]]] = []

    for deg in range(0, 360, 4):
        theta = math.radians(deg)
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        in_run = False
        run_outer: Optional[Tuple[float, float, float]] = None
        clusters: List[Tuple[float, float, float]] = []

        for r in np.arange(r_lo, r_hi, 1.0):
            x = int(round(cx + r * sin_t))
            y = int(round(cy - r * cos_t))
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            hit = bool(board[y, x] > 0)
            if hit:
                in_run = True
                run_outer = (float(x), float(y), float(r))
            elif in_run:
                if run_outer is not None:
                    clusters.append(run_outer)
                in_run = False
                run_outer = None
        if in_run and run_outer is not None:
            clusters.append(run_outer)
        if clusters:
            per_angle.append(clusters)

    if not per_angle:
        return []

    # Tipični vanjski double rub; daleki outlieri = broj zona / šum.
    outer_rs = [cl[-1][2] for cl in per_angle]
    soft_cap = float(np.percentile(outer_rs, 70)) * 1.08

    outer_points: List[Tuple[float, float]] = []
    for clusters in per_angle:
        chosen = clusters[-1]
        if chosen[2] > soft_cap and len(clusters) >= 2:
            prev = clusters[-2]
            # Triple+double: uzmi vanjski; šum u brojevima: padni na prethodni.
            if abs(prev[2] - soft_cap) < abs(chosen[2] - soft_cap):
                chosen = prev
        elif chosen[2] > soft_cap * 1.12 and len(clusters) >= 2:
            chosen = clusters[-2]
        outer_points.append((chosen[0], chosen[1]))

    return outer_points


def _sample_ellipse_fit_points(
    red: np.ndarray,
    green: np.ndarray,
    cx: float,
    cy: float,
    bull_r: float,
) -> List[Tuple[float, float]]:
    """Točke za fit elipse: najudaljeniji RG po zraci (prati perspektivu, ne krug).

    Fiksni radijalni pojas daje krug u image prostoru i često sjeda na triple —
    zato legacy outermost ima prioritet.
    """
    outer_legacy = _sample_double_ring_outer_points_legacy(red, green, cx, cy, bull_r)
    if len(outer_legacy) >= 16:
        return outer_legacy

    h, w = red.shape[:2]
    board = cv2.bitwise_or(red, green)
    cv2.circle(board, (int(round(cx)), int(round(cy))), max(2, int(bull_r * 3.5)), 0, -1)

    r_lo, r_hi = _double_ring_sample_radii(red, green, (cx, cy), bull_r)

    points: List[Tuple[float, float]] = []
    for deg in range(0, 360, 2):
        theta = math.radians(deg)
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        outer_xy: Optional[Tuple[float, float]] = None
        for r in np.arange(r_lo, r_hi + 0.01, 0.6):
            x = int(round(cx + r * sin_t))
            y = int(round(cy - r * cos_t))
            if 0 <= x < w and 0 <= y < h and board[y, x] > 0:
                outer_xy = (float(x), float(y))
        if outer_xy is not None:
            points.append(outer_xy)

    if len(points) >= 20:
        return points
    if len(outer_legacy) >= 12:
        return outer_legacy
    return points


def _sample_double_ring_points_legacy(
    red: np.ndarray,
    green: np.ndarray,
    cx: float,
    cy: float,
    bull_r: float,
) -> List[Tuple[float, float]]:
    """Robusno uzorkovanje: radialni sken do vanjskog RG ruba."""
    h, w = red.shape[:2]
    board = cv2.bitwise_or(red, green)
    cv2.circle(board, (int(round(cx)), int(round(cy))), max(2, int(bull_r * 3.5)), 0, -1)

    outer_points = _sample_double_ring_outer_points_legacy(red, green, cx, cy, bull_r)
    band_points: List[Tuple[float, float]] = []

    for deg in range(0, 360, 4):
        theta = math.radians(deg)
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        outer_r: Optional[float] = None

        for r in np.arange(bull_r * 7.0, _max_ray_radius(h, w, cx, cy) * 0.98, 1.2):
            x = int(round(cx + r * sin_t))
            y = int(round(cy - r * cos_t))
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            if red[y, x] > 0 or green[y, x] > 0:
                outer_r = float(r)

        if outer_r is None:
            continue

        for r in np.arange(outer_r * 0.92, outer_r * 1.02, 0.6):
            x = int(round(cx + r * sin_t))
            y = int(round(cy - r * cos_t))
            if 0 <= x < w and 0 <= y < h and board[y, x] > 0:
                band_points.append((float(x), float(y)))

    if len(band_points) >= 24:
        return band_points
    if len(outer_points) >= 20:
        return outer_points
    return band_points


def _sample_double_ring_points(
    red: np.ndarray,
    green: np.ndarray,
    cx: float,
    cy: float,
    bull_r: float,
) -> List[Tuple[float, float]]:
    """Prvo uski double prsten, zatim legacy fallback ako nema dovoljno tocaka."""
    h, w = red.shape[:2]
    board = cv2.bitwise_or(red, green)
    cv2.circle(board, (int(round(cx)), int(round(cy))), max(2, int(bull_r * 3.5)), 0, -1)

    r_lo, r_hi = _double_ring_sample_radii(red, green, (cx, cy), bull_r)
    if r_hi - r_lo < bull_r * 1.5:
        r_mid = (r_lo + r_hi) * 0.5
        r_lo = r_mid * 0.92
        r_hi = r_mid * 1.08

    band_points: List[Tuple[float, float]] = []
    for deg in range(0, 360, 3):
        theta = math.radians(deg)
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        outer_xy: Optional[Tuple[float, float]] = None
        for r in np.arange(r_lo, r_hi + 0.01, 0.8):
            x = int(round(cx + r * sin_t))
            y = int(round(cy - r * cos_t))
            if 0 <= x < w and 0 <= y < h and board[y, x] > 0:
                outer_xy = (float(x), float(y))
        if outer_xy is not None:
            band_points.append(outer_xy)

    legacy_points = _sample_double_ring_points_legacy(red, green, cx, cy, bull_r)
    if len(band_points) >= 12 and len(legacy_points) >= 12:
        return band_points + legacy_points
    if len(legacy_points) >= len(band_points):
        return legacy_points
    if len(band_points) >= 12:
        return band_points
    return legacy_points


def _contour_double_points(
    red: np.ndarray,
    green: np.ndarray,
    bull: Tuple[float, float],
    bull_r: float,
) -> List[Tuple[float, float]]:
    board = cv2.bitwise_or(red, green)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    board = cv2.morphologyEx(board, cv2.MORPH_CLOSE, k)
    cv2.circle(board, (int(round(bull[0])), int(round(bull[1]))), max(2, int(bull_r * 3.5)), 0, -1)

    r_lo, r_hi = _double_ring_sample_radii(red, green, bull, bull_r)
    r_lo2 = r_lo * 0.94
    r_hi2 = r_hi * 1.06

    contours, _ = cv2.findContours(board, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best: Optional[np.ndarray] = None
    best_area = 0.0
    for cnt in contours:
        if len(cnt) < 20 or cv2.pointPolygonTest(cnt, bull, False) < 0:
            continue
        area = cv2.contourArea(cnt)
        if area > best_area:
            best_area = area
            best = cnt

    if best is None:
        return []
    bx, by = bull
    pts: List[Tuple[float, float]] = []
    step = max(1, len(best) // 100)
    for p in best[::step]:
        px, py = float(p[0][0]), float(p[0][1])
        r = math.hypot(px - bx, py - by)
        if r_lo2 <= r <= r_hi2:
            pts.append((px, py))
    if len(pts) >= 12:
        return pts
    return [(float(p[0][0]), float(p[0][1])) for p in best[::max(1, len(best) // 80)]]


def _ellipse_point_distance(ellipse: Ellipse, px: float, py: float) -> float:
    (ecx, ecy), (width, height), angle_deg = ellipse
    a = max(2.0, width * 0.5)
    b = max(2.0, height * 0.5)
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    dx, dy = px - ecx, py - ecy
    xr = cos_a * dx + sin_a * dy
    yr = -sin_a * dx + cos_a * dy
    return abs(1.0 - math.sqrt((xr / a) ** 2 + (yr / b) ** 2)) * max(a, b)


def _scale_ellipse(ellipse: Ellipse, scale: float) -> Ellipse:
    (ecx, ecy), (width, height), angle = ellipse
    return ((ecx, ecy), (width * scale, height * scale), angle)


def _ellipse_mean_semi(ellipse: Ellipse) -> float:
    (_, (width, height), _) = ellipse
    return 0.25 * (float(width) + float(height))


def _refine_ellipse_to_outer_double(
    ellipse: Ellipse,
    bull: Tuple[float, float],
    bull_r: float,
    red: np.ndarray,
    green: np.ndarray,
) -> Ellipse:
    """Poravnaj elipsu na VANJSKI rub double RG boje (prava ploča, bez 2D outside-scale)."""
    outer_rs = _farthest_rg_radii_on_rays(red, green, bull, bull_r)
    semi = _ellipse_mean_semi(ellipse)
    if semi < 1.0:
        return ellipse

    if len(outer_rs) >= 12:
        # 70. percentil = vanjski rub double boje; cap sprječava brojčanu zonu.
        target = float(np.percentile(outer_rs, 70))
        soft_cap = float(np.percentile(outer_rs, 55)) * 1.10
        target = min(target, soft_cap)
        scale = target / semi
        # Samo fino poravnanje — ne širi izvan doublea.
        scale = float(np.clip(scale, 0.94, 1.08))
        return _scale_ellipse(ellipse, scale)

    return ellipse


def _ellipse_ovality(ellipse: Ellipse) -> float:
    """0 = krug; veće = jajastije (|a-b|/max)."""
    _c, (w, h), _a = ellipse
    a = max(float(w), float(h))
    b = min(float(w), float(h))
    a = max(a, 1e-6)
    b = max(b, 1e-6)
    return float(abs(a - b) / a)


def _farthest_rg_radii_on_rays(
    red: np.ndarray,
    green: np.ndarray,
    bull: Tuple[float, float],
    bull_r: float,
    step_deg: int = 5,
) -> List[float]:
    """Vanjski rub double RG klastera po zraci (bez outliera u brojevima)."""
    h, w = red.shape[:2]
    board = cv2.bitwise_or(red, green)
    cx, cy = bull
    r_lo = bull_r * 6.0
    r_hi = _max_ray_radius(h, w, cx, cy) * 0.98
    per_angle: List[List[float]] = []
    for deg in range(0, 360, step_deg):
        theta = math.radians(float(deg))
        st, ct = math.sin(theta), -math.cos(theta)
        in_run = False
        run_outer: Optional[float] = None
        clusters: List[float] = []
        for r in np.arange(r_lo, r_hi, 1.0):
            x = int(round(cx + r * st))
            y = int(round(cy + r * ct))
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            hit = bool(board[y, x] > 0)
            if hit:
                in_run = True
                run_outer = float(r)
            elif in_run:
                if run_outer is not None:
                    clusters.append(run_outer)
                in_run = False
                run_outer = None
        if in_run and run_outer is not None:
            clusters.append(run_outer)
        if clusters:
            per_angle.append(clusters)

    if not per_angle:
        return []

    raw_outer = [cl[-1] for cl in per_angle]
    soft_cap = float(np.percentile(raw_outer, 70)) * 1.08
    out: List[float] = []
    for clusters in per_angle:
        chosen = clusters[-1]
        if chosen > soft_cap and len(clusters) >= 2:
            prev = clusters[-2]
            if abs(prev - soft_cap) < abs(chosen - soft_cap):
                chosen = prev
        elif chosen > soft_cap * 1.12 and len(clusters) >= 2:
            chosen = clusters[-2]
        out.append(chosen)
    return out


def _ray_to_ellipse_edge(
    bull: Tuple[float, float],
    angle_deg: float,
    ellipse: Ellipse,
) -> Optional[Tuple[int, int]]:
    """Točka presjeka zrake od bulla s elipsom (analitički, točno na elipsi)."""
    pt = _ray_to_ellipse_edge_f(bull, angle_deg, ellipse)
    if pt is None:
        return None
    return (int(round(pt[0])), int(round(pt[1])))


def _ray_to_ellipse_edge_f(
    bull: Tuple[float, float],
    angle_deg: float,
    ellipse: Ellipse,
) -> Optional[Tuple[float, float]]:
    (ecx, ecy), (width, height), ellipse_angle_deg = ellipse
    a = max(2.0, width * 0.5)
    b = max(2.0, height * 0.5)
    rad = math.radians(ellipse_angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    theta = math.radians(angle_deg)
    dx, dy = math.sin(theta), -math.cos(theta)

    bx, by = bull[0] - ecx, bull[1] - ecy
    ox = cos_a * bx + sin_a * by
    oy = -sin_a * bx + cos_a * by
    ux = cos_a * dx + sin_a * dy
    uy = -sin_a * dx + cos_a * dy

    inv_a2 = 1.0 / (a * a)
    inv_b2 = 1.0 / (b * b)
    qa = ux * ux * inv_a2 + uy * uy * inv_b2
    qb = 2.0 * (ox * ux * inv_a2 + oy * uy * inv_b2)
    qc = ox * ox * inv_a2 + oy * oy * inv_b2 - 1.0
    if abs(qa) < 1e-14:
        return None

    disc = qb * qb - 4.0 * qa * qc
    if disc < 0.0:
        return None

    sqrt_d = math.sqrt(disc)
    t1 = (-qb - sqrt_d) / (2.0 * qa)
    t2 = (-qb + sqrt_d) / (2.0 * qa)
    t_candidates = [t for t in (t1, t2) if t > 1e-6]
    if not t_candidates:
        return None

    t = max(t_candidates)
    return (bull[0] + t * dx, bull[1] + t * dy)


def _angle_sin_cos(angle_deg: float) -> Tuple[float, float]:
    """0°=gore, 90°=desno (image koordinate)."""
    rad = math.radians(angle_deg)
    return math.sin(rad), -math.cos(rad)


def _sample_xy(bull: Tuple[float, float], angle_deg: float, radius: float) -> Tuple[int, int]:
    st, ct = _angle_sin_cos(angle_deg)
    return int(round(bull[0] + radius * st)), int(round(bull[1] + radius * ct))


def _ray_dist_to_ellipse(bull: Tuple[float, float], angle_deg: float, ellipse: Ellipse) -> float:
    pt = _ray_to_ellipse_edge(bull, angle_deg, ellipse)
    if pt is None:
        (_, (ew, eh), _) = ellipse
        return max(ew, eh) * 0.5
    return math.hypot(pt[0] - bull[0], pt[1] - bull[1])


def _wire_boundary_distances(
    bull: Tuple[float, float],
    ellipse: Ellipse,
    wire_angles_deg: Tuple[float, ...],
) -> np.ndarray:
    dists: List[float] = []
    for ang in wire_angles_deg[:SEGMENT_COUNT]:
        dists.append(_ray_dist_to_ellipse(bull, float(ang), ellipse))
    while len(dists) < SEGMENT_COUNT:
        dists.append(dists[-1] if dists else 1.0)
    return np.array(dists[:SEGMENT_COUNT], dtype=np.float32)


def _wire_ellipse_intersections(
    bull: Tuple[float, float],
    ellipse: Ellipse,
    wire_angles_deg: Tuple[float, ...],
) -> List[Optional[Tuple[int, int]]]:
    pts: List[Optional[Tuple[int, int]]] = []
    for ang in wire_angles_deg[:SEGMENT_COUNT]:
        pts.append(_ray_to_ellipse_edge(bull, float(ang), ellipse))
    return pts


def _interp_boundary_distance(
    board_angles: np.ndarray,
    boundary_dists: np.ndarray,
) -> np.ndarray:
    """Udaljenost do elipse po board kutu (20 referentnih tocaka)."""
    n = len(boundary_dists)
    slot = (board_angles % 360.0) / SEGMENT_STEP_DEG
    idx = slot.astype(np.int32) % n
    frac = (slot - np.floor(slot)).astype(np.float32)
    d0 = boundary_dists[idx]
    d1 = boundary_dists[(idx + 1) % n]
    return (d0 + frac * (d1 - d0)).astype(np.float32)


def _draw_wires_to_ellipse(
    img: np.ndarray,
    bull: Tuple[float, float],
    ellipse: Ellipse,
    wire_angles_deg: Tuple[float, ...],
    *,
    line_color=(0, 255, 255),
    dot_color=(0, 255, 255),
    line_thickness: int = 1,
    dot_radius: int = 3,
    draw_dots: bool = True,
) -> None:
    """20 zica od bulla do elipse + tocke na sjecistu."""
    cx, cy = int(round(bull[0])), int(round(bull[1]))
    for ang in wire_angles_deg[:SEGMENT_COUNT]:
        edge = _ray_to_ellipse_edge(bull, float(ang), ellipse)
        if edge is None:
            continue
        cv2.line(img, (cx, cy), edge, line_color, line_thickness, cv2.LINE_AA)
        if draw_dots:
            cv2.circle(img, edge, dot_radius, dot_color, -1, cv2.LINE_AA)
            cv2.circle(img, edge, dot_radius + 1, dot_color, 1, cv2.LINE_AA)


def _draw_wire_sector_wedge(
    img: np.ndarray,
    bull: Tuple[float, float],
    ellipse: Ellipse,
    wire_angles_deg: Tuple[float, ...],
    wire_start: int,
    color: Tuple[int, int, int],
    *,
    wire_end: Optional[int] = None,
    alpha: float = SEG20_OVERLAY_ALPHA,
    draw_edges: bool = True,
) -> None:
    """Iscrtaj klin segmenta izmedu zica wire_start i wire_end (clockwise)."""
    n = min(len(wire_angles_deg), SEGMENT_COUNT)
    if wire_start < 0 or n < 8:
        return
    k0 = wire_start % n
    k1 = (wire_end if wire_end is not None else (k0 + 1)) % n
    e0 = _ray_to_ellipse_edge(bull, float(wire_angles_deg[k0]), ellipse)
    e1 = _ray_to_ellipse_edge(bull, float(wire_angles_deg[k1]), ellipse)
    if e0 is None or e1 is None:
        return
    cx, cy = int(round(bull[0])), int(round(bull[1]))
    tri = np.array([[cx, cy], list(e0), list(e1)], dtype=np.int32)
    overlay = img.copy()
    cv2.fillConvexPoly(overlay, tri, color)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
    if draw_edges:
        cv2.line(img, (cx, cy), e0, color, 3, cv2.LINE_AA)
        cv2.line(img, (cx, cy), e1, color, 3, cv2.LINE_AA)


def _mid_angle_between_wires(a0: float, a1: float) -> float:
    """Sredina kuta izmedu dvije zice (clockwise)."""
    start = float(a0) % 360.0
    end = float(a1) % 360.0
    if end <= start:
        end += 360.0
    return (start + end) * 0.5 % 360.0


def _segment20_wire_pair(
    wire_angles_deg: Tuple[float, ...],
    sorted_idx: List[int],
    angular_idx: int,
) -> Tuple[int, int, float]:
    n = min(len(wire_angles_deg), SEGMENT_COUNT)
    k0 = sorted_idx[angular_idx % n]
    k1 = sorted_idx[(angular_idx + 1) % n]
    center = _mid_angle_between_wires(float(wire_angles_deg[k0]), float(wire_angles_deg[k1]))
    return k0, k1, center


# Jedna boja za cijeli virtualni board overlay na top-down warpu.
_WARP_OVERLAY_BGR = (255, 160, 40)  # plava (BGR)
_SEG_FILL_ALPHA = 0.20
_SEG20_FILL_ALPHA = 0.80


def _punch_bull_disk(img: np.ndarray, before: np.ndarray, cx: float, cy: float, r: float) -> None:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), max(1, int(round(r))), 255, -1)
    img[mask > 0] = before[mask > 0]


def _draw_segment_fills_on_aligned_warp(img: np.ndarray) -> None:
    """Svi segmenti 50% plavo; segment 20 (gore) 100% plavo. Bull zona ostaje čista."""
    h, w = img.shape[:2]
    cx_out, _view_r, board_r = _warp_radii(w)
    rings = scoring_ring_radii(w)
    r_bull = max(1.0, float(rings["bull_outer"]))
    cx = cy = float(cx_out)
    before = img.copy()
    solid = np.zeros_like(img)
    soft = np.zeros_like(img)
    solid_mask = np.zeros((h, w), dtype=np.uint8)
    soft_mask = np.zeros((h, w), dtype=np.uint8)
    for k in range(SEGMENT_COUNT):
        ang0 = (SEG20_LEFT_WIRE_DST_DEG + k * SEGMENT_STEP_DEG) % 360.0
        ang1 = (SEG20_LEFT_WIRE_DST_DEG + (k + 1) * SEGMENT_STEP_DEG) % 360.0
        st0, ct0 = _angle_sin_cos(ang0)
        st1, ct1 = _angle_sin_cos(ang1)
        e0 = (int(round(cx + board_r * st0)), int(round(cy + board_r * ct0)))
        e1 = (int(round(cx + board_r * st1)), int(round(cy + board_r * ct1)))
        tri = np.array([[int(round(cx)), int(round(cy))], e0, e1], dtype=np.int32)
        if k == 0:
            cv2.fillConvexPoly(solid, tri, _WARP_OVERLAY_BGR)
            cv2.fillConvexPoly(solid_mask, tri, 255)
        else:
            cv2.fillConvexPoly(soft, tri, _WARP_OVERLAY_BGR)
            cv2.fillConvexPoly(soft_mask, tri, 255)
    if np.any(soft_mask):
        a = float(_SEG_FILL_ALPHA)
        m = soft_mask > 0
        img[m] = (
            (1.0 - a) * img[m].astype(np.float32) + a * soft[m].astype(np.float32)
        ).astype(np.uint8)
    if np.any(solid_mask):
        a20 = float(_SEG20_FILL_ALPHA)
        m = solid_mask > 0
        img[m] = (
            (1.0 - a20) * img[m].astype(np.float32) + a20 * solid[m].astype(np.float32)
        ).astype(np.uint8)
    _punch_bull_disk(img, before, cx, cy, r_bull)


def _draw_segment20_on_aligned_warp(img: np.ndarray) -> None:
    """Kompatibilnost: puni fill svih segmenata (seg20 100%)."""
    _draw_segment_fills_on_aligned_warp(img)


def _draw_segment_wires_on_aligned_warp(
    img: np.ndarray,
    *,
    r_inner: float,
    r_outer: float,
    color: Tuple[int, int, int],
) -> None:
    """20 tankih žica od vanjskog bulla do double outer — bez bull zone."""
    h, w = img.shape[:2]
    cx_out, _view_r, _board_r = _warp_radii(w)
    r0 = max(0.0, float(r_inner))
    r1 = max(r0 + 1.0, float(r_outer))
    for k in range(SEGMENT_COUNT):
        ang = (SEG20_LEFT_WIRE_DST_DEG + k * SEGMENT_STEP_DEG) % 360.0
        st, ct = _angle_sin_cos(ang)
        x0 = int(round(cx_out + r0 * st))
        y0 = int(round(cx_out + r0 * ct))
        x1 = int(round(cx_out + r1 * st))
        y1 = int(round(cx_out + r1 * ct))
        cv2.line(img, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)


def _draw_scoring_rings_on_aligned_warp(
    img: np.ndarray,
    cal: Optional[BoardCalibration] = None,
    *,
    bull_center: Optional[Tuple[float, float]] = None,
    source_bgr: Optional[np.ndarray] = None,
) -> None:
    """Virtualna ploča: savršeni D/T/bull krugovi + segment žice (plavo).

    Overlay je idealni model; Stage2 warp mora sjesti na ove krugove.
    """
    _ = bull_center
    _ = source_bgr
    h, w = img.shape[:2]
    cx_out, _view_r, board_r = _warp_radii(w)
    cx = cy = int(round(cx_out))
    color = _WARP_OVERLAY_BGR
    thick = int(WARP_RING_DRAW_THICKNESS)
    # Uvijek nominalni krugovi (board_r model) — ne "snapaj" overlay na krivi warp.
    rings = standard_ring_radii(board_r)
    _ = cal  # cal reserved for future debug; overlay stays ideal

    _draw_segment_wires_on_aligned_warp(
        img,
        r_inner=float(rings["bull_outer"]),
        r_outer=float(rings.get("double_outer", board_r)),
        color=color,
    )

    for key in ("triple_inner", "triple_outer", "double_inner", "double_outer"):
        r = max(1, int(round(rings[key])))
        cv2.circle(img, (cx, cy), r, color, thick, cv2.LINE_AA)

    b50 = max(1, int(round(rings["bull_inner"])))
    b25 = max(b50 + 1, int(round(rings["bull_outer"])))
    cv2.circle(img, (cx, cy), b50, color, thick, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), b25, color, thick, cv2.LINE_AA)


def _sample_rg_outer_points_in_band(
    board: np.ndarray,
    cx: float,
    cy: float,
    r_lo: float,
    r_hi: float,
) -> List[Tuple[float, float]]:
    h, w = board.shape[:2]
    if r_hi <= r_lo + 1.0:
        return []
    points: List[Tuple[float, float]] = []
    for deg in range(0, 360, 2):
        theta = math.radians(float(deg))
        st, ct = math.sin(theta), -math.cos(theta)
        outer_xy: Optional[Tuple[float, float]] = None
        for r in np.arange(r_lo, r_hi + 0.01, 0.5):
            x = int(round(cx + r * st))
            y = int(round(cy + r * ct))
            if 0 <= x < w and 0 <= y < h and board[y, x] > 0:
                outer_xy = (float(x), float(y))
        if outer_xy is not None:
            points.append(outer_xy)
    return points


def _fit_warp_ring_ellipse(
    points: List[Tuple[float, float]],
    center_hint: Tuple[float, float],
    min_dim: float,
) -> Optional[Ellipse]:
    if len(points) < 16:
        return None
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    try:
        ellipse = cv2.fitEllipse(pts)
    except cv2.error:
        return None
    (ecx, ecy), (width, height), ang = ellipse
    semi_max = max(width, height) * 0.5
    semi_min = min(width, height) * 0.5
    if semi_max < min_dim * 0.18 or semi_max > min_dim * 0.62:
        return None
    if semi_min < min_dim * 0.12:
        return None
    # Zadrži stvarni centar elipse (nakon Stage1 bull često nije u sredini warpa).
    # Odbaci samo divlje fitove daleko od hint-a.
    if math.hypot(ecx - center_hint[0], ecy - center_hint[1]) > semi_max * 0.35:
        return None
    return ellipse


def _ellipse_center(ell: Ellipse) -> Tuple[float, float]:
    (cx, cy), _, _ = ell
    return float(cx), float(cy)


def detect_bull_center_on_warp(warped: np.ndarray) -> Optional[Tuple[float, float]]:
    """Pronađi bull na Stage1/2 warpu (za recenter Stage2)."""
    if warped is None or warped.size == 0:
        return None
    h, w = warped.shape[:2]
    min_dim = float(min(h, w))
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    red, green = _bull_color_masks(hsv)
    hit = _find_bull(red, green, min_dim)
    if hit is None:
        return None
    (cx, cy), _r, _score = hit
    cx_out, _, _ = _warp_radii(w)
    # Bull mora biti blizu središta warpa (inače je false positive).
    if math.hypot(cx - cx_out, cy - cx_out) > min_dim * 0.18:
        return None
    return (float(cx), float(cy))


def _ray_outer_rg_radius(
    board: np.ndarray,
    cx: float,
    cy: float,
    angle_deg: float,
    r_lo: float,
    r_hi: float,
) -> Optional[float]:
    """Vanjski rub RG klastera duž zrake (ili None)."""
    h, w = board.shape[:2]
    th = math.radians(float(angle_deg))
    st, ct = math.sin(th), -math.cos(th)
    in_run = False
    run_outer: Optional[float] = None
    last: Optional[float] = None
    for r in np.arange(r_lo, r_hi + 0.01, 0.4):
        x = int(round(cx + r * st))
        y = int(round(cy + r * ct))
        if x < 0 or x >= w or y < 0 or y >= h:
            break
        hit = bool(board[y, x] > 0)
        if hit:
            in_run = True
            run_outer = float(r)
        elif in_run:
            last = run_outer
            in_run = False
            run_outer = None
    if in_run and run_outer is not None:
        last = run_outer
    return last


def detect_center_from_diameters_on_warp(
    warped: np.ndarray,
    *,
    hint: Optional[Tuple[float, float]] = None,
    board_r: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    """Središte ploče = prosjek središta 10 promjera (suprotni rubovi double RG).

    Stabilnije od samog color-bulla kad je bull blago pomaknut / blur.
    """
    if warped is None or warped.size == 0:
        return None
    h, w = warped.shape[:2]
    cx_out, _view_r, br = _warp_radii(w)
    if board_r is not None and board_r > 1.0:
        br = float(board_r)
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    red, green = _bull_color_masks(hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    board = cv2.bitwise_or(red, green)
    board = cv2.morphologyEx(board, cv2.MORPH_CLOSE, kernel)

    cx = float(hint[0]) if hint is not None else float(cx_out)
    cy = float(hint[1]) if hint is not None else float(cx_out)
    r_lo = br * 0.78
    r_hi = br * 1.08

    for _ in range(3):
        mids_x: List[float] = []
        mids_y: List[float] = []
        for deg in range(0, 180, 18):
            r0 = _ray_outer_rg_radius(board, cx, cy, float(deg), r_lo, r_hi)
            r1 = _ray_outer_rg_radius(board, cx, cy, float(deg + 180), r_lo, r_hi)
            if r0 is None or r1 is None:
                continue
            th0 = math.radians(float(deg))
            th1 = math.radians(float(deg + 180))
            st0, ct0 = math.sin(th0), -math.cos(th0)
            st1, ct1 = math.sin(th1), -math.cos(th1)
            x0, y0 = cx + r0 * st0, cy + r0 * ct0
            x1, y1 = cx + r1 * st1, cy + r1 * ct1
            mids_x.append(0.5 * (x0 + x1))
            mids_y.append(0.5 * (y0 + y1))
        if len(mids_x) < 4:
            return None
        nx = float(np.median(mids_x))
        ny = float(np.median(mids_y))
        if math.hypot(nx - cx, ny - cy) < 0.25:
            cx, cy = nx, ny
            break
        cx, cy = nx, ny

    if math.hypot(cx - cx_out, cy - cx_out) > br * 0.22:
        return None
    return (cx, cy)


def _stage2_source_center(
    warped: Optional[np.ndarray],
    double_ell: Ellipse,
) -> Tuple[float, float]:
    """Izvor Stage2 = bull (scoring origin). Fallback: promjeri / centar double elipse."""
    ecx, ecy = _ellipse_center(double_ell)
    if warped is None:
        return ecx, ecy
    d_semi = _mean_ellipse_semi(double_ell)
    max_bull = max(10.0, d_semi * 0.20)
    max_diam = max(10.0, d_semi * 0.16)
    bull = detect_bull_center_on_warp(warped)
    if bull is not None and math.hypot(bull[0] - ecx, bull[1] - ecy) <= max_bull:
        # Bull je origin overlaya — ne blendaj prema double elipsi (to ostavlja bull pomaknut).
        return (float(bull[0]), float(bull[1]))
    diam = detect_center_from_diameters_on_warp(
        warped, hint=(ecx, ecy), board_r=d_semi
    )
    if diam is not None and math.hypot(diam[0] - ecx, diam[1] - ecy) <= max_diam:
        return (float(diam[0]), float(diam[1]))
    return ecx, ecy


def _warp_bull_center_offset(
    warped: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """(dx, dy) = color-bull − cx_out na final/Stage2 warpu. None ako nema bulla."""
    if warped is None or warped.size == 0:
        return None
    h, w = warped.shape[:2]
    cx_out, _, _ = _warp_radii(w)
    bull = detect_bull_center_on_warp(warped)
    if bull is None:
        return None
    return (float(bull[0] - cx_out), float(bull[1] - cx_out))


def detect_scoring_rings_on_warp(
    warped: np.ndarray,
    *,
    nominal_board_r: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    """Na top-down warpu izmjeri double/triple radijuse (RG maska) — kruzni fallback."""
    ell = detect_scoring_ring_ellipses_on_warp(warped, nominal_board_r=nominal_board_r)
    if ell is None:
        return None
    d_out = _mean_ellipse_semi(ell["double_outer"])
    d_in = _mean_ellipse_semi(ell["double_inner"])
    t_out = _mean_ellipse_semi(ell["triple_outer"])
    t_in = _mean_ellipse_semi(ell["triple_inner"])
    return {
        "double_outer": float(d_out),
        "double_inner": float(d_in),
        "triple_outer": float(t_out),
        "triple_inner": float(t_in),
        "bull_outer": float(d_out * RING_BULL_OUTER_FRAC),
        "bull_inner": float(d_out * RING_BULL_INNER_FRAC),
        "sisal_edge": float(d_out * BOARD_SISAL_EDGE_FRAC),
    }


def _circle_as_ellipse(cx: float, cy: float, radius: float) -> Ellipse:
    d = max(2.0, 2.0 * float(radius))
    return ((float(cx), float(cy)), (d, d), 0.0)


def _ellipse_to_circle_maps(
    ellipse: Ellipse,
    *,
    center: Tuple[float, float],
    target_r: float,
    out_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mapa 2. warpa: final -> stage1 pikseli + view mask (samo double → board_r)."""
    size = int(out_size)
    cx_out, view_r, _board_r = _warp_radii(size)
    bx, by = float(center[0]), float(center[1])
    tgt = max(8.0, float(target_r))

    edge_table = np.array(
        [_ray_dist_to_ellipse((bx, by), float(a), ellipse) for a in range(360)],
        dtype=np.float32,
    )
    edge_table = np.maximum(edge_table, 1.0)

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dx = xx - cx_out
    dy = yy - cx_out
    r_px = np.sqrt(dx * dx + dy * dy)
    theta = (np.degrees(np.arctan2(dx, -dy)) % 360.0).astype(np.float32)
    edge_dist = np.interp(theta, np.arange(360, dtype=np.float32), edge_table)

    inside = r_px <= tgt
    beyond = np.maximum(r_px - tgt, 0.0)
    margin = max(view_r - tgt, 1.0)
    extrap = 1.0 + (beyond / margin) * WARP_OUTSIDE_EXTRAPOL
    scale = np.where(inside, r_px / tgt, extrap)
    src_r = scale * edge_dist
    sin_t = np.sin(np.deg2rad(theta))
    cos_t = -np.cos(np.deg2rad(theta))
    map_x = (bx + src_r * sin_t).astype(np.float32)
    map_y = (by + src_r * cos_t).astype(np.float32)
    mask = (r_px <= view_r).astype(np.uint8)
    return map_x, map_y, mask


def _multi_ring_radial_maps(
    double_ell: Ellipse,
    triple_ell: Ellipse,
    *,
    center: Tuple[float, float],
    board_r: float,
    out_size: int,
    double_inner_frac: Optional[float] = None,
    triple_inner_frac: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stage2: piecewise radijalni warp na idealne D/T krugove (inner+outer).

    Na svakoj zraci: izmjerena T/D inner+outer → nominalni krugovi.
    """
    size = int(out_size)
    cx_out, view_r, _ = _warp_radii(size)
    bx, by = float(center[0]), float(center[1])
    dst_do = max(8.0, float(board_r))
    dst_di = max(4.0, dst_do * float(RING_DOUBLE_INNER_FRAC))
    dst_to = max(4.0, dst_do * float(RING_TRIPLE_OUTER_FRAC))
    dst_ti = max(2.0, dst_do * float(RING_TRIPLE_INNER_FRAC))
    # Osiguraj strogi poredak destinacija.
    dst_ti = min(dst_ti, dst_to - 1.0)
    dst_to = min(dst_to, dst_di - 1.0)
    dst_di = min(dst_di, dst_do - 1.0)

    d_table = np.array(
        [_ray_dist_to_ellipse((bx, by), float(a), double_ell) for a in range(360)],
        dtype=np.float32,
    )
    t_table = np.array(
        [_ray_dist_to_ellipse((bx, by), float(a), triple_ell) for a in range(360)],
        dtype=np.float32,
    )
    d_table = np.maximum(d_table, 1.0)
    t_cap = d_table * float(RING_TRIPLE_OUTER_FRAC) * 1.15
    t_floor = d_table * float(RING_TRIPLE_OUTER_FRAC) * 0.75
    t_table = np.clip(t_table, t_floor, np.minimum(t_cap, d_table * 0.97))
    t_table = np.maximum(t_table, 1.0)

    di_frac = float(double_inner_frac) if double_inner_frac is not None else float(RING_DOUBLE_INNER_FRAC)
    ti_frac = float(triple_inner_frac) if triple_inner_frac is not None else float(
        RING_TRIPLE_INNER_FRAC / max(RING_TRIPLE_OUTER_FRAC, 1e-6)
    )
    # di_frac = r_inner/r_double_outer; ti_frac = r_t_inner/r_t_outer
    di_frac = float(np.clip(di_frac, 0.88, 0.985))
    ti_frac = float(np.clip(ti_frac, 0.82, 0.985))
    di_table = np.maximum(d_table * di_frac, 1.0)
    ti_table = np.maximum(t_table * ti_frac, 1.0)
    # Strogi poredak izvora: ti < to < di < do
    ti_table = np.minimum(ti_table, t_table * 0.98)
    di_table = np.minimum(di_table, d_table * 0.98)
    di_table = np.maximum(di_table, t_table * 1.02)

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dx = xx - cx_out
    dy = yy - cx_out
    r_px = np.sqrt(dx * dx + dy * dy)
    theta = (np.degrees(np.arctan2(dx, -dy)) % 360.0).astype(np.float32)
    ang = np.arange(360, dtype=np.float32)
    src_do = np.interp(theta, ang, d_table)
    src_di = np.interp(theta, ang, di_table)
    src_to = np.interp(theta, ang, t_table)
    src_ti = np.interp(theta, ang, ti_table)

    def _lerp(r, r0, r1, s0, s1):
        span = np.maximum(r1 - r0, 1e-3)
        return s0 + ((r - r0) / span) * (s1 - s0)

    src_r = np.where(
        r_px <= dst_ti,
        (r_px / np.maximum(dst_ti, 1e-3)) * src_ti,
        np.where(
            r_px <= dst_to,
            _lerp(r_px, dst_ti, dst_to, src_ti, src_to),
            np.where(
                r_px <= dst_di,
                _lerp(r_px, dst_to, dst_di, src_to, src_di),
                np.where(
                    r_px <= dst_do,
                    _lerp(r_px, dst_di, dst_do, src_di, src_do),
                    src_do
                    * (
                        1.0
                        + np.maximum(r_px - dst_do, 0.0)
                        / max(view_r - dst_do, 1.0)
                        * WARP_OUTSIDE_EXTRAPOL
                    ),
                ),
            ),
        ),
    )
    sin_t = np.sin(np.deg2rad(theta))
    cos_t = -np.cos(np.deg2rad(theta))
    map_x = (bx + src_r * sin_t).astype(np.float32)
    map_y = (by + src_r * cos_t).astype(np.float32)
    mask = (r_px <= view_r).astype(np.uint8)
    return map_x, map_y, mask


def warp_ellipse_to_circle(
    img: np.ndarray,
    ellipse: Ellipse,
    *,
    center: Optional[Tuple[float, float]] = None,
    target_r: float,
    out_size: int = 0,
) -> np.ndarray:
    """2. warp: mapiraj elipsu (npr. double) u savršen krug radijusa target_r."""
    if img is None or img.size == 0:
        return img
    h, w = img.shape[:2]
    size = int(out_size) if out_size > 0 else int(min(h, w))
    cx_out, _view_r, _board_r = _warp_radii(size)
    if center is None:
        center = (cx_out, cx_out)
    map_x, map_y, mask = _ellipse_to_circle_maps(
        ellipse, center=center, target_r=target_r, out_size=size
    )
    warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return cv2.bitwise_and(warped, warped, mask=mask)


def _sample_double_outer_points_on_stage1(
    board: np.ndarray,
    cx: float,
    cy: float,
    board_r: float,
) -> List[Tuple[float, float]]:
    """Po zraci: vanjski rub RG u pojasu oko očekivanog doublea (Stage1 mapira double→board_r)."""
    h, w = board.shape[:2]
    # Stage1 stavlja double na ~board_r; traži samo u tom pojasu (ne cijelu ploču).
    r_lo = max(4.0, board_r * 0.82)
    r_hi = max(r_lo + 2.0, board_r * 1.06)
    points: List[Tuple[float, float]] = []
    radii: List[float] = []
    for deg in range(0, 360, 2):
        theta = math.radians(float(deg))
        st, ct = math.sin(theta), -math.cos(theta)
        in_run = False
        run_outer: Optional[float] = None
        clusters: List[float] = []
        for r in np.arange(r_lo, r_hi + 0.01, 0.4):
            x = int(round(cx + r * st))
            y = int(round(cy + r * ct))
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            hit = bool(board[y, x] > 0)
            if hit:
                in_run = True
                run_outer = float(r)
            elif in_run:
                if run_outer is not None:
                    clusters.append(run_outer)
                in_run = False
                run_outer = None
        if in_run and run_outer is not None:
            clusters.append(run_outer)
        if not clusters:
            continue
        chosen = clusters[-1]
        radii.append(chosen)
        points.append((cx + chosen * st, cy + chosen * ct))

    if len(radii) < 16:
        return points
    soft_cap = float(np.percentile(radii, 80)) * 1.04
    soft_floor = float(np.percentile(radii, 20)) * 0.96
    cleaned: List[Tuple[float, float]] = []
    for (px, py), rr in zip(points, radii):
        if soft_floor <= rr <= soft_cap:
            cleaned.append((px, py))
    return cleaned if len(cleaned) >= 16 else points


def _sample_triple_outer_points_on_stage1(
    board: np.ndarray,
    cx: float,
    cy: float,
    board_r: float,
    double_semi: float,
) -> List[Tuple[float, float]]:
    """Vanjski rub triple RG klastera u očekivanom pojasu (između bulla i doublea)."""
    h, w = board.shape[:2]
    d_ref = max(8.0, min(float(double_semi), float(board_r) * 1.05))
    exp = d_ref * float(RING_TRIPLE_OUTER_FRAC)
    r_lo = max(4.0, exp * 0.78)
    r_hi = min(d_ref * 0.93, exp * 1.20)
    if r_hi <= r_lo + 1.5:
        return []
    points: List[Tuple[float, float]] = []
    radii: List[float] = []
    for deg in range(0, 360, 2):
        theta = math.radians(float(deg))
        st, ct = math.sin(theta), -math.cos(theta)
        in_run = False
        run_outer: Optional[float] = None
        clusters: List[float] = []
        for r in np.arange(r_lo, r_hi + 0.01, 0.4):
            x = int(round(cx + r * st))
            y = int(round(cy + r * ct))
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            hit = bool(board[y, x] > 0)
            if hit:
                in_run = True
                run_outer = float(r)
            elif in_run:
                if run_outer is not None:
                    clusters.append(run_outer)
                in_run = False
                run_outer = None
        if in_run and run_outer is not None:
            clusters.append(run_outer)
        if not clusters:
            continue
        # Najbliži očekivanom triple outeru (ne nužno zadnji — izbjegni single bleed).
        chosen = min(clusters, key=lambda rr: abs(rr - exp))
        radii.append(chosen)
        points.append((cx + chosen * st, cy + chosen * ct))

    if len(radii) < 16:
        return points
    soft_cap = float(np.percentile(radii, 80)) * 1.05
    soft_floor = float(np.percentile(radii, 20)) * 0.95
    cleaned: List[Tuple[float, float]] = []
    for (px, py), rr in zip(points, radii):
        if soft_floor <= rr <= soft_cap:
            cleaned.append((px, py))
    return cleaned if len(cleaned) >= 16 else points


def detect_double_outer_ellipse_on_warp(
    warped: np.ndarray,
    *,
    nominal_board_r: Optional[float] = None,
) -> Optional[Ellipse]:
    """Stage2 ulaz: elipsa kroz točke na VANJSKOM rubu double prstena (Stage1 warp)."""
    if warped is None or warped.size == 0:
        return None
    h, w = warped.shape[:2]
    cx_out, _view_r, board_r = _warp_radii(w)
    if nominal_board_r is not None and nominal_board_r > 1.0:
        board_r = float(nominal_board_r)
    min_dim = float(min(h, w))

    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    red, green = _bull_color_masks(hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    board = cv2.bitwise_or(red, green)
    board = cv2.morphologyEx(board, cv2.MORPH_CLOSE, kernel)

    bull = detect_bull_center_on_warp(warped)
    diam = detect_center_from_diameters_on_warp(
        warped,
        hint=bull if bull is not None else (float(cx_out), float(cx_out)),
        board_r=board_r,
    )
    if diam is not None:
        cx, cy = diam
    elif bull is not None:
        cx, cy = bull
    else:
        cx, cy = float(cx_out), float(cx_out)

    d_pts = _sample_double_outer_points_on_stage1(board, cx, cy, board_r)
    if len(d_pts) < 16:
        d_pts = _sample_rg_outer_points_in_band(
            board, cx, cy, board_r * 0.86, board_r * 1.06
        )
    d_outer = _fit_warp_ring_ellipse(d_pts, (cx, cy), min_dim)
    if d_outer is None:
        return _circle_as_ellipse(cx, cy, board_r)

    semi = _mean_ellipse_semi(d_outer)
    # Double nakon Stage1 treba biti blizu board_r.
    if semi < board_r * 0.72 or semi > board_r * 1.18:
        return _circle_as_ellipse(cx, cy, board_r)
    # Bez ručnog shrink/expand — residual korekcija radi to nakon Stage2.
    return d_outer


def detect_triple_outer_ellipse_on_warp(
    warped: np.ndarray,
    double_outer: Ellipse,
) -> Optional[Ellipse]:
    """Izmjeni triple outer na Stage1 warpu (za multi-ring Stage2)."""
    if warped is None or warped.size == 0 or double_outer is None:
        return None
    h, w = warped.shape[:2]
    cx_out, _view_r, board_r = _warp_radii(w)
    min_dim = float(min(h, w))
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    red, green = _bull_color_masks(hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    board = cv2.bitwise_or(red, green)
    board = cv2.morphologyEx(board, cv2.MORPH_CLOSE, kernel)

    bull = detect_bull_center_on_warp(warped)
    cx, cy = bull if bull is not None else _ellipse_center(double_outer)
    d_semi = _mean_ellipse_semi(double_outer)
    t_pts = _sample_triple_outer_points_on_stage1(board, cx, cy, board_r, d_semi)
    if len(t_pts) < 16:
        exp = d_semi * RING_TRIPLE_OUTER_FRAC
        t_pts = _sample_rg_outer_points_in_band(board, cx, cy, exp * 0.80, exp * 1.18)
    t_outer = _fit_warp_ring_ellipse(t_pts, (cx, cy), min_dim)
    if t_outer is None:
        return None
    t_semi = _mean_ellipse_semi(t_outer)
    exp = d_semi * RING_TRIPLE_OUTER_FRAC
    if t_semi < exp * 0.72 or t_semi > exp * 1.28:
        return None
    if t_semi >= d_semi * 0.96:
        return None
    return t_outer


def _measure_rg_ring_band_radii(
    warped: np.ndarray,
    *,
    board_r: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    """Na Stage2 warpu: unutarnji+vanjski rub RG pojasa za double i triple (bez morph close).

    Rubovi su rubovi boje (sisal), ne metalne žice — ali to je scoring zona.
    """
    if warped is None or warped.size == 0:
        return None
    h, w = warped.shape[:2]
    cx_out, _, br = _warp_radii(w)
    if board_r is not None and board_r > 1.0:
        br = float(board_r)
    cx = cy = float(cx_out)
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    red, green = _bull_color_masks(hsv)
    board = cv2.bitwise_or(red, green)

    d_exp = br
    t_exp = br * float(RING_TRIPLE_OUTER_FRAC)
    d_outs: List[float] = []
    d_inns: List[float] = []
    t_outs: List[float] = []
    t_inns: List[float] = []

    for deg in range(0, 360, 3):
        th = math.radians(float(deg))
        st, ct = math.sin(th), -math.cos(th)
        clusters: List[Tuple[float, float]] = []
        in_run = False
        r0 = r1 = 0.0
        for r in np.arange(br * 0.42, br * 1.12 + 0.01, 0.4):
            x = int(round(cx + r * st))
            y = int(round(cy + r * ct))
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            hit = bool(board[y, x] > 0)
            if hit:
                if not in_run:
                    in_run = True
                    r0 = float(r)
                r1 = float(r)
            elif in_run:
                if r1 - r0 >= 1.2:
                    clusters.append((r0, r1))
                in_run = False
        if in_run and r1 - r0 >= 1.2:
            clusters.append((r0, r1))

        best_d: Optional[Tuple[float, float]] = None
        best_t: Optional[Tuple[float, float]] = None
        best_d_err = 1e9
        best_t_err = 1e9
        for a, b in clusters:
            mid = 0.5 * (a + b)
            d_err = abs(mid - d_exp)
            t_err = abs(mid - t_exp)
            if d_err < best_d_err and mid >= br * 0.86:
                best_d_err = d_err
                best_d = (a, b)
            if t_err < best_t_err and br * 0.48 <= mid <= br * 0.78:
                best_t_err = t_err
                best_t = (a, b)
        if best_d is not None and best_d_err <= br * 0.10:
            d_inns.append(best_d[0])
            d_outs.append(best_d[1])
        if best_t is not None and best_t_err <= br * 0.12:
            t_inns.append(best_t[0])
            t_outs.append(best_t[1])

    if len(d_outs) < 24 or len(t_outs) < 24:
        return None

    d_out = float(np.median(d_outs))
    d_in = float(np.median(d_inns))
    t_out = float(np.median(t_outs))
    t_in = float(np.median(t_inns))

    # Soft clamp oko nominale — odbaci divlje fitove.
    nom_di = br * RING_DOUBLE_INNER_FRAC
    nom_to = br * RING_TRIPLE_OUTER_FRAC
    nom_ti = br * RING_TRIPLE_INNER_FRAC
    if abs(d_out - br) > br * 0.06:
        d_out = br
    if abs(d_in - nom_di) > br * 0.06:
        d_in = nom_di
    if abs(t_out - nom_to) > br * 0.08:
        t_out = nom_to
    if abs(t_in - nom_ti) > br * 0.08:
        t_in = nom_ti
    if d_in >= d_out - 1.0:
        d_in = max(1.0, d_out - 1.0)
    if t_in >= t_out - 1.0:
        t_in = max(1.0, t_out - 1.0)
    return {
        "double_outer": d_out,
        "double_inner": d_in,
        "triple_outer": t_out,
        "triple_inner": t_in,
    }


def detect_circular_rings_on_aligned_warp(
    warped: np.ndarray,
    *,
    nominal_board_r: Optional[float] = None,
) -> Optional[Dict[str, Ellipse]]:
    """Na 2. warpu izmjeri double/triple inner+outer kao koncentrične krugove."""
    if warped is None or warped.size == 0:
        return None
    h, w = warped.shape[:2]
    cx_out, _view_r, board_r = _warp_radii(w)
    if nominal_board_r is not None and nominal_board_r > 1.0:
        board_r = float(nominal_board_r)
    cx = cy = float(cx_out)
    measured = _measure_rg_ring_band_radii(warped, board_r=board_r)
    if measured is None:
        return _nominal_final_ring_ellipses(w)
    return {
        "double_outer": _circle_as_ellipse(cx, cy, measured["double_outer"]),
        "double_inner": _circle_as_ellipse(cx, cy, measured["double_inner"]),
        "triple_outer": _circle_as_ellipse(cx, cy, measured["triple_outer"]),
        "triple_inner": _circle_as_ellipse(cx, cy, measured["triple_inner"]),
    }


def detect_scoring_ring_ellipses_on_warp(
    warped: np.ndarray,
    *,
    nominal_board_r: Optional[float] = None,
) -> Optional[Dict[str, Ellipse]]:
    """Legacy: jedan prolaz elipsi (bez 2. warpa). Preferiraj enrich pipeline."""
    d_outer = detect_double_outer_ellipse_on_warp(warped, nominal_board_r=nominal_board_r)
    if d_outer is None:
        return None
    d_semi = _mean_ellipse_semi(d_outer)
    h, w = warped.shape[:2]
    cx_out, _, board_r = _warp_radii(w)
    if nominal_board_r is not None and nominal_board_r > 1.0:
        board_r = float(nominal_board_r)
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    red, green = _bull_color_masks(hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    board = cv2.bitwise_or(red, green)
    board = cv2.morphologyEx(board, cv2.MORPH_CLOSE, kernel)
    t_pts = _sample_rg_outer_points_in_band(
        board, cx_out, cx_out, d_semi * 0.48, d_semi * 0.82
    )
    t_outer = _fit_warp_ring_ellipse(t_pts, (cx_out, cx_out), float(min(h, w)))
    expected_t = _scale_ellipse_axes(d_outer, RING_TRIPLE_OUTER_FRAC)
    if t_outer is None:
        t_outer = expected_t
    else:
        if abs(_mean_ellipse_semi(t_outer) - _mean_ellipse_semi(expected_t)) > _mean_ellipse_semi(expected_t) * 0.22:
            t_outer = expected_t
    d_inner = _scale_ellipse_axes(d_outer, RING_DOUBLE_INNER_FRAC * WARP_RING_INNER_TIGHTEN)
    t_inner = _scale_ellipse_axes(
        t_outer,
        (RING_TRIPLE_INNER_FRAC / max(RING_TRIPLE_OUTER_FRAC, 1e-6)) * WARP_RING_INNER_TIGHTEN,
    )
    return {
        "double_outer": d_outer,
        "double_inner": d_inner,
        "triple_outer": t_outer,
        "triple_inner": t_inner,
    }


def warp_topdown_stage1(
    frame: np.ndarray,
    cal: BoardCalibration,
    *,
    out_size: int = DEBUG_WARP_SIZE,
) -> Optional[np.ndarray]:
    """Stage1: kamera double-elipsa + žice → seg.20 gore (double → board_r)."""
    if not cal.is_valid() or cal.double_ellipse is None:
        return None
    size = int(out_size) if out_size > 0 else DEBUG_WARP_SIZE
    warped, _b, _br, _dr, _sh = warp_board_topdown(
        frame,
        cal.center,
        cal.bull_radius,
        cal.double_ellipse,
        size,
        wire_angles_deg=cal.wire_angles_deg if cal.has_wires() else (),
        segment20_offset=cal.segment20_offset,
    )
    return warped


def apply_ring_align_warp(
    warped1: np.ndarray,
    warp1_double_outer: Ellipse,
    *,
    out_size: int = 0,
    warp1_triple_outer: Optional[Ellipse] = None,
    center: Optional[Tuple[float, float]] = None,
    double_inner_ratio: Optional[float] = None,
    triple_inner_ratio: Optional[float] = None,
) -> np.ndarray:
    """Stage2: double (+triple) elipsa → koncentrični krugovi board_r."""
    h, w = warped1.shape[:2]
    size = int(out_size) if out_size > 0 else int(min(h, w))
    _cx_out, _view_r, board_r = _warp_radii(size)
    if center is None:
        center = _stage2_source_center(warped1, warp1_double_outer)
    if warp1_triple_outer is not None:
        map_x, map_y, mask = _multi_ring_radial_maps(
            warp1_double_outer,
            warp1_triple_outer,
            center=center,
            board_r=board_r,
            out_size=size,
            double_inner_frac=double_inner_ratio,
            triple_inner_frac=triple_inner_ratio,
        )
        warped = cv2.remap(
            warped1, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        return cv2.bitwise_and(warped, warped, mask=mask)
    return warp_ellipse_to_circle(
        warped1,
        warp1_double_outer,
        center=center,
        target_r=board_r,
        out_size=size,
    )


def _nominal_final_ring_ellipses(out_size: int) -> Dict[str, Ellipse]:
    cx_out, _, board_r = _warp_radii(int(out_size))
    cx = cy = float(cx_out)
    return {
        "double_outer": _circle_as_ellipse(cx, cy, board_r),
        "double_inner": _circle_as_ellipse(
            cx, cy, board_r * RING_DOUBLE_INNER_FRAC * WARP_RING_INNER_TIGHTEN
        ),
        "triple_outer": _circle_as_ellipse(cx, cy, board_r * RING_TRIPLE_OUTER_FRAC),
        "triple_inner": _circle_as_ellipse(
            cx, cy, board_r * RING_TRIPLE_INNER_FRAC * WARP_RING_INNER_TIGHTEN
        ),
    }


def _stage2_needed(double_ell: Ellipse, board_r: float, center: Tuple[float, float]) -> bool:
    """True ako Stage1 double još nije dovoljno krug / centriran / na board_r."""
    oval = _ellipse_ovality(double_ell)
    semi = _mean_ellipse_semi(double_ell)
    ecx, ecy = _ellipse_center(double_ell)
    rad_err = abs(semi - board_r) / max(board_r, 1.0)
    cen_err = math.hypot(ecx - center[0], ecy - center[1]) / max(board_r, 1.0)
    return (
        oval >= STAGE2_OVALITY_MIN
        or rad_err >= STAGE2_RADIUS_MAX_DEV
        or cen_err >= STAGE2_CENTER_MAX_FRAC
    )


def ensure_ring_align_warp(
    cal: BoardCalibration,
    *,
    out_size: int = DEBUG_WARP_SIZE,
) -> BoardCalibration:
    """Ako nema Stage2 podataka (stari JSON), dodaj nominalni double→board_r align."""
    if cal.has_ring_align_warp():
        return cal
    size = int(out_size) if out_size > 0 else DEBUG_WARP_SIZE
    cx_out, _, board_r = _warp_radii(size)
    src = (float(cx_out), float(cx_out))
    # Identity Stage2: Stage1 već stavlja double na board_r.
    d1 = _circle_as_ellipse(src[0], src[1], board_r)
    ell = _nominal_final_ring_ellipses(size)
    return replace(
        cal,
        measured_double_outer_frac=1.0,
        measured_double_inner_frac=float(RING_DOUBLE_INNER_FRAC * WARP_RING_INNER_TIGHTEN),
        measured_triple_outer_frac=float(RING_TRIPLE_OUTER_FRAC),
        measured_triple_inner_frac=float(RING_TRIPLE_INNER_FRAC * WARP_RING_INNER_TIGHTEN),
        warp_ring_size=size,
        warp_double_outer=ell["double_outer"],
        warp_double_inner=ell["double_inner"],
        warp_triple_outer=ell["triple_outer"],
        warp_triple_inner=ell["triple_inner"],
        warp1_double_outer=d1,
        warp1_triple_outer=None,
        warp1_center=src,
    )


def _warp_ring_center_offset(
    warped: np.ndarray,
    *,
    board_r: float,
) -> Optional[Tuple[float, float, float, float]]:
    """Na final/Stage2 warpu: (dx, dy, d_semi, t_semi) za double/triple RG pojaseve.

    dx/dy = centar doublea − cx_out (pomak ploče u warpu).
    d_semi/t_semi = vanjski rub RG pojasa.
    """
    if warped is None or warped.size == 0:
        return None
    h, w = warped.shape[:2]
    cx_out, _, br = _warp_radii(w)
    if board_r > 1.0:
        br = float(board_r)

    measured = _measure_rg_ring_band_radii(warped, board_r=br)
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    red, green = _bull_color_masks(hsv)
    board = cv2.bitwise_or(red, green)
    d_pts = _sample_double_outer_points_on_stage1(board, cx_out, cx_out, br)
    d_ell = _fit_warp_ring_ellipse(d_pts, (cx_out, cx_out), float(min(h, w)))
    if d_ell is not None:
        ecx, ecy = _ellipse_center(d_ell)
    else:
        ecx = ecy = float(cx_out)

    if measured is not None:
        return (
            float(ecx - cx_out),
            float(ecy - cx_out),
            float(measured["double_outer"]),
            float(measured["triple_outer"]),
        )

    board = cv2.morphologyEx(
        board, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    d_pts = _sample_double_outer_points_on_stage1(board, cx_out, cx_out, br)
    d_ell = _fit_warp_ring_ellipse(d_pts, (cx_out, cx_out), float(min(h, w)))
    if d_ell is None:
        return None
    (ecx, ecy), _, _ = d_ell
    d_semi = _mean_ellipse_semi(d_ell)
    t_pts = _sample_triple_outer_points_on_stage1(board, ecx, ecy, br, d_semi)
    t_ell = _fit_warp_ring_ellipse(t_pts, (ecx, ecy), float(min(h, w)))
    t_semi = _mean_ellipse_semi(t_ell) if t_ell is not None else br * RING_TRIPLE_OUTER_FRAC
    return (float(ecx - cx_out), float(ecy - cx_out), float(d_semi), float(t_semi))


def _per_angle_rg_outer_radii(
    warped: np.ndarray,
    *,
    board_r: float,
    expected_r: float,
    r_lo: float,
    r_hi: float,
) -> Optional[np.ndarray]:
    """Po stupnju: vanjski rub RG klastera u pojasu. NaN ako nema uzorka."""
    if warped is None or warped.size == 0:
        return None
    h, w = warped.shape[:2]
    cx_out, _, _ = _warp_radii(w)
    cx = cy = float(cx_out)
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    red, green = _bull_color_masks(hsv)
    board = cv2.bitwise_or(red, green)
    out = np.full(360, np.nan, dtype=np.float64)
    exp = float(expected_r)
    for deg in range(0, 360, 2):
        th = math.radians(float(deg))
        st, ct = math.sin(th), -math.cos(th)
        clusters: List[float] = []
        in_run = False
        run_outer: Optional[float] = None
        for r in np.arange(r_lo, r_hi + 0.01, 0.35):
            x = int(round(cx + r * st))
            y = int(round(cy + r * ct))
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            hit = bool(board[y, x] > 0)
            if hit:
                in_run = True
                run_outer = float(r)
            elif in_run:
                if run_outer is not None:
                    clusters.append(run_outer)
                in_run = False
                run_outer = None
        if in_run and run_outer is not None:
            clusters.append(run_outer)
        if not clusters:
            continue
        chosen = min(clusters, key=lambda rr: abs(rr - exp))
        if abs(chosen - exp) <= board_r * 0.08:
            out[deg] = chosen
            out[(deg + 1) % 360] = chosen
    if np.count_nonzero(np.isfinite(out)) < 40:
        return None
    # Odbaci outlier kutove (brojevi / bleed) — Stage2 smije pratiti samo konzistentan rub.
    finite = out[np.isfinite(out)]
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med))) + 1e-3
    keep = np.isfinite(out) & (np.abs(out - med) <= max(1.6, 3.5 * mad))
    cleaned = np.full(360, np.nan, dtype=np.float64)
    cleaned[keep] = out[keep]
    if np.count_nonzero(np.isfinite(cleaned)) < 36:
        return out
    return cleaned


def _refine_stage2_ellipse_from_residual(
    warp1_ell: Ellipse,
    center: Tuple[float, float],
    r_meas: np.ndarray,
    target_r: float,
    min_dim: float,
) -> Optional[Ellipse]:
    """Ispravi Stage1 elipsu tako da idući Stage2 dovodi r_meas → target_r po kutovima."""
    bx, by = float(center[0]), float(center[1])
    tgt = max(1.0, float(target_r))
    points: List[Tuple[float, float]] = []
    for ang in range(0, 360, 2):
        rm = float(r_meas[ang])
        if not math.isfinite(rm):
            continue
        old = _ray_dist_to_ellipse((bx, by), float(ang), warp1_ell)
        if old is None or old < 1.0:
            continue
        # Prigušena korekcija — elipsa ne može pojesti sve kutne greške odjednom.
        raw = rm / tgt
        corr = float(np.clip(1.0 + 0.55 * (raw - 1.0), 0.92, 1.10))
        new_r = float(old) * corr
        th = math.radians(float(ang))
        st, ct = math.sin(th), -math.cos(th)
        points.append((bx + new_r * st, by + new_r * ct))
    if len(points) < 24:
        return None
    return _fit_warp_ring_ellipse(points, (bx, by), float(min_dim))


def enrich_calibration_with_warp_rings(
    frame: np.ndarray,
    cal: BoardCalibration,
    *,
    out_size: int = DEBUG_WARP_SIZE,
) -> BoardCalibration:
    """Stage1 (double→board_r) → Stage2 egg→circle (+ multi-ring) + residual.

    Overlay ostaje savršeni krugovi; Stage2 iterativno dovodi fizički D/T na njih.
    """
    if frame is None or not cal.has_board_ellipse():
        return ensure_ring_align_warp(cal, out_size=out_size)
    size = int(out_size) if out_size > 0 else DEBUG_WARP_SIZE
    warped1 = warp_topdown_stage1(frame, cal, out_size=size)
    if warped1 is None:
        return ensure_ring_align_warp(cal, out_size=out_size)

    d1 = detect_double_outer_ellipse_on_warp(warped1)
    if d1 is None:
        return ensure_ring_align_warp(cal, out_size=out_size)
    src = _stage2_source_center(warped1, d1)
    cx_out, _, board_r = _warp_radii(size)
    d_semi = max(_mean_ellipse_semi(d1), 1.0)
    t1 = detect_triple_outer_ellipse_on_warp(warped1, d1)
    target_t = board_r * RING_TRIPLE_OUTER_FRAC
    bull1 = detect_bull_center_on_warp(warped1)
    bull_off1 = (
        math.hypot(bull1[0] - cx_out, bull1[1] - cx_out) if bull1 is not None else 0.0
    )
    need_s2 = _stage2_needed(d1, board_r, (cx_out, cx_out)) or (
        t1 is not None
        and abs(_mean_ellipse_semi(t1) - target_t) / board_r > STAGE2_RADIUS_MAX_DEV
    ) or (bull_off1 >= 0.6)

    warp1_d, warp1_t = d1, t1
    # Stage1 omjeri pojasa (inner/outer) — 4-ring Stage2 dovodi ih na nominalne krugove.
    s1_bands = _measure_rg_ring_band_radii(warped1, board_r=_mean_ellipse_semi(d1))
    di_ratio = float(RING_DOUBLE_INNER_FRAC)
    ti_ratio = float(RING_TRIPLE_INNER_FRAC / max(RING_TRIPLE_OUTER_FRAC, 1e-6))
    if s1_bands is not None:
        d_o = max(float(s1_bands["double_outer"]), 1.0)
        t_o = max(float(s1_bands["triple_outer"]), 1.0)
        di_ratio = float(np.clip(float(s1_bands["double_inner"]) / d_o, 0.88, 0.985))
        ti_ratio = float(np.clip(float(s1_bands["triple_inner"]) / t_o, 0.82, 0.985))

    if not need_s2:
        print(
            f"[calib] Stage2 skip (Stage1 OK) D~{d_semi:.1f} oval={_ellipse_ovality(d1):.3f}",
            flush=True,
        )
        src = (float(cx_out), float(cx_out))
        warp1_d = _circle_as_ellipse(cx_out, cx_out, board_r)
        warp1_t = None
    else:
        mode = "multi" if t1 is not None else "double"
        print(
            f"[calib] Stage2 {mode}-align "
            f"D~{d_semi:.1f} oval={_ellipse_ovality(d1):.3f} "
            f"src0=({src[0]:.1f},{src[1]:.1f}) bull_off1={bull_off1:.2f} "
            f"di={di_ratio:.4f} ti={ti_ratio:.4f}",
            flush=True,
        )

    # Residual: bull → cx_out, zatim po-kutu D/T elipse da warp sjede na idealne krugove.
    best_score = 1e9
    for _iter in range(7):
        warped2 = apply_ring_align_warp(
            warped1,
            warp1_d,
            out_size=size,
            warp1_triple_outer=warp1_t,
            center=src,
            double_inner_ratio=di_ratio,
            triple_inner_ratio=ti_ratio,
        )
        scale = board_r / max(_mean_ellipse_semi(warp1_d), 1.0)
        did = False

        bull_off = _warp_bull_center_offset(warped2)
        if bull_off is not None:
            bdx, bdy = bull_off
            bull_err = math.hypot(bdx, bdy)
            if bull_err >= 0.25:
                src = (src[0] + 0.95 * bdx / scale, src[1] + 0.95 * bdy / scale)
                did = True
                print(
                    f"[calib] Stage2 bull-center err={bull_err:.2f}px "
                    f"off=({bdx:+.2f},{bdy:+.2f}) -> src=({src[0]:.1f},{src[1]:.1f})",
                    flush=True,
                )

        score = 0.0
        r_d = _per_angle_rg_outer_radii(
            warped2,
            board_r=board_r,
            expected_r=board_r,
            r_lo=board_r * 0.88,
            r_hi=board_r * 1.10,
        )
        if r_d is not None:
            finite = r_d[np.isfinite(r_d)]
            d_err_mean = float(np.mean(finite) - board_r)
            d_err_p90 = float(np.percentile(np.abs(finite - board_r), 90))
            score += d_err_p90 + abs(d_err_mean)
            if abs(d_err_mean) >= 0.25:
                corr = float(np.clip(float(np.mean(finite)) / board_r, 0.92, 1.08))
                warp1_d = _scale_ellipse(warp1_d, corr)
                if warp1_t is not None:
                    warp1_t = _scale_ellipse(warp1_t, corr)
                did = True
                print(
                    f"[calib] Stage2 D-radius corr={corr:.4f} (mean={d_err_mean:+.2f}px)",
                    flush=True,
                )
            elif d_err_p90 >= 0.55:
                refined = _refine_stage2_ellipse_from_residual(
                    warp1_d, src, r_d, board_r, float(size)
                )
                if refined is not None:
                    warp1_d = refined
                    did = True
                    print(
                        f"[calib] Stage2 D-angle refine "
                        f"p90={d_err_p90:.2f} mean={d_err_mean:+.2f}",
                        flush=True,
                    )

        if warp1_t is not None:
            r_t = _per_angle_rg_outer_radii(
                warped2,
                board_r=board_r,
                expected_r=target_t,
                r_lo=target_t * 0.82,
                r_hi=min(board_r * 0.92, target_t * 1.18),
            )
            if r_t is not None:
                finite_t = r_t[np.isfinite(r_t)]
                t_err_mean = float(np.mean(finite_t) - target_t)
                t_err_p90 = float(np.percentile(np.abs(finite_t - target_t), 90))
                score += t_err_p90 + abs(t_err_mean)
                if abs(t_err_mean) >= 0.22:
                    corr_t = float(
                        np.clip(float(np.mean(finite_t)) / max(target_t, 1.0), 0.90, 1.12)
                    )
                    warp1_t = _scale_ellipse(warp1_t, corr_t)
                    did = True
                    print(
                        f"[calib] Stage2 T-radius corr={corr_t:.4f} (mean={t_err_mean:+.2f}px)",
                        flush=True,
                    )
                elif t_err_p90 >= 0.55:
                    refined_t = _refine_stage2_ellipse_from_residual(
                        warp1_t, src, r_t, target_t, float(size)
                    )
                    if refined_t is not None:
                        warp1_t = refined_t
                        did = True
                        print(
                            f"[calib] Stage2 T-angle refine "
                            f"p90={t_err_p90:.2f} mean={t_err_mean:+.2f}",
                            flush=True,
                        )

        if score < best_score - 0.02:
            best_score = score
        elif did and score >= best_score - 0.02 and _iter >= 1:
            # Nema napretka — prestani da ne oscilira.
            break

        if not did:
            break

    final_img = apply_ring_align_warp(
        warped1,
        warp1_d,
        out_size=size,
        warp1_triple_outer=warp1_t,
        center=src,
        double_inner_ratio=di_ratio,
        triple_inner_ratio=ti_ratio,
    )
    # Overlay/scoring = idealni krugovi; warp mora sjesti na njih.
    ell = _nominal_final_ring_ellipses(size)
    d_out_f = 1.0
    d_in_f = float(RING_DOUBLE_INNER_FRAC * WARP_RING_INNER_TIGHTEN)
    t_out_f = float(RING_TRIPLE_OUTER_FRAC)
    t_in_f = float(RING_TRIPLE_INNER_FRAC * WARP_RING_INNER_TIGHTEN)

    stats = _warp_ring_center_offset(final_img, board_r=board_r)
    bull_final = _warp_bull_center_offset(final_img)
    if stats is not None:
        dx, dy, d_meas, t_meas = stats
        btxt = ""
        if bull_final is not None:
            btxt = f" bull_off=({bull_final[0]:+.2f},{bull_final[1]:+.2f})"
        print(
            f"[calib] final D_off=({dx:+.2f},{dy:+.2f}) "
            f"D_err={d_meas - board_r:+.2f} T_err={t_meas - target_t:+.2f}{btxt}",
            flush=True,
        )
    r_d_f = _per_angle_rg_outer_radii(
        final_img,
        board_r=board_r,
        expected_r=board_r,
        r_lo=board_r * 0.88,
        r_hi=board_r * 1.10,
    )
    if r_d_f is not None:
        fin = r_d_f[np.isfinite(r_d_f)]
        print(
            f"[calib] warp-on-circle D mean={float(np.mean(fin))-board_r:+.2f} "
            f"max={float(np.max(np.abs(fin-board_r))):.2f}px",
            flush=True,
        )

    return replace(
        cal,
        measured_double_outer_frac=d_out_f,
        measured_double_inner_frac=d_in_f,
        measured_triple_outer_frac=t_out_f,
        measured_triple_inner_frac=t_in_f,
        warp_ring_size=size,
        warp_double_outer=ell["double_outer"],
        warp_double_inner=ell["double_inner"],
        warp_triple_outer=ell["triple_outer"],
        warp_triple_inner=ell["triple_inner"],
        warp1_double_outer=warp1_d,
        warp1_triple_outer=warp1_t,
        warp1_center=src,
        stage2_double_inner_ratio=float(di_ratio),
        stage2_triple_inner_ratio=float(ti_ratio),
    )


def _warp_output_size(frame: np.ndarray) -> int:
    """Warp u punoj rezoluciji kamere (bez smanjivanja na 220px)."""
    h, w = frame.shape[:2]
    return int(min(h, w))


def _rg_label_at(
    red: np.ndarray, green: np.ndarray, bull: Tuple[float, float], angle_deg: float, radius: float
) -> int:
    h, w = red.shape[:2]
    x, y = _sample_xy(bull, angle_deg, radius)
    if x < 0 or x >= w or y < 0 or y >= h:
        return 0
    rv, gv = red[y, x] > 0, green[y, x] > 0
    if rv and not gv:
        return 1
    if gv and not rv:
        return -1
    return 0


def _rg_boundary_profile(
    red: np.ndarray, green: np.ndarray, bull: Tuple[float, float], bull_r: float
) -> np.ndarray:
    """Jačina prijelaza crveno/zeleno po kutu (granice segmenata)."""
    n = 360
    rg = np.zeros(n, dtype=np.int8)
    for radius, wgt in ((bull_r * 5.2, 2), (bull_r * 7.8, 1.5), (bull_r * 8.6, 1.2)):
        ring = np.zeros(n, dtype=np.int8)
        for i in range(n):
            ring[i] = _rg_label_at(red, green, bull, float(i), radius)
        hit = (ring != 0) & (np.roll(ring, 1) != 0) & (ring != np.roll(ring, 1))
        rg = rg + hit.astype(np.int8) * int(wgt)
    profile = rg.astype(np.float32)
    return np.convolve(profile, np.ones(5) / 5.0, mode="same")


def _score_boundary_grid(profile: np.ndarray, rotation_deg: float) -> float:
    angles = (rotation_deg + np.arange(SEGMENT_COUNT, dtype=np.float32) * SEGMENT_STEP_DEG) % 360.0
    grid = np.arange(360, dtype=np.float32)
    return float(np.interp(angles, grid, profile).sum())


def _find_boundary_rotation(profile: np.ndarray) -> float:
    best_rot = 0.0
    best_score = -1.0
    for rot in np.arange(0.0, 360.0, 0.5, dtype=np.float32):
        score = _score_boundary_grid(profile, float(rot))
        if score > best_score:
            best_score = score
            best_rot = float(rot)
    return best_rot


def _board_ellipse_mask(shape: Tuple[int, int], ellipse: Ellipse, shrink: float = 0.99) -> np.ndarray:
    """Maska unutar double elipse (bez brojeva i okoline)."""
    h, w = shape[:2]
    (ecx, ecy), (ew, eh), ang = ellipse
    ax = max(1, int(round(ew * 0.5 * shrink)))
    ay = max(1, int(round(eh * 0.5 * shrink)))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (int(round(ecx)), int(round(ecy))), (ax, ay), ang, 0, 360, 255, -1)
    return mask


def _canny_edge_map(gray: np.ndarray, ellipse: Optional[Ellipse] = None) -> np.ndarray:
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    mask: Optional[np.ndarray] = None
    if ellipse is not None:
        mask = _board_ellipse_mask(gray.shape, ellipse)
        gray = cv2.bitwise_and(gray, gray, mask=mask)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    if mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=mask)
    return edges


def _canny_radial_profile(
    edges: np.ndarray,
    bull: Tuple[float, float],
    bull_r: float,
    board_r: float,
) -> np.ndarray:
    """Brzi vektorizirani profil: Canny suma duz 360 radijalnih zraka."""
    h, w = edges.shape[:2]
    bx, by = bull
    rad = np.deg2rad(np.arange(360, dtype=np.float32))
    sin_t = np.sin(rad)[:, None]
    cos_t = (-np.cos(rad))[:, None]
    # Širi opseg da uključimo cijeli vanjski double rub.
    radii = np.arange(bull_r * 3.8, board_r * 0.99, 1.0, dtype=np.float32)[None, :]
    xs = np.clip((bx + radii * sin_t).astype(np.int32), 0, w - 1)
    ys = np.clip((by + radii * cos_t).astype(np.int32), 0, h - 1)
    return edges[ys, xs].mean(axis=1).astype(np.float32)


def _find_rotation_from_profile(profile: np.ndarray) -> float:
    rots = np.arange(0.0, 360.0, 1.0, dtype=np.float32)
    slots = np.arange(SEGMENT_COUNT, dtype=np.float32) * SEGMENT_STEP_DEG
    angles = (rots[:, None] + slots[None, :]) % 360.0
    scores = np.interp(angles.ravel(), np.arange(360, dtype=np.float32), profile).reshape(-1, SEGMENT_COUNT)
    return float(rots[int(np.argmax(scores.sum(axis=1)))])


def _refine_angles_from_profile(profile: np.ndarray, rotation: float) -> Tuple[float, ...]:
    grid = np.arange(360, dtype=np.float32)
    refined: List[float] = []
    deltas = np.arange(-EDGE_REFINE_RANGE_DEG, EDGE_REFINE_RANGE_DEG + 0.01, EDGE_REFINE_STEP_DEG)
    for k in range(SEGMENT_COUNT):
        nominal = (rotation + k * SEGMENT_STEP_DEG) % 360.0
        local = (nominal + deltas) % 360.0
        vals = np.interp(local, grid, profile)
        refined.append(float(local[int(np.argmax(vals))]))
    return tuple(refined)


def _angle_dist_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _fitline_angle(vx: float, vy: float) -> float:
    return math.degrees(math.atan2(vx, -vy)) % 360.0


def _collect_annulus_edge_points(
    edges: np.ndarray,
    bull: Tuple[float, float],
    bull_r: float,
    board_r: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.where(edges > 0)
    if len(xs) < FITLINE_MIN_POINTS:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        )

    bx, by = bull
    dx = xs.astype(np.float32) - bx
    dy = ys.astype(np.float32) - by
    dist = np.hypot(dx, dy)
    valid = (dist >= bull_r * 2.8) & (dist <= board_r * 0.94)
    xs, ys, dx, dy, dist = xs[valid], ys[valid], dx[valid], dy[valid], dist[valid]
    if len(xs) < FITLINE_MIN_POINTS:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        )

    angles = np.degrees(np.arctan2(dx, -dy))
    angles = (angles + 360.0) % 360.0
    dist_out = dist.astype(np.float32)
    return xs.astype(np.float32), ys.astype(np.float32), dx, dy, angles.astype(np.float32), dist_out


def _perp_dist_to_diameter(dx: np.ndarray, dy: np.ndarray, angle_deg: float) -> np.ndarray:
    """Udaljenost tocke od promjera kroz bull pod kutom angle_deg."""
    st, ct = _angle_sin_cos(angle_deg)
    return np.abs(dx.astype(np.float32) * ct - dy.astype(np.float32) * st)


def _mask_points_on_diameter(
    dx: np.ndarray,
    dy: np.ndarray,
    angle_deg: float,
    line_dist_px: float,
) -> np.ndarray:
    return _perp_dist_to_diameter(dx, dy, angle_deg) < line_dist_px


def _diameter_sep_deg(a: float, b: float) -> float:
    d = abs((a % 180.0) - (b % 180.0))
    return min(d, 180.0 - d)


def _fit_diameter_angle(xs: np.ndarray, ys: np.ndarray) -> Optional[float]:
    if len(xs) < FITLINE_MIN_POINTS:
        return None
    pts = np.column_stack([xs, ys])
    line = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    return _fitline_angle(float(line[0]), float(line[1]))


def _fit_diameter_at_angle(
    xs: np.ndarray,
    ys: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    seed_angle: float,
) -> float:
    ang: Optional[float] = None
    for line_dist in (DIAMETER_LINE_DIST_PX, DIAMETER_LINE_DIST_PX + 2.0, DIAMETER_LINE_DIST_PX + 4.0):
        mask = _mask_points_on_diameter(dx, dy, seed_angle, line_dist)
        ang = _fit_diameter_angle(xs[mask], ys[mask])
        if ang is not None:
            break
    if ang is None:
        return seed_angle % 360.0

    refine_mask = _mask_points_on_diameter(dx, dy, ang, FITLINE_REFINE_LINE_DIST_PX)
    if int(refine_mask.sum()) >= FITLINE_MIN_POINTS:
        refined = _fit_diameter_angle(xs[refine_mask], ys[refine_mask])
        if refined is not None:
            ang = refined
    return ang % 360.0


def _diameter_vote_scores(
    dx: np.ndarray,
    dy: np.ndarray,
    line_dist_px: float,
) -> Tuple[np.ndarray, np.ndarray]:
    grid = np.arange(0.0, 180.0, DIAMETER_VOTE_STEP_DEG, dtype=np.float32)
    rad = np.deg2rad(grid)
    st = np.sin(rad)
    ct = -np.cos(rad)
    perp = np.abs(dx[:, None] * ct[None, :] - dy[:, None] * st[None, :])
    scores = (perp < line_dist_px).sum(axis=0).astype(np.float32)
    kernel_w = max(3, int(round(DIAMETER_VOTE_SMOOTH_DEG / DIAMETER_VOTE_STEP_DEG)))
    kernel = np.ones(kernel_w, dtype=np.float32) / float(kernel_w)
    scores = np.convolve(scores, kernel, mode="same")
    return grid, scores


def _diameter_fit_score(dx: np.ndarray, dy: np.ndarray, angle_deg: float) -> int:
    return int(_mask_points_on_diameter(dx, dy, angle_deg, FITLINE_REFINE_LINE_DIST_PX).sum())


def _dedupe_diameters(
    diameter_angles: List[float],
    dx: np.ndarray,
    dy: np.ndarray,
) -> List[float]:
    scored = [(ang, _diameter_fit_score(dx, dy, ang)) for ang in diameter_angles]
    kept: List[float] = []
    for ang, _fit_score in sorted(scored, key=lambda item: -item[1]):
        if all(_diameter_sep_deg(ang, existing) >= DIAMETER_MERGE_SEP_DEG for existing in kept):
            kept.append(ang % 360.0)
    kept.sort(key=lambda a: a % 180.0)
    return kept[:DIAMETER_COUNT]


def _largest_diameter_gap_mod180(diameter_angles: List[float]) -> Tuple[float, float]:
    """Najveca praznina izmedu susjednih promjera (mod 180°) i njezin srediste."""
    mods = sorted((float(a) % 180.0 for a in diameter_angles))
    if len(mods) < 2:
        return 0.0, 180.0
    best_gap = 0.0
    best_mid = 0.0
    for i, a0 in enumerate(mods):
        a1 = mods[(i + 1) % len(mods)]
        gap = (a1 - a0) % 180.0
        if gap <= 0.0:
            gap = 180.0
        if gap > best_gap:
            best_gap = gap
            best_mid = (a0 + gap * 0.5) % 180.0
    return best_mid, best_gap


def _fill_one_missing_diameter(
    diameter_angles: List[float],
    xs: np.ndarray,
    ys: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
) -> List[float]:
    """
    Ako fali tocno jedan promjer (tipicno 12h-6h), dodaj ga u srediste najvece praznine.
    Ne dira postojece promjere kad ih je vec 10.
    """
    if len(diameter_angles) >= DIAMETER_COUNT:
        return diameter_angles[:DIAMETER_COUNT]
    if len(diameter_angles) != DIAMETER_COUNT - 1:
        return diameter_angles

    gap_mid, gap_deg = _largest_diameter_gap_mod180(diameter_angles)
    if gap_deg < DIAMETER_MISSING_GAP_MIN_DEG:
        return diameter_angles

    candidate = _fit_diameter_at_angle(xs, ys, dx, dy, gap_mid)
    if _diameter_fit_score(dx, dy, candidate) < FITLINE_MIN_POINTS:
        return diameter_angles

    filled = list(diameter_angles) + [candidate]
    return _dedupe_diameters(filled, dx, dy)


def _pick_diameter_seeds(
    angle_grid: np.ndarray,
    scores: np.ndarray,
    count: int,
    min_sep_deg: float,
    score_ratio: float,
) -> List[float]:
    if len(angle_grid) == 0 or float(scores.max()) <= 0.0:
        return []

    thresh = float(scores.max()) * score_ratio
    seeds: List[float] = []
    for idx in np.argsort(-scores):
        if float(scores[idx]) < thresh:
            break
        ang = float(angle_grid[int(idx)])
        if all(_diameter_sep_deg(ang, kept) >= min_sep_deg for kept in seeds):
            seeds.append(ang)
        if len(seeds) >= count:
            break
    return seeds


def _detect_ten_diameters(
    xs: np.ndarray,
    ys: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
) -> List[float]:
    grid, scores = _diameter_vote_scores(dx, dy, DIAMETER_LINE_DIST_PX)
    seeds = _pick_diameter_seeds(
        grid, scores, DIAMETER_COUNT, DIAMETER_MIN_SEP_DEG, DIAMETER_VOTE_SCORE_RATIO
    )

    if len(seeds) < DIAMETER_COUNT:
        grid2, scores2 = _diameter_vote_scores(dx, dy, DIAMETER_LINE_DIST_PX + 2.0)
        extra = _pick_diameter_seeds(
            grid2, scores2, DIAMETER_COUNT, DIAMETER_MIN_SEP_DEG, DIAMETER_VOTE_SCORE_RATIO * 0.85
        )
        for seed in extra:
            if all(_diameter_sep_deg(seed, kept) >= DIAMETER_MIN_SEP_DEG for kept in seeds):
                seeds.append(seed)
            if len(seeds) >= DIAMETER_COUNT:
                break

    if not seeds:
        return []

    diameter_angles = [_fit_diameter_at_angle(xs, ys, dx, dy, seed) for seed in seeds]
    diameter_angles = _dedupe_diameters(diameter_angles, dx, dy)
    return _fill_one_missing_diameter(diameter_angles, xs, ys, dx, dy)


def _wires_from_diameters(diameter_angles: List[float]) -> Tuple[float, ...]:
    wires: List[float] = []
    for diam in diameter_angles[:DIAMETER_COUNT]:
        wires.append(float(diam) % 360.0)
        wires.append((float(diam) + 180.0) % 360.0)
    return tuple(wires)


def _detect_wire_angles_fitline(
    edges: np.ndarray,
    bull: Tuple[float, float],
    bull_r: float,
    board_r: float,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """10 promjera iz Canny edge piksela, fitLine, svaki -> 2 zice."""
    xs, ys, dx, dy, _angles, _dist = _collect_annulus_edge_points(edges, bull, bull_r, board_r)
    if len(xs) < FITLINE_MIN_POINTS:
        return (), ()

    diameter_angles = _detect_ten_diameters(xs, ys, dx, dy)
    if len(diameter_angles) < DIAMETER_COUNT - 2:
        return (), ()

    wires = tuple(sorted(_wires_from_diameters(diameter_angles)))
    return wires, tuple(diameter_angles)


def _pick_wire_angles_from_profile(profile: np.ndarray) -> Tuple[float, ...]:
    """
    Iz radijalnog Canny profila procijeni 20 zica tako da budu strogo razmaknute 18°.

    Implementacija:
    1) nađi najbolji globalni offset `t0` (modulo 18°) preko comb-sume profila
    2) za svaku nominalnu žicu (t0 + k*18°) nađi lokalni maksimum u malom prozoru
    3) uzmi medijanu svih lokalnih korekcija i primijeni je globalno na sve žice

    Time razmak ostaje uniforman, a žice se i dalje “zalijepe” na maksimum profila.
    """
    prof = np.convolve(profile.astype(np.float32), np.ones(7, dtype=np.float32) / 7.0, mode="same")
    grid = np.arange(360, dtype=np.float32)

    k_steps = np.arange(SEGMENT_COUNT, dtype=np.float32) * SEGMENT_STEP_DEG
    t0_candidates = np.arange(0.0, SEGMENT_STEP_DEG, 0.15, dtype=np.float32)

    best_t0 = 0.0
    best_score = -1e30
    for t0 in t0_candidates:
        angles = (t0 + k_steps) % 360.0
        vals = np.interp(angles, grid, prof)
        score = float(vals.sum())
        if score > best_score:
            best_score = score
            best_t0 = float(t0)

    # Lokalni vrh profila oko svake zice (bez globalnog pomaka koji pomiče sve odjednom).
    window = np.arange(-5.0, 5.01, 0.25, dtype=np.float32)
    final: List[float] = []
    for k in range(SEGMENT_COUNT):
        nominal = (best_t0 + k * SEGMENT_STEP_DEG) % 360.0
        cands = (nominal + window) % 360.0
        cand_vals = np.interp(cands, grid, prof)
        best_i = int(np.argmax(cand_vals))
        final.append(float(cands[best_i]))

    return tuple(final)


def _detect_segment_boundaries(
    frame: np.ndarray,
    red_full: np.ndarray,
    green_full: np.ndarray,
    bull: Tuple[float, float],
    bull_r: float,
    ellipse: Ellipse,
    cam_idx: Optional[int] = None,
) -> Tuple[float, ...]:
    """Canny unutar elipse, 10 promjera fitLine."""
    _ = cam_idx, red_full, green_full
    scale = WIRE_DETECT_SCALE
    if abs(scale - 1.0) > 1e-3:
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        bull_s = (bull[0] * scale, bull[1] * scale)
        bull_r_s = bull_r * scale
        ellipse_s = _scale_ellipse(ellipse, scale)
    else:
        small, bull_s, bull_r_s, ellipse_s = frame, bull, bull_r, ellipse

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = _canny_edge_map(gray, ellipse_s)
    board_r = max(ellipse_s[1][0], ellipse_s[1][1]) * 0.5

    wire_angles, _ = _detect_wire_angles_fitline(edges, bull_s, bull_r_s, board_r)
    if not wire_angles:
        profile = _canny_radial_profile(edges, bull_s, bull_r_s, board_r)
        profile = np.convolve(profile, np.ones(5, dtype=np.float32) / 5.0, mode="same")
        if float(profile.max()) >= 1.0:
            wire_angles = _pick_wire_angles_from_profile(profile)

    return wire_angles


def _interp_image_angle_from_board(
    board_angles: np.ndarray,
    boundaries: np.ndarray,
) -> np.ndarray:
    """Uniformni kut ploce (18°) -> stvarni kut u slici (perspektiva)."""
    n = len(boundaries)
    slot = (board_angles % 360.0) / SEGMENT_STEP_DEG
    idx = slot.astype(np.int32) % n
    frac = slot - np.floor(slot)
    a0 = boundaries[idx]
    a1 = boundaries[(idx + 1) % n]
    span = (a1 - a0) % 360.0
    span = np.where(span < 1.0, SEGMENT_STEP_DEG, span)
    return (a0 + frac * span) % 360.0


def _wire_sector_index(angle_deg: float, wire_angles_deg: Tuple[float, ...]) -> Optional[int]:
    """Indeks segmenta (izmedu zica k i k+1) koji sadrzi zadani kut."""
    n = min(len(wire_angles_deg), SEGMENT_COUNT)
    if n < 8:
        return None
    ang = angle_deg % 360.0
    for k in range(n):
        a0 = float(wire_angles_deg[k]) % 360.0
        a1 = float(wire_angles_deg[(k + 1) % n]) % 360.0
        if a1 <= a0:
            if ang >= a0 or ang < a1:
                return k
        elif a0 <= ang < a1:
            return k
    return 0


def _clockwise_wire_ring(
    bull: Tuple[float, float],
    ellipse: Ellipse,
    wire_angles_deg: Tuple[float, ...],
) -> Tuple[np.ndarray, List[int]]:
    """20 sjecista zica/elipse sortirano clockwise; vraca (tocke, originalni indeksi)."""
    indexed: List[Tuple[float, int, float, float]] = []
    for idx, ang in enumerate(wire_angles_deg[:SEGMENT_COUNT]):
        pt = _ray_to_ellipse_edge_f(bull, float(ang), ellipse)
        if pt is not None:
            indexed.append((float(ang) % 360.0, idx, float(pt[0]), float(pt[1])))
    if len(indexed) < 8:
        return np.zeros((0, 2), dtype=np.float32), []
    indexed.sort(key=lambda item: item[0])
    order = [item[1] for item in indexed]
    pts = np.array([[item[2], item[3]] for item in indexed], dtype=np.float32)
    return pts, order


def _dst_angle_for_clockwise_wire(clockwise_idx: int, segment20_offset: int, n: int) -> float:
    """Kut zice na top-down slici (clockwise od 12h). Seg.20: lijevo 351°, desno 9°."""
    off = int(segment20_offset) % max(1, n)
    return (SEG20_LEFT_WIRE_DST_DEG + (int(clockwise_idx) - off) * SEGMENT_STEP_DEG) % 360.0


def _clockwise_wire_source_angles(wire_angles_deg: Tuple[float, ...]) -> Tuple[np.ndarray, int]:
    n = min(len(wire_angles_deg), SEGMENT_COUNT)
    if n < 8:
        return np.zeros(0, dtype=np.float32), 0
    sorted_idx = sorted(range(n), key=lambda k: float(wire_angles_deg[k]) % 360.0)
    src = np.array([float(wire_angles_deg[k]) for k in sorted_idx], dtype=np.float32)
    return src, n


def _interp_source_angle_seg20_topdown(
    dst_angles: np.ndarray,
    clockwise_src: np.ndarray,
    segment20_offset: int,
) -> np.ndarray:
    """Za svaki kut na top-down slici -> odgovarajuci kut u izvornoj slici."""
    n = len(clockwise_src)
    if n < 8:
        return dst_angles
    off = int(segment20_offset) % n
    rel = (dst_angles - SEG20_LEFT_WIRE_DST_DEG) % 360.0
    wire_slot = rel / SEGMENT_STEP_DEG
    idx = (off + np.floor(wire_slot).astype(np.int32)) % n
    frac = (wire_slot - np.floor(wire_slot)).astype(np.float32)
    a0 = clockwise_src[idx]
    a1 = clockwise_src[(idx + 1) % n]
    start = np.mod(a0, 360.0)
    end = np.mod(a1, 360.0)
    span = np.mod(end - start, 360.0)
    span = np.where(span < 1.0, SEGMENT_STEP_DEG, span)
    return np.mod(start + frac * span, 360.0).astype(np.float32)


def _homography_topdown_seg20(
    bull: Tuple[float, float],
    ellipse: Ellipse,
    wire_angles_deg: Tuple[float, ...],
    out_size: int,
    segment20_offset: int,
) -> Optional[np.ndarray]:
    """Homografija: bull->centar, zice + medusegmentne tocke na elipsi -> krug."""
    clockwise_pts, _order = _clockwise_wire_ring(bull, ellipse, wire_angles_deg)
    if len(clockwise_pts) < 8:
        return None

    n_pts = len(clockwise_pts)
    cx_out, view_r, board_r = _warp_radii(out_size)
    dst_ring = []
    for j in range(n_pts):
        board_ang = _dst_angle_for_clockwise_wire(j, segment20_offset, n_pts)
        st, ct = _angle_sin_cos(board_ang)
        dst_ring.append((cx_out + board_r * st, cx_out + board_r * ct))
    dst_ring_arr = np.array(dst_ring, dtype=np.float32)

    # Dodatne tocke na elipsi (sredina svakog segmenta) — stabilniji krug, manje
    # "tankog" doublea na jednoj strani zbog pogreske samo 20 zica.
    extra_src: List[Tuple[float, float]] = []
    extra_dst: List[Tuple[float, float]] = []
    sorted_ang = sorted(float(a) % 360.0 for a in wire_angles_deg[:SEGMENT_COUNT])
    if len(sorted_ang) >= 8:
        for i, a0 in enumerate(sorted_ang):
            a1 = sorted_ang[(i + 1) % len(sorted_ang)]
            span = (a1 - a0) % 360.0
            mid_src = (a0 + 0.5 * span) % 360.0
            pt = _ray_to_ellipse_edge_f(bull, mid_src, ellipse)
            if pt is None:
                continue
            # Isto pomicanje kao zica i (clockwise index i) -> sredina segmenta na dst.
            board_ang = _dst_angle_for_clockwise_wire(i, segment20_offset, n_pts)
            board_mid = (board_ang + 0.5 * SEGMENT_STEP_DEG) % 360.0
            st, ct = _angle_sin_cos(board_mid)
            extra_src.append((float(pt[0]), float(pt[1])))
            extra_dst.append((cx_out + board_r * st, cx_out + board_r * ct))

    src_list = [np.array([[bull[0], bull[1]]], dtype=np.float32), clockwise_pts]
    dst_list = [np.array([[cx_out, cx_out]], dtype=np.float32), dst_ring_arr]
    if extra_src:
        src_list.append(np.array(extra_src, dtype=np.float32))
        dst_list.append(np.array(extra_dst, dtype=np.float32))
    src_pts = np.vstack(src_list).astype(np.float32)
    dst_pts = np.vstack(dst_list).astype(np.float32)
    H, _mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.5)
    if H is None:
        H, _mask = cv2.findHomography(src_pts, dst_pts, 0)
    return H


def _resolve_segment20_geometry(
    wire_angles_deg: Tuple[float, ...],
    offset: int,
) -> Tuple[int, int, float]:
    n = min(len(wire_angles_deg), SEGMENT_COUNT)
    if n < 8:
        return -1, -1, 0.0
    sorted_idx = sorted(range(n), key=lambda k: float(wire_angles_deg[k]) % 360.0)
    angular_idx = int(offset) % n
    k0, k1, center = _segment20_wire_pair(wire_angles_deg, sorted_idx, angular_idx)
    return k0, k1, center % 360.0


def _segment20_align_rotation_deg(
    wire_angles_deg: Tuple[float, ...],
    segment20_offset: int,
    homography_shift: int,
) -> float:
    """Rotacija nije potrebna — seg.20 je u homografiji."""
    _ = wire_angles_deg, segment20_offset, homography_shift
    return 0.0


def refresh_segment20_fields(cal: BoardCalibration) -> None:
    if not cal.has_wires():
        cal.segment20_wire_sector = -1
        cal.segment20_wire_end = -1
        cal.align_rotation_deg = 0.0
        return
    k0, k1, _center = _resolve_segment20_geometry(cal.wire_angles_deg, cal.segment20_offset)
    cal.segment20_wire_sector = k0
    cal.segment20_wire_end = k1
    cal.align_rotation_deg = _segment20_align_rotation_deg(
        cal.wire_angles_deg,
        cal.segment20_offset,
        cal.homography_shift,
    )


def rotate_board_warp(warped: np.ndarray, align_rotation_deg: float) -> np.ndarray:
    """Rotira warpiranu sliku tako da segment 20 bude gore."""
    if abs(align_rotation_deg) < 0.05:
        return warped
    h, w = warped.shape[:2]
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    M = cv2.getRotationMatrix2D((cx, cy), align_rotation_deg, 1.0)
    rotated = cv2.warpAffine(
        warped,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(min(cx, cy) - 2), 255, -1)
    return cv2.bitwise_and(rotated, rotated, mask=mask)


def _warp_homography_topdown(
    frame: np.ndarray,
    bull: Tuple[float, float],
    bull_r: float,
    ellipse: Ellipse,
    wire_angles_deg: Tuple[float, ...],
    out_size: int,
    segment20_offset: int = 0,
) -> Optional[Tuple[np.ndarray, Tuple[float, float], float, float, int]]:
    """
    Top-down warp: homografija bull + 20 zica -> seg.20 na vrhu (351°/9°).
    """
    clockwise_pts, _order = _clockwise_wire_ring(bull, ellipse, wire_angles_deg)
    if len(clockwise_pts) < 8:
        return None

    H = _homography_topdown_seg20(bull, ellipse, wire_angles_deg, out_size, segment20_offset)
    if H is None:
        return None

    cx_out, view_r, board_r = _warp_radii(out_size)

    warped = cv2.warpPerspective(frame, H, (out_size, out_size), flags=cv2.INTER_CUBIC)
    circle_mask = np.zeros((out_size, out_size), dtype=np.uint8)
    cv2.circle(circle_mask, (int(cx_out), int(cx_out)), int(view_r), 255, -1)
    warped = cv2.bitwise_and(warped, warped, mask=circle_mask)

    mean_d = float(
        np.mean([math.hypot(x - bull[0], y - bull[1]) for x, y in clockwise_pts])
    )
    bull_w = (cx_out, cx_out)
    bull_r_w = bull_r * (board_r / max(mean_d, 1.0))
    return warped, bull_w, bull_r_w, board_r, int(segment20_offset) % len(clockwise_pts)


def _warp_radial_topdown(
    frame: np.ndarray,
    bull: Tuple[float, float],
    bull_r: float,
    ellipse: Ellipse,
    out_size: int = 0,
    wire_angles_deg: Tuple[float, ...] = (),
    segment20_offset: int = 0,
) -> Tuple[np.ndarray, Tuple[float, float], float, float]:
    """
    Top-down warp: bull u sredinu, double prsten -> krug.
    Kutno mapiranje uskladeno s seg.20 na vrhu (351°/9°).
    """
    if out_size <= 0:
        out_size = _warp_output_size(frame)
    bx, by = bull
    cx_out, view_r, board_r = _warp_radii(out_size)

    yy, xx = np.mgrid[0:out_size, 0:out_size].astype(np.float32)
    dx = xx - cx_out
    dy = yy - cx_out
    r_px = np.sqrt(dx * dx + dy * dy)
    theta_board = (np.degrees(np.arctan2(dx, -dy)) % 360.0).astype(np.float32)

    if len(wire_angles_deg) >= SEGMENT_COUNT - 2:
        clockwise_src, _n = _clockwise_wire_source_angles(wire_angles_deg)
        boundary_dists = _wire_boundary_distances(bull, ellipse, wire_angles_deg)
        mean_edge = float(np.mean(boundary_dists))
        theta_image = _interp_source_angle_seg20_topdown(
            theta_board, clockwise_src, segment20_offset
        )
        sin_t = np.sin(np.deg2rad(theta_image))
        cos_t = -np.cos(np.deg2rad(theta_image))
        edge_dist = _interp_boundary_distance(theta_board, boundary_dists)
    else:
        edge_table = np.array([_ray_dist_to_ellipse(bull, float(a), ellipse) for a in range(360)], dtype=np.float32)
        mean_edge = float(np.mean(edge_table))
        sin_t = np.sin(np.deg2rad(theta_board))
        cos_t = -np.cos(np.deg2rad(theta_board))
        edge_dist = np.interp(theta_board, np.arange(360, dtype=np.float32), edge_table)

    inside_board = r_px <= board_r
    beyond = np.maximum(r_px - board_r, 0.0)
    margin = max(view_r - board_r, 1.0)
    extrap = 1.0 + (beyond / margin) * WARP_OUTSIDE_EXTRAPOL
    scale = np.where(inside_board, r_px / max(board_r, 1.0), extrap)
    src_r = scale * edge_dist
    map_x = (bx + src_r * sin_t).astype(np.float32)
    map_y = (by + src_r * cos_t).astype(np.float32)

    warped = cv2.remap(frame, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    inside = (r_px <= view_r).astype(np.uint8)
    warped = cv2.bitwise_and(warped, warped, mask=inside)

    bull_w = (cx_out, cx_out)
    bull_r_w = bull_r * (board_r / max(mean_edge, 1.0))
    return warped, bull_w, bull_r_w, board_r


def warp_board_topdown(
    frame: np.ndarray,
    bull: Tuple[float, float],
    bull_r: float,
    ellipse: Ellipse,
    out_size: int = 0,
    wire_angles_deg: Tuple[float, ...] = (),
    segment20_offset: int = 0,
) -> Tuple[np.ndarray, Tuple[float, float], float, float, int]:
    """Top-down warp: homografija (seg.20 gore) ili radijalni fallback."""
    if out_size <= 0:
        out_size = _warp_output_size(frame)
    if len(wire_angles_deg) >= SEGMENT_COUNT - 2:
        homography = _warp_homography_topdown(
            frame,
            bull,
            bull_r,
            ellipse,
            wire_angles_deg,
            out_size,
            segment20_offset=segment20_offset,
        )
        if homography is not None:
            return homography
    warped, bull_w, bull_r_w, out_radius = _warp_radial_topdown(
        frame,
        bull,
        bull_r,
        ellipse,
        out_size,
        wire_angles_deg,
        segment20_offset=segment20_offset,
    )
    return warped, bull_w, bull_r_w, out_radius, int(segment20_offset) % SEGMENT_COUNT


def _warp_board_circular(
    frame: np.ndarray,
    bull: Tuple[float, float],
    bull_r: float,
    ellipse: Ellipse,
    out_size: int = WARP_SIZE,
) -> Tuple[np.ndarray, Tuple[float, float], float, float]:
    warped, bull_w, bull_r_w, out_radius, _shift = warp_board_topdown(
        frame, bull, bull_r, ellipse, out_size
    )
    return warped, bull_w, bull_r_w, out_radius


def _sample_rg_ring(
    red: np.ndarray,
    green: np.ndarray,
    bull: Tuple[float, float],
    bull_r: float,
    board_r: float,
) -> np.ndarray:
    """360 uzoraka crveno/zeleno na triple+double prstenu (+1/-1/0)."""
    n = 360
    angles = np.arange(n, dtype=np.float32)
    h, w = red.shape[:2]
    samples = np.zeros(n, dtype=np.float32)
    weights = np.zeros(n, dtype=np.float32)
    for radius, wgt in (
        (bull_r * 5.0, 1.0),
        (bull_r * 8.0, 0.85),
        (bull_r * 6.5, 0.6),
    ):
        for i, ang in enumerate(angles):
            x, y = _sample_xy(bull, float(ang), radius)
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            rv, gv = red[y, x] > 0, green[y, x] > 0
            if rv and not gv:
                samples[i] += wgt
                weights[i] += wgt
            elif gv and not rv:
                samples[i] -= wgt
                weights[i] += wgt
    weights[weights < 1e-6] = 1.0
    out = samples / weights
    out[weights < 0.5] = 0.0
    return out


def _expected_rg_color(angle_deg: float, rotation_deg: float) -> float:
    """Ocekivana boja segmenta: +1 zeleno, -1 crveno (segment 20 gore)."""
    rel = (angle_deg - rotation_deg) % 360.0
    seg_idx = int(rel / SEGMENT_STEP_DEG) % SEGMENT_COUNT
    return 1.0 if seg_idx % 2 == 0 else -1.0


def _detect_board_rotation(red: np.ndarray, green: np.ndarray, bull: Tuple[float, float], bull_r: float, board_r: float) -> float:
    """Usporedi 360 RG uzoraka s ocekivanim uzorkom (period 36°) -> rotacija ploce."""
    samples = _sample_rg_ring(red, green, bull, bull_r, board_r)
    if float(np.max(np.abs(samples))) < 0.15:
        return 0.0

    rots = np.arange(360, dtype=np.float32)
    idx = np.arange(360, dtype=np.float32)
    seg_idx = ((idx[:, None] - rots[None, :]) % 360.0 / SEGMENT_STEP_DEG).astype(np.int32) % SEGMENT_COUNT
    expected = np.where(seg_idx % 2 == 0, 1.0, -1.0)
    scores = (samples[:, None] * expected).sum(axis=0)
    return float(rots[int(np.argmax(scores))])


def _sobel_edge_map(gray: np.ndarray) -> np.ndarray:
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _edge_score_at_angle(
    edges: np.ndarray,
    bull: Tuple[float, float],
    angle_deg: float,
    bull_r: float,
    board_r: float,
) -> float:
    h, w = edges.shape[:2]
    st, ct = _angle_sin_cos(angle_deg)
    score = 0.0
    for radius, wgt in (
        (bull_r * 4.8, 1.6),
        (bull_r * 5.6, 1.8),
        (bull_r * 7.6, 1.6),
        (bull_r * 8.4, 1.4),
        (bull_r * 6.2, 1.0),
    ):
        x = int(round(bull[0] + radius * st))
        y = int(round(bull[1] + radius * ct))
        if 0 <= x < w and 0 <= y < h:
            score += float(edges[y, x]) * wgt
    return score


def _refine_segment_angles_from_edges(
    edges: np.ndarray,
    bull: Tuple[float, float],
    bull_r: float,
    board_r: float,
    initial_angles: List[float],
) -> List[float]:
    refined: List[float] = []
    for a0 in initial_angles:
        best_angle = a0
        best_score = -1.0
        candidate = a0
        while candidate < a0 + EDGE_REFINE_RANGE_DEG + 0.01:
            s = _edge_score_at_angle(edges, bull, candidate, bull_r, board_r)
            if s > best_score:
                best_score = s
                best_angle = candidate
            candidate += EDGE_REFINE_STEP_DEG
        candidate = a0 - EDGE_REFINE_STEP_DEG
        while candidate >= a0 - EDGE_REFINE_RANGE_DEG:
            s = _edge_score_at_angle(edges, bull, candidate, bull_r, board_r)
            if s > best_score:
                best_score = s
                best_angle = candidate
            candidate -= EDGE_REFINE_STEP_DEG
        refined.append(best_angle % 360.0)
    return refined


def _segment_boundary_angles_from_ring(
    red: np.ndarray,
    green: np.ndarray,
    bull: Tuple[float, float],
    bull_r: float,
    initial_angles: List[float],
) -> List[float]:
    """Grube granice iz prijelaza crveno/zeleno na triple prstenu."""
    n = 360
    radius = bull_r * 5.2
    h, w = red.shape[:2]
    rg = np.zeros(n, dtype=np.int8)
    for i in range(n):
        x, y = _sample_xy(bull, float(i), radius)
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        if red[y, x] > 0:
            rg[i] = 1
        elif green[y, x] > 0:
            rg[i] = -1

    transitions: List[float] = []
    for i in range(1, n):
        if rg[i] != 0 and rg[i - 1] != 0 and rg[i] != rg[i - 1]:
            transitions.append((i - 0.5) % 360.0)

    if len(transitions) < 8:
        return list(initial_angles)

    def _nearest_transition(angle: float) -> float:
        return min(transitions, key=lambda t: min(abs(t - angle), 360.0 - abs(t - angle)))

    refined: List[float] = []
    for a0 in initial_angles:
        nearest = _nearest_transition(a0)
        if min(abs(nearest - a0), 360.0 - abs(nearest - a0)) <= SEGMENT_STEP_DEG * 0.5:
            refined.append(nearest)
        else:
            refined.append(a0)
    return refined


def _blend_angles_deg(a: float, b: float, weight_a: float) -> float:
    ra, rb = math.radians(a), math.radians(b)
    x = weight_a * math.cos(ra) + (1.0 - weight_a) * math.cos(rb)
    y = weight_a * math.sin(ra) + (1.0 - weight_a) * math.sin(rb)
    return math.degrees(math.atan2(y, x)) % 360.0


def _detect_wire_angles(
    frame: np.ndarray,
    red_full: np.ndarray,
    green_full: np.ndarray,
    bull: Tuple[float, float],
    bull_r: float,
    ellipse: Ellipse,
) -> Tuple[float, ...]:
    return _detect_segment_boundaries(frame, red_full, green_full, bull, bull_r, ellipse)


def _fit_double_ellipse(
    points: List[Tuple[float, float]],
    bull: Tuple[float, float],
    bull_r: float,
    min_dim: float,
) -> Tuple[Optional[Ellipse], float]:
    if len(points) < 10:
        return None, 0.0

    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    ellipse = cv2.fitEllipse(pts)
    (ecx, ecy), (width, height), _ = ellipse

    dists = [_ellipse_point_distance(ellipse, px, py) for px, py in points]
    med = float(np.median(dists))
    inliers = [(px, py) for (px, py), d in zip(points, dists) if d <= med * 2.0 + 2.0]
    if len(inliers) >= 10:
        pts2 = np.array(inliers, dtype=np.float32).reshape(-1, 1, 2)
        ellipse = cv2.fitEllipse(pts2)
        (ecx, ecy), (width, height), _ = ellipse
        dists = [_ellipse_point_distance(ellipse, px, py) for px, py in inliers]

    semi_max = max(width, height) * 0.5
    semi_min = min(width, height) * 0.5
    if semi_max < bull_r * 9.5 or semi_max > min_dim * 0.56:
        return None, 0.0
    if semi_min < bull_r * 5.0:
        return None, 0.0

    bcx, bcy = bull
    if math.hypot(ecx - bcx, ecy - bcy) > semi_max * 0.25:
        return None, 0.0

    mean_err = float(np.mean(dists))
    fit_score = 1.0 - min(mean_err / (semi_max * 0.12 + 1e-6), 1.0)
    coverage = min(len(inliers if len(inliers) >= 10 else points) / 90.0, 1.0)
    flatness = 1.0 - min(semi_min / semi_max, 1.0)
    confidence = 0.55 * fit_score + 0.30 * coverage + 0.15 * flatness
    return ellipse, float(confidence)


def _detect_double_ellipse(
    red: np.ndarray,
    green: np.ndarray,
    bull: Tuple[float, float],
    bull_r: float,
    min_dim: float,
    extra_points: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[Optional[Ellipse], float]:
    cx, cy = bull
    points = _sample_ellipse_fit_points(red, green, cx, cy, bull_r)
    extra = [p for p in (extra_points or []) if p is not None and len(p) >= 2]
    if extra:
        # Ručne točke na vanjskom rubu — ponovi da nadvladaju šum surrounda.
        boost = extra * 10
        points = list(points) + boost
    if len(points) < 12:
        points = _contour_double_points(red, green, bull, bull_r)
        if extra:
            points = list(points) + extra * 10
    ellipse, conf = _fit_double_ellipse(points, bull, bull_r, min_dim)
    if ellipse is not None:
        return _refine_ellipse_to_outer_double(ellipse, bull, bull_r, red, green), conf

    outer = _contour_double_points(red, green, bull, bull_r)
    if extra:
        outer = list(outer) + extra * 10
    ellipse, conf = _fit_double_ellipse(outer, bull, bull_r, min_dim)
    if ellipse is not None:
        return _refine_ellipse_to_outer_double(ellipse, bull, bull_r, red, green), conf
    return None, 0.0


def _angle_lerp_deg(a: float, b: float, t: float) -> float:
    a = float(a) % 360.0
    b = float(b) % 360.0
    diff = ((b - a + 180.0) % 360.0) - 180.0
    return (a + float(t) * diff) % 360.0


def _blend_ellipse(old: Ellipse, new: Ellipse, t: float) -> Ellipse:
    (oecx, oecy), (ow, oh), oang = old
    (necx, necy), (nw, nh), nang = new
    return (
        (oecx * (1.0 - t) + necx * t, oecy * (1.0 - t) + necy * t),
        (ow * (1.0 - t) + nw * t, oh * (1.0 - t) + nh * t),
        _angle_lerp_deg(oang, nang, t),
    )


def _blend_wire_angles(
    old_wires: Tuple[float, ...],
    new_wires: Tuple[float, ...],
    t: float,
) -> Tuple[float, ...]:
    n = min(len(old_wires), len(new_wires))
    if n < 8:
        return new_wires
    old_sorted = sorted(range(n), key=lambda k: float(old_wires[k]) % 360.0)
    new_sorted = sorted(range(n), key=lambda k: float(new_wires[k]) % 360.0)
    out = list(new_wires)
    for i in range(n):
        oa = float(old_wires[old_sorted[i]])
        nk = new_sorted[i]
        na = float(new_wires[nk])
        out[nk] = _angle_lerp_deg(oa, na, t)
    return tuple(out)


def _recover_calibration_geometry(old: BoardCalibration, new: BoardCalibration) -> BoardCalibration:
    """Ako nova detekcija ima samo bull, zadrzi elipsu/zice/seg.20 iz prethodne."""
    if not old.is_valid() or new is None or not new.is_valid():
        if new is not None and old is not None:
            patched = replace(new, segment20_offset=int(old.segment20_offset))
            refresh_segment20_fields(patched)
            return patched
        return new
    patched = new
    if not new.has_board_ellipse() and old.has_board_ellipse():
        patched = replace(
            patched,
            double_ellipse=old.double_ellipse,
            board_confidence=old.board_confidence,
        )
    if not patched.has_wires() and old.has_wires():
        patched = replace(patched, wire_angles_deg=old.wire_angles_deg)
    if not patched.has_measured_rings() and old.has_measured_rings():
        patched = replace(
            patched,
            measured_double_outer_frac=old.measured_double_outer_frac,
            measured_double_inner_frac=old.measured_double_inner_frac,
            measured_triple_outer_frac=old.measured_triple_outer_frac,
            measured_triple_inner_frac=old.measured_triple_inner_frac,
            warp_ring_size=old.warp_ring_size,
            warp_double_outer=old.warp_double_outer,
            warp_double_inner=old.warp_double_inner,
            warp_triple_outer=old.warp_triple_outer,
            warp_triple_inner=old.warp_triple_inner,
            warp1_double_outer=old.warp1_double_outer,
            warp1_triple_outer=old.warp1_triple_outer,
            warp1_center=old.warp1_center,
        )
    elif not patched.has_ring_align_warp() and old.has_ring_align_warp():
        patched = replace(
            patched,
            warp1_double_outer=old.warp1_double_outer,
            warp1_triple_outer=old.warp1_triple_outer,
            warp1_center=old.warp1_center,
        )
    patched = replace(patched, segment20_offset=int(old.segment20_offset))
    refresh_segment20_fields(patched)
    return patched


def _stabilize_calibration(old: BoardCalibration, new: BoardCalibration, alpha: float) -> BoardCalibration:
    """Usporavanje jittera: spoji novu detekciju s prethodnom kalibracijom."""
    t = max(0.0, min(1.0, float(alpha)))
    if t <= 0.0:
        return new
    ox, oy = old.center
    nx, ny = new.center
    center = (ox * (1.0 - t) + nx * t, oy * (1.0 - t) + ny * t)
    bull_r = old.bull_radius * (1.0 - t) + new.bull_radius * t
    confidence = old.confidence * (1.0 - t) + new.confidence * t

    ellipse = new.double_ellipse
    board_conf = new.board_confidence
    if old.double_ellipse is not None and new.double_ellipse is not None:
        ellipse = _blend_ellipse(old.double_ellipse, new.double_ellipse, t)
        board_conf = old.board_confidence * (1.0 - t) + new.board_confidence * t

    wires = new.wire_angles_deg
    if old.has_wires() and new.has_wires():
        wires = _blend_wire_angles(old.wire_angles_deg, new.wire_angles_deg, t)

    d_out = new.measured_double_outer_frac
    d_in = new.measured_double_inner_frac
    t_out = new.measured_triple_outer_frac
    t_in = new.measured_triple_inner_frac
    if old.has_measured_rings() and new.has_measured_rings():
        d_out = old.measured_double_outer_frac * (1.0 - t) + new.measured_double_outer_frac * t
        d_in = old.measured_double_inner_frac * (1.0 - t) + new.measured_double_inner_frac * t
        t_out = old.measured_triple_outer_frac * (1.0 - t) + new.measured_triple_outer_frac * t
        t_in = old.measured_triple_inner_frac * (1.0 - t) + new.measured_triple_inner_frac * t
    elif old.has_measured_rings() and not new.has_measured_rings():
        d_out = old.measured_double_outer_frac
        d_in = old.measured_double_inner_frac
        t_out = old.measured_triple_outer_frac
        t_in = old.measured_triple_inner_frac

    warp_size = new.warp_ring_size or old.warp_ring_size
    w_d_out = new.warp_double_outer or old.warp_double_outer
    w_d_in = new.warp_double_inner or old.warp_double_inner
    w_t_out = new.warp_triple_outer or old.warp_triple_outer
    w_t_in = new.warp_triple_inner or old.warp_triple_inner
    w1_d = new.warp1_double_outer or old.warp1_double_outer
    w1_t = new.warp1_triple_outer or old.warp1_triple_outer
    w1_c = new.warp1_center or old.warp1_center
    if old.has_warp_ring_ellipses() and new.has_warp_ring_ellipses():
        assert old.warp_double_outer and new.warp_double_outer
        assert old.warp_triple_outer and new.warp_triple_outer
        w_d_out = _blend_ellipse(old.warp_double_outer, new.warp_double_outer, t)
        w_t_out = _blend_ellipse(old.warp_triple_outer, new.warp_triple_outer, t)
        if old.warp_double_inner and new.warp_double_inner:
            w_d_in = _blend_ellipse(old.warp_double_inner, new.warp_double_inner, t)
        if old.warp_triple_inner and new.warp_triple_inner:
            w_t_in = _blend_ellipse(old.warp_triple_inner, new.warp_triple_inner, t)
        warp_size = new.warp_ring_size
    if old.warp1_double_outer is not None and new.warp1_double_outer is not None:
        w1_d = _blend_ellipse(old.warp1_double_outer, new.warp1_double_outer, t)
    if old.warp1_triple_outer is not None and new.warp1_triple_outer is not None:
        w1_t = _blend_ellipse(old.warp1_triple_outer, new.warp1_triple_outer, t)
    if old.warp1_center is not None and new.warp1_center is not None:
        w1_c = (
            old.warp1_center[0] * (1.0 - t) + new.warp1_center[0] * t,
            old.warp1_center[1] * (1.0 - t) + new.warp1_center[1] * t,
        )

    cal = BoardCalibration(
        center=center,
        bull_radius=bull_r,
        confidence=confidence,
        double_ellipse=ellipse,
        board_confidence=board_conf,
        wire_angles_deg=wires,
        homography_shift=0,
        segment20_offset=new.segment20_offset,
        measured_double_outer_frac=d_out,
        measured_double_inner_frac=d_in,
        measured_triple_outer_frac=t_out,
        measured_triple_inner_frac=t_in,
        warp_ring_size=int(warp_size),
        warp_double_outer=w_d_out,
        warp_double_inner=w_d_in,
        warp_triple_outer=w_t_out,
        warp_triple_inner=w_t_in,
        warp1_double_outer=w1_d,
        warp1_triple_outer=w1_t,
        warp1_center=w1_c,
        stage2_double_inner_ratio=(
            new.stage2_double_inner_ratio
            if new.stage2_double_inner_ratio > 0.5
            else old.stage2_double_inner_ratio
        ),
        stage2_triple_inner_ratio=(
            new.stage2_triple_inner_ratio
            if new.stage2_triple_inner_ratio > 0.5
            else old.stage2_triple_inner_ratio
        ),
    )
    refresh_segment20_fields(cal)
    return cal


def scale_board_calibration(cal: BoardCalibration, scale: float) -> BoardCalibration:
    """Skaliraj geometriju kalibracije (npr. nakon upscale detekcije natrag na capture rezoluciju)."""
    s = float(scale)
    if abs(s - 1.0) < 1e-6:
        return cal
    ellipse = cal.double_ellipse
    if ellipse is not None:
        (ecx, ecy), (ew, eh), ang = ellipse
        ellipse = ((ecx * s, ecy * s), (ew * s, eh * s), ang)
    out = replace(
        cal,
        center=(cal.center[0] * s, cal.center[1] * s),
        bull_radius=cal.bull_radius * s,
        double_ellipse=ellipse,
    )
    refresh_segment20_fields(out)
    return out


def calibrate_board(
    frame: np.ndarray,
    *,
    cam_idx: Optional[int] = None,
    segment20_offset: int = 0,
    bull_hint_xy: Optional[Tuple[float, float]] = None,
    prior_bull_xy: Optional[Tuple[float, float]] = None,
    ellipse_hint_xy: Optional[List[Tuple[float, float]]] = None,
) -> Optional[BoardCalibration]:
    if frame is None or frame.size == 0:
        return None

    h0, w0 = frame.shape[:2]
    min_dim0 = float(min(h0, w0))
    upscale = CALIBRATE_UPSCALE if min_dim0 < CALIBRATE_UPSCALE_IF_MIN_DIM_LT else 1
    work = frame
    if upscale > 1:
        work = cv2.resize(
            frame,
            (w0 * upscale, h0 * upscale),
            interpolation=cv2.INTER_CUBIC,
        )

    h, w = work.shape[:2]
    min_dim = float(min(h, w))
    hint_xy: Optional[Tuple[float, float]] = None
    if bull_hint_xy is not None:
        hint_xy = (float(bull_hint_xy[0]) * upscale, float(bull_hint_xy[1]) * upscale)
    prior_xy: Optional[Tuple[float, float]] = None
    if prior_bull_xy is not None and hint_xy is None:
        prior_xy = (float(prior_bull_xy[0]) * upscale, float(prior_bull_xy[1]) * upscale)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    red, green, black, _white = _board_color_masks(hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel)

    hint_pts_up: List[Tuple[float, float]] = []
    for p in ellipse_hint_xy or []:
        if p is None or len(p) < 2:
            continue
        hint_pts_up.append((float(p[0]) * upscale, float(p[1]) * upscale))
    hint_ell = _ellipse_from_hint_points(hint_pts_up, min_dim) if len(hint_pts_up) >= 3 else None

    # 1) Najveća R/G elipsa ploce (gdje god u kadru) → 2) bull SAMO unutar nje.
    # Ručne točke vanjske elipse: maskiraj surround (npr. crvena spužva).
    board_guess = hint_ell
    if board_guess is None:
        board_guess = _find_largest_rg_board_ellipse(red, green, min_dim)
    search_red, search_green = red, green
    if board_guess is not None:
        mask = _board_ellipse_mask((h, w), board_guess, shrink=1.08)
        search_red = cv2.bitwise_and(red, mask)
        search_green = cv2.bitwise_and(green, mask)
        if hint_ell is not None:
            red = search_red
            green = search_green

    def _search(
        sr: np.ndarray,
        sg: np.ndarray,
        ellipse: Optional[Ellipse],
        *,
        relaxed: bool = False,
        boost_xy: Optional[Tuple[float, float]] = None,
    ) -> Optional[Tuple[Tuple[float, float], float, float]]:
        return _find_bull(
            sr,
            sg,
            min_dim,
            search_ellipse=ellipse,
            hint_xy=boost_xy if boost_xy is not None else hint_xy,
            relaxed=relaxed,
        )

    bull_hit: Optional[Tuple[Tuple[float, float], float, float]] = None
    hint_path = ""

    # Optional manual seed: restrict bull search to ROI around click first.
    if hint_xy is not None:
        # Relaxed R/G only inside wide ROI — do NOT gate by possibly-wrong board_guess.
        red_rx, green_rx = _bull_color_masks_relaxed(hsv)
        red_rx = cv2.morphologyEx(red_rx, cv2.MORPH_OPEN, kernel)
        red_rx = cv2.morphologyEx(red_rx, cv2.MORPH_CLOSE, kernel)
        green_rx = cv2.morphologyEx(green_rx, cv2.MORPH_OPEN, kernel)
        green_rx = cv2.morphologyEx(green_rx, cv2.MORPH_CLOSE, kernel)

        for frac in (0.25, 0.35, 0.42):
            radius = max(36.0, min_dim * frac)
            roi = _hint_roi_mask((h, w), hint_xy, radius)
            # Unmasked full-frame colors in ROI first (board_guess can wipe true bull).
            for sr_src, sg_src, tag in (
                (red_rx, green_rx, "relaxed"),
                (red, green, "strict"),
            ):
                hr = cv2.bitwise_and(sr_src, roi)
                hg = cv2.bitwise_and(sg_src, roi)
                bull_hit = _search(hr, hg, None, relaxed=True)
                if bull_hit is not None:
                    hint_path = f"roi_{tag}_{frac:.2f}"
                    break
            if bull_hit is not None:
                break
            # Then AND with board_guess if available.
            if board_guess is not None:
                hr = cv2.bitwise_and(search_red, roi)
                hg = cv2.bitwise_and(search_green, roi)
                bull_hit = _search(hr, hg, board_guess, relaxed=True)
                if bull_hit is None:
                    bull_hit = _search(hr, hg, None, relaxed=True)
                if bull_hit is not None:
                    hint_path = f"roi_board_{frac:.2f}"
                    break

        # Full-frame with hint score bias (still relaxed).
        if bull_hit is None:
            bull_hit = _search(red_rx, green_rx, None, relaxed=True)
            if bull_hit is None:
                bull_hit = _search(search_red, search_green, board_guess, relaxed=True)
            if bull_hit is None and board_guess is not None:
                bull_hit = _search(search_red, search_green, None, relaxed=True)
            if bull_hit is not None:
                hint_path = "full_relaxed"

        # Board ellipse near click → seed bull at ellipse center / refine near hint.
        if bull_hit is None:
            near_ell = _find_largest_board_ellipse_near_hint(
                red_rx,
                green_rx,
                black,
                hint_xy,
                min_dim,
                max_center_dist=max(48.0, min_dim * 0.42),
            )
            if near_ell is None and board_guess is not None:
                if normalized_ellipse_radius(board_guess, hint_xy[0], hint_xy[1]) < 1.2:
                    near_ell = board_guess
            if near_ell is not None:
                (ecx, ecy), (ew, eh), _ = near_ell
                semi_min = 0.5 * min(float(ew), float(eh))
                # Prefer click when it is clearly the bull; else ellipse center.
                seed_cx, seed_cy = float(ecx), float(ecy)
                if math.hypot(hint_xy[0] - ecx, hint_xy[1] - ecy) < semi_min * 0.35:
                    seed_cx, seed_cy = hint_xy[0], hint_xy[1]
                seed_r = max(2.5, semi_min * (6.35 / 170.0))
                # Local refine with relaxed masks around seed.
                r_loc = max(24.0, min_dim * 0.22)
                roi = _hint_roi_mask((h, w), (seed_cx, seed_cy), r_loc)
                hr = cv2.bitwise_and(red_rx, roi)
                hg = cv2.bitwise_and(green_rx, roi)
                refined = _search(hr, hg, near_ell, relaxed=True)
                if refined is None:
                    refined = _search(hr, hg, None, relaxed=True)
                if refined is not None:
                    bull_hit = refined
                    hint_path = "ellipse_refine"
                else:
                    bull_hit = ((seed_cx, seed_cy), seed_r, 0.28)
                    hint_path = "ellipse_seed"
                    if board_guess is None:
                        board_guess = near_ell

        # Last resort: soft bull = click itself; double-ellipse / wires still run.
        if bull_hit is None:
            soft_r = max(3.0, min_dim * 0.018)
            if board_guess is not None:
                (_ecx, _ecy), (ew, eh), _ = board_guess
                soft_r = max(soft_r, 0.5 * min(float(ew), float(eh)) * (6.35 / 170.0))
            bull_hit = (hint_xy, soft_r, 0.22)
            hint_path = "click_seed"

        cam_tag = f"cam{cam_idx}" if cam_idx is not None else "cam?"
        print(
            f"[cal] {cam_tag} bull hint applied ({hint_xy[0] / upscale:.1f},"
            f"{hint_xy[1] / upscale:.1f}) -> {hint_path}",
            flush=True,
        )
    else:
        # Auto path: soft prior from last good cal (score bias, not forced click).
        boost = prior_xy
        bull_hit = _search(search_red, search_green, board_guess, boost_xy=boost)
        if bull_hit is None and board_guess is not None:
            # Fallback: malo širi search (cijela maska bez radius cut).
            bull_hit = _search(search_red, search_green, None, boost_xy=boost)

        # Ako strict padne ili je kandidat daleko od priora sa slabim scoreom —
        # ROI oko prijašnjeg bulla (bez click_seed).
        need_prior_roi = bull_hit is None
        if (
            not need_prior_roi
            and prior_xy is not None
            and bull_hit is not None
        ):
            dist = math.hypot(bull_hit[0][0] - prior_xy[0], bull_hit[0][1] - prior_xy[1])
            if dist > min_dim * CAL_BULL_REJECT_SHIFT_FRAC and float(bull_hit[2]) < 0.62:
                need_prior_roi = True

        if need_prior_roi and prior_xy is not None:
            red_rx, green_rx = _bull_color_masks_relaxed(hsv)
            red_rx = cv2.morphologyEx(red_rx, cv2.MORPH_OPEN, kernel)
            red_rx = cv2.morphologyEx(red_rx, cv2.MORPH_CLOSE, kernel)
            green_rx = cv2.morphologyEx(green_rx, cv2.MORPH_OPEN, kernel)
            green_rx = cv2.morphologyEx(green_rx, cv2.MORPH_CLOSE, kernel)
            prior_hit: Optional[Tuple[Tuple[float, float], float, float]] = None
            for frac in (0.16, 0.24, 0.34):
                radius = max(28.0, min_dim * frac)
                roi = _hint_roi_mask((h, w), prior_xy, radius)
                for sr_src, sg_src in ((red, green), (red_rx, green_rx)):
                    hr = cv2.bitwise_and(sr_src, roi)
                    hg = cv2.bitwise_and(sg_src, roi)
                    hit = _search(
                        hr, hg, board_guess, relaxed=True, boost_xy=prior_xy
                    )
                    if hit is None:
                        hit = _search(hr, hg, None, relaxed=True, boost_xy=prior_xy)
                    if hit is not None and (
                        prior_hit is None or float(hit[2]) > float(prior_hit[2])
                    ):
                        prior_hit = hit
                if prior_hit is not None and float(prior_hit[2]) >= 0.40:
                    break
            if prior_hit is not None:
                if bull_hit is None or float(prior_hit[2]) + 0.04 >= float(bull_hit[2]):
                    bull_hit = prior_hit
                    hint_path = "prior_roi"
                    cam_tag = f"cam{cam_idx}" if cam_idx is not None else "cam?"
                    print(
                        f"[cal] {cam_tag} prior bull ROI "
                        f"({prior_xy[0] / upscale:.1f},{prior_xy[1] / upscale:.1f})",
                        flush=True,
                    )
    if bull_hit is None:
        if hint_xy is not None:
            cam_tag = f"cam{cam_idx}" if cam_idx is not None else "cam?"
            print(f"[cal] {cam_tag} bull hint failed (no candidate)", flush=True)
        return None

    center, bull_r, bull_conf = bull_hit
    ellipse, board_conf = _detect_double_ellipse(
        red, green, center, bull_r, min_dim, extra_points=hint_pts_up or None
    )
    if ellipse is None and board_guess is not None:
        # Ako classic double-fit padne, koristi najveću RG elipsu (bez outside-scale).
        ellipse = board_guess
        board_conf = 0.55 if hint_ell is not None else 0.45
    # Hint: if double-fit still missing, try dark/RG ellipse near bull.
    if ellipse is None and hint_xy is not None:
        near_ell = _find_largest_board_ellipse_near_hint(
            red, green, black, center, min_dim, max_center_dist=max(48.0, min_dim * 0.45)
        )
        if near_ell is not None:
            ellipse = near_ell
            board_conf = 0.40
    # Prior path: same ellipse recovery near detected / prior bull.
    if ellipse is None and prior_xy is not None:
        near_ell = _find_largest_board_ellipse_near_hint(
            red, green, black, center, min_dim, max_center_dist=max(48.0, min_dim * 0.45)
        )
        if near_ell is not None:
            ellipse = near_ell
            board_conf = 0.40
    wire_angles: Tuple[float, ...] = ()
    if ellipse is not None:
        wire_angles = _detect_segment_boundaries(work, red, green, center, bull_r, ellipse, cam_idx=cam_idx)

    seg_offset = int(segment20_offset) % SEGMENT_COUNT
    cal = BoardCalibration(
        center=center,
        bull_radius=bull_r,
        confidence=bull_conf,
        double_ellipse=ellipse,
        board_confidence=board_conf,
        wire_angles_deg=wire_angles,
        homography_shift=0,
        segment20_offset=seg_offset,
    )
    refresh_segment20_fields(cal)
    if upscale > 1:
        cal = scale_board_calibration(cal, 1.0 / float(upscale))
    return cal


def _draw_segment_lines(
    out: np.ndarray,
    bull: Tuple[float, float],
    ellipse: Ellipse,
    wire_angles_deg: Tuple[float, ...],
    *,
    uniform_board_angles: bool = False,
    board_center: Optional[Tuple[float, float]] = None,
    board_radius: float = 0.0,
) -> None:
    cx, cy = int(round(bull[0])), int(round(bull[1]))
    if uniform_board_angles and board_center is not None:
        bcx, bcy = int(round(board_center[0])), int(round(board_center[1]))
        br = max(4, int(round(board_radius)))
        for k in range(SEGMENT_COUNT):
            ang = k * SEGMENT_STEP_DEG
            st, ct = _angle_sin_cos(ang)
            ex, ey = int(round(bcx + br * st)), int(round(bcy + br * ct))
            cv2.line(out, (bcx, bcy), (ex, ey), (0, 255, 255), 1, cv2.LINE_AA)
    else:
        for ang in wire_angles_deg:
            edge = _ray_to_ellipse_edge(bull, ang, ellipse)
            if edge is not None:
                cv2.line(out, (cx, cy), edge, (0, 255, 255), 1, cv2.LINE_AA)


def draw_board_overlay(frame: np.ndarray, cal: BoardCalibration) -> np.ndarray:
    """Klasični kalibracijski overlay: double elipsa, žice, bull točka, seg.20 boja."""
    out = frame.copy()
    _paint_board_wire_overlay(out, cal)
    return out


def _fill_camera_segment_wedges(
    img: np.ndarray,
    cal: BoardCalibration,
    color: Tuple[int, int, int],
) -> None:
    """Svi segmenti 50% plavo; trenutni segment 20 = 100% plavo. Bull čist."""
    if cal.double_ellipse is None or not cal.has_wires():
        return
    wires = cal.wire_angles_deg
    n = min(len(wires), SEGMENT_COUNT)
    if n < 8:
        return
    ellipse = cal.double_ellipse
    bull = cal.center
    before = img.copy()
    sorted_idx = sorted(range(n), key=lambda k: float(wires[k]) % 360.0)
    seg20_k0 = int(cal.segment20_wire_sector) if cal.segment20_wire_sector >= 0 else -1

    solid = np.zeros_like(img)
    soft = np.zeros_like(img)
    solid_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    soft_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cx, cy = int(round(bull[0])), int(round(bull[1]))

    for i in range(n):
        k0 = sorted_idx[i]
        k1 = sorted_idx[(i + 1) % n]
        e0 = _ray_to_ellipse_edge(bull, float(wires[k0]), ellipse)
        e1 = _ray_to_ellipse_edge(bull, float(wires[k1]), ellipse)
        if e0 is None or e1 is None:
            continue
        tri = np.array([[cx, cy], list(e0), list(e1)], dtype=np.int32)
        if k0 == seg20_k0:
            cv2.fillConvexPoly(solid, tri, color)
            cv2.fillConvexPoly(solid_mask, tri, 255)
        else:
            cv2.fillConvexPoly(soft, tri, color)
            cv2.fillConvexPoly(soft_mask, tri, 255)

    if np.any(soft_mask):
        a = float(_SEG_FILL_ALPHA)
        m = soft_mask > 0
        img[m] = (
            (1.0 - a) * img[m].astype(np.float32) + a * soft[m].astype(np.float32)
        ).astype(np.uint8)
    if np.any(solid_mask):
        a20 = float(_SEG20_FILL_ALPHA)
        m = solid_mask > 0
        img[m] = (
            (1.0 - a20) * img[m].astype(np.float32) + a20 * solid[m].astype(np.float32)
        ).astype(np.uint8)

    mean_semi = _mean_ellipse_semi(ellipse)
    rings = scoring_ring_radii(DEBUG_WARP_SIZE, cal=cal)
    d_ref = max(float(rings["double_outer"]), 1.0)
    r_bull = max(1.0, mean_semi * float(rings["bull_outer"]) / d_ref)
    _punch_bull_disk(img, before, bull[0], bull[1], r_bull)


def _sample_remap_xy(
    map_x: np.ndarray,
    map_y: np.ndarray,
    tx: float,
    ty: float,
) -> Optional[Tuple[float, float]]:
    h, w = map_x.shape[:2]
    if tx < 0.0 or ty < 0.0 or tx > w - 1.0 or ty > h - 1.0:
        return None
    x0 = int(math.floor(tx))
    y0 = int(math.floor(ty))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    fx = tx - x0
    fy = ty - y0
    sx = (
        map_x[y0, x0] * (1 - fx) * (1 - fy)
        + map_x[y0, x1] * fx * (1 - fy)
        + map_x[y1, x0] * (1 - fx) * fy
        + map_x[y1, x1] * fx * fy
    )
    sy = (
        map_y[y0, x0] * (1 - fx) * (1 - fy)
        + map_y[y0, x1] * fx * (1 - fy)
        + map_y[y1, x0] * (1 - fx) * fy
        + map_y[y1, x1] * fx * fy
    )
    if not (math.isfinite(sx) and math.isfinite(sy)) or sx < 0.0 or sy < 0.0:
        return None
    return (float(sx), float(sy))


def _map_td_point(
    map_x: np.ndarray, map_y: np.ndarray, tx: float, ty: float
) -> Optional[Tuple[int, int]]:
    sample = _sample_remap_xy(map_x, map_y, tx, ty)
    if sample is None:
        return None
    return (int(round(sample[0])), int(round(sample[1])))


def _map_td_polyline(
    map_x: np.ndarray,
    map_y: np.ndarray,
    points_td: List[Tuple[float, float]],
) -> Optional[np.ndarray]:
    pts: List[List[int]] = []
    for tx, ty in points_td:
        p = _map_td_point(map_x, map_y, tx, ty)
        if p is None:
            if len(pts) >= 2:
                break
            pts = []
            continue
        pts.append([p[0], p[1]])
    if len(pts) < 2:
        return None
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


def project_topdown_overlay_onto_frame(
    frame: np.ndarray,
    cal: BoardCalibration,
    *,
    out_size: int = DEBUG_WARP_SIZE,
) -> np.ndarray:
    """Projiciraj Stage2 topdown overlay (fill+žice+prsteni) natrag na kameru."""
    if frame is None or frame.size == 0 or not cal.is_valid():
        return frame
    fh, fw = frame.shape[:2]
    size = int(out_size) if out_size > 0 else DEBUG_WARP_SIZE
    maps = build_topdown_warp_maps(cal, out_size=size, frame_wh=(fw, fh))
    if maps is None:
        return draw_board_overlay(frame, cal)
    map_x, map_y, _mask = maps
    cx_out, _view_r, board_r = _warp_radii(size)
    rings = scoring_ring_radii(size, cal=cal)
    color = _WARP_OVERLAY_BGR
    out = frame.copy()
    before = out.copy()

    # Segment fillovi: mapiraj trokute topdown → kamera.
    soft = np.zeros_like(out)
    solid = np.zeros_like(out)
    soft_mask = np.zeros(out.shape[:2], dtype=np.uint8)
    solid_mask = np.zeros(out.shape[:2], dtype=np.uint8)
    for k in range(SEGMENT_COUNT):
        ang0 = (SEG20_LEFT_WIRE_DST_DEG + k * SEGMENT_STEP_DEG) % 360.0
        ang1 = (SEG20_LEFT_WIRE_DST_DEG + (k + 1) * SEGMENT_STEP_DEG) % 360.0
        st0, ct0 = _angle_sin_cos(ang0)
        st1, ct1 = _angle_sin_cos(ang1)
        pts_td = [
            (cx_out, cx_out),
            (cx_out + board_r * st0, cx_out + board_r * ct0),
            (cx_out + board_r * st1, cx_out + board_r * ct1),
        ]
        cam_pts: List[List[int]] = []
        ok = True
        for tx, ty in pts_td:
            p = _map_td_point(map_x, map_y, tx, ty)
            if p is None or not (0 <= p[0] < fw and 0 <= p[1] < fh):
                ok = False
                break
            cam_pts.append([p[0], p[1]])
        if not ok or len(cam_pts) < 3:
            continue
        tri = np.array(cam_pts, dtype=np.int32)
        if k == 0:
            cv2.fillConvexPoly(solid, tri, color)
            cv2.fillConvexPoly(solid_mask, tri, 255)
        else:
            cv2.fillConvexPoly(soft, tri, color)
            cv2.fillConvexPoly(soft_mask, tri, 255)
    if np.any(soft_mask):
        a = float(_SEG_FILL_ALPHA)
        m = soft_mask > 0
        out[m] = ((1.0 - a) * out[m].astype(np.float32) + a * soft[m].astype(np.float32)).astype(
            np.uint8
        )
    if np.any(solid_mask):
        a20 = float(_SEG20_FILL_ALPHA)
        m = solid_mask > 0
        out[m] = (
            (1.0 - a20) * out[m].astype(np.float32) + a20 * solid[m].astype(np.float32)
        ).astype(np.uint8)

    layer = np.zeros_like(out)
    for key in (
        "bull_inner",
        "bull_outer",
        "triple_inner",
        "triple_outer",
        "double_inner",
        "double_outer",
    ):
        r = float(rings[key])
        pts_td = [
            (cx_out + r * _angle_sin_cos(float(a))[0], cx_out + r * _angle_sin_cos(float(a))[1])
            for a in range(0, 360, 2)
        ]
        pts_td.append(pts_td[0])
        poly = _map_td_polyline(map_x, map_y, pts_td)
        if poly is not None:
            cv2.polylines(layer, [poly], False, color, 1, cv2.LINE_AA)

    r0 = float(rings["bull_outer"])
    r1 = float(rings["double_outer"])
    for k in range(SEGMENT_COUNT):
        ang = (SEG20_LEFT_WIRE_DST_DEG + k * SEGMENT_STEP_DEG) % 360.0
        st, ct = _angle_sin_cos(ang)
        pts_td = [
            (cx_out + (r0 + (r1 - r0) * float(t)) * st, cx_out + (r0 + (r1 - r0) * float(t)) * ct)
            for t in np.linspace(0.0, 1.0, 24)
        ]
        poly = _map_td_polyline(map_x, map_y, pts_td)
        if poly is not None:
            cv2.polylines(layer, [poly], False, color, 1, cv2.LINE_AA)

    line_m = layer.max(axis=2) > 0
    if np.any(line_m):
        out[line_m] = layer[line_m]

    # Bull rupa: vrati original unutar b25 (projicirano grubo oko centra).
    c_pt = _map_td_point(map_x, map_y, cx_out, cx_out)
    if c_pt is not None:
        br = max(2, int(round(float(rings["bull_outer"]) * 1.15)))
        _punch_bull_disk(out, before, float(c_pt[0]), float(c_pt[1]), float(br))
    return out


def draw_warp_overlay(
    warped: np.ndarray,
    bull_w: Tuple[float, float],
    bull_r_w: float,
    double_r_w: float,
    wire_angles_deg: Tuple[float, ...] = (),
    *,
    show_lines: bool = False,
) -> np.ndarray:
    """Overlay na warpiranoj slici: bull, krug (linije opcionalno)."""
    out = warped.copy()
    cx, cy = int(round(bull_w[0])), int(round(bull_w[1]))
    br = max(2, int(round(bull_r_w)))
    dr = max(4, int(round(double_r_w)))

    if show_lines:
        circle_ellipse: Ellipse = ((bull_w[0], bull_w[1]), (dr * 2.0, dr * 2.0), 0.0)
        if len(wire_angles_deg) >= SEGMENT_COUNT - 2:
            # Žice na topdownu: 351°, 9°, … (ne sredine segmenata 0°, 18°, …).
            uniform_wires = tuple(
                (SEG20_LEFT_WIRE_DST_DEG + k * SEGMENT_STEP_DEG) % 360.0
                for k in range(SEGMENT_COUNT)
            )
            _draw_wires_to_ellipse(
                out,
                bull_w,
                circle_ellipse,
                uniform_wires,
                line_color=(0, 255, 255),
                dot_color=(0, 255, 255),
                line_thickness=1,
                dot_radius=2,
            )
        else:
            for k in range(SEGMENT_COUNT):
                ang = (SEG20_LEFT_WIRE_DST_DEG + k * SEGMENT_STEP_DEG) % 360.0
                st, ct = _angle_sin_cos(ang)
                ex, ey = int(round(cx + dr * st)), int(round(cy + dr * ct))
                cv2.line(out, (cx, cy), (ex, ey), (0, 255, 255), 1, cv2.LINE_AA)

    cv2.circle(out, (cx, cy), dr, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), max(3, br // 2 + 2), (0, 255, 255), -1, cv2.LINE_AA)
    return out


def _homography_to_remap_maps(H: np.ndarray, out_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """dst (topdown) -> src (kamera) mape iz homografije."""
    inv = np.linalg.inv(H.astype(np.float64))
    yy, xx = np.mgrid[0:out_size, 0:out_size].astype(np.float64)
    ones = np.ones_like(xx)
    pts = np.stack([xx.ravel(), yy.ravel(), ones.ravel()], axis=0)
    src = inv @ pts
    w = np.maximum(src[2], 1e-9)
    map_x = (src[0] / w).reshape(out_size, out_size).astype(np.float32)
    map_y = (src[1] / w).reshape(out_size, out_size).astype(np.float32)
    return map_x, map_y


def _radial_stage1_maps(
    bull: Tuple[float, float],
    ellipse: Ellipse,
    out_size: int,
    wire_angles_deg: Tuple[float, ...] = (),
    segment20_offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """dst -> kamera mape za radijalni topdown + view mask."""
    bx, by = bull
    cx_out, view_r, board_r = _warp_radii(out_size)
    yy, xx = np.mgrid[0:out_size, 0:out_size].astype(np.float32)
    dx = xx - cx_out
    dy = yy - cx_out
    r_px = np.sqrt(dx * dx + dy * dy)
    theta_board = (np.degrees(np.arctan2(dx, -dy)) % 360.0).astype(np.float32)

    if len(wire_angles_deg) >= SEGMENT_COUNT - 2:
        clockwise_src, _n = _clockwise_wire_source_angles(wire_angles_deg)
        boundary_dists = _wire_boundary_distances(bull, ellipse, wire_angles_deg)
        theta_image = _interp_source_angle_seg20_topdown(
            theta_board, clockwise_src, segment20_offset
        )
        sin_t = np.sin(np.deg2rad(theta_image))
        cos_t = -np.cos(np.deg2rad(theta_image))
        edge_dist = _interp_boundary_distance(theta_board, boundary_dists)
    else:
        edge_table = np.array(
            [_ray_dist_to_ellipse(bull, float(a), ellipse) for a in range(360)],
            dtype=np.float32,
        )
        sin_t = np.sin(np.deg2rad(theta_board))
        cos_t = -np.cos(np.deg2rad(theta_board))
        edge_dist = np.interp(theta_board, np.arange(360, dtype=np.float32), edge_table)

    inside_board = r_px <= board_r
    beyond = np.maximum(r_px - board_r, 0.0)
    margin = max(view_r - board_r, 1.0)
    extrap = 1.0 + (beyond / margin) * WARP_OUTSIDE_EXTRAPOL
    scale = np.where(inside_board, r_px / max(board_r, 1.0), extrap)
    src_r = scale * edge_dist
    map_x = (bx + src_r * sin_t).astype(np.float32)
    map_y = (by + src_r * cos_t).astype(np.float32)
    mask = (r_px <= view_r).astype(np.uint8)
    return map_x, map_y, mask


def _compose_remap_maps(
    map1_x: np.ndarray,
    map1_y: np.ndarray,
    map2_x: np.ndarray,
    map2_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Spoji stage1 (dst1->cam) i stage2 (dst2->dst1) u jednu (dst2->cam)."""
    map_x = cv2.remap(
        map1_x, map2_x, map2_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0
    )
    map_y = cv2.remap(
        map1_y, map2_x, map2_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0
    )
    return map_x, map_y


def build_topdown_warp_maps(
    cal: BoardCalibration,
    *,
    out_size: int,
    frame_wh: Tuple[int, int],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Jednokratno izgradi spojene remap mape (kamera -> finalni krug)."""
    if not cal.is_valid() or cal.double_ellipse is None:
        return None
    size = int(out_size) if out_size > 0 else DEBUG_WARP_SIZE
    wires = cal.wire_angles_deg if cal.has_wires() else ()
    mask1: Optional[np.ndarray] = None

    if len(wires) >= SEGMENT_COUNT - 2:
        H = _homography_topdown_seg20(
            cal.center,
            cal.double_ellipse,
            wires,
            size,
            cal.segment20_offset,
        )
        if H is not None:
            map1_x, map1_y = _homography_to_remap_maps(H, size)
            cx_out, view_r, _ = _warp_radii(size)
            mask1 = np.zeros((size, size), dtype=np.uint8)
            cv2.circle(mask1, (int(cx_out), int(cx_out)), int(view_r), 255, -1)
        else:
            map1_x, map1_y, mask1 = _radial_stage1_maps(
                cal.center,
                cal.double_ellipse,
                size,
                wire_angles_deg=wires,
                segment20_offset=cal.segment20_offset,
            )
    else:
        map1_x, map1_y, mask1 = _radial_stage1_maps(
            cal.center,
            cal.double_ellipse,
            size,
            wire_angles_deg=wires,
            segment20_offset=cal.segment20_offset,
        )

    if cal.has_ring_align_warp() and cal.warp1_double_outer is not None:
        s = float(size) / float(max(1, cal.warp_ring_size))
        ell = (
            _resize_ellipse(cal.warp1_double_outer, s)
            if abs(s - 1.0) > 1e-6
            else cal.warp1_double_outer
        )
        _cx_out, _view_r, board_r = _warp_radii(size)
        # Bull (warp1_center) → sredina; double (+triple) → board_r krugovi.
        if cal.warp1_center is not None:
            center = (float(cal.warp1_center[0]) * s, float(cal.warp1_center[1]) * s)
        else:
            center = _ellipse_center(ell)
        triple = None
        if cal.warp1_triple_outer is not None:
            triple = (
                _resize_ellipse(cal.warp1_triple_outer, s)
                if abs(s - 1.0) > 1e-6
                else cal.warp1_triple_outer
            )
        if triple is not None:
            di_r = (
                float(cal.stage2_double_inner_ratio)
                if cal.stage2_double_inner_ratio > 0.5
                else None
            )
            ti_r = (
                float(cal.stage2_triple_inner_ratio)
                if cal.stage2_triple_inner_ratio > 0.5
                else None
            )
            map2_x, map2_y, mask2 = _multi_ring_radial_maps(
                ell,
                triple,
                center=center,
                board_r=board_r,
                out_size=size,
                double_inner_frac=di_r,
                triple_inner_frac=ti_r,
            )
        else:
            map2_x, map2_y, mask2 = _ellipse_to_circle_maps(
                ell, center=center, target_r=board_r, out_size=size
            )
        map_x, map_y = _compose_remap_maps(map1_x, map1_y, map2_x, map2_y)
        mask = cv2.bitwise_and(mask1, mask2) if mask1 is not None else mask2
    else:
        map_x, map_y = map1_x, map1_y
        mask = mask1 if mask1 is not None else np.ones((size, size), dtype=np.uint8)

    _ = frame_wh  # za cache ključ / buduću validaciju
    return map_x, map_y, mask


def apply_topdown_warp_maps(
    frame: np.ndarray,
    maps: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    map_x, map_y, mask = maps
    warped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return cv2.bitwise_and(warped, warped, mask=mask)


def warp_topdown_from_calibration(
    frame: np.ndarray,
    cal: BoardCalibration,
    *,
    out_size: int = DEBUG_WARP_SIZE,
) -> Optional[np.ndarray]:
    """Top-down: 1) kamera->seg.20  2) double elipsa->krug (ako je kalibrirano)."""
    size = int(out_size) if out_size > 0 else DEBUG_WARP_SIZE
    h, w = frame.shape[:2]
    maps = build_topdown_warp_maps(cal, out_size=size, frame_wh=(w, h))
    if maps is None:
        return None
    return apply_topdown_warp_maps(frame, maps)


def render_aligned_topdown(frame: np.ndarray, cal: BoardCalibration, out_size: int = 0) -> Optional[np.ndarray]:
    """Cisti aligned top-down warp (segment 20 gore), bez overlay linija."""
    size = out_size if out_size > 0 else DEBUG_WARP_SIZE
    return warp_topdown_from_calibration(frame, cal, out_size=size)


def render_warped_frame(frame: np.ndarray, cal: BoardCalibration, out_size: int = 0) -> Optional[np.ndarray]:
    if not cal.is_valid() or cal.double_ellipse is None or not cal.has_board_ellipse():
        return None
    h, w = frame.shape[:2]
    upscale = 2 if min(h, w) < 400 else 1

    if upscale > 1:
        frame2 = cv2.resize(frame, (w * upscale, h * upscale), interpolation=cv2.INTER_CUBIC)
        bull = (cal.center[0] * upscale, cal.center[1] * upscale)
        bull_r = cal.bull_radius * upscale
        ellipse = _scale_ellipse(cal.double_ellipse, float(upscale))
        # out_size = 0 -> koristi min(h,w) od većeg ulaza.
        out_size2 = 0
    else:
        frame2 = frame
        bull = cal.center
        bull_r = cal.bull_radius
        ellipse = cal.double_ellipse
        out_size2 = out_size if out_size > 0 else 0

    warped, bull_w, bull_r_w, double_r_w, _shift = warp_board_topdown(
        frame2,
        bull,
        bull_r,
        ellipse,
        out_size2,
        wire_angles_deg=cal.wire_angles_deg if cal.has_wires() else (),
        segment20_offset=cal.segment20_offset,
    )
    return draw_warp_overlay(
        warped,
        bull_w,
        bull_r_w,
        double_r_w,
        cal.wire_angles_deg if cal.has_wires() else (),
        show_lines=cal.has_wires(),
    )


# Razlozi zašto kamera nije "kompletna" nakon detect_all — NIJE isto što USB missing.
_DETECT_REASON_HR: Dict[str, str] = {
    "no_signal": "nema signala",
    "no_bull": "ploča/bull nije nađen",
    "no_ellipse": "elipsa nije nađena",
    "no_wires": "žice nisu nađene",
    "incomplete": "nepotpuno",
}


def _classify_incomplete_cal(
    cal: Optional[BoardCalibration],
    *,
    frame_missing: bool,
) -> str:
    if frame_missing:
        return "no_signal"
    if cal is None or not cal.is_valid():
        return "no_bull"
    if not cal.has_board_ellipse():
        return "no_ellipse"
    if not cal.has_wires():
        return "no_wires"
    return "incomplete"


class BoardCalibrator:
    def __init__(self) -> None:
        self._cals: Dict[int, Optional[BoardCalibration]] = {}
        self._seg20_offsets: Dict[int, int] = {}
        # (cam_idx, out_size, frame_w, frame_h) -> (map_x, map_y, mask)
        self._warp_maps: Dict[Tuple[int, int, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        # cam_idx -> no_signal | no_bull | no_ellipse | no_wires | incomplete
        self._last_detect_reasons: Dict[int, str] = {}
        self.show_overlay = True
        self.show_topdown = False
        self.load()

    def reset(self) -> None:
        self._cals.clear()
        self._seg20_offsets.clear()
        self._warp_maps.clear()
        self._last_detect_reasons.clear()

    def invalidate_warp_cache(self, cam_idx: Optional[int] = None) -> None:
        if cam_idx is None:
            self._warp_maps.clear()
            return
        dead = [k for k in self._warp_maps if k[0] == int(cam_idx)]
        for k in dead:
            self._warp_maps.pop(k, None)

    def warp_topdown(
        self,
        cam_idx: int,
        frame: np.ndarray,
        *,
        out_size: int = DEBUG_WARP_SIZE,
    ) -> Optional[np.ndarray]:
        """Brzi topdown: spojene remap mape (1 remap/frame), cache po kameri."""
        cal = self._cals.get(cam_idx)
        if cal is None or not cal.is_valid() or cal.double_ellipse is None or frame is None:
            return None
        size = int(out_size) if out_size > 0 else DEBUG_WARP_SIZE
        fh, fw = frame.shape[:2]
        key = (int(cam_idx), size, int(fw), int(fh))
        maps = self._warp_maps.get(key)
        if maps is None:
            maps = build_topdown_warp_maps(cal, out_size=size, frame_wh=(fw, fh))
            if maps is None:
                return None
            self._warp_maps[key] = maps
        return apply_topdown_warp_maps(frame, maps)

    def map_topdown_click_to_camera(
        self,
        cam_idx: int,
        dx: float,
        dy: float,
        *,
        display_size: Tuple[int, int],
        frame_size: Tuple[int, int],
    ) -> Optional[Tuple[float, float]]:
        """Map a click on the displayed topdown frame back to capture-frame coords."""
        cal = self._cals.get(int(cam_idx))
        if cal is None or not cal.is_valid() or cal.double_ellipse is None:
            return None
        dw = max(1, int(display_size[0]))
        dh = max(1, int(display_size[1]))
        fw = max(1, int(frame_size[0]))
        fh = max(1, int(frame_size[1]))
        size = int(DEBUG_WARP_SIZE)
        tx = float(dx) * (float(size) / float(dw))
        ty = float(dy) * (float(size) / float(dh))
        key = (int(cam_idx), size, int(fw), int(fh))
        maps = self._warp_maps.get(key)
        if maps is None:
            maps = build_topdown_warp_maps(cal, out_size=size, frame_wh=(fw, fh))
            if maps is not None:
                self._warp_maps[key] = maps
        if maps is None:
            return None
        map_x, map_y, _mask = maps
        sample = _sample_remap_xy(map_x, map_y, tx, ty)
        if sample is None:
            return None
        sx = max(0.0, min(float(fw - 1), float(sample[0])))
        sy = max(0.0, min(float(fh - 1), float(sample[1])))
        return (sx, sy)

    def map_camera_to_topdown_display(
        self,
        cam_idx: int,
        fx: float,
        fy: float,
        *,
        display_size: Tuple[int, int],
    ) -> Optional[Tuple[float, float]]:
        """Project a capture-frame point onto the displayed topdown image."""
        cal = self._cals.get(int(cam_idx))
        if cal is None or not cal.is_valid() or cal.double_ellipse is None:
            return None
        wires = cal.wire_angles_deg if cal.has_wires() else ()
        size = int(DEBUG_WARP_SIZE)
        H = None
        if len(wires) >= SEGMENT_COUNT - 2:
            H = _homography_topdown_seg20(
                cal.center,
                cal.double_ellipse,
                wires,
                size,
                cal.segment20_offset,
            )
        if H is None:
            return None
        pts = np.array([[[float(fx), float(fy)]]], dtype=np.float32)
        try:
            out = cv2.perspectiveTransform(pts, H)
        except cv2.error:
            return None
        tx, ty = float(out[0, 0, 0]), float(out[0, 0, 1])
        dw = max(1, int(display_size[0]))
        dh = max(1, int(display_size[1]))
        return (tx * (float(dw) / float(size)), ty * (float(dh) / float(size)))

    def _segment20_offset_for(self, cam_idx: int) -> int:
        old = self._cals.get(cam_idx)
        if old is not None:
            return int(old.segment20_offset) % SEGMENT_COUNT
        return int(self._seg20_offsets.get(cam_idx, 0)) % SEGMENT_COUNT

    def _remember_segment20(self, cam_idx: int, cal: BoardCalibration) -> None:
        self._seg20_offsets[cam_idx] = int(cal.segment20_offset) % SEGMENT_COUNT

    def get(self, cam_idx: int) -> Optional[BoardCalibration]:
        return self._cals.get(cam_idx)

    def set(self, cam_idx: int, cal: Optional[BoardCalibration]) -> None:
        if cal is None:
            self._cals.pop(cam_idx, None)
        else:
            self._cals[cam_idx] = cal
            self._remember_segment20(cam_idx, cal)
        self.invalidate_warp_cache(cam_idx)

    def detect_all(
        self,
        frames: Dict[int, Optional[np.ndarray]],
        *,
        target_size: Optional[Tuple[int, int]] = None,
        bull_hints: Optional[Dict[int, Tuple[float, float]]] = None,
        ellipse_hints: Optional[Dict[int, List[Tuple[float, float]]]] = None,
        prefer_stable: bool = False,
    ) -> List[int]:
        """Detektira ploču po kameri. Vraća indekse bez *kompletne* kalibracije.

        To NIJE lista USB/offline kamera — frame može postojati, a bull/elipsa/žice
        padaju. Koristi format_incomplete_status() / _last_detect_reasons za razlog.

        target_size=(width, height): ako su frameovi veći (npr. 640x480), skaliraj
        kalibraciju na radnu capture rezoluciju (npr. 320x240).

        bull_hints: opcioni per-cam (x,y) u frame koordinatama — seed za bull search.

        prefer_stable: True za auto-kalibraciju pri startu igre — soft prior iz
        prethodne kalibracije, blend, i ne briše dobar cal lošim/nepotpunim.
        False za ručni KALIBRIRAJ (force replace + incomplete overwrite).
        """
        hints = bull_hints or {}
        # Normalize keys (int / str / numpy) so cam0/cam2 hints are never missed.
        hints_norm: Dict[int, Tuple[float, float]] = {}
        for hk, hv in hints.items():
            try:
                ik = int(hk)
            except (TypeError, ValueError):
                continue
            if hv is None or len(hv) < 2:
                continue
            hints_norm[ik] = (float(hv[0]), float(hv[1]))
        ell_norm: Dict[int, List[Tuple[float, float]]] = {}
        for hk, hv in (ellipse_hints or {}).items():
            try:
                ik = int(hk)
            except (TypeError, ValueError):
                continue
            pts: List[Tuple[float, float]] = []
            if not isinstance(hv, (list, tuple)):
                continue
            for p in hv:
                if p is None or len(p) < 2:
                    continue
                pts.append((float(p[0]), float(p[1])))
            if pts:
                ell_norm[ik] = pts[:4]
        # Kamere čiji *ovaj* detect nije dao kompletnu kalibraciju (nakon politike).
        failed_this_run: List[int] = []
        for cam_idx, frame in frames.items():
            old = self._cals.get(cam_idx)
            offset = self._segment20_offset_for(cam_idx)
            cam_i = int(cam_idx)

            if frame is None:
                failed_this_run.append(cam_i)
                continue

            hint = hints_norm.get(cam_i)
            ell_hint = ell_norm.get(cam_i)
            prior: Optional[Tuple[float, float]] = None
            if hint is None and old is not None and old.is_valid():
                prior = (float(old.center[0]), float(old.center[1]))
            if hint is not None:
                print(
                    f"[cal] detect_all cam{cam_i}: using bull hint ({hint[0]:.1f},{hint[1]:.1f})",
                    flush=True,
                )
            elif hints:
                print(
                    f"[cal] detect_all cam{cam_i}: no bull hint for this cam "
                    f"(have {sorted(hints_norm.keys())})",
                    flush=True,
                )
            elif prior is not None and prefer_stable:
                print(
                    f"[cal] detect_all cam{cam_i}: soft prior bull "
                    f"({prior[0]:.1f},{prior[1]:.1f})",
                    flush=True,
                )
            if ell_hint:
                print(
                    f"[cal] detect_all cam{cam_i}: ellipse hints n={len(ell_hint)}",
                    flush=True,
                )
            new_cal = calibrate_board(
                frame,
                cam_idx=cam_i,
                segment20_offset=offset,
                bull_hint_xy=hint,
                prior_bull_xy=prior,
                ellipse_hint_xy=ell_hint,
            )
            if new_cal is None:
                if hint is not None:
                    print(
                        f"[cal] detect_all cam{cam_i}: hint present but calibrate_board returned None",
                        flush=True,
                    )
                if prefer_stable and _is_calibration_complete(old):
                    # Auto-start: zadrži zadnji dobar cal umjesto fail-a.
                    print(
                        f"[cal] detect_all cam{cam_i}: detect fail — keep previous complete cal",
                        flush=True,
                    )
                    self._remember_segment20(cam_idx, old)
                    continue
                failed_this_run.append(cam_i)
                continue

            diag = _frame_diag(frame)
            stable_shift = max(
                CAL_BULL_STABLE_SHIFT_PX, CAL_BULL_STABLE_SHIFT_FRAC * diag
            )
            reject_shift = max(stable_shift, CAL_BULL_REJECT_SHIFT_FRAC * diag)
            bull_shift = 0.0
            if old is not None and old.is_valid():
                bull_shift = _bull_shift_px(old, new_cal)

            if old is not None:
                if bull_shift <= stable_shift:
                    new_cal = _recover_calibration_geometry(old, new_cal)
                else:
                    print(
                        f"[cal] detect_all cam{cam_i}: bull shifted {bull_shift:.1f}px "
                        f"(>{stable_shift:.1f}) — skip geometry recover",
                        flush=True,
                    )
                    new_cal = replace(new_cal, segment20_offset=offset)
                    refresh_segment20_fields(new_cal)
            else:
                new_cal = replace(new_cal, segment20_offset=offset)
                refresh_segment20_fields(new_cal)

            # Enrich na frame rezoluciji (prije scale na capture), da 2. warp vidi ispravnu elipsu.
            if _is_calibration_complete(new_cal):
                # Stabilize jitter kad je nova blizu stare.
                if (
                    prefer_stable
                    and old is not None
                    and _is_calibration_complete(old)
                    and bull_shift <= stable_shift
                ):
                    new_cal = _stabilize_calibration(
                        old, new_cal, CALIBRATION_BLEND_ALPHA
                    )
                new_cal = enrich_calibration_with_warp_rings(frame, new_cal)
                # Jamstvo: Stage2 podaci uvijek postoje i spremaju se.
                if not new_cal.has_ring_align_warp():
                    new_cal = ensure_ring_align_warp(new_cal)
                _save_debug_calibration_pack(
                    cam_idx=cam_idx,
                    frame_bgr=frame,
                    cal=new_cal,
                )

            if target_size is not None:
                tw, th = int(target_size[0]), int(target_size[1])
                fh, fw = frame.shape[:2]
                if fw > 0 and fh > 0 and (fw != tw or fh != th):
                    scale = 0.5 * (float(tw) / float(fw) + float(th) / float(fh))
                    new_cal = scale_board_calibration(new_cal, scale)

            if _is_calibration_complete(new_cal):
                # Veliki skok bulla u auto-modu: primi samo ako je jasno bolji.
                if (
                    prefer_stable
                    and old is not None
                    and _is_calibration_complete(old)
                    and bull_shift > reject_shift
                    and _calibration_quality(new_cal)
                    < _calibration_quality(old) * 1.12
                ):
                    print(
                        f"[cal] detect_all cam{cam_i}: reject far jump "
                        f"({bull_shift:.1f}px) — keep previous cal "
                        f"(q_new={_calibration_quality(new_cal):.2f} "
                        f"q_old={_calibration_quality(old):.2f})",
                        flush=True,
                    )
                    self._remember_segment20(cam_idx, old)
                    continue
                self._cals[cam_idx] = new_cal
                self._remember_segment20(cam_idx, new_cal)
                refresh_segment20_fields(new_cal)
                self.invalidate_warp_cache(cam_idx)
            elif prefer_stable and _is_calibration_complete(old):
                # Auto-start: ne zamijeni kompletan cal nepotpunim.
                print(
                    f"[cal] detect_all cam{cam_i}: incomplete new — keep previous complete cal",
                    flush=True,
                )
                self._remember_segment20(cam_idx, old)
            else:
                failed_this_run.append(cam_i)
                # Settings / force: zamijeni stari complete nepotpunim da UI ne ostane
                # zaključan dok se ne klikne bull.
                if new_cal.is_valid():
                    self._cals[cam_idx] = new_cal
                    self._remember_segment20(cam_idx, new_cal)
                    self.invalidate_warp_cache(cam_idx)
                elif old is not None:
                    self._remember_segment20(cam_idx, old)

        incomplete = sorted(
            set(failed_this_run)
            | {
                int(cam_idx)
                for cam_idx in frames
                if not _is_calibration_complete(self._cals.get(cam_idx))
            }
        )
        self._last_detect_reasons = {
            int(cam_idx): _classify_incomplete_cal(
                self._cals.get(cam_idx),
                frame_missing=frames.get(cam_idx) is None,
            )
            for cam_idx in incomplete
        }
        self.save()
        return incomplete

    def recalibrate_for_play(
        self,
        grab_frames,
        *,
        attempts: int = 2,
        bull_hints: Optional[Dict[int, Tuple[float, float]]] = None,
        ellipse_hints: Optional[Dict[int, List[Tuple[float, float]]]] = None,
    ) -> Tuple[List[int], Dict[int, Optional[np.ndarray]]]:
        """Stabilna auto-kalibracija pri startu igre (bez KLIKNI BULL).

        Soft prior + prefer_stable; jedan kompletan prolaz dovoljan.
        Vraća (incomplete, last_frames).
        """
        hints = bull_hints or None
        last_frames: Dict[int, Optional[np.ndarray]] = {}
        incomplete: List[int] = []
        for attempt in range(max(1, int(attempts))):
            flush = 2 if attempt == 0 else 1
            try:
                last_frames = grab_frames(flush=flush)
            except TypeError:
                last_frames = grab_frames()
            incomplete = self.detect_all(
                last_frames,
                bull_hints=hints,
                ellipse_hints=ellipse_hints,
                prefer_stable=True,
            )
            if not incomplete:
                break
            if attempt + 1 >= attempts:
                break
            # Nema signala = kamere još nisu bile spremne — duže pričekaj.
            no_sig = self.incomplete_no_signal(incomplete)
            if no_sig and len(no_sig) >= len(incomplete):
                time.sleep(0.35)
            else:
                time.sleep(0.08)
        return incomplete, last_frames

    def format_incomplete_status(self, incomplete: Optional[List[int]] = None) -> str:
        """Čitljiv status: npr. 'cam0:nema signala, cam2:ploča/bull nije nađen'."""
        reasons = self._last_detect_reasons or {}
        cams = list(incomplete) if incomplete is not None else list(reasons.keys())
        if not cams:
            return ""
        parts: List[str] = []
        for cam_idx in sorted(int(c) for c in cams):
            code = reasons.get(cam_idx, "incomplete")
            label = _DETECT_REASON_HR.get(code, code)
            parts.append(f"cam{cam_idx}:{label}")
        return ", ".join(parts)

    def incomplete_no_signal(self, incomplete: Optional[List[int]] = None) -> List[int]:
        """Subset incomplete kamera koje stvarno nemaju kadar (USB/open/read)."""
        reasons = self._last_detect_reasons or {}
        cams = list(incomplete) if incomplete is not None else list(reasons.keys())
        return [int(c) for c in sorted(cams) if reasons.get(int(c)) == "no_signal"]

    def nudge_segment20(self, cam_idx: int, delta: int) -> None:
        """Pomak segmenta 20: delta -1 = lijevo, +1 = desno (clockwise po ploci)."""
        cal = self._cals.get(cam_idx)
        if cal is None or not cal.has_wires():
            return
        cal.segment20_offset = (cal.segment20_offset + int(delta)) % SEGMENT_COUNT
        refresh_segment20_fields(cal)
        self._remember_segment20(cam_idx, cal)
        self.invalidate_warp_cache(cam_idx)
        self.save()

    def capture_topdown_all(self, frames: Dict[int, Optional[np.ndarray]]) -> int:
        """Snimi rotirani top-down pregled svih kalibriranih kamera u debug_edges/."""
        saved = 0
        for cam_idx, frame in frames.items():
            if frame is None:
                continue
            cal = self._cals.get(cam_idx)
            if cal is None:
                continue
            if save_topdown_warp_from_calibration(cam_idx, frame, cal):
                saved += 1
        return saved

    def load(self, path: Optional[str] = None) -> None:
        fpath = path or calibration_file_path()
        if not os.path.isfile(fpath):
            return
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self._warp_maps.clear()
            dirty = False
            for entry in payload.get("cameras", []):
                cal = _calibration_from_dict(entry)
                if cal is not None:
                    refresh_segment20_fields(cal)
                    # Stari JSON bez warp_rings → dodaj nominalni Stage2 da double sjeda.
                    if _is_calibration_complete(cal) and not cal.has_ring_align_warp():
                        cal = ensure_ring_align_warp(cal)
                        dirty = True
                    self._cals[int(entry["cam_idx"])] = cal
                    self._remember_segment20(int(entry["cam_idx"]), cal)
            if dirty:
                self.save(fpath)
        except (OSError, json.JSONDecodeError, ValueError, KeyError):
            pass

    def save(self, path: Optional[str] = None) -> None:
        fpath = path or calibration_file_path()
        payload = {
            "cameras": [
                _calibration_to_dict(cam_idx, cal)
                for cam_idx, cal in sorted(self._cals.items())
                if cal is not None and cal.is_valid()
            ]
        }
        try:
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except OSError:
            pass

    def render_frames(self, frames: Dict[int, Optional[np.ndarray]]) -> Dict[int, Optional[np.ndarray]]:
        if self.show_topdown:
            return self.render_topdown_frames(frames)
        out: Dict[int, Optional[np.ndarray]] = {}
        for cam_idx, frame in frames.items():
            if frame is None:
                out[cam_idx] = None
                continue
            cal = self._cals.get(cam_idx)
            if cal is None or not cal.is_valid():
                out[cam_idx] = frame
                continue
            out[cam_idx] = draw_board_overlay(frame, cal)
        return out

    def render_topdown_frames(
        self, frames: Dict[int, Optional[np.ndarray]]
    ) -> Dict[int, Optional[np.ndarray]]:
        """Stage2 topdown warp + overlay (segment fill + prsteni/žice) za sve kamere."""
        out: Dict[int, Optional[np.ndarray]] = {}
        for cam_idx, frame in frames.items():
            if frame is None:
                out[cam_idx] = None
                continue
            cal = self._cals.get(cam_idx)
            if cal is None or not cal.is_valid():
                out[cam_idx] = frame
                continue
            warped = self.warp_topdown(int(cam_idx), frame, out_size=DEBUG_WARP_SIZE)
            if warped is None:
                out[cam_idx] = frame
                continue
            vis = warped.copy()
            _draw_segment_fills_on_aligned_warp(vis)
            _draw_scoring_rings_on_aligned_warp(vis, cal=cal, source_bgr=warped)
            out[cam_idx] = vis
        return out

    def status_summary(self) -> str:
        if self._last_detect_reasons:
            detail = self.format_incomplete_status()
            return f"Kalibracija nepotpuna — {detail}" if detail else "Kalibracija nepotpuna"
        n_ok = sum(1 for cal in self._cals.values() if _is_calibration_complete(cal))
        if n_ok <= 0 and not self._cals:
            return "Pritisni B za kalibraciju  |  strelice: pomak segmenta 20"
        if n_ok > 0:
            return f"Kompletno kalibrirano: {n_ok} kam."
        return ""
