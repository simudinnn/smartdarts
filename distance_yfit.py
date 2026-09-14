#!/usr/bin/env python3
"""Production port of distance_yfit_warppolar_final_v2.py.

Algorithm (unchanged):
  per-cam usable mask → CC denoise (no dilation) → distanceTransform
  → Gaussian support + coverage → pair-aware P seeds → cv2.warpPolar rays
  → per-cam independent best angle → 0.5*mean + 0.5*min consensus
  → fine ±3 px / 360 angle bins.

Production input is per-camera warped motion (grayscale/soft), not fused RGB.
RGB extract_layers() is only for comparing against standalone debug PNGs.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ============================================================
# Tunables (copied from distance_yfit_warppolar_final_v2.py)
# ============================================================

MIN_LAYER_PIXELS = 8

# Fused debug PNGs contain colored HUD/legend text in the top-left.
# Production motion layers have no HUD — ignore_hud stays False there.
IGNORE_HUD_X = 88
IGNORE_HUD_Y = 46

BOARD_CENTER = (149.5, 149.5)
MAX_IMPACT_RADIUS = 126.0

SEED_COUNT = 6
SEED_NMS_RADIUS = 10

P_LOCAL_RADIUS = 12
MIN_LOCAL_CAM_SUPPORT = 2
MAX_LOCAL_DIST_PX = 10.0
EDGE_MARGIN_PX = 8

FINE_P_RADIUS = 3
FINE_P_STEP = 1

RAY_LENGTH = 82
DIST_SIGMA = 3.0
COVERAGE_DISTANCE_PX = 3.5

COARSE_ANGLE_BINS = 120
FINE_ANGLE_BINS = 360

NEAR_WEIGHT_END = 22
MID_WEIGHT_END = 52
NEAR_WEIGHT = 1.00
MID_WEIGHT = 0.80
FAR_WEIGHT = 0.40

WEIGHT_SOFT_SUPPORT = 0.70
WEIGHT_COVERAGE = 0.30

CONSENSUS_MEAN_WEIGHT = 0.50
CONSENSUS_MIN_WEIGHT = 0.50

THIRD_LAYER_SEED_WEIGHT = 0.35

CAM_NAMES = {0: "cam0", 2: "cam2", 4: "cam4"}
CAM_COLORS_BGR = {
    0: (0, 0, 255),
    2: (0, 255, 0),
    4: (255, 255, 0),
}


@dataclass
class Layer:
    name: str
    color_bgr: tuple
    mask: np.ndarray
    dist: np.ndarray
    support: np.ndarray
    coverage: np.ndarray
    pixels: int


@dataclass
class RayResult:
    angle_deg: float
    score: float
    soft_support: float
    coverage: float


@dataclass
class DistanceYFitEstimate:
    valid: bool
    x: float = 0.0
    y: float = 0.0
    score: float = 0.0
    usable_cams: List[int] = field(default_factory=list)
    per_cam_angle: Dict[int, float] = field(default_factory=dict)
    per_cam_score: Dict[int, float] = field(default_factory=dict)
    runtime_ms: float = 0.0
    mask_ms: float = 0.0
    distance_transform_ms: float = 0.0
    seed_ms: float = 0.0
    coarse_ms: float = 0.0
    fine_ms: float = 0.0
    reject_reason: str = ""
    coarse_seed: Optional[Tuple[float, float]] = None
    ray_results: Dict[str, RayResult] = field(default_factory=dict)

    @property
    def point(self) -> Optional[Tuple[float, float]]:
        if not self.valid:
            return None
        return (float(self.x), float(self.y))

    def as_dict(self) -> dict:
        return {
            "valid": bool(self.valid),
            "point": self.point,
            "score": float(self.score),
            "usable_cams": list(self.usable_cams),
            "per_cam_angle": dict(self.per_cam_angle),
            "per_cam_score": dict(self.per_cam_score),
            "runtime_ms": float(self.runtime_ms),
            "mask_ms": float(self.mask_ms),
            "distance_transform_ms": float(self.distance_transform_ms),
            "seed_ms": float(self.seed_ms),
            "coarse_ms": float(self.coarse_ms),
            "fine_ms": float(self.fine_ms),
            "reject_reason": str(self.reject_reason),
        }


def cam_id_from_name(name: str) -> int:
    return int(str(name).replace("cam", ""))


def _empty_estimate(**kwargs) -> DistanceYFitEstimate:
    est = DistanceYFitEstimate(valid=False)
    for k, v in kwargs.items():
        setattr(est, k, v)
    return est


def _clean_motion_mask(mask: np.ndarray, *, ignore_hud: bool) -> np.ndarray:
    """Binary 0/255 mask. Tiny isolated noise removed. No dilation."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8)
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    if ignore_hud:
        binary[:IGNORE_HUD_Y, :IGNORE_HUD_X] = 0
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if nlabels <= 1:
        return np.zeros_like(binary)
    keep = stats[:, cv2.CC_STAT_AREA] >= 2
    keep[0] = False
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def _finish_layer(name: str, color_bgr: tuple, clean: np.ndarray) -> Optional[Layer]:
    pixels = int(np.count_nonzero(clean))
    if pixels < MIN_LAYER_PIXELS:
        return None
    inv = np.where(clean > 0, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5).astype(np.float32)
    support = np.exp(-(dist * dist) / (2.0 * DIST_SIGMA * DIST_SIGMA)).astype(np.float32)
    coverage = (dist <= COVERAGE_DISTANCE_PX).astype(np.float32)
    return Layer(name, color_bgr, clean, dist, support, coverage, pixels)


def extract_layers(img_bgr: np.ndarray) -> List[Layer]:
    """Standalone fused-PNG parser (cam0 red / cam2 green / cam4 cyan). Not used in production."""
    b, g, r = cv2.split(img_bgr)
    b = b.astype(np.int16)
    g = g.astype(np.int16)
    r = r.astype(np.int16)

    red = (r >= 18) & (r >= g * 1.18 + 3) & (r >= b * 1.18 + 3)
    green = (g >= 18) & (g >= r * 1.15 + 3) & (g >= b * 1.08 + 1)
    cyan = (
        (b >= 14)
        & (g >= 14)
        & (np.maximum(b, g) >= r * 1.22 + 3)
        & (np.abs(b - g) <= np.maximum(22, (0.60 * np.maximum(b, g)).astype(np.int16)))
    )

    raw = [
        ("cam0", (0, 0, 255), red),
        ("cam2", (0, 255, 0), green),
        ("cam4", (255, 255, 0), cyan),
    ]
    layers: List[Layer] = []
    for name, color, m in raw:
        mask = (m.astype(np.uint8) * 255)
        clean = _clean_motion_mask(mask, ignore_hud=True)
        layer = _finish_layer(name, color, clean)
        if layer is not None:
            layers.append(layer)
    return layers


def layers_from_motion(
    cam_motion_layers: Dict[int, np.ndarray],
    *,
    ignore_hud: bool = False,
) -> Tuple[List[Layer], float, float]:
    """Build prototype Layer objects from per-camera warped grayscale/soft motion."""
    t_mask = 0.0
    t_dist = 0.0
    layers: List[Layer] = []
    for cam in sorted(int(c) for c in cam_motion_layers.keys()):
        src = cam_motion_layers.get(cam)
        if src is None or getattr(src, "size", 0) <= 0:
            continue
        t0 = time.perf_counter()
        if src.ndim == 3:
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        else:
            gray = src
        if gray.dtype != np.uint8:
            gray = np.clip(np.round(gray), 0, 255).astype(np.uint8)
        clean = _clean_motion_mask(gray, ignore_hud=ignore_hud)
        t_mask += (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        layer = _finish_layer(
            CAM_NAMES.get(int(cam), f"cam{int(cam)}"),
            CAM_COLORS_BGR.get(int(cam), (200, 200, 200)),
            clean,
        )
        t_dist += (time.perf_counter() - t1) * 1000.0
        if layer is not None:
            layers.append(layer)
    return layers, t_mask, t_dist


def joint_proximity_map(layers: Sequence[Layer]) -> np.ndarray:
    stack = np.stack([x.dist for x in layers], axis=0)
    if len(layers) == 2:
        return stack[0] + stack[1]
    s = np.sort(stack, axis=0)
    return s[0] + s[1] + THIRD_LAYER_SEED_WEIGHT * s[2]


def _nms_pick_minima(
    cost: np.ndarray,
    count: int,
    nms_radius: int,
    *,
    board_center: Tuple[float, float],
    max_impact_radius: float,
    edge_margin: int = EDGE_MARGIN_PX,
) -> List[Tuple[float, float, float]]:
    work = cost.copy().astype(np.float32)
    h, w = work.shape

    yy0, xx0 = np.ogrid[:h, :w]
    cx, cy = float(board_center[0]), float(board_center[1])
    outside_board = (xx0 - cx) ** 2 + (yy0 - cy) ** 2 > float(max_impact_radius) ** 2
    work[outside_board] = np.inf

    work[:edge_margin, :] = np.inf
    work[h - edge_margin :, :] = np.inf
    work[:, :edge_margin] = np.inf
    work[:, w - edge_margin :] = np.inf

    out: List[Tuple[float, float, float]] = []
    yy, xx = np.ogrid[:h, :w]
    for _ in range(count):
        idx = int(np.argmin(work))
        y, x = np.unravel_index(idx, work.shape)
        val = float(work[y, x])
        if not np.isfinite(val):
            break
        out.append((float(x), float(y), val))
        region = (xx - x) ** 2 + (yy - y) ** 2 <= nms_radius ** 2
        work[region] = np.inf
    return out


def valid_p_seed(
    layers: Sequence[Layer],
    p: Tuple[float, float],
    shape,
    *,
    board_center: Tuple[float, float] = BOARD_CENTER,
    max_impact_radius: float = MAX_IMPACT_RADIUS,
) -> Tuple[bool, int, list]:
    x, y = float(p[0]), float(p[1])
    h, w = shape[:2]
    if (
        x < EDGE_MARGIN_PX
        or y < EDGE_MARGIN_PX
        or x >= (w - EDGE_MARGIN_PX)
        or y >= (h - EDGE_MARGIN_PX)
    ):
        return False, 0, []
    cx, cy = float(board_center[0]), float(board_center[1])
    if (x - cx) ** 2 + (y - cy) ** 2 > float(max_impact_radius) ** 2:
        return False, 0, []
    xi = int(round(x))
    yi = int(round(y))
    supporting = []
    for layer in layers:
        d = float(layer.dist[yi, xi])
        if d <= MAX_LOCAL_DIST_PX:
            supporting.append((layer.name, d))
    return len(supporting) >= MIN_LOCAL_CAM_SUPPORT, len(supporting), supporting


def choose_seeds(
    layers: Sequence[Layer],
    count: int = SEED_COUNT,
    nms_radius: int = SEED_NMS_RADIUS,
    *,
    board_center: Tuple[float, float] = BOARD_CENTER,
    max_impact_radius: float = MAX_IMPACT_RADIUS,
) -> List[Tuple[float, float, float]]:
    if len(layers) < 2:
        return []
    candidates = []
    for i in range(len(layers)):
        for j in range(i + 1, len(layers)):
            li, lj = layers[i], layers[j]
            pair_cost = li.dist + lj.dist
            pair_seeds = _nms_pick_minima(
                pair_cost,
                count=max(3, count // 2),
                nms_radius=nms_radius,
                board_center=board_center,
                max_impact_radius=max_impact_radius,
            )
            for x, y, val in pair_seeds:
                candidates.append(
                    {
                        "x": x,
                        "y": y,
                        "pair_cost": float(val),
                        "pair": (li.name, lj.name),
                    }
                )

    prox = joint_proximity_map(layers)
    for x, y, val in _nms_pick_minima(
        prox,
        count=count,
        nms_radius=nms_radius,
        board_center=board_center,
        max_impact_radius=max_impact_radius,
    ):
        candidates.append(
            {
                "x": x,
                "y": y,
                "pair_cost": float(val),
                "pair": ("joint", "joint"),
            }
        )

    merged = []
    merge_r2 = 4.0 ** 2
    for c in sorted(candidates, key=lambda q: q["pair_cost"]):
        matched = False
        for m in merged:
            if (c["x"] - m["x"]) ** 2 + (c["y"] - m["y"]) ** 2 <= merge_r2:
                matched = True
                if c["pair_cost"] < m["pair_cost"]:
                    m.update(c)
                break
        if not matched:
            merged.append(c)

    valid = []
    shape = layers[0].dist.shape
    for c in merged:
        p = (c["x"], c["y"])
        ok, cam_count, local_support = valid_p_seed(
            layers, p, shape, board_center=board_center, max_impact_radius=max_impact_radius
        )
        if not ok:
            continue
        score = cam_count * 1000.0 - c["pair_cost"] - sum(d for _, d in local_support)
        c["seed_rank"] = float(score)
        c["local_support"] = local_support
        valid.append(c)

    valid.sort(key=lambda q: q["seed_rank"], reverse=True)
    valid = valid[:count]
    return [(c["x"], c["y"], c["pair_cost"]) for c in valid]


def radial_weights(length: int) -> np.ndarray:
    r = np.arange(length, dtype=np.float32)
    w = np.where(
        r <= NEAR_WEIGHT_END,
        NEAR_WEIGHT,
        np.where(r <= MID_WEIGHT_END, MID_WEIGHT, FAR_WEIGHT),
    ).astype(np.float32)
    if len(w) > 2:
        w[:2] = 0.0
    return w


RADIAL_W = radial_weights(RAY_LENGTH)


def polar_angle_scores(src: np.ndarray, p: Tuple[float, float], angle_bins: int) -> np.ndarray:
    polar = cv2.warpPolar(
        src,
        (RAY_LENGTH, angle_bins),
        (float(p[0]), float(p[1])),
        float(RAY_LENGTH),
        cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS,
    )
    denom = float(np.sum(RADIAL_W))
    if denom <= 1e-6:
        return np.zeros(angle_bins, dtype=np.float32)
    return (polar * RADIAL_W[None, :]).sum(axis=1) / denom


def best_ray_for_layer(layer: Layer, p: Tuple[float, float], angle_bins: int) -> RayResult:
    soft = polar_angle_scores(layer.support, p, angle_bins)
    cov = polar_angle_scores(layer.coverage, p, angle_bins)
    score = WEIGHT_SOFT_SUPPORT * soft + WEIGHT_COVERAGE * cov
    idx = int(np.argmax(score))
    angle_deg = (idx * 360.0) / angle_bins
    return RayResult(
        angle_deg=float(angle_deg),
        score=float(score[idx]),
        soft_support=float(soft[idx]),
        coverage=float(cov[idx]),
    )


def consensus_score(ray_results: Dict[str, RayResult]) -> float:
    scores = np.asarray([r.score for r in ray_results.values()], dtype=np.float32)
    if len(scores) == 0:
        return -1.0
    if len(scores) == 1:
        return float(scores[0])
    return CONSENSUS_MEAN_WEIGHT * float(np.mean(scores)) + CONSENSUS_MIN_WEIGHT * float(np.min(scores))


def evaluate_p(layers: Sequence[Layer], p: Tuple[float, float], angle_bins: int):
    rr = {}
    for layer in layers:
        rr[layer.name] = best_ray_for_layer(layer, p, angle_bins)
    return consensus_score(rr), rr


def fine_points(center, radius, step, shape):
    cx, cy = center
    h, w = shape[:2]
    pts = []
    for dy in range(-radius, radius + 1, step):
        for dx in range(-radius, radius + 1, step):
            x = cx + dx
            y = cy + dy
            if 0 <= x < w and 0 <= y < h:
                pts.append((float(x), float(y)))
    return pts


def _search_on_layers(
    layers: List[Layer],
    shape,
    *,
    board_center: Tuple[float, float],
    max_impact_radius: float,
) -> DistanceYFitEstimate:
    t0 = time.perf_counter()
    if len(layers) < 2:
        return _empty_estimate(
            reject_reason="need_2cam",
            usable_cams=[cam_id_from_name(x.name) for x in layers],
            runtime_ms=(time.perf_counter() - t0) * 1000.0,
        )

    t_seed = time.perf_counter()
    seeds = choose_seeds(
        layers, board_center=board_center, max_impact_radius=max_impact_radius
    )
    seed_ms = (time.perf_counter() - t_seed) * 1000.0

    t_coarse = time.perf_counter()
    best_score = -1.0
    best_p = None
    best_rr = None
    best_seed = None

    def _geom_kw():
        return dict(board_center=board_center, max_impact_radius=max_impact_radius)

    for sx, sy, _ in seeds:
        p = (sx, sy)
        ok_seed, _local_cam_count, _local_support = valid_p_seed(
            layers, p, shape, **_geom_kw()
        )
        if not ok_seed:
            continue
        score, rr = evaluate_p(layers, p, COARSE_ANGLE_BINS)
        if score > best_score:
            best_score = score
            best_p = p
            best_rr = rr
            best_seed = p
    coarse_ms = (time.perf_counter() - t_coarse) * 1000.0

    if best_p is None and not seeds:
        emergency = []
        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                cost = layers[i].dist + layers[j].dist
                cost = cost.copy()
                h, w = cost.shape
                cost[:EDGE_MARGIN_PX, :] = np.inf
                cost[h - EDGE_MARGIN_PX :, :] = np.inf
                cost[:, :EDGE_MARGIN_PX] = np.inf
                cost[:, w - EDGE_MARGIN_PX :] = np.inf
                idx = int(np.argmin(cost))
                yy, xx = np.unravel_index(idx, cost.shape)
                if np.isfinite(cost[yy, xx]):
                    emergency.append((float(xx), float(yy), float(cost[yy, xx])))
        seeds = emergency
        t_coarse2 = time.perf_counter()
        for sx, sy, _ in seeds:
            p = (sx, sy)
            ok_seed, _, _ = valid_p_seed(layers, p, shape, **_geom_kw())
            if not ok_seed:
                continue
            score, rr = evaluate_p(layers, p, COARSE_ANGLE_BINS)
            if score > best_score:
                best_score = score
                best_p = p
                best_rr = rr
                best_seed = p
        coarse_ms += (time.perf_counter() - t_coarse2) * 1000.0

    if best_p is None:
        relaxed_max_dist = MAX_LOCAL_DIST_PX + 6.0
        t_coarse3 = time.perf_counter()
        for sx, sy, _ in seeds:
            p = (sx, sy)
            x, y = p
            h, w = shape[:2]
            if (
                x < EDGE_MARGIN_PX
                or y < EDGE_MARGIN_PX
                or x >= (w - EDGE_MARGIN_PX)
                or y >= (h - EDGE_MARGIN_PX)
            ):
                continue
            xi = int(round(x))
            yi = int(round(y))
            near_count = sum(
                float(layer.dist[yi, xi]) <= relaxed_max_dist for layer in layers
            )
            if near_count < MIN_LOCAL_CAM_SUPPORT:
                continue
            score, rr = evaluate_p(layers, p, COARSE_ANGLE_BINS)
            if score > best_score:
                best_score = score
                best_p = p
                best_rr = rr
                best_seed = p
        coarse_ms += (time.perf_counter() - t_coarse3) * 1000.0

    if best_p is None:
        return _empty_estimate(
            reject_reason="no_seed",
            usable_cams=[cam_id_from_name(x.name) for x in layers],
            runtime_ms=(time.perf_counter() - t0) * 1000.0,
            seed_ms=seed_ms,
            coarse_ms=coarse_ms,
        )

    t_fine = time.perf_counter()
    fine_best_score = best_score
    fine_best_p = best_p
    fine_best_rr = best_rr
    center_int = (int(round(best_p[0])), int(round(best_p[1])))
    for p in fine_points(center_int, FINE_P_RADIUS, FINE_P_STEP, shape):
        ok_p, _, _ = valid_p_seed(layers, p, shape, **_geom_kw())
        if not ok_p:
            continue
        score, rr = evaluate_p(layers, p, FINE_ANGLE_BINS)
        if score > fine_best_score:
            fine_best_score = score
            fine_best_p = p
            fine_best_rr = rr
    fine_ms = (time.perf_counter() - t_fine) * 1000.0

    final_ok, _final_cam_count, _final_support = valid_p_seed(
        layers, fine_best_p, shape, **_geom_kw()
    )
    usable = [cam_id_from_name(x.name) for x in layers]
    if not final_ok:
        return _empty_estimate(
            reject_reason="unsupported_p",
            usable_cams=usable,
            runtime_ms=(time.perf_counter() - t0) * 1000.0,
            seed_ms=seed_ms,
            coarse_ms=coarse_ms,
            fine_ms=fine_ms,
            score=float(fine_best_score),
            x=float(fine_best_p[0]),
            y=float(fine_best_p[1]),
        )

    rr = fine_best_rr or {}
    return DistanceYFitEstimate(
        valid=True,
        x=float(fine_best_p[0]),
        y=float(fine_best_p[1]),
        score=float(fine_best_score),
        usable_cams=usable,
        per_cam_angle={cam_id_from_name(n): float(v.angle_deg) for n, v in rr.items()},
        per_cam_score={cam_id_from_name(n): float(v.score) for n, v in rr.items()},
        runtime_ms=(time.perf_counter() - t0) * 1000.0,
        seed_ms=seed_ms,
        coarse_ms=coarse_ms,
        fine_ms=fine_ms,
        coarse_seed=(float(best_seed[0]), float(best_seed[1])) if best_seed is not None else None,
        ray_results=rr,
    )


def estimate_dart_tip_distance_yfit(
    cam_motion_layers: Dict[int, np.ndarray],
    board_center: Tuple[float, float] = BOARD_CENTER,
    board_radius: float = MAX_IMPACT_RADIUS,
) -> DistanceYFitEstimate:
    """Find dart impact P from 2–3 warped motion layers. No FitLine, no ML.

    ``board_radius`` is the scoring-board cap for P (prototype MAX_IMPACT_RADIUS=126
    on the 300px canvas). Shafts may extend outside; P may not.
    """
    t0 = time.perf_counter()
    max_r = float(board_radius) if float(board_radius) > 1.0 else float(MAX_IMPACT_RADIUS)
    layers, mask_ms, dist_ms = layers_from_motion(cam_motion_layers, ignore_hud=False)
    if not layers:
        return _empty_estimate(
            reject_reason="need_2cam",
            runtime_ms=(time.perf_counter() - t0) * 1000.0,
            mask_ms=mask_ms,
            distance_transform_ms=dist_ms,
        )
    est = _search_on_layers(
        layers,
        layers[0].dist.shape,
        board_center=(float(board_center[0]), float(board_center[1])),
        max_impact_radius=max_r,
    )
    est.mask_ms = float(mask_ms)
    est.distance_transform_ms = float(dist_ms)
    est.runtime_ms = (time.perf_counter() - t0) * 1000.0
    if len(est.usable_cams) == 2 and est.valid:
        # Integration-only: 2-cam confidence is down-weighted by the caller.
        pass
    return est


def estimate_from_fused_debug_png(
    img_bgr: np.ndarray,
    *,
    board_center: Tuple[float, float] = BOARD_CENTER,
    max_impact_radius: float = MAX_IMPACT_RADIUS,
) -> DistanceYFitEstimate:
    """Same search as standalone final_v2 on a fused_motion_raw PNG (HUD ignored)."""
    t0 = time.perf_counter()
    t_mask = time.perf_counter()
    layers = extract_layers(img_bgr)
    mask_ms = (time.perf_counter() - t_mask) * 1000.0
    if len(layers) < 2:
        return _empty_estimate(
            reject_reason="need_2cam",
            usable_cams=[cam_id_from_name(x.name) for x in layers],
            runtime_ms=(time.perf_counter() - t0) * 1000.0,
            mask_ms=mask_ms,
        )
    est = _search_on_layers(
        layers,
        img_bgr.shape,
        board_center=board_center,
        max_impact_radius=max_impact_radius,
    )
    est.mask_ms = mask_ms
    est.runtime_ms = (time.perf_counter() - t0) * 1000.0
    return est


def ray_direction(angle_deg: float) -> Tuple[float, float]:
    a = math.radians(float(angle_deg))
    return float(math.cos(a)), float(math.sin(a))


def compare_with_standalone_module(img_bgr: np.ndarray, standalone_mod) -> Tuple[float, DistanceYFitEstimate, object]:
    """Return hypot(P_prod - P_standalone) for the same fused PNG."""
    prod = estimate_from_fused_debug_png(img_bgr)
    st, *_rest = standalone_mod.run_yfit(img_bgr)
    if not prod.valid:
        return 1.0e9, prod, st
    err = float(math.hypot(prod.x - float(st.p[0]), prod.y - float(st.p[1])))
    return err, prod, st
