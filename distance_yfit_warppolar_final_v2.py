#!/usr/bin/env python3
"""
SmartDarts standalone fast distance-Y estimator using cv2.warpPolar.

Input:
    Existing *_fused_motion_raw.png where:
      cam0 = red
      cam2 = green
      cam4 = cyan

The detector:
- does NOT use FitLine
- does NOT use ML
- works with 2 or 3 usable camera layers
- uses a distance-transform support image
- evaluates all ray angles at once with cv2.warpPolar
- searches only a few candidate shared points P, then fine-refines around the best

Usage:
    python distance_yfit_warppolar.py hit_xxx_fused_motion_raw.png --show
    python distance_yfit_warppolar.py hit_xxx_fused_motion_raw.png --out result.png
"""

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Tunables
# ============================================================

MIN_LAYER_PIXELS = 8

# Fused debug PNGs contain colored HUD/legend text in the top-left.
# That must NEVER be treated as dart motion.
IGNORE_HUD_X = 88
IGNORE_HUD_Y = 46

# The impact point itself must lie inside the actual scoring board.
# Motion/shaft may extend outside this circle, but P may not.
BOARD_CENTER = (149.5, 149.5)
MAX_IMPACT_RADIUS = 126.0

# Candidate P generation from joint distance map.
SEED_COUNT = 6
SEED_NMS_RADIUS = 10

# Reject degenerate shared-point seeds near frame edges / far from the actual
# meeting region of visible camera blobs.
P_LOCAL_RADIUS = 12
MIN_LOCAL_CAM_SUPPORT = 2       # at least this many camera layers near P
MAX_LOCAL_DIST_PX = 10.0       # nearest motion from each supporting layer
EDGE_MARGIN_PX = 8             # avoid trivial minima on image border

# Fine P refinement around the best coarse seed.
FINE_P_RADIUS = 3
FINE_P_STEP = 1

# Ray geometry / support.
RAY_LENGTH = 82
DIST_SIGMA = 3.0
COVERAGE_DISTANCE_PX = 3.5

# Angle resolution.
COARSE_ANGLE_BINS = 120   # 3 degrees/bin
FINE_ANGLE_BINS = 360     # 1 degree/bin

# Radial weighting: near-tip shaft matters more than distant flight.
NEAR_WEIGHT_END = 22
MID_WEIGHT_END = 52
NEAR_WEIGHT = 1.00
MID_WEIGHT = 0.80
FAR_WEIGHT = 0.40

# Per-angle score blend.
WEIGHT_SOFT_SUPPORT = 0.70
WEIGHT_COVERAGE = 0.30

# Cross-camera consensus.
CONSENSUS_MEAN_WEIGHT = 0.50
CONSENSUS_MIN_WEIGHT = 0.50

# Seed map: when 3 layers exist, the 2 closest dominate.
THIRD_LAYER_SEED_WEIGHT = 0.35


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
class YResult:
    p: tuple
    score: float
    ray_results: dict
    coarse_seed: tuple
    runtime_ms: float
    coarse_ms: float
    fine_ms: float
    usable_cams: list


def extract_layers(img_bgr):
    b, g, r = cv2.split(img_bgr)
    b = b.astype(np.int16)
    g = g.astype(np.int16)
    r = r.astype(np.int16)

    red = (
        (r >= 18)
        & (r >= g * 1.18 + 3)
        & (r >= b * 1.18 + 3)
    )

    green = (
        (g >= 18)
        & (g >= r * 1.15 + 3)
        & (g >= b * 1.08 + 1)
    )

    cyan = (
        (b >= 14)
        & (g >= 14)
        & (np.maximum(b, g) >= r * 1.22 + 3)
        & (np.abs(b - g) <= np.maximum(
            22, (0.60 * np.maximum(b, g)).astype(np.int16)
        ))
    )

    raw = [
        ("cam0", (0, 0, 255), red),
        ("cam2", (0, 255, 0), green),
        ("cam4", (255, 255, 0), cyan),
    ]

    layers = []
    for name, color, m in raw:
        mask = (m.astype(np.uint8) * 255)

        # IMPORTANT: fused debug images contain colored camera legend/HUD pixels.
        # Ignore that fixed overlay region so it cannot become a fake motion blob.
        mask[:IGNORE_HUD_Y, :IGNORE_HUD_X] = 0

        # Remove only tiny isolated noise. No dilation.
        nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        clean = np.zeros_like(mask)
        for i in range(1, nlabels):
            if stats[i, cv2.CC_STAT_AREA] >= 2:
                clean[labels == i] = 255

        pixels = int(np.count_nonzero(clean))
        if pixels < MIN_LAYER_PIXELS:
            continue

        # Distance to nearest motion pixel.
        inv = np.where(clean > 0, 0, 255).astype(np.uint8)
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5).astype(np.float32)

        # Precompute support maps ONCE per hit.
        support = np.exp(
            -(dist * dist) / (2.0 * DIST_SIGMA * DIST_SIGMA)
        ).astype(np.float32)

        coverage = (dist <= COVERAGE_DISTANCE_PX).astype(np.float32)

        layers.append(
            Layer(name, color, clean, dist, support, coverage, pixels)
        )

    return layers


def joint_proximity_map(layers):
    stack = np.stack([x.dist for x in layers], axis=0)

    if len(layers) == 2:
        return stack[0] + stack[1]

    s = np.sort(stack, axis=0)
    return s[0] + s[1] + THIRD_LAYER_SEED_WEIGHT * s[2]



def _nms_pick_minima(cost, count, nms_radius, edge_margin=EDGE_MARGIN_PX):
    work = cost.copy().astype(np.float32)
    h, w = work.shape

    # Hard-mask positions that cannot physically be a dart impact.
    yy0, xx0 = np.ogrid[:h, :w]
    cx, cy = BOARD_CENTER
    outside_board = (xx0 - cx) ** 2 + (yy0 - cy) ** 2 > MAX_IMPACT_RADIUS ** 2
    work[outside_board] = np.inf

    # Hard-mask borders to prevent degenerate edge minima.
    work[:edge_margin, :] = np.inf
    work[h-edge_margin:, :] = np.inf
    work[:, :edge_margin] = np.inf
    work[:, w-edge_margin:] = np.inf

    out = []
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


def choose_seeds(layers, count=SEED_COUNT, nms_radius=SEED_NMS_RADIUS):
    """
    Pair-aware shared-point seed generation.

    The previous version could choose a bogus frame-edge point in 2-camera cases.
    Here we explicitly generate candidates from every visible camera pair:
        cam0+cam2
        cam0+cam4
        cam2+cam4

    A real dart impact must be geometrically close to at least two camera-motion
    layers, even if the exact tip pixel itself is invisible.
    """
    if len(layers) < 2:
        return []

    candidates = []

    # Pair-specific minima are important when one camera layer is absent or weak.
    for i in range(len(layers)):
        for j in range(i + 1, len(layers)):
            li, lj = layers[i], layers[j]
            pair_cost = li.dist + lj.dist

            pair_seeds = _nms_pick_minima(
                pair_cost,
                count=max(3, count // 2),
                nms_radius=nms_radius,
            )

            for x, y, val in pair_seeds:
                candidates.append({
                    "x": x,
                    "y": y,
                    "pair_cost": float(val),
                    "pair": (li.name, lj.name),
                })

    # Also include minima from the all-layer proximity map for 3-camera consensus.
    prox = joint_proximity_map(layers)
    for x, y, val in _nms_pick_minima(prox, count=count, nms_radius=nms_radius):
        candidates.append({
            "x": x,
            "y": y,
            "pair_cost": float(val),
            "pair": ("joint", "joint"),
        })

    # Merge near-duplicate candidate points and retain the best cost.
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

    # Keep only candidates geometrically close to >=2 visible layers.
    valid = []
    for c in merged:
        p = (c["x"], c["y"])
        ok, cam_count, local_support = valid_p_seed(
            layers, p, layers[0].dist.shape
        )
        if not ok:
            continue

        # Prefer candidates with more camera support and smaller pair cost.
        score = (
            cam_count * 1000.0
            - c["pair_cost"]
            - sum(d for _, d in local_support)
        )
        c["seed_rank"] = float(score)
        c["local_support"] = local_support
        valid.append(c)

    valid.sort(key=lambda q: q["seed_rank"], reverse=True)
    valid = valid[:count]

    return [(c["x"], c["y"], c["pair_cost"]) for c in valid]


def radial_weights(length):
    r = np.arange(length, dtype=np.float32)

    w = np.where(
        r <= NEAR_WEIGHT_END,
        NEAR_WEIGHT,
        np.where(r <= MID_WEIGHT_END, MID_WEIGHT, FAR_WEIGHT)
    ).astype(np.float32)

    # Ignore P itself / first couple pixels because the actual tip may be absent.
    if len(w) > 2:
        w[:2] = 0.0

    return w


RADIAL_W = radial_weights(RAY_LENGTH)


def polar_angle_scores(src, p, angle_bins):
    """
    Evaluate all ray angles at once.

    warpPolar output:
      width  = radial position
      height = angle
    """
    polar = cv2.warpPolar(
        src,
        (RAY_LENGTH, angle_bins),
        (float(p[0]), float(p[1])),
        float(RAY_LENGTH),
        cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS,
    )

    # Weighted mean along each ray.
    denom = float(np.sum(RADIAL_W))
    if denom <= 1e-6:
        return np.zeros(angle_bins, dtype=np.float32)

    return (polar * RADIAL_W[None, :]).sum(axis=1) / denom


def best_ray_for_layer(layer, p, angle_bins):
    soft = polar_angle_scores(layer.support, p, angle_bins)
    cov = polar_angle_scores(layer.coverage, p, angle_bins)

    score = (
        WEIGHT_SOFT_SUPPORT * soft
        + WEIGHT_COVERAGE * cov
    )

    idx = int(np.argmax(score))

    # OpenCV warpPolar angle rows map [0, 360).
    angle_deg = (idx * 360.0) / angle_bins

    return RayResult(
        angle_deg=float(angle_deg),
        score=float(score[idx]),
        soft_support=float(soft[idx]),
        coverage=float(cov[idx]),
    )


def consensus_score(ray_results):
    scores = np.asarray(
        [r.score for r in ray_results.values()],
        dtype=np.float32
    )

    if len(scores) == 0:
        return -1.0
    if len(scores) == 1:
        return float(scores[0])

    return (
        CONSENSUS_MEAN_WEIGHT * float(np.mean(scores))
        + CONSENSUS_MIN_WEIGHT * float(np.min(scores))
    )


def evaluate_p(layers, p, angle_bins):
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



def valid_p_seed(layers, p, shape):
    """
    Cheap geometric guard for candidate impact point P.

    A valid P does NOT need motion exactly on the pixel (tip may be invisible),
    but at least MIN_LOCAL_CAM_SUPPORT camera layers must have motion reasonably
    near P. Also reject trivial frame-edge minima.
    """
    x, y = float(p[0]), float(p[1])
    h, w = shape[:2]

    if (
        x < EDGE_MARGIN_PX
        or y < EDGE_MARGIN_PX
        or x >= (w - EDGE_MARGIN_PX)
        or y >= (h - EDGE_MARGIN_PX)
    ):
        return False, 0, []

    # The dart impact must be on the scoring board.
    cx, cy = BOARD_CENTER
    if (x - cx) ** 2 + (y - cy) ** 2 > MAX_IMPACT_RADIUS ** 2:
        return False, 0, []

    xi = int(round(x))
    yi = int(round(y))

    supporting = []
    for layer in layers:
        # Distance-transform gives nearest motion distance directly.
        d = float(layer.dist[yi, xi])
        if d <= MAX_LOCAL_DIST_PX:
            supporting.append((layer.name, d))

    return len(supporting) >= MIN_LOCAL_CAM_SUPPORT, len(supporting), supporting


def run_yfit(img_bgr):
    t0 = time.perf_counter()

    layers = extract_layers(img_bgr)
    if len(layers) < 2:
        raise RuntimeError(
            f"Need >=2 usable layers, got "
            f"{[(x.name, x.pixels) for x in layers]}"
        )

    proximity = joint_proximity_map(layers)
    seeds = choose_seeds(layers)

    # --------------------------------------------
    # COARSE:
    # only evaluate a few proximity minima,
    # all angles at once with warpPolar.
    # --------------------------------------------
    t_coarse = time.perf_counter()

    best_score = -1.0
    best_p = None
    best_rr = None
    best_seed = None

    valid_seed_count = 0

    for sx, sy, _ in seeds:
        p = (sx, sy)

        ok_seed, local_cam_count, local_support = valid_p_seed(
            layers, p, img_bgr.shape
        )
        if not ok_seed:
            continue

        valid_seed_count += 1
        score, rr = evaluate_p(layers, p, COARSE_ANGLE_BINS)

        if score > best_score:
            best_score = score
            best_p = p
            best_rr = rr
            best_seed = p

    coarse_ms = (time.perf_counter() - t_coarse) * 1000.0

    if best_p is None and not seeds:
        # Emergency fallback: use the global minimum of every camera-pair distance sum.
        emergency = []
        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                cost = layers[i].dist + layers[j].dist
                cost = cost.copy()
                h, w = cost.shape
                cost[:EDGE_MARGIN_PX, :] = np.inf
                cost[h-EDGE_MARGIN_PX:, :] = np.inf
                cost[:, :EDGE_MARGIN_PX] = np.inf
                cost[:, w-EDGE_MARGIN_PX:] = np.inf

                idx = int(np.argmin(cost))
                yy, xx = np.unravel_index(idx, cost.shape)
                if np.isfinite(cost[yy, xx]):
                    emergency.append((float(xx), float(yy), float(cost[yy, xx])))

        seeds = emergency

        for sx, sy, _ in seeds:
            p = (sx, sy)
            ok_seed, _, _ = valid_p_seed(layers, p, img_bgr.shape)
            if not ok_seed:
                continue

            score, rr = evaluate_p(layers, p, COARSE_ANGLE_BINS)
            if score > best_score:
                best_score = score
                best_p = p
                best_rr = rr
                best_seed = p

    if best_p is None:
        # Fallback: scan the proximity minima more permissively, but still avoid edges.
        # This keeps 2-camera cases working when the actual tip is missing by several px.
        relaxed_max_dist = MAX_LOCAL_DIST_PX + 6.0

        for sx, sy, _ in seeds:
            p = (sx, sy)
            x, y = p
            h, w = img_bgr.shape[:2]

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
                float(layer.dist[yi, xi]) <= relaxed_max_dist
                for layer in layers
            )
            if near_count < MIN_LOCAL_CAM_SUPPORT:
                continue

            score, rr = evaluate_p(layers, p, COARSE_ANGLE_BINS)
            if score > best_score:
                best_score = score
                best_p = p
                best_rr = rr
                best_seed = p

    if best_p is None:
        raise RuntimeError(
            "No geometrically plausible coarse candidate: "
            "need >=2 camera motion layers near the impact region"
        )

    # --------------------------------------------
    # FINE:
    # only ±3 px around best coarse P.
    # Full 1° angular resolution.
    # --------------------------------------------
    t_fine = time.perf_counter()

    fine_best_score = best_score
    fine_best_p = best_p
    fine_best_rr = best_rr

    center_int = (
        int(round(best_p[0])),
        int(round(best_p[1])),
    )

    for p in fine_points(
        center_int,
        FINE_P_RADIUS,
        FINE_P_STEP,
        img_bgr.shape
    ):
        ok_p, _, _ = valid_p_seed(layers, p, img_bgr.shape)
        if not ok_p:
            continue

        score, rr = evaluate_p(layers, p, FINE_ANGLE_BINS)

        if score > fine_best_score:
            fine_best_score = score
            fine_best_p = p
            fine_best_rr = rr

    fine_ms = (time.perf_counter() - t_fine) * 1000.0
    # Final geometric plausibility guard.
    final_ok, final_cam_count, final_support = valid_p_seed(
        layers, fine_best_p, img_bgr.shape
    )
    if not final_ok:
        raise RuntimeError(
            f"Final Y point is not supported by >=2 camera layers: "
            f"P={fine_best_p} support={final_support}"
        )

    runtime_ms = (time.perf_counter() - t0) * 1000.0

    result = YResult(
        p=fine_best_p,
        score=fine_best_score,
        ray_results=fine_best_rr,
        coarse_seed=best_seed,
        runtime_ms=runtime_ms,
        coarse_ms=coarse_ms,
        fine_ms=fine_ms,
        usable_cams=[x.name for x in layers],
    )

    return result, layers, proximity, seeds


def draw_debug(img_bgr, result, layers, seeds):
    out = img_bgr.copy()
    x, y = result.p

    # Candidate seeds.
    for sx, sy, _ in seeds:
        cv2.circle(
            out,
            (int(round(sx)), int(round(sy))),
            2,
            (90, 90, 90),
            -1,
            cv2.LINE_AA,
        )

    color_map = {x.name: x.color_bgr for x in layers}

    # Rays.
    for name, rr in result.ray_results.items():
        a = math.radians(rr.angle_deg)

        x2 = x + RAY_LENGTH * math.cos(a)
        y2 = y + RAY_LENGTH * math.sin(a)

        color = color_map[name]

        cv2.line(
            out,
            (int(round(x)), int(round(y))),
            (int(round(x2)), int(round(y2))),
            color,
            2,
            cv2.LINE_AA,
        )

        tx = int(round(x2))
        ty = int(round(y2))
        cv2.putText(
            out,
            f"{name} {rr.score:.2f}",
            (max(2, tx - 25), max(12, ty)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

    # Final point.
    cv2.circle(
        out,
        (int(round(x)), int(round(y))),
        5,
        (0, 255, 255),
        -1,
        cv2.LINE_AA,
    )
    cv2.circle(
        out,
        (int(round(x)), int(round(y))),
        7,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    text1 = (
        f"P=({x:.1f},{y:.1f}) Y={result.score:.3f} "
        f"{result.runtime_ms:.1f}ms"
    )
    text2 = (
        f"coarse={result.coarse_ms:.1f} "
        f"fine={result.fine_ms:.1f} "
        f"cams={','.join(result.usable_cams)}"
    )

    cv2.putText(
        out, text1, (5, 17),
        cv2.FONT_HERSHEY_SIMPLEX, 0.40,
        (0, 255, 255), 1, cv2.LINE_AA
    )
    cv2.putText(
        out, text2, (5, 33),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
        (0, 255, 255), 1, cv2.LINE_AA
    )

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(str(args.input))
    if img is None:
        raise SystemExit(f"Cannot read {args.input}")

    result, layers, proximity, seeds = run_yfit(img)
    debug = draw_debug(img, result, layers, seeds)

    print(
        f"[DIST_YFIT_POLAR] "
        f"P=({result.p[0]:.2f},{result.p[1]:.2f}) "
        f"score={result.score:.4f} "
        f"runtime={result.runtime_ms:.1f}ms "
        f"coarse={result.coarse_ms:.1f}ms "
        f"fine={result.fine_ms:.1f}ms "
        f"usable={result.usable_cams} "
        f"seeds={len(seeds)}"
    )

    seed_ok, seed_cam_count, seed_support = valid_p_seed(
        layers, result.p, img.shape
    )
    print(
        f"[DIST_YFIT_POLAR] local_support ok={int(seed_ok)} "
        f"cams={seed_cam_count} details={seed_support}"
    )
    print(
        f"[DIST_YFIT_POLAR] guards hud=({IGNORE_HUD_X}x{IGNORE_HUD_Y}) "
        f"impact_center={BOARD_CENTER} max_r={MAX_IMPACT_RADIUS}"
    )

    for name, rr in result.ray_results.items():
        print(
            f"  {name}: angle={rr.angle_deg:.1f} "
            f"score={rr.score:.3f} "
            f"soft={rr.soft_support:.3f} "
            f"coverage={rr.coverage:.3f}"
        )

    out_path = args.out
    if out_path is None:
        out_path = args.input.with_name(
            args.input.stem + "_distance_yfit_polar.png"
        )

    cv2.imwrite(str(out_path), debug)
    print(f"[DIST_YFIT_POLAR] debug={out_path}")

    if args.show:
        cv2.imshow("distance_yfit_warppolar", debug)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
