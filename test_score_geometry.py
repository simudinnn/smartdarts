"""Deterministic scoring-geometry audit. No cameras / detection."""
from __future__ import annotations

import math
import os
import sys

import cv2

from board_calibration import (
    BOARD_SEGMENT_NUMBERS,
    RING_DOUBLE_INNER_FRAC,
    RING_DOUBLE_OUTER_FRAC,
    RING_TRIPLE_INNER_FRAC,
    RING_TRIPLE_OUTER_FRAC,
    SEG20_LEFT_WIRE_DST_DEG,
    SEGMENT_COUNT,
    SEGMENT_STEP_DEG,
    BoardCalibration,
    BoardCalibrator,
    calibrated_topdown_wire_thetas,
    canonical_topdown_wire_thetas,
    scoring_ring_ellipses,
    scoring_ring_radii,
    topdown_point_to_score,
    warp_topdown_from_calibration,
    _angle_sin_cos,
    _mean_ellipse_semi,
    _warp_radii,
)
from dart_detection import (
    FITLINE_SIZE,
    _board_center_and_r,
    draw_score_function_boundaries,
    score_topdown_point,
)


def _pt(cx: float, cy: float, ang_deg: float, radius: float) -> tuple[float, float]:
    st, ct = _angle_sin_cos(ang_deg)
    return cx + radius * st, cy + radius * ct


def _score_nom(px, py, center, board_r):
    rings = scoring_ring_radii(FITLINE_SIZE)
    return topdown_point_to_score(
        px, py, center, board_r, 0, rings=rings, log_geom=False
    )


def _theta(px: float, py: float, cx: float, cy: float) -> float:
    return math.degrees(math.atan2(float(px) - cx, -(float(py) - cy))) % 360.0


def _landmark_cal() -> BoardCalibration:
    return BoardCalibration(
        center=(160.0, 120.0),
        bull_radius=8.0,
        confidence=1.0,
        warp_mode="homography_landmarks",
        landmark_H=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        landmark_H_size=FITLINE_SIZE,
        landmark_H_offset=0,
        # Poisoned RG-measure leftovers must not move dest rings.
        measured_double_outer_frac=1.08,
        measured_double_inner_frac=0.80,
        measured_triple_outer_frac=0.50,
        measured_triple_inner_frac=0.40,
        warp_ring_size=FITLINE_SIZE,
    )


def _audit_homography_canonical_wires() -> int:
    """Landmark H dest already is the 18° grid — scoring must use that, not Stage1 rays."""
    fails = 0
    canon = canonical_topdown_wire_thetas()
    got = calibrated_topdown_wire_thetas(FITLINE_SIZE, cal=_landmark_cal())
    if got is None or len(got) != SEGMENT_COUNT:
        print("HOMOGRAPHY WIRES FAIL: missing canonical set")
        return 1
    for k, (a, b) in enumerate(zip(got, canon)):
        if abs(((float(a) - float(b) + 180.0) % 360.0) - 180.0) > 1e-6:
            print(f"HOMOGRAPHY WIRES FAIL k={k} got={a:.4f} expect={b:.4f}")
            fails += 1
    # Screenshot: FIT S12 / Y S9 on the 9/12 wire. Drifted virtual wire splits them;
    # dest-canonical wires keep both on the same side of 315°.
    center, board_r = _board_center_and_r(FITLINE_SIZE)
    cx, cy = center
    rings = scoring_ring_radii(FITLINE_SIZE)
    fit, ypt = (85.7, 84.5), (84.0, 83.0)
    drifted = list(canon)
    drifted[18] = (float(canon[18]) + 0.48) % 360.0
    n_fit_d, _, _ = topdown_point_to_score(
        *fit, center, board_r, 0, rings=rings, wire_thetas_deg=tuple(drifted), log_geom=False
    )
    n_y_d, _, _ = topdown_point_to_score(
        *ypt, center, board_r, 0, rings=rings, wire_thetas_deg=tuple(drifted), log_geom=False
    )
    n_fit, _, _ = topdown_point_to_score(
        *fit, center, board_r, 0, rings=rings, wire_thetas_deg=canon, log_geom=False
    )
    n_y, _, _ = topdown_point_to_score(
        *ypt, center, board_r, 0, rings=rings, wire_thetas_deg=canon, log_geom=False
    )
    print(
        f"wire-edge FIT theta={_theta(*fit, cx, cy):.3f} Y theta={_theta(*ypt, cx, cy):.3f} "
        f"canon_wire_9_12={canon[18]:.3f} drifted={n_fit_d}/{n_y_d} canonical={n_fit}/{n_y}"
    )
    if n_fit != n_y:
        print(f"HOMOGRAPHY WIRES FAIL canonical split FIT={n_fit} Y={n_y}")
        fails += 1
    if n_fit_d == n_y_d:
        print("HOMOGRAPHY WIRES NOTE drifted wires did not split FIT/Y (geometry check)")
    hcal = _landmark_cal()
    n_fit_h, _, _ = score_topdown_point(*fit, out_size=FITLINE_SIZE, cal=hcal, log_geom=False)
    n_y_h, _, _ = score_topdown_point(*ypt, out_size=FITLINE_SIZE, cal=hcal, log_geom=False)
    if n_fit_h != n_y_h:
        print(f"HOMOGRAPHY WIRES FAIL score_topdown_point split FIT={n_fit_h} Y={n_y_h}")
        fails += 1
    if fails == 0:
        print("homography canonical wires: OK")
    rings = scoring_ring_radii(FITLINE_SIZE, cal=_landmark_cal())
    _, _, board_r = _warp_radii(FITLINE_SIZE)
    std = scoring_ring_radii(FITLINE_SIZE)
    for key in ("double_outer", "double_inner", "triple_outer", "triple_inner"):
        if abs(float(rings[key]) - float(std[key])) > 1e-6:
            print(f"HOMOGRAPHY RINGS FAIL {key} got={rings[key]:.4f} expect={std[key]:.4f}")
            fails += 1
    if fails == 0:
        print("homography poisoned rings ignored: OK")
    fails += _audit_homography_measured_rings()
    return fails


def _landmark_cal_measured(triple_outer_shift_px: float) -> BoardCalibration:
    _, _, br = _warp_radii(FITLINE_SIZE)
    t_out = RING_TRIPLE_OUTER_FRAC + float(triple_outer_shift_px) / br
    return BoardCalibration(
        center=(160.0, 120.0),
        bull_radius=8.0,
        confidence=1.0,
        warp_mode="homography_landmarks",
        landmark_H=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        landmark_H_size=FITLINE_SIZE,
        landmark_H_offset=0,
        measured_double_outer_frac=float(RING_DOUBLE_OUTER_FRAC),
        measured_double_inner_frac=float(RING_DOUBLE_INNER_FRAC),
        measured_triple_outer_frac=float(t_out),
        measured_triple_inner_frac=float(RING_TRIPLE_INNER_FRAC),
        warp_ring_size=FITLINE_SIZE,
    )


def _audit_homography_measured_rings() -> int:
    """Sane warp residual moves S/T; overlay ellipses match scoring radii."""
    fails = 0
    _, _, br = _warp_radii(FITLINE_SIZE)
    shift = -2.0
    cal = _landmark_cal_measured(shift)
    rings = scoring_ring_radii(FITLINE_SIZE, cal=cal)
    expect_t = br * RING_TRIPLE_OUTER_FRAC + shift
    if abs(float(rings["triple_outer"]) - expect_t) > 1e-6:
        print(
            f"HOMOGRAPHY MEASURED RINGS FAIL triple_outer "
            f"got={rings['triple_outer']:.4f} expect={expect_t:.4f}"
        )
        fails += 1
    ell = scoring_ring_ellipses(FITLINE_SIZE, cal=cal)
    if ell is None:
        print("HOMOGRAPHY MEASURED RINGS FAIL missing ellipses")
        return fails + 1
    semi = _mean_ellipse_semi(ell["triple_outer"])
    if abs(semi - float(rings["triple_outer"])) > 1e-6:
        print(
            f"HOMOGRAPHY MEASURED RINGS FAIL overlay!=score "
            f"ellipse={semi:.4f} radii={rings['triple_outer']:.4f}"
        )
        fails += 1
    center, _board_r = _board_center_and_r(FITLINE_SIZE)
    cx, cy = center
    std = scoring_ring_radii(FITLINE_SIZE)
    on_nominal_t = _pt(cx, cy, 0.0, float(std["triple_outer"]))
    _n_m, z_m, _s_m = score_topdown_point(
        *on_nominal_t, out_size=FITLINE_SIZE, cal=cal, log_geom=False
    )
    _n_n, z_n, _s_n = score_topdown_point(
        *on_nominal_t, out_size=FITLINE_SIZE, log_geom=False
    )
    if z_n != "triple":
        print(f"HOMOGRAPHY MEASURED RINGS FAIL nominal edge zone={z_n}")
        fails += 1
    if z_m != "single":
        print(
            f"HOMOGRAPHY MEASURED RINGS FAIL 2px-inward T should make "
            f"canonical T-outer a single, got {z_m}"
        )
        fails += 1
    inside = _pt(cx, cy, 0.0, float(rings["triple_outer"]) - 0.8)
    _ni, z_in, _si = score_topdown_point(
        *inside, out_size=FITLINE_SIZE, cal=cal, log_geom=False
    )
    if z_in != "triple":
        print(f"HOMOGRAPHY MEASURED RINGS FAIL inside measured T zone={z_in}")
        fails += 1
    if fails == 0:
        print("homography measured rings: OK")
    return fails


def main() -> int:
    h_fail = _audit_homography_canonical_wires()
    center, board_r = _board_center_and_r(FITLINE_SIZE)
    cx, cy = center
    print("FITLINE_SIZE", FITLINE_SIZE)
    print("center", center, "expected", ((FITLINE_SIZE - 1) * 0.5,) * 2)
    print("board_r", board_r, "warp", _warp_radii(FITLINE_SIZE))
    r_safe = float(board_r) * 0.50

    cal = BoardCalibrator()
    cals = [cal.get(0), cal.get(2), cal.get(4)]
    wires = calibrated_topdown_wire_thetas(FITLINE_SIZE, cals=cals)
    print("calibrated_wires", None if wires is None else [round(float(w), 3) for w in wires])

    print("--- reported debug points ---")
    for name, px, py in (("FIT", 89.5, 182.0), ("Y", 81.0, 183.0)):
        dx, dy = px - cx, py - cy
        raw = math.atan2(dx, -dy)
        th = math.degrees(raw) % 360.0
        n, z, s = score_topdown_point(px, py, out_size=FITLINE_SIZE, cals=cals, log_geom=False)
        n2, z2, s2 = _score_nom(px, py, center, board_r)
        print(
            f"{name} point=({px},{py}) dx={dx:.3f} dy={dy:.3f} radius={math.hypot(dx, dy):.3f} "
            f"raw_angle={raw:.6f} normalized_angle={th:.3f} "
            f"calibrated={n}/{z}/{s} nominal={n2}/{z2}/{s2}"
        )

    print("--- cardinals ---")
    card_fail = 0
    for label, ang, expect in (("12h", 0.0, 20), ("3h", 90.0, 6), ("6h", 180.0, 3), ("9h", 270.0, 11)):
        px, py = _pt(cx, cy, ang, r_safe)
        n, z, s = score_topdown_point(px, py, out_size=FITLINE_SIZE, cals=cals, log_geom=False)
        n2, _, _ = _score_nom(px, py, center, board_r)
        ok = n == expect and n2 == expect
        card_fail += int(not ok)
        print(f"{label} expect={expect} cal={n} nom={n2} {'OK' if ok else 'FAIL'} pt=({px:.2f},{py:.2f})")

    print("--- segment centers ---")
    cen_fail = []
    for k, num in enumerate(BOARD_SEGMENT_NUMBERS):
        ang = k * SEGMENT_STEP_DEG
        px, py = _pt(cx, cy, ang, r_safe)
        n, _, _ = score_topdown_point(px, py, out_size=FITLINE_SIZE, cals=cals, log_geom=False)
        n2, _, _ = _score_nom(px, py, center, board_r)
        ok = n == num and n2 == num
        if not ok:
            cen_fail.append((k, num, n, n2, ang))
        print(f"k={k:02d} ang={ang:6.1f} expect={num:2d} cal={n:2d} nom={n2:2d} {'OK' if ok else 'FAIL'}")

    print("--- wire epsilon ---")
    eps = 0.35
    wfail = []
    for k in range(SEGMENT_COUNT):
        w = (
            float(wires[k])
            if wires is not None
            else (SEG20_LEFT_WIRE_DST_DEG + k * SEGMENT_STEP_DEG) % 360.0
        )
        left_seg = BOARD_SEGMENT_NUMBERS[(k - 1) % SEGMENT_COUNT]
        right_seg = BOARD_SEGMENT_NUMBERS[k]
        for side, d, expect in (("L", -eps, left_seg), ("R", +eps, right_seg)):
            ang = (w + d) % 360.0
            px, py = _pt(cx, cy, ang, r_safe)
            n, _, _ = score_topdown_point(px, py, out_size=FITLINE_SIZE, cals=cals, log_geom=False)
            if n != expect:
                wfail.append((k, side, w, ang, expect, n))
                print(f"WIRE FAIL W{k} {side} wire={w:.3f} ang={ang:.3f} expect={expect} got={n}")
    if not wfail:
        print("wire epsilon: all OK")

    print(
        f"SUMMARY card_fail={card_fail} center_fail={len(cen_fail)} wire_fail={len(wfail)}"
    )

    # Visual: actual score_topdown_point wires on the same 300x300 topdown.

    frame = None
    for p in (
        os.path.join("fake_cams", "cam_0.png"),
        os.path.join("dart_motion_refs", "empty_raw_cam_0.png"),
    ):
        if os.path.isfile(p):
            frame = cv2.imread(p)
            if frame is not None:
                break
    c0 = cal.get(0)
    if frame is not None and c0 is not None:
        warped = warp_topdown_from_calibration(frame, c0, out_size=FITLINE_SIZE)
        if warped is not None:
            draw_score_function_boundaries(
                warped, out_size=FITLINE_SIZE, cals=cals, thickness=1, label=True
            )
            out_dir = os.path.join("debug_edges", "calibration")
            os.makedirs(out_dir, exist_ok=True)
            out_p = os.path.join(out_dir, "scoring_boundaries.png")
            cv2.imwrite(out_p, warped)
            print("wrote", out_p)

    return 0 if (h_fail == 0 and card_fail == 0 and not cen_fail and not wfail) else 1


if __name__ == "__main__":
    sys.exit(main())
