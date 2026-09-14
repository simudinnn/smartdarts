"""Replay BoardCalibrator on saved KALIBRIRAJ frames (development only).

Collect on the Pi (same 3 frames + hints the live button used):

    CAL_SNAPSHOT_ENABLED = True  in board_calibration.py
    press KALIBRIRAJ

Then:

    python replay_calibration_snapshot.py
    python replay_calibration_snapshot.py --write-golden
    python replay_calibration_snapshot.py --compare

Does not change production calibration. Future speedups must pass --compare
against debug_calibration/golden_output.json on this frozen input.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import board_calibration as bc


def _snapshot_dir(path: str) -> str:
    return os.path.abspath(path)


def _load_hints(meta_path: str) -> Tuple[dict, dict, List[int]]:
    if not os.path.isfile(meta_path):
        return {}, {}, [0, 2, 4]
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    bull: Dict[int, Tuple[float, float]] = {}
    for k, v in (meta.get("bull_hints") or {}).items():
        if v is None or len(v) < 2:
            continue
        bull[int(k)] = (float(v[0]), float(v[1]))
    ell: Dict[int, List[Tuple[float, float]]] = {}
    for k, v in (meta.get("ellipse_hints") or {}).items():
        pts = []
        for p in v or []:
            if p is None or len(p) < 2:
                continue
            pts.append((float(p[0]), float(p[1])))
        if pts:
            ell[int(k)] = pts
    cameras = [int(c) for c in (meta.get("cameras") or [0, 2, 4])]
    return bull, ell, cameras


def _load_frames(d: str, cameras: List[int]) -> Dict[int, Optional[np.ndarray]]:
    frames: Dict[int, Optional[np.ndarray]] = {}
    for cam in cameras:
        path = os.path.join(d, f"input_cam{int(cam)}.png")
        if not os.path.isfile(path):
            print(f"[replay] missing {path}", flush=True)
            frames[int(cam)] = None
            continue
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            print(f"[replay] failed to read {path}", flush=True)
            frames[int(cam)] = None
        else:
            frames[int(cam)] = img
    return frames


def _payload_from_cals(
    cals: Dict[int, Optional[bc.BoardCalibration]], cameras: List[int]
) -> dict:
    return {
        "cameras": {
            str(int(cam)): bc.calibration_golden_entry(int(cam), cals.get(cam))
            for cam in cameras
        }
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_calibration"),
        help="Directory with input_cam*.png and input_hints.json",
    )
    parser.add_argument(
        "--write-golden",
        action="store_true",
        help="Write debug_calibration/golden_output.json from this replay",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare replay geometry to existing golden_output.json",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Numeric tolerance for --compare (default 1e-3)",
    )
    args = parser.parse_args(argv)

    d = _snapshot_dir(args.dir)
    hints_path = os.path.join(d, "input_hints.json")
    bull_hints, ellipse_hints, cameras = _load_hints(hints_path)
    if not cameras:
        cameras = [0, 2, 4]
    frames = _load_frames(d, cameras)
    n_ok = sum(1 for f in frames.values() if f is not None)
    if n_ok == 0:
        print(f"[replay] no input_cam*.png in {d}", flush=True)
        return 1

    print(
        f"[replay] dir={d} cams={cameras} frames={n_ok} "
        f"bull_hints={sorted(bull_hints)} ellipse_hints={sorted(ellipse_hints)}",
        flush=True,
    )

    expected_path = os.path.join(d, "golden_output.json")
    expected = None
    if args.compare:
        if not os.path.isfile(expected_path):
            print(f"[replay] --compare but missing {expected_path}", flush=True)
            return 1
        with open(expected_path, encoding="utf-8") as fh:
            expected = json.load(fh)

    calibrator = bc.BoardCalibrator()
    calibrator.reset()
    incomplete = calibrator.detect_all(
        frames,
        bull_hints=bull_hints or None,
        ellipse_hints=ellipse_hints or None,
        prefer_stable=False,
        persist=False,
    )
    payload = _payload_from_cals(calibrator._cals, cameras)
    replay_path = os.path.join(d, "replay_output.json")
    with open(replay_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[replay] incomplete={incomplete} wrote {replay_path}", flush=True)

    if args.write_golden:
        with open(expected_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[replay] wrote golden {expected_path}", flush=True)

    if expected is not None:
        mismatches = bc.compare_golden_calibration(payload, expected, atol=float(args.atol))
        if mismatches:
            print(f"[replay] GOLDEN FAIL ({len(mismatches)} mismatches):", flush=True)
            for line in mismatches[:80]:
                print(f"  {line}", flush=True)
            if len(mismatches) > 80:
                print(f"  ... {len(mismatches) - 80} more", flush=True)
            return 2
        print("[replay] GOLDEN PASS", flush=True)

    all_complete = n_ok > 0 and not incomplete
    for cam, frame in frames.items():
        if frame is None:
            continue
        if not bc._is_calibration_complete(calibrator.get(cam)):
            all_complete = False
            print(f"[replay] cam{cam} incomplete", flush=True)
    return 0 if all_complete else 2


if __name__ == "__main__":
    sys.exit(main())
