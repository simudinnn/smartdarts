"""Bounded, delayed hit-debug writer.

The detection and UI threads only copy/enqueue data.  OpenCV encoding and disk
I/O start after a short delay, once the UI has had a chance to paint the hit.
"""
from __future__ import annotations

import copy
import os
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np


_JOBS: "queue.Queue[tuple]" = queue.Queue(maxsize=24)
_STARTED = False
_LOCK = threading.Lock()


def _start() -> None:
    global _STARTED
    with _LOCK:
        if _STARTED:
            return
        _STARTED = True
        threading.Thread(target=_worker, name="debug-image-writer", daemon=True).start()


def enqueue_hit_debug(fused, *, canvas=None, delay_s: float = 0.15) -> bool:
    """Queue one immutable hit snapshot; never wait for a full disk queue."""
    _start()
    snapshot = copy.copy(fused)
    snapshot.per_cam = []
    if canvas is not None:
        canvas = canvas.copy()
    else:
        for result in fused.per_cam:
            item = copy.copy(result)
            item.fit_gray = None if result.fit_gray is None else result.fit_gray.copy()
            item.motion_gray = None if result.motion_gray is None else result.motion_gray.copy()
            item.debug_bgr = None
            snapshot.per_cam.append(item)
    try:
        _JOBS.put_nowait(("hit", snapshot, canvas, time.monotonic() + max(0.0, delay_s)))
        return True
    except queue.Full:
        print("[DEBUG] red za slike je pun; preskacem samo debug sliku", flush=True)
        return False


def enqueue_single_motion(
    cam_idx: int,
    frame,
    motion,
    *,
    tip_xy=None,
    label: str = "",
    delay_s: float = 0.05,
) -> bool:
    """Save native single-camera input without blocking capture/detection."""
    if frame is None or motion is None:
        return False
    _start()
    try:
        _JOBS.put_nowait(
            (
                "single",
                int(cam_idx),
                frame.copy(),
                motion.copy(),
                None if tip_xy is None else (float(tip_xy[0]), float(tip_xy[1])),
                str(label or ""),
                time.monotonic() + max(0.0, delay_s),
            )
        )
        return True
    except queue.Full:
        print("[DEBUG] red za slike je pun; preskacem single debug", flush=True)
        return False


def _worker() -> None:
    while True:
        job = _JOBS.get()
        try:
            if job[0] == "single":
                _kind, cam_idx, frame, motion, tip_xy, label, not_before = job
                wait = not_before - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                from dart_detection import _debug_dir_path

                out_dir = os.path.join(_debug_dir_path(), "single")
                os.makedirs(out_dir, exist_ok=True)
                stamp = time.strftime("%H%M%S")
                ms = int((time.time() % 1.0) * 1000.0)
                stem = f"hit_{stamp}_{ms:03d}_cam{cam_idx}"
                marked = frame.copy()
                if tip_xy is not None:
                    tx, ty = int(round(tip_xy[0])), int(round(tip_xy[1]))
                    cv2.circle(marked, (tx, ty), 10, (0, 0, 255), 3, cv2.LINE_AA)
                    cv2.drawMarker(
                        marked, (tx, ty), (0, 255, 255),
                        markerType=cv2.MARKER_CROSS, markerSize=28,
                        thickness=2, line_type=cv2.LINE_AA,
                    )
                    if label:
                        cv2.putText(
                            marked, label,
                            (min(marked.shape[1] - 100, tx + 14), max(34, ty - 14)),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2,
                            cv2.LINE_AA,
                        )
                cv2.imwrite(os.path.join(out_dir, stem + "_frame.png"), marked)
                raw_motion = motion
                if raw_motion.ndim == 3:
                    raw_motion = cv2.cvtColor(raw_motion, cv2.COLOR_BGR2GRAY)
                visible = np.zeros_like(raw_motion, dtype=np.uint8)
                nonzero = raw_motion[raw_motion > 0]
                if nonzero.size:
                    lo = float(np.percentile(nonzero, 2.0))
                    hi = float(np.percentile(nonzero, 99.5))
                    if hi <= lo:
                        hi = lo + 1.0
                    visible = np.clip(
                        (raw_motion.astype(np.float32) - lo) * (255.0 / (hi - lo)),
                        0,
                        255,
                    ).astype(np.uint8)
                    visible[raw_motion <= 0] = 0
                cv2.imwrite(os.path.join(out_dir, stem + "_motion.png"), visible)
                print(f"[DEBUG SINGLE] spremljen {stem}", flush=True)
                continue
            _kind, fused, canvas, not_before = job
            wait = not_before - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            from dart_detection import _compose_cam_colored_motion, save_hit_fusion_debug

            if canvas is None:
                canvas = _compose_cam_colored_motion(fused.per_cam, 200)
            marked_xy: Optional[tuple] = None
            if getattr(fused, "tip_source", "fitline") == "ml":
                marked_xy = tuple(float(v) for v in fused.tip_xy)
            save_hit_fusion_debug(fused, {}, canvas=canvas, prediction_xy=marked_xy)
        except Exception as exc:
            print(f"[DEBUG] spremanje slike nije uspjelo: {exc}", flush=True)
        finally:
            _JOBS.task_done()
