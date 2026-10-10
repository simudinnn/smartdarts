"""Shared low-bandwidth settings for every SmartDarts camera capture."""

from __future__ import annotations

from typing import Optional

import cv2


import os
SINGLE_CAMERA = os.environ.get("SMARTDARTS_SINGLE_CAMERA","0") == "1"
CAPTURE_WIDTH = 1280 if SINGLE_CAMERA else 320
CAPTURE_HEIGHT = 720 if SINGLE_CAMERA else 240
CAPTURE_FPS = 15
def force_capture_settings(cap: cv2.VideoCapture) -> None:
    """Apply the bandwidth-sensitive settings in driver negotiation order."""
    # One 1280x720 stream must use MJPEG on the OV9732.  Without this V4L2 can
    # accept the dimensions while cap.read() keeps returning an empty frame.
    # Three-camera mode deliberately keeps the old conservative negotiation.
    use_single_profile = bool(SINGLE_CAMERA)
    width = 1280 if use_single_profile else CAPTURE_WIDTH
    height = 720 if use_single_profile else CAPTURE_HEIGHT
    for _ in range(2):
        if use_single_profile:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def open_limited_camera(index: int) -> Optional[cv2.VideoCapture]:
    """Open one UVC camera using the mode selected for this process."""
    use_single_profile = bool(SINGLE_CAMERA)
    width = 1280 if use_single_profile else CAPTURE_WIDTH
    height = 720 if use_single_profile else CAPTURE_HEIGHT
    params = [
        *(
            [cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")]
            if use_single_profile
            else []
        ),
        cv2.CAP_PROP_FRAME_WIDTH, width,
        cv2.CAP_PROP_FRAME_HEIGHT, height,
        cv2.CAP_PROP_FPS, CAPTURE_FPS,
    ]
    try:
        cap = cv2.VideoCapture(int(index), cv2.CAP_V4L2, params)
    except (TypeError, cv2.error):
        cap = cv2.VideoCapture(int(index), cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(int(index), cv2.CAP_V4L2)
        force_capture_settings(cap)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(int(index))
    if not cap.isOpened():
        cap.release()
        return None
    force_capture_settings(cap)
    return cap
