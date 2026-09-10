"""Shared fused-motion → 200×200 preprocessing and board inverse map."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

INPUT_SIZE = 200


@dataclass(frozen=True)
class WarpMapInfo:
    """Source fused-map size before stretch-resize to INPUT_SIZE."""

    src_w: int
    src_h: int

    @property
    def scale_x(self) -> float:
        return float(self.src_w) / float(INPUT_SIZE)

    @property
    def scale_y(self) -> float:
        return float(self.src_h) / float(INPUT_SIZE)


def _as_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        return np.clip(img, 0, 255).astype(np.uint8)
    return img


def stretch_to_input_size(gray_u8: np.ndarray) -> Tuple[np.ndarray, WarpMapInfo]:
    """Stretch-resize grayscale map to INPUT_SIZE×INPUT_SIZE (uint8).

    Policy: full-frame stretch (no letterbox / no center-crop). Same scale for
    train, label, export, dataset dump, and kiosk inference.
    """
    g = _as_gray_u8(gray_u8)
    h, w = int(g.shape[0]), int(g.shape[1])
    info = WarpMapInfo(src_w=w, src_h=h)
    if h == INPUT_SIZE and w == INPUT_SIZE:
        return g, info
    interp = (
        cv2.INTER_AREA
        if (h > INPUT_SIZE or w > INPUT_SIZE)
        else cv2.INTER_LINEAR
    )
    out = cv2.resize(g, (INPUT_SIZE, INPUT_SIZE), interpolation=interp)
    return out, info


def fused_motion_to_model_input(fused_u8: np.ndarray) -> Tuple[np.ndarray, WarpMapInfo]:
    """Return float32 HxW in [0, 1] plus WarpMapInfo for inverse mapping."""
    u8, info = stretch_to_input_size(fused_u8)
    return (u8.astype(np.float32) * (1.0 / 255.0)), info


def uint8_model_image(fused_u8: np.ndarray) -> Tuple[np.ndarray, WarpMapInfo]:
    """Same preprocessing as inference, kept as uint8 for dataset PNG save."""
    return stretch_to_input_size(fused_u8)


def model_xy_to_board(
    x: float,
    y: float,
    info: WarpMapInfo,
) -> Tuple[float, float]:
    """Map 200×200 pixel tip → full fused / warp board coordinates."""
    return float(x) * info.scale_x, float(y) * info.scale_y


def board_xy_to_model(
    x: float,
    y: float,
    info: WarpMapInfo,
) -> Tuple[float, float]:
    """Map board warp coords → 200×200 pixel coords (for labeling aids)."""
    sx = info.scale_x if info.scale_x > 1e-9 else 1.0
    sy = info.scale_y if info.scale_y > 1e-9 else 1.0
    return float(x) / sx, float(y) / sy
