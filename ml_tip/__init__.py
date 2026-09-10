"""Experimental ML tip detector (ONNX) — alternative to fitLine tip (x,y) only.

Coordinate convention
---------------------
- Model input: 1×200×200 grayscale, pixels in [0, 1] (uint8 / 255).
- Model output / labels.csv: tip (x, y) in **pixel coordinates** of the 200×200
  image, with origin at top-left: x ∈ [0, 199], y ∈ [0, 199].
- Warp mapping: full fused motion map is **stretch-resized** to 200×200
  (cv2.INTER_AREA when downscaling, INTER_LINEAR when upscaling). Inverse:
      board_x = pred_x * (src_w / 200)
      board_y = pred_y * (src_h / 200)
  When DEBUG_WARP_SIZE == 200 this is identity.

Runtime inference uses onnxruntime only (no torch). Training/export need torch.
"""

from __future__ import annotations

from ml_tip.preprocess import (
    INPUT_SIZE,
    WarpMapInfo,
    fused_motion_to_model_input,
    model_xy_to_board,
    uint8_model_image,
)

__all__ = [
    "INPUT_SIZE",
    "WarpMapInfo",
    "fused_motion_to_model_input",
    "model_xy_to_board",
    "uint8_model_image",
]
