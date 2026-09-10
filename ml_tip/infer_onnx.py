"""Python ONNX Runtime tip detector (kiosk inference — no torch)."""

from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import numpy as np

from ml_tip.paths import default_onnx_path
from ml_tip.preprocess import INPUT_SIZE, WarpMapInfo, fused_motion_to_model_input, model_xy_to_board


class TipOnnxDetector:
    """Load tip_detector.onnx once; predict (x, y) in 200×200 pixel coords."""

    def __init__(self, model_path: Optional[str] = None, *, providers: Optional[list] = None) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for ML tip detection.\n"
                "  pip install onnxruntime\n"
                "Or: pip install -r ml_tip/requirements-ml.txt"
            ) from exc

        path = model_path or default_onnx_path()
        if not os.path.isfile(path):
            raise FileNotFoundError(f"ONNX model not found: {path}")

        avail = providers or ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(path, providers=avail)
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if not inputs or not outputs:
            raise RuntimeError(f"Invalid ONNX model (no I/O): {path}")
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self.model_path = path

    def predict_xy(self, gray_01: np.ndarray) -> Tuple[float, float]:
        """Predict tip in 200×200 pixel coordinates.

        gray_01: HxW or 1xHxW or 1x1xHxW float in [0, 1].
        """
        x = np.asarray(gray_01, dtype=np.float32)
        if x.ndim == 2:
            x = x[None, None, :, :]
        elif x.ndim == 3:
            x = x[None, :, :, :] if x.shape[0] == 1 else x[None, None, :, :]
        elif x.ndim != 4:
            raise ValueError(f"expected 2..4D input, got shape {x.shape}")
        if x.shape[-2] != INPUT_SIZE or x.shape[-1] != INPUT_SIZE:
            raise ValueError(f"expected *x{INPUT_SIZE}x{INPUT_SIZE}, got {x.shape}")

        out = self._session.run([self._output_name], {self._input_name: x})[0]
        arr = np.asarray(out, dtype=np.float32).reshape(-1)
        if arr.size < 2:
            raise RuntimeError(f"ONNX output too small: {out.shape}")
        px = float(np.clip(arr[0], 0.0, float(INPUT_SIZE - 1)))
        py = float(np.clip(arr[1], 0.0, float(INPUT_SIZE - 1)))
        return px, py

    def predict_on_fused(
        self,
        fused_u8: np.ndarray,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], WarpMapInfo]:
        """Full pipeline: fused u8 map → (board_xy, model_xy, WarpMapInfo)."""
        gray_01, info = fused_motion_to_model_input(fused_u8)
        mx, my = self.predict_xy(gray_01)
        bx, by = model_xy_to_board(mx, my, info)
        return (bx, by), (mx, my), info


_detector: Optional[TipOnnxDetector] = None
_detector_path: Optional[str] = None
_detector_error: Optional[str] = None


def get_tip_detector(
    model_path: Optional[str] = None,
    *,
    reload: bool = False,
) -> TipOnnxDetector:
    """Process-wide singleton. Raises if model / onnxruntime unavailable."""
    global _detector, _detector_path, _detector_error
    path = os.path.abspath(model_path or default_onnx_path())
    if reload or _detector is None or _detector_path != path:
        _detector = TipOnnxDetector(path)
        _detector_path = path
        _detector_error = None
    return _detector


def try_get_tip_detector(model_path: Optional[str] = None) -> Union[TipOnnxDetector, None]:
    """Return detector or None (prints once on failure)."""
    global _detector_error
    try:
        return get_tip_detector(model_path)
    except Exception as exc:
        msg = str(exc)
        if _detector_error != msg:
            print(f"[ml_tip] detector unavailable: {exc}", flush=True)
            _detector_error = msg
        return None
