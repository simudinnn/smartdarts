#!/usr/bin/env python3
"""Export TipCNN checkpoint → tip_detector.onnx."""

from __future__ import annotations

import argparse
import os
from typing import Any, List, Optional

from ml_tip.paths import default_checkpoint_path, default_onnx_path
from ml_tip.preprocess import INPUT_SIZE


def _export_onnx(model: Any, dummy: Any, output: str, opset: int) -> None:
    """Export with legacy TorchScript ONNX path (no onnxscript / dynamo)."""
    import torch

    kwargs = dict(
        input_names=["input"],
        output_names=["tip_xy"],
        dynamic_axes={"input": {0: "batch"}, "tip_xy": {0: "batch"}},
        opset_version=int(opset),
        do_constant_folding=True,
    )
    # Torch ≥2.x modern exporter defaults dynamo=True and needs onnxscript.
    try:
        torch.onnx.export(model, dummy, output, dynamo=False, **kwargs)
        return
    except TypeError:
        pass
    try:
        torch.onnx.export(model, dummy, output, **kwargs)
    except ModuleNotFoundError as exc:
        if "onnxscript" in str(exc):
            raise SystemExit(
                "ONNX export needs the legacy exporter (dynamo=False) or onnxscript.\n"
                "  Upgrade torch, or: pip install onnxscript\n"
                "  See ml_tip/requirements-ml.txt"
            ) from exc
        raise


def main(argv: Optional[List[str]] = None) -> int:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch required for export.\n"
            "  pip install -r ml_tip/requirements-ml.txt"
        ) from exc

    from ml_tip.model import build_model

    p = argparse.ArgumentParser(description="Export tip detector to ONNX")
    p.add_argument("--checkpoint", default=default_checkpoint_path())
    p.add_argument("--output", default=default_onnx_path())
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args(argv)

    model = build_model()
    if os.path.isfile(args.checkpoint):
        blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"] if isinstance(blob, dict) and "model" in blob else blob)
        print(f"[export] loaded checkpoint {args.checkpoint}", flush=True)
    else:
        print(
            f"[export] checkpoint not found: {args.checkpoint}\n"
            f"[export] exporting randomly initialized weights (smoke test only).\n"
            f"[export] WARNING: kiosk will reject collapsed/untrained ONNX and "
            f"keep FitLine — train a checkpoint before relying on ML tip.",
            flush=True,
        )
    model.eval()

    dummy = torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    _export_onnx(model, dummy, args.output, int(args.opset))
    print(f"[export] wrote {args.output}", flush=True)

    # Optional sanity check with onnxruntime if installed.
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"input": dummy.numpy()})[0]
        print(f"[export] onnxruntime smoke: shape={out.shape} tip={out.reshape(-1)[:2]}", flush=True)
    except Exception as exc:
        print(f"[export] onnxruntime check skipped: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
