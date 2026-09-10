"""Tiny CNN tip regressor (Pi4-friendly, ONNX-exportable). No pretrained backbone."""

from __future__ import annotations

import math
from typing import Tuple

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ml_tip.model requires PyTorch. Install training extras:\n"
        "  pip install -r ml_tip/requirements-ml.txt"
    ) from exc

from ml_tip.preprocess import INPUT_SIZE


class AdaptiveAvgPool2dExportable(nn.Module):
    """AdaptiveAvgPool2d via slice+mean (legacy ONNX; 13→4 is not divisible).

    Matches ``F.adaptive_avg_pool2d`` numerically; no parameters (checkpoint-safe).
    """

    def __init__(self, output_size: Tuple[int, int]) -> None:
        super().__init__()
        self.output_size = (int(output_size[0]), int(output_size[1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        oh, ow = self.output_size
        # Always use slice+mean so legacy ONNX export does not hit
        # AdaptiveAvgPool2d's "output size not factor of input" limitation.
        # After TipCNN's 4× stride-2 convs on INPUT_SIZE=200, features are 13×13.
        H = W = 13
        rows = []
        for i in range(oh):
            i0 = int(math.floor(float(i * H) / oh))
            i1 = int(math.ceil(float((i + 1) * H) / oh))
            cols = []
            for j in range(ow):
                j0 = int(math.floor(float(j * W) / ow))
                j1 = int(math.ceil(float((j + 1) * W) / ow))
                cols.append(x[:, :, i0:i1, j0:j1].mean(dim=(2, 3), keepdim=True))
            rows.append(torch.cat(cols, dim=3))
        return torch.cat(rows, dim=2)


class TipCNN(nn.Module):
    """Very small grayscale tip regressor.

    Input:  (N, 1, 200, 200) float in [0, 1]
    Output: (N, 2) pixel coords (x, y) in the 200×200 image (≈ [0, 199])
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),  # 100
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 50
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 25
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # 13
            nn.ReLU(inplace=True),
            AdaptiveAvgPool2dExportable((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model() -> TipCNN:
    return TipCNN()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = build_model()
    x = torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE)
    y = m(x)
    print(f"params={count_parameters(m)} out={tuple(y.shape)} sample={y.detach().numpy()}")
