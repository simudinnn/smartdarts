#!/usr/bin/env python3
"""Train TipCNN on ml_tip/dataset (images/ + labels.csv)."""

from __future__ import annotations

import argparse
import os
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ml_tip.paths import (
    default_checkpoint_path,
    default_dataset_dir,
    default_images_dir,
    default_labels_path,
    load_labels,
)
from ml_tip.preprocess import INPUT_SIZE


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise SystemExit(
            "PyTorch required for training.\n"
            "  pip install -r ml_tip/requirements-ml.txt"
        ) from exc
    return torch, nn, DataLoader, Dataset


torch, nn, DataLoader, Dataset = _require_torch()

from ml_tip.model import TipCNN, build_model, count_parameters  # noqa: E402


class TipDataset(Dataset):
    def __init__(self, items: List[Tuple[str, float, float]], images_dir: str) -> None:
        self.items = items
        self.images_dir = images_dir

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        fn, x, y = self.items[idx]
        path = os.path.join(self.images_dir, fn)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        if img.shape[0] != INPUT_SIZE or img.shape[1] != INPUT_SIZE:
            img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        target = torch.tensor([x, y], dtype=torch.float32)
        return t, target


def _collect_labeled(
    dataset_dir: str,
) -> List[Tuple[str, float, float]]:
    images_dir = default_images_dir(dataset_dir)
    labels = load_labels(default_labels_path(dataset_dir))
    items: List[Tuple[str, float, float]] = []
    for fn, (x, y) in labels.items():
        path = os.path.join(images_dir, fn)
        if not os.path.isfile(path):
            print(f"[train] skip missing image: {fn}", flush=True)
            continue
        items.append((fn, float(x), float(y)))
    return items


def _split(
    items: List[Tuple[str, float, float]],
    val_frac: float,
    seed: int,
) -> Tuple[List, List]:
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_frac))) if len(shuffled) >= 5 else max(
        0, min(1, len(shuffled) // 5)
    )
    if len(shuffled) < 2:
        return shuffled, []
    if n_val >= len(shuffled):
        n_val = max(1, len(shuffled) // 5) if len(shuffled) >= 5 else 0
    val = shuffled[:n_val]
    train = shuffled[n_val:] or shuffled
    return train, val


def _epoch_loss(model, loader, device, optimizer=None) -> float:
    train = optimizer is not None
    model.train(train)
    total = 0.0
    n = 0
    crit = nn.MSELoss(reduction="sum")
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = crit(pred, yb)
        if train:
            loss.backward()
            optimizer.step()
        total += float(loss.item())
        n += int(yb.shape[0])
    return total / max(n, 1)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Train ml_tip TipCNN")
    p.add_argument("--dataset", default=default_dataset_dir(), help="Dataset root")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", default=default_checkpoint_path())
    p.add_argument("--resume", action="store_true", help="Resume from --checkpoint if present")
    p.add_argument("--device", default="", help="cpu | cuda | empty=auto")
    args = p.parse_args(argv)

    items = _collect_labeled(args.dataset)
    if not items:
        print(
            f"[train] no labeled samples under {args.dataset}\n"
            "  Add images/ + labels.csv (see ml_tip/README.md)",
            flush=True,
        )
        return 1

    train_items, val_items = _split(items, args.val_frac, args.seed)
    images_dir = default_images_dir(args.dataset)
    train_ds = TipDataset(train_items, images_dir)
    val_ds = TipDataset(val_items, images_dir) if val_items else None

    device_s = args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_s)

    model = build_model().to(device)
    start_epoch = 0
    best_val = float("inf")
    ckpt_path = args.checkpoint
    os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)) or ".", exist_ok=True)

    if args.resume and os.path.isfile(ckpt_path):
        blob = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(blob["model"])
        start_epoch = int(blob.get("epoch", 0))
        best_val = float(blob.get("best_val", best_val))
        print(
            f"[train] resumed epoch={start_epoch} best_val={best_val:.4f} from {ckpt_path}",
            flush=True,
        )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(
        train_ds,
        batch_size=min(args.batch_size, len(train_ds)),
        shuffle=True,
        num_workers=0,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=min(args.batch_size, len(val_ds)), shuffle=False, num_workers=0)
        if val_ds and len(val_ds) > 0
        else None
    )

    print(
        f"[train] samples={len(items)} train={len(train_ds)} val={len(val_ds) if val_ds else 0} "
        f"params={count_parameters(model)} device={device}",
        flush=True,
    )

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        tr = _epoch_loss(model, train_loader, device, optimizer=opt)
        if val_loader is not None:
            with torch.no_grad():
                va = _epoch_loss(model, val_loader, device, optimizer=None)
        else:
            va = tr
        print(f"[train] epoch {epoch:4d}  train_loss={tr:.4f}  val_loss={va:.4f}", flush=True)

        if va <= best_val:
            best_val = va
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_val": best_val,
                    "input_size": INPUT_SIZE,
                    "coord": "pixels_0_199",
                },
                ckpt_path,
            )
            print(f"[train] saved best → {ckpt_path} (val={best_val:.4f})", flush=True)

    print(f"[train] done. best_val={best_val:.4f} checkpoint={ckpt_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
