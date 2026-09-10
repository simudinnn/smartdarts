"""Paths and dataset helpers for ml_tip."""

from __future__ import annotations

import csv
import os
from typing import List, Optional, Tuple

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def package_dir() -> str:
    return _PKG_DIR


def default_dataset_dir() -> str:
    return os.path.join(_PKG_DIR, "dataset")


def default_images_dir(dataset_dir: Optional[str] = None) -> str:
    return os.path.join(dataset_dir or default_dataset_dir(), "images")


def default_labels_path(dataset_dir: Optional[str] = None) -> str:
    return os.path.join(dataset_dir or default_dataset_dir(), "labels.csv")


def default_checkpoint_path() -> str:
    return os.path.join(_PKG_DIR, "checkpoints", "tip_cnn_best.pt")


def default_onnx_path() -> str:
    return os.path.join(_PKG_DIR, "tip_detector.onnx")


def ensure_dataset_dirs(dataset_dir: Optional[str] = None) -> Tuple[str, str, str]:
    root = dataset_dir or default_dataset_dir()
    images = os.path.join(root, "images")
    labels = os.path.join(root, "labels.csv")
    os.makedirs(images, exist_ok=True)
    if not os.path.isfile(labels):
        with open(labels, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "x", "y"])
    return root, images, labels


def next_image_filename(images_dir: str) -> str:
    """Next auto-increment name: 000001.png, 000002.png, ..."""
    max_n = 0
    if os.path.isdir(images_dir):
        for name in os.listdir(images_dir):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
                continue
            try:
                max_n = max(max_n, int(stem))
            except ValueError:
                continue
    return f"{max_n + 1:06d}.png"


def load_labels(labels_path: str) -> dict:
    """filename -> (x, y) float."""
    out: dict = {}
    if not os.path.isfile(labels_path):
        return out
    with open(labels_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = (row.get("filename") or "").strip()
            if not fn:
                continue
            try:
                out[fn] = (float(row["x"]), float(row["y"]))
            except (KeyError, ValueError, TypeError):
                continue
    return out


def save_labels(labels_path: str, labels: dict) -> None:
    rows: List[Tuple[str, float, float]] = []
    for fn in sorted(labels.keys()):
        x, y = labels[fn]
        rows.append((fn, float(x), float(y)))
    os.makedirs(os.path.dirname(os.path.abspath(labels_path)) or ".", exist_ok=True)
    with open(labels_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "x", "y"])
        for fn, x, y in rows:
            w.writerow([fn, f"{x:.4f}", f"{y:.4f}"])


def list_dataset_images(images_dir: str) -> List[str]:
    if not os.path.isdir(images_dir):
        return []
    names = []
    for name in os.listdir(images_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
            names.append(name)
    return sorted(names)
