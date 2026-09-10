#!/usr/bin/env python3
"""Simple OpenCV labeling tool for ml_tip dataset tip (x, y).

Keys (tipke)
------------
  Left click     set tip / postavi vrh (auto-save labels.csv)
  n / →          next / sljedeća
  p / ←          previous / prethodna
  s              save labels.csv (also auto-saves on click)
  d / Delete     clear label / obriši oznaku
  q / Esc        quit / izlaz
  + / -          zoom

Labels are pixel coordinates in the 200×200 image (x right, y down).
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ml_tip.paths import (
    default_dataset_dir,
    default_images_dir,
    default_labels_path,
    ensure_dataset_dirs,
    list_dataset_images,
    load_labels,
    save_labels,
)
from ml_tip.preprocess import INPUT_SIZE

WINDOW = "ml_tip label | click=tip | n/p=next/prev | d=delete | q=quit"


class LabelApp:
    def __init__(self, dataset_dir: str) -> None:
        ensure_dataset_dirs(dataset_dir)
        self.dataset_dir = dataset_dir
        self.images_dir = default_images_dir(dataset_dir)
        self.labels_path = default_labels_path(dataset_dir)
        self.names = list_dataset_images(self.images_dir)
        self.labels = load_labels(self.labels_path)
        self.idx = 0
        self.scale = 3  # display magnification for easier clicking
        # Size of the last image passed to imshow (for click → image mapping)
        self._disp_w = INPUT_SIZE * self.scale
        self._disp_h = INPUT_SIZE * self.scale
        self._status = "Klikni = postavi vrh | Click = set tip"

    def _cur_name(self) -> Optional[str]:
        if not self.names:
            return None
        return self.names[self.idx]

    def _load_u8(self) -> Optional[np.ndarray]:
        name = self._cur_name()
        if name is None:
            return None
        path = os.path.join(self.images_dir, name)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        if img.shape[0] != INPUT_SIZE or img.shape[1] != INPUT_SIZE:
            img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        return img

    @staticmethod
    def _draw_tip_marker(vis: np.ndarray, dx: int, dy: int) -> None:
        """High-contrast tip marker on dark grayscale motion (display pixels)."""
        h, w = vis.shape[:2]
        if not (0 <= dx < w and 0 <= dy < h):
            return
        # Outer glow / outline for dark backgrounds
        cv2.circle(vis, (dx, dy), 18, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(vis, (dx, dy), 14, (0, 255, 255), 2, cv2.LINE_AA)  # yellow ring
        # Magenta filled core
        cv2.circle(vis, (dx, dy), 8, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, (dx, dy), 8, (0, 0, 0), 2, cv2.LINE_AA)
        # Bright yellow crosshair
        arm = 28
        cv2.line(vis, (dx - arm, dy), (dx + arm, dy), (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(vis, (dx, dy - arm), (dx, dy + arm), (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(vis, (dx - arm, dy), (dx + arm, dy), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(vis, (dx, dy - arm), (dx, dy + arm), (0, 255, 255), 2, cv2.LINE_AA)

    def _empty_canvas(self) -> np.ndarray:
        h = max(INPUT_SIZE * self.scale, 360)
        w = max(INPUT_SIZE * self.scale, 640)
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = (28, 28, 28)
        lines = [
            "NEMA SLIKA / NO IMAGES",
            "",
            f"Folder: {self.images_dir}",
            "",
            "Stavi 200x200 PNG ovdje, ili u kiosku ukljuci:",
            "  Settings → ML tip dataset save: On",
            "",
            "Put 200x200 PNGs here, or enable in kiosk:",
            "  Settings → ML tip dataset save: On",
            "",
            "q = quit / izlaz",
        ]
        y0 = 36
        for line in lines:
            color = (0, 220, 255) if line.startswith("NEMA") else (220, 220, 220)
            thickness = 2 if line.startswith("NEMA") else 1
            cv2.putText(
                canvas, line, (24, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA
            )
            cv2.putText(
                canvas, line, (24, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thickness, cv2.LINE_AA
            )
            y0 += 28
        return canvas

    def _draw(self) -> np.ndarray:
        gray = self._load_u8()
        if gray is None:
            vis = self._empty_canvas()
            self._disp_w, self._disp_h = vis.shape[1], vis.shape[0]
            return vis

        # Upscale first, then draw marker in display pixels (crisp + visible on dark motion)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        vis = cv2.resize(
            bgr,
            (INPUT_SIZE * self.scale, INPUT_SIZE * self.scale),
            interpolation=cv2.INTER_NEAREST,
        )
        self._disp_w, self._disp_h = vis.shape[1], vis.shape[0]

        name = self._cur_name() or "?"
        tip = self.labels.get(name)
        if tip is not None:
            dx = int(round(tip[0] * self.scale))
            dy = int(round(tip[1] * self.scale))
            self._draw_tip_marker(vis, dx, dy)

        labeled = "DA/yes" if tip is not None else "NE/no"
        if tip is not None:
            tip_s = f"tip: {tip[0]:.1f}, {tip[1]:.1f}"
        else:
            tip_s = "tip: (nema / none)"
        # Hershey font has no Croatian diacritics — keep HUD ASCII-safe
        hud = [
            f"{self.idx + 1}/{len(self.names)}  {name}  oznaceno={labeled}  {tip_s}",
            "Klik=vrh | n/Right=sljedeca | p/Left=prethodna | s=spremi | d=obrisi | q=izlaz",
            "Click=tip | n/Right=next | p/Left=prev | s=save | d=delete | q=quit",
            self._status,
        ]
        y0 = 22
        for line in hud:
            cv2.putText(vis, line, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, line, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
            y0 += 20
        return vis

    def _window_display_size(self) -> Tuple[int, int]:
        """Actual image area in the window (handles resize / DPI quirks)."""
        try:
            _x, _y, ww, wh = cv2.getWindowImageRect(WINDOW)
            if ww > 1 and wh > 1:
                return int(ww), int(wh)
        except cv2.error:
            pass
        return max(1, self._disp_w), max(1, self._disp_h)

    def _click_to_image(self, x: int, y: int) -> Tuple[float, float]:
        dw, dh = self._window_display_size()
        mx = x * float(INPUT_SIZE) / float(dw)
        my = y * float(INPUT_SIZE) / float(dh)
        mx = float(np.clip(mx, 0.0, float(INPUT_SIZE - 1)))
        my = float(np.clip(my, 0.0, float(INPUT_SIZE - 1)))
        return mx, my

    def _refresh(self) -> None:
        vis = self._draw()
        cv2.imshow(WINDOW, vis)

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        name = self._cur_name()
        if name is None:
            self._status = "Nema slika - put PNGs in dataset/images"
            self._refresh()
            return
        mx, my = self._click_to_image(x, y)
        self.labels[name] = (mx, my)
        save_labels(self.labels_path, self.labels)
        self._status = f"Spremljeno / saved  tip: {mx:.1f}, {my:.1f}"
        print(f"[label] {name} -> ({mx:.2f}, {my:.2f})  auto-saved", flush=True)
        # Redraw immediately so the marker is visible without waiting for waitKey
        self._refresh()

    def run(self) -> int:
        if not self.names:
            print(
                f"[label] NEMA SLIKA / NO IMAGES in:\n  {self.images_dir}\n"
                "Stavi 200×200 PNG ovdje, ili u kiosku:\n"
                "  Settings → ML tip dataset save: On\n"
                "Put 200×200 PNGs here, or enable Save hits in kiosk settings.",
                flush=True,
            )
        # WINDOW_AUTOSIZE: window tracks image size; clicks still mapped via getWindowImageRect
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self._on_mouse)
        self._refresh()
        print(
            "[label] Tipke / keys:\n"
            "  click = set tip (auto-save)\n"
            "  n / → = next    p / ← = prev\n"
            "  s = save        d = delete label\n"
            "  q / Esc = quit  +/− = zoom",
            flush=True,
        )
        # Arrow codes from waitKeyEx (do NOT use 81/83 — those are 'Q'/'S')
        RIGHT = {2555904, 65363}
        LEFT = {2424832, 65361}
        DELETE = {3014656, 65535, 8, 127}

        while True:
            self._refresh()
            key = cv2.waitKeyEx(30)
            if key == -1:
                continue
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("n"), ord("N")) or key in RIGHT:
                if self.names:
                    self.idx = (self.idx + 1) % len(self.names)
                    self._status = "Sljedeca / next"
            elif key in (ord("p"), ord("P")) or key in LEFT:
                if self.names:
                    self.idx = (self.idx - 1) % len(self.names)
                    self._status = "Prethodna / prev"
            elif key in (ord("s"), ord("S")):
                save_labels(self.labels_path, self.labels)
                self._status = f"Spremljeno / saved"
                print(f"[label] saved {self.labels_path}", flush=True)
            elif key in (ord("d"), ord("D")) or key in DELETE:
                name = self._cur_name()
                if name and name in self.labels:
                    del self.labels[name]
                    save_labels(self.labels_path, self.labels)
                    self._status = f"Obrisano / cleared  {name}"
                    print(f"[label] cleared {name}", flush=True)
            elif key in (ord("+"), ord("=")):
                self.scale = min(8, self.scale + 1)
            elif key == ord("-"):
                self.scale = max(1, self.scale - 1)
        save_labels(self.labels_path, self.labels)
        cv2.destroyAllWindows()
        return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Label tip positions for ml_tip dataset")
    ap.add_argument("--dataset", default=default_dataset_dir())
    args = ap.parse_args(argv)
    return LabelApp(args.dataset).run()


if __name__ == "__main__":
    raise SystemExit(main())
