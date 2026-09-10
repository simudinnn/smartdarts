"""Camera calibration screen — live preview of 3 board cameras."""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from board_calibration import BoardCalibrator

# UVC na Linuxu: video0/2/4 = capture; 1/3/5 = metadata (nije capture → hang).
CAMERA_INDICES: Tuple[int, ...] = (0, 2, 4)
CAMERA_LABELS: Tuple[str, ...] = ("Kamera 0", "Kamera 2", "Kamera 4")
CALIBRATION_FPS = 15
CALIBRATION_FRAME_INTERVAL = 1.0 / CALIBRATION_FPS

# Live preview / detekcija: niska rezolucija da Pi izdrži 3 USB kamere.
# Veća točnost kalibracije ide softverskim 2× upscaleom u calibrate_board —
# NE native 640 (USB renegotiation smrzava 3 kamere).
CAPTURE_WIDTH = 320
CAPTURE_HEIGHT = 240
_CAPTURE_LOOP_SLEEP_SEC = 0.05


def normalize_capture_frame(frame: np.ndarray) -> np.ndarray:
    """Soft resize na 320x240 — ne dira hardware format/exposure."""
    if frame is None or frame.size == 0:
        return frame
    h, w = frame.shape[:2]
    if int(w) == CAPTURE_WIDTH and int(h) == CAPTURE_HEIGHT:
        return frame
    return cv2.resize(
        frame, (CAPTURE_WIDTH, CAPTURE_HEIGHT), interpolation=cv2.INTER_AREA
    )


def _open_camera(index: int) -> Optional[cv2.VideoCapture]:
    """Raspberry Pi / V4L2: otvori /dev/video{0,2,4}, blagi 320x240, bez DSHOW.

    Feed se ionako soft-resizea na 320x240. Ne forsira fourcc/exposure.
    Ako postoji camera_controls.json (iz camera_controls.py), primijeni spremljene V4L2 kontrole.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"[cam] OPEN FAIL index={index}", flush=True)
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, CALIBRATION_FPS)
    try:
        from camera_controls import apply_saved_controls

        apply_saved_controls(index)
    except Exception:
        pass
    prop_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    prop_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_text = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
    print(
        f"[cam] OPEN OK index={index} "
        f"prop={prop_w}x{prop_h} @ {actual_fps:.1f} FPS, {fourcc_text} "
        f"(soft->{CAPTURE_WIDTH}x{CAPTURE_HEIGHT})",
        flush=True,
    )
    return cap


_FAKE_CAM_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
# e.g. hit_155441_966_S12_cam0_raw.png → hit_id + cam index 0/2/4
_FAKE_HIT_CAM_RE = re.compile(
    r"^(?P<hit_id>.+)_cam(?P<cam>\d+)_raw\.(?:png|jpg|jpeg|bmp|webp)$",
    re.IGNORECASE,
)


def _resolve_fake_cam_image(fake_dir: str, index: int) -> Optional[str]:
    """Traži cam_{index}.png / .jpg ... u fake_cams folderu."""
    base = f"cam_{index}"
    for ext in _FAKE_CAM_EXTS:
        path = os.path.join(fake_dir, base + ext)
        if os.path.isfile(path):
            return path
    # Dozvoli i cam_0 bez extensiona / bilo koji prefix match
    try:
        for name in os.listdir(fake_dir):
            stem, ext = os.path.splitext(name)
            if stem.lower() == base.lower() and ext.lower() in _FAKE_CAM_EXTS:
                return os.path.join(fake_dir, name)
    except OSError:
        pass
    return None


class FakeCameraCapture:
    """Virtualna kamera: vraća trenutnu sliku (za test bez USB kamera)."""

    def __init__(self, image_path: str) -> None:
        self._path = image_path
        self._frame: Optional[np.ndarray] = None
        self.set_image(image_path)

    def set_image(self, image_path: str) -> bool:
        """Učitaj novi kadar (J/L simulacija bacanja)."""
        frame = cv2.imread(image_path)
        if frame is None:
            return False
        h, w = frame.shape[:2]
        if w != CAPTURE_WIDTH or h != CAPTURE_HEIGHT:
            frame = normalize_capture_frame(frame)
        self._path = image_path
        self._frame = frame
        return True

    def isOpened(self) -> bool:
        return self._frame is not None

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._frame is None:
            return False, None
        return True, self._frame.copy()

    def release(self) -> None:
        self._frame = None

    def set(self, *_args, **_kwargs) -> bool:
        return True


def default_fake_cams_dir() -> str:
    """fake_cams pored glavnog skripta / projekta."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "fake_cams")


def default_fake_darts_dir() -> str:
    """fake_darts — raw hit tripletovi (cam0/2/4) za J tipku."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "fake_darts")


def index_fake_dart_hits(
    fake_darts_dir: str,
    *,
    limit: Optional[int] = None,
) -> List[Tuple[str, Dict[int, str]]]:
    """Grupiraj hit_*_cam{0,2,4}_raw.* u potpune triplete, sortirane po hit_id.

    Po defaultu vraća sve potpune triplete (J tipka wrapa kroz cijelu listu).
    ``limit`` opcionalno skraćuje playlistu (testovi).
    """
    wanted = set(CAMERA_INDICES)
    groups: Dict[str, Dict[int, str]] = {}
    try:
        names = os.listdir(fake_darts_dir)
    except OSError:
        return []
    for name in names:
        m = _FAKE_HIT_CAM_RE.match(name)
        if not m:
            continue
        cam = int(m.group("cam"))
        if cam not in wanted:
            continue
        hit_id = m.group("hit_id")
        groups.setdefault(hit_id, {})[cam] = os.path.join(fake_darts_dir, name)
    hits: List[Tuple[str, Dict[int, str]]] = []
    for hit_id in sorted(groups.keys()):
        paths = groups[hit_id]
        if all(idx in paths for idx in CAMERA_INDICES):
            hits.append((hit_id, paths))
    if limit is not None:
        hits = hits[: max(0, int(limit))]
    return hits


class CameraManager:
    """Opens and reads from the three dart-board cameras (background thread)."""

    def __init__(self, fake_cams_dir: Optional[str] = None) -> None:
        self._caps: Dict[int, Optional[Union[cv2.VideoCapture, FakeCameraCapture]]] = {}
        self._active = False
        self._fake_cams_dir = fake_cams_dir
        self._fake_hits: List[Tuple[str, Dict[int, str]]] = []
        self._fake_hit_index: int = -1  # -1 = empty board
        self._lock = threading.Lock()
        self._latest: Dict[int, Optional[np.ndarray]] = {i: None for i in CAMERA_INDICES}
        self._thread: Optional[threading.Thread] = None
        self._flush_request = 0
        self._open_done = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def fake_mode(self) -> bool:
        return bool(self._fake_cams_dir)

    def start(self) -> None:
        """Pokreni capture nit. Za kalibraciju zovi wait_for_frames() prije prvog crtanja."""
        if self._active:
            return
        self._active = True
        self._open_done = False
        with self._lock:
            self._latest = {i: None for i in CAMERA_INDICES}
        if self._fake_cams_dir:
            self._start_fake()
            self._open_done = True
            return
        print("[cam] start (open 0/2/4)...", flush=True)
        self._thread = threading.Thread(
            target=self._capture_loop, name="cam-capture", daemon=True
        )
        self._thread.start()

    def _start_fake(self) -> None:
        fake_dir = os.path.abspath(self._fake_cams_dir or "")
        print(f"[cam] FAKECAMS — {fake_dir}", flush=True)
        self._fake_hit_index = -1
        for idx in CAMERA_INDICES:
            path = _resolve_fake_cam_image(fake_dir, idx)
            if path is None:
                print(f"[cam] FAKECAMS: nema slike za cam_{idx}", flush=True)
                self._caps[idx] = None
                continue
            cap = FakeCameraCapture(path)
            if not cap.isOpened():
                print(f"[cam] FAKECAMS: ne mogu ucitati {path}", flush=True)
                self._caps[idx] = None
            else:
                print(f"[cam] FAKECAMS cam_{idx} <- {path}", flush=True)
                self._caps[idx] = cap
        darts_dir = default_fake_darts_dir()
        self._fake_hits = index_fake_dart_hits(darts_dir) if os.path.isdir(darts_dir) else []
        hit_ids = ", ".join(h for h, _ in self._fake_hits) or "(none)"
        n_hits = len(self._fake_hits)
        print(
            f"[cam] FAKECAMS hits: {n_hits} "
            f"playlist u {darts_dir} [{hit_ids}] (J=next hit wrap, L=next player)",
            flush=True,
        )
        # Jednokratni snapshot u _latest (fake je statičan dok J ne promijeni).
        self._publish_from_caps(flush=0)

    def _capture_loop(self) -> None:
        try:
            # Otvori sve 3 prije prvog publisha — kalibracija ne smije prvo pokazati NEMA SIGNALA.
            for idx in CAMERA_INDICES:
                if not self._active:
                    return
                print(f"[cam] opening index={idx}...", flush=True)
                try:
                    cap = _open_camera(idx)
                except Exception as exc:
                    print(f"[cam] OPEN ERR index={idx}: {exc}", flush=True)
                    cap = None
                if not self._active:
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
                    return
                self._caps[idx] = cap
            failed = [i for i in CAMERA_INDICES if self._caps.get(i) is None]
            for idx in failed:
                if not self._active:
                    return
                time.sleep(0.3)
                print(f"[cam] retry index={idx}...", flush=True)
                try:
                    cap = _open_camera(idx)
                except Exception as exc:
                    print(f"[cam] OPEN ERR index={idx}: {exc}", flush=True)
                    cap = None
                self._caps[idx] = cap
            print(
                f"[cam] capture loop — open: {self.opened_indices()}",
                flush=True,
            )
            self._publish_from_caps(flush=0)
            self._open_done = True
            while self._active:
                flush = 0
                with self._lock:
                    flush = int(self._flush_request)
                    self._flush_request = 0
                self._publish_from_caps(flush=flush)
                time.sleep(_CAPTURE_LOOP_SLEEP_SEC)
        except Exception as exc:
            print(f"[cam] capture loop ERR: {exc}", flush=True)
        finally:
            self._open_done = True
            print("[cam] capture loop exit", flush=True)

    def _publish_from_caps(self, *, flush: int = 0) -> None:
        frames: Dict[int, Optional[np.ndarray]] = {}
        reads = max(1, int(flush) + 1)
        for idx in CAMERA_INDICES:
            cap = self._caps.get(idx)
            if cap is not None and cap.isOpened():
                ok, frame = False, None
                try:
                    for _ in range(reads):
                        ok, frame = cap.read()
                except Exception:
                    ok, frame = False, None
                if ok and frame is not None:
                    frames[idx] = normalize_capture_frame(frame)
                else:
                    frames[idx] = None
            else:
                frames[idx] = None
        with self._lock:
            self._latest = frames

    def stop(self) -> None:
        self._active = False
        self._open_done = False
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=8.0)
        self._thread = None
        for cap in list(self._caps.values()):
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
        self._caps.clear()
        self._fake_hits = []
        self._fake_hit_index = -1
        with self._lock:
            self._latest = {i: None for i in CAMERA_INDICES}

    def frames_ready_count(self) -> int:
        with self._lock:
            return sum(1 for f in self._latest.values() if f is not None)

    def wait_for_frames(
        self,
        *,
        timeout_sec: float = 12.0,
        min_cams: Optional[int] = None,
    ) -> bool:
        """Pričekaj da capture nit da dovoljno kadrova (clear-board / start igre)."""
        need = int(min_cams) if min_cams is not None else len(CAMERA_INDICES)
        need = max(1, min(need, len(CAMERA_INDICES)))
        deadline = time.perf_counter() + float(timeout_sec)
        while time.perf_counter() < deadline:
            if not self._active:
                return False
            n = self.frames_ready_count()
            if n >= need:
                return True
            if self._open_done:
                opened = len(self.opened_indices())
                if opened <= 0:
                    return False
                # Open gotov — dovoljno je imati kadar sa svih uspješno otvorenih.
                if n >= min(need, opened) and n > 0:
                    return True
            time.sleep(0.08)
        n = self.frames_ready_count()
        opened = len(self.opened_indices()) if self._open_done else 0
        if n >= need:
            return True
        if opened > 0 and n >= min(need, opened):
            return True
        return False

    def _apply_fake_paths(self, paths: Dict[int, str]) -> bool:
        """Zamijeni kadrove svih FakeCameraCapture instanata."""
        ok = True
        for idx in CAMERA_INDICES:
            cap = self._caps.get(idx)
            path = paths.get(idx)
            if path is None or not isinstance(cap, FakeCameraCapture):
                ok = False
                continue
            if not cap.set_image(path):
                print(f"[cam] FAKECAMS: ne mogu ucitati {path}", flush=True)
                ok = False
        if ok:
            self._publish_from_caps(flush=0)
        return ok

    def cycle_fake_hit(self) -> Optional[str]:
        """J: idući hit u cijeloj fake_darts playlisti (wrap). Vraća hit_id ili None.

        Početni index=-1 (prazna ploča); prvi J učitava hit 0, zatim 1…n-1→0…
        """
        if not self.fake_mode or not self._active or not self._fake_hits:
            return None
        n = len(self._fake_hits)
        self._fake_hit_index = (self._fake_hit_index + 1) % n
        hit_id, paths = self._fake_hits[self._fake_hit_index]
        self._apply_fake_paths(paths)
        print(
            f"[fakecams] J -> hit {self._fake_hit_index + 1}/{n} {hit_id}",
            flush=True,
        )
        return hit_id

    def grab_frames(self, *, flush: int = 0) -> Dict[int, Optional[np.ndarray]]:
        """Vrati zadnje kadrove iz capture niti (nikad ne radi USB read na UI niti)."""
        if not self._active:
            return {i: None for i in CAMERA_INDICES}
        if self._fake_cams_dir:
            if flush > 0:
                self._publish_from_caps(flush=flush)
            with self._lock:
                return {i: (None if f is None else f.copy()) for i, f in self._latest.items()}
        if flush > 0:
            with self._lock:
                self._flush_request = max(int(self._flush_request), int(flush))
            # Kratko pričekaj da capture nit odbaci buffer (max ~150ms).
            deadline = time.perf_counter() + 0.15
            while time.perf_counter() < deadline:
                with self._lock:
                    if self._flush_request == 0:
                        break
                time.sleep(0.01)
        with self._lock:
            return {i: (None if f is None else f.copy()) for i, f in self._latest.items()}

    def opened_indices(self) -> List[int]:
        return [
            idx
            for idx in CAMERA_INDICES
            if self._caps.get(idx) is not None and self._caps[idx].isOpened()
        ]

    def status_text(self) -> str:
        ok_count = sum(
            1 for idx in CAMERA_INDICES if self._caps.get(idx) is not None and self._caps[idx].isOpened()
        )
        prefix = "FAKE " if self._fake_cams_dir else ""
        return f"{prefix}{ok_count}/{len(CAMERA_INDICES)} kamera spojeno"


def _placeholder(w: int, h: int, label: str, msg: str) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (28, 28, 32)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (70, 70, 80), 2)
    cv2.putText(img, label, (16, 36), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 2)
    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.55, 2)
    cv2.putText(img, msg, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_DUPLEX, 0.55, (80, 80, 200), 2)
    return img


def _draw_button(
    panel: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    label: str,
    *,
    active: bool = False,
) -> None:
    fill = (0, 200, 200) if active else (0, 140, 140)
    border = (0, 180, 180)
    cv2.rectangle(panel, (x1, y1), (x2, y2), fill, -1)
    cv2.rectangle(panel, (x1, y1), (x2, y2), border, 2)
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    # Veći gumbi (seg.20 strelice) → veći font.
    scale = max(0.45, min(1.35, min(bw, bh) / 48.0))
    thickness = 2 if scale < 0.85 else 3
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)
    tx = x1 + (bw - tw) // 2
    ty = y1 + (bh + th) // 2 - bl // 2
    cv2.putText(panel, label, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thickness)


def _draw_bull_hint_marker(img: np.ndarray, x: float, y: float) -> None:
    """Cross + circle at approximate bull click (frame coords)."""
    cx, cy = int(round(x)), int(round(y))
    h, w = img.shape[:2]
    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        return
    color = (0, 255, 255)
    r = max(10, int(min(h, w) * 0.045))
    cv2.circle(img, (cx, cy), r, color, 2, cv2.LINE_AA)
    cv2.drawMarker(
        img,
        (cx, cy),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=max(14, r * 2),
        thickness=2,
        line_type=cv2.LINE_AA,
    )


def _map_hint_to_display(
    board_calibrator: Optional[BoardCalibrator],
    cam_idx: int,
    hx: float,
    hy: float,
    display_frame: np.ndarray,
    raw_frame: Optional[np.ndarray],
) -> Tuple[float, float]:
    """Capture-frame hint → currently displayed frame (overlay or topdown)."""
    fh_d, fw_d = display_frame.shape[:2]
    if (
        board_calibrator is not None
        and board_calibrator.show_topdown
        and raw_frame is not None
    ):
        mapped = board_calibrator.map_camera_to_topdown_display(
            int(cam_idx),
            hx,
            hy,
            display_size=(int(fw_d), int(fh_d)),
        )
        if mapped is not None:
            return float(mapped[0]), float(mapped[1])
        fh_raw, fw_raw = raw_frame.shape[:2]
        if fw_raw > 0 and fh_raw > 0 and (fw_raw != fw_d or fh_raw != fh_d):
            return hx * (fw_d / float(fw_raw)), hy * (fh_d / float(fh_raw))
        return hx, hy
    if raw_frame is not None:
        fh_raw, fw_raw = raw_frame.shape[:2]
        if fw_raw > 0 and fh_raw > 0 and (fw_raw != fw_d or fh_raw != fh_d):
            return hx * (fw_d / float(fw_raw)), hy * (fh_d / float(fh_raw))
    return hx, hy


def _draw_ellipse_hint_marker(img: np.ndarray, x: float, y: float) -> None:
    """Orange rim marker — click target is the centre, not a label."""
    cx, cy = int(round(x)), int(round(y))
    h, w = img.shape[:2]
    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        return
    color = (40, 90, 255)
    r = max(7, int(min(h, w) * 0.028))
    cv2.circle(img, (cx, cy), r, color, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 3, color, -1, cv2.LINE_AA)


def _draw_ellipse_hint_preview(img: np.ndarray, points: List[Tuple[float, float]]) -> None:
    """Ellipse through the clicked rim points (no chords)."""
    if len(points) < 3:
        return
    try:
        from board_calibration import _ellipse_from_hint_points

        h, w = img.shape[:2]
        ell = _ellipse_from_hint_points(
            [(float(p[0]), float(p[1])) for p in points],
            float(min(h, w)),
            strict=False,
        )
        if ell is not None:
            cv2.ellipse(img, ell, (40, 90, 255), 2, cv2.LINE_AA)
    except Exception:
        pass


def build_calibration_view(
    width: int,
    height: int,
    frames: Dict[int, Optional[np.ndarray]],
    *,
    status: str = "",
    board_calibrator: Optional[BoardCalibrator] = None,
    buttons_out: Optional[list] = None,
    draw_action_buttons: bool = True,
    bull_hints: Optional[Dict[int, Tuple[float, float]]] = None,
    ellipse_hints: Optional[Dict[int, List[Tuple[float, float]]]] = None,
    tiles_out: Optional[list] = None,
    click_bull_mode: bool = False,
    click_bull_banner: str = "",
    click_ellipse_mode: bool = False,
) -> np.ndarray:
    """Render calibration screen with 3 camera previews and board overlay."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (0, 0, 0)
    hints = bull_hints or {}
    ell_hints = ellipse_hints or {}

    margin = max(16, int(min(width, height) * 0.025))
    title = "KALIBRACIJA KAMERA I PLOCE"
    title_scale = max(0.65, min(1.3, width / 950.0))
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, title_scale, 2)
    title_y = margin + th
    cv2.putText(panel, title, ((width - tw) // 2, title_y), cv2.FONT_HERSHEY_DUPLEX, title_scale, (235, 235, 235), 2)

    hint = f"2/ESC izlaz  |  B kalibriraj  |  O elipsa  |  T topdown  |  < > seg.20"
    if (click_bull_mode or click_ellipse_mode) and click_bull_banner:
        hint = click_bull_banner
    hint_scale = max(0.36, min(0.48, width / 1500.0))
    (hw, hh), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, hint_scale, 1)
    if click_bull_mode:
        hint_color = (0, 220, 255)
    elif click_ellipse_mode:
        hint_color = (40, 90, 255)
    else:
        hint_color = (130, 130, 130)
    cv2.putText(
        panel,
        hint,
        ((width - hw) // 2, title_y + int(th * 0.55) + hh + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        hint_scale,
        hint_color,
        1,
    )

    board_status = board_calibrator.status_summary() if board_calibrator is not None else ""

    btn_h = max(40, int(height * 0.065)) if draw_action_buttons else 0
    btn_gap = max(8, int(width * 0.01))
    btn_y2 = height - margin
    btn_y1 = btn_y2 - btn_h if draw_action_buttons else height - margin
    footer_h = 44 if draw_action_buttons else 8

    if board_status:
        st_scale = max(0.48, min(0.62, width / 1200.0))
        cv2.putText(
            panel,
            board_status,
            (margin, max(title_y + th + 24, btn_y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            st_scale,
            (255, 200, 80),
            2,
        )
    if status:
        st_scale = max(0.42, min(0.52, width / 1400.0))
        cv2.putText(
            panel,
            status,
            (margin, btn_y1 - footer_h + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            st_scale,
            (100, 220, 100),
            1,
        )

    btn_specs = [
        ("calibration_back", "NAZAD"),
        ("calibration_click_bull", "BULL"),
        ("calibration_click_ellipse", "ELIPSA"),
        ("calibration_capture", "TOPDOWN"),
        ("calibration_detect", "KALIBRIRAJ"),
    ]
    btn_w = max(100, int((width - 2 * margin - btn_gap * (len(btn_specs) - 1)) / len(btn_specs)))
    if buttons_out is not None:
        buttons_out.clear()
    if tiles_out is not None:
        tiles_out.clear()

    if draw_action_buttons:
        for i, (bid, blabel) in enumerate(btn_specs):
            bx1 = margin + i * (btn_w + btn_gap)
            bx2 = min(width - margin, bx1 + btn_w)
            active = False
            if board_calibrator is not None and bid == "calibration_capture":
                active = bool(board_calibrator.show_topdown)
            if bid == "calibration_click_bull":
                active = bool(click_bull_mode)
            if bid == "calibration_click_ellipse":
                active = bool(click_ellipse_mode)
            _draw_button(panel, bx1, btn_y1, bx2, btn_y2, blabel, active=active)
            if buttons_out is not None:
                buttons_out.append({"id": bid, "rect": (bx1, btn_y1, bx2, btn_y2)})

    top = title_y + int(th * 1.6) + margin
    bottom = btn_y1 - margin - footer_h
    avail_h = max(1, bottom - top)
    gap = max(8, int(width * 0.012))
    cell_w = max(1, (width - 2 * margin - gap * (len(CAMERA_INDICES) - 1)) // len(CAMERA_INDICES))
    arrow_h = max(56, int(height * 0.095))
    arrow_gap = max(8, int(cell_w * 0.05))
    video_h = max(1, avail_h - arrow_h - 8)

    # Overlay is always on (topdown is a separate warp view). Clicks unwarp when needed.
    display_frames = frames
    if board_calibrator is not None:
        display_frames = board_calibrator.render_frames(frames)

    for i, cam_idx in enumerate(CAMERA_INDICES):
        x1 = margin + i * (cell_w + gap)
        x2 = x1 + cell_w
        y1 = top
        y2 = top + avail_h
        label = CAMERA_LABELS[i] if i < len(CAMERA_LABELS) else f"Kamera {cam_idx}"

        frame = display_frames.get(cam_idx)
        raw_frame = frames.get(cam_idx)
        cal = board_calibrator.get(cam_idx) if board_calibrator is not None else None
        hint_pt = hints.get(int(cam_idx))
        if hint_pt is None:
            hint_pt = hints.get(cam_idx)
        ell_pts = ell_hints.get(int(cam_idx))
        if ell_pts is None:
            ell_pts = ell_hints.get(cam_idx)
        ell_pts = [
            (float(p[0]), float(p[1]))
            for p in (ell_pts or [])
            if p is not None and len(p) >= 2
        ][:4]
        show_ell = bool(click_ellipse_mode) and bool(ell_pts)

        if frame is not None:
            # Draw hints on a copy of the displayed frame (scale if overlay size ≠ capture).
            draw_src = frame
            if hint_pt is not None or show_ell:
                draw_src = frame.copy()
                if hint_pt is not None:
                    hx, hy = _map_hint_to_display(
                        board_calibrator,
                        int(cam_idx),
                        float(hint_pt[0]),
                        float(hint_pt[1]),
                        draw_src,
                        raw_frame,
                    )
                    _draw_bull_hint_marker(draw_src, hx, hy)
                if show_ell:
                    mapped_ell: List[Tuple[float, float]] = []
                    for ex, ey in ell_pts:
                        mx, my = _map_hint_to_display(
                            board_calibrator,
                            int(cam_idx),
                            ex,
                            ey,
                            draw_src,
                            raw_frame,
                        )
                        mapped_ell.append((mx, my))
                    _draw_ellipse_hint_preview(draw_src, mapped_ell)
                    for mx, my in mapped_ell:
                        _draw_ellipse_hint_marker(draw_src, mx, my)
            fh, fw = draw_src.shape[:2]
            scale = min(cell_w / float(fw), video_h / float(fh))
            nw = max(1, int(fw * scale))
            nh = max(1, int(fh * scale))
            resized = cv2.resize(draw_src, (nw, nh), interpolation=cv2.INTER_CUBIC)
            tile = np.zeros((avail_h, cell_w, 3), dtype=np.uint8)
            tile[:] = (18, 18, 22)
            ox = (cell_w - nw) // 2
            oy = (video_h - nh) // 2
            tile[oy : oy + nh, ox : ox + nw] = resized
            if cal is not None and cal.is_valid():
                if cal.has_board_ellipse() and cal.has_wires():
                    status_col = (60, 200, 60)
                    status_msg = "OK"
                elif cal.has_board_ellipse():
                    # Bull+elipsa bez žica — NIJE kompletna kalibracija (ranije zeleno "BULL").
                    status_col = (0, 165, 255)
                    status_msg = "ELIPSA"
                else:
                    status_col = (0, 200, 255)
                    status_msg = "BULL"
            else:
                status_col = (0, 180, 255)
                status_msg = "CEKAJ B"
            if tiles_out is not None and raw_frame is not None:
                rh, rw = raw_frame.shape[:2]
                # Map clicks via raw capture size even if overlay is shown (same dims).
                # Include display letterbox offsets so widget→frame stays aligned.
                tiles_out.append(
                    {
                        "cam_idx": int(cam_idx),
                        "video_rect": (x1 + ox, y1 + oy, x1 + ox + nw, y1 + oy + nh),
                        "frame_size": (int(rw), int(rh)),
                        "display_size": (int(fw), int(fh)),
                    }
                )
        else:
            tile = np.zeros((avail_h, cell_w, 3), dtype=np.uint8)
            tile[:] = (18, 18, 22)
            tile[0:video_h, 0:cell_w] = _placeholder(cell_w, video_h, label, "NEMA SIGNALA")
            status_col = (60, 60, 220)
            status_msg = "OFFLINE"

        cv2.rectangle(tile, (0, 0), (cell_w - 1, video_h - 1), (90, 90, 100), 2)
        cv2.putText(tile, label, (10, 28), cv2.FONT_HERSHEY_DUPLEX, 0.55, (220, 220, 220), 2)
        cv2.circle(tile, (cell_w - 22, 22), 8, status_col, -1)
        (smw, _), _ = cv2.getTextSize(status_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(tile, status_msg, (cell_w - smw - 38, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_col, 2)

        arrow_y1 = video_h + 4
        arrow_y2 = arrow_y1 + arrow_h
        arrow_btn_w = max(36, (cell_w - arrow_gap) // 2)
        left_x1 = (cell_w - 2 * arrow_btn_w - arrow_gap) // 2
        left_x2 = left_x1 + arrow_btn_w
        right_x1 = left_x2 + arrow_gap
        right_x2 = right_x1 + arrow_btn_w
        seg20_ready = cal is not None and cal.has_wires()
        _draw_button(tile, left_x1, arrow_y1, left_x2, arrow_y2, "<", active=seg20_ready)
        _draw_button(tile, right_x1, arrow_y1, right_x2, arrow_y2, ">", active=seg20_ready)
        if buttons_out is not None:
            buttons_out.append(
                {"id": f"calibration_seg20_left_{cam_idx}", "rect": (x1 + left_x1, y1 + arrow_y1, x1 + left_x2, y1 + arrow_y2)}
            )
            buttons_out.append(
                {"id": f"calibration_seg20_right_{cam_idx}", "rect": (x1 + right_x1, y1 + arrow_y1, x1 + right_x2, y1 + arrow_y2)}
            )

        panel[y1:y2, x1:x2] = tile

    return panel
