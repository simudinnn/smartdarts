"""Game + OpenCV vision session. Qt UI drives this; no OpenCV window drawing."""

from __future__ import annotations

import copy
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from calibration import (
    CALIBRATION_FRAME_INTERVAL,
    CAMERA_INDICES,
    CameraManager,
    build_calibration_view,
)
from board_calibration import BoardCalibrator
from dart_detection import (
    DartDetectResult,
    DartMotionDetector,
    FusedDartResult,
    EMPTY_BOARD_MAX_DIFF_MEAN,
    EMPTY_BOARD_MAX_PIXELS,
    EMPTY_BOARD_RAW_HAND_PIXELS,
    MOTION_CLEAR_PIXELS,
    MOTION_LOG_MIN_PIXELS,
    summarize_fused_log,
    summarize_motion_log,
)

import main_manual as mm
from kiosk_settings import get_settings, load_settings, start_qr_image_path, update_settings
from league_qr import ensure_game_qr


OnChange = Optional[Callable[[], None]]


def _bgr_to_hex(bgr: Tuple[int, int, int]) -> str:
    b, g, r = bgr
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


class KioskSession:
    def __init__(self, *, kiosk: bool = False, fake_cams_dir: Optional[str] = None) -> None:
        self.kiosk = bool(kiosk)
        self._on_change: OnChange = None
        self.all_hits: List[mm.DetectedHit] = []
        self.settings = load_settings()
        self.state: Dict[str, Any] = {
            "screen": "select_game" if self.settings.start_mode == "always_on" else "standby",
            "selected_game_mode": None,
            "num_players": 0,
            "current_player_idx": 0,
            "player_slots": [None, None, None, None],
            "_prev_screen": None,
            "rules_pending": {
                "double_in": False,
                "double_out": False,
                "max_rounds_20": False,
                "cricket_mode": "standard",
                "killer_lives": 3,
                "killer_assign": "random",
                "killer_activation": "double",
                "around_mult": "any",
                "around_order": "standard",
                "around_backstep": 0,
                "halve_mode": "standard",
            },
        }
        self.ctx: Dict[str, Any] = {
            "game": None,
            "input_multiplier": 1,
            "camera_mgr": CameraManager(fake_cams_dir=fake_cams_dir),
            "board_calibrator": BoardCalibrator(),
            "dart_detector": DartMotionDetector(),
            "_dart_last_tick": 0.0,
            "_dart_log_last_tick": 0.0,
            "_dart_no_ref_warned": False,
            "_dart_status": "",
            "_dart_wait_idle_after_switch": False,
            "_dart_idle_frames": 0,
            "_dart_empty_frames": 0,
            "_dart_turn_start_pending": False,
            "_dart_empty_log_last": 0.0,
            "_dart_score_locked": False,
            "_dart_block_auto_advance": False,
            "_dart_block_auto_detect": False,
            "_dart_undo_target_player": None,
            "_dart_undo_pending_hits": None,
            "_dart_undo_pending_bull_steals": None,
            "_dart_can_submit_since": None,
            "_dart_manual_advance_confirm": False,
            "_dart_no_empty_ref_warned": False,
            "_dart_wait_board_clear": False,
            "_dart_board_clear_frames": 0,
            "_last_cal_frames": {},
            "_cal_last_tick": 0.0,
            "_cal_buttons": [],
            "_cal_tiles": [],
            "bull_hints": {},
            "ellipse_hints": {},
            "_cal_click_bull_mode": False,
            "_cal_click_ellipse_mode": False,
            "_ingame_cal_open": False,
            "_ingame_cal_phase": "",
            "_ingame_cal_status": "",
            "_ingame_cal_stills": {},
            "_ingame_cal_frames": {},
            "_clear_board_status": "",
            "editing_hit_slot": None,
            "_dart_absorb_pending": False,
        }
        self._vision_lock = threading.RLock()
        self._bg_stop = threading.Event()
        self._bg_pending_lock = threading.Lock()
        self._bg_pending: Optional[Dict[str, Any]] = None
        self._cal_view_wh = (1280, 720)
        self._bg_cal_pack: Optional[Dict[str, Any]] = None
        self._bg_thread = threading.Thread(
            target=self._vision_loop,
            name="smartdarts-vision",
            daemon=True,
        )
        self._bg_thread.start()
        self._load_cal_hints()

    def start_qr_path(self) -> Optional[str]:
        path = start_qr_image_path(get_settings().device_id)
        return path if os.path.isfile(path) else None

    def set_on_change(self, cb: OnChange) -> None:
        self._on_change = cb

    def notify(self) -> None:
        if self._on_change is not None:
            self._on_change()

    @property
    def screen(self) -> str:
        return str(self.state.get("screen", "standby"))

    @property
    def game(self) -> Optional[Dict]:
        return self.ctx.get("game")

    def close_hit_editor(self) -> None:
        self.ctx["editing_hit_slot"] = None
        self.ctx["input_multiplier"] = 1

    def is_manual_mode(self) -> bool:
        try:
            return bool(get_settings().manual_mode)
        except Exception:
            return False

    def toggle_manual_mode(self) -> None:
        """Prebaci AUTO ↔ MANUAL tijekom igre (sprema u postavke)."""
        want = not self.is_manual_mode()
        update_settings(manual_mode=want)
        if want:
            dart_det = self.ctx.get("dart_detector")
            if dart_det is not None:
                with self._vision_lock:
                    dart_det.reset_motion_state()
            self.ensure_manual_hit_editor()
            print("[dart] nacin: MANUAL (tipkovnica)", flush=True)
            return
        self.close_hit_editor()
        self.ctx["_dart_block_auto_detect"] = False
        self.ctx["_dart_block_auto_advance"] = False
        self.ctx["_dart_score_locked"] = False
        self.resume_live_motion_after_pause()
        print("[dart] nacin: AUTO (kamere) — motion/diff ociscen", flush=True)

    def resume_live_motion_after_pause(self) -> None:
        """Isto kao MANUAL→AUTO: trenutna ploča = nova nula, stari diff se ne scorea."""
        if self.screen != "playing" or self.is_manual_mode():
            self.ctx["_dart_absorb_pending"] = False
            return
        g = self.ctx.get("game")
        if not g:
            self.ctx["_dart_absorb_pending"] = False
            return
        self.ctx["_dart_empty_frames"] = 0
        self.ctx["_dart_idle_frames"] = 0
        self.ctx["_dart_wait_idle_after_switch"] = True
        cam_mgr = self.ctx.get("camera_mgr")
        dart_det = self.ctx.get("dart_detector")
        board_cal = self.ctx.get("board_calibrator")
        if (
            cam_mgr is not None
            and dart_det is not None
            and board_cal is not None
            and cam_mgr.active
        ):
            with self._vision_lock:
                frames = cam_mgr.grab_frames(flush=1)
                dart_det.absorb_live_board_and_idle(board_cal, frames)
            self.ctx["_dart_absorb_pending"] = False
            print("[dart] povratak u igru — motion/diff ociscen", flush=True)
            return
        self.ctx["_dart_absorb_pending"] = True
        if dart_det is not None:
            with self._vision_lock:
                dart_det.reset_motion_state()

    def ensure_manual_hit_editor(self) -> None:
        """U ručnom načinu tipkovnica ostaje otvorena na prvom praznom slotu."""
        if not self.is_manual_mode() or self.screen != "playing":
            return
        g = self.ctx.get("game")
        if not g or g.get("winner_player_idx") is not None:
            return
        if self.ctx.get("killer_bull_pending"):
            return
        if self.ctx.get("editing_hit_slot") is not None:
            return
        turn_hits = g.get("turn_hits", [None, None, None])
        cur = int(self.state.get("current_player_idx", 0))
        for slot in range(3):
            if turn_hits[slot] is not None:
                continue
            if mm._turn_blocks_new_hits(g, cur, turn_hits):
                break
            self.open_hit_editor(slot)
            return

    def set_turn_hit(self, slot: int, hit: mm.DetectedHit) -> None:
        game = self.ctx.get("game") or {}
        turn_hits = game.get("turn_hits", [None, None, None])
        if 0 <= slot < 3:
            turn_hits[slot] = hit
            game["turn_hits"] = turn_hits
            self.all_hits.append(hit)
            cur = int(self.state.get("current_player_idx", 0))
            mm._killer_after_hit_registered(self.ctx, game, cur, slot, hit)

    def dart_hit_from_fused(self, fused: FusedDartResult) -> mm.DetectedHit:
        zone = str(fused.zone_name).lower()
        if zone == "inner_bull":
            return mm._make_bull_hit(True)
        if zone == "outer_bull":
            return mm._make_bull_hit(False)
        if zone == "miss" or int(fused.score) <= 0:
            return mm._make_miss_hit()
        mult = {"single": 1, "double": 2, "triple": 3}.get(zone, 1)
        return mm._make_number_hit(int(fused.segment_number), mult)

    def fused_hit_label(self, fused: FusedDartResult) -> str:
        zone = str(fused.zone_name).lower()
        if zone in ("inner_bull", "outer_bull"):
            return "B50" if zone == "inner_bull" else "B25"
        if zone == "miss":
            return "MISS"
        prefix = {"single": "S", "double": "D", "triple": "T"}.get(zone, "S")
        return f"{prefix}{fused.segment_number}"

    def capture_dart_reference(self) -> int:
        cam_mgr = self.ctx.get("camera_mgr")
        dart_det = self.ctx.get("dart_detector")
        board_cal = self.ctx.get("board_calibrator")
        if cam_mgr is None or dart_det is None or board_cal is None or not cam_mgr.active:
            print("[dart] REF: kamere nisu aktivne", flush=True)
            return 0
        frames = cam_mgr.grab_frames()
        saved = dart_det.capture_references(board_cal, frames)
        if saved > 0:
            self.ctx["_dart_status"] = f"REF {saved} cam"
        self.ctx["_dart_no_ref_warned"] = False
        print(f"[dart] REF: spremljeno {saved} kamera", flush=True)
        return saved

    def log_dart_motion(
        self,
        probe: List[DartDetectResult],
        *,
        fused: Optional[FusedDartResult] = None,
        state: str = "",
        force: bool = False,
    ) -> None:
        any_motion = any(r.motion_pixels >= MOTION_LOG_MIN_PIXELS for r in probe)
        now = time.perf_counter()
        last_log = float(self.ctx.get("_dart_log_last_tick", 0.0))
        if not force and not any_motion and fused is None and (now - last_log) < 5.0:
            return
        self.ctx["_dart_log_last_tick"] = now
        line = summarize_motion_log(probe)
        if state:
            line = f"[{state}] {line}"
        print(f"[dart] {line}", flush=True)
        if fused is not None:
            print(f"[dart] {summarize_fused_log(fused)}", flush=True)

    def run_calibration_detect(self) -> Tuple[bool, int]:
        cam_mgr = self.ctx.get("camera_mgr")
        board_cal = self.ctx.get("board_calibrator")
        frames = self.ctx.get("_last_cal_frames") or {}
        if cam_mgr is not None and cam_mgr.active:
            frames = cam_mgr.grab_frames(flush=2)
            self.ctx["_last_cal_frames"] = frames
        if board_cal is None or not frames:
            print("[dart] DETEKTIRAJ: nema kadrova", flush=True)
            self.ctx["_dart_status"] = "DET FAIL"
            return False, 0
        incomplete: List[int] = []
        # Max 3 pokušaja na 320x240 — softverski upscale je u calibrate_board.
        hints = dict(self.ctx.get("bull_hints") or {})
        ell_hints = dict(self.ctx.get("ellipse_hints") or {})
        for _attempt in range(3):
            if cam_mgr is not None and cam_mgr.active:
                frames = cam_mgr.grab_frames(flush=2)
                self.ctx["_last_cal_frames"] = frames
            incomplete = board_cal.detect_all(
                frames,
                bull_hints=hints or None,
                ellipse_hints=ell_hints or None,
            )
            if not incomplete:
                break
            time.sleep(0.08)
        n_ref = self.capture_dart_reference()
        all_ok = not incomplete and n_ref > 0
        if incomplete:
            detail = board_cal.format_incomplete_status(incomplete)
            self.ctx["_dart_status"] = f"KAL {len(CAMERA_INDICES) - len(incomplete)}/{len(CAMERA_INDICES)}"
            print(f"[dart] DETEKTIRAJ: kalibracija nepotpuna — {detail}", flush=True)
            no_signal = board_cal.incomplete_no_signal(incomplete)
            if no_signal:
                print(
                    f"[dart] DETEKTIRAJ: stvarno bez kadra (USB/open/read): {no_signal}",
                    flush=True,
                )
            print(
                "[dart] DETEKTIRAJ: nepotpuna — klikni BULL ili ELIPSA pa opet KALIBRIRAJ",
                flush=True,
            )
        else:
            self.ctx["_dart_status"] = f"REF {n_ref} cam" if n_ref > 0 else "REF FAIL"
        self.ctx["_cal_click_bull_mode"] = False
        self.ctx["_cal_click_ellipse_mode"] = False
        self._persist_cal_hints()
        if n_ref > 0:
            self.ctx["_dart_no_empty_ref_warned"] = False
        print(f"[dart] DETEKTIRAJ: kalibracija + ref ({n_ref} cam)", flush=True)
        return all_ok, n_ref

    def _load_cal_hints(self) -> None:
        board_cal = self.ctx.get("board_calibrator")
        if board_cal is None:
            return
        self.ctx["bull_hints"] = dict(board_cal.bull_hints)
        self.ctx["ellipse_hints"] = {
            int(k): list(v) for k, v in (board_cal.ellipse_hints or {}).items()
        }

    def _persist_cal_hints(self) -> None:
        board_cal = self.ctx.get("board_calibrator")
        if board_cal is None:
            return
        board_cal.sync_manual_hints(
            dict(self.ctx.get("bull_hints") or {}),
            dict(self.ctx.get("ellipse_hints") or {}),
        )

    def start_playing_after_clear_board(self) -> None:
        self.state["screen"] = "playing"
        self.ctx["_dart_turn_start_pending"] = True
        self.ctx["_dart_no_ref_warned"] = False
        self.ctx["_dart_block_auto_advance"] = False
        self.ctx["_dart_block_auto_detect"] = False
        self.ctx["_dart_undo_target_player"] = None
        self.ctx["_dart_undo_pending_hits"] = None
        self.ctx["_dart_undo_pending_bull_steals"] = None
        self.ctx["_dart_score_locked"] = False
        self.ctx["_dart_empty_frames"] = 0
        self.ctx["_dart_status"] = ""
        self.ctx["_clear_board_status"] = ""
        self.ctx["editing_hit_slot"] = None
        dart_det = self.ctx.get("dart_detector")
        if dart_det is not None:
            dart_det.reset_motion_state()
        self.ensure_manual_hit_editor()
        print("[dart] igra pokrenuta", flush=True)

    def sync_board_refs_after_undo(self) -> None:
        cam_mgr = self.ctx.get("camera_mgr")
        dart_det = self.ctx.get("dart_detector")
        board_cal = self.ctx.get("board_calibrator")
        if dart_det is None:
            return
        if cam_mgr is not None and board_cal is not None and cam_mgr.active:
            frames = cam_mgr.grab_frames()
            dart_det.commit_board_snapshot(board_cal, frames)
        dart_det.reset_motion_state()
        self.ctx["_dart_wait_idle_after_switch"] = False
        self.ctx["_dart_idle_frames"] = 0
        self.ctx["_dart_score_locked"] = False
        self.ctx["_dart_can_submit_since"] = None
        self.ctx["_dart_manual_advance_confirm"] = False

    def confirm_clear_board_and_start(self) -> None:
        if self.ctx.get("_clear_board_busy"):
            return
        self.ctx["_clear_board_busy"] = True
        self.ctx["_clear_board_pending_run"] = False
        self.ctx["_clear_board_pending_at"] = None
        try:
            with self._vision_lock:
                self._confirm_clear_board_and_start_impl()
        finally:
            self.ctx["_clear_board_busy"] = False

    def _confirm_clear_board_and_start_impl(self) -> None:
        cam_mgr = self.ctx.get("camera_mgr")
        dart_det = self.ctx.get("dart_detector")
        board_cal = self.ctx.get("board_calibrator")
        if cam_mgr is None or dart_det is None or board_cal is None or not cam_mgr.active:
            self.ctx["_clear_board_status"] = "Kamere nisu aktivne"
            return
        # Prvi klik često padne jer su kamere još u background openu.
        wait_msg = (
            get_settings().t("calibrating")
            if get_settings().auto_calibrate
            else get_settings().t("saving_ref")
        )
        self.ctx["_clear_board_status"] = wait_msg
        print("[dart] clear_board: cekam kamere...", flush=True)
        if not cam_mgr.wait_for_frames(timeout_sec=12.0, min_cams=len(CAMERA_INDICES)):
            if not cam_mgr.wait_for_frames(timeout_sec=0.5, min_cams=1):
                self.ctx["_clear_board_status"] = "Kamere se još otvaraju — pokušaj ponovo"
                print("[dart] clear_board: kamere nisu bile spremne", flush=True)
                return
            print(
                f"[dart] clear_board: djelomicno spremno "
                f"({cam_mgr.frames_ready_count()}/{len(CAMERA_INDICES)})",
                flush=True,
            )
        # Kratki warmup da buffer nije crn/stari.
        frames = cam_mgr.grab_frames(flush=2)
        for _ in range(3):
            time.sleep(0.05)
            frames = cam_mgr.grab_frames(flush=1)
        if get_settings().auto_calibrate:
            self.ctx["_clear_board_status"] = get_settings().t("calibrating")
            hints = dict(self.ctx.get("bull_hints") or {})
            incomplete, frames = board_cal.recalibrate_for_play(
                lambda flush=1: cam_mgr.grab_frames(flush=flush),
                attempts=4,
                bull_hints=hints or None,
                ellipse_hints=dict(self.ctx.get("ellipse_hints") or {}) or None,
            )
            if incomplete:
                # Još jedan prolaz ako je bio samo "nema signala".
                no_sig = board_cal.incomplete_no_signal(incomplete)
                if no_sig and len(no_sig) >= len(incomplete):
                    print("[dart] clear_board: retry nakon no_signal...", flush=True)
                    time.sleep(0.4)
                    cam_mgr.wait_for_frames(timeout_sec=4.0, min_cams=3)
                    incomplete, frames = board_cal.recalibrate_for_play(
                        lambda flush=1: cam_mgr.grab_frames(flush=flush),
                        attempts=3,
                        bull_hints=hints or None,
                        ellipse_hints=dict(self.ctx.get("ellipse_hints") or {}) or None,
                    )
            if incomplete:
                detail = board_cal.format_incomplete_status(incomplete)
                self.ctx["_clear_board_status"] = get_settings().t("cal_incomplete")
                print(f"[dart] clear_board: kalibracija nepotpuna — {detail}", flush=True)
                return
            self.ctx["_cal_click_bull_mode"] = False
            self.ctx["_cal_click_ellipse_mode"] = False
            print("[dart] clear_board: ploca ponovo kalibrirana (stable)", flush=True)

        n_ref = dart_det.update_empty_board_references(
            board_cal, frames, include_raw=True
        )
        if n_ref <= 0:
            self.ctx["_clear_board_status"] = get_settings().t(
                "no_saved_cal" if not get_settings().auto_calibrate else "ref_not_saved"
            )
            return
        dart_det.commit_board_snapshot(board_cal, frames)
        dart_det.reset_motion_state()
        self.ctx["_clear_board_status"] = ""
        print(f"[dart] prazna ploca ref ({n_ref} cam)", flush=True)
        self.start_playing_after_clear_board()

    def prepare_dart_turn_start(
        self, *, commit_snapshot: bool = True, skip_idle_wait: bool = False
    ) -> None:
        with self._vision_lock:
            self._prepare_dart_turn_start_impl(
                commit_snapshot=commit_snapshot, skip_idle_wait=skip_idle_wait
            )

    def _prepare_dart_turn_start_impl(
        self, *, commit_snapshot: bool = True, skip_idle_wait: bool = False
    ) -> None:
        cam_mgr = self.ctx.get("camera_mgr")
        dart_det = self.ctx.get("dart_detector")
        board_cal = self.ctx.get("board_calibrator")
        if (
            commit_snapshot
            and cam_mgr is not None
            and dart_det is not None
            and board_cal is not None
            and cam_mgr.active
        ):
            dart_det.commit_board_snapshot(board_cal, cam_mgr.grab_frames())
        if dart_det is not None:
            dart_det.reset_motion_state()
        self.ctx["_dart_wait_idle_after_switch"] = not skip_idle_wait
        self.ctx["_dart_idle_frames"] = 0
        self.ctx["_dart_empty_frames"] = 0
        self.ctx["_dart_score_locked"] = False
        self.ctx["_dart_last_tick"] = 0.0
        g = self.ctx.get("game")
        if g is not None:
            self.ctx["_turn_players_snap"] = copy.deepcopy(g.get("players") or [])
            self.ctx["_turn_elim_snap"] = copy.deepcopy(g.get("eliminated_order") or [])
            self.ctx["_turn_winner_snap"] = g.get("winner_player_idx")
            self.ctx["_turn_placements_snap"] = copy.deepcopy(g.get("placements") or [])
        self.ctx["killer_bull_pending"] = None
        self.ctx["killer_bull_steals"] = {}

    def reset_same_player_after_board_clear(self, *, skip_idle_wait: bool = True) -> None:
        """Clear dart slots and stay on current player (Killer assign retry)."""
        g = self.ctx.get("game")
        if not g:
            return
        g["turn_hits"] = [None, None, None]
        self.ctx["_dart_status"] = ""
        self.ctx["_dart_wait_board_clear"] = False
        self.ctx["_dart_board_clear_frames"] = 0
        self.ctx["_dart_can_submit_since"] = None
        self.ctx["_dart_manual_advance_confirm"] = False
        self.ctx["_dart_undo_target_player"] = None
        self.ctx["_dart_undo_pending_hits"] = None
        self.ctx["_dart_undo_pending_bull_steals"] = None
        self.ctx["_dart_block_auto_advance"] = False
        self.ctx["_dart_block_auto_detect"] = False
        self.ctx["_dart_empty_frames"] = 0
        self.close_hit_editor()
        self.ensure_manual_hit_editor()
        self.prepare_dart_turn_start(commit_snapshot=True, skip_idle_wait=skip_idle_wait)
        print("[dart] Killer assign — ploca prazna, isti igrac ponavlja", flush=True)

    def dart_scoring_paused(
        self, game: Dict, cur_idx: int, turn_hits: List[Optional[mm.DetectedHit]]
    ) -> bool:
        if self.ctx.get("_ingame_cal_open"):
            return True
        if self.ctx.get("killer_bull_pending"):
            return True
        if all(h is not None for h in turn_hits):
            return True
        if mm._turn_is_early_finish(game, cur_idx, turn_hits):
            return True
        if mm._x01_turn_is_bust(game, cur_idx, turn_hits):
            return True
        return False

    def confirm_undo_advance_to_next(self) -> None:
        g = self.ctx.get("game")
        target = self.ctx.get("_dart_undo_target_player")
        if g is None or target is None:
            return
        pending = list(self.ctx.get("_dart_undo_pending_hits") or [None, None, None])
        pending_steals_raw = self.ctx.get("_dart_undo_pending_bull_steals") or {}
        pending_steals = (
            {int(k): int(v) for k, v in pending_steals_raw.items()}
            if isinstance(pending_steals_raw, dict)
            else {}
        )
        finished_idx = int(self.state.get("current_player_idx", 0))
        finished_hits = list(g.get("turn_hits", [None, None, None]))
        cam_mgr = self.ctx.get("camera_mgr")
        dart_det = self.ctx.get("dart_detector")
        board_cal = self.ctx.get("board_calibrator")
        # Stanje za undo snimku = turn-start baseline (ne mid-turn mutacije)
        players_before = copy.deepcopy(
            self.ctx.get("_turn_players_snap") or g.get("players") or []
        )
        round_before = int(g.get("round_number", 1))
        winner_before = self.ctx.get("_turn_winner_snap", g.get("winner_player_idx"))
        winner_at_before = g.get("winner_at")
        placements_before = copy.deepcopy(
            self.ctx.get("_turn_placements_snap") or g.get("placements") or []
        )
        elim_before = copy.deepcopy(
            self.ctx.get("_turn_elim_snap") or g.get("eliminated_order") or []
        )
        killer_phase_before = g.get("killer_phase")
        if str(g.get("mode", "")).lower() == "killer":
            mm._killer_commit_turn(self.ctx, g, finished_idx, finished_hits)
        else:
            # Restore pre-edit scores then apply (players may still be at baseline)
            if isinstance(self.ctx.get("_turn_players_snap"), list):
                g["players"] = copy.deepcopy(self.ctx["_turn_players_snap"])
            mm._apply_turn_score(g, finished_idx, finished_hits)
        if g.get("winner_player_idx") is not None and g.get("winner_at") is None:
            g["winner_at"] = time.time()
        nxt = int(target)
        if g.get("winner_player_idx") is None:
            self.state["current_player_idx"] = nxt
            mm._advance_round_if_needed(g, finished_idx, nxt)
            g["turn_hits"] = pending
            mm._push_undo_after_advance(
                g,
                finished_idx,
                finished_hits,
                nxt,
                players_before=players_before,
                round_before=round_before,
                winner_before=winner_before,
                winner_at_before=winner_at_before,
                placements_before=placements_before,
                eliminated_order_before=elim_before,
                killer_phase_before=killer_phase_before,
            )
        else:
            # Edited turn ended the game — drop interrupted next-player hits
            g["turn_hits"] = [None, None, None]
            g["undo_stack"] = []
        self.ctx["_dart_undo_target_player"] = None
        self.ctx["_dart_undo_pending_hits"] = None
        self.ctx["_dart_undo_pending_bull_steals"] = None
        self.ctx["_dart_block_auto_advance"] = False
        self.ctx["_dart_block_auto_detect"] = False
        self.ctx["_dart_can_submit_since"] = None
        self.ctx["_dart_manual_advance_confirm"] = False
        self.ctx["_dart_wait_board_clear"] = False
        self.ctx["_dart_board_clear_frames"] = 0
        self.ctx["_dart_empty_frames"] = 0
        self.close_hit_editor()
        if dart_det is not None:
            dart_det.reset_motion_state()
        if cam_mgr is not None and dart_det is not None and board_cal is not None and cam_mgr.active:
            frames = cam_mgr.grab_frames()
            dart_det.commit_board_snapshot(board_cal, frames)
        # Baseline snap = post-commit players; then re-apply interrupted pending hits
        # so Killer lives / is_killer match the restored turn_hits immediately.
        self.prepare_dart_turn_start(commit_snapshot=False, skip_idle_wait=True)
        if g.get("winner_player_idx") is None and mm._killer_is_play_phase(g):
            if pending_steals:
                self.ctx["killer_bull_steals"] = dict(pending_steals)
            mm._killer_resync_mid_turn(
                self.ctx, g, int(nxt), list(g.get("turn_hits") or [None, None, None])
            )
        print(f"[dart] P{nxt + 1} — preuzima detektirane strelice (score azuriran)", flush=True)

    def advance_to_next_player(
        self, *, auto: bool = False, skip_idle_wait: bool = False, commit_snapshot: bool = True
    ) -> None:
        g = self.ctx.get("game")
        if not g:
            return
        turn_hits = g.get("turn_hits", [None, None, None])
        cur = int(self.state.get("current_player_idx", 0))
        # Assign: never advance without a successful unique claim.
        if (
            str(g.get("mode", "")).lower() == "killer"
            and str(g.get("killer_phase", "play")).lower() == "assign"
            and not mm._killer_number_claimed_in_turn(g, cur, turn_hits)
        ):
            if mm._turn_fully_entered(turn_hits):
                self.reset_same_player_after_board_clear(skip_idle_wait=skip_idle_wait)
            return
        final_hits = list(turn_hits)
        players_before = copy.deepcopy(
            self.ctx.get("_turn_players_snap") or g.get("players") or []
        )
        round_before = int(g.get("round_number", 1))
        winner_before = self.ctx.get("_turn_winner_snap", g.get("winner_player_idx"))
        winner_at_before = g.get("winner_at")
        placements_before = copy.deepcopy(
            self.ctx.get("_turn_placements_snap") or g.get("placements") or []
        )
        elim_before = copy.deepcopy(
            self.ctx.get("_turn_elim_snap") or g.get("eliminated_order") or []
        )
        killer_phase_before = g.get("killer_phase")
        if str(g.get("mode", "")).lower() == "killer":
            mm._killer_commit_turn(self.ctx, g, cur, final_hits)
        else:
            mm._apply_turn_score(g, cur, final_hits)
        if g.get("winner_player_idx") is not None and g.get("winner_at") is None:
            g["winner_at"] = time.time()
        nxt_for_undo = None
        if g.get("winner_player_idx") is None:
            nxt_for_undo = mm._next_active_player_idx(g, cur)
            self.state["current_player_idx"] = nxt_for_undo
            mm._advance_round_if_needed(g, cur, nxt_for_undo)
        g["turn_hits"] = [None, None, None]
        self.ctx["_dart_status"] = ""
        self.ctx["_dart_wait_board_clear"] = False
        self.ctx["_dart_board_clear_frames"] = 0
        self.ctx["_dart_can_submit_since"] = None
        self.ctx["_dart_manual_advance_confirm"] = False
        self.ctx["_dart_undo_target_player"] = None
        self.ctx["_dart_undo_pending_hits"] = None
        self.ctx["_dart_undo_pending_bull_steals"] = None
        self.close_hit_editor()
        self.ensure_manual_hit_editor()
        if not auto:
            self.ctx["_dart_block_auto_advance"] = False
            self.ctx["_dart_block_auto_detect"] = False
        if nxt_for_undo is not None:
            mm._push_undo_after_advance(
                g,
                cur,
                final_hits,
                nxt_for_undo,
                players_before=players_before,
                round_before=round_before,
                winner_before=winner_before,
                winner_at_before=winner_at_before,
                placements_before=placements_before,
                eliminated_order_before=elim_before,
                killer_phase_before=killer_phase_before,
            )
        self.prepare_dart_turn_start(commit_snapshot=commit_snapshot, skip_idle_wait=skip_idle_wait)
        if auto:
            print("[dart] AUTO IDUCI IGRAC: ploca prazna", flush=True)
        else:
            print("[dart] IDUCI IGRAC (preskoci)", flush=True)

    def force_advance_past_board_clear(self, *, update_empty_ref: bool = False) -> None:
        g = self.ctx.get("game")
        if not g:
            return
        turn_hits = g.get("turn_hits", [None, None, None])
        cur = int(self.state.get("current_player_idx", 0))
        needs_retry = mm._killer_assign_needs_retry(g, cur, turn_hits)
        if not needs_retry and not mm._turn_can_submit(g, cur, turn_hits):
            return
        cam_mgr = self.ctx.get("camera_mgr")
        dart_det = self.ctx.get("dart_detector")
        board_cal = self.ctx.get("board_calibrator")
        if dart_det is not None:
            dart_det.reset_motion_state()
        self.ctx["_dart_wait_board_clear"] = False
        self.ctx["_dart_board_clear_frames"] = 0
        self.ctx["_dart_empty_frames"] = 0
        self.ctx["_dart_block_auto_advance"] = False
        self.ctx["_dart_block_auto_detect"] = False
        self.ctx["_dart_manual_advance_confirm"] = False
        refresh_empty = bool(update_empty_ref) and not self.is_manual_mode()
        if cam_mgr is not None and dart_det is not None and board_cal is not None and cam_mgr.active:
            with self._vision_lock:
                frames = cam_mgr.grab_frames()
                if refresh_empty:
                    n_empty = dart_det.update_empty_board_references(board_cal, frames)
                    dart_det.commit_board_snapshot(board_cal, frames)
                    dart_det.reset_motion_state()
                    print(
                        f"[dart] rucno — empty ref azuriran ({n_empty} cam) + motion ref",
                        flush=True,
                    )
                else:
                    dart_det.commit_board_snapshot(board_cal, frames)
                    dart_det.reset_motion_state()
                    print("[dart] rucno — motion ref azuriran (empty ref netaknut)", flush=True)
        if needs_retry:
            self.reset_same_player_after_board_clear(skip_idle_wait=True)
        else:
            self.advance_to_next_player(auto=True, skip_idle_wait=True)

    def try_auto_empty_board_advance(self) -> bool:
        with self._vision_lock:
            result = self._empty_advance_vision()
            if not result:
                return False
            return self._apply_empty_advance_result(result)

    def _empty_advance_vision(self) -> Optional[Dict[str, Any]]:
        if self.is_manual_mode():
            return None
        g = self.ctx.get("game")
        if self.ctx.get("_dart_block_auto_advance"):
            self.ctx["_dart_empty_frames"] = 0
            return None
        if not g:
            self.ctx["_dart_empty_frames"] = 0
            return None
        if g.get("winner_player_idx") is not None and not mm._winner_pending_board_clear(g):
            self.ctx["_dart_empty_frames"] = 0
            return None
        turn_hits = g.get("turn_hits", [None, None, None])
        cur = int(self.state.get("current_player_idx", 0))
        needs_retry = mm._killer_assign_needs_retry(g, cur, turn_hits)
        if not needs_retry and not mm._turn_can_submit(g, cur, turn_hits):
            self.ctx["_dart_empty_frames"] = 0
            return None
        cam_mgr = self.ctx.get("camera_mgr")
        dart_det = self.ctx.get("dart_detector")
        board_cal = self.ctx.get("board_calibrator")
        if cam_mgr is None or dart_det is None or board_cal is None or not cam_mgr.active:
            return None
        if not dart_det.has_empty_references():
            if not self.ctx.get("_dart_no_empty_ref_warned"):
                print(
                    "[dart] auto-prebacivanje: nema prazne reference — DETEKTIRAJ na praznoj ploci",
                    flush=True,
                )
                self.ctx["_dart_no_empty_ref_warned"] = True
            return None
        frames = cam_mgr.grab_frames()
        empty_ok, max_px, max_diff = dart_det.is_board_like_empty(board_cal, frames)
        now = time.perf_counter()
        last_empty_log = float(self.ctx.get("_dart_empty_log_last", 0.0))
        if now - last_empty_log >= 2.0:
            self.ctx["_dart_empty_log_last"] = now
            print(
                f"[dart] empty probe px={max_px} td={int(getattr(dart_det, '_last_empty_td_px', 0))} "
                f"raw={int(getattr(dart_det, '_last_empty_raw_px', 0))} diff={max_diff:.2f} "
                f"(prag td<={EMPTY_BOARD_MAX_PIXELS} raw_hand>={EMPTY_BOARD_RAW_HAND_PIXELS} "
                f"diff<={EMPTY_BOARD_MAX_DIFF_MEAN})",
                flush=True,
            )
        if empty_ok:
            self.ctx["_dart_empty_frames"] = int(self.ctx.get("_dart_empty_frames", 0)) + 1
        else:
            cur_n = int(self.ctx.get("_dart_empty_frames", 0))
            self.ctx["_dart_empty_frames"] = max(0, cur_n - int(mm.EMPTY_BOARD_FAIL_PENALTY))
        if self.ctx["_dart_empty_frames"] < mm.EMPTY_BOARD_CONFIRM_FRAMES:
            return None
        self.ctx["_dart_empty_frames"] = 0
        dart_det.reset_motion_state()
        n_empty = dart_det.update_empty_board_references(board_cal, frames)
        dart_det.commit_board_snapshot(board_cal, frames, persist=False)
        print(
            f"[dart] auto-prebacivanje — empty ref azuriran ({n_empty} cam) + motion ref",
            flush=True,
        )
        return {"kind": "empty_advance", "needs_retry": bool(needs_retry)}

    def _apply_empty_advance_result(self, result: Dict[str, Any]) -> bool:
        if result.get("needs_retry"):
            self.reset_same_player_after_board_clear(skip_idle_wait=True)
        else:
            self.advance_to_next_player(
                auto=True, skip_idle_wait=True, commit_snapshot=False
            )
        return True

    def dart_should_enter_rearm(
        self, game: Dict, cur_idx: int, turn_hits: List[Optional[mm.DetectedHit]]
    ) -> bool:
        if self.ctx.get("_dart_block_auto_advance"):
            return False
        if not mm._turn_fully_entered(turn_hits):
            return False
        return mm._turn_can_submit(game, cur_idx, turn_hits) or mm._killer_assign_needs_retry(
            game, cur_idx, turn_hits
        )

    def apply_detected_hit(
        self, slot: int, hit: mm.DetectedHit, *, undo_mode: bool = False
    ) -> None:
        if undo_mode:
            pending = list(self.ctx.get("_dart_undo_pending_hits") or [None, None, None])
            if 0 <= slot < 3:
                pending[slot] = hit
                self.ctx["_dart_undo_pending_hits"] = pending
                self.all_hits.append(hit)
            return
        self.set_turn_hit(slot, hit)

    def try_auto_detect_dart(self, *, force_log: bool = False, undo_mode: bool = False) -> bool:
        """Sinhrono (hotkey D) — vision nit koristi _auto_detect_vision + queue."""
        with self._vision_lock:
            result = self._auto_detect_vision(force_log=force_log, undo_mode=undo_mode)
            if not result:
                return False
            return self._apply_detect_result(result)

    def _auto_detect_vision(
        self, *, force_log: bool = False, undo_mode: bool = False
    ) -> Optional[Dict[str, Any]]:
        g = self.ctx.get("game")
        if not g or g.get("winner_player_idx") is not None:
            return None
        if self.is_manual_mode() and not undo_mode:
            return None
        if undo_mode:
            if not self.ctx.get("_dart_block_auto_advance"):
                return None
        elif self.ctx.get("_dart_block_auto_detect"):
            return None
        turn_hits = g.get("turn_hits", [None, None, None])
        cur = int(self.state.get("current_player_idx", 0))
        if undo_mode:
            hit_slots = list(self.ctx.get("_dart_undo_pending_hits") or [None, None, None])
        else:
            hit_slots = turn_hits
            if self.dart_scoring_paused(g, cur, turn_hits):
                return None
        slots_full = all(h is not None for h in hit_slots)
        if undo_mode and slots_full:
            return None
        cam_mgr = self.ctx.get("camera_mgr")
        dart_det = self.ctx.get("dart_detector")
        board_cal = self.ctx.get("board_calibrator")
        if cam_mgr is None or dart_det is None or board_cal is None or not cam_mgr.active:
            if force_log:
                print("[dart] AUTO: kamere nisu aktivne", flush=True)
            return None
        if not dart_det.has_references():
            if force_log:
                print("[dart] AUTO: nema reference (DETEKTIRAJ u kalibraciji)", flush=True)
            return None
        wait_idle = False if undo_mode else bool(self.ctx.get("_dart_wait_idle_after_switch", False))
        frames = cam_mgr.grab_frames()
        probe, fused, state = dart_det.tick_and_detect(
            board_cal,
            frames,
            save_debug=not undo_mode,
            accept_fire=not slots_full and not wait_idle,
        )
        if wait_idle:
            max_px = max((r.motion_pixels for r in probe), default=0)
            if state == "idle" and max_px < MOTION_CLEAR_PIXELS:
                self.ctx["_dart_idle_frames"] = int(self.ctx.get("_dart_idle_frames", 0)) + 1
                if self.ctx["_dart_idle_frames"] >= mm.DART_IDLE_FRAMES_AFTER_SWITCH:
                    self.ctx["_dart_wait_idle_after_switch"] = False
                    self.ctx["_dart_idle_frames"] = 0
                    print("[dart] spreman za novu strijelu", flush=True)
            else:
                self.ctx["_dart_idle_frames"] = 0
        if force_log or state.startswith("armed") or state in ("hand_block", "hand_cooldown") or fused is not None:
            self.log_dart_motion(probe, fused=fused, state=state, force=force_log)
            if state in ("hand_block", "hand_cooldown"):
                self.ctx["_dart_status"] = "CEKAJ RUKA"
        elif any(r.motion_pixels >= MOTION_LOG_MIN_PIXELS for r in probe):
            self.log_dart_motion(probe, state=state)

        dart_det.try_commit_deferred_snapshot(board_cal, frames)

        if fused is None:
            if state == "fired_fail":
                fire_frames = dart_det.last_fire_frames()
                commit_frames = fire_frames if len(fire_frames) >= 2 else cam_mgr.grab_frames()
                dart_det.commit_board_snapshot(board_cal, commit_frames, persist=False)
                dart_det.finalize_shot(enter_rearm=False)
                print("[dart] AUTO: fusion fail — ignore (noise)", flush=True)
                return None
            if state == "post_fire":
                dart_det.abort_shot()
            return None

        is_miss = str(fused.zone_name).lower() == "miss" or int(fused.score) <= 0
        low_conf = not is_miss and float(fused.confidence) < mm.DART_MIN_CONFIDENCE
        if low_conf:
            fire_frames = dart_det.last_fire_frames()
            commit_frames = fire_frames if len(fire_frames) >= 2 else cam_mgr.grab_frames()
            dart_det.commit_board_snapshot(board_cal, commit_frames, persist=False)
            dart_det.abort_shot()
            if force_log:
                print(
                    f"[dart] AUTO: odbijeno conf={fused.confidence:.2f} < {mm.DART_MIN_CONFIDENCE}",
                    flush=True,
                )
            return None

        slot = mm._next_insert_slot(hit_slots)
        if slot >= 3:
            dart_det.abort_shot()
            return None
        return {
            "kind": "hit",
            "fused": fused,
            "slot": slot,
            "is_miss": is_miss,
            "undo_mode": undo_mode,
        }

    def _apply_detect_result(self, result: Dict[str, Any]) -> bool:
        fused = result.get("fused")
        slot = int(result.get("slot", 3))
        undo_mode = bool(result.get("undo_mode"))
        is_miss = bool(result.get("is_miss"))
        dart_det = self.ctx.get("dart_detector")
        if fused is None or slot >= 3 or dart_det is None:
            return False
        if is_miss:
            self.apply_detected_hit(slot, mm._make_miss_hit(), undo_mode=undo_mode)
            label = "MISS"
        else:
            self.apply_detected_hit(slot, self.dart_hit_from_fused(fused), undo_mode=undo_mode)
            label = self.fused_hit_label(fused)
        cams = ",".join(str(c) for c in fused.cam_indices)
        self.ctx["_dart_status"] = f"AUTO {label} [{cams}]"
        tip_txt = ""
        if fused.tip_xy is not None:
            tip_txt = f" tip=({fused.tip_xy[0]:.1f},{fused.tip_xy[1]:.1f})"
        tag = "UNDO " if undo_mode else ""
        print(
            f"[dart] {tag}HIT {label} cams=[{cams}] conf={fused.confidence:.2f}{tip_txt}",
            flush=True,
        )
        dart_det.finalize_shot(enter_rearm=False)
        dart_det.mark_commit_after_ui()
        return True

    def cycle_fake_dart_hit(self) -> bool:
        """J (fakecams): sljedeći hit u cijeloj fake_darts playlisti (wrap)."""
        cam_mgr = self.ctx.get("camera_mgr")
        if cam_mgr is None or not cam_mgr.fake_mode:
            return False
        if not cam_mgr.active:
            cam_mgr.start()
        return cam_mgr.cycle_fake_hit() is not None

    def fake_cams_advance_next_player(self) -> bool:
        """L (fakecams): isto kao UI '>' — manual_advance_confirm (next player / end turn)."""
        cam_mgr = self.ctx.get("camera_mgr")
        if cam_mgr is None or not cam_mgr.fake_mode:
            return False
        if self.state.get("screen") != "playing":
            return False
        print("[fakecams] L -> next player", flush=True)
        self.handle_action("manual_advance_confirm")
        return True

    def is_in_game_calibrating(self) -> bool:
        return bool(self.ctx.get("_ingame_cal_open"))

    def open_in_game_calibrate(self) -> None:
        if self.screen != "playing" or self.ctx.get("_ingame_cal_open"):
            return
        g = self.ctx.get("game")
        if g and g.get("winner_player_idx") is not None:
            return
        if self.ctx.get("killer_bull_pending"):
            return
        self.ctx["_ingame_cal_open"] = True
        self.ctx["_ingame_cal_phase"] = "prompt"
        self.ctx["_ingame_cal_status"] = ""
        self.ctx["_ingame_cal_stills"] = {}
        self.ctx["_ingame_cal_frames"] = {}
        print("[dart] in-game cal: pauza", flush=True)

    def close_in_game_calibrate(self) -> None:
        was_open = bool(self.ctx.get("_ingame_cal_open"))
        self.ctx["_ingame_cal_open"] = False
        self.ctx["_ingame_cal_phase"] = ""
        self.ctx["_ingame_cal_status"] = ""
        self.ctx["_ingame_cal_stills"] = {}
        self.ctx["_ingame_cal_frames"] = {}
        if was_open:
            print("[dart] in-game cal: nastavak igre", flush=True)
            self.resume_live_motion_after_pause()

    def finish_in_game_calibrate(self) -> None:
        """Gotovo: spremi kalibraciju + novi empty-board ref, nastavi igru."""
        with self._vision_lock:
            cam_mgr = self.ctx.get("camera_mgr")
            dart_det = self.ctx.get("dart_detector")
            board_cal = self.ctx.get("board_calibrator")
            frames = self.ctx.get("_ingame_cal_frames") or {}
            if dart_det is not None and board_cal is not None and frames:
                n_ref = dart_det.update_empty_board_references(
                    board_cal, frames, include_raw=True
                )
                dart_det.commit_board_snapshot(board_cal, frames)
                dart_det.reset_motion_state()
                print(f"[dart] in-game cal: empty ref ({n_ref} cam)", flush=True)
            elif dart_det is not None:
                dart_det.reset_motion_state()
            if board_cal is not None:
                board_cal.save()
            if cam_mgr is not None and cam_mgr.active and dart_det is not None and board_cal is not None:
                if not frames:
                    live = cam_mgr.grab_frames(flush=1)
                    dart_det.commit_board_snapshot(board_cal, live)
        self.close_in_game_calibrate()

    def run_in_game_calibrate(self) -> bool:
        """Kalibriraj s prazne ploče i spremi still-slike s overlayem (nije live)."""
        self.ctx["_ingame_cal_phase"] = "busy"
        self.ctx["_ingame_cal_status"] = get_settings().t("calibrating")
        with self._vision_lock:
            return self._run_in_game_calibrate_impl()

    def _run_in_game_calibrate_impl(self) -> bool:
        cam_mgr = self.ctx.get("camera_mgr")
        board_cal = self.ctx.get("board_calibrator")
        if cam_mgr is None or board_cal is None or not cam_mgr.active:
            self.ctx["_ingame_cal_status"] = "Kamere nisu aktivne"
            self.ctx["_ingame_cal_phase"] = "prompt"
            return False
        print("[dart] in-game cal: cekam kamere...", flush=True)
        if not cam_mgr.wait_for_frames(timeout_sec=12.0, min_cams=len(CAMERA_INDICES)):
            if not cam_mgr.wait_for_frames(timeout_sec=0.5, min_cams=1):
                self.ctx["_ingame_cal_status"] = "Kamere se još otvaraju — pokušaj ponovo"
                self.ctx["_ingame_cal_phase"] = "prompt"
                print("[dart] in-game cal: kamere nisu spremne", flush=True)
                return False
        frames = cam_mgr.grab_frames(flush=2)
        for _ in range(3):
            time.sleep(0.05)
            frames = cam_mgr.grab_frames(flush=1)
        hints = dict(self.ctx.get("bull_hints") or {})
        ell_hints = dict(self.ctx.get("ellipse_hints") or {}) or None
        incomplete, frames = board_cal.recalibrate_for_play(
            lambda flush=1: cam_mgr.grab_frames(flush=flush),
            attempts=4,
            bull_hints=hints or None,
            ellipse_hints=ell_hints,
        )
        if incomplete:
            no_sig = board_cal.incomplete_no_signal(incomplete)
            if no_sig and len(no_sig) >= len(incomplete):
                print("[dart] in-game cal: retry nakon no_signal...", flush=True)
                time.sleep(0.4)
                cam_mgr.wait_for_frames(timeout_sec=4.0, min_cams=len(CAMERA_INDICES))
                incomplete, frames = board_cal.recalibrate_for_play(
                    lambda flush=1: cam_mgr.grab_frames(flush=flush),
                    attempts=3,
                    bull_hints=hints or None,
                    ellipse_hints=ell_hints,
                )
        old_ov = bool(board_cal.show_overlay)
        old_td = bool(board_cal.show_topdown)
        board_cal.show_overlay = True
        board_cal.show_topdown = False
        try:
            rendered = board_cal.render_frames(frames)
        finally:
            board_cal.show_overlay = old_ov
            board_cal.show_topdown = old_td
        stills: Dict[int, Optional[np.ndarray]] = {}
        for idx in CAMERA_INDICES:
            fr = rendered.get(idx) if isinstance(rendered, dict) else None
            stills[int(idx)] = None if fr is None else fr.copy()
        self.ctx["_ingame_cal_stills"] = stills
        self.ctx["_ingame_cal_frames"] = dict(frames or {})
        if incomplete:
            detail = board_cal.format_incomplete_status(incomplete)
            self.ctx["_ingame_cal_status"] = get_settings().t("cal_incomplete")
            print(f"[dart] in-game cal: nepotpuna — {detail}", flush=True)
        else:
            self.ctx["_cal_click_bull_mode"] = False
            self.ctx["_cal_click_ellipse_mode"] = False
            self.ctx["_ingame_cal_status"] = board_cal.status_summary() or ""
            print("[dart] in-game cal: ploca kalibrirana", flush=True)
        self.ctx["_ingame_cal_phase"] = "result"
        return not bool(incomplete)

    def enter_calibration(self) -> None:
        if self.state.get("screen") == "calibration":
            return
        self.state["_prev_screen"] = self.state.get("screen", "standby")
        cam_mgr = self.ctx.get("camera_mgr")
        if cam_mgr is not None:
            cam_mgr.start()
            # Pričekaj feedove pa tek onda prikaži ekran (bez NEMA SIGNALA flasha).
            cam_mgr.wait_for_frames(timeout_sec=8.0, min_cams=len(CAMERA_INDICES))
        self.state["screen"] = "calibration"
        self.ctx["_cal_click_bull_mode"] = False
        self.ctx["_cal_click_ellipse_mode"] = False
        self._load_cal_hints()
        board_cal = self.ctx.get("board_calibrator")
        if board_cal is not None:
            board_cal.show_overlay = True

    def exit_calibration(self) -> None:
        if self.state.get("screen") != "calibration":
            return
        cam_mgr = self.ctx.get("camera_mgr")
        if cam_mgr is not None:
            cam_mgr.stop()
        self.ctx["_last_cal_frames"] = {}
        self.ctx["_cal_click_bull_mode"] = False
        self.ctx["_cal_click_ellipse_mode"] = False
        board_cal = self.ctx.get("board_calibrator")
        if board_cal is not None:
            board_cal.save()
        prev = self.state.get("_prev_screen") or "standby"
        self.state["screen"] = prev
        self.state["_prev_screen"] = None
        if prev == "playing":
            self.resume_live_motion_after_pause()

    def _cal_topdown_on(self) -> bool:
        board_cal = self.ctx.get("board_calibrator")
        return bool(board_cal is not None and getattr(board_cal, "show_topdown", False))

    def clear_bull_hints(self) -> None:
        self.ctx["bull_hints"] = {}
        self.ctx["_cal_click_bull_mode"] = False
        self.ctx["_cal_last_tick"] = 0.0
        self._persist_cal_hints()

    def toggle_cal_click_bull_mode(self) -> None:
        if self._cal_topdown_on():
            return
        on = not bool(self.ctx.get("_cal_click_bull_mode"))
        self.ctx["_cal_click_bull_mode"] = on
        if on:
            self.ctx["_cal_click_ellipse_mode"] = False
        self.ctx["_cal_last_tick"] = 0.0
        print(
            f"[dart] BULL: {'ON — klikni otprilike na bull po kameri' if on else 'OFF'}",
            flush=True,
        )

    def toggle_cal_click_ellipse_mode(self) -> None:
        if self._cal_topdown_on():
            return
        on = not bool(self.ctx.get("_cal_click_ellipse_mode"))
        self.ctx["_cal_click_ellipse_mode"] = on
        if on:
            self.ctx["_cal_click_bull_mode"] = False
        self.ctx["_cal_last_tick"] = 0.0
        print(
            f"[dart] ELIPSA: {'ON — klikni do 4 tocke na vanjskom rubu' if on else 'OFF'}",
            flush=True,
        )

    def enter_settings(self) -> None:
        if self.state.get("screen") == "settings":
            return
        # Iz kalibracije ne idi u postavke dok kamere rade — zatvori prvo.
        if self.state.get("screen") == "calibration":
            self.exit_calibration()
        # Zapamti ekran za povratak (ne diraj kad se vraćamo iz kalibracije u postavke).
        if not self.state.get("_settings_return_screen"):
            prev = self.state.get("screen", "standby")
            if prev not in ("settings", "calibration"):
                self.state["_settings_return_screen"] = prev
            else:
                self.state["_settings_return_screen"] = self._home_screen()
        self.state["screen"] = "settings"
        self.settings = get_settings()

    def exit_settings(self) -> None:
        if self.state.get("screen") != "settings":
            return
        prev = self.state.get("_settings_return_screen") or "standby"
        self.state["_settings_return_screen"] = None
        if prev in ("settings", "calibration"):
            prev = self._home_screen()
        self.state["screen"] = prev
        if prev == "playing":
            self.resume_live_motion_after_pause()

    def _home_screen(self) -> str:
        mode = get_settings().start_mode
        if mode == "always_on":
            return "select_game"
        return "standby"

    def _apply_start_mode_home(self) -> None:
        """Ako smo na standby/select_game, uskladi s načinom pokretanja."""
        if self.state.get("screen") in ("standby", "select_game") and self.ctx.get("game") is None:
            self.state["screen"] = self._home_screen()

    def handle_nav_back(self) -> None:
        screen = self.state.get("screen", "standby")
        mode = str(self.state.get("selected_game_mode") or "")
        if screen == "settings":
            self.exit_settings()
            return
        if screen == "select_game":
            self.state["screen"] = self._home_screen()
            if self.state["screen"] == "select_game":
                # always_on: nazad s select_game ostaje tu (nema standby).
                pass
        elif screen == "rules_x01":
            self.state["screen"] = "select_game"
        elif screen == "rules_cricket":
            self.state["screen"] = "select_game"
        elif screen == "rules_killer":
            self.state["screen"] = "select_game"
        elif screen == "rules_around":
            self.state["screen"] = "select_game"
        elif screen == "rules_halve":
            self.state["screen"] = "select_game"
        elif screen == "tutorial_x01":
            self.state["screen"] = "rules_x01"
        elif screen == "tutorial_cricket":
            self.state["screen"] = "rules_cricket"
        elif screen == "tutorial_killer":
            self.state["screen"] = "rules_killer"
        elif screen == "tutorial_around":
            self.state["screen"] = "rules_around"
        elif screen == "tutorial_halve":
            self.state["screen"] = "rules_halve"
        elif screen == "select_players":
            if mode in ("301", "501"):
                self.state["screen"] = "rules_x01"
            elif mode == "cricket":
                self.state["screen"] = "rules_cricket"
            elif mode == "killer":
                self.state["screen"] = "rules_killer"
            elif mode == "around":
                self.state["screen"] = "rules_around"
            elif mode in ("halve", "halve_it"):
                self.state["screen"] = "rules_halve"
            else:
                self.state["screen"] = "select_game"
        elif screen == "clear_board":
            self.state["screen"] = "select_players"
            self.ctx["game"] = None
            self.ctx["_clear_board_status"] = ""
            self.all_hits.clear()

    def clear_to_standby(self, *, mqtt_off: bool = False) -> None:
        self.state["screen"] = self._home_screen()
        self.state["selected_game_mode"] = None
        self.state["num_players"] = 0
        self.state["current_player_idx"] = 0
        self.ctx["game"] = None
        self.all_hits.clear()
        self.close_hit_editor()
        self.close_in_game_calibrate()
        self.state["player_slots"] = [None, None, None, None]
        # always_on: nikad ne šalji MQTT (ni off pri izlazu).
        if mqtt_off and get_settings().start_mode != "always_on":
            try:
                mm.send_mqtt_off()
            except Exception:
                pass

    def mqtt_wake(self) -> None:
        self.state["screen"] = "select_game"
        self.state["selected_game_mode"] = None
        self.state["num_players"] = 0
        self.state["current_player_idx"] = 0
        self.ctx["game"] = None
        self.all_hits.clear()
        self.state["player_slots"] = [None, None, None, None]

    def start_from_button(self) -> None:
        """Gumb POKRENI na standby ekranu — lokalni start + MQTT 'on' za LED ring."""
        if get_settings().start_mode != "always_on":
            try:
                mm.send_mqtt_on()
            except Exception:
                pass
        self.mqtt_wake()

    @staticmethod
    def sanitize_player_name(raw: str) -> str:
        s = str(raw or "").strip().upper()
        if s.startswith("PLAYER"):
            rest = s[6:].strip()
            if rest.isdigit() and 1 <= int(rest) <= 4:
                return f"PLAYER {int(rest)}"
        return "".join(ch for ch in s if ch.isalpha())[:8]

    def player_slots(self) -> List[Optional[Dict[str, Any]]]:
        slots = self.state.get("player_slots")
        if not isinstance(slots, list) or len(slots) != 4:
            slots = [None, None, None, None]
            self.state["player_slots"] = slots
        return slots

    def set_player_slot(self, idx: int, name: str) -> None:
        if not (0 <= int(idx) < 4):
            return
        slots = self.player_slots()
        cleaned = self.sanitize_player_name(name)
        if not cleaned:
            cleaned = f"PLAYER {int(idx) + 1}"
        slots[int(idx)] = {"name": cleaned}

    def clear_player_slot(self, idx: int) -> None:
        if not (0 <= int(idx) < 4):
            return
        slots = self.player_slots()
        slots[int(idx)] = None

    def start_from_player_slots(self) -> None:
        color_slots: List[int] = []
        selected: List[Dict[str, Any]] = []
        for i, s in enumerate(self.player_slots()):
            if isinstance(s, dict):
                selected.append(s)
                color_slots.append(i)
        n = len(selected)
        mode = self.state.get("selected_game_mode")
        if n <= 0:
            return
        if str(mode).lower() == "killer" and n < 2:
            return
        names = [str(s.get("name") or "") for s in selected]
        self.state["num_players"] = n
        self.state["current_player_idx"] = 0
        rules_pending = dict(self.state.get("rules_pending") or {})
        self.ctx["game"] = mm._create_game(
            mode, n, rules_pending, player_names=names, color_slots=color_slots
        )
        self.ctx["input_multiplier"] = 1
        self.all_hits.clear()
        self.ctx["editing_hit_slot"] = None
        self.ctx["_clear_board_status"] = ""
        self.state["screen"] = "clear_board"

    def open_hit_editor(self, slot: int) -> None:
        g = self.ctx.get("game")
        if not g or g.get("winner_player_idx") is not None:
            return
        if self.ctx.get("killer_bull_pending"):
            return
        turn_hits = g.get("turn_hits", [None, None, None])
        cur = int(self.state.get("current_player_idx", 0))
        if mm._turn_blocks_new_hits(g, cur, turn_hits) and turn_hits[slot] is None:
            return
        if 0 <= slot < 3:
            self.ctx["editing_hit_slot"] = slot
            self.ctx["input_multiplier"] = 1
            self.ctx["_dart_can_submit_since"] = time.time()
            self.ctx["_dart_manual_advance_confirm"] = False
            mm._click_sound()

    def handle_keypad(self, kid: str) -> None:
        g = self.ctx.get("game")
        editing_slot = self.ctx.get("editing_hit_slot")
        if not g or editing_slot is None:
            return
        if kid.startswith("mult:"):
            m = int(kid.split(":", 1)[1])
            cur_mult = int(self.ctx.get("input_multiplier", 1))
            self.ctx["input_multiplier"] = 1 if cur_mult == m else m
            mm._click_sound()
            return
        if kid == "clear_one":
            turn_hits = g.get("turn_hits", [None, None, None])
            if 0 <= int(editing_slot) < 3:
                turn_hits[int(editing_slot)] = None
            else:
                mm._clear_one_slot(turn_hits)
            g["turn_hits"] = turn_hits
            cur = int(self.state.get("current_player_idx", 0))
            mm._killer_resync_mid_turn(self.ctx, g, cur, turn_hits)
            # Snapshot upije strijelu u referencu — detekcija mora nastaviti.
            self.ctx["_dart_block_auto_detect"] = False
            self.ctx["_dart_score_locked"] = False
            self.ctx["_dart_can_submit_since"] = None
            self.ctx["_dart_manual_advance_confirm"] = False
            cam_mgr = self.ctx.get("camera_mgr")
            dart_det = self.ctx.get("dart_detector")
            board_cal = self.ctx.get("board_calibrator")
            if cam_mgr is not None and dart_det is not None and board_cal is not None and cam_mgr.active:
                dart_det.reset_motion_state()
                dart_det.commit_board_snapshot(board_cal, cam_mgr.grab_frames())
            if self.is_manual_mode():
                self.ctx["editing_hit_slot"] = int(editing_slot)
                self.ctx["input_multiplier"] = 1
            else:
                self.close_hit_editor()
            mm._click_sound()
            print("[dart] CLR — detekcija ponovno aktivna", flush=True)
            return
        hit: Optional[mm.DetectedHit] = None
        if kid == "hit:miss":
            hit = mm._make_miss_hit()
        elif kid == "bull:25":
            hit = mm._make_bull_hit(False)
        elif kid == "bull:50":
            hit = mm._make_bull_hit(True)
        elif kid.startswith("num:"):
            n = int(kid.split(":", 1)[1])
            hit = mm._make_number_hit(n, int(self.ctx.get("input_multiplier", 1)))
        if hit is not None and 0 <= int(editing_slot) < 3:
            self.set_turn_hit(int(editing_slot), hit)
            if int(self.ctx.get("input_multiplier", 1)) in (2, 3):
                self.ctx["input_multiplier"] = 1
            if self.is_manual_mode():
                # Ostavi tipkovnicu; otvori sljedeći prazan slot.
                self.ctx["editing_hit_slot"] = None
                self.ensure_manual_hit_editor()
                if self.ctx.get("editing_hit_slot") is None:
                    # Sva 3 popunjena — drži zadnji slot za ispravke.
                    self.ctx["editing_hit_slot"] = int(editing_slot)
            else:
                self.close_hit_editor()
            mm._click_sound()

    def handle_action(self, bid: str) -> None:
        if self.ctx.get("_ingame_cal_open") and bid not in (
            "ingame_cal_open",
            "ingame_cal_back",
            "ingame_cal_confirm_empty",
            "ingame_cal_done",
            "ingame_cal_retry",
        ):
            return
        if bid == "ingame_cal_open":
            mm._click_sound()
            self.open_in_game_calibrate()
            return
        if bid == "ingame_cal_back":
            mm._click_sound()
            self.close_in_game_calibrate()
            return
        if bid == "ingame_cal_confirm_empty":
            mm._click_sound()
            if not self.ctx.get("_ingame_cal_open"):
                return
            self.ctx["_ingame_cal_phase"] = "busy"
            self.ctx["_ingame_cal_status"] = get_settings().t("calibrating")
            return
        if bid == "ingame_cal_retry":
            mm._click_sound()
            if not self.ctx.get("_ingame_cal_open"):
                return
            self.ctx["_ingame_cal_phase"] = "busy"
            self.ctx["_ingame_cal_status"] = get_settings().t("calibrating")
            return
        if bid == "ingame_cal_done":
            mm._click_sound()
            self.finish_in_game_calibrate()
            return
        if bid == "settings_open":
            mm._click_sound()
            self.enter_settings()
            return
        if bid == "settings_back":
            mm._click_sound()
            self.exit_settings()
            return
        if bid == "settings_calibration":
            mm._click_sound()
            self.state["_prev_screen"] = "settings"
            self.enter_calibration()
            # enter_calibration overwrites _prev_screen — keep settings as return target.
            self.state["_prev_screen"] = "settings"
            return
        if bid.startswith("settings_lang:"):
            mm._click_sound()
            lang = bid.split(":", 1)[1]
            self.settings = update_settings(language=lang)
            return
        if bid.startswith("settings_bg:"):
            mm._click_sound()
            self.settings = update_settings(bg_color=bid.split(":", 1)[1])
            return
        if bid.startswith("settings_btn:"):
            mm._click_sound()
            self.settings = update_settings(button_color=bid.split(":", 1)[1])
            return
        if bid.startswith("settings_start:"):
            mm._click_sound()
            mode = bid.split(":", 1)[1]
            self.settings = update_settings(start_mode=mode)
            return
        if bid == "standby_start":
            mm._click_sound()
            self.start_from_button()
            return
        if bid == "calibration_back":
            mm._click_sound()
            self.exit_calibration()
            return
        if bid == "calibration_detect":
            mm._click_sound()
            self.run_calibration_detect()
            return
        if bid == "calibration_click_bull":
            mm._click_sound()
            self.toggle_cal_click_bull_mode()
            return
        if bid == "calibration_click_ellipse":
            mm._click_sound()
            self.toggle_cal_click_ellipse_mode()
            return
        if bid == "calibration_overlay":
            mm._click_sound()
            self.toggle_cal_click_ellipse_mode()
            return
        if bid == "calibration_clear_bull_hints":
            mm._click_sound()
            self.clear_bull_hints()
            return
        if bid == "calibration_capture":
            mm._click_sound()
            board_cal = self.ctx.get("board_calibrator")
            if board_cal is not None:
                board_cal.show_topdown = not board_cal.show_topdown
                if board_cal.show_topdown:
                    board_cal.show_overlay = True
                    self.ctx["_cal_click_bull_mode"] = False
                    self.ctx["_cal_click_ellipse_mode"] = False
                self.ctx["_cal_last_tick"] = 0.0
            return
        if bid.startswith("calibration_seg20_left_"):
            mm._click_sound()
            board_cal = self.ctx.get("board_calibrator")
            if board_cal is not None:
                try:
                    cam_idx = int(bid.rsplit("_", 1)[-1])
                except ValueError:
                    return
                board_cal.nudge_segment20(cam_idx, -1)
                self.ctx["_cal_last_tick"] = 0.0
            return
        if bid.startswith("calibration_seg20_right_"):
            mm._click_sound()
            board_cal = self.ctx.get("board_calibrator")
            if board_cal is not None:
                try:
                    cam_idx = int(bid.rsplit("_", 1)[-1])
                except ValueError:
                    return
                board_cal.nudge_segment20(cam_idx, 1)
                self.ctx["_cal_last_tick"] = 0.0
            return
        if bid == "nav_back":
            mm._click_sound()
            self.handle_nav_back()
            return
        if bid.startswith("game:"):
            mm._click_sound()
            mode = bid.split(":", 1)[1]
            self.state["selected_game_mode"] = mode
            self.state["rules_pending"] = {
                "double_in": False,
                "double_out": False,
                "max_rounds_20": False,
                "cricket_mode": "standard",
                "killer_lives": 3,
                "killer_assign": "random",
                "killer_activation": "double",
                "around_mult": "any",
                "around_order": "standard",
                "around_backstep": 0,
                "halve_mode": "standard",
            }
            if mode in ("301", "501"):
                self.state["screen"] = "rules_x01"
            elif mode == "cricket":
                self.state["screen"] = "rules_cricket"
            elif mode == "killer":
                self.state["screen"] = "rules_killer"
            elif mode == "around":
                self.state["screen"] = "rules_around"
            elif mode in ("halve", "halve_it"):
                self.state["screen"] = "rules_halve"
            else:
                self.state["screen"] = "select_players"
            return
        if bid == "tutorial_open":
            mm._click_sound()
            mode = str(self.state.get("selected_game_mode") or "")
            if mode in ("301", "501"):
                self.state["screen"] = "tutorial_x01"
            elif mode == "cricket":
                self.state["screen"] = "tutorial_cricket"
            elif mode == "killer":
                self.state["screen"] = "tutorial_killer"
            elif mode == "around":
                self.state["screen"] = "tutorial_around"
            elif mode in ("halve", "halve_it"):
                self.state["screen"] = "tutorial_halve"
            return
        if bid == "rule:double_in":
            mm._click_sound()
            rp = self.state.setdefault("rules_pending", {})
            rp["double_in"] = not bool(rp.get("double_in"))
            return
        if bid == "rule:double_out":
            mm._click_sound()
            rp = self.state.setdefault("rules_pending", {})
            rp["double_out"] = not bool(rp.get("double_out"))
            return
        if bid == "rule:max_rounds_off":
            mm._click_sound()
            self.state.setdefault("rules_pending", {})["max_rounds_20"] = False
            return
        if bid == "rule:max_rounds_20":
            mm._click_sound()
            self.state.setdefault("rules_pending", {})["max_rounds_20"] = True
            return
        if bid == "rule:cricket_standard":
            mm._click_sound()
            self.state.setdefault("rules_pending", {})["cricket_mode"] = "standard"
            return
        if bid == "rule:cricket_no_score":
            mm._click_sound()
            self.state.setdefault("rules_pending", {})["cricket_mode"] = "no_score"
            return
        if bid == "rule:cricket_cut_throat":
            mm._click_sound()
            self.state.setdefault("rules_pending", {})["cricket_mode"] = "cut_throat"
            return
        if bid.startswith("rule:killer_lives:"):
            mm._click_sound()
            try:
                lives = int(bid.split(":")[-1])
            except ValueError:
                return
            if lives in (3, 5, 7, 10):
                self.state.setdefault("rules_pending", {})["killer_lives"] = lives
            return
        if bid.startswith("rule:killer_assign:"):
            mm._click_sound()
            assign = bid.split(":")[-1]
            if assign in ("random", "throw"):
                self.state.setdefault("rules_pending", {})["killer_assign"] = assign
            return
        if bid.startswith("rule:killer_activation:"):
            mm._click_sound()
            act = bid.split(":")[-1]
            if act in ("single", "double", "triple"):
                self.state.setdefault("rules_pending", {})["killer_activation"] = act
            return
        if bid.startswith("rule:around_mult:"):
            mm._click_sound()
            mult = bid.split(":")[-1]
            if mult in ("single", "double", "triple", "any"):
                self.state.setdefault("rules_pending", {})["around_mult"] = mult
            return
        if bid.startswith("rule:around_order:"):
            mm._click_sound()
            order = bid.split(":")[-1]
            if order in ("standard", "random", "reverse"):
                self.state.setdefault("rules_pending", {})["around_order"] = order
            return
        if bid.startswith("rule:around_backstep:"):
            mm._click_sound()
            try:
                bs = int(bid.split(":")[-1])
            except ValueError:
                return
            if bs in (0, 1, 2):
                self.state.setdefault("rules_pending", {})["around_backstep"] = bs
            return
        if bid.startswith("rule:halve_mode:"):
            mm._click_sound()
            hm = bid.split(":")[-1].lower()
            if hm in ("standard", "extended"):
                self.state.setdefault("rules_pending", {})["halve_mode"] = hm
            return
        if bid.startswith("killer_bull_steal:"):
            mm._click_sound()
            g = self.ctx.get("game")
            if not g:
                return
            try:
                ti = int(bid.split(":")[-1])
            except ValueError:
                return
            if mm._killer_confirm_bull_steal(self.ctx, g, ti):
                self.notify()
            return
        if bid == "killer_bull_false":
            mm._click_sound()
            g = self.ctx.get("game")
            if g and mm._killer_false_bull(self.ctx, g):
                self.notify()
            return
        if bid == "rules_continue":
            mm._click_sound()
            self.state["player_slots"] = [None, None, None, None]
            self.state["screen"] = "select_players"
            return
        if bid == "clear_board_confirm":
            mm._click_sound()
            self.ctx["_clear_board_status"] = get_settings().t("calibrating")
            # Odgodi kalibraciju da UI stigne iscrtati poruku (Qt: QTimer; OpenCV: pending_at).
            self.ctx["_clear_board_pending_run"] = True
            self.ctx["_clear_board_pending_at"] = time.perf_counter()
            return
        if bid.startswith("players:"):
            mm._click_sound()
            n = int(bid.split(":", 1)[1])
            mode = self.state.get("selected_game_mode")
            # Killer requires at least 2 players.
            if str(mode).lower() == "killer" and n < 2:
                return
            self.state["num_players"] = n
            self.state["current_player_idx"] = 0
            rules_pending = dict(self.state.get("rules_pending") or {})
            self.ctx["game"] = mm._create_game(mode, n, rules_pending)
            self.ctx["input_multiplier"] = 1
            self.all_hits.clear()
            self.ctx["editing_hit_slot"] = None
            self.ctx["_clear_board_status"] = ""
            self.state["screen"] = "clear_board"
            return
        if bid == "players_start":
            mm._click_sound()
            self.start_from_player_slots()
            return
        if bid == "request_exit_game":
            mm._click_sound()
            g = self.ctx.get("game")
            if self.state.get("screen") == "playing" and g and mm._winner_screen_ready(g):
                self.clear_to_standby(mqtt_off=True)
            else:
                self.state["screen"] = "exit_confirm"
            return
        if bid == "confirm_exit_yes":
            mm._click_sound()
            self.clear_to_standby(mqtt_off=True)
            return
        if bid == "confirm_exit_no":
            mm._click_sound()
            self.state["screen"] = "playing"
            return
        if bid == "undo_turn":
            g = self.ctx.get("game")
            if not g or not (g.get("undo_stack") or []):
                return
            # Vec u undo modu — ne dopusti drugi NAZAD
            if self.ctx.get("_dart_block_auto_advance"):
                return
            # Dok je OČISTI PLOČU / ruka spremna za predaju — undo off
            turn_hits = g.get("turn_hits", [None, None, None])
            cur = int(self.state.get("current_player_idx", 0))
            if mm._turn_can_submit(g, cur, turn_hits) or mm._killer_assign_needs_retry(
                g, cur, turn_hits
            ):
                return
            mm._click_sound()
            if mm._pop_undo(g, self.state, self.ctx):
                self.ctx["_dart_block_auto_advance"] = True
                self.ctx["_dart_block_auto_detect"] = False
                self.ctx["_dart_empty_frames"] = 0
                # One-shot: UI reverse turn slide (2-player heuristic is ambiguous)
                self.ctx["_ui_turn_anim_reverse"] = True
                self.close_hit_editor()
                self.sync_board_refs_after_undo()
                print("[dart] nazad — uredi ruku / prazni slotovi; iduci igrac pauziran", flush=True)
            return
        if bid == "manual_advance_confirm":
            g = self.ctx.get("game")
            if not g:
                return
            if self.ctx.get("_dart_block_auto_advance"):
                mm._click_sound()
                self.ctx["_dart_manual_advance_confirm"] = False
                self.confirm_undo_advance_to_next()
                return
            turn_hits = g.get("turn_hits", [None, None, None])
            cur_chk = int(self.state.get("current_player_idx", 0))
            if not (
                mm._turn_can_submit(g, cur_chk, turn_hits)
                or mm._killer_assign_needs_retry(g, cur_chk, turn_hits)
            ):
                return
            # Ručni: odmah PREBACI IGRACA (bez potvrde / 8s delay).
            if self.is_manual_mode():
                mm._click_sound()
                self.ctx["_dart_manual_advance_confirm"] = False
                self.force_advance_past_board_clear()
                return
            submit_since = self.ctx.get("_dart_can_submit_since")
            if (
                submit_since is not None
                and time.time() - float(submit_since) < mm.FORCE_NEXT_BUTTON_DELAY_SEC
            ):
                return
            mm._click_sound()
            if not self.ctx.get("_dart_manual_advance_confirm"):
                self.ctx["_dart_manual_advance_confirm"] = True
                print("[dart] rucno prebacivanje — potvrdi: ploca je cista?", flush=True)
                return
            self.ctx["_dart_manual_advance_confirm"] = False
            self.force_advance_past_board_clear(update_empty_ref=True)
            return
        if bid.startswith("hit_slot:"):
            try:
                slot = int(bid.split(":", 1)[1])
            except ValueError:
                return
            self.open_hit_editor(slot)
            return
        if bid == "keypad_close":
            mm._click_sound()
            if self.is_manual_mode():
                # U ručnom načinu tipkovnica ostaje; X samo gasi D/T.
                self.ctx["input_multiplier"] = 1
            else:
                self.close_hit_editor()
            return
        if bid == "toggle_manual_mode":
            mm._click_sound()
            self.toggle_manual_mode()
            return
        if bid.startswith(("mult:", "num:", "bull:", "hit:", "clear_one")) or bid == "clear_one":
            self.handle_keypad(bid)
            return

    def ensure_cameras(self) -> None:
        cam_mgr = self.ctx.get("camera_mgr")
        if cam_mgr is None:
            return
        need = self.screen in ("calibration", "clear_board", "playing")
        if need and not cam_mgr.active:
            cam_mgr.start()
        elif not need and cam_mgr.active:
            cam_mgr.stop()
            self.ctx["_last_cal_frames"] = {}
            dart_det = self.ctx.get("dart_detector")
            if dart_det is not None:
                dart_det.reset_motion_state()

    def request_calibration_view_size(self, width: int, height: int) -> None:
        self._cal_view_wh = (max(2, int(width)), max(2, int(height)))

    def get_calibration_frame(self, width: int = 1280, height: int = 720) -> Optional[np.ndarray]:
        """Zadnji kalibracijski kadar s vision niti (GUI samo prikaže)."""
        self.request_calibration_view_size(width, height)
        pack = self._bg_cal_pack
        if not pack:
            return None
        self.ctx["_cal_buttons"] = pack.get("buttons") or []
        self.ctx["_cal_tiles"] = pack.get("tiles") or []
        img = pack.get("img")
        return img if img is not None else None

    def _build_calibration_frame(self, width: int, height: int) -> Optional[np.ndarray]:
        if self.screen != "calibration":
            return None
        cam_mgr = self.ctx.get("camera_mgr")
        board_cal = self.ctx.get("board_calibrator")
        if cam_mgr is None or not cam_mgr.active:
            return None
        now = time.perf_counter()
        if now - float(self.ctx.get("_cal_last_tick", 0.0)) < CALIBRATION_FRAME_INTERVAL:
            pack = self._bg_cal_pack
            return None if pack is None else pack.get("img")
        self.ctx["_cal_last_tick"] = now
        frames = cam_mgr.grab_frames()
        self.ctx["_last_cal_frames"] = frames
        status = ""
        if board_cal is not None:
            status = board_cal.status_summary()
        if not cam_mgr.fake_mode and not cam_mgr.opened_indices():
            status = "Otvaranje kamera..."
        buttons: list = []
        tiles: list = []
        click_bull = bool(self.ctx.get("_cal_click_bull_mode")) and not self._cal_topdown_on()
        click_ell = bool(self.ctx.get("_cal_click_ellipse_mode")) and not self._cal_topdown_on()
        banner = ""
        if click_bull:
            try:
                banner = get_settings().t("click_bull_hint")
            except Exception:
                banner = "Klikni otprilike na bull"
        elif click_ell:
            try:
                banner = get_settings().t("click_ellipse_hint")
            except Exception:
                banner = "Klikni do 4 tocke na vanjskom rubu"
        img = build_calibration_view(
            width,
            height,
            frames,
            status=status,
            board_calibrator=board_cal,
            buttons_out=buttons,
            draw_action_buttons=False,
            bull_hints=dict(self.ctx.get("bull_hints") or {}),
            ellipse_hints=dict(self.ctx.get("ellipse_hints") or {}),
            tiles_out=tiles,
            click_bull_mode=click_bull,
            click_bull_banner=banner,
            click_ellipse_mode=click_ell,
        )
        self._bg_cal_pack = {"img": img, "buttons": buttons, "tiles": tiles}
        return img

    def handle_calibration_click(self, x: int, y: int, width: int, height: int) -> None:
        """Map click on calibration QLabel to bull / ellipse hints."""
        buttons = self.ctx.get("_cal_buttons") or []
        for btn in buttons:
            x1, y1, x2, y2 = btn["rect"]
            if x1 <= x < x2 and y1 <= y < y2:
                self.handle_action(btn["id"])
                return

        bull_mode = bool(self.ctx.get("_cal_click_bull_mode"))
        ell_mode = bool(self.ctx.get("_cal_click_ellipse_mode"))
        if self._cal_topdown_on() or (not bull_mode and not ell_mode):
            return

        tiles = self.ctx.get("_cal_tiles") or []
        for tile in tiles:
            vx1, vy1, vx2, vy2 = tile["video_rect"]
            if not (vx1 <= x < vx2 and vy1 <= y < vy2):
                continue
            fw, fh = tile["frame_size"]
            dw, dh = tile.get("display_size") or (fw, fh)
            vw = max(1, vx2 - vx1)
            vh = max(1, vy2 - vy1)
            dx = (float(x - vx1) / float(vw)) * float(dw)
            dy = (float(y - vy1) / float(vh)) * float(dh)
            cam_idx = int(tile["cam_idx"])
            fx, fy = dx, dy
            board_cal = self.ctx.get("board_calibrator")
            if board_cal is not None and bool(getattr(board_cal, "show_topdown", False)):
                mapped = board_cal.map_topdown_click_to_camera(
                    cam_idx,
                    dx,
                    dy,
                    display_size=(int(dw), int(dh)),
                    frame_size=(int(fw), int(fh)),
                )
                if mapped is not None:
                    fx, fy = mapped
                elif int(dw) > 0 and int(dh) > 0 and (int(dw) != int(fw) or int(dh) != int(fh)):
                    fx = dx * (float(fw) / float(dw))
                    fy = dy * (float(fh) / float(dh))
            elif int(dw) > 0 and int(dh) > 0 and (int(dw) != int(fw) or int(dh) != int(fh)):
                fx = dx * (float(fw) / float(dw))
                fy = dy * (float(fh) / float(dh))
            fx = max(0.0, min(float(fw - 1), fx))
            fy = max(0.0, min(float(fh - 1), fy))
            if bull_mode:
                hints = dict(self.ctx.get("bull_hints") or {})
                hints[cam_idx] = (fx, fy)
                self.ctx["bull_hints"] = hints
                print(
                    f"[dart] bull hint cam{cam_idx}: ({fx:.1f},{fy:.1f}) — ponovo KALIBRIRAJ",
                    flush=True,
                )
            else:
                raw = dict(self.ctx.get("ellipse_hints") or {})
                pts = list(raw.get(cam_idx) or raw.get(str(cam_idx)) or [])
                pts = [(float(p[0]), float(p[1])) for p in pts if p is not None and len(p) >= 2]
                near = max(18.0, min(float(fw), float(fh)) * 0.10)
                moved = False
                if pts:
                    dists = [
                        ((fx - px) ** 2 + (fy - py) ** 2) ** 0.5
                        for px, py in pts
                    ]
                    best = int(min(range(len(dists)), key=lambda i: dists[i]))
                    if dists[best] <= near or len(pts) >= 4:
                        pts[best] = (fx, fy)
                        moved = True
                if not moved and len(pts) < 4:
                    pts.append((fx, fy))
                raw[cam_idx] = pts[:4]
                self.ctx["ellipse_hints"] = raw
                print(
                    f"[dart] ellipse hint cam{cam_idx}: n={len(pts)} "
                    f"last=({fx:.1f},{fy:.1f}) — ponovo KALIBRIRAJ",
                    flush=True,
                )
            self.ctx["_cal_last_tick"] = 0.0
            self._persist_cal_hints()
            return

    def _bg_has_pending(self) -> bool:
        with self._bg_pending_lock:
            return self._bg_pending is not None

    def _bg_set_pending(self, result: Dict[str, Any]) -> None:
        with self._bg_pending_lock:
            self._bg_pending = result

    def _drain_vision_results(self) -> None:
        with self._bg_pending_lock:
            pending = self._bg_pending
        if not pending:
            return
        with self._vision_lock:
            with self._bg_pending_lock:
                pending = self._bg_pending
                if pending is None:
                    return
            kind = pending.get("kind")
            try:
                if kind == "hit":
                    self._apply_detect_result(pending)
                elif kind == "empty_advance":
                    self._apply_empty_advance_result(pending)
            finally:
                with self._bg_pending_lock:
                    self._bg_pending = None

    def _vision_loop(self) -> None:
        print("[vision] background thread on", flush=True)
        while not self._bg_stop.wait(0.02):
            try:
                self._vision_loop_once()
            except Exception as exc:
                print(f"[vision] error (ignored): {exc}", flush=True)

    def _vision_loop_once(self) -> None:
        if self._bg_stop.is_set():
            return
        if self.ctx.get("_ingame_cal_open"):
            return
        if self._bg_has_pending():
            return
        screen = self.screen
        if screen == "calibration":
            if not self._vision_lock.acquire(blocking=False):
                return
            try:
                w, h = self._cal_view_wh
                self._build_calibration_frame(w, h)
            finally:
                self._vision_lock.release()
            return
        if screen != "playing":
            return
        if not self._vision_lock.acquire(blocking=False):
            return
        try:
            self._vision_playing_cycle()
        finally:
            self._vision_lock.release()

    def _vision_playing_cycle(self) -> None:
        if self._bg_has_pending():
            return
        g_play = self.ctx.get("game")
        if not g_play:
            return
        if g_play.get("winner_player_idx") is not None and not mm._winner_pending_board_clear(
            g_play
        ):
            return
        if self.is_manual_mode():
            return
        dart_det = self.ctx.get("dart_detector")
        cam_mgr = self.ctx.get("camera_mgr")
        if dart_det is None or cam_mgr is None or not cam_mgr.active:
            return
        if self.ctx.get("_dart_turn_start_pending"):
            if dart_det.has_references():
                self.prepare_dart_turn_start(commit_snapshot=True)
                self.ctx["_dart_turn_start_pending"] = False
                print("[dart] igra pokrenuta — cekam mirnu plocu", flush=True)
        if not dart_det.has_references() and not self.ctx.get("_dart_no_ref_warned"):
            print(
                "[dart] Nema reference — pritisni KALIBRIRAJ u kalibraciji (prazna ploca)",
                flush=True,
            )
            self.ctx["_dart_status"] = "CEKAJ REF"
            self.ctx["_dart_no_ref_warned"] = True
        now_dart = time.perf_counter()
        if now_dart - float(self.ctx.get("_dart_last_tick", 0.0)) < mm.DART_DETECT_INTERVAL:
            return
        self.ctx["_dart_last_tick"] = now_dart
        turn_hits = g_play.get("turn_hits", [None, None, None])
        cur_play = int(self.state.get("current_player_idx", 0))
        checkout_ready = mm._turn_is_early_finish(g_play, cur_play, turn_hits)
        bust_ready = mm._x01_turn_is_bust(g_play, cur_play, turn_hits)
        score_paused = self.dart_scoring_paused(g_play, cur_play, turn_hits)
        prev_locked = bool(self.ctx.get("_dart_score_locked", False))
        if score_paused and not prev_locked:
            self.close_hit_editor()
            if bust_ready or checkout_ready:
                dart_det.reset_motion_state()
            if bust_ready:
                print("[dart] bust — detekcija pauzirana", flush=True)
            elif checkout_ready:
                print("[dart] kraj ruke spreman — detekcija pauzirana", flush=True)
        self.ctx["_dart_score_locked"] = score_paused
        result: Optional[Dict[str, Any]] = None
        if self.ctx.get("_dart_block_auto_advance"):
            result = self._auto_detect_vision(undo_mode=False)
        elif (
            (
                mm._turn_can_submit(g_play, cur_play, turn_hits)
                or mm._killer_assign_needs_retry(g_play, cur_play, turn_hits)
            )
            and not self.ctx.get("killer_bull_pending")
        ):
            result = self._empty_advance_vision()
        else:
            result = self._auto_detect_vision()
        if result:
            self._bg_set_pending(result)

    def tick(self) -> None:
        if self.ctx.get("_ingame_cal_open"):
            if self.screen != "playing":
                self.close_in_game_calibrate()
            else:
                self.ensure_cameras()
            self._drain_vision_results()
            return
        g_auto = self.ctx.get("game")
        if self.screen == "playing" and g_auto and mm._winner_screen_ready(g_auto):
            t0 = g_auto.get("winner_shown_at")
            if t0 is None:
                g_auto["winner_shown_at"] = time.time()
            elif time.time() - float(t0) >= mm.WINNER_AUTO_EXIT_SEC:
                self.clear_to_standby(mqtt_off=True)
        elif isinstance(g_auto, dict):
            g_auto.pop("winner_shown_at", None)

        self.ensure_cameras()
        if self.ctx.get("_dart_absorb_pending"):
            self.resume_live_motion_after_pause()
        self._drain_vision_results()
        if self.screen == "playing":
            self.ensure_manual_hit_editor()

    def snapshot_for_ui(self) -> Dict[str, Any]:
        g = self.ctx.get("game")
        cur = int(self.state.get("current_player_idx", 0))
        turn_hits = (g or {}).get("turn_hits", [None, None, None]) if g else [None, None, None]
        can_submit = bool(
            g
            and mm._turn_can_submit(g, cur, turn_hits)
            and not self.ctx.get("killer_bull_pending")
        )
        needs_assign_retry = bool(
            g
            and mm._killer_assign_needs_retry(g, cur, turn_hits)
            and not self.ctx.get("killer_bull_pending")
        )
        waiting_clear = can_submit or needs_assign_retry
        early = bool(g and mm._turn_is_early_finish(g, cur, turn_hits))
        bust = bool(g and mm._x01_turn_is_bust(g, cur, turn_hits))
        block_adv = bool(self.ctx.get("_dart_block_auto_advance"))
        now_submit = time.time()
        submit_since = self.ctx.get("_dart_can_submit_since")
        if waiting_clear and submit_since is None:
            self.ctx["_dart_can_submit_since"] = now_submit
            submit_since = now_submit
        elif not waiting_clear and not block_adv:
            self.ctx["_dart_can_submit_since"] = None
            submit_since = None
        show_manual = block_adv or (
            waiting_clear
            and submit_since is not None
            and (
                self.is_manual_mode()
                or now_submit - float(submit_since) >= mm.FORCE_NEXT_BUTTON_DELAY_SEC
            )
        )
        if block_adv or self.is_manual_mode():
            advance_label = get_settings().t("switch_player")
        elif self.ctx.get("_dart_manual_advance_confirm"):
            advance_label = get_settings().t("board_clean_q")
        else:
            advance_label = get_settings().t("not_advanced_q")
        board_msg = ""
        if waiting_clear or block_adv:
            if show_manual:
                board_msg = ""
            else:
                board_msg = (
                    get_settings().t("board_clear_end")
                    if early
                    else get_settings().t("board_clear")
                )

        players_ui = []
        if g:
            # Live score s hitovima (uklj. undo uredivanje) — simulacija bez mutacije igre
            preview_players = None
            if any(h is not None for h in turn_hits):
                try:
                    preview_players = mm._preview_players_after_turn(g, cur, turn_hits)
                except Exception:
                    preview_players = None
            for pi, pstate in enumerate(g.get("players") or []):
                show = (
                    preview_players[pi]
                    if preview_players is not None and 0 <= pi < len(preview_players)
                    else pstate
                )
                players_ui.append(
                    {
                        "index": pi,
                        "label": f"P{pi + 1}",
                        "name": str(pstate.get("name") or ""),
                        "score_text": mm._player_score_text(g, pi, show),
                        "color": _bgr_to_hex(mm._player_color_from_state(pstate, pi)),
                        "active": pi == cur,
                        "rank": mm._player_place_rank(g, pi),
                        "marks": dict(show.get("wickets") or show.get("marks") or {}),
                        "points": int(show.get("points", show.get("score", 0)) or 0),
                        "remaining": show.get("remaining"),
                        "completed": bool(show.get("completed")),
                        "next_target_idx": show.get("next_target_idx"),
                        "number": show.get("number"),
                        "lives": show.get("lives"),
                        "is_killer": bool(show.get("is_killer")),
                        "score": int(show.get("score", 0) or 0),
                        "score_halved": bool(show.get("score_just_halved")),
                    }
                )

        hit_tags = [mm._hit_tag(h) for h in turn_hits]
        blocks_new = bool(g and mm._turn_blocks_new_hits(g, cur, turn_hits))
        slot_enabled = [
            not (blocks_new and turn_hits[i] is None) for i in range(3)
        ]

        # Undo: ima snimke, nismo vec u undo, i nije OČISTI PLOČU / predaja ruke
        has_stack = bool(g and (g.get("undo_stack") or []))
        waiting_clear_ui = bool(board_msg) or (waiting_clear and not block_adv)
        has_undo = has_stack and not block_adv and not waiting_clear_ui
        turn_anim_reverse = bool(self.ctx.pop("_ui_turn_anim_reverse", False))

        # Gate winner/leaderboard until finishing turn is submitted (board cleared).
        winner_ready = bool(g and mm._winner_screen_ready(g))
        winner_idx = (g or {}).get("winner_player_idx") if winner_ready else None
        league_qr_payload = (
            ensure_game_qr(g, get_settings().device_id, winner_ready=winner_ready)
            if g and get_settings().league_qr_enabled
            else None
        )

        return {
            "screen": self.screen,
            "selected_game_mode": self.state.get("selected_game_mode"),
            "player_slots": [
                None if s is None else {"name": str((s or {}).get("name") or "")}
                for s in (self.player_slots())
            ],
            "rules_pending": dict(self.state.get("rules_pending") or {}),
            "game_title": mm._game_display_title(g) if g else "",
            "mode": str((g or {}).get("mode", "")).lower() if g else "",
            "players": players_ui,
            "current_player_idx": cur,
            "turn_hit_tags": hit_tags,
            "slot_enabled": slot_enabled,
            "editing_hit_slot": self.ctx.get("editing_hit_slot"),
            "input_multiplier": int(self.ctx.get("input_multiplier", 1)),
            "manual_mode": self.is_manual_mode(),
            "can_submit": can_submit,
            "is_early_finish": early,
            "is_bust": bust,
            "block_adv": block_adv,
            "show_manual_advance": show_manual,
            "advance_label": advance_label,
            "board_clear_message": board_msg,
            "has_undo": has_undo,
            "turn_anim_reverse": turn_anim_reverse,
            "winner": winner_idx,
            "leaderboard": mm._leaderboard_entries(g) if winner_ready else [],
            "league_qr_payload": league_qr_payload,
            "live_score_line": (
                None
                if g and str(g.get("mode", "")).lower() == "killer"
                else (
                    mm._halve_live_score_line(g)
                    if g and str(g.get("mode", "")).lower() in ("halve", "halve_it")
                    else (mm._x01_live_score_line(g, cur) if g else None)
                )
            ),
            "dart_status": str(self.ctx.get("_dart_status") or ""),
            "clear_board_status": str(self.ctx.get("_clear_board_status") or ""),
            "ingame_cal_open": bool(self.ctx.get("_ingame_cal_open")),
            "ingame_cal_phase": str(self.ctx.get("_ingame_cal_phase") or ""),
            "ingame_cal_status": str(self.ctx.get("_ingame_cal_status") or ""),
            "round_number": int((g or {}).get("round_number", 1)) if g else 1,
            "rules": dict((g or {}).get("rules") or {}) if g else {},
            "killer_phase": str((g or {}).get("killer_phase") or "") if g else "",
            "killer_bull_pending": (
                dict(self.ctx["killer_bull_pending"])
                if isinstance(self.ctx.get("killer_bull_pending"), dict)
                else None
            ),
            "killer_bull_targets": (
                [
                    {
                        "index": i,
                        "label": f"P{i + 1}",
                        "color": _bgr_to_hex(
                            mm._player_color_from_state(
                                (g.get("players") or [])[i], i
                            )
                        ),
                        "number": (g.get("players") or [])[i].get("number"),
                        "lives": int((g.get("players") or [])[i].get("lives") or 0),
                        "is_killer": bool((g.get("players") or [])[i].get("is_killer")),
                    }
                    for i in mm._killer_bull_steal_targets(
                        g, int((self.ctx.get("killer_bull_pending") or {}).get("attacker", cur))
                    )
                ]
                if g and isinstance(self.ctx.get("killer_bull_pending"), dict)
                else []
            ),
            "killer_life_flash": self.ctx.pop("_ui_killer_life_flash", None),
            "qr_path": self.start_qr_path(),
            "kiosk": self.kiosk,
            "settings": {
                "language": get_settings().language,
                "bg_color": get_settings().bg_color,
                "button_color": get_settings().button_color,
                "start_mode": get_settings().start_mode,
                "manual_mode": get_settings().manual_mode,
                "auto_calibrate": get_settings().auto_calibrate,
                "device_id": get_settings().device_id,
                "league_qr_enabled": get_settings().league_qr_enabled,
            },
            "cal_click_bull_mode": bool(self.ctx.get("_cal_click_bull_mode"))
            and not bool(
                getattr(self.ctx.get("board_calibrator"), "show_topdown", False)
            ),
            "cal_click_ellipse_mode": bool(self.ctx.get("_cal_click_ellipse_mode"))
            and not bool(
                getattr(self.ctx.get("board_calibrator"), "show_topdown", False)
            ),
            "cal_show_topdown": bool(
                getattr(self.ctx.get("board_calibrator"), "show_topdown", False)
            ),
            "cal_show_overlay": True,
            "bull_hints": {
                int(k): (float(v[0]), float(v[1]))
                for k, v in (self.ctx.get("bull_hints") or {}).items()
                if v is not None and len(v) >= 2
            },
            "ellipse_hints": {
                int(k): [
                    (float(p[0]), float(p[1]))
                    for p in (v or [])
                    if p is not None and len(p) >= 2
                ][:4]
                for k, v in (self.ctx.get("ellipse_hints") or {}).items()
            },
            "undo_pending_tags": [
                mm._hit_tag(h)
                for h in (self.ctx.get("_dart_undo_pending_hits") or [None, None, None])
            ]
            if block_adv
            else None,
        }

    def shutdown(self) -> None:
        self._bg_stop.set()
        t = getattr(self, "_bg_thread", None)
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=1.5)
        cam_mgr = self.ctx.get("camera_mgr")
        if cam_mgr is not None:
            cam_mgr.stop()
        if get_settings().start_mode == "always_on":
            return
        try:
            mm.send_mqtt_off()
        except Exception:
            pass
