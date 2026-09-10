#!/usr/bin/env python3
"""
Manual-input darts kiosk app (no cameras, no detection).

Keeps the same high-level flow as the camera app:
- MQTT on/off control
- standby (QR) -> game selection -> player selection -> playing
- fullscreen kiosk mode option

Gameplay input is a big keypad:
- Numbers 1..20
- D / T one-shot mode (default is always single)
- MISS
- B25 / B50
- CLR (single-step clear; repeatedly removes latest entered hit)
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import math
import os
import random
import signal
import ssl
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import paho.mqtt.client as mqtt

from mqtt import BROKER_HOST, BROKER_PORT, PASSWORD, TOPIC, USERNAME
from calibration import (
    CALIBRATION_FRAME_INTERVAL,
    CAMERA_INDICES,
    CameraManager,
    build_calibration_view,
    default_fake_cams_dir,
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

DART_DETECT_INTERVAL = 1.0 / 15.0
DART_MIN_CONFIDENCE = 0.18
DART_IDLE_FRAMES_AFTER_SWITCH = 6
EMPTY_BOARD_CONFIRM_FRAMES = 5
# Jedan loš kadar ne briše streak — samo smanji (šum / trenutni flicker).
EMPTY_BOARD_FAIL_PENALTY = 2
FORCE_NEXT_BUTTON_DELAY_SEC = 8.0


@dataclass
class HitScore:
    segment_index: int
    segment_number: int
    zone_index: int
    zone_name: str
    score: int


@dataclass
class DetectedHit:
    cam_indices: List[int]
    image_pos: Tuple[float, float]
    canonical_r: float
    canonical_theta: float
    hit_score: HitScore
    per_cam_hits: List[object]

shutdown_requested = False
# winsound više nije potreban (klik bez zvuka)


def _on_sigint(_signum, _frame):
    global shutdown_requested
    shutdown_requested = True


def send_mqtt_off():
    try:
        client = mqtt.Client()
        client.username_pw_set(USERNAME, PASSWORD)
        client.tls_set(
            ca_certs=None,
            certfile=None,
            keyfile=None,
            cert_reqs=ssl.CERT_REQUIRED,
            tls_version=ssl.PROTOCOL_TLS_CLIENT,
        )
        client.connect(BROKER_HOST, BROKER_PORT, 60)
        client.loop_start()
        info = client.publish(TOPIC, "off", qos=1)
        try:
            info.wait_for_publish(timeout=5.0)
        except Exception:
            time.sleep(0.5)
        client.loop_stop()
        client.disconnect()
        print(f"[MQTT] published 'off' -> {TOPIC}", flush=True)
    except Exception as e:
        print(f"[MQTT] send_mqtt_off failed: {e}", flush=True)


def send_mqtt_on():
    try:
        client = mqtt.Client()
        client.username_pw_set(USERNAME, PASSWORD)
        client.tls_set(
            ca_certs=None,
            certfile=None,
            keyfile=None,
            cert_reqs=ssl.CERT_REQUIRED,
            tls_version=ssl.PROTOCOL_TLS_CLIENT,
        )
        client.connect(BROKER_HOST, BROKER_PORT, 60)
        client.loop_start()
        info = client.publish(TOPIC, "on", qos=1)
        try:
            info.wait_for_publish(timeout=5.0)
        except Exception:
            time.sleep(0.5)
        client.loop_stop()
        client.disconnect()
        print(f"[MQTT] published 'on' -> {TOPIC}", flush=True)
    except Exception as e:
        print(f"[MQTT] send_mqtt_on failed: {e}", flush=True)


def _click_sound():
    """Bez zvuka — nema click asseta; Windows SystemAsterisk je uklonjen."""
    return


def _make_detected(hit_score: HitScore) -> DetectedHit:
    return DetectedHit(
        cam_indices=[],
        image_pos=(0.0, 0.0),
        canonical_r=0.0,
        canonical_theta=0.0,
        hit_score=hit_score,
        per_cam_hits=[],
    )


def _make_number_hit(number: int, mult: int) -> DetectedHit:
    if mult == 3:
        hs = HitScore(segment_index=-1, segment_number=number, zone_index=3, zone_name="triple", score=3 * number)
    elif mult == 2:
        hs = HitScore(segment_index=-1, segment_number=number, zone_index=5, zone_name="double", score=2 * number)
    else:
        hs = HitScore(segment_index=-1, segment_number=number, zone_index=2, zone_name="single", score=number)
    return _make_detected(hs)


def _make_miss_hit() -> DetectedHit:
    return _make_detected(HitScore(segment_index=-1, segment_number=0, zone_index=-1, zone_name="miss", score=0))


def _make_bull_hit(inner: bool) -> DetectedHit:
    if inner:
        return _make_detected(HitScore(segment_index=-1, segment_number=50, zone_index=0, zone_name="inner_bull", score=50))
    return _make_detected(HitScore(segment_index=-1, segment_number=25, zone_index=1, zone_name="outer_bull", score=25))


def _hit_tag(h: Optional[DetectedHit]) -> str:
    if h is None:
        return "-"
    hs = h.hit_score
    zn = str(getattr(hs, "zone_name", "single")).lower()
    if getattr(hs, "score", 0) <= 0 or zn == "miss":
        return "MISS"
    seg = int(getattr(hs, "segment_number", 0))
    if zn == "triple":
        return f"T{seg}"
    if zn == "double":
        return f"D{seg}"
    if zn == "inner_bull":
        return "B50"
    if zn == "outer_bull":
        return "B25"
    return f"S{seg}"


def _next_insert_slot(turn_hits: List[Optional[DetectedHit]]) -> int:
    for i, h in enumerate(turn_hits):
        if h is None:
            return i
    return 2


def _clear_one_slot(turn_hits: List[Optional[DetectedHit]]) -> int:
    for i in (2, 1, 0):
        if turn_hits[i] is not None:
            turn_hits[i] = None
            return i
    return -1


def _is_double_hit(h: Optional[DetectedHit]) -> bool:
    if h is None:
        return False
    zn = str(getattr(h.hit_score, "zone_name", "single")).lower()
    return zn == "double" or zn == "inner_bull"


def _filter_x01_turn_hits(
    hits: List[Optional[DetectedHit]], double_in: bool, player_is_in: bool
) -> List[DetectedHit]:
    """Apply double-in: only darts from first double onward count (B50 counts as double)."""
    if not double_in or player_is_in:
        return [h for h in hits if h is not None]
    out: List[DetectedHit] = []
    opened = False
    for h in hits:
        if h is None:
            continue
        if not opened:
            if _is_double_hit(h):
                opened = True
                out.append(h)
        else:
            out.append(h)
    return out


def _apply_x01_turn(
    game: Dict, cur_idx: int, final_hits: List[Optional[DetectedHit]]
) -> None:
    rules = game.get("rules") or {}
    double_in = bool(rules.get("double_in"))
    double_out = bool(rules.get("double_out"))
    pstate = game["players"][cur_idx]
    start_remaining = int(pstate["remaining"])
    player_is_in = bool(pstate.get("is_in"))
    scoring_hits = _filter_x01_turn_hits(final_hits, double_in, player_is_in)
    total_sub = sum(h.hit_score.score for h in scoring_hits)
    new_remaining = start_remaining - int(total_sub)
    bust = False
    if new_remaining < 0:
        bust = True
    elif double_out and new_remaining == 1:
        bust = True
    elif new_remaining == 0 and double_out:
        if not scoring_hits or not _is_double_hit(scoring_hits[-1]):
            bust = True
    if bust:
        new_remaining = start_remaining
    else:
        if double_in and scoring_hits and any(_is_double_hit(h) for h in scoring_hits):
            pstate["is_in"] = True
        elif not double_in:
            pstate["is_in"] = True
    pstate["remaining"] = int(new_remaining)
    pstate["last_turn"] = final_hits
    if not bust and int(new_remaining) == 0:
        _record_player_finish(game, cur_idx)


def _cricket_all_closed(pstate: Dict) -> bool:
    wicket_values = [15, 16, 17, 18, 19, 20, 25]
    w = pstate.get("wickets") or {}
    for v in wicket_values:
        n = w.get(v, w.get(str(v), 0))
        if int(n or 0) < 3:
            return False
    return True


def _cricket_resolve_remaining_by_points(game: Dict) -> None:
    """Kad nitko više ne može bodovati (sve zatvoreno) — poredaj preostale po bodovima."""
    players = game.get("players") or []
    if not players:
        return
    placements = game.setdefault("placements", [])
    unplaced = [i for i in range(len(players)) if i not in placements]
    if not unplaced:
        if placements and game.get("winner_player_idx") is None:
            game["winner_player_idx"] = placements[0]
        return
    rules = game.get("rules") or {}
    cricket_mode = str(rules.get("cricket_mode", "standard")).lower()
    if cricket_mode == "cut_throat":
        unplaced.sort(key=lambda i: (int(players[i].get("points", 0)), i))
    else:
        # standard / no_score: više bodova = bolje; isti score → manji index prvi
        unplaced.sort(key=lambda i: (-int(players[i].get("points", 0)), i))
    for i in unplaced:
        placements.append(i)
        print(f"[game] P{i + 1} zavrsio na {len(placements)}. mjestu (cricket resolve)", flush=True)
    if placements:
        game["winner_player_idx"] = placements[0]


def _cricket_set_winner_if_any(game: Dict) -> None:
    rules = game.get("rules") or {}
    cricket_mode = str(rules.get("cricket_mode", "standard")).lower()
    players = game.get("players") or []
    if not players:
        return
    placed = set(game.get("placements") or [])
    unplaced = [i for i in range(len(players)) if i not in placed]

    # Deadlock: svi preostali zatvorili — više se ne može bodovati → poredaj po bodovima
    if unplaced and all(_cricket_all_closed(players[i]) for i in unplaced):
        _cricket_resolve_remaining_by_points(game)
        return

    if cricket_mode == "no_score":
        for i in unplaced:
            if _cricket_all_closed(players[i]):
                _record_player_finish(game, i)
                return
        return
    if cricket_mode == "cut_throat":
        for i in unplaced:
            p = players[i]
            if not _cricket_all_closed(p):
                continue
            my_pts = int(p.get("points", 0))
            # Usporedi samo s onima koji još igraju (ne s već plasiranima)
            if all(
                my_pts <= int(players[j].get("points", 0))
                for j in unplaced
                if j != i
            ):
                _record_player_finish(game, i)
                return
        return
    # standard: zatvorio sve + najbolji (ili dijeljeni) među onima koji JOŠ igraju
    for i in unplaced:
        p = players[i]
        if not _cricket_all_closed(p):
            continue
        my_pts = int(p.get("points", 0))
        rivals = [j for j in unplaced if j != i]
        if rivals and not all(
            my_pts >= int(players[j].get("points", 0)) for j in rivals
        ):
            continue
        # Isti score kao netko tko još nije zatvorio → čekaj (može te prestići)
        tied_open = [
            j
            for j in rivals
            if int(players[j].get("points", 0)) == my_pts
            and not _cricket_all_closed(players[j])
        ]
        if tied_open:
            continue
        _record_player_finish(game, i)
        return


def _cricket_round_limit_winner(game: Dict) -> None:
    rules = game.get("rules") or {}
    if not rules.get("max_rounds_20"):
        return
    limit = int(rules.get("max_rounds_limit", 20))
    if int(game.get("round_number", 1)) <= limit:
        return
    players = game.get("players") or []
    if not players:
        return
    cricket_mode = str(rules.get("cricket_mode", "standard")).lower()
    if cricket_mode == "cut_throat":
        pts = [(i, int(p.get("points", 0))) for i, p in enumerate(players)]
        min_pts = min(p for _, p in pts)
        tops = [i for i, p in pts if p == min_pts]
    else:
        pts = [(i, int(p.get("points", 0))) for i, p in enumerate(players)]
        max_pts = max(p for _, p in pts)
        tops = [i for i, p in pts if p == max_pts]
    game["winner_player_idx"] = min(tops)


PLAYER_COLORS_BGR: Tuple[Tuple[int, int, int], ...] = (
    (60, 60, 255),    # P1 crvena
    (255, 140, 30),   # P2 plava
    (220, 60, 220),   # P3 ljubicasta
    (0, 165, 255),    # P4 narancasta
    (255, 255, 255),  # bijela
    (160, 160, 168),  # siva
    (0, 0, 0),        # crna (UI label/icon → bijela radi kontrasta)
)

WINNER_AUTO_EXIT_SEC = 30.0


def _winner_screen_ready(game: Optional[Dict]) -> bool:
    """True when winner is set and the finishing turn is already submitted.

    Killer (and any mid-turn finish) may set winner_player_idx while darts are
    still on the board — keep clear-board / advance flow until turn_hits clear.
    """
    if not game or game.get("winner_player_idx") is None:
        return False
    turn_hits = game.get("turn_hits") or [None, None, None]
    return not any(h is not None for h in turn_hits)


def _winner_pending_board_clear(game: Optional[Dict]) -> bool:
    """Winner decided mid-turn but board still has darts to clear."""
    if not game or game.get("winner_player_idx") is None:
        return False
    turn_hits = game.get("turn_hits") or [None, None, None]
    return any(h is not None for h in turn_hits)


def _player_color(pi: int) -> Tuple[int, int, int]:
    return PLAYER_COLORS_BGR[int(pi) % len(PLAYER_COLORS_BGR)]


def _player_color_from_state(pstate: Optional[Dict], fallback_pi: int) -> Tuple[int, int, int]:
    """Boja iz originalnog slota (P1 crvena … P4 narančasta), ne sabijenog indexa u igri."""
    slot = fallback_pi
    if isinstance(pstate, dict):
        raw = pstate.get("color_slot", fallback_pi)
        try:
            slot = int(raw)
        except (TypeError, ValueError):
            slot = fallback_pi
    return _player_color(slot)


def _player_color_is_dark(bgr: Tuple[int, int, int]) -> bool:
    b, g, r = bgr
    return (int(r) + int(g) + int(b)) < 60


def _player_label_bgr(pi: int, pstate: Optional[Dict] = None) -> Tuple[int, int, int]:
    """BGR for text/icons: black player uses white for contrast on dark UI."""
    c = _player_color_from_state(pstate, pi)
    if _player_color_is_dark(c):
        return (255, 255, 255)
    return c


def _player_place_rank(game: Dict, pi: int) -> Optional[int]:
    placements = game.get("placements") or []
    if pi in placements:
        return placements.index(pi) + 1
    # Killer mid-game: provisional place from elimination order (first out = last place).
    if str(game.get("mode", "")).lower() == "killer":
        elim = game.get("eliminated_order") or []
        if pi in elim:
            npl = int(game.get("num_players", 1))
            return int(npl) - int(elim.index(pi))
    return None


def _player_score_text(game: Dict, pi: int, pstate: Dict) -> str:
    rank = _player_place_rank(game, pi)
    if rank is not None:
        return f"#{rank}"
    if str(game.get("mode", "")).lower() == "killer" or "lives" in pstate:
        num = pstate.get("number")
        lives = int(pstate.get("lives", 0))
        if num is None:
            return f"?/{lives}"
        tag = "K" if pstate.get("is_killer") else ""
        return f"{tag}{int(num)}:{lives}"
    if "next_target_idx" in pstate:
        targets = _around_targets(game)
        if pstate.get("completed"):
            return "DONE"
        nxt = int(pstate.get("next_target_idx", 0))
        if nxt < 0 or nxt >= len(targets):
            return "DONE"
        want = targets[nxt]
        return "BULL" if str(want).lower() == "bull" else str(want)
    if str(game.get("mode", "")).lower() in ("halve", "halve_it"):
        return str(int(pstate.get("score", 0) or 0))
    return str(pstate.get("remaining", pstate.get("points", pstate.get("score", pstate.get("progress", 0)))))


def _around_targets(game: Dict) -> List:
    """Shared Around-the-World target sequence (ints 1–20 or 'bull')."""
    cached = game.get("around_targets")
    if isinstance(cached, list) and len(cached) > 0:
        out = []
        for t in cached:
            if str(t).lower() == "bull":
                out.append("bull")
            else:
                out.append(int(t))
        return out
    rules = game.get("rules") or {}
    order = str(rules.get("around_order", "standard")).lower()
    if order == "reverse":
        # Conventional reverse: bull first, then 20→1.
        return ["bull"] + list(range(20, 0, -1))
    return list(range(1, 21)) + ["bull"]


def _around_build_targets(order: str) -> List:
    order = str(order or "standard").lower()
    if order == "random":
        seq: List = list(range(1, 21)) + ["bull"]
        random.shuffle(seq)
        return seq
    if order == "reverse":
        return ["bull"] + list(range(20, 0, -1))
    return list(range(1, 21)) + ["bull"]


def _around_hit_matches(h: Optional[DetectedHit], want, required_mult: str) -> bool:
    """True if dart hits the required ATW target with the required multiplier.

    Bull mapping: Any = 25 or 50; Single = outer (25); Double/Triple = inner (50)
    (no triple-bull exists). Numbers 1–20 use zone single/double/triple; Any accepts all.
    """
    if h is None or int(getattr(h.hit_score, "score", 0) or 0) <= 0:
        return False
    seg = int(getattr(h.hit_score, "segment_number", 0) or 0)
    zn = str(getattr(h.hit_score, "zone_name", "single") or "single").lower()
    req = str(required_mult or "any").lower()
    if req not in ("single", "double", "triple", "any"):
        req = "any"

    if str(want).lower() == "bull":
        is_outer = seg == 25 or zn == "outer_bull"
        is_inner = seg == 50 or zn == "inner_bull"
        if not (is_outer or is_inner):
            return False
        if req == "any":
            return True
        if req == "single":
            return is_outer
        # double / triple → inner bull only
        return is_inner

    if seg != int(want):
        return False
    mult = _hit_zone_mult(h)
    if req == "any":
        return mult in (1, 2, 3)
    if req == "single":
        return mult == 1
    if req == "double":
        return mult == 2
    if req == "triple":
        return mult == 3
    return False


def _around_apply_hits(
    pstate: Dict,
    targets: List,
    rules: Dict,
    hits: List[Optional[DetectedHit]],
) -> None:
    """Advance through ATW targets per dart; backstep only for a failed turn.

    Success (correct target + mult): +1 step immediately.
    Mid-turn misses do not move the index.
    After a full 3-dart turn with zero successful advances: subtract
    around_backstep steps (0 = Off). Index never goes below 0 (start of sequence).
    """
    backstep = int(rules.get("around_backstep", 0) or 0)
    if backstep not in (0, 1, 2):
        backstep = 0
    req_mult = str(rules.get("around_mult", "any")).lower()
    advanced = False
    for h in hits:
        if pstate.get("completed"):
            break
        if h is None:
            continue
        nxt = int(pstate.get("next_target_idx", 0))
        if nxt >= len(targets):
            pstate["completed"] = True
            break
        want = targets[nxt]
        if _around_hit_matches(h, want, req_mult):
            pstate["next_target_idx"] = nxt + 1
            advanced = True
            if int(pstate["next_target_idx"]) >= len(targets):
                pstate["completed"] = True
        # Miss / wrong target: no per-dart backstep — wait for end of turn.
    # Whole-turn penalty: only when all three darts are in and none advanced.
    if (
        backstep > 0
        and not advanced
        and not pstate.get("completed")
        and _turn_fully_entered(hits)
    ):
        nxt = int(pstate.get("next_target_idx", 0))
        pstate["next_target_idx"] = max(0, nxt - backstep)


# ---------------------------------------------------------------------------
# Halve It
# ---------------------------------------------------------------------------

def _halve_build_targets(mode: str) -> List[Dict]:
    """Target sequence for Halve It.

    Each entry: label (UI), want (seg int | 'bull' | 'any_double' | 'any_triple'),
    mult ('any'|'double'|'triple').

    Extended order:
      20→15, ANY DOUBLE, 14→7, ANY TRIPLE, 6→1, BULL
    (not specific D14–D7 / T6–T1).
    """
    mode = str(mode or "standard").lower()
    if mode not in ("standard", "extended"):
        mode = "standard"
    out: List[Dict] = []
    for n in (20, 19, 18, 17, 16, 15):
        out.append({"label": str(n), "want": n, "mult": "any"})
    if mode == "extended":
        out.append({"label": "ANY DOUBLE", "want": "any_double", "mult": "double"})
        for n in range(14, 6, -1):
            out.append({"label": str(n), "want": n, "mult": "any"})
        out.append({"label": "ANY TRIPLE", "want": "any_triple", "mult": "triple"})
        for n in range(6, 0, -1):
            out.append({"label": str(n), "want": n, "mult": "any"})
    out.append({"label": "BULL", "want": "bull", "mult": "any"})
    return out


def _halve_targets(game: Dict) -> List[Dict]:
    cached = game.get("halve_targets")
    if isinstance(cached, list) and cached:
        return list(cached)
    rules = game.get("rules") or {}
    return _halve_build_targets(str(rules.get("halve_mode", "standard")))


def _halve_current_target(game: Dict) -> Optional[Dict]:
    targets = _halve_targets(game)
    ri = int(game.get("round_number", 1)) - 1
    if ri < 0 or ri >= len(targets):
        return None
    return targets[ri]


def _halve_target_display_label(target: Dict) -> str:
    """Localized label for live UI (ANY DOUBLE / ANY TRIPLE / number / BULL)."""
    want = str(target.get("want") or "").lower()
    raw = str(target.get("label") or "")
    try:
        from kiosk_settings import get_settings

        t = get_settings().t
    except Exception:
        t = None
    if want == "any_double" or raw.upper() in ("ANY DOUBLE", "D"):
        return t("halve_any_double") if t else "ANY DOUBLE"
    if want == "any_triple" or raw.upper() in ("ANY TRIPLE", "T"):
        return t("halve_any_triple") if t else "ANY TRIPLE"
    return raw


def _halve_hit_points(h: Optional[DetectedHit], target: Dict) -> int:
    """Points for one dart on the current Halve It target (0 if invalid).

    Number rounds: S/D/T of that number → face value × mult (hit score).
    ANY DOUBLE: any double on the board (incl. inner bull) → 2×n / 50.
    ANY TRIPLE: any triple → 3×n.
    BULL: 25 / 50.
    """
    if h is None or target is None:
        return 0
    pts = int(getattr(h.hit_score, "score", 0) or 0)
    if pts <= 0:
        return 0
    want = target.get("want")
    want_s = str(want).lower() if want is not None else ""
    mult = str(target.get("mult", "any")).lower()

    if want_s == "any_double" or mult == "any_double":
        # Any double ring 1–20, or double-bull (inner).
        return pts if _is_double_hit(h) else 0
    if want_s == "any_triple" or mult == "any_triple":
        return pts if _hit_zone_mult(h) == 3 else 0
    if not _around_hit_matches(h, want, mult):
        return 0
    return pts


def _halve_turn_points(hits: List[Optional[DetectedHit]], target: Dict) -> int:
    return sum(_halve_hit_points(h, target) for h in hits)


def _apply_halve_turn(
    game: Dict, cur_idx: int, final_hits: List[Optional[DetectedHit]]
) -> None:
    players = game.get("players") or []
    if cur_idx < 0 or cur_idx >= len(players):
        return
    target = _halve_current_target(game)
    pstate = players[cur_idx]
    if target is None:
        pstate["last_turn"] = final_hits
        pstate["score_just_halved"] = False
        return
    turn_pts = _halve_turn_points(final_hits, target)
    score = int(pstate.get("score", 0) or 0)
    fully = _turn_fully_entered(final_hits)
    if turn_pts > 0:
        pstate["score"] = score + turn_pts
        pstate["score_just_halved"] = False
    elif fully:
        # Classic Halve It: miss the round target with all 3 → floor(score / 2).
        pstate["score"] = score // 2
        pstate["score_just_halved"] = True
    else:
        pstate["score_just_halved"] = False
    pstate["last_turn"] = final_hits


def _halve_set_winner(game: Dict) -> None:
    if game.get("winner_player_idx") is not None:
        return
    players = game.get("players") or []
    if not players:
        return
    order = sorted(
        range(len(players)),
        key=lambda i: (-int(players[i].get("score", 0) or 0), i),
    )
    game["placements"] = list(order)
    game["winner_player_idx"] = int(order[0])
    print(f"[game] Halve It pobjednik P{int(order[0]) + 1}", flush=True)


def _halve_live_score_line(game: Dict) -> Optional[str]:
    target = _halve_current_target(game)
    if target is None:
        return None
    targets = _halve_targets(game)
    ri = int(game.get("round_number", 1))
    n = len(targets)
    label = _halve_target_display_label(target)
    return f"{ri}/{n}  ·  {label}"


def _play_for_placements(game: Dict) -> bool:
    # Killer always skips eliminated players and uses active-set round advance.
    if str(game.get("mode", "")).lower() == "killer":
        return True
    return int(game.get("num_players", 1)) > 2


def _active_player_indices(game: Dict) -> List[int]:
    npl = int(game.get("num_players", 1))
    if str(game.get("mode", "")).lower() == "killer":
        players = game.get("players") or []
        return [
            i
            for i in range(npl)
            if i < len(players) and int(players[i].get("lives", 0)) > 0
        ]
    placed = set(game.get("placements") or [])
    return [i for i in range(npl) if i not in placed]


def _next_active_player_idx(game: Dict, cur: int) -> int:
    npl = int(game.get("num_players", 1))
    if not _play_for_placements(game):
        return (cur + 1) % npl
    active = set(_active_player_indices(game))
    if not active:
        return cur
    for step in range(1, npl + 1):
        nxt = (cur + step) % npl
        if nxt in active:
            return nxt
    return cur


def _killer_activation_mult(rules: Dict) -> int:
    act = str(rules.get("killer_activation", "double")).lower()
    if act == "triple":
        return 3
    if act == "single":
        return 1
    return 2


def _hit_zone_mult(h: Optional[DetectedHit]) -> int:
    if h is None:
        return 0
    zn = str(getattr(h.hit_score, "zone_name", "single")).lower()
    if zn == "triple":
        return 3
    if zn == "double":
        return 2
    if zn in ("inner_bull", "outer_bull", "miss"):
        return 0
    return 1


def _hit_matches_killer_activation(h: Optional[DetectedHit], rules: Dict) -> bool:
    """Activation / damage only count on the chosen multiplier (bull never counts)."""
    return _hit_zone_mult(h) == _killer_activation_mult(rules)


def _killer_claim_number_from_hit(h: Optional[DetectedHit]) -> Optional[int]:
    """Gađanjem: any S/D/T on 1–20 claims that sector; bull/miss ignored."""
    if h is None or int(getattr(h.hit_score, "score", 0) or 0) <= 0:
        return None
    seg = int(getattr(h.hit_score, "segment_number", 0) or 0)
    if 1 <= seg <= 20:
        return seg
    return None


def _killer_taken_numbers(game: Dict) -> Set[int]:
    out: Set[int] = set()
    for p in game.get("players") or []:
        num = p.get("number")
        if num is not None:
            out.add(int(num))
    return out


def _killer_number_claimed_in_turn(
    game: Dict, cur_idx: int, turn_hits: List[Optional[DetectedHit]]
) -> bool:
    """True when assign-phase throw already claimed a free number this turn."""
    if str(game.get("mode", "")).lower() != "killer":
        return False
    if str(game.get("killer_phase", "play")).lower() != "assign":
        return False
    players = game.get("players") or []
    if cur_idx < 0 or cur_idx >= len(players):
        return False
    if players[cur_idx].get("number") is not None:
        return True
    taken = _killer_taken_numbers(game)
    for h in turn_hits:
        num = _killer_claim_number_from_hit(h)
        if num is not None and num not in taken:
            return True
    return False


def _killer_assign_needs_retry(
    game: Dict, cur_idx: int, turn_hits: List[Optional[DetectedHit]]
) -> bool:
    """Assign phase: 3 darts in, no unique claim — clear board and retry same player."""
    if str(game.get("mode", "")).lower() != "killer":
        return False
    if str(game.get("killer_phase", "play")).lower() != "assign":
        return False
    if _killer_number_claimed_in_turn(game, cur_idx, turn_hits):
        return False
    return _turn_fully_entered(turn_hits)


def _is_bull_hit(h: Optional[DetectedHit]) -> bool:
    """True for inner (B50) and outer (B25) bull — both trigger Killer life-steal."""
    if h is None:
        return False
    score = int(getattr(h.hit_score, "score", 0) or 0)
    if score <= 0:
        return False
    zn = str(getattr(h.hit_score, "zone_name", "") or "").lower().replace(" ", "_")
    if zn in ("inner_bull", "outer_bull", "bull", "bullseye"):
        return True
    seg = int(getattr(h.hit_score, "segment_number", 0) or 0)
    if seg in (25, 50):
        return True
    # Single-dart scores 25/50 are always bull on a standard board.
    return score in (25, 50)


def _killer_is_play_phase(game: Dict) -> bool:
    return (
        str(game.get("mode", "")).lower() == "killer"
        and str(game.get("killer_phase", "play")).lower() == "play"
    )


def _killer_bull_steal_targets(game: Dict, attacker_idx: int) -> List[int]:
    """Other alive players (bull life-steal targets). Uses live mid-turn lives."""
    players = game.get("players") or []
    out: List[int] = []
    for i, p in enumerate(players):
        if i == attacker_idx:
            continue
        if int(p.get("lives", 0)) > 0:
            out.append(i)
    return out


def _killer_is_killer_before_slot(
    game: Dict,
    cur_idx: int,
    turn_hits: List[Optional[DetectedHit]],
    slot: int,
) -> bool:
    """True if current player is Killer when the dart in `slot` is thrown.

    Uses persisted is_killer OR activation on an earlier dart this same turn
    (slots < slot). Does not mutate game state — mid-turn bull checks must not
    wait for _apply_killer_turn at submit.
    """
    players = game.get("players") or []
    if cur_idx < 0 or cur_idx >= len(players):
        return False
    pstate = players[cur_idx]
    if int(pstate.get("lives", 0)) <= 0:
        return False
    if bool(pstate.get("is_killer")):
        return True
    if not _killer_is_play_phase(game):
        return False
    my_num = pstate.get("number")
    if my_num is None:
        return False
    my_num_i = int(my_num)
    rules = game.get("rules") or {}
    for i, h in enumerate(turn_hits):
        if i >= int(slot):
            break
        if h is None or int(getattr(h.hit_score, "score", 0) or 0) <= 0:
            continue
        if _is_bull_hit(h):
            continue
        seg = int(getattr(h.hit_score, "segment_number", 0) or 0)
        if seg != my_num_i:
            continue
        if _hit_matches_killer_activation(h, rules):
            return True
    return False


def _killer_should_prompt_bull_steal(
    game: Dict,
    cur_idx: int,
    hit: Optional[DetectedHit],
    *,
    turn_hits: Optional[List[Optional[DetectedHit]]] = None,
    slot: Optional[int] = None,
) -> bool:
    """Killer hitting bull with 2+ other alive players → show life-steal picker.

    Exactly 1 other alive target → auto-steal (no picker). Not Killer yet → nothing.
    When turn_hits/slot are provided, becoming Killer earlier this turn counts.
    Lives must already reflect earlier darts this turn (mid-turn resync).
    """
    if not _killer_is_play_phase(game):
        return False
    if not _is_bull_hit(hit):
        return False
    players = game.get("players") or []
    if cur_idx < 0 or cur_idx >= len(players):
        return False
    if int(players[cur_idx].get("lives", 0)) <= 0:
        return False
    hits = turn_hits if turn_hits is not None else list(game.get("turn_hits") or [None, None, None])
    slot_i = int(slot) if slot is not None else 0
    if not _killer_is_killer_before_slot(game, cur_idx, hits, slot_i):
        return False
    return len(_killer_bull_steal_targets(game, cur_idx)) >= 2


def _killer_activation_prefix(rules: Optional[Dict]) -> str:
    act = str((rules or {}).get("killer_activation", "double")).lower()
    if act == "triple":
        return "T"
    if act == "single":
        return "S"
    return "D"


def _killer_lose_life(game: Dict, idx: int) -> None:
    players = game.get("players") or []
    if idx < 0 or idx >= len(players):
        return
    p = players[idx]
    lives = int(p.get("lives", 0))
    if lives <= 0:
        return
    p["lives"] = lives - 1
    if int(p["lives"]) <= 0:
        elim = game.setdefault("eliminated_order", [])
        if idx not in elim:
            elim.append(idx)
            print(f"[game] P{idx + 1} eliminiran (Killer)", flush=True)
        _killer_set_winner_if_any(game)


def _killer_restore_life(game: Dict, idx: int) -> None:
    """Undo one life loss (e.g. cleared bull-steal dart)."""
    players = game.get("players") or []
    if idx < 0 or idx >= len(players):
        return
    p = players[idx]
    max_lives = int((game.get("rules") or {}).get("killer_lives", 3) or 3)
    lives = int(p.get("lives", 0))
    if lives >= max_lives:
        return
    p["lives"] = lives + 1
    elim = game.get("eliminated_order")
    if isinstance(elim, list) and idx in elim and int(p["lives"]) > 0:
        game["eliminated_order"] = [i for i in elim if i != idx]
    alive = [i for i, pl in enumerate(players) if int(pl.get("lives", 0)) > 0]
    if len(alive) > 1 and game.get("winner_player_idx") is not None:
        game["winner_player_idx"] = None
        game["winner_at"] = None
        game["placements"] = []


def _killer_revert_bull_steal_slot(ctx: Dict, game: Dict, slot: int) -> None:
    steals = ctx.get("killer_bull_steals")
    if not isinstance(steals, dict) or slot not in steals:
        return
    target = int(steals.pop(slot))
    _killer_restore_life(game, target)
    pend = ctx.get("killer_bull_pending")
    if isinstance(pend, dict) and int(pend.get("slot", -1)) == slot:
        ctx["killer_bull_pending"] = None


def _killer_apply_one_play_hit(game: Dict, cur_idx: int, hit: Optional[DetectedHit]) -> None:
    """Apply one non-bull Killer play-phase hit to live players (activation / damage).

    Each matching dart costs exactly one life (2× S15 → −2). Becoming Killer on
    the activating dart does not cost a life; later self-hits do.
    """
    if hit is None or int(getattr(hit.hit_score, "score", 0) or 0) <= 0:
        return
    if _is_bull_hit(hit):
        return
    if game.get("winner_player_idx") is not None:
        return
    players = game.get("players") or []
    if cur_idx < 0 or cur_idx >= len(players):
        return
    pstate = players[cur_idx]
    if int(pstate.get("lives", 0)) <= 0:
        return
    rules = game.get("rules") or {}
    seg = int(getattr(hit.hit_score, "segment_number", 0) or 0)
    if not (1 <= seg <= 20):
        return
    if not _hit_matches_killer_activation(hit, rules):
        return

    my_num = pstate.get("number")
    is_killer = bool(pstate.get("is_killer"))

    if not is_killer:
        # Become Killer on own number + activation mult (no self-damage on activate).
        if my_num is not None and seg == int(my_num):
            pstate["is_killer"] = True
            print(f"[game] P{cur_idx + 1} postao Killer", flush=True)
        return

    # Self-hit always on: Killer hitting own number with activation loses a life.
    if my_num is not None and seg == int(my_num):
        _killer_lose_life(game, cur_idx)
        return

    # Friendly fire always on: −1 life per dart on the living owner of that number.
    for oi, op in enumerate(players):
        if oi == cur_idx:
            continue
        if int(op.get("lives", 0)) <= 0:
            continue
        onum = op.get("number")
        if onum is not None and int(onum) == seg:
            _killer_lose_life(game, oi)
            break


def _killer_restore_turn_baseline(ctx: Dict, game: Dict) -> bool:
    """Restore players/elim/winner/placements to turn-start snapshot. False if no snap."""
    snap = ctx.get("_turn_players_snap")
    if not isinstance(snap, list):
        return False
    game["players"] = copy.deepcopy(snap)
    game["eliminated_order"] = copy.deepcopy(ctx.get("_turn_elim_snap") or [])
    game["winner_player_idx"] = ctx.get("_turn_winner_snap")
    game["placements"] = copy.deepcopy(ctx.get("_turn_placements_snap") or [])
    if game.get("winner_player_idx") is None:
        game["winner_at"] = None
    return True


def _killer_apply_bull_steal_for_slot(
    ctx: Dict,
    game: Dict,
    cur_idx: int,
    slot: int,
    hit: Optional[DetectedHit],
    turn_hits: List[Optional[DetectedHit]],
    *,
    prefer_target: Optional[int] = None,
) -> None:
    """Auto-steal (1 target) or open picker (2+). No-op if not Killer / not bull / already stolen."""
    if not _is_bull_hit(hit):
        return
    if not _killer_is_play_phase(game):
        return
    players = game.get("players") or []
    if cur_idx < 0 or cur_idx >= len(players):
        return
    if int(players[cur_idx].get("lives", 0)) <= 0:
        return
    if not _killer_is_killer_before_slot(game, cur_idx, turn_hits, slot):
        return
    steals = ctx.get("killer_bull_steals")
    if isinstance(steals, dict) and int(slot) in steals:
        return
    targets = _killer_bull_steal_targets(game, cur_idx)
    if not targets:
        return
    if prefer_target is not None and int(prefer_target) in targets:
        target = int(prefer_target)
        _killer_lose_life(game, target)
        if not isinstance(steals, dict):
            steals = {}
            ctx["killer_bull_steals"] = steals
        steals[int(slot)] = target
        ctx["killer_bull_pending"] = None
        ctx["_ui_killer_life_flash"] = target
        print(f"[game] Bull steal (kept): P{cur_idx + 1} -> P{target + 1}", flush=True)
        return
    if len(targets) == 1:
        target = int(targets[0])
        _killer_lose_life(game, target)
        if not isinstance(steals, dict):
            steals = {}
            ctx["killer_bull_steals"] = steals
        steals[int(slot)] = target
        ctx["killer_bull_pending"] = None
        ctx["_ui_killer_life_flash"] = target
        print(f"[game] Bull steal (auto): P{cur_idx + 1} -> P{target + 1}", flush=True)
        return
    ctx["killer_bull_pending"] = {"slot": int(slot), "attacker": int(cur_idx)}
    print(f"[game] Killer bull — izbor zivota (P{cur_idx + 1})", flush=True)


def _killer_resync_mid_turn(
    ctx: Dict, game: Dict, cur_idx: int, turn_hits: List[Optional[DetectedHit]]
) -> None:
    """Rebuild live Killer state from turn-start snap + registered hits (incl. bull steals).

    Ensures bull picker / sole-survivor checks see lives after earlier darts this turn.
    """
    if not _killer_is_play_phase(game):
        return
    prev_steals: Dict = {}
    raw = ctx.get("killer_bull_steals")
    if isinstance(raw, dict):
        prev_steals = {int(k): int(v) for k, v in raw.items()}
    if not _killer_restore_turn_baseline(ctx, game):
        # No snap yet — keep legacy bull-only path on current players.
        ctx["killer_bull_pending"] = None
        for key in list(prev_steals.keys()):
            _killer_revert_bull_steal_slot(ctx, game, int(key))
        ctx["killer_bull_steals"] = {}
        for si, h in enumerate(turn_hits):
            if isinstance(ctx.get("killer_bull_pending"), dict):
                break
            prefer = prev_steals.get(int(si))
            _killer_apply_bull_steal_for_slot(
                ctx, game, cur_idx, si, h, turn_hits, prefer_target=prefer
            )
        return

    ctx["killer_bull_steals"] = {}
    ctx["killer_bull_pending"] = None
    for si, h in enumerate(turn_hits):
        if game.get("winner_player_idx") is not None:
            break
        players = game.get("players") or []
        if cur_idx < 0 or cur_idx >= len(players):
            break
        if int(players[cur_idx].get("lives", 0)) <= 0:
            break
        if h is None:
            continue
        if _is_bull_hit(h):
            prefer = prev_steals.get(int(si))
            _killer_apply_bull_steal_for_slot(
                ctx, game, cur_idx, si, h, turn_hits, prefer_target=prefer
            )
            if isinstance(ctx.get("killer_bull_pending"), dict):
                break
            continue
        _killer_apply_one_play_hit(game, cur_idx, h)
    _killer_set_winner_if_any(game)
    # Flash a victim when this turn removed lives (number hits / self-hits).
    # Bull steals set _ui_killer_life_flash themselves; don't overwrite.
    if ctx.get("_ui_killer_life_flash") is None:
        base = ctx.get("_turn_players_snap")
        if isinstance(base, list):
            flash_i = None
            for i, p in enumerate(game.get("players") or []):
                if i >= len(base):
                    break
                if int(p.get("lives", 0)) < int(base[i].get("lives", 0)):
                    flash_i = i
            if flash_i is not None:
                ctx["_ui_killer_life_flash"] = flash_i


def _killer_after_hit_registered(
    ctx: Dict, game: Dict, cur_idx: int, slot: int, hit: Optional[DetectedHit]
) -> None:
    """After a turn slot changes: resync live Killer lives/activation/bull steals."""
    turn_hits = list(game.get("turn_hits") or [None, None, None])
    if 0 <= int(slot) < 3:
        turn_hits[int(slot)] = hit
        game["turn_hits"] = turn_hits
    _killer_resync_mid_turn(ctx, game, cur_idx, turn_hits)


def _killer_confirm_bull_steal(ctx: Dict, game: Dict, target_idx: int) -> bool:
    pend = ctx.get("killer_bull_pending")
    if not isinstance(pend, dict):
        return False
    slot = int(pend.get("slot", -1))
    attacker = int(pend.get("attacker", -1))
    targets = _killer_bull_steal_targets(game, attacker)
    if target_idx not in targets:
        return False
    steals = ctx.setdefault("killer_bull_steals", {})
    if not isinstance(steals, dict):
        steals = {}
        ctx["killer_bull_steals"] = steals
    steals[slot] = int(target_idx)
    ctx["killer_bull_pending"] = None
    ctx["_ui_killer_life_flash"] = int(target_idx)
    print(f"[game] Bull steal: P{attacker + 1} -> P{target_idx + 1}", flush=True)
    # Resync so later registered darts / sole-survivor state stay consistent.
    turn_hits = list(game.get("turn_hits") or [None, None, None])
    _killer_resync_mid_turn(ctx, game, attacker, turn_hits)
    return True


def _killer_false_bull(ctx: Dict, game: Dict) -> bool:
    """False bull detection — treat dart as miss; no life stolen."""
    pend = ctx.get("killer_bull_pending")
    if not isinstance(pend, dict):
        return False
    slot = int(pend.get("slot", -1))
    attacker = int(pend.get("attacker", -1))
    ctx["killer_bull_pending"] = None
    turn_hits = list(game.get("turn_hits", [None, None, None]))
    if 0 <= slot < 3:
        turn_hits[slot] = _make_miss_hit()
        game["turn_hits"] = turn_hits
    print("[game] Killer bull — false detection (MISS)", flush=True)
    cur = attacker if attacker >= 0 else 0
    _killer_resync_mid_turn(ctx, game, cur, turn_hits)
    return True


def _killer_set_winner_if_any(game: Dict) -> None:
    """Last player with lives > 0 wins. Sequential hit processing avoids same-dart ties."""
    if game.get("winner_player_idx") is not None:
        return
    if str(game.get("killer_phase", "play")).lower() == "assign":
        return
    players = game.get("players") or []
    if not players:
        return
    alive = [i for i, p in enumerate(players) if int(p.get("lives", 0)) > 0]
    if len(alive) > 1:
        return
    elim = list(game.get("eliminated_order") or [])
    if len(alive) == 1:
        # Solo game: never auto-win just for being the only player.
        if int(game.get("num_players", 1)) <= 1:
            return
        winner = alive[0]
    else:
        # Should be rare (processed dart-by-dart). Prefer survivor who lasted longest.
        if len(elim) >= 2:
            winner = int(elim[-2])
        elif len(elim) == 1:
            winner = int(elim[-1])
        else:
            return
    rest = [i for i in reversed(elim) if i != winner]
    others = [i for i in range(len(players)) if i != winner and i not in rest]
    game["placements"] = [winner] + rest + others
    game["winner_player_idx"] = winner
    print(f"[game] Killer pobjednik P{winner + 1}", flush=True)


def _killer_commit_turn(
    ctx: Dict, game: Dict, cur_idx: int, final_hits: List[Optional[DetectedHit]]
) -> None:
    """Commit Killer turn once from turn-start baseline + confirmed bull steals."""
    steals = ctx.get("killer_bull_steals")
    steals_copy = (
        {int(k): int(v) for k, v in steals.items()} if isinstance(steals, dict) else {}
    )
    _killer_restore_turn_baseline(ctx, game)
    _apply_killer_turn(game, cur_idx, final_hits, bull_steals=steals_copy)


def _apply_killer_turn(
    game: Dict,
    cur_idx: int,
    final_hits: List[Optional[DetectedHit]],
    *,
    bull_steals: Optional[Dict[int, int]] = None,
) -> None:
    players = game.get("players") or []
    if cur_idx < 0 or cur_idx >= len(players):
        return
    pstate = players[cur_idx]
    phase = str(game.get("killer_phase", "play")).lower()

    if phase == "assign":
        # Claim first free sector 1–20; duplicate / bull / miss → ignore that dart.
        taken = _killer_taken_numbers(game)
        if pstate.get("number") is None:
            for h in final_hits:
                num = _killer_claim_number_from_hit(h)
                if num is None or num in taken:
                    continue
                pstate["number"] = int(num)
                taken.add(int(num))
                print(f"[game] P{cur_idx + 1} uzeo broj {num}", flush=True)
                break
        pstate["last_turn"] = final_hits
        if players and all(p.get("number") is not None for p in players):
            game["killer_phase"] = "play"
            game["round_number"] = 1
            print("[game] Killer: svi brojevi dodijeljeni — play", flush=True)
        return

    steals = bull_steals if isinstance(bull_steals, dict) else {}
    for si, h in enumerate(final_hits):
        if game.get("winner_player_idx") is not None:
            break
        if int(pstate.get("lives", 0)) <= 0:
            break
        if h is None or int(getattr(h.hit_score, "score", 0) or 0) <= 0:
            continue
        # Bull life-steal: apply confirmed mid-turn choices (not re-prompted at commit).
        if _is_bull_hit(h):
            if int(si) in steals:
                _killer_lose_life(game, int(steals[int(si)]))
            continue
        _killer_apply_one_play_hit(game, cur_idx, h)

    pstate["last_turn"] = final_hits
    _killer_set_winner_if_any(game)


def _record_player_finish(game: Dict, cur_idx: int) -> None:
    """Zabiljezi pobjedu; kod 3+ igraca nastavlja bez pobjednika."""
    if not _play_for_placements(game):
        game["winner_player_idx"] = cur_idx
        npl = int(game.get("num_players", 1))
        if npl == 2:
            game["placements"] = [cur_idx, (cur_idx + 1) % 2]
        return
    placements = game.setdefault("placements", [])
    if cur_idx in placements:
        return
    placements.append(cur_idx)
    place = len(placements)
    print(f"[game] P{cur_idx + 1} zavrsio na {place}. mjestu", flush=True)
    npl = int(game.get("num_players", 1))
    unplaced = [i for i in range(npl) if i not in placements]
    if len(unplaced) <= 1:
        if len(unplaced) == 1:
            placements.append(unplaced[0])
            print(f"[game] P{unplaced[0] + 1} zavrsio na {len(placements)}. mjestu", flush=True)
        if placements:
            game["winner_player_idx"] = placements[0]


def _cricket_round_limit_finish(game: Dict) -> None:
    rules = game.get("rules") or {}
    if not rules.get("max_rounds_20"):
        return
    limit = int(rules.get("max_rounds_limit", 20))
    if int(game.get("round_number", 1)) <= limit:
        return
    if not _play_for_placements(game):
        _cricket_round_limit_winner(game)
        return
    players = game.get("players") or []
    if not players:
        return
    placements = game.setdefault("placements", [])
    unplaced = [i for i in range(len(players)) if i not in placements]
    cricket_mode = str(rules.get("cricket_mode", "standard")).lower()
    if cricket_mode == "cut_throat":
        unplaced.sort(key=lambda i: int(players[i].get("points", 0)))
    else:
        unplaced.sort(key=lambda i: int(players[i].get("points", 0)), reverse=True)
    for i in unplaced:
        placements.append(i)
    if placements:
        game["winner_player_idx"] = placements[0]


def _leaderboard_entries(game: Dict) -> List[Tuple[int, int]]:
    placements = game.get("placements") or []
    if placements:
        return [(rank + 1, pi) for rank, pi in enumerate(placements)]
    winner = game.get("winner_player_idx")
    if winner is None:
        return []
    npl = int(game.get("num_players", 1))
    win_pi = int(winner)
    rest = [i for i in range(npl) if i != win_pi]
    return [(1, win_pi)] + [(rank + 2, pi) for rank, pi in enumerate(rest)]


def _create_game(
    mode: str,
    n: int,
    rules_pending: Dict,
    player_names: Optional[List[str]] = None,
    color_slots: Optional[List[int]] = None,
) -> Dict:
    players = []
    killer_phase = "play"
    names = [str(x or "")[:8] for x in (player_names or [])]
    for _ in range(n):
        if mode in ("301", "501"):
            players.append({"remaining": int(mode), "last_turn": [None, None, None], "is_in": False})
        elif mode in ("around", "around_the_world"):
            players.append({"next_target_idx": 0, "completed": False, "last_turn": [None, None, None]})
        elif mode == "cricket":
            players.append(
                {
                    "points": 0,
                    "wickets": {15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 25: 0},
                    "last_turn": [None, None, None],
                }
            )
        elif mode == "killer":
            lives = int(rules_pending.get("killer_lives", 3) or 3)
            if lives not in (3, 5, 7, 10):
                lives = 3
            players.append(
                {
                    "number": None,
                    "lives": lives,
                    "is_killer": False,
                    "last_turn": [None, None, None],
                }
            )
        elif mode in ("halve", "halve_it"):
            players.append({"score": 0, "last_turn": [None, None, None]})
        else:
            players.append({"remaining": 0, "last_turn": [None, None, None]})

    for i, p in enumerate(players):
        p["name"] = names[i] if i < len(names) else ""
        if color_slots is not None and i < len(color_slots):
            try:
                p["color_slot"] = int(color_slots[i])
            except (TypeError, ValueError):
                p["color_slot"] = i
        else:
            p["color_slot"] = i

    if mode == "killer":
        assign = str(rules_pending.get("killer_assign", "random")).lower()
        if assign == "throw":
            killer_phase = "assign"
        else:
            # Random: unique sectors 1–20 at start; nobody starts as Killer.
            nums = random.sample(range(1, 21), n)
            for i, p in enumerate(players):
                p["number"] = int(nums[i])
            killer_phase = "play"

    act = str(rules_pending.get("killer_activation", "double")).lower()
    if act not in ("single", "double", "triple"):
        act = "double"
    lives_rule = int(rules_pending.get("killer_lives", 3) or 3)
    if lives_rule not in (3, 5, 7, 10):
        lives_rule = 3

    around_mult = str(rules_pending.get("around_mult", "any")).lower()
    if around_mult not in ("single", "double", "triple", "any"):
        around_mult = "any"
    around_order = str(rules_pending.get("around_order", "standard")).lower()
    if around_order not in ("standard", "random", "reverse"):
        around_order = "standard"
    around_backstep = int(rules_pending.get("around_backstep", 0) or 0)
    if around_backstep not in (0, 1, 2):
        around_backstep = 0

    around_targets = None
    if mode in ("around", "around_the_world"):
        # One shared sequence for all players (incl. shuffled random order).
        around_targets = _around_build_targets(around_order)

    halve_mode = str(rules_pending.get("halve_mode", "standard")).lower()
    if halve_mode not in ("standard", "extended"):
        halve_mode = "standard"
    halve_targets = None
    if mode in ("halve", "halve_it"):
        halve_targets = _halve_build_targets(halve_mode)

    game_rules = {
        "double_in": bool(rules_pending.get("double_in")),
        "double_out": bool(rules_pending.get("double_out")),
        "max_rounds_20": bool(rules_pending.get("max_rounds_20")),
        "max_rounds_limit": 20,
        "cricket_mode": str(rules_pending.get("cricket_mode", "standard")),
        "killer_lives": lives_rule,
        "killer_assign": str(rules_pending.get("killer_assign", "random")).lower(),
        "killer_activation": act,
        "around_mult": around_mult,
        "around_order": around_order,
        "around_backstep": around_backstep,
        "halve_mode": halve_mode,
    }
    return {
        "mode": mode,
        "num_players": n,
        "players": players,
        "turn_hits": [None, None, None],
        "winner_player_idx": None,
        "winner_at": None,
        "placements": [],
        "eliminated_order": [],
        "killer_phase": killer_phase,
        "rules": game_rules,
        "around_targets": around_targets,
        "halve_targets": halve_targets,
        "round_number": 1,
        "undo_stack": [],
    }


def _dart_tag_from_score(score: int, double_out: bool, is_last: bool) -> Optional[str]:
    if score <= 0:
        return None
    if double_out and is_last:
        if score == 50:
            return "B50"
        if score % 2 == 0 and score <= 40 and score // 2 <= 20:
            return f"D{score // 2}"
        return None
    if score == 50:
        return "B50"
    if score == 25:
        return "B25"
    if score <= 20:
        return f"S{score}"
    if score % 3 == 0 and score <= 60 and score // 3 <= 20:
        return f"T{score // 3}"
    if score % 2 == 0 and score <= 40 and score // 2 <= 20:
        return f"D{score // 2}"
    return None


def _find_checkout_path(remaining: int, darts_left: int, double_out: bool) -> Optional[List[str]]:
    if remaining > 170 or remaining <= 0 or darts_left <= 0:
        return None
    if double_out and remaining == 2:
        return ["D1"] if darts_left >= 1 else None

    def _solve(rem: int, left: int, require_double: bool) -> Optional[List[int]]:
        if rem < 0 or left <= 0:
            return None
        if rem == 0:
            return [] if not require_double else None
        if left == 1:
            if require_double:
                if rem == 50:
                    return [50]
                if rem <= 40 and rem % 2 == 0 and rem // 2 <= 20:
                    return [rem]
            else:
                if rem <= 20:
                    return [rem]
                if rem <= 40 and rem % 2 == 0 and rem // 2 <= 20:
                    return [rem]
                if rem <= 60 and rem % 3 == 0 and rem // 3 <= 20:
                    return [rem]
                if rem == 25:
                    return [25]
                if rem == 50:
                    return [50]
            return None
        scores: List[int] = []
        for n in range(20, 0, -1):
            scores.extend([3 * n, 2 * n, n])
        scores.extend([50, 25])
        for sc in scores:
            if sc <= 0 or sc > rem:
                continue
            is_dbl = sc == 50 or (sc % 2 == 0 and sc <= 40 and sc // 2 <= 20)
            sub = _solve(rem - sc, left - 1, require_double and not is_dbl)
            if sub is not None:
                return [sc] + sub
        return None

    path = _solve(remaining, darts_left, double_out)
    if not path:
        return None
    tags: List[str] = []
    for i, sc in enumerate(path):
        is_last = i == len(path) - 1
        tag = _dart_tag_from_score(sc, double_out, is_last)
        if tag is None:
            return None
        tags.append(tag)
    return tags


def _x01_live_score_line(game: Dict, cur_idx: int) -> Optional[str]:
    """Checkout prijedlog (bez live remaining — scoreboard se sam animira)."""
    mode = game.get("mode")
    if mode not in ("301", "501"):
        return None
    rules = game.get("rules") or {}
    double_out = bool(rules.get("double_out"))
    pstate = game["players"][cur_idx]
    remaining = int(pstate.get("remaining", 0))
    turn_hits = game.get("turn_hits", [None, None, None])
    entered = [h for h in turn_hits if h is not None]
    spent = sum(h.hit_score.score for h in entered)
    left_score = remaining - spent
    darts_left = 3 - len(entered)
    if left_score <= 0 or darts_left <= 0 or left_score > 170:
        return None

    path = _find_checkout_path(left_score, darts_left, double_out)
    if not path:
        return None
    return ", ".join(tag.upper() for tag in path)


def _turn_fully_entered(turn_hits: List[Optional[DetectedHit]]) -> bool:
    return all(h is not None for h in turn_hits)


def _x01_turn_is_bust(game: Dict, cur_idx: int, turn_hits: List[Optional[DetectedHit]]) -> bool:
    """True when entered darts (1–3) would bust the leg."""
    if str(game.get("mode", "")).lower() not in ("301", "501"):
        return False
    entered = [h for h in turn_hits if h is not None]
    if not entered:
        return False
    rules = game.get("rules") or {}
    double_in = bool(rules.get("double_in"))
    double_out = bool(rules.get("double_out"))
    pstate = game["players"][cur_idx]
    if "remaining" not in pstate:
        return False
    start_remaining = int(pstate["remaining"])
    player_is_in = bool(pstate.get("is_in"))
    scoring_hits = _filter_x01_turn_hits(entered, double_in, player_is_in)
    total_sub = sum(h.hit_score.score for h in scoring_hits)
    new_remaining = start_remaining - int(total_sub)
    if new_remaining < 0:
        return True
    if double_out and new_remaining == 1:
        return True
    if new_remaining == 0 and double_out:
        if not scoring_hits or not _is_double_hit(scoring_hits[-1]):
            return True
    return False


def _x01_turn_is_valid_checkout(game: Dict, cur_idx: int, turn_hits: List[Optional[DetectedHit]]) -> bool:
    """True when entered darts (1–3) legally finish the leg (incl. double-out on last scoring dart)."""
    if str(game.get("mode", "")).lower() not in ("301", "501"):
        return False
    entered = [h for h in turn_hits if h is not None]
    if not entered:
        return False
    rules = game.get("rules") or {}
    double_in = bool(rules.get("double_in"))
    double_out = bool(rules.get("double_out"))
    pstate = game["players"][cur_idx]
    if "remaining" not in pstate:
        return False
    start_remaining = int(pstate["remaining"])
    player_is_in = bool(pstate.get("is_in"))
    scoring_hits = _filter_x01_turn_hits(entered, double_in, player_is_in)
    if double_in and not player_is_in and not scoring_hits:
        return False
    total_sub = sum(h.hit_score.score for h in scoring_hits)
    new_remaining = start_remaining - int(total_sub)
    if new_remaining != 0:
        return False
    if double_out:
        if not scoring_hits or not _is_double_hit(scoring_hits[-1]):
            return False
    return True


def _around_turn_completes(game: Dict, cur_idx: int, turn_hits: List[Optional[DetectedHit]]) -> bool:
    """True kad Around the World zavrsi unutar ove ruke (npr. bull s 1. strijelom)."""
    if str(game.get("mode", "")).lower() not in ("around", "around_the_world"):
        return False
    players = game.get("players") or []
    if cur_idx < 0 or cur_idx >= len(players):
        return False
    pstate = dict(players[cur_idx])
    if pstate.get("completed"):
        return False
    targets = _around_targets(game)
    rules = game.get("rules") or {}
    start_idx = int(pstate.get("next_target_idx", 0))
    _around_apply_hits(pstate, targets, rules, turn_hits)
    return bool(pstate.get("completed")) and int(pstate.get("next_target_idx", 0)) > start_idx


def _killer_turn_ends_early(
    game: Dict, cur_idx: int, turn_hits: List[Optional[DetectedHit]]
) -> bool:
    """True if Killer play should finish before 3 darts.

    Uses live mid-turn player state (resync applies number hits + bull steals):
    - current player at 0 lives (self-elim), or
    - only one player left alive (sole survivor / winner decided).
    """
    if not _killer_is_play_phase(game):
        return False
    players = game.get("players") or []
    if cur_idx < 0 or cur_idx >= len(players):
        return False
    if not any(h is not None for h in turn_hits):
        return False
    if int(players[cur_idx].get("lives", 0)) <= 0:
        return True
    if int(game.get("num_players", 1)) <= 1:
        return False
    if game.get("winner_player_idx") is not None:
        return True
    alive = [i for i, p in enumerate(players) if int(p.get("lives", 0)) > 0]
    return len(alive) <= 1


def _killer_current_dies_in_turn(
    game: Dict, cur_idx: int, turn_hits: List[Optional[DetectedHit]]
) -> bool:
    """Backward-compatible alias: current player eliminated mid-turn."""
    if not _killer_is_play_phase(game):
        return False
    players = game.get("players") or []
    if cur_idx < 0 or cur_idx >= len(players):
        return False
    if not any(h is not None for h in turn_hits):
        return False
    return int(players[cur_idx].get("lives", 0)) <= 0


def _turn_is_early_finish(game: Dict, cur_idx: int, turn_hits: List[Optional[DetectedHit]]) -> bool:
    """Checkout / Around finish / Killer assign claim / Killer mid-turn end prije 3. strijele."""
    return (
        _x01_turn_is_valid_checkout(game, cur_idx, turn_hits)
        or _around_turn_completes(game, cur_idx, turn_hits)
        or _killer_number_claimed_in_turn(game, cur_idx, turn_hits)
        or _killer_turn_ends_early(game, cur_idx, turn_hits)
    )


def _turn_can_submit(game: Dict, cur_idx: int, turn_hits: List[Optional[DetectedHit]]) -> bool:
    mode = str(game.get("mode", "")).lower()
    # Killer assign: advance ONLY after a successful unique claim — never on
    # 3 duplicates/misses (those must clear board and retry the same player).
    if mode == "killer" and str(game.get("killer_phase", "play")).lower() == "assign":
        return _killer_number_claimed_in_turn(game, cur_idx, turn_hits)
    if _turn_fully_entered(turn_hits):
        return True
    if mode in ("301", "501"):
        if _x01_turn_is_valid_checkout(game, cur_idx, turn_hits):
            return True
        if _x01_turn_is_bust(game, cur_idx, turn_hits):
            return True
    if mode in ("around", "around_the_world"):
        if _around_turn_completes(game, cur_idx, turn_hits):
            return True
    if mode == "killer" and _killer_turn_ends_early(game, cur_idx, turn_hits):
        return True
    return False


def _x01_turn_blocks_input(game: Dict, cur_idx: int, turn_hits: List[Optional[DetectedHit]]) -> bool:
    return _x01_turn_is_valid_checkout(game, cur_idx, turn_hits) or _x01_turn_is_bust(
        game, cur_idx, turn_hits
    )


def _x01_checkout_blocks_input(game: Dict, cur_idx: int, turn_hits: List[Optional[DetectedHit]]) -> bool:
    return _x01_turn_blocks_input(game, cur_idx, turn_hits)


def _turn_blocks_new_hits(game: Dict, cur_idx: int, turn_hits: List[Optional[DetectedHit]]) -> bool:
    """True kad bust/checkout/Around/Killer-assign/Killer mid-turn finish prije 3. pikadera — prazni slotovi zakljucani."""
    if (
        _x01_turn_blocks_input(game, cur_idx, turn_hits)
        or _around_turn_completes(game, cur_idx, turn_hits)
        or _killer_number_claimed_in_turn(game, cur_idx, turn_hits)
        or _killer_turn_ends_early(game, cur_idx, turn_hits)
    ):
        return not _turn_fully_entered(turn_hits)
    return False


def _turn_hits_display_line(game: Dict, cur_idx: int, turn_hits: List[Optional[DetectedHit]]) -> str:
    if _turn_is_early_finish(game, cur_idx, turn_hits):
        return ",".join(_hit_tag(h) for h in turn_hits if h is not None)
    return ",".join(_hit_tag(turn_hits[i]) for i in range(3))


def _game_display_title(game: Dict) -> str:
    mode = str(game.get("mode", "GAME")).lower()
    rules = game.get("rules") or {}
    if mode in ("301", "501"):
        parts: List[str] = []
        if rules.get("double_in"):
            parts.append("DOUBLE IN")
        if rules.get("double_out"):
            parts.append("DOUBLE OUT")
        if parts:
            return f"{mode.upper()} - {', '.join(parts)}"
        return mode.upper()
    if mode == "cricket":
        cmode = str(rules.get("cricket_mode", "standard")).lower()
        mode_labels = {
            "standard": "STANDARD",
            "no_score": "NO SCORE",
            "cut_throat": "CUT THROAT",
        }
        return f"CRICKET - {mode_labels.get(cmode, cmode.upper())}"
    if mode == "killer":
        act = str(rules.get("killer_activation", "double")).upper()
        phase = str(game.get("killer_phase", "play")).lower()
        if phase == "assign":
            return f"KILLER - ASSIGN ({act})"
        return f"KILLER - {act}"
    if mode in ("around", "around_the_world"):
        mult = str(rules.get("around_mult", "any")).upper()
        order = str(rules.get("around_order", "standard")).upper()
        return f"AROUND - {mult}/{order}"
    if mode in ("halve", "halve_it"):
        hmode = str(rules.get("halve_mode", "standard")).upper()
        return f"HALVE IT - {hmode}"
    return str(game.get("mode", "GAME")).upper()


def _killer_live_score_line(game: Dict, cur_idx: int) -> Optional[str]:
    """Killer has no LiveScore hint text (number + lives shown on scoreboard)."""
    _ = game, cur_idx
    return None


def _push_undo_after_advance(
    game: Dict,
    finished_idx: int,
    finished_hits: List[Optional[DetectedHit]],
    next_idx: int,
    *,
    players_before: List[Dict],
    round_before: int,
    winner_before: Optional[int],
    winner_at_before: Optional[float],
    placements_before: List[int],
    eliminated_order_before: Optional[List[int]] = None,
    killer_phase_before: Optional[str] = None,
) -> None:
    """Snimka nakon prebacivanja: stanje PRIJE bodovanja zavrsenog poteza."""
    game["undo_stack"] = [
        {
            "finished_player_idx": int(finished_idx),
            "finished_turn_hits": copy.deepcopy(finished_hits),
            "next_player_idx": int(next_idx),
            "players_before": copy.deepcopy(players_before),
            "round_before": int(round_before),
            "winner_player_idx": winner_before,
            "winner_at": winner_at_before,
            "placements": copy.deepcopy(placements_before),
            "eliminated_order": copy.deepcopy(
                eliminated_order_before
                if eliminated_order_before is not None
                else []
            ),
            "killer_phase": killer_phase_before,
        }
    ]


def _enter_undo_rewind(
    game: Dict, gui_state: Dict, gui_ctx: Dict, snap: Dict
) -> None:
    """NAZAD: prikazi zavrsenog igraca (P1) s scoreom PRIJE poteza.

    Sacuvaj prekinute hitove iduceg igraca u `_dart_undo_pending_hits` (zamrznuto —
    ne dopuni ih detekcijom u pozadini). Dok je undo aktivan, nove strijelice idu
    u prazne slotove trenutne (vracene) ruke.
    """
    # Save current (next) player's in-progress darts / bull steals BEFORE rewind.
    interrupted_hits = copy.deepcopy(game.get("turn_hits") or [None, None, None])
    interrupted_steals: Dict[int, int] = {}
    raw_steals = gui_ctx.get("killer_bull_steals")
    if isinstance(raw_steals, dict):
        interrupted_steals = {int(k): int(v) for k, v in raw_steals.items()}
    if "players_before" in snap:
        game["players"] = copy.deepcopy(snap["players_before"])
        game["round_number"] = int(snap.get("round_before", snap.get("round_number", 1)))
    else:
        # Stari undo format (score vec upisan) — degradirani fallback
        game["players"] = copy.deepcopy(snap.get("players") or [])
        game["round_number"] = int(snap.get("round_number", 1))
    game["winner_player_idx"] = snap.get("winner_player_idx")
    game["winner_at"] = snap.get("winner_at")
    game["placements"] = copy.deepcopy(snap.get("placements") or [])
    if "eliminated_order" in snap:
        game["eliminated_order"] = copy.deepcopy(snap.get("eliminated_order") or [])
    if snap.get("killer_phase") is not None:
        game["killer_phase"] = snap.get("killer_phase")
    finished_idx = int(snap["finished_player_idx"])
    gui_state["current_player_idx"] = finished_idx
    game["turn_hits"] = copy.deepcopy(snap["finished_turn_hits"])
    gui_ctx["_dart_undo_target_player"] = int(snap["next_player_idx"])
    gui_ctx["_dart_undo_pending_hits"] = interrupted_hits
    gui_ctx["_dart_undo_pending_bull_steals"] = interrupted_steals
    # Turn baseline for Killer mid-turn resync while editing the finished turn.
    gui_ctx["_turn_players_snap"] = copy.deepcopy(game.get("players") or [])
    gui_ctx["_turn_elim_snap"] = copy.deepcopy(game.get("eliminated_order") or [])
    gui_ctx["_turn_winner_snap"] = game.get("winner_player_idx")
    gui_ctx["_turn_placements_snap"] = copy.deepcopy(game.get("placements") or [])
    gui_ctx["killer_bull_pending"] = None
    gui_ctx["killer_bull_steals"] = {}
    if _killer_is_play_phase(game):
        _killer_resync_mid_turn(
            gui_ctx, game, finished_idx, list(game.get("turn_hits") or [None, None, None])
        )


def _preview_players_after_turn(
    game: Dict, cur_idx: int, turn_hits: List[Optional[DetectedHit]]
) -> List[Dict]:
    """Kopie igraca nakon simuliranog bodovanja poteza (samo za UI)."""
    # Killer play: live players must already reflect turn_hits via
    # _killer_resync_mid_turn (incl. after undo-continue restores pending hits).
    if _killer_is_play_phase(game):
        return copy.deepcopy(game.get("players") or [])
    stub: Dict = {
        "mode": game.get("mode"),
        "rules": copy.deepcopy(game.get("rules") or {}),
        "players": copy.deepcopy(game.get("players") or []),
        "placements": list(game.get("placements") or []),
        "eliminated_order": list(game.get("eliminated_order") or []),
        "killer_phase": game.get("killer_phase"),
        "winner_player_idx": game.get("winner_player_idx"),
        "round_number": int(game.get("round_number", 1)),
        "num_players": int(game.get("num_players", 1)),
        "around_targets": list(game.get("around_targets") or [])
        if game.get("around_targets") is not None
        else None,
        "halve_targets": list(game.get("halve_targets") or [])
        if game.get("halve_targets") is not None
        else None,
    }
    _apply_turn_score(stub, cur_idx, turn_hits)
    return list(stub.get("players") or [])


def _advance_round_if_needed(game: Dict, finished_idx: int, next_idx: int) -> None:
    """Pomakni broj runde kad se krug igraca zatvori (isto kao kod IDUCI IGRAC)."""
    if game.get("winner_player_idx") is not None:
        return
    active = _active_player_indices(game)
    round_advanced = False
    if _play_for_placements(game):
        if active and len(active) > 1 and finished_idx in active and next_idx in active:
            if active.index(next_idx) <= active.index(finished_idx):
                round_advanced = True
        elif active and len(active) == 1:
            # Solo survivor / solo game: each turn closes the round.
            round_advanced = True
    elif int(game.get("num_players", 1)) <= 1:
        # Solo Halve It / etc.: next wraps to same player — still advance round.
        round_advanced = True
    elif int(next_idx) == 0 and int(finished_idx) != int(next_idx):
        round_advanced = True
    if not round_advanced:
        return
    game["round_number"] = int(game.get("round_number", 1)) + 1
    if str(game.get("mode", "")).lower() == "cricket":
        _cricket_round_limit_finish(game)
        if game.get("winner_player_idx") is not None and game.get("winner_at") is None:
            game["winner_at"] = time.time()
    elif str(game.get("mode", "")).lower() in ("halve", "halve_it"):
        targets = _halve_targets(game)
        if int(game.get("round_number", 1)) > len(targets):
            _halve_set_winner(game)
            if game.get("winner_player_idx") is not None and game.get("winner_at") is None:
                game["winner_at"] = time.time()


def _pop_undo(game: Dict, gui_state: Dict, gui_ctx: Dict) -> bool:
    stack = game.get("undo_stack") or []
    if not stack:
        return False
    snap = stack.pop()
    if "finished_player_idx" not in snap:
        return False
    _enter_undo_rewind(game, gui_state, gui_ctx, snap)
    return True


def _apply_turn_score(game: Dict, cur_idx: int, final_hits: List[Optional[DetectedHit]]) -> None:
    mode = game.get("mode")
    if mode in ("301", "501"):
        _apply_x01_turn(game, cur_idx, final_hits)
        return

    if mode in ("around", "around_the_world"):
        targets = _around_targets(game)
        rules = game.get("rules") or {}
        pstate = game["players"][cur_idx]
        _around_apply_hits(pstate, targets, rules, final_hits)
        pstate["last_turn"] = final_hits
        if pstate.get("completed"):
            _record_player_finish(game, cur_idx)
        return

    if mode == "cricket":
        rules = game.get("rules") or {}
        cricket_mode = str(rules.get("cricket_mode", "standard")).lower()
        wicket_values = [15, 16, 17, 18, 19, 20, 25]
        pstate = game["players"][cur_idx]
        for h in final_hits:
            if h is None or h.hit_score.score <= 0:
                continue
            seg_num = int(h.hit_score.segment_number)
            zone = str(getattr(h.hit_score, "zone_name", "single")).lower()
            if seg_num in (25, 50):
                key = 25
                marks = 2 if zone == "inner_bull" or seg_num == 50 else 1
            else:
                key = seg_num
                if key not in wicket_values:
                    continue
                if zone == "triple":
                    marks = 3
                elif zone == "double":
                    marks = 2
                else:
                    marks = 1
            if key not in wicket_values:
                continue
            prev_count = int(pstate["wickets"][key])
            new_count = min(3, prev_count + marks)
            overflow = max(0, prev_count + marks - 3)
            pstate["wickets"][key] = new_count
            if overflow > 0 and cricket_mode != "no_score":
                if cricket_mode == "cut_throat":
                    for oi, op in enumerate(game["players"]):
                        if oi == cur_idx:
                            continue
                        if int(op["wickets"].get(key, 0)) < 3:
                            op["points"] = int(op.get("points", 0)) + int(overflow * key)
                else:
                    any_open = any(
                        int(op["wickets"].get(key, 0)) < 3
                        for oi, op in enumerate(game["players"])
                        if oi != cur_idx
                    )
                    if any_open:
                        pstate["points"] = int(pstate.get("points", 0)) + int(overflow * key)
        pstate["last_turn"] = final_hits
        _cricket_set_winner_if_any(game)
        return

    if mode == "killer":
        _apply_killer_turn(game, cur_idx, final_hits)
        return

    if mode in ("halve", "halve_it"):
        _apply_halve_turn(game, cur_idx, final_hits)
        return


def main():
    global shutdown_requested
    parser = argparse.ArgumentParser(description="Manual darts kiosk")
    parser.add_argument("--gui", action="store_true", help="Open GUI (recommended)")
    parser.add_argument("--kiosk", action="store_true", help="Fullscreen kiosk mode (q exits)")
    parser.add_argument(
        "--legacy-gui",
        action="store_true",
        help="Use old OpenCV-drawn GUI instead of Qt",
    )
    parser.add_argument(
        "--fakecams",
        action="store_true",
        help="Use fake_cams/cam_0,cam_2,cam_4 images instead of USB cameras",
    )
    args = parser.parse_args()

    fake_cams_dir = default_fake_cams_dir() if args.fakecams else None
    if fake_cams_dir is not None and not os.path.isdir(fake_cams_dir):
        print(f"[cam] --fakecams: folder ne postoji: {fake_cams_dir}", flush=True)
        return

    if args.gui and not args.legacy_gui:
        try:
            from qt_ui import run_qt_app
        except ImportError as e:
            print(
                "[gui] PySide6 nije instaliran — pip install PySide6 "
                f"(ili pokreni s --legacy-gui). Greska: {e}",
                flush=True,
            )
            return
        raise SystemExit(run_qt_app(kiosk=args.kiosk, fake_cams_dir=fake_cams_dir))

    signal.signal(signal.SIGINT, _on_sigint)

    UI = {
        "bg": (0, 0, 0),
        "text": (235, 235, 235),
        "muted": (170, 170, 170),
        "good": (60, 200, 60),
        "game_bg": (0, 0, 0),
        "btn_yellow": (0, 255, 255),
        "btn_red": (60, 60, 220),
        "hits_green": (0, 255, 100),
        "text_black": (0, 0, 0),
    }

    gui_state = {
        "screen": "standby",  # standby|select_game|rules_x01|rules_cricket|rules_killer|select_players|clear_board|playing|exit_confirm|calibration
        "selected_game_mode": None,
        "num_players": 0,
        "current_player_idx": 0,
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
    gui_ctx = {
        "buttons": [],
        "keypad_buttons": [],
        "game": None,
        "input_multiplier": 1,  # one-shot 2|3; default 1 (single)
        "_display_w": 1120,
        "_display_h": 640,
        "_display_scale": 1.0,
        "_display_off_x": 0,
        "_display_off_y": 0,
        "_layout_split": False,
        "_stack_control_h": 0,
        "_stack_view_w": 0,
        "_stack_view_h": 0,
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
        "_cal_canvas_disp": None,
        "_cal_tiles": [],
        "bull_hints": {},
        "ellipse_hints": {},
        "_cal_click_bull_mode": False,
        "_cal_click_ellipse_mode": False,
        "editing_hit_slot": None,
    }

    # Keep a flat history list for debugging/consistency with camera app behavior.
    all_hits: List[DetectedHit] = []

    qr_start_img_bgr = None
    qr_start_loaded_path = None

    def _panel_draw_button(
        img: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        label: str,
        active: bool = False,
        font_scale: float = 0.45,
        style: str = "default",
        text_thick: Optional[int] = None,
    ):
        if style == "primary":
            fill = UI["btn_yellow"] if active else (0, 200, 200)
            border = (0, 160, 160)
            txt_col = UI["text_black"]
        elif style == "danger":
            fill = UI["btn_red"] if active else (45, 45, 120)
            border = (90, 90, 160)
            txt_col = UI["text_black"]
        elif style == "miss":
            fill = UI["btn_red"]
            border = (90, 90, 160)
            txt_col = UI["text_black"]
        elif style == "confirm_gray":
            fill = (72, 72, 78)
            border = (110, 110, 118)
            txt_col = (235, 235, 235)
        elif style == "accent":
            fill = (0, 180, 120) if active else (0, 130, 90)
            border = (0, 120, 80)
            txt_col = UI["text_black"]
        elif style == "rule_off":
            fill = (72, 72, 78)
            border = (110, 110, 118)
            txt_col = (235, 235, 235)
        elif style == "rule_on":
            fill = UI["btn_yellow"]
            border = (0, 160, 160)
            txt_col = UI["text_black"]
        elif style == "disabled":
            fill = (50, 50, 50)
            border = (80, 80, 80)
            txt_col = (120, 120, 120)
        elif style == "skip":
            fill = (72, 72, 78)
            border = (110, 110, 118)
            txt_col = (235, 235, 235)
        elif style == "hit_slot":
            fill = (38, 38, 42)
            border = (72, 72, 80)
            txt_col = (255, 255, 255)
        else:
            fill = (70, 70, 70) if not active else (0, 140, 255)
            border = (200, 200, 200)
            txt_col = (255, 255, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), border, 1)
        btn_w = max(1, x2 - x1)
        btn_h = max(1, y2 - y1)
        thick = text_thick if text_thick is not None else (2 if font_scale < 0.7 else 3)
        fs = font_scale
        pad_x = max(8, int(btn_w * 0.12))
        for _ in range(16):
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, thick)
            if tw <= btn_w - pad_x or fs <= 0.32:
                break
            fs *= 0.88
        tx = x1 + (btn_w - tw) // 2
        ty = y1 + (btn_h + th) // 2 - bl // 2
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, fs, txt_col, thick)

    def _draw_clear_board_card(
        panel: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        *,
        title: str,
        subtitle: str,
        subtitle2: str = "",
        status: str = "",
        accent: Optional[Tuple[int, int, int]] = None,
        title_only: bool = False,
        subtitle_scale: float = 1.0,
        title_scale: float = 1.0,
    ) -> None:
        accent = accent or UI["btn_yellow"]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        bg = (16, 24, 38)
        border_t = 3
        cv2.rectangle(panel, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(panel, (x1, y1), (x2, y2), accent, border_t)
        pad_x = max(12, w // 24)
        cx = x1 + w // 2
        title_fs = max(0.78, min(1.55, w / 340.0))
        if title_only:
            title_fs = max(0.90, min(1.55, min(w, h) / 200.0)) * title_scale
        for _ in range(10):
            (tw, th), bl = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, title_fs, 3)
            if tw <= w - 2 * pad_x or title_fs <= 0.55:
                break
            title_fs *= 0.92
        if title_only:
            ty = y1 + (h + th) // 2 - bl // 2
            cv2.putText(
                panel,
                title,
                (cx - tw // 2, ty),
                cv2.FONT_HERSHEY_DUPLEX,
                title_fs,
                accent,
                3,
            )
            return
        y_cur = y1 + max(18, h // 8)
        cv2.putText(
            panel,
            title,
            (cx - tw // 2, y_cur + th - bl),
            cv2.FONT_HERSHEY_DUPLEX,
            title_fs,
            accent,
            3,
        )
        y_cur += th + max(10, h // 16)
        sub_fs = max(0.62, min(0.98, w / 400.0)) * subtitle_scale
        for line in (subtitle, subtitle2):
            if not line:
                continue
            (sw, sh), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, sub_fs, 1)
            cv2.putText(
                panel,
                line,
                (cx - sw // 2, y_cur + sh),
                cv2.FONT_HERSHEY_SIMPLEX,
                sub_fs,
                UI["muted"],
                1,
                cv2.LINE_AA,
            )
            y_cur += sh + max(6, h // 28)
        if status:
            st_fs = max(0.44, min(0.62, w / 620.0))
            (stw, sth), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, st_fs, 1)
            st_y = min(y2 - max(10, h // 16), y_cur + sth + 6)
            cv2.putText(
                panel,
                status,
                (cx - stw // 2, st_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                st_fs,
                (90, 190, 255),
                1,
                cv2.LINE_AA,
            )

    def _build_control_panel(width: int, height: int) -> np.ndarray:
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel.fill(0)
        buttons = []
        screen = gui_state.get("screen", "standby")
        game = gui_ctx.get("game") or {}

        def add_button(
            bid: str,
            x1: int,
            y1: int,
            x2: int,
            y2: int,
            label: str,
            *,
            active: bool = False,
            style: str = "default",
            text_thick: Optional[int] = None,
            font_scale: Optional[float] = None,
        ):
            btn_h = max(1, y2 - y1)
            if font_scale is None:
                font_scale = max(0.5, min(1.15, btn_h / 54.0))
            _panel_draw_button(
                panel, x1, y1, x2, y2, label, active=active, font_scale=font_scale, style=style, text_thick=text_thick
            )
            buttons.append({"id": bid, "rect": (x1, y1, x2, y2)})

        def add_nav_back() -> Tuple[int, int, int]:
            """← — mali kvadrat donji lijevi kut."""
            margin = max(10, int(min(width, height) * 0.02))
            sq = max(52, int(min(width, height) * 0.085))
            x1 = margin
            y1 = height - margin - sq
            add_button(
                "nav_back",
                x1,
                y1,
                x1 + sq,
                y1 + sq,
                "←",
                active=True,
                style="default",
                text_thick=3,
                font_scale=1.35,
            )
            return margin, margin, sq

        def add_nav_continue(bid: str = "rules_continue") -> None:
            """→ — mali kvadrat donji desni kut."""
            margin = max(10, int(min(width, height) * 0.02))
            sq = max(52, int(min(width, height) * 0.085))
            x2 = width - margin
            y1 = height - margin - sq
            add_button(
                bid,
                x2 - sq,
                y1,
                x2,
                y1 + sq,
                "→",
                active=True,
                style="primary",
                text_thick=3,
                font_scale=1.35,
            )

        if screen == "standby":
            pad = int(height * 0.05)
            title_scale = max(1.0, min(1.85, float(height) / 380.0))
            title_y = pad + int(height * 0.12)
            line1 = "SKENIRAJ QR KOD"
            line2 = "ZA POCETAK IGRE"
            (tw1, th1), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_DUPLEX, title_scale, 3)
            (tw2, th2), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_DUPLEX, title_scale * 0.95, 3)
            cv2.putText(
                panel,
                line1,
                ((width - tw1) // 2, title_y),
                cv2.FONT_HERSHEY_DUPLEX,
                title_scale,
                UI["text"],
                3,
            )
            cv2.putText(
                panel,
                line2,
                ((width - tw2) // 2, title_y + th1 + int(height * 0.05)),
                cv2.FONT_HERSHEY_DUPLEX,
                title_scale * 0.95,
                UI["text"],
                3,
            )
            nonlocal qr_start_img_bgr, qr_start_loaded_path
            from kiosk_settings import start_qr_image_path as _qr_path
            want_path = _qr_path()
            if qr_start_img_bgr is None or qr_start_loaded_path != want_path:
                qr_start_img_bgr = cv2.imread(want_path, cv2.IMREAD_COLOR)
                qr_start_loaded_path = want_path
            if qr_start_img_bgr is not None:
                qr_size = max(80, min(int(width * 0.52), int(height * 0.52)))
                qr_resized = cv2.resize(qr_start_img_bgr, (qr_size, qr_size), interpolation=cv2.INTER_NEAREST)
                qr_top = title_y + th1 + th2 + int(height * 0.07)
                qr_left = max(0, (width - qr_size) // 2)
                qr_top = max(0, min(qr_top, height - qr_size - pad))
                qr_pad = 10
                cv2.rectangle(
                    panel,
                    (qr_left - qr_pad, qr_top - qr_pad),
                    (qr_left + qr_size + qr_pad, qr_top + qr_size + qr_pad),
                    (255, 255, 255),
                    -1,
                )
                panel[qr_top : qr_top + qr_size, qr_left : qr_left + qr_size] = qr_resized

        elif screen == "exit_confirm":
            q = "Izlaz iz igre?"
            q_scale = max(0.82, min(1.15, width / 520.0))
            (tw, th), _ = cv2.getTextSize(q, cv2.FONT_HERSHEY_SIMPLEX, q_scale, 2)
            qx = (width - tw) // 2
            btn_h = max(52, int(height * 0.09))
            btn_w = max(160, int(width * 0.22))
            gap = max(24, int(width * 0.04))
            total_w = btn_w * 2 + gap
            x0 = (width - total_w) // 2
            text_gap = max(22, int(height * 0.035))
            block_h = th + text_gap + btn_h
            y_text_top = (height - block_h) // 2
            cv2.putText(panel, q, (qx, y_text_top + th - 2), cv2.FONT_HERSHEY_SIMPLEX, q_scale, UI["text"], 2)
            y0 = y_text_top + th + text_gap
            add_button("confirm_exit_yes", x0, y0, x0 + btn_w, y0 + btn_h, "DA", active=True, style="confirm_gray")
            add_button(
                "confirm_exit_no",
                x0 + btn_w + gap,
                y0,
                x0 + btn_w + gap + btn_w,
                y0 + btn_h,
                "NE",
                active=True,
                style="confirm_gray",
            )

        elif screen == "select_game":
            title = "IZABERI IGRU"
            (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
            cv2.putText(panel, title, ((width - tw) // 2, max(35, int(height * 0.08)) + th), cv2.FONT_HERSHEY_SIMPLEX, 1.2, UI["text"], 2)
            margin_x = max(10, int(width * 0.05))
            margin_bottom = max(10, int(height * 0.06))
            gap_x = max(8, int(width * 0.02))
            gap_y = max(8, int(height * 0.02))
            top = int(height * 0.16)
            cell_w = int((width - 2 * margin_x - gap_x) / 2)
            n_rows = 3
            cell_h = int((height - top - margin_bottom - gap_y * (n_rows - 1)) / n_rows)
            modes = [
                ("game:301", "301"),
                ("game:501", "501"),
                ("game:around", "AROUND"),
                ("game:cricket", "CRICKET"),
                ("game:killer", "KILLER"),
                ("game:halve", "HALVE IT"),
            ]
            for idx, (bid, label) in enumerate(modes):
                r = idx // 2
                c = idx % 2
                x1 = margin_x + c * (cell_w + gap_x)
                y1 = top + r * (cell_h + gap_y)
                add_button(bid, x1, y1, x1 + cell_w, y1 + cell_h, label, active=True, style="primary")

        elif screen == "rules_x01":
            mode = str(gui_state.get("selected_game_mode", "301")).upper()
            rules = gui_state.get("rules_pending") or {}
            title = f"{mode} - PRAVILA"
            title_scale = max(1.0, min(1.45, width / 420.0))
            (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, 2)
            content_w = max(320, int(width * 0.90))
            x_left = (width - content_w) // 2
            btn_gap = max(14, int(width * 0.035))
            btn_h = max(68, int(height * 0.13))
            cont_h = max(68, int(height * 0.13))
            btn_w = (content_w - btn_gap) // 2
            title_gap = max(32, int(height * 0.07))
            rules_cont_gap = max(28, int(height * 0.055))
            block_h = th + title_gap + btn_h + rules_cont_gap + cont_h
            block_top = max(20, (height - block_h) // 2)
            title_y = block_top + th
            cv2.putText(panel, title, ((width - tw) // 2, title_y), cv2.FONT_HERSHEY_SIMPLEX, title_scale, UI["text"], 2)
            y_rules = title_y + title_gap
            di_on = bool(rules.get("double_in"))
            do_on = bool(rules.get("double_out"))
            add_button(
                "rule:double_in",
                x_left,
                y_rules,
                x_left + btn_w,
                y_rules + btn_h,
                f"DOUBLE IN: {'DA' if di_on else 'NE'}",
                active=True,
                style="rule_on" if di_on else "rule_off",
                text_thick=3,
            )
            add_button(
                "rule:double_out",
                x_left + btn_w + btn_gap,
                y_rules,
                x_left + content_w,
                y_rules + btn_h,
                f"DOUBLE OUT: {'DA' if do_on else 'NE'}",
                active=True,
                style="rule_on" if do_on else "rule_off",
                text_thick=3,
            )
            y_cont = y_rules + btn_h + rules_cont_gap
            add_nav_continue("rules_continue")
            add_nav_back()

        elif screen == "rules_cricket":
            rules = gui_state.get("rules_pending") or {}
            title = "CRICKET - PRAVILA"
            title_scale = max(1.0, min(1.45, width / 420.0))
            title_th = 3
            (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, title_scale, title_th)
            content_w = max(320, int(width * 0.92))
            x_left = (width - content_w) // 2
            btn_gap = max(12, int(width * 0.03))
            btn_h = max(62, int(height * 0.115))
            cont_h = max(68, int(height * 0.13))
            lbl_fs = max(0.55, min(0.78, width / 560.0))
            lbl_th = 2
            (_, lbl_h), _ = cv2.getTextSize("MAX RUNDE", cv2.FONT_HERSHEY_DUPLEX, lbl_fs, lbl_th)
            section_gap = max(18, int(height * 0.04))
            lbl_btn_gap = max(12, int(height * 0.028))
            title_gap = max(28, int(height * 0.055))
            cont_gap = max(22, int(height * 0.045))
            block_h = th + title_gap + lbl_h + lbl_btn_gap + btn_h + section_gap + lbl_h + lbl_btn_gap + btn_h + cont_gap + cont_h
            block_top = max(14, (height - block_h) // 2)
            title_y = block_top + th
            cv2.putText(
                panel, title, ((width - tw) // 2, title_y), cv2.FONT_HERSHEY_DUPLEX, title_scale, UI["text"], title_th
            )
            y0 = title_y + title_gap
            cv2.putText(
                panel, "MAX RUNDE", (x_left, y0 + lbl_h), cv2.FONT_HERSHEY_DUPLEX, lbl_fs, UI["muted"], lbl_th
            )
            y0 += lbl_h + lbl_btn_gap
            max20 = bool(rules.get("max_rounds_20"))
            half_w = (content_w - btn_gap) // 2
            add_button(
                "rule:max_rounds_off",
                x_left,
                y0,
                x_left + half_w,
                y0 + btn_h,
                "BEZ LIMITA",
                active=True,
                style="rule_on" if not max20 else "rule_off",
                text_thick=3,
            )
            add_button(
                "rule:max_rounds_20",
                x_left + half_w + btn_gap,
                y0,
                x_left + content_w,
                y0 + btn_h,
                "20 RUNDI",
                active=True,
                style="rule_on" if max20 else "rule_off",
                text_thick=3,
            )
            y0 += btn_h + section_gap
            cv2.putText(
                panel, "GAME MODE", (x_left, y0 + lbl_h), cv2.FONT_HERSHEY_DUPLEX, lbl_fs, UI["muted"], lbl_th
            )
            y0 += lbl_h + lbl_btn_gap
            cmode = str(rules.get("cricket_mode", "standard")).lower()
            third_w = (content_w - 2 * btn_gap) // 3
            modes_cr = [
                ("rule:cricket_standard", "STANDARD", "standard"),
                ("rule:cricket_no_score", "NO SCORE", "no_score"),
                ("rule:cricket_cut_throat", "CUT THROAT", "cut_throat"),
            ]
            for mi, (bid, lab, key) in enumerate(modes_cr):
                x1 = x_left + mi * (third_w + btn_gap)
                add_button(
                    bid,
                    x1,
                    y0,
                    x1 + third_w,
                    y0 + btn_h,
                    lab,
                    active=True,
                    style="rule_on" if cmode == key else "rule_off",
                    text_thick=3,
                )
            y_cont = y0 + btn_h + cont_gap
            add_nav_continue("rules_continue")
            add_nav_back()

        elif screen == "rules_killer":
            rules = gui_state.get("rules_pending") or {}
            title = "KILLER - PRAVILA"
            title_scale = max(0.85, min(1.25, width / 480.0))
            (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, title_scale, 2)
            content_w = max(320, int(width * 0.92))
            x_left = (width - content_w) // 2
            btn_gap = max(8, int(width * 0.02))
            btn_h = max(48, int(height * 0.085))
            cont_h = max(56, int(height * 0.10))
            lbl_fs = max(0.48, min(0.68, width / 600.0))
            lbl_th = 2
            (_, lbl_h), _ = cv2.getTextSize("LIVES", cv2.FONT_HERSHEY_DUPLEX, lbl_fs, lbl_th)
            section_gap = max(10, int(height * 0.02))
            lbl_btn_gap = max(6, int(height * 0.015))
            title_gap = max(14, int(height * 0.03))
            cont_gap = max(12, int(height * 0.025))
            block_h = (
                th
                + title_gap
                + 3 * (lbl_h + lbl_btn_gap + btn_h + section_gap)
                + cont_gap
                + cont_h
            )
            block_top = max(8, (height - block_h) // 2)
            title_y = block_top + th
            cv2.putText(
                panel, title, ((width - tw) // 2, title_y), cv2.FONT_HERSHEY_DUPLEX, title_scale, UI["text"], 2
            )
            y0 = title_y + title_gap

            def _row_label(txt: str) -> None:
                nonlocal y0
                cv2.putText(
                    panel, txt, (x_left, y0 + lbl_h), cv2.FONT_HERSHEY_DUPLEX, lbl_fs, UI["muted"], lbl_th
                )
                y0 += lbl_h + lbl_btn_gap

            lives = int(rules.get("killer_lives", 3) or 3)
            _row_label("BROJ ZIVOTA")
            life_opts = [3, 5, 7, 10]
            life_w = (content_w - 3 * btn_gap) // 4
            for i, lv in enumerate(life_opts):
                x1 = x_left + i * (life_w + btn_gap)
                add_button(
                    f"rule:killer_lives:{lv}",
                    x1,
                    y0,
                    x1 + life_w,
                    y0 + btn_h,
                    str(lv),
                    active=True,
                    style="rule_on" if lives == lv else "rule_off",
                    text_thick=2,
                )
            y0 += btn_h + section_gap

            assign = str(rules.get("killer_assign", "random")).lower()
            _row_label("DODJELA BROJEVA")
            half_w = (content_w - btn_gap) // 2
            add_button(
                "rule:killer_assign:random",
                x_left,
                y0,
                x_left + half_w,
                y0 + btn_h,
                "RANDOM",
                active=True,
                style="rule_on" if assign == "random" else "rule_off",
                text_thick=2,
            )
            add_button(
                "rule:killer_assign:throw",
                x_left + half_w + btn_gap,
                y0,
                x_left + content_w,
                y0 + btn_h,
                "GADJANJEM",
                active=True,
                style="rule_on" if assign == "throw" else "rule_off",
                text_thick=2,
            )
            y0 += btn_h + section_gap

            act = str(rules.get("killer_activation", "double")).lower()
            _row_label("ACTIVATION")
            third_w = (content_w - 2 * btn_gap) // 3
            for i, (key, lab) in enumerate(
                (("single", "SINGLE"), ("double", "DOUBLE"), ("triple", "TRIPLE"))
            ):
                x1 = x_left + i * (third_w + btn_gap)
                add_button(
                    f"rule:killer_activation:{key}",
                    x1,
                    y0,
                    x1 + third_w,
                    y0 + btn_h,
                    lab,
                    active=True,
                    style="rule_on" if act == key else "rule_off",
                    text_thick=2,
                )
            y0 += btn_h + cont_gap
            add_nav_continue("rules_continue")
            add_nav_back()

        elif screen == "select_players":
            title = "KOLIKO IGRACA?"
            (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
            cv2.putText(panel, title, ((width - tw) // 2, max(35, int(height * 0.08)) + th), cv2.FONT_HERSHEY_SIMPLEX, 1.2, UI["text"], 2)
            margin_x = max(10, int(width * 0.05))
            nav_sq = max(52, int(min(width, height) * 0.085))
            margin_bottom = max(10, int(height * 0.04)) + nav_sq + max(8, int(height * 0.02))
            gap_x = max(8, int(width * 0.02))
            gap_y = max(12, int(height * 0.03))
            top = int(height * 0.16)
            cell_w = int((width - 2 * margin_x - gap_x) / 2)
            available = height - top - margin_bottom - gap_y
            cell_h = max(20, int(available / 2))
            mode = str(gui_state.get("selected_game_mode") or "").lower()
            for n in range(1, 5):
                bid = f"players:{n}"
                r = (n - 1) // 2
                c = (n - 1) % 2
                x1 = margin_x + c * (cell_w + gap_x)
                y1 = top + r * (cell_h + gap_y)
                if mode == "killer" and n == 1:
                    # Disabled visually; click ignored in handler.
                    add_button(
                        "noop_killer_1p",
                        x1,
                        y1,
                        x1 + cell_w,
                        y1 + cell_h,
                        "NEMOGUCE S 1",
                        active=False,
                        style="default",
                    )
                else:
                    add_button(bid, x1, y1, x1 + cell_w, y1 + cell_h, str(n), active=True, style="primary")
            add_nav_back()

        elif screen == "clear_board":
            pad = max(18, int(min(width, height) * 0.05))
            margin_x = max(10, int(width * 0.05))
            margin_bottom = max(10, int(height * 0.04))
            back_h = max(44, int(height * 0.075))
            back_gap = max(28, int(height * 0.05))
            card_bottom_limit = height - margin_bottom - back_h - back_gap
            card_w = max(300, int(width * 0.90))
            card_h = max(300, min(int(height * 0.72), card_bottom_limit - pad))
            card_x1 = (width - card_w) // 2
            card_y1 = max(pad, (height - card_h) // 2 - int(height * 0.03))
            card_x2 = card_x1 + card_w
            card_y2 = min(card_bottom_limit, card_y1 + card_h)
            status = str(gui_ctx.get("_clear_board_status") or "").strip()
            _draw_clear_board_card(
                panel,
                card_x1,
                card_y1,
                card_x2,
                card_y2,
                title="OČISTITE PLOČU",
                subtitle="Uklonite sve strelice s ploče",
                subtitle2="prije početka igre",
                status=status,
                subtitle_scale=1.12,
            )
            btn_h = max(88, int((card_y2 - card_y1) * 0.20))
            btn_w = max(260, int(card_w * 0.76))
            bx1 = (width - btn_w) // 2
            by2 = card_y2 - max(18, int(card_h * 0.08))
            by1 = by2 - btn_h
            add_button(
                "clear_board_confirm",
                bx1,
                by1,
                bx1 + btn_w,
                by2,
                "PLOČA JE PRAZNA",
                active=True,
                style="primary",
                text_thick=3,
            )
            add_nav_back()

        elif screen == "playing":
            pad = max(14, int(width * 0.05))
            score_scale = max(0.98, min(1.48, height / 380.0))
            line_h = int(44 + score_scale * 9)
            y_top = pad + 36
            num_players = int(game.get("num_players", gui_state.get("num_players", 1)) or 1)
            players = game.get("players", [])
            cur = int(gui_state.get("current_player_idx", 0))
            winner = game.get("winner_player_idx")
            turn_hits_ui = game.get("turn_hits", [None, None, None])
            preview_players = None
            if any(h is not None for h in turn_hits_ui):
                try:
                    preview_players = _preview_players_after_turn(game, cur, turn_hits_ui)
                except Exception:
                    preview_players = None

            def _player_line(pi: int) -> str:
                if preview_players is not None and 0 <= pi < len(preview_players):
                    pstate = preview_players[pi]
                else:
                    pstate = players[pi] if pi < len(players) else {}
                return _player_score_text(game, pi, pstate)

            # Winner state: only after finishing turn submitted (board cleared).
            if winner is not None and not _winner_pending_board_clear(game):
                entries = _leaderboard_entries(game)
                t0 = float(game.get("winner_at") or time.time())
                t_anim = max(0.0, time.time() - t0)
                pulse = 0.5 + 0.5 * math.sin(t_anim * 5.5)
                head_fs = max(1.45, min(2.05, width / 220.0)) + 0.22 * pulse
                sub_fs = max(1.0, min(1.42, width / 320.0))
                btn_exit_h = max(58, int(height * 0.12))
                leaderboard_btn_gap = max(56, int(height * 0.12))
                y_bot = height - pad
                y_exit_top = y_bot - btn_exit_h
                leaderboard_bottom_limit = y_exit_top - leaderboard_btn_gap
                line_gap_head = max(32, int(height * 0.048))
                line_gap_sub = max(24, int(height * 0.038))
                total_lb_h = 0
                for idx, (rank, pi) in enumerate(entries):
                    line_txt = f"#{rank} P{pi + 1}"
                    fs = head_fs if idx == 0 else sub_fs
                    th = 3 if idx == 0 else 2
                    (_, th_px), bl = cv2.getTextSize(line_txt, cv2.FONT_HERSHEY_DUPLEX, fs, th)
                    total_lb_h += th_px + bl
                    if idx < len(entries) - 1:
                        total_lb_h += line_gap_head if idx == 0 else line_gap_sub
                start_y = max(pad + 24, (leaderboard_bottom_limit - total_lb_h) // 2)
                y_next = start_y
                for idx, (rank, pi) in enumerate(entries):
                    line_txt = f"#{rank} P{pi + 1}"
                    fs = head_fs if idx == 0 else sub_fs
                    th = 3 if idx == 0 else 2
                    col = _player_label_bgr(pi, (players[pi] if pi < len(players) else None))
                    if idx == 0 and int(t_anim * 2.5) % 2 == 0:
                        col = tuple(min(255, int(c * 1.15)) for c in col)
                    (_, th_px), bl = cv2.getTextSize(line_txt, cv2.FONT_HERSHEY_DUPLEX, fs, th)
                    (tw, _), _ = cv2.getTextSize(line_txt, cv2.FONT_HERSHEY_DUPLEX, fs, th)
                    ty = y_next + th_px - bl
                    cv2.putText(
                        panel,
                        line_txt,
                        ((width - tw) // 2, ty),
                        cv2.FONT_HERSHEY_DUPLEX,
                        fs,
                        col,
                        th,
                    )
                    y_next = ty + bl + (line_gap_head if idx == 0 else line_gap_sub)
                add_button(
                    "request_exit_game",
                    pad,
                    y_exit_top,
                    width - pad,
                    y_bot,
                    "IZLAZ IZ IGRE",
                    active=True,
                    style="primary",
                    text_thick=3,
                )
            else:
                turn_hits = game.get("turn_hits", [None, None, None])
                can_submit = _turn_can_submit(game, cur, turn_hits)
                needs_assign_retry = _killer_assign_needs_retry(game, cur, turn_hits)
                waiting_clear = can_submit or needs_assign_retry
                is_checkout = _turn_is_early_finish(game, cur, turn_hits)
                block_adv = bool(gui_ctx.get("_dart_block_auto_advance"))
                hit_fs = max(1.08, min(1.55, width / 300.0))
                hit_step = int(max(42, line_h * 0.95))
                btn_exit_h = max(58, int(height * 0.12))
                undo_w = max(btn_exit_h, int(btn_exit_h * 1.08))
                gap_btn = max(8, int(width * 0.015))
                gap_row = max(10, int(height * 0.022))
                btn_lift = max(12, int(height * 0.02))
                y_bot = height - pad
                y_exit_top = y_bot - btn_exit_h
                action_x2 = width - pad
                action_x1 = action_x2 - undo_w
                undo_x2 = action_x1 - gap_btn
                # Fiksno rezerviraj donji red (OCISTI + UNDO/X) da scoreboard ne skace.
                y_clear_bot_layout = y_exit_top - gap_row
                y_clear_top_layout = y_clear_bot_layout - btn_exit_h
                y_clear_bot = y_clear_bot_layout - btn_lift
                y_clear_top = y_clear_bot - btn_exit_h
                y_next_top = y_clear_top_layout - 22

                is_cricket = str(game.get("mode", "")).lower() == "cricket"
                is_around = str(game.get("mode", "")).lower() in ("around", "around_the_world")
                rules = game.get("rules") or {}
                cricket_round_band = (
                    34 if is_cricket and rules.get("max_rounds_20") else 0
                )

                def _around_target_label(pi: int) -> str:
                    pstate = players[pi] if pi < len(players) else {}
                    if pstate.get("completed"):
                        return "DONE"
                    nxt = int(pstate.get("next_target_idx", 0))
                    targets = _around_targets(game)
                    if nxt < 0 or nxt >= len(targets):
                        return "DONE"
                    want = targets[nxt]
                    return "BULL" if str(want).lower() == "bull" else str(want)

                def _draw_cricket_mark_bars(
                    x0: int, y0: int, w: int, h_total: int, marks: int, pi: int,
                    pstate: Optional[Dict] = None,
                ) -> None:
                    """Three horizontal bars; fills bottom→top (1st hit = bottom bar)."""
                    bar_gray = (74, 74, 74)
                    bar_fill = _player_color_from_state(pstate, pi)
                    gap = max(2, int(h_total * 0.10))
                    bar_h = max(2, (h_total - 2 * gap) // 3)
                    m = min(3, max(0, int(marks)))
                    y = y0
                    for i in range(3):
                        col = bar_fill if i >= 3 - m else bar_gray
                        cv2.rectangle(panel, (x0, y), (x0 + w - 1, y + bar_h - 1), col, -1)
                        y += bar_h + gap

                if is_cricket:
                    # Header row (15–20, B) + one row per player: Pn | 7 stacks | points. D1–D3 centered below.
                    cricket_rules = game.get("rules") or {}
                    show_cricket_pts = str(cricket_rules.get("cricket_mode", "standard")).lower() != "no_score"
                    tgt_keys = [15, 16, 17, 18, 19, 20, 25]
                    tgt_hdr = ["15", "16", "17", "18", "19", "20", "B"]
                    n_tgt = len(tgt_keys)
                    label_col_w = min(62, max(42, int(width * 0.11)))
                    row_fs = max(0.72, min(1.05, width / 380.0))
                    row_th = 3
                    hdr_fs = max(0.58, min(0.82, width / 420.0))
                    hdr_th = 2
                    score_txt_w = (
                        int(max(cv2.getTextSize("#1", cv2.FONT_HERSHEY_DUPLEX, row_fs, row_th)[0][0], 44))
                        if show_cricket_pts
                        else 0
                    )
                    icons_area_w = max(1, width - 2 * pad - label_col_w - score_txt_w - (12 if show_cricket_pts else 0))
                    icon_gap = 4
                    icon_w = max(8, (icons_area_w - (n_tgt - 1) * icon_gap) // n_tgt)
                    bar_w = min(icon_w, max(3, int(icon_w * 0.48)) + 4)
                    (_, hdr_h), _ = cv2.getTextSize("20", cv2.FONT_HERSHEY_DUPLEX, hdr_fs, hdr_th)
                    header_row_h = int(hdr_h + 10)
                    row_gap = 10
                    inner_top = pad + 8 + cricket_round_band
                    if cricket_round_band > 0:
                        rnd = int(game.get("round_number", 1))
                        limit = int(rules.get("max_rounds_limit", 20))
                        rnd_txt = f"Runda: {rnd}/{limit}"
                        rnd_fs = max(0.58, min(0.72, width / 560.0))
                        (rtw, rth), _ = cv2.getTextSize(rnd_txt, cv2.FONT_HERSHEY_DUPLEX, rnd_fs, 2)
                        cv2.putText(
                            panel,
                            rnd_txt,
                            (width - pad - rtw, pad + rth + 4),
                            cv2.FONT_HERSHEY_DUPLEX,
                            rnd_fs,
                            UI["btn_yellow"],
                            2,
                        )
                    y_row0 = inner_top + header_row_h
                    row_h = max(52, min(66, int(height * 0.095)))
                    stride = row_h + row_gap
                    x_ic0 = pad + label_col_w
                    for ci, hlab in enumerate(tgt_hdr):
                        cx = x_ic0 + ci * (icon_w + icon_gap) + icon_w // 2
                        (tw, _), _ = cv2.getTextSize(hlab, cv2.FONT_HERSHEY_DUPLEX, hdr_fs, hdr_th)
                        hdr_baseline = inner_top + header_row_h - 6
                        cv2.putText(
                            panel,
                            hlab,
                            (int(cx - tw // 2), hdr_baseline),
                            cv2.FONT_HERSHEY_DUPLEX,
                            hdr_fs,
                            UI["text"],
                            hdr_th,
                        )
                    for pi in range(num_players):
                        pstate = players[pi] if pi < len(players) else {}
                        wmap = pstate.get("wickets", {}) or {}
                        pts = int(pstate.get("points", 0))
                        y_row = y_row0 + pi * stride
                        plab = f"P{pi + 1}"
                        pcol = _player_label_bgr(pi, pstate)
                        (_, th), bl = cv2.getTextSize(plab, cv2.FONT_HERSHEY_DUPLEX, row_fs, row_th)
                        ty = y_row + (row_h + th) // 2 - bl
                        if pi == cur and _player_place_rank(game, pi) is None:
                            cv2.rectangle(
                                panel,
                                (pad - 6, y_row + 2),
                                (pad + label_col_w - 4, y_row + row_h - 2),
                                pcol,
                                2,
                            )
                        cv2.putText(panel, plab, (pad, ty), cv2.FONT_HERSHEY_DUPLEX, row_fs, pcol, row_th)
                        icon_h = max(18, row_h - 8)
                        x_ic = x_ic0
                        for key in tgt_keys:
                            m = min(3, int(wmap.get(key, 0)))
                            iy = y_row + (row_h - icon_h) // 2
                            x_bar = x_ic + (icon_w - bar_w) // 2
                            _draw_cricket_mark_bars(x_bar, iy, bar_w, icon_h, m, pi, pstate)
                            x_ic += icon_w + icon_gap
                        if show_cricket_pts:
                            rank = _player_place_rank(game, pi)
                            pts_s = f"#{rank}" if rank is not None else str(pts)
                            (tw, _), _ = cv2.getTextSize(pts_s, cv2.FONT_HERSHEY_DUPLEX, row_fs, row_th)
                            cv2.putText(
                                panel, pts_s, (width - pad - tw, ty), cv2.FONT_HERSHEY_DUPLEX, row_fs, pcol, row_th
                            )
                    if num_players > 0:
                        scoreboard_bottom = y_row0 + (num_players - 1) * stride + row_h
                    else:
                        scoreboard_bottom = y_row0
                elif is_around:
                    # One row per player; larger labels, tight highlight around text only.
                    row_gap = 12
                    inner_top = y_top + 4
                    y_row0 = inner_top
                    row_fs = max(0.88, min(1.22, width / 280.0))
                    row_th = 3
                    row_h = max(44, min(58, int(height * 0.082)))
                    stride = row_h + row_gap
                    box_pad_x = 12
                    box_pad_y = 10
                    for pi in range(num_players):
                        y_row = y_row0 + pi * stride
                        line_txt = f"P{pi + 1}: {_around_target_label(pi)}"
                        rank = _player_place_rank(game, pi)
                        if rank is not None:
                            line_txt = f"P{pi + 1}: #{rank}"
                        pcol = _player_label_bgr(
                            pi, players[pi] if pi < len(players) else None
                        )
                        tfs = row_fs
                        for _ in range(14):
                            (tw, _), _ = cv2.getTextSize(line_txt, cv2.FONT_HERSHEY_DUPLEX, tfs, row_th)
                            if tw <= width - 2 * pad or tfs < 0.40:
                                break
                            tfs *= 0.94
                        (_, th), bl = cv2.getTextSize(line_txt, cv2.FONT_HERSHEY_DUPLEX, tfs, row_th)
                        tx = pad
                        ty = y_row + (row_h + th) // 2 - bl
                        if pi == cur and rank is None:
                            bx1 = tx - box_pad_x
                            by1 = ty - th - box_pad_y
                            bx2 = tx + tw + box_pad_x
                            by2 = ty + bl + box_pad_y
                            cv2.rectangle(panel, (bx1, by1), (bx2, by2), pcol, 2)
                        cv2.putText(panel, line_txt, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, tfs, pcol, row_th)
                    if num_players > 0:
                        scoreboard_bottom = y_row0 + (num_players - 1) * stride + row_h
                    else:
                        scoreboard_bottom = y_row0
                else:
                    score_bottom = y_top
                    x01_row_gap = max(10, int(line_h * 0.3))
                    x01_row_h = max(38, int(line_h * 0.95))
                    x01_stride = x01_row_h + x01_row_gap
                    box_pad_y = 10
                    box_pad_x = 10
                    for pi in range(num_players):
                        y_row = y_top + pi * x01_stride
                        label = f"P{pi+1}: {_player_line(pi)}"
                        pcol = _player_label_bgr(
                            pi, players[pi] if pi < len(players) else None
                        )
                        thick = 3 if pi == cur else 2
                        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, score_scale, thick)
                        tx = pad
                        ty = y_row + (x01_row_h + th) // 2 - bl
                        by2 = ty + bl + box_pad_y
                        if pi == cur and _player_place_rank(game, pi) is None:
                            bx1 = tx - box_pad_x
                            by1 = ty - th - box_pad_y
                            bx2 = tx + tw + box_pad_x
                            cv2.rectangle(panel, (bx1, by1), (bx2, by2), pcol, 2)
                        score_bottom = max(score_bottom, by2)
                        cv2.putText(panel, label, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, score_scale, pcol, thick)
                    live_score = _x01_live_score_line(game, cur) or _killer_live_score_line(game, cur)
                    is_bust = _x01_turn_is_bust(game, cur, turn_hits)
                    sug_fs = max(0.78, score_scale * 0.82)
                    hits_x = pad + 6
                    sug_gap_below_scores = max(34, int(line_h * 0.52))
                    if is_bust:
                        bust_fs = max(0.95, score_scale * 0.98)
                        (_, bust_th), _ = cv2.getTextSize("BUST", cv2.FONT_HERSHEY_DUPLEX, bust_fs, 3)
                        y_bust = score_bottom + sug_gap_below_scores + bust_th
                        cv2.putText(
                            panel,
                            "BUST",
                            (hits_x, y_bust),
                            cv2.FONT_HERSHEY_DUPLEX,
                            bust_fs,
                            UI["btn_red"],
                            3,
                        )
                    elif live_score:
                        sug_txt = live_score
                        (_, sug_th), _ = cv2.getTextSize(sug_txt, cv2.FONT_HERSHEY_DUPLEX, sug_fs, 2)
                        y_sug = score_bottom + sug_gap_below_scores + sug_th
                        cv2.putText(
                            panel,
                            sug_txt,
                            (hits_x, y_sug),
                            cv2.FONT_HERSHEY_DUPLEX,
                            sug_fs,
                            UI["hits_green"],
                            3,
                        )

                if waiting_clear or block_adv:
                    from kiosk_settings import get_settings as _gs_draw

                    now_submit = time.time()
                    submit_since = gui_ctx.get("_dart_can_submit_since")
                    if submit_since is None and waiting_clear:
                        gui_ctx["_dart_can_submit_since"] = now_submit
                        submit_since = now_submit
                    manual_on = bool(_gs_draw().manual_mode)
                    show_manual_advance = block_adv or (
                        waiting_clear
                        and submit_since is not None
                        and (
                            manual_on
                            or now_submit - float(submit_since) >= FORCE_NEXT_BUTTON_DELAY_SEC
                        )
                    )
                    if show_manual_advance:
                        manual_btn_fs = max(0.82, min(1.38, btn_exit_h / 40.0))
                        if block_adv or manual_on:
                            add_button(
                                "manual_advance_confirm",
                                pad,
                                y_clear_top,
                                action_x2,
                                y_clear_bot,
                                _gs_draw().t("switch_player"),
                                active=True,
                                style="skip",
                                text_thick=3,
                                font_scale=manual_btn_fs,
                            )
                        elif gui_ctx.get("_dart_manual_advance_confirm"):
                            add_button(
                                "manual_advance_confirm",
                                pad,
                                y_clear_top,
                                action_x2,
                                y_clear_bot,
                                "PLOČA JE ČISTA?",
                                active=True,
                                style="accent",
                                text_thick=3,
                                font_scale=manual_btn_fs,
                            )
                        else:
                            add_button(
                                "manual_advance_confirm",
                                pad,
                                y_clear_top,
                                action_x2,
                                y_clear_bot,
                                "NIJE PREBACILO?",
                                active=True,
                                style="skip",
                                text_thick=3,
                                font_scale=manual_btn_fs,
                            )
                    else:
                        gui_ctx["_dart_manual_advance_confirm"] = False
                        _draw_clear_board_card(
                            panel,
                            pad,
                            y_clear_top,
                            action_x2,
                            y_clear_bot,
                            title=(
                                "OČISTITE PLOČU ZA KRAJ" if is_checkout else "OČISTI PLOČU"
                            ),
                            subtitle="",
                            title_only=True,
                            title_scale=1.22,
                        )
                else:
                    gui_ctx["_dart_can_submit_since"] = None
                    gui_ctx["_dart_manual_advance_confirm"] = False

                has_undo = bool((game.get("undo_stack") or [])) and not block_adv and not waiting_clear
                add_button(
                    "undo_turn",
                    pad,
                    y_exit_top,
                    undo_x2,
                    y_bot,
                    "NAZAD",
                    active=has_undo,
                    style="default" if has_undo else "disabled",
                )
                add_button(
                    "request_exit_game",
                    action_x1,
                    y_exit_top,
                    action_x2,
                    y_bot,
                    "X",
                    active=(not getattr(args, "kiosk", False)),
                    style="danger",
                    text_thick=3,
                )

            # Killer bull life-steal picker (OpenCV fallback UI)
            pend = gui_ctx.get("killer_bull_pending")
            if isinstance(pend, dict) and winner is None:
                overlay = panel.copy()
                cv2.rectangle(overlay, (0, 0), (width - 1, height - 1), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.72, panel, 0.28, 0, panel)
                try:
                    from kiosk_settings import get_settings

                    prompt = get_settings().t("killer_bull_prompt")
                    false_lab = get_settings().t("killer_bull_false")
                except Exception:
                    prompt = "Pogodili ste bull — izaberite igraca"
                    false_lab = "Nisam pogodio bull"
                attacker = int(pend.get("attacker", cur))
                targets = _killer_bull_steal_targets(game, attacker)
                card_w = max(280, int(width * 0.88))
                card_x1 = (width - card_w) // 2
                card_x2 = card_x1 + card_w
                btn_h = max(48, int(height * 0.09))
                gap = max(8, int(height * 0.015))
                title_fs = max(0.55, min(0.85, width / 520.0))
                lines = []
                words = prompt.split()
                cur_line = ""
                for word in words:
                    trial = (cur_line + " " + word).strip()
                    (tw, _), _ = cv2.getTextSize(trial, cv2.FONT_HERSHEY_DUPLEX, title_fs, 2)
                    if tw > card_w - 24 and cur_line:
                        lines.append(cur_line)
                        cur_line = word
                    else:
                        cur_line = trial
                if cur_line:
                    lines.append(cur_line)
                n_btns = len(targets) + 1
                block_h = len(lines) * 28 + 16 + n_btns * (btn_h + gap)
                y0 = max(12, (height - block_h) // 2)
                for li, line in enumerate(lines):
                    (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_DUPLEX, title_fs, 2)
                    cv2.putText(
                        panel,
                        line,
                        ((width - tw) // 2, y0 + th),
                        cv2.FONT_HERSHEY_DUPLEX,
                        title_fs,
                        UI["text"],
                        2,
                    )
                    y0 += th + 8
                y0 += 8
                # Replace clickable buttons with picker only while pending
                buttons.clear()
                for ti in targets:
                    lab = f"P{ti + 1}  ({int((players[ti].get('lives') or 0))} zivota)"
                    add_button(
                        f"killer_bull_steal:{ti}",
                        card_x1,
                        y0,
                        card_x2,
                        y0 + btn_h,
                        lab,
                        active=True,
                        style="confirm_gray",
                        text_thick=2,
                    )
                    y0 += btn_h + gap
                add_button(
                    "killer_bull_false",
                    card_x1,
                    y0,
                    card_x2,
                    y0 + btn_h,
                    false_lab,
                    active=True,
                    style="danger",
                    text_thick=2,
                )

        gui_ctx["buttons"] = buttons
        return panel

    def _build_keypad_view(size: int = 640) -> np.ndarray:
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:] = UI["game_bg"]
        buttons = []
        game = gui_ctx.get("game") or {}
        mult = int(gui_ctx.get("input_multiplier", 1))

        def add_kbtn(kid: str, x1: int, y1: int, x2: int, y2: int, text: str, style: str = "default", active: bool = False):
            h = max(1, y2 - y1)
            fs = max(0.65, min(1.25, h / 50.0))
            if kid.startswith("num:"):
                fs = max(fs, 1.05)
            _panel_draw_button(img, x1, y1, x2, y2, text, style=style, active=active, font_scale=fs)
            buttons.append({"id": kid, "rect": (x1, y1, x2, y2)})

        pad = 12
        w = size - 2 * pad
        game_title = _game_display_title(game) if game else "GAME"
        title_fs = 0.72
        title_th = 2
        for _ in range(12):
            (ttw, _), _ = cv2.getTextSize(game_title, cv2.FONT_HERSHEY_DUPLEX, title_fs, title_th)
            if ttw <= w or title_fs <= 0.42:
                break
            title_fs *= 0.92
        cv2.putText(img, game_title, (pad, 30), cv2.FONT_HERSHEY_DUPLEX, title_fs, UI["text"], title_th)

        gap = 8

        # Top controls: D/T one-shot multiplier + quick actions (6 buttons).
        top_y = 40
        bh = 58
        bw = (w - 5 * gap) // 6
        x = pad
        for m, lab in ((2, "D"), (3, "T")):
            add_kbtn(
                f"mult:{m}", x, top_y, x + bw, top_y + bh, lab,
                style="rule_on" if mult == m else "rule_off", active=(mult == m),
            )
            if mult == m:
                cv2.rectangle(img, (x + 2, top_y + 2), (x + bw - 2, top_y + bh - 2), (0, 200, 255), 3)
            x += bw + gap
        add_kbtn("bull:25", x, top_y, x + bw, top_y + bh, "B25", style="primary", active=False)
        x += bw + gap
        add_kbtn("bull:50", x, top_y, x + bw, top_y + bh, "B50", style="primary", active=False)
        x += bw + gap
        add_kbtn("hit:miss", x, top_y, x + bw, top_y + bh, "MISS", style="miss", active=False)
        x += bw + gap
        add_kbtn("clear_one", x, top_y, x + bw, top_y + bh, "CLR", style="default", active=False)

        # Number grid 1..20 with square buttons.
        grid_y = top_y + bh + 12
        cols = 5
        rows = 4
        cw = (w - (cols - 1) * gap) // cols
        max_ch = (size - grid_y - pad - (rows - 1) * gap) // rows
        cell = min(cw, max_ch)
        nums = list(range(1, 21))
        idx = 0
        for r in range(rows):
            for c in range(cols):
                n = nums[idx]
                x1 = pad + c * (cell + gap)
                y1 = grid_y + r * (cell + gap)
                add_kbtn(f"num:{n}", x1, y1, x1 + cell, y1 + cell, str(n), style="primary", active=False)
                idx += 1

        gui_ctx["keypad_buttons"] = buttons
        gui_ctx["hit_slot_buttons"] = []
        return img

    def _build_hits_view(size: int = 640) -> np.ndarray:
        """Tri automatska pikadera — klik otvara tipkovnicu za ispravak."""
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:] = UI["game_bg"]
        buttons: List[dict] = []
        game = gui_ctx.get("game") or {}
        turn_hits = game.get("turn_hits", [None, None, None])
        cur = int(gui_state.get("current_player_idx", 0))
        blocks_new_hits = _turn_blocks_new_hits(game, cur, turn_hits)

        pad = 12
        w = size - 2 * pad
        game_title = _game_display_title(game) if game else "GAME"
        title_fs = 0.72
        title_th = 2
        for _ in range(12):
            (ttw, _), _ = cv2.getTextSize(game_title, cv2.FONT_HERSHEY_DUPLEX, title_fs, title_th)
            if ttw <= w or title_fs <= 0.42:
                break
            title_fs *= 0.92
        cv2.putText(img, game_title, (pad, 30), cv2.FONT_HERSHEY_DUPLEX, title_fs, UI["text"], title_th)

        gap = 10
        title_band = 40
        slot_h = max(72, (size - title_band - pad - 2 * gap) // 3)
        for i in range(3):
            y1 = title_band + i * (slot_h + gap)
            y2 = min(size - pad, y1 + slot_h)
            tag = _hit_tag(turn_hits[i])
            slot_clickable = not (blocks_new_hits and turn_hits[i] is None)
            slot_style = "hit_slot" if slot_clickable else "disabled"
            _panel_draw_button(img, pad, y1, size - pad, y2, "", style=slot_style, active=False, font_scale=0.65)
            if slot_clickable:
                buttons.append({"id": f"hit_slot:{i}", "rect": (pad, y1, size - pad, y2)})

            tag_fs = max(1.75, min(3.15, slot_h / 22.0))
            tag_th = 5
            for _ in range(10):
                (tw, th), bl = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, tag_fs, tag_th)
                if tw <= w - 28 or tag_fs <= 1.0:
                    break
                tag_fs *= 0.92
            tx = pad + max(0, (w - tw) // 2)
            ty = y1 + (y2 - y1 + th) // 2 - bl // 2
            if tag == "MISS":
                tag_col = UI["btn_red"]
            elif tag == "-":
                tag_col = UI["muted"]
            else:
                tag_col = UI["hits_green"]
            cv2.putText(img, tag, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, tag_fs, tag_col, tag_th)

        gui_ctx["hit_slot_buttons"] = buttons
        gui_ctx["keypad_buttons"] = []
        return img

    def _close_hit_editor() -> None:
        gui_ctx["editing_hit_slot"] = None
        gui_ctx["input_multiplier"] = 1

    def _hit_test(x: int, y: int, rect: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = rect
        return x1 <= x < x2 and y1 <= y < y2

    def _map_click_to_view_in_canvas(x: int, y: int) -> Optional[Tuple[float, float]]:
        scale = float(gui_ctx.get("_display_scale", 1.0)) or 1.0
        off_x = float(gui_ctx.get("_display_off_x", 0.0))
        off_y = float(gui_ctx.get("_display_off_y", 0.0))
        x_canvas = int((x - off_x) / max(1e-9, scale))
        y_canvas = int((y - off_y) / max(1e-9, scale))
        vw = int(gui_ctx.get("_stack_view_w", 0))
        vh = int(gui_ctx.get("_stack_view_h", 0))
        if vw <= 0 or vh <= 0:
            return None
        if gui_ctx.get("_layout_split"):
            if 0 <= x_canvas < vw and 0 <= y_canvas < vh:
                return (float(x_canvas), float(y_canvas))
            return None
        ch = int(gui_ctx.get("_stack_control_h", 0))
        if y_canvas < ch:
            return None
        vx = x_canvas
        vy = y_canvas - ch
        if 0 <= vx < vw and 0 <= vy < vh:
            return (float(vx), float(vy))
        return None

    def _set_turn_hit(slot: int, hit: DetectedHit):
        game = gui_ctx.get("game") or {}
        turn_hits = game.get("turn_hits", [None, None, None])
        if 0 <= slot < 3:
            turn_hits[slot] = hit
            game["turn_hits"] = turn_hits
            all_hits.append(hit)
            cur = int(gui_state.get("current_player_idx", 0))
            _killer_after_hit_registered(gui_ctx, game, cur, slot, hit)

    def _dart_hit_from_fused(fused: FusedDartResult) -> DetectedHit:
        zone = str(fused.zone_name).lower()
        if zone == "inner_bull":
            return _make_bull_hit(True)
        if zone == "outer_bull":
            return _make_bull_hit(False)
        if zone == "miss" or int(fused.score) <= 0:
            return _make_miss_hit()
        mult = {"single": 1, "double": 2, "triple": 3}.get(zone, 1)
        return _make_number_hit(int(fused.segment_number), mult)

    def _fused_hit_label(fused: FusedDartResult) -> str:
        zone = str(fused.zone_name).lower()
        if zone in ("inner_bull", "outer_bull"):
            return "B50" if zone == "inner_bull" else "B25"
        if zone == "miss":
            return "MISS"
        prefix = {"single": "S", "double": "D", "triple": "T"}.get(zone, "S")
        return f"{prefix}{fused.segment_number}"

    def _capture_dart_reference() -> int:
        cam_mgr = gui_ctx.get("camera_mgr")
        dart_det = gui_ctx.get("dart_detector")
        board_cal = gui_ctx.get("board_calibrator")
        if cam_mgr is None or dart_det is None or board_cal is None or not cam_mgr.active:
            print("[dart] REF: kamere nisu aktivne", flush=True)
            return 0
        frames = cam_mgr.grab_frames()
        saved = dart_det.capture_references(board_cal, frames)
        if saved > 0:
            gui_ctx["_dart_status"] = f"REF {saved} cam"
        gui_ctx["_dart_no_ref_warned"] = False
        print(f"[dart] REF: spremljeno {saved} kamera", flush=True)
        return saved

    def _log_dart_motion(
        probe: List[DartDetectResult],
        *,
        fused: Optional[FusedDartResult] = None,
        state: str = "",
        force: bool = False,
    ) -> None:
        any_motion = any(r.motion_pixels >= MOTION_LOG_MIN_PIXELS for r in probe)
        now = time.perf_counter()
        last_log = float(gui_ctx.get("_dart_log_last_tick", 0.0))
        if not force and not any_motion and fused is None and (now - last_log) < 5.0:
            return
        gui_ctx["_dart_log_last_tick"] = now
        line = summarize_motion_log(probe)
        if state:
            line = f"[{state}] {line}"
        print(f"[dart] {line}", flush=True)
        if fused is not None:
            print(f"[dart] {summarize_fused_log(fused)}", flush=True)

    def _run_calibration_detect() -> Tuple[bool, int]:
        """Kalibracija + reference na praznoj ploci. Vraca (sve_kamere_ok, broj_ref)."""
        cam_mgr = gui_ctx.get("camera_mgr")
        board_cal = gui_ctx.get("board_calibrator")
        frames = gui_ctx.get("_last_cal_frames") or {}
        if cam_mgr is not None and cam_mgr.active:
            frames = cam_mgr.grab_frames(flush=2)
            gui_ctx["_last_cal_frames"] = frames
        if board_cal is None or not frames:
            print("[dart] DETEKTIRAJ: nema kadrova", flush=True)
            gui_ctx["_dart_status"] = "DET FAIL"
            return False, 0
        incomplete: List[int] = []
        hints = dict(gui_ctx.get("bull_hints") or {})
        for attempt in range(3):
            if cam_mgr is not None and cam_mgr.active:
                frames = cam_mgr.grab_frames(flush=2)
                gui_ctx["_last_cal_frames"] = frames
            incomplete = board_cal.detect_all(
                frames,
                bull_hints=hints or None,
                ellipse_hints=dict(gui_ctx.get("ellipse_hints") or {}) or None,
            )
            if not incomplete:
                break
            time.sleep(0.08)
        n_ref = _capture_dart_reference()
        all_ok = not incomplete and n_ref > 0
        if incomplete:
            detail = board_cal.format_incomplete_status(incomplete)
            gui_ctx["_dart_status"] = f"KAL {len(CAMERA_INDICES) - len(incomplete)}/{len(CAMERA_INDICES)}"
            print(f"[dart] DETEKTIRAJ: kalibracija nepotpuna — {detail}", flush=True)
            no_signal = board_cal.incomplete_no_signal(incomplete)
            if no_signal:
                print(
                    f"[dart] DETEKTIRAJ: stvarno bez kadra (USB/open/read): {no_signal}",
                    flush=True,
                )
            gui_ctx["_cal_click_bull_mode"] = True
            print(
                "[dart] DETEKTIRAJ: uključen KLIKNI BULL — klikni bull po kameri, pa opet KALIBRIRAJ",
                flush=True,
            )
        else:
            gui_ctx["_dart_status"] = f"REF {n_ref} cam" if n_ref > 0 else "REF FAIL"
            gui_ctx["bull_hints"] = {}
            gui_ctx["_cal_click_bull_mode"] = False
        if n_ref > 0:
            gui_ctx["_dart_no_empty_ref_warned"] = False
        print(f"[dart] DETEKTIRAJ: kalibracija + ref ({n_ref} cam)", flush=True)
        return all_ok, n_ref

    def _start_playing_after_clear_board() -> None:
        gui_state["screen"] = "playing"
        gui_ctx["_dart_turn_start_pending"] = True
        gui_ctx["_dart_no_ref_warned"] = False
        gui_ctx["_dart_block_auto_advance"] = False
        gui_ctx["_dart_block_auto_detect"] = False
        gui_ctx["_dart_undo_target_player"] = None
        gui_ctx["_dart_undo_pending_hits"] = None
        gui_ctx["_dart_undo_pending_bull_steals"] = None
        gui_ctx["_dart_score_locked"] = False
        gui_ctx["_dart_empty_frames"] = 0
        gui_ctx["_dart_status"] = ""
        gui_ctx["_clear_board_status"] = ""
        gui_ctx["editing_hit_slot"] = None
        dart_det = gui_ctx.get("dart_detector")
        if dart_det is not None:
            dart_det.reset_motion_state()
        print("[dart] igra pokrenuta", flush=True)

    def _sync_board_refs_after_undo() -> None:
        """Nakon NAZAD: postojece strijele na ploci postaju referenca, nova se detektiraju."""
        cam_mgr = gui_ctx.get("camera_mgr")
        dart_det = gui_ctx.get("dart_detector")
        board_cal = gui_ctx.get("board_calibrator")
        if dart_det is None:
            return
        if (
            cam_mgr is not None
            and board_cal is not None
            and cam_mgr.active
        ):
            frames = cam_mgr.grab_frames()
            dart_det.commit_board_snapshot(board_cal, frames)
        dart_det.reset_motion_state()
        gui_ctx["_dart_wait_idle_after_switch"] = False
        gui_ctx["_dart_idle_frames"] = 0
        gui_ctx["_dart_score_locked"] = False
        gui_ctx["_dart_can_submit_since"] = None
        gui_ctx["_dart_manual_advance_confirm"] = False

    def _confirm_clear_board_and_start() -> None:
        cam_mgr = gui_ctx.get("camera_mgr")
        dart_det = gui_ctx.get("dart_detector")
        board_cal = gui_ctx.get("board_calibrator")
        if (
            cam_mgr is None
            or dart_det is None
            or board_cal is None
            or not cam_mgr.active
        ):
            gui_ctx["_clear_board_status"] = "Kamere nisu aktivne"
            return
        from kiosk_settings import get_settings

        auto_cal = bool(get_settings().auto_calibrate)
        gui_ctx["_clear_board_status"] = get_settings().t(
            "calibrating" if auto_cal else "saving_ref"
        )
        print("[dart] clear_board: cekam kamere...", flush=True)
        from calibration import CAMERA_INDICES as _CAM_IDX

        if not cam_mgr.wait_for_frames(timeout_sec=12.0, min_cams=len(_CAM_IDX)):
            if not cam_mgr.wait_for_frames(timeout_sec=0.5, min_cams=1):
                gui_ctx["_clear_board_status"] = "Kamere se još otvaraju — pokušaj ponovo"
                print("[dart] clear_board: kamere nisu bile spremne", flush=True)
                return
        frames = cam_mgr.grab_frames(flush=2)
        for _ in range(3):
            time.sleep(0.05)
            frames = cam_mgr.grab_frames(flush=1)
        if auto_cal:
            gui_ctx["_clear_board_status"] = get_settings().t("calibrating")
            incomplete: List[int] = []
            hints = dict(gui_ctx.get("bull_hints") or {})
            incomplete, frames = board_cal.recalibrate_for_play(
                lambda flush=1: cam_mgr.grab_frames(flush=flush),
                attempts=4,
                bull_hints=hints or None,
                ellipse_hints=dict(gui_ctx.get("ellipse_hints") or {}) or None,
            )
            if incomplete:
                no_sig = board_cal.incomplete_no_signal(incomplete)
                if no_sig and len(no_sig) >= len(incomplete):
                    print("[dart] clear_board: retry nakon no_signal...", flush=True)
                    time.sleep(0.4)
                    cam_mgr.wait_for_frames(timeout_sec=4.0, min_cams=len(_CAM_IDX))
                    incomplete, frames = board_cal.recalibrate_for_play(
                        lambda flush=1: cam_mgr.grab_frames(flush=flush),
                        attempts=3,
                        bull_hints=hints or None,
                        ellipse_hints=dict(gui_ctx.get("ellipse_hints") or {}) or None,
                    )
            if incomplete:
                detail = board_cal.format_incomplete_status(incomplete)
                gui_ctx["_clear_board_status"] = get_settings().t("cal_incomplete")
                print(f"[dart] clear_board: kalibracija nepotpuna — {detail}", flush=True)
                return
            gui_ctx["bull_hints"] = {}
            gui_ctx["_cal_click_bull_mode"] = False
            print("[dart] clear_board: ploca ponovo kalibrirana (stable)", flush=True)
        else:
            gui_ctx["_clear_board_status"] = get_settings().t("saving_ref")
        n_ref = dart_det.update_empty_board_references(
            board_cal, frames, include_raw=True
        )
        if n_ref <= 0:
            gui_ctx["_clear_board_status"] = get_settings().t(
                "no_saved_cal" if not auto_cal else "ref_not_saved"
            )
            return
        dart_det.commit_board_snapshot(board_cal, frames)
        dart_det.reset_motion_state()
        gui_ctx["_clear_board_status"] = ""
        print(f"[dart] prazna ploca ref ({n_ref} cam)", flush=True)
        _start_playing_after_clear_board()

    def _prepare_dart_turn_start(
        *, commit_snapshot: bool = True, skip_idle_wait: bool = False
    ) -> None:
        cam_mgr = gui_ctx.get("camera_mgr")
        dart_det = gui_ctx.get("dart_detector")
        board_cal = gui_ctx.get("board_calibrator")
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
        gui_ctx["_dart_wait_idle_after_switch"] = not skip_idle_wait
        gui_ctx["_dart_idle_frames"] = 0
        gui_ctx["_dart_empty_frames"] = 0
        gui_ctx["_dart_score_locked"] = False
        gui_ctx["_dart_last_tick"] = 0.0
        g = gui_ctx.get("game")
        if g is not None:
            gui_ctx["_turn_players_snap"] = copy.deepcopy(g.get("players") or [])
            gui_ctx["_turn_elim_snap"] = copy.deepcopy(g.get("eliminated_order") or [])
            gui_ctx["_turn_winner_snap"] = g.get("winner_player_idx")
            gui_ctx["_turn_placements_snap"] = copy.deepcopy(g.get("placements") or [])
        gui_ctx["killer_bull_pending"] = None
        gui_ctx["killer_bull_steals"] = {}

    def _dart_scoring_paused(
        game: Dict, cur_idx: int, turn_hits: List[Optional[DetectedHit]]
    ) -> bool:
        if gui_ctx.get("killer_bull_pending"):
            return True
        if all(h is not None for h in turn_hits):
            return True
        if _turn_is_early_finish(game, cur_idx, turn_hits):
            return True
        if _x01_turn_is_bust(game, cur_idx, turn_hits):
            return True
        return False

    def _can_auto_advance_on_empty_board(
        game: Dict, cur_idx: int, turn_hits: List[Optional[DetectedHit]]
    ) -> bool:
        return _turn_can_submit(game, cur_idx, turn_hits) or _killer_assign_needs_retry(
            game, cur_idx, turn_hits
        )

    def _reset_same_player_after_board_clear(*, skip_idle_wait: bool = True) -> None:
        """Clear dart slots and stay on current player (Killer assign retry)."""
        g = gui_ctx.get("game")
        if not g:
            return
        g["turn_hits"] = [None, None, None]
        gui_ctx["_dart_status"] = ""
        gui_ctx["_dart_wait_board_clear"] = False
        gui_ctx["_dart_board_clear_frames"] = 0
        gui_ctx["_dart_can_submit_since"] = None
        gui_ctx["_dart_manual_advance_confirm"] = False
        gui_ctx["_dart_undo_target_player"] = None
        gui_ctx["_dart_undo_pending_hits"] = None
        gui_ctx["_dart_undo_pending_bull_steals"] = None
        gui_ctx["_dart_block_auto_advance"] = False
        gui_ctx["_dart_block_auto_detect"] = False
        gui_ctx["_dart_empty_frames"] = 0
        _close_hit_editor()
        _prepare_dart_turn_start(commit_snapshot=True, skip_idle_wait=skip_idle_wait)
        print("[dart] Killer assign — ploca prazna, isti igrac ponavlja", flush=True)

    def _confirm_undo_advance_to_next() -> None:
        """NAZAD mod: ponovno boduj uredjene P1 hitove, zatim predaj P2 detektirane."""
        g = gui_ctx.get("game")
        target = gui_ctx.get("_dart_undo_target_player")
        if g is None or target is None:
            return
        pending = list(gui_ctx.get("_dart_undo_pending_hits") or [None, None, None])
        pending_steals_raw = gui_ctx.get("_dart_undo_pending_bull_steals") or {}
        pending_steals = (
            {int(k): int(v) for k, v in pending_steals_raw.items()}
            if isinstance(pending_steals_raw, dict)
            else {}
        )
        finished_idx = int(gui_state.get("current_player_idx", 0))
        finished_hits = list(g.get("turn_hits", [None, None, None]))
        cam_mgr = gui_ctx.get("camera_mgr")
        dart_det = gui_ctx.get("dart_detector")
        board_cal = gui_ctx.get("board_calibrator")
        players_before = copy.deepcopy(
            gui_ctx.get("_turn_players_snap") or g.get("players") or []
        )
        round_before = int(g.get("round_number", 1))
        winner_before = gui_ctx.get("_turn_winner_snap", g.get("winner_player_idx"))
        winner_at_before = g.get("winner_at")
        placements_before = copy.deepcopy(
            gui_ctx.get("_turn_placements_snap") or g.get("placements") or []
        )
        elim_before = copy.deepcopy(
            gui_ctx.get("_turn_elim_snap") or g.get("eliminated_order") or []
        )
        killer_phase_before = g.get("killer_phase")
        if str(g.get("mode", "")).lower() == "killer":
            _killer_commit_turn(gui_ctx, g, finished_idx, finished_hits)
        else:
            if isinstance(gui_ctx.get("_turn_players_snap"), list):
                g["players"] = copy.deepcopy(gui_ctx["_turn_players_snap"])
            _apply_turn_score(g, finished_idx, finished_hits)
        if g.get("winner_player_idx") is not None and g.get("winner_at") is None:
            g["winner_at"] = time.time()
        nxt = int(target)
        if g.get("winner_player_idx") is None:
            gui_state["current_player_idx"] = nxt
            _advance_round_if_needed(g, finished_idx, nxt)
            g["turn_hits"] = pending
            _push_undo_after_advance(
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
            g["turn_hits"] = [None, None, None]
            g["undo_stack"] = []
        gui_ctx["_dart_undo_target_player"] = None
        gui_ctx["_dart_undo_pending_hits"] = None
        gui_ctx["_dart_undo_pending_bull_steals"] = None
        gui_ctx["_dart_block_auto_advance"] = False
        gui_ctx["_dart_block_auto_detect"] = False
        gui_ctx["_dart_can_submit_since"] = None
        gui_ctx["_dart_manual_advance_confirm"] = False
        gui_ctx["_dart_wait_board_clear"] = False
        gui_ctx["_dart_board_clear_frames"] = 0
        gui_ctx["_dart_empty_frames"] = 0
        _close_hit_editor()
        if dart_det is not None:
            dart_det.reset_motion_state()
        if (
            cam_mgr is not None
            and dart_det is not None
            and board_cal is not None
            and cam_mgr.active
        ):
            frames = cam_mgr.grab_frames()
            dart_det.commit_board_snapshot(board_cal, frames)
        # Baseline snap = post-commit players; then re-apply interrupted pending hits
        # so Killer lives / is_killer match the restored turn_hits immediately.
        _prepare_dart_turn_start(commit_snapshot=False, skip_idle_wait=True)
        if g.get("winner_player_idx") is None and _killer_is_play_phase(g):
            if pending_steals:
                gui_ctx["killer_bull_steals"] = dict(pending_steals)
            _killer_resync_mid_turn(
                gui_ctx, g, int(nxt), list(g.get("turn_hits") or [None, None, None])
            )
        print(f"[dart] P{nxt + 1} — preuzima detektirane strelice (score azuriran)", flush=True)

    def _advance_to_next_player(*, auto: bool = False, skip_idle_wait: bool = False) -> None:
        g = gui_ctx.get("game")
        if not g:
            return
        turn_hits = g.get("turn_hits", [None, None, None])
        cur = int(gui_state.get("current_player_idx", 0))
        # Assign: never advance without a successful unique claim.
        if (
            str(g.get("mode", "")).lower() == "killer"
            and str(g.get("killer_phase", "play")).lower() == "assign"
            and not _killer_number_claimed_in_turn(g, cur, turn_hits)
        ):
            if _turn_fully_entered(turn_hits):
                _reset_same_player_after_board_clear(skip_idle_wait=skip_idle_wait)
            return
        final_hits = list(turn_hits)
        players_before = copy.deepcopy(
            gui_ctx.get("_turn_players_snap") or g.get("players") or []
        )
        round_before = int(g.get("round_number", 1))
        winner_before = gui_ctx.get("_turn_winner_snap", g.get("winner_player_idx"))
        winner_at_before = g.get("winner_at")
        placements_before = copy.deepcopy(
            gui_ctx.get("_turn_placements_snap") or g.get("placements") or []
        )
        elim_before = copy.deepcopy(
            gui_ctx.get("_turn_elim_snap") or g.get("eliminated_order") or []
        )
        killer_phase_before = g.get("killer_phase")
        if str(g.get("mode", "")).lower() == "killer":
            _killer_commit_turn(gui_ctx, g, cur, final_hits)
        else:
            _apply_turn_score(g, cur, final_hits)
        if g.get("winner_player_idx") is not None and g.get("winner_at") is None:
            g["winner_at"] = time.time()
        nxt_for_undo = None
        if g.get("winner_player_idx") is None:
            nxt_for_undo = _next_active_player_idx(g, cur)
            gui_state["current_player_idx"] = nxt_for_undo
            _advance_round_if_needed(g, cur, nxt_for_undo)
        g["turn_hits"] = [None, None, None]
        gui_ctx["_dart_status"] = ""
        gui_ctx["_dart_wait_board_clear"] = False
        gui_ctx["_dart_board_clear_frames"] = 0
        gui_ctx["_dart_can_submit_since"] = None
        gui_ctx["_dart_manual_advance_confirm"] = False
        gui_ctx["_dart_undo_target_player"] = None
        gui_ctx["_dart_undo_pending_hits"] = None
        gui_ctx["_dart_undo_pending_bull_steals"] = None
        _close_hit_editor()
        if not auto:
            gui_ctx["_dart_block_auto_advance"] = False
            gui_ctx["_dart_block_auto_detect"] = False
        if nxt_for_undo is not None:
            _push_undo_after_advance(
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
        _prepare_dart_turn_start(commit_snapshot=True, skip_idle_wait=skip_idle_wait)
        if auto:
            print("[dart] AUTO IDUCI IGRAC: ploca prazna", flush=True)
        else:
            print("[dart] IDUCI IGRAC (preskoci)", flush=True)

    def _force_advance_past_board_clear(*, update_empty_ref: bool = False) -> None:
        """Ručni preskok ako auto detekcija prazne ploče zapne."""
        g = gui_ctx.get("game")
        if not g:
            return
        turn_hits = g.get("turn_hits", [None, None, None])
        cur = int(gui_state.get("current_player_idx", 0))
        needs_retry = _killer_assign_needs_retry(g, cur, turn_hits)
        if not needs_retry and not _turn_can_submit(g, cur, turn_hits):
            return
        cam_mgr = gui_ctx.get("camera_mgr")
        dart_det = gui_ctx.get("dart_detector")
        board_cal = gui_ctx.get("board_calibrator")
        if dart_det is not None:
            dart_det.reset_motion_state()
        gui_ctx["_dart_wait_board_clear"] = False
        gui_ctx["_dart_board_clear_frames"] = 0
        gui_ctx["_dart_empty_frames"] = 0
        gui_ctx["_dart_block_auto_advance"] = False
        gui_ctx["_dart_block_auto_detect"] = False
        gui_ctx["_dart_manual_advance_confirm"] = False
        from kiosk_settings import get_settings

        refresh_empty = bool(update_empty_ref) and not bool(get_settings().manual_mode)
        if (
            cam_mgr is not None
            and dart_det is not None
            and board_cal is not None
            and cam_mgr.active
        ):
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
            _reset_same_player_after_board_clear(skip_idle_wait=True)
        else:
            _advance_to_next_player(auto=True, skip_idle_wait=True)

    def _try_auto_empty_board_advance() -> bool:
        g = gui_ctx.get("game")
        if gui_ctx.get("_dart_block_auto_advance"):
            gui_ctx["_dart_empty_frames"] = 0
            return False
        if not g:
            gui_ctx["_dart_empty_frames"] = 0
            return False
        # Winner mid-turn (Killer end): still allow clear → commit → leaderboard.
        if g.get("winner_player_idx") is not None and not _winner_pending_board_clear(g):
            gui_ctx["_dart_empty_frames"] = 0
            return False
        turn_hits = g.get("turn_hits", [None, None, None])
        cur = int(gui_state.get("current_player_idx", 0))
        needs_retry = _killer_assign_needs_retry(g, cur, turn_hits)
        if not needs_retry and not _can_auto_advance_on_empty_board(g, cur, turn_hits):
            gui_ctx["_dart_empty_frames"] = 0
            return False
        cam_mgr = gui_ctx.get("camera_mgr")
        dart_det = gui_ctx.get("dart_detector")
        board_cal = gui_ctx.get("board_calibrator")
        if (
            cam_mgr is None
            or dart_det is None
            or board_cal is None
            or not cam_mgr.active
        ):
            return False
        if not dart_det.has_empty_references():
            if not gui_ctx.get("_dart_no_empty_ref_warned"):
                print(
                    "[dart] auto-prebacivanje: nema prazne reference — DETEKTIRAJ na praznoj ploci",
                    flush=True,
                )
                gui_ctx["_dart_no_empty_ref_warned"] = True
            return False
        frames = cam_mgr.grab_frames()
        empty_ok, max_px, max_diff = dart_det.is_board_like_empty(board_cal, frames)
        now = time.perf_counter()
        last_empty_log = float(gui_ctx.get("_dart_empty_log_last", 0.0))
        if now - last_empty_log >= 2.0:
            gui_ctx["_dart_empty_log_last"] = now
            print(
                f"[dart] empty probe px={max_px} td={int(getattr(dart_det, '_last_empty_td_px', 0))} "
                f"raw={int(getattr(dart_det, '_last_empty_raw_px', 0))} diff={max_diff:.2f} "
                f"(prag td<={EMPTY_BOARD_MAX_PIXELS} raw_hand>={EMPTY_BOARD_RAW_HAND_PIXELS} "
                f"diff<={EMPTY_BOARD_MAX_DIFF_MEAN})",
                flush=True,
            )
        if empty_ok:
            gui_ctx["_dart_empty_frames"] = int(gui_ctx.get("_dart_empty_frames", 0)) + 1
        else:
            cur_n = int(gui_ctx.get("_dart_empty_frames", 0))
            gui_ctx["_dart_empty_frames"] = max(0, cur_n - int(EMPTY_BOARD_FAIL_PENALTY))
        if gui_ctx["_dart_empty_frames"] < EMPTY_BOARD_CONFIRM_FRAMES:
            return False
        gui_ctx["_dart_empty_frames"] = 0
        if dart_det is not None:
            dart_det.reset_motion_state()
        n_empty = dart_det.update_empty_board_references(board_cal, frames)
        dart_det.commit_board_snapshot(board_cal, frames, persist=False)
        print(
            f"[dart] auto-prebacivanje — empty ref azuriran ({n_empty} cam) + motion ref",
            flush=True,
        )
        if needs_retry:
            _reset_same_player_after_board_clear(skip_idle_wait=True)
        else:
            _advance_to_next_player(auto=True, skip_idle_wait=True)
        return True

    def _dart_should_enter_rearm(
        game: Dict, cur_idx: int, turn_hits: List[Optional[DetectedHit]]
    ) -> bool:
        """Rearm samo: 3/3 upisano, IDUCI IGRAC dostupan, bez undo blokade."""
        if gui_ctx.get("_dart_block_auto_advance"):
            return False
        if not _turn_fully_entered(turn_hits):
            return False
        return _turn_can_submit(game, cur_idx, turn_hits) or _killer_assign_needs_retry(
            game, cur_idx, turn_hits
        )

    def _apply_detected_hit(
        slot: int, hit: DetectedHit, *, undo_mode: bool = False
    ) -> None:
        if undo_mode:
            pending = list(gui_ctx.get("_dart_undo_pending_hits") or [None, None, None])
            if 0 <= slot < 3:
                pending[slot] = hit
                gui_ctx["_dart_undo_pending_hits"] = pending
                all_hits.append(hit)
            return
        _set_turn_hit(slot, hit)

    def _try_auto_detect_dart(*, force_log: bool = False, undo_mode: bool = False) -> bool:
        g = gui_ctx.get("game")
        if not g or g.get("winner_player_idx") is not None:
            return False
        if undo_mode:
            if not gui_ctx.get("_dart_block_auto_advance"):
                return False
        elif gui_ctx.get("_dart_block_auto_detect"):
            return False
        turn_hits = g.get("turn_hits", [None, None, None])
        cur = int(gui_state.get("current_player_idx", 0))
        if undo_mode:
            hit_slots = list(gui_ctx.get("_dart_undo_pending_hits") or [None, None, None])
        else:
            hit_slots = turn_hits
            if _dart_scoring_paused(g, cur, turn_hits):
                return False
        slots_full = all(h is not None for h in hit_slots)
        if undo_mode and slots_full:
            return False
        cam_mgr = gui_ctx.get("camera_mgr")
        dart_det = gui_ctx.get("dart_detector")
        board_cal = gui_ctx.get("board_calibrator")
        if cam_mgr is None or dart_det is None or board_cal is None or not cam_mgr.active:
            if force_log:
                print("[dart] AUTO: kamere nisu aktivne", flush=True)
            return False
        if not dart_det.has_references():
            if force_log:
                print("[dart] AUTO: nema reference (DETEKTIRAJ u kalibraciji)", flush=True)
            return False
        wait_idle = False if undo_mode else bool(gui_ctx.get("_dart_wait_idle_after_switch", False))
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
                gui_ctx["_dart_idle_frames"] = int(gui_ctx.get("_dart_idle_frames", 0)) + 1
                if gui_ctx["_dart_idle_frames"] >= DART_IDLE_FRAMES_AFTER_SWITCH:
                    gui_ctx["_dart_wait_idle_after_switch"] = False
                    gui_ctx["_dart_idle_frames"] = 0
                    print("[dart] spreman za novu strijelu", flush=True)
            else:
                gui_ctx["_dart_idle_frames"] = 0
        if force_log or state.startswith("armed") or state in ("hand_block", "hand_cooldown") or fused is not None:
            _log_dart_motion(probe, fused=fused, state=state, force=force_log)
            if state == "hand_block":
                gui_ctx["_dart_status"] = "CEKAJ RUKA"
            elif state == "hand_cooldown":
                gui_ctx["_dart_status"] = "CEKAJ RUKA"
        elif any(r.motion_pixels >= MOTION_LOG_MIN_PIXELS for r in probe):
            _log_dart_motion(probe, state=state)

        dart_det.try_commit_deferred_snapshot(board_cal, frames)

        if fused is None:
            if state == "fired_fail":
                fire_frames = dart_det.last_fire_frames()
                commit_frames = fire_frames if len(fire_frames) >= 2 else cam_mgr.grab_frames()
                dart_det.commit_board_snapshot(board_cal, commit_frames, persist=False)
                dart_det.finalize_shot(enter_rearm=False)
                print("[dart] AUTO: fusion fail — ignore (noise)", flush=True)
                return False
            if state == "post_fire":
                dart_det.abort_shot()
            return False

        is_miss = str(fused.zone_name).lower() == "miss" or int(fused.score) <= 0
        low_conf = not is_miss and float(fused.confidence) < DART_MIN_CONFIDENCE

        if low_conf:
            fire_frames = dart_det.last_fire_frames()
            commit_frames = fire_frames if len(fire_frames) >= 2 else cam_mgr.grab_frames()
            dart_det.commit_board_snapshot(board_cal, commit_frames, persist=False)
            dart_det.abort_shot()
            if force_log:
                print(
                    f"[dart] AUTO: odbijeno conf={fused.confidence:.2f} < {DART_MIN_CONFIDENCE}",
                    flush=True,
                )
            return False

        slot = _next_insert_slot(hit_slots)
        if slot >= 3:
            dart_det.abort_shot()
            return False
        if is_miss:
            _apply_detected_hit(slot, _make_miss_hit(), undo_mode=undo_mode)
            label = "MISS"
        else:
            _apply_detected_hit(slot, _dart_hit_from_fused(fused), undo_mode=undo_mode)
            label = _fused_hit_label(fused)
        if not undo_mode:
            turn_hits = g.get("turn_hits", [None, None, None])
        cams = ",".join(str(c) for c in fused.cam_indices)
        gui_ctx["_dart_status"] = f"AUTO {label} [{cams}]"
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

    def _enter_calibration():
        if gui_state.get("screen") == "calibration":
            return
        gui_state["_prev_screen"] = gui_state.get("screen", "standby")
        cam_mgr = gui_ctx.get("camera_mgr")
        if cam_mgr is not None:
            cam_mgr.start()
            cam_mgr.wait_for_frames(timeout_sec=8.0, min_cams=len(CAMERA_INDICES))
        gui_state["screen"] = "calibration"
        gui_ctx["buttons"] = []

    def _exit_calibration():
        if gui_state.get("screen") != "calibration":
            return
        cam_mgr = gui_ctx.get("camera_mgr")
        if cam_mgr is not None:
            cam_mgr.stop()
        gui_ctx["_cal_canvas_disp"] = None
        gui_ctx["_last_cal_frames"] = {}
        gui_ctx["_cal_click_bull_mode"] = False
        board_cal = gui_ctx.get("board_calibrator")
        if board_cal is not None:
            board_cal.save()
        prev = gui_state.get("_prev_screen") or "standby"
        gui_state["screen"] = prev
        gui_state["_prev_screen"] = None
        gui_ctx["buttons"] = []

    def _handle_nav_back() -> None:
        screen = gui_state.get("screen", "standby")
        mode = str(gui_state.get("selected_game_mode") or "")
        if screen == "select_game":
            gui_state["screen"] = "standby"
        elif screen == "rules_x01":
            gui_state["screen"] = "select_game"
        elif screen == "rules_cricket":
            gui_state["screen"] = "select_game"
        elif screen == "rules_killer":
            gui_state["screen"] = "select_game"
        elif screen == "select_players":
            if mode in ("301", "501"):
                gui_state["screen"] = "rules_x01"
            elif mode == "cricket":
                gui_state["screen"] = "rules_cricket"
            elif mode == "killer":
                gui_state["screen"] = "rules_killer"
            else:
                gui_state["screen"] = "select_game"
        elif screen == "clear_board":
            gui_state["screen"] = "select_players"
            gui_ctx["game"] = None
            gui_ctx["_clear_board_status"] = ""
            all_hits.clear()
        else:
            return

    def _on_main_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONUP:
            return
        scale = float(gui_ctx.get("_display_scale", 1.0)) or 1.0
        off_x = float(gui_ctx.get("_display_off_x", 0.0))
        off_y = float(gui_ctx.get("_display_off_y", 0.0))
        x_canvas = int((x - off_x) / max(1e-9, scale))
        y_canvas = int((y - off_y) / max(1e-9, scale))

        # Panel buttons first
        for btn in gui_ctx.get("buttons", []):
            if not _hit_test(x_canvas, y_canvas, btn["rect"]):
                continue
            bid = btn["id"]
            if bid == "calibration_back":
                _click_sound()
                _exit_calibration()
                return
            if bid == "calibration_detect":
                _click_sound()
                _run_calibration_detect()
                return
            if bid == "calibration_click_bull":
                _click_sound()
                on = not bool(gui_ctx.get("_cal_click_bull_mode"))
                gui_ctx["_cal_click_bull_mode"] = on
                if on:
                    gui_ctx["_cal_click_ellipse_mode"] = False
                gui_ctx["_cal_last_tick"] = 0.0
                return
            if bid in ("calibration_click_ellipse", "calibration_overlay"):
                _click_sound()
                on = not bool(gui_ctx.get("_cal_click_ellipse_mode"))
                gui_ctx["_cal_click_ellipse_mode"] = on
                if on:
                    gui_ctx["_cal_click_bull_mode"] = False
                gui_ctx["_cal_last_tick"] = 0.0
                return
            if bid == "calibration_capture":
                _click_sound()
                board_cal = gui_ctx.get("board_calibrator")
                if board_cal is not None:
                    board_cal.show_topdown = not board_cal.show_topdown
                    if board_cal.show_topdown:
                        board_cal.show_overlay = True
                    gui_ctx["_cal_last_tick"] = 0.0
                return
            if bid.startswith("calibration_seg20_left_"):
                _click_sound()
                board_cal = gui_ctx.get("board_calibrator")
                if board_cal is not None:
                    try:
                        cam_idx = int(bid.rsplit("_", 1)[-1])
                    except ValueError:
                        return
                    board_cal.nudge_segment20(cam_idx, -1)
                    gui_ctx["_cal_last_tick"] = 0.0
                return
            if bid.startswith("calibration_seg20_right_"):
                _click_sound()
                board_cal = gui_ctx.get("board_calibrator")
                if board_cal is not None:
                    try:
                        cam_idx = int(bid.rsplit("_", 1)[-1])
                    except ValueError:
                        return
                    board_cal.nudge_segment20(cam_idx, 1)
                    gui_ctx["_cal_last_tick"] = 0.0
                return
            if bid == "nav_back":
                _click_sound()
                _handle_nav_back()
                return
            if bid.startswith("game:"):
                _click_sound()
                mode = bid.split(":", 1)[1]
                gui_state["selected_game_mode"] = mode
                gui_state["rules_pending"] = {
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
                    gui_state["screen"] = "rules_x01"
                elif mode == "cricket":
                    gui_state["screen"] = "rules_cricket"
                elif mode == "killer":
                    gui_state["screen"] = "rules_killer"
                else:
                    gui_state["screen"] = "select_players"
                return
            if bid == "rule:double_in":
                _click_sound()
                rp = gui_state.setdefault("rules_pending", {})
                rp["double_in"] = not bool(rp.get("double_in"))
                return
            if bid == "rule:double_out":
                _click_sound()
                rp = gui_state.setdefault("rules_pending", {})
                rp["double_out"] = not bool(rp.get("double_out"))
                return
            if bid == "rule:max_rounds_off":
                _click_sound()
                gui_state.setdefault("rules_pending", {})["max_rounds_20"] = False
                return
            if bid == "rule:max_rounds_20":
                _click_sound()
                gui_state.setdefault("rules_pending", {})["max_rounds_20"] = True
                return
            if bid == "rule:cricket_standard":
                _click_sound()
                gui_state.setdefault("rules_pending", {})["cricket_mode"] = "standard"
                return
            if bid == "rule:cricket_no_score":
                _click_sound()
                gui_state.setdefault("rules_pending", {})["cricket_mode"] = "no_score"
                return
            if bid == "rule:cricket_cut_throat":
                _click_sound()
                gui_state.setdefault("rules_pending", {})["cricket_mode"] = "cut_throat"
                return
            if bid.startswith("rule:killer_lives:"):
                _click_sound()
                try:
                    lives = int(bid.split(":")[-1])
                except ValueError:
                    return
                if lives in (3, 5, 7, 10):
                    gui_state.setdefault("rules_pending", {})["killer_lives"] = lives
                return
            if bid.startswith("rule:killer_assign:"):
                _click_sound()
                assign = bid.split(":")[-1]
                if assign in ("random", "throw"):
                    gui_state.setdefault("rules_pending", {})["killer_assign"] = assign
                return
            if bid.startswith("rule:killer_activation:"):
                _click_sound()
                act = bid.split(":")[-1]
                if act in ("single", "double", "triple"):
                    gui_state.setdefault("rules_pending", {})["killer_activation"] = act
                return
            if bid.startswith("killer_bull_steal:"):
                _click_sound()
                g = gui_ctx.get("game")
                if not g:
                    return
                try:
                    ti = int(bid.split(":")[-1])
                except ValueError:
                    return
                _killer_confirm_bull_steal(gui_ctx, g, ti)
                return
            if bid == "killer_bull_false":
                _click_sound()
                g = gui_ctx.get("game")
                if g:
                    _killer_false_bull(gui_ctx, g)
                return
            if bid == "rules_continue":
                _click_sound()
                gui_state["screen"] = "select_players"
                return
            if bid == "clear_board_confirm":
                _click_sound()
                try:
                    from kiosk_settings import get_settings

                    gui_ctx["_clear_board_status"] = get_settings().t("calibrating")
                except Exception:
                    gui_ctx["_clear_board_status"] = "KALIBRIRAM..."
                gui_ctx["_clear_board_pending_run"] = True
                gui_ctx["_clear_board_pending_at"] = time.perf_counter()
                return
            if bid.startswith("players:"):
                _click_sound()
                n = int(bid.split(":", 1)[1])
                mode = gui_state.get("selected_game_mode")
                if str(mode).lower() == "killer" and n < 2:
                    return
                gui_state["num_players"] = n
                gui_state["current_player_idx"] = 0
                rules_pending = dict(gui_state.get("rules_pending") or {})
                gui_ctx["game"] = _create_game(mode, n, rules_pending)
                gui_ctx["input_multiplier"] = 1
                all_hits.clear()
                gui_ctx["editing_hit_slot"] = None
                gui_ctx["_clear_board_status"] = ""
                gui_state["screen"] = "clear_board"
                return
            if bid == "request_exit_game":
                _click_sound()
                g = gui_ctx.get("game")
                if gui_state.get("screen") == "playing" and g and _winner_screen_ready(g):
                    gui_state["screen"] = "standby"
                    gui_state["selected_game_mode"] = None
                    gui_state["num_players"] = 0
                    gui_state["current_player_idx"] = 0
                    gui_ctx["game"] = None
                    all_hits.clear()
                    try:
                        send_mqtt_off()
                    except Exception:
                        pass
                else:
                    gui_state["screen"] = "exit_confirm"
                return
            if bid == "confirm_exit_yes":
                _click_sound()
                gui_state["screen"] = "standby"
                gui_state["selected_game_mode"] = None
                gui_state["num_players"] = 0
                gui_state["current_player_idx"] = 0
                gui_ctx["game"] = None
                all_hits.clear()
                try:
                    send_mqtt_off()
                except Exception:
                    pass
                return
            if bid == "confirm_exit_no":
                _click_sound()
                gui_state["screen"] = "playing"
                return
            if bid == "undo_turn":
                g = gui_ctx.get("game")
                if not g or not (g.get("undo_stack") or []):
                    return
                if gui_ctx.get("_dart_block_auto_advance"):
                    return
                turn_hits = g.get("turn_hits", [None, None, None])
                cur = int(gui_state.get("current_player_idx", 0))
                if _turn_can_submit(g, cur, turn_hits) or _killer_assign_needs_retry(
                    g, cur, turn_hits
                ):
                    return
                _click_sound()
                if _pop_undo(g, gui_state, gui_ctx):
                    gui_ctx["_dart_block_auto_advance"] = True
                    gui_ctx["_dart_block_auto_detect"] = False
                    gui_ctx["_dart_empty_frames"] = 0
                    _close_hit_editor()
                    _sync_board_refs_after_undo()
                    print("[dart] nazad — uredi ruku / prazni slotovi; iduci igrac pauziran", flush=True)
                    return
            if bid == "manual_advance_confirm":
                g = gui_ctx.get("game")
                if not g:
                    return
                if gui_ctx.get("_dart_block_auto_advance"):
                    _click_sound()
                    gui_ctx["_dart_manual_advance_confirm"] = False
                    _confirm_undo_advance_to_next()
                    return
                turn_hits = g.get("turn_hits", [None, None, None])
                cur_chk = int(gui_state.get("current_player_idx", 0))
                if not (
                    _turn_can_submit(g, cur_chk, turn_hits)
                    or _killer_assign_needs_retry(g, cur_chk, turn_hits)
                ):
                    return
                from kiosk_settings import get_settings as _gs_adv

                if bool(_gs_adv().manual_mode):
                    _click_sound()
                    gui_ctx["_dart_manual_advance_confirm"] = False
                    _force_advance_past_board_clear()
                    return
                submit_since = gui_ctx.get("_dart_can_submit_since")
                if (
                    submit_since is not None
                    and time.time() - float(submit_since) < FORCE_NEXT_BUTTON_DELAY_SEC
                ):
                    return
                _click_sound()
                if not gui_ctx.get("_dart_manual_advance_confirm"):
                    gui_ctx["_dart_manual_advance_confirm"] = True
                    print("[dart] rucno prebacivanje — potvrdi: ploca je cista?", flush=True)
                    return
                gui_ctx["_dart_manual_advance_confirm"] = False
                _force_advance_past_board_clear(update_empty_ref=True)
                return

        # Manual bull / ellipse seed on calibration camera preview
        if gui_state.get("screen") == "calibration" and (
            gui_ctx.get("_cal_click_bull_mode") or gui_ctx.get("_cal_click_ellipse_mode")
        ):
            for tile in gui_ctx.get("_cal_tiles") or []:
                vx1, vy1, vx2, vy2 = tile["video_rect"]
                if not (vx1 <= x_canvas < vx2 and vy1 <= y_canvas < vy2):
                    continue
                fw, fh = tile["frame_size"]
                dw, dh = tile.get("display_size") or (fw, fh)
                vw = max(1, vx2 - vx1)
                vh = max(1, vy2 - vy1)
                dx = (float(x_canvas - vx1) / float(vw)) * float(dw)
                dy = (float(y_canvas - vy1) / float(vh)) * float(dh)
                if int(dw) > 0 and int(dh) > 0 and (int(dw) != int(fw) or int(dh) != int(fh)):
                    fx = dx * (float(fw) / float(dw))
                    fy = dy * (float(fh) / float(dh))
                else:
                    fx, fy = dx, dy
                fx = max(0.0, min(float(fw - 1), fx))
                fy = max(0.0, min(float(fh - 1), fy))
                cam_idx = int(tile["cam_idx"])
                if gui_ctx.get("_cal_click_bull_mode"):
                    hints = dict(gui_ctx.get("bull_hints") or {})
                    hints[cam_idx] = (fx, fy)
                    gui_ctx["bull_hints"] = hints
                    print(
                        f"[dart] bull hint cam{cam_idx}: ({fx:.1f},{fy:.1f}) — ponovo KALIBRIRAJ",
                        flush=True,
                    )
                else:
                    raw = dict(gui_ctx.get("ellipse_hints") or {})
                    pts = list(raw.get(cam_idx) or raw.get(str(cam_idx)) or [])
                    pts = [(float(p[0]), float(p[1])) for p in pts if p is not None and len(p) >= 2]
                    near = max(18.0, min(float(fw), float(fh)) * 0.10)
                    moved = False
                    if pts:
                        dists = [((fx - px) ** 2 + (fy - py) ** 2) ** 0.5 for px, py in pts]
                        best = int(min(range(len(dists)), key=lambda i: dists[i]))
                        if dists[best] <= near or len(pts) >= 4:
                            pts[best] = (fx, fy)
                            moved = True
                    if not moved and len(pts) < 4:
                        pts.append((fx, fy))
                    raw[cam_idx] = pts[:4]
                    gui_ctx["ellipse_hints"] = raw
                    print(
                        f"[dart] ellipse hint cam{cam_idx}: n={len(pts)} "
                        f"last=({fx:.1f},{fy:.1f}) — ponovo KALIBRIRAJ",
                        flush=True,
                    )
                gui_ctx["_cal_last_tick"] = 0.0
                return

        # Hits / keypad area (only on playing)
        if gui_state.get("screen") != "playing":
            return
        pt = _map_click_to_view_in_canvas(x, y)
        if pt is None:
            return
        vx, vy = int(pt[0]), int(pt[1])
        g = gui_ctx.get("game")
        if not g:
            return
        if g.get("winner_player_idx") is not None:
            return

        turn_hits = g.get("turn_hits", [None, None, None])
        cur = int(gui_state.get("current_player_idx", 0))

        editing_slot = gui_ctx.get("editing_hit_slot")
        if editing_slot is None:
            for hb in gui_ctx.get("hit_slot_buttons", []):
                if not _hit_test(vx, vy, hb["rect"]):
                    continue
                try:
                    slot = int(str(hb["id"]).rsplit(":", 1)[-1])
                except ValueError:
                    return
                if 0 <= slot < 3:
                    gui_ctx["editing_hit_slot"] = slot
                    gui_ctx["input_multiplier"] = 1
                    gui_ctx["_dart_can_submit_since"] = time.time()
                    gui_ctx["_dart_manual_advance_confirm"] = False
                    _click_sound()
                return
            return

        for kb in gui_ctx.get("keypad_buttons", []):
            if not _hit_test(vx, vy, kb["rect"]):
                continue
            kid = kb["id"]
            if kid.startswith("mult:"):
                m = int(kid.split(":", 1)[1])
                cur_mult = int(gui_ctx.get("input_multiplier", 1))
                gui_ctx["input_multiplier"] = 1 if cur_mult == m else m
                _click_sound()
                return
            if kid == "clear_one":
                turn_hits = g.get("turn_hits", [None, None, None])
                if 0 <= int(editing_slot) < 3:
                    turn_hits[int(editing_slot)] = None
                else:
                    _clear_one_slot(turn_hits)
                g["turn_hits"] = turn_hits
                cur_clr = int(gui_state.get("current_player_idx", 0))
                _killer_resync_mid_turn(gui_ctx, g, cur_clr, turn_hits)
                # Snapshot upije strijelu u referencu — detekcija mora nastaviti.
                gui_ctx["_dart_block_auto_detect"] = False
                gui_ctx["_dart_score_locked"] = False
                gui_ctx["_dart_can_submit_since"] = None
                gui_ctx["_dart_manual_advance_confirm"] = False
                cam_mgr = gui_ctx.get("camera_mgr")
                dart_det = gui_ctx.get("dart_detector")
                board_cal = gui_ctx.get("board_calibrator")
                if (
                    cam_mgr is not None
                    and dart_det is not None
                    and board_cal is not None
                    and cam_mgr.active
                ):
                    dart_det.reset_motion_state()
                    dart_det.commit_board_snapshot(board_cal, cam_mgr.grab_frames())
                print("[dart] CLR — detekcija ponovno aktivna", flush=True)
                _close_hit_editor()
                _click_sound()
                return

            hit: Optional[DetectedHit] = None
            if kid == "hit:miss":
                hit = _make_miss_hit()
            elif kid == "bull:25":
                hit = _make_bull_hit(False)
            elif kid == "bull:50":
                hit = _make_bull_hit(True)
            elif kid.startswith("num:"):
                n = int(kid.split(":", 1)[1])
                hit = _make_number_hit(n, int(gui_ctx.get("input_multiplier", 1)))
            if hit is not None and 0 <= int(editing_slot) < 3:
                _set_turn_hit(int(editing_slot), hit)
                if int(gui_ctx.get("input_multiplier", 1)) in (2, 3):
                    gui_ctx["input_multiplier"] = 1
                _close_hit_editor()
                _click_sound()
            return

    mqtt_cmd_lock = None
    mqtt_pending_cmd = None
    mqtt_client = None
    if args.gui:
        import threading

        mqtt_cmd_lock = threading.Lock()

        def _on_mqtt_message(_mqttc, _userdata, msg):
            nonlocal mqtt_pending_cmd
            try:
                payload = msg.payload.decode().strip().lower()
            except Exception:
                payload = ""
            if payload not in ("on", "off"):
                return
            with mqtt_cmd_lock:
                mqtt_pending_cmd = payload

        def _on_mqtt_connect(mqttc, _userdata, _flags, rc, properties=None):
            try:
                if rc == 0:
                    mqttc.subscribe(TOPIC)
            except Exception:
                pass

        try:
            mqtt_client = mqtt.Client()
            mqtt_client.username_pw_set(USERNAME, PASSWORD)
            mqtt_client.tls_set(
                ca_certs=None,
                certfile=None,
                keyfile=None,
                cert_reqs=ssl.CERT_REQUIRED,
                tls_version=ssl.PROTOCOL_TLS_CLIENT,
            )
            mqtt_client.on_message = _on_mqtt_message
            mqtt_client.on_connect = _on_mqtt_connect
            mqtt_client.connect(BROKER_HOST, BROKER_PORT, 60)
            mqtt_client.loop_start()
            print("[MQTT] GUI subscriber started")
        except Exception as e:
            print("[MQTT] GUI subscriber failed:", e)

        try:
            from kiosk_settings import get_settings as _gs

            _skip_mqtt = _gs().start_mode == "always_on"
        except Exception:
            _skip_mqtt = False
        if _skip_mqtt:
            print("[MQTT] Startup skipped (always_on)")
        else:
            send_mqtt_on()
            print("[MQTT] Startup 'on' sent")
    else:
        try:
            from kiosk_settings import get_settings as _gs

            _skip_mqtt = _gs().start_mode == "always_on"
        except Exception:
            _skip_mqtt = False
        if _skip_mqtt:
            print("[MQTT] Startup skipped (always_on, no GUI)")
        else:
            send_mqtt_on()
            print("[MQTT] Startup 'on' sent (no GUI)")

    if args.gui:
        cv2.namedWindow("Pikado", cv2.WINDOW_NORMAL)
        if args.kiosk:
            try:
                cv2.setWindowProperty("Pikado", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            except Exception:
                pass
            try:
                user32 = ctypes.windll.user32
                gui_ctx["_display_w"] = int(user32.GetSystemMetrics(0))
                gui_ctx["_display_h"] = int(user32.GetSystemMetrics(1))
            except Exception:
                gui_ctx["_display_w"] = 1136
                gui_ctx["_display_h"] = 640
            try:
                cv2.resizeWindow("Pikado", int(gui_ctx["_display_w"]), int(gui_ctx["_display_h"]))
            except Exception:
                pass
        else:
            cv2.resizeWindow("Pikado", 1120, 640)
            gui_ctx["_display_w"] = 1120
            gui_ctx["_display_h"] = 640
        cv2.setMouseCallback("Pikado", _on_main_mouse, gui_ctx)

    try:
        while not shutdown_requested:
            if args.gui and mqtt_cmd_lock is not None:
                cmd = None
                with mqtt_cmd_lock:
                    if mqtt_pending_cmd is not None:
                        cmd = mqtt_pending_cmd
                        mqtt_pending_cmd = None
                if cmd == "on":
                    gui_state["screen"] = "select_game"
                    gui_state["selected_game_mode"] = None
                    gui_state["num_players"] = 0
                    gui_state["current_player_idx"] = 0
                    gui_ctx["game"] = None
                    all_hits.clear()
                elif cmd == "off":
                    gui_state["screen"] = "standby"
                    gui_state["selected_game_mode"] = None
                    gui_state["num_players"] = 0
                    gui_state["current_player_idx"] = 0
                    gui_ctx["game"] = None
                    all_hits.clear()

            # Winner auto-return: after WINNER_AUTO_EXIT_SEC on winner screen, go back to standby.
            g_auto = gui_ctx.get("game")
            if (
                gui_state.get("screen") == "playing"
                and g_auto
                and _winner_screen_ready(g_auto)
            ):
                t0 = g_auto.get("winner_shown_at")
                if t0 is None:
                    g_auto["winner_shown_at"] = time.time()
                elif time.time() - float(t0) >= WINNER_AUTO_EXIT_SEC:
                    gui_state["screen"] = "standby"
                    gui_state["selected_game_mode"] = None
                    gui_state["num_players"] = 0
                    gui_state["current_player_idx"] = 0
                    gui_ctx["game"] = None
                    all_hits.clear()
                    try:
                        send_mqtt_off()
                    except Exception:
                        pass

            if args.gui:
                screen = gui_state.get("screen", "standby")

                # Kamere: kalibracija + igra (auto-detekcija strijelica).
                cam_mgr_guard = gui_ctx.get("camera_mgr")
                need_cams = screen in ("calibration", "clear_board", "playing")
                if cam_mgr_guard is not None:
                    if need_cams and not cam_mgr_guard.active:
                        cam_mgr_guard.start()
                    elif not need_cams and cam_mgr_guard.active:
                        cam_mgr_guard.stop()
                        gui_ctx["_cal_canvas_disp"] = None
                        gui_ctx["_last_cal_frames"] = {}
                        dart_det = gui_ctx.get("dart_detector")
                        if dart_det is not None:
                            dart_det.reset_motion_state()
                        gui_ctx["_dart_status"] = ""

                if screen == "playing":
                    g_play = gui_ctx.get("game")
                    # Keep dart/clear-board loop while winner is pending board clear.
                    if g_play and (
                        g_play.get("winner_player_idx") is None
                        or _winner_pending_board_clear(g_play)
                    ):
                        dart_det = gui_ctx.get("dart_detector")
                        if dart_det is not None and cam_mgr_guard is not None and cam_mgr_guard.active:
                            if gui_ctx.get("_dart_turn_start_pending"):
                                if dart_det.has_references():
                                    _prepare_dart_turn_start(commit_snapshot=True)
                                    gui_ctx["_dart_turn_start_pending"] = False
                                    print("[dart] igra pokrenuta — cekam mirnu plocu", flush=True)
                            if not dart_det.has_references() and not gui_ctx.get("_dart_no_ref_warned"):
                                print(
                                    "[dart] Nema reference — pritisni DETEKTIRAJ u kalibraciji (prazna ploca)",
                                    flush=True,
                                )
                                gui_ctx["_dart_status"] = "CEKAJ REF"
                                gui_ctx["_dart_no_ref_warned"] = True
                            now_dart = time.perf_counter()
                            if now_dart - float(gui_ctx.get("_dart_last_tick", 0.0)) >= DART_DETECT_INTERVAL:
                                gui_ctx["_dart_last_tick"] = now_dart
                                turn_hits = g_play.get("turn_hits", [None, None, None])
                                cur_play = int(gui_state.get("current_player_idx", 0))
                                checkout_ready = _turn_is_early_finish(
                                    g_play, cur_play, turn_hits
                                )
                                bust_ready = _x01_turn_is_bust(g_play, cur_play, turn_hits)
                                score_paused = _dart_scoring_paused(
                                    g_play, cur_play, turn_hits
                                )
                                prev_locked = bool(gui_ctx.get("_dart_score_locked", False))
                                if score_paused and not prev_locked:
                                    _close_hit_editor()
                                    if bust_ready or checkout_ready:
                                        dart_det.reset_motion_state()
                                    if bust_ready:
                                        print("[dart] bust — detekcija pauzirana", flush=True)
                                    elif checkout_ready:
                                        print(
                                            "[dart] kraj ruke spreman — detekcija pauzirana",
                                            flush=True,
                                        )
                                gui_ctx["_dart_score_locked"] = score_paused
                                if gui_ctx.get("_dart_block_auto_advance"):
                                    # Undo: prazni slotovi trenutne ruke — ne scoreaj iduceg.
                                    _try_auto_detect_dart(undo_mode=False)
                                elif _can_auto_advance_on_empty_board(
                                    g_play, cur_play, turn_hits
                                ):
                                    _try_auto_empty_board_advance()
                                else:
                                    _try_auto_detect_dart()

                disp_w = int(gui_ctx.get("_display_w", 1120))
                disp_h = int(gui_ctx.get("_display_h", 640))
                screen_margin = max(18, int(min(disp_w, disp_h) * 0.04))
                avail_w = max(1, disp_w - 2 * screen_margin)
                avail_h = max(1, disp_h - 2 * screen_margin)

                if screen in ("standby", "select_game", "rules_x01", "rules_cricket", "rules_killer", "select_players", "clear_board", "exit_confirm"):
                    control_img = _build_control_panel(width=avail_w, height=avail_h)
                    canvas_disp = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
                    canvas_disp[:] = UI["bg"]
                    canvas_disp[
                        screen_margin : screen_margin + avail_h,
                        screen_margin : screen_margin + avail_w,
                    ] = control_img
                    gui_ctx["_layout_split"] = False
                    gui_ctx["_stack_control_h"] = disp_h
                    gui_ctx["_stack_view_w"] = 0
                    gui_ctx["_stack_view_h"] = 0
                    gui_ctx["_display_scale"] = 1.0
                    gui_ctx["_display_off_x"] = screen_margin
                    gui_ctx["_display_off_y"] = screen_margin
                    cv2.imshow("Pikado", canvas_disp)
                elif screen == "calibration":
                    now = time.perf_counter()
                    last_tick = float(gui_ctx.get("_cal_last_tick", 0.0))
                    should_capture = (now - last_tick) >= CALIBRATION_FRAME_INTERVAL

                    if should_capture:
                        gui_ctx["_cal_last_tick"] = now
                        cam_mgr = gui_ctx.get("camera_mgr")
                        frames = cam_mgr.grab_frames() if cam_mgr is not None else {}
                        gui_ctx["_last_cal_frames"] = frames
                        status = cam_mgr.status_text() if cam_mgr is not None else ""
                        board_cal = gui_ctx.get("board_calibrator")
                        cal_buttons: List[dict] = []
                        cal_tiles: List[dict] = []
                        click_bull = bool(gui_ctx.get("_cal_click_bull_mode"))
                        click_ell = bool(gui_ctx.get("_cal_click_ellipse_mode"))
                        banner = ""
                        if click_bull:
                            try:
                                from kiosk_settings import get_settings

                                banner = get_settings().t("click_bull_hint")
                            except Exception:
                                banner = "Klikni otprilike na bull"
                        elif click_ell:
                            try:
                                from kiosk_settings import get_settings

                                banner = get_settings().t("click_ellipse_hint")
                            except Exception:
                                banner = "Klikni do 4 tocke na vanjskom rubu"
                        cal_img = build_calibration_view(
                            avail_w,
                            avail_h,
                            frames,
                            status=status,
                            board_calibrator=board_cal,
                            buttons_out=cal_buttons,
                            bull_hints=dict(gui_ctx.get("bull_hints") or {}),
                            ellipse_hints=dict(gui_ctx.get("ellipse_hints") or {}),
                            tiles_out=cal_tiles,
                            click_bull_mode=click_bull,
                            click_bull_banner=banner,
                            click_ellipse_mode=click_ell,
                        )
                        gui_ctx["buttons"] = cal_buttons
                        gui_ctx["_cal_tiles"] = cal_tiles
                        canvas_disp = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
                        canvas_disp[:] = UI["bg"]
                        canvas_disp[
                            screen_margin : screen_margin + avail_h,
                            screen_margin : screen_margin + avail_w,
                        ] = cal_img
                        gui_ctx["_layout_split"] = False
                        gui_ctx["_stack_control_h"] = disp_h
                        gui_ctx["_stack_view_w"] = 0
                        gui_ctx["_stack_view_h"] = 0
                        gui_ctx["_display_scale"] = 1.0
                        gui_ctx["_display_off_x"] = screen_margin
                        gui_ctx["_display_off_y"] = screen_margin
                        gui_ctx["_cal_canvas_disp"] = canvas_disp
                        cv2.imshow("Pikado", canvas_disp)
                    else:
                        prev_canvas = gui_ctx.get("_cal_canvas_disp")
                        if prev_canvas is not None:
                            cv2.imshow("Pikado", prev_canvas)
                else:
                    g_layout = gui_ctx.get("game") or {}
                    keypad_size = 640
                    cp_w = max(360, int(avail_w * 0.44))
                    if gui_ctx.get("editing_hit_slot") is not None:
                        view_img = _build_keypad_view(size=keypad_size)
                    else:
                        view_img = _build_hits_view(size=keypad_size)
                    control_img = _build_control_panel(width=cp_w, height=view_img.shape[0])
                    vh, vw = view_img.shape[:2]
                    ch, cw = control_img.shape[:2]
                    win_w = vw + cw
                    win_h = max(vh, ch)
                    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                    canvas[:] = UI["game_bg"]
                    canvas[0:vh, 0:vw] = view_img
                    py = (win_h - ch) // 2
                    canvas[py : py + ch, vw : vw + cw] = control_img
                    # Shift panel button rects into combined canvas coords.
                    for btn in gui_ctx.get("buttons", []):
                        x1, y1, x2, y2 = btn["rect"]
                        btn["rect"] = (x1 + vw, y1 + py, x2 + vw, y2 + py)

                    gui_ctx["_layout_split"] = True
                    gui_ctx["_stack_control_h"] = 0
                    gui_ctx["_stack_view_w"] = int(vw)
                    gui_ctx["_stack_view_h"] = int(vh)

                    scale = min(avail_w / float(win_w), avail_h / float(win_h)) if (win_w > 0 and win_h > 0) else 1.0
                    new_w = int(win_w * scale)
                    new_h = int(win_h * scale)
                    x_off = screen_margin + max(0, (avail_w - new_w) // 2)
                    y_off = screen_margin + max(0, (avail_h - new_h) // 2)
                    gui_ctx["_display_scale"] = float(scale)
                    gui_ctx["_display_off_x"] = int(x_off)
                    gui_ctx["_display_off_y"] = int(y_off)

                    if (new_w, new_h) != (win_w, win_h):
                        resized = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    else:
                        resized = canvas
                    canvas_disp = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
                    canvas_disp[:] = UI["bg"]
                    canvas_disp[y_off : y_off + new_h, x_off : x_off + new_w] = resized
                    cv2.imshow("Pikado", canvas_disp)

                dart_det = gui_ctx.get("dart_detector")
                if dart_det is not None:
                    dart_det.flush_commit_after_ui(gui_ctx.get("board_calibrator"))

                # Tek nakon imshow: KALIBRIRAM... je na ekranu, pa nakon delay-a kalibriraj.
                pending_at = gui_ctx.get("_clear_board_pending_at")
                if (
                    gui_ctx.get("_clear_board_pending_run")
                    and gui_state.get("screen") == "clear_board"
                    and pending_at is not None
                    and (time.perf_counter() - float(pending_at)) >= 0.55
                ):
                    gui_ctx["_clear_board_pending_run"] = False
                    gui_ctx["_clear_board_pending_at"] = None
                    _confirm_clear_board_and_start()

                if screen == "calibration":
                    elapsed = time.perf_counter() - float(gui_ctx.get("_cal_last_tick", 0.0))
                    wait_ms = max(1, int((CALIBRATION_FRAME_INTERVAL - elapsed) * 1000))
                    key = cv2.waitKey(wait_ms) & 0xFF
                else:
                    key = cv2.waitKey(16) & 0xFF
                if key == ord("2"):
                    if gui_state.get("screen") == "calibration":
                        _exit_calibration()
                    else:
                        _enter_calibration()
                elif key == ord("3"):
                    send_mqtt_on()
                elif gui_state.get("screen") == "calibration" and key == 27:
                    _exit_calibration()
                elif gui_state.get("screen") == "calibration" and key == ord("b"):
                    _run_calibration_detect()
                elif gui_state.get("screen") == "calibration" and key == ord("s"):
                    board_cal = gui_ctx.get("board_calibrator")
                    if board_cal is not None:
                        board_cal.show_topdown = not board_cal.show_topdown
                        if board_cal.show_topdown:
                            board_cal.show_overlay = True
                        gui_ctx["_cal_last_tick"] = 0.0
                elif gui_state.get("screen") == "calibration" and key == ord("t"):
                    board_cal = gui_ctx.get("board_calibrator")
                    if board_cal is not None:
                        board_cal.show_topdown = not board_cal.show_topdown
                        if board_cal.show_topdown:
                            board_cal.show_overlay = True
                        gui_ctx["_cal_last_tick"] = 0.0
                elif gui_state.get("screen") == "calibration" and key == ord("o"):
                    on = not bool(gui_ctx.get("_cal_click_ellipse_mode"))
                    gui_ctx["_cal_click_ellipse_mode"] = on
                    if on:
                        gui_ctx["_cal_click_bull_mode"] = False
                    gui_ctx["_cal_last_tick"] = 0.0
                elif gui_state.get("screen") == "playing" and key == ord("d"):
                    _try_auto_detect_dart(force_log=True)
                elif args.kiosk:
                    if key == ord("q"):
                        shutdown_requested = True
                elif key in (27, ord("q")):
                    shutdown_requested = True
            else:
                time.sleep(0.03)

    finally:
        try:
            if mqtt_client is not None:
                mqtt_client.loop_stop()
                mqtt_client.disconnect()
        except Exception:
            pass
        if args.gui:
            cam_mgr = gui_ctx.get("camera_mgr")
            if cam_mgr is not None:
                cam_mgr.stop()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        send_mqtt_off()


if __name__ == "__main__":
    main()

