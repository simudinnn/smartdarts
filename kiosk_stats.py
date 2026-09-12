"""Trajni izvještaj kioska: auto/ručni hitovi, ispravke, igrane igre."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

STATS_FILENAME = "kiosk_stats.json"

_GAME_LABELS = {
    "301": "301",
    "501": "501",
    "cricket": "Cricket",
    "killer": "Killer",
    "around": "Around the Clock",
    "halve": "Halve It",
    "halve_it": "Halve It",
}


def stats_path() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, STATS_FILENAME)


def _empty() -> Dict[str, Any]:
    return {
        "auto_hits": 0,
        "manual_hits": 0,
        "corrected_hits": 0,
        "corrected_numbers": {},
        "games": {},
    }


def load_stats() -> Dict[str, Any]:
    path = stats_path()
    data = _empty()
    if not os.path.isfile(path):
        return data
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f) or {}
    except Exception:
        return data
    if not isinstance(raw, dict):
        return data
    data["auto_hits"] = max(0, int(raw.get("auto_hits", 0) or 0))
    data["manual_hits"] = max(0, int(raw.get("manual_hits", 0) or 0))
    data["corrected_hits"] = max(0, int(raw.get("corrected_hits", 0) or 0))
    nums = raw.get("corrected_numbers") or {}
    if isinstance(nums, dict):
        data["corrected_numbers"] = {
            str(k): max(0, int(v or 0)) for k, v in nums.items() if str(k)
        }
    games = raw.get("games") or {}
    if isinstance(games, dict):
        data["games"] = {
            str(k): max(0, int(v or 0)) for k, v in games.items() if str(k)
        }
    return data


def _save(data: Dict[str, Any]) -> None:
    try:
        with open(stats_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def record_auto_hit() -> None:
    data = load_stats()
    data["auto_hits"] = int(data.get("auto_hits", 0)) + 1
    _save(data)


def record_manual_hit() -> None:
    data = load_stats()
    data["manual_hits"] = int(data.get("manual_hits", 0)) + 1
    _save(data)


def record_correction(auto_tag: str) -> None:
    tag = str(auto_tag or "").strip() or "?"
    data = load_stats()
    data["corrected_hits"] = int(data.get("corrected_hits", 0)) + 1
    nums = data.setdefault("corrected_numbers", {})
    nums[tag] = int(nums.get(tag, 0)) + 1
    _save(data)


def record_game(mode: str) -> None:
    key = str(mode or "").strip().lower() or "unknown"
    if key == "halve_it":
        key = "halve"
    data = load_stats()
    games = data.setdefault("games", {})
    games[key] = int(games.get(key, 0)) + 1
    _save(data)


def game_label(mode: str) -> str:
    key = str(mode or "").strip().lower()
    if key == "halve_it":
        key = "halve"
    return _GAME_LABELS.get(key, (mode or "?").upper())


def top_corrected(limit: int = 5) -> List[Tuple[str, int]]:
    nums = load_stats().get("corrected_numbers") or {}
    items = [(str(k), int(v)) for k, v in nums.items() if int(v) > 0]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return items[: max(1, int(limit))]


def top_games(limit: int = 6) -> List[Tuple[str, int]]:
    games = load_stats().get("games") or {}
    items = [(str(k), int(v)) for k, v in games.items() if int(v) > 0]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return items[: max(1, int(limit))]
