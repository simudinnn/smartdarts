"""League QR payload for the winner screen.

Plain-text format (pipe-delimited):
    DART:v1|<device>|<GAME>|<epoch_seconds>|<token>
"""

from __future__ import annotations

import secrets
import time
from typing import Any, Dict, List, Optional

LEAGUE_GAME_CODES = frozenset({"X01", "AROUND", "CRICKET", "KILLER", "HALVE_IT"})

_MODE_TO_GAME = {
    "301": "X01",
    "501": "X01",
    "x01": "X01",
    "around": "AROUND",
    "around_the_world": "AROUND",
    "cricket": "CRICKET",
    "killer": "KILLER",
    "halve": "HALVE_IT",
    "halve_it": "HALVE_IT",
}


def sanitize_device_id(raw: Any, *, fallback: str = "pikado-1") -> str:
    text = str(raw or "").replace("|", "").strip()
    return text or fallback


def game_code_from_mode(mode: Any) -> str:
    key = str(mode or "").strip().lower().replace(" ", "_")
    code = _MODE_TO_GAME.get(key)
    if code:
        return code
    upper = key.upper()
    if upper in LEAGUE_GAME_CODES:
        return upper
    return "X01"


def build_payload(device_id: str, game_code: str, epoch_seconds: int, token: str) -> str:
    device = sanitize_device_id(device_id)
    game = str(game_code or "X01").strip().upper().replace(" ", "_")
    if game not in LEAGUE_GAME_CODES:
        game = "X01"
    tok = str(token or "").replace("|", "").strip()
    if len(tok) < 4:
        tok = secrets.token_hex(8)
    return f"DART:v1|{device}|{game}|{int(epoch_seconds)}|{tok}"


def game_player_count(game: Optional[Dict[str, Any]]) -> int:
    if not isinstance(game, dict):
        return 0
    try:
        n = int(game.get("num_players") or 0)
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        n = len(game.get("players") or [])
    return n


def ensure_game_qr(
    game: Optional[Dict[str, Any]],
    device_id: str,
    *,
    winner_ready: bool,
) -> Optional[str]:
    """Mint the league payload once per finished game. Returns payload when the
    winner screen is ready; otherwise None. Token stays stable across UI refreshes.
    Solo games (1 player) never get a QR.
    """
    if not isinstance(game, dict):
        return None
    if game.get("winner_player_idx") is None or game_player_count(game) < 2:
        game.pop("league_qr", None)
        return None

    existing = game.get("league_qr")
    payload = None
    if isinstance(existing, dict) and existing.get("payload"):
        payload = str(existing["payload"])
    elif isinstance(existing, str) and existing.startswith("DART:"):
        payload = existing

    if payload:
        return payload if winner_ready else None
    if not winner_ready:
        return None

    epoch_s = int(time.time())
    if game.get("winner_at") is None:
        game["winner_at"] = float(epoch_s)
    token = secrets.token_hex(8)
    game_code = game_code_from_mode(game.get("mode"))
    payload = build_payload(device_id, game_code, epoch_s, token)
    game["league_qr"] = {
        "payload": payload,
        "token": token,
        "epoch": epoch_s,
        "device": sanitize_device_id(device_id),
        "game": game_code,
    }
    return payload


def qr_matrix(payload: str) -> Optional[List[List[bool]]]:
    """QR boolean matrix with a 2-module quiet zone. No extra pip packages."""
    text = str(payload or "").strip()
    if not text:
        return None
    try:
        return _encode_qr_m(text, border=2)
    except Exception as e:
        print(f"[league_qr] encode fail: {e}", flush=True)
        return None


# --- Built-in QR (byte mode, ECC-M, versions 1–10). No third-party deps. ---

_EXP = [0] * 512
_LOG = [0] * 256
_x = 1
for _i in range(255):
    _EXP[_i] = _x
    _LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D
for _i in range(255, 512):
    _EXP[_i] = _EXP[_i - 255]

# version -> (ec_per_block, [(n_blocks, data_per_block), ...])  ECC-M
_ECC_M = {
    1: (10, ((1, 16),)),
    2: (16, ((1, 28),)),
    3: (26, ((1, 44),)),
    4: (18, ((2, 32),)),
    5: (24, ((2, 43),)),
    6: (16, ((4, 27),)),
    7: (18, ((4, 31),)),
    8: (22, ((2, 38), (2, 39))),
    9: (22, ((3, 36), (2, 37))),
    10: (26, ((4, 43), (1, 44))),
}

_ALIGN = {
    2: (18,),
    3: (22,),
    4: (26,),
    5: (30,),
    6: (34,),
    7: (6, 22, 38),
    8: (6, 24, 42),
    9: (6, 26, 46),
    10: (6, 28, 50),
}

_VERSION_BITS = {
    7: 0x07C94,
    8: 0x085BC,
    9: 0x09A64,
    10: 0x0A4D4,
}


def _gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _EXP[_LOG[a] + _LOG[b]]


def _rs_remainder(data: List[int], degree: int) -> List[int]:
    gen = [1]
    root = 1
    for _ in range(degree):
        gen.append(0)
        for j in range(len(gen) - 1, 0, -1):
            gen[j] ^= _gf_mul(gen[j - 1], root)
        root = _gf_mul(root, 2)
    rem = [0] * degree
    for b in data:
        factor = b ^ rem[0]
        rem = rem[1:] + [0]
        if factor:
            for i in range(degree):
                rem[i] ^= _gf_mul(gen[i + 1], factor)
    return rem


def _append_bits(bits: List[int], value: int, n: int) -> None:
    for i in range(n - 1, -1, -1):
        bits.append((value >> i) & 1)


def _data_codewords(data: bytes, version: int) -> List[int]:
    ec_n, groups = _ECC_M[version]
    data_cw = sum(nb * dc for nb, dc in groups)
    cci = 8 if version <= 9 else 16
    bits: List[int] = []
    _append_bits(bits, 0b0100, 4)
    _append_bits(bits, len(data), cci)
    for b in data:
        _append_bits(bits, b, 8)
    remain = data_cw * 8 - len(bits)
    _append_bits(bits, 0, min(4, remain))
    while len(bits) % 8:
        bits.append(0)
    pad = (0xEC, 0x11)
    pi = 0
    while len(bits) < data_cw * 8:
        _append_bits(bits, pad[pi], 8)
        pi ^= 1
    codewords = []
    for i in range(0, data_cw * 8, 8):
        v = 0
        for b in bits[i : i + 8]:
            v = (v << 1) | b
        codewords.append(v)
    blocks: List[List[int]] = []
    offset = 0
    for nb, dc in groups:
        for _ in range(nb):
            block = codewords[offset : offset + dc]
            offset += dc
            blocks.append(block + _rs_remainder(block, ec_n))
    interleaved: List[int] = []
    max_len = max(len(b) for b in blocks)
    for i in range(max_len):
        for b in blocks:
            if i < len(b):
                interleaved.append(b[i])
    out_bits: List[int] = []
    for cw in interleaved:
        _append_bits(out_bits, cw, 8)
    return out_bits


def _set(mod: List[List[Optional[bool]]], res: List[List[bool]], x: int, y: int, val: bool) -> None:
    n = len(mod)
    if 0 <= x < n and 0 <= y < n:
        mod[y][x] = val
        res[y][x] = True


def _add_finder(mod: List[List[Optional[bool]]], res: List[List[bool]], ox: int, oy: int) -> None:
    n = len(mod)
    for dy in range(-1, 8):
        for dx in range(-1, 8):
            x, y = ox + dx, oy + dy
            if not (0 <= x < n and 0 <= y < n):
                continue
            if 0 <= dx <= 6 and 0 <= dy <= 6:
                dark = dx in (0, 6) or dy in (0, 6) or (2 <= dx <= 4 and 2 <= dy <= 4)
            else:
                dark = False
            _set(mod, res, x, y, dark)


def _add_align(mod: List[List[Optional[bool]]], res: List[List[bool]], cx: int, cy: int) -> None:
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            dark = dx in (-2, 2) or dy in (-2, 2) or (dx == 0 and dy == 0)
            _set(mod, res, cx + dx, cy + dy, dark)


def _add_function(mod: List[List[Optional[bool]]], res: List[List[bool]], version: int) -> None:
    n = len(mod)
    _add_finder(mod, res, 0, 0)
    _add_finder(mod, res, n - 7, 0)
    _add_finder(mod, res, 0, n - 7)
    for i in range(8, n - 8):
        _set(mod, res, i, 6, i % 2 == 0)
        _set(mod, res, 6, i, i % 2 == 0)
    _set(mod, res, 8, n - 8, True)
    pos = _ALIGN.get(version, ())
    for r in pos:
        for c in pos:
            if (r <= 8 and c <= 8) or (r <= 8 and c >= n - 9) or (r >= n - 9 and c <= 8):
                continue
            _add_align(mod, res, c, r)
    for i in range(9):
        res[i][8] = True
        res[8][i] = True
    for i in range(8):
        res[8][n - 1 - i] = True
        res[n - 1 - i][8] = True
    if version >= 7:
        for i in range(18):
            res[n - 11 + i % 3][i // 3] = True
            res[i // 3][n - 11 + i % 3] = True


def _format_bits(mask: int) -> int:
    data = mask  # ECC-M = 00, so 5-bit field is just the mask
    bits = data << 10
    for i in range(4, -1, -1):
        if (bits >> (i + 10)) & 1:
            bits ^= 0x537 << i
    return ((data << 10) | (bits & 0x3FF)) ^ 0x5412


def _place_format(mod: List[List[Optional[bool]]], mask: int) -> None:
    n = len(mod)
    bits = _format_bits(mask)
    coords_a = [
        (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 7), (8, 8),
        (7, 8), (5, 8), (4, 8), (3, 8), (2, 8), (1, 8), (0, 8),
    ]
    coords_b = [(n - 1 - i, 8) for i in range(8)] + [(8, n - 7 + i) for i in range(7)]
    for i, (x, y) in enumerate(coords_a):
        mod[y][x] = bool((bits >> i) & 1)
    for i, (x, y) in enumerate(coords_b):
        mod[y][x] = bool((bits >> i) & 1)
    mod[n - 8][8] = True


def _place_version(mod: List[List[Optional[bool]]], version: int) -> None:
    if version < 7:
        return
    n = len(mod)
    bits = _VERSION_BITS[version]
    for i in range(18):
        bit = bool((bits >> i) & 1)
        a, b = i // 3, n - 11 + i % 3
        mod[b][a] = bit
        mod[a][b] = bit


def _mask_bit(mask: int, x: int, y: int) -> bool:
    if mask == 0:
        return (x + y) % 2 == 0
    if mask == 1:
        return y % 2 == 0
    if mask == 2:
        return x % 3 == 0
    if mask == 3:
        return (x + y) % 3 == 0
    if mask == 4:
        return (y // 2 + x // 3) % 2 == 0
    if mask == 5:
        return (x * y) % 2 + (x * y) % 3 == 0
    if mask == 6:
        return ((x * y) % 2 + (x * y) % 3) % 2 == 0
    return ((x + y) % 2 + (x * y) % 3) % 2 == 0


def _place_data(
    mod: List[List[Optional[bool]]],
    res: List[List[bool]],
    data_bits: List[int],
    mask: int,
) -> None:
    n = len(mod)
    idx = 0
    upward = True
    x = n - 1
    while x > 0:
        if x == 6:
            x -= 1
        ys = range(n - 1, -1, -1) if upward else range(n)
        for y in ys:
            for dx in (0, -1):
                xx = x + dx
                if res[y][xx]:
                    continue
                bit = data_bits[idx] if idx < len(data_bits) else 0
                idx += 1
                if _mask_bit(mask, xx, y):
                    bit ^= 1
                mod[y][xx] = bool(bit)
        upward = not upward
        x -= 2


def _penalty(mod: List[List[bool]]) -> int:
    n = len(mod)
    score = 0
    for y in range(n):
        run = 1
        for x in range(1, n):
            if mod[y][x] == mod[y][x - 1]:
                run += 1
            else:
                if run >= 5:
                    score += run - 2
                run = 1
        if run >= 5:
            score += run - 2
    for x in range(n):
        run = 1
        for y in range(1, n):
            if mod[y][x] == mod[y - 1][x]:
                run += 1
            else:
                if run >= 5:
                    score += run - 2
                run = 1
        if run >= 5:
            score += run - 2
    for y in range(n - 1):
        for x in range(n - 1):
            v = mod[y][x]
            if v == mod[y][x + 1] == mod[y + 1][x] == mod[y + 1][x + 1]:
                score += 3
    pat = (True, False, True, True, True, False, True)
    for y in range(n):
        row = mod[y]
        for x in range(n - 6):
            if tuple(row[x : x + 7]) != pat:
                continue
            left = x >= 4 and not any(row[x - 4 : x])
            right = x + 10 <= n and not any(row[x + 7 : x + 11])
            if left or right:
                score += 40
    for x in range(n):
        col = [mod[y][x] for y in range(n)]
        for y in range(n - 6):
            if tuple(col[y : y + 7]) != pat:
                continue
            up = y >= 4 and not any(col[y - 4 : y])
            down = y + 10 <= n and not any(col[y + 7 : y + 11])
            if up or down:
                score += 40
    dark = sum(1 for y in range(n) for x in range(n) if mod[y][x])
    k = abs(dark * 20 - n * n * 10) // (n * n)
    score += k * 10
    return score


def _encode_qr_m(text: str, border: int = 2) -> List[List[bool]]:
    data = text.encode("utf-8")
    version = None
    for v in range(1, 11):
        cci = 8 if v <= 9 else 16
        data_cw = sum(nb * dc for nb, dc in _ECC_M[v][1])
        if (data_cw * 8 - 4 - cci) // 8 >= len(data):
            version = v
            break
    if version is None:
        raise ValueError("payload too long for QR versions 1–10")
    n = 21 + 4 * (version - 1)
    bits = _data_codewords(data, version)
    best: Optional[List[List[bool]]] = None
    best_score = None
    for mask in range(8):
        mod: List[List[Optional[bool]]] = [[None] * n for _ in range(n)]
        res = [[False] * n for _ in range(n)]
        _add_function(mod, res, version)
        _place_data(mod, res, bits, mask)
        _place_format(mod, mask)
        _place_version(mod, version)
        grid = [[bool(mod[y][x]) for x in range(n)] for y in range(n)]
        score = _penalty(grid)
        if best_score is None or score < best_score:
            best_score = score
            best = grid
    assert best is not None
    dim = n + 2 * border
    out = [[False] * dim for _ in range(dim)]
    for y in range(n):
        for x in range(n):
            if best[y][x]:
                out[y + border][x + border] = True
    return out
