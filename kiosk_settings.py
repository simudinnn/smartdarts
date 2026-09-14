"""Kiosk postavke (jezik, tema, način pokretanja) — trajno u JSON."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

SETTINGS_FILENAME = "kiosk_settings.json"

LANG_OPTIONS: List[Tuple[str, str, str]] = [
    ("hr", "Hrvatski", "🇭🇷"),
    ("en", "English", "🇬🇧"),
    ("de", "Deutsch", "🇩🇪"),
    ("fr", "Français", "🇫🇷"),
    ("it", "Italiano", "🇮🇹"),
    ("es", "Español", "🇪🇸"),
]

# Svijetli izbor; UI pozadina = gradijent crna → ova boja.
BG_PRESETS: List[Tuple[str, str]] = [
    ("#e02020", "bg_red"),
    ("#2060e0", "bg_blue"),
    ("#20c040", "bg_green"),
    ("#e0c020", "bg_yellow"),
    ("#a0a0a8", "bg_gray"),
    ("#a040e0", "bg_purple"),
    ("#f2f2f2", "bg_white"),
    ("#000000", "bg_black"),
]

BTN_PRESETS: List[Tuple[str, str]] = [
    ("#ffe000", "btn_yellow"),
    ("#36d700", "btn_green"),
    ("#3aa0ff", "btn_blue"),
    ("#ff8c00", "btn_orange"),
    ("#f2f2f2", "btn_white"),
    ("#a0a0a8", "btn_gray"),
    ("#000000", "btn_black"),
    ("#00e5c0", "btn_cyan"),
    ("#ff4d6d", "btn_pink"),
]

START_MODES: List[Tuple[str, str]] = [
    ("mqtt_qr", "start_mqtt_qr"),
    ("button", "start_button"),
    ("always_on", "start_always"),
]

# League QR <device> field — one kiosk per Pi.
DEVICE_IDS: List[str] = ["pikado-1", "pikado-2", "pikado-3"]

# Tip (x,y) source after multicam fuse. FitLine = existing line intersection.
TIP_DETECTION_MODES: List[Tuple[str, str]] = [
    ("fitline", "tip_fitline"),
    ("ml_onnx", "tip_ml_onnx"),
]

# Downscale factor for FitLine motion blobs (string keys for SettingsChoiceButton).
FITLINE_SCALE_OPTIONS: List[Tuple[str, str]] = [
    ("0.75", "0.75"),
    ("0.5", "0.5"),
    ("0.375", "0.375"),
    ("0.25", "0.25"),
]
FITLINE_SCALE_DEFAULT = 0.5
_FITLINE_SCALE_ALLOWED = {0.75, 0.5, 0.375, 0.25}

ML_TIP_DATASET_OPTIONS: List[Tuple[str, str]] = [
    ("off", "ml_tip_dataset_off"),
    ("on", "ml_tip_dataset_on"),
]

AUTO_CALIBRATE_OPTIONS: List[Tuple[str, str]] = [
    ("on", "auto_calibrate_on"),
    ("off", "auto_calibrate_off"),
]

POWER_OPTIONS: List[Tuple[str, str]] = [
    ("", "power_choose"),
    ("shutdown", "power_off"),
    ("reboot", "power_reboot"),
    ("exit", "power_exit"),
]

LEAGUE_QR_OPTIONS: List[Tuple[str, str]] = [
    ("on", "league_qr_on"),
    ("off", "league_qr_off"),
]

_I18N: Dict[str, Dict[str, str]] = {
    "hr": {
        "settings": "POSTAVKE",
        "language": "Jezik",
        "bg_color": "Boja pozadine",
        "btn_color": "Boja gumba",
        "start_mode": "Pokretanje igre",
        "start_mqtt_qr": "MQTT / QR kod",
        "start_button": "Gumb POKRENI",
        "start_always": "Stalno upaljeno",
        "device": "Uređaj",
        "league_qr": "Ligaški QR",
        "league_qr_on": "Uključeno",
        "league_qr_off": "Isključeno",
        "tip_detection": "Detekcija tipa",
        "tip_fitline": "FitLine",
        "tip_ml_onnx": "ML (ONNX)",
        "fitline_scale": "FitLine skala",
        "ml_tip_dataset": "ML tip dataset",
        "ml_tip_dataset_off": "Isključeno",
        "ml_tip_dataset_on": "Spremaj hitove",
        "manual_mode": "Ručni unos",
        "manual_mode_off": "Isključeno",
        "manual_mode_on": "Uključeno",
        "auto_calibrate": "Auto kalibracija",
        "auto_calibrate_on": "Uključeno",
        "auto_calibrate_off": "Isključeno",
        "saving_ref": "SPREMAM REF...",
        "ref_not_saved": "Reference nije spremljena — pokušaj ponovo",
        "no_saved_cal": "Nema kalibracije — prvo KALIBRACIJA u postavkama",
        "play_mode_auto": "AUTO",
        "play_mode_manual": "MANUAL",
        "switch_player": "PREBACI IGRAČA",
        "calibration": "KALIBRACIJA",
        "calibrate": "KALIBRIRAJ",
        "calibrating": "KALIBRIRAM...",
        "calibrating_hint": "Tražim ploču na kamerama",
        "starting_game": "POKREĆEM IGRU",
        "starting_game_hint": "Kalibracija i priprema detekcije",
        "please_wait": "Pričekajte",
        "select_game_sub": "Odaberi način igre",
        "cal_incomplete": "Kalibracija nepotpuna — pokušaj ponovo",
        "ingame_cal_title": "PONOVI KALIBRACIJU?",
        "ingame_cal_s1": "Program loše detektira hitove?",
        "ingame_cal_s2": "Uklonite strelice i kalibrirajte ploču ispočetka za precizniju detekciju.",
        "ingame_cal_done": "GOTOVO",
        "ingame_cal_retry": "PONOVNO KALIBRIRAJ",
        "click_bull": "BULL",
        "click_bull_hint": "Klikni otprilike na bull",
        "click_ellipse": "ELIPSA",
        "click_ellipse_hint": "Klikni do 4 točke na vanjskom rubu ploče",
        "overlay": "ELIPSA",
        "enter_name": "IME IGRAČA",
        "name_skip": "PRESKOK",
        "name_remove": "OBRIŠI",
        "back": "NAZAD",
        "standby_qr": "Skeniraj QR za početak igre!",
        "standby_button": "Pritisni POKRENI za početak",
        "start_game": "POKRENI",
        "select_game": "ODABERI IGRU",
        "select_players": "BROJ IGRAČA",
        "rules": "PRAVILA",
        "continue": "NASTAVI",
        "tutorial": "UPUTE",
        "tutorial_options": "OPCIJE",
        "tutorial_basics": "KAKO SE IGRA",
        "tutorial_ex_double_in": "DOUBLE IN\nprvo double",
        "tutorial_ex_double_out": "DOUBLE OUT\nzavrši doubleom",
        "tutorial_ex_cut_scorer": "VIŠAK MARKI",
        "tutorial_ex_cut_rivals": "BODUJE RIVALE",
        "tutorial_ex_killer_lives": "Životi: tko ostane zadnji — pobjeđuje",
        "tutorial_ex_around_seg": "S / D / T / ANY = koja zona broji",
        "clear_board_title": "OČISTITE PLOČU",
        "clear_board_s1": "Uklonite sve strelice s ploče",
        "clear_board_s2": "prije početka igre",
        "board_empty": "PLOČA JE PRAZNA",
        "exit_confirm": "IZAĐI IZ IGRE?",
        "yes": "DA",
        "no": "NE",
        "exit_game": "IZAĐI IZ IGRE",
        "idle_title": "IGRA JE NEAKTIVNA",
        "idle_s1": "Igra je idle već 5 minuta i uskoro će se ugasiti.",
        "idle_s2": "Igra će se ugasiti.",
        "idle_quit": "UGASI",
        "idle_continue": "NASTAVI",
        "game_mode": "GAME MODE",
        "double_in": "DOUBLE IN",
        "double_out": "DOUBLE OUT",
        "no_limit": "BEZ LIMITA",
        "rounds_20": "20 RUNDI",
        "cricket_standard": "STANDARD",
        "cricket_no_score": "NO SCORE",
        "cricket_cut": "CUT THROAT",
        "killer_lives": "BROJ ŽIVOTA",
        "killer_assign": "DODJELA BROJEVA",
        "killer_assign_random": "RANDOM",
        "killer_assign_throw": "GAĐANJEM",
        "killer_activation": "AKTIVACIJA",
        "killer_act_single": "SINGLE",
        "killer_act_double": "DOUBLE",
        "killer_act_triple": "TRIPLE",
        "killer_dead": "MRTAV",
        "killer_bull_prompt": "POGODILI STE BULL, IZABERITE KOME ĆETE UZETI ŽIVOT",
        "killer_bull_false": "NISAM POGODIO BULL",
        "killer_one_player_impossible": "NEMOGUĆE\nS 1 IGRAČEM",
        "halve_rounds": "BROJ RUNDI",
        "halve_standard": "STANDARD",
        "halve_extended": "PRODUŽENO",
        "halve_any_double": "BILO KOJI DOUBLE",
        "halve_any_triple": "BILO KOJI TRIPLE",
        "around_target_mult": "SEGMENTI",
        "around_mult_single": "SINGLE",
        "around_mult_double": "DOUBLE",
        "around_mult_triple": "TRIPLE",
        "around_mult_any": "ANY",
        "around_number_order": "REDOSLIJED BROJEVA",
        "around_order_standard": "STANDARD",
        "around_order_random": "RANDOM",
        "around_order_reverse": "REVERSE",
        "around_backstep": "KAZNA UNAZAD",
        "around_back": "NATRAG",
        "around_backstep_off": "ISKLJUČENO",
        "around_backstep_1": "NATRAG 1",
        "around_backstep_2": "NATRAG 2",
        "board_clear_end": "OČISTITE PLOČU ZA KRAJ",
        "board_clear": "OČISTI PLOČU",
        "board_clean_q": "PLOČA JE ČISTA?",
        "not_advanced_q": "NIJE PREBACILO?",
        "next_prefix": "Sljedeći:",
        "round": "Runda",
        "bust": "BUST",
        "tutorial_x01_body": (
            "• Krećete s {start} bodova.\n"
            "• 3 strelice po potezu — oduzimajte od zbira.\n"
            "• Prvi na točno 0 pobjeđuje.\n"
            "• Ispod 0 = BUST (potez se briše)."
        ),
        "tutorial_x01_opts": (
            "• DOUBLE IN OFF: bodovanje odmah.\n"
            "• DOUBLE IN ON: prvo double (ili B50).\n"
            "• DOUBLE OUT OFF: bilo koji pogodak na 0.\n"
            "• DOUBLE OUT ON: završetak doubleom; ostatak 1 = BUST."
        ),
        "tutorial_cricket_body": (
            "• Zatvorite 15–20 i bull (3 marke po broju).\n"
            "• Single=1, double=2, triple=3 marke.\n"
            "• Zatvorite sve i ispunite uvjet pobjede moda."
        ),
        "tutorial_cricket_opts": (
            "• BEZ LIMITA / 20 RUNDI — trajanje igre.\n"
            "• STANDARD: višak marki = bodovi vama.\n"
            "• NO SCORE: samo marke, bez bodova.\n"
            "• CUT THROAT: višak daje bodove rivalima; manje = bolje."
        ),
        "tutorial_around_body": (
            "• Gađajte brojeve redom (pa bull, ovisno o opciji).\n"
            "• Pogodite traženi broj + segment da napredujete.\n"
            "• Tko prvi prođe cijeli krug — pobjednik."
        ),
        "tutorial_around_opts": (
            "• SEGMENTI: SINGLE / DOUBLE / TRIPLE / ANY.\n"
            "• REDOSLIJED: STANDARD 1→20→bull, REVERSE, ili RANDOM.\n"
            "• KAZNA UNAZAD: OFF / NATRAG 1 / NATRAG 2 ako promašite cijeli potez."
        ),
        "tutorial_killer_body": (
            "• Svaki ima svoj broj i živote.\n"
            "• Postanite Killer svojim brojem + multiplikatorom aktivacije.\n"
            "• AKTIVACIJA (Single/Double/Triple): točno taj segment mora pogoditi "
            "da postanete Killer i da rivalima skidate živote.\n"
            "• Kao Killer gađajte rivale da im uzmete život.\n"
            "• Pogodak vlastitog broja kao Killer = gubite život.\n"
            "• Zadnji s životima pobjeđuje."
        ),
        "tutorial_killer_opts": (
            "• ŽIVOTI: 3 / 5 / 7 / 10.\n"
            "• RANDOM ili GAĐANJEM — dodjela brojeva.\n"
            "• AKTIVACIJA: SINGLE / DOUBLE / TRIPLE — obavezni multiplikator "
            "za ulazak u Killer i za oduzimanje života."
        ),
        "tutorial_halve_body": (
            "• Počinjete s 0 bodova; svaka runda ima jedan cilj "
            "(npr. 20, BILO KOJI DOUBLE, BULL).\n"
            "• Zbrojite pogotke na cilj u 3 strelice "
            "(S/D/T broja; na DOUBLE/TRIPLE rundi — bilo koji double/triple).\n"
            "• Ako u rundi ne pogodite ništa valjano — bodovi se prepolove (zaokruženo dolje).\n"
            "• Nakon svih rundi najviši zbroj pobjeđuje."
        ),
        "tutorial_halve_opts": (
            "• STANDARD: 20→15 i BULL (7 rundi).\n"
            "• PRODUŽENO: 20→15, BILO KOJI DOUBLE, 14→7, "
            "BILO KOJI TRIPLE, 6→1, BULL."
        ),
        "bg_red": "Crvena",
        "bg_blue": "Plava",
        "bg_green": "Zelena",
        "bg_yellow": "Žuta",
        "bg_gray": "Siva",
        "bg_purple": "Ljubičasta",
        "bg_white": "Bijela",
        "bg_black": "Crna",
        "btn_yellow": "Žuta",
        "btn_green": "Zelena",
        "btn_blue": "Plava",
        "btn_orange": "Narančasta",
        "btn_white": "Bijela",
        "btn_gray": "Siva",
        "btn_black": "Crna",
        "btn_cyan": "Cyan",
        "btn_pink": "Roza",
        "winner_title": "POBJEDNIK!",
        "league_qr_hint": "Skeniraj za ligu",
        "change_password": "PROMIJENI LOZINKU",
        "change_password_short": "Promijeni",
        "password": "Lozinka",
        "application": "Aplikacija",
        "pin_enter": "Unesi lozinku",
        "pin_new": "Nova lozinka",
        "pin_confirm": "Potvrdi lozinku",
        "pin_wrong": "Pogrešna lozinka",
        "pin_mismatch": "Lozinke se ne podudaraju",
        "pin_saved": "Lozinka spremljena",
        "pin_cancel": "ODUSTANI",
        "pin_ok": "OK",
        "quit_app": "IZAĐI",
        "quit_confirm": "IZAĆI IZ APLIKACIJE?",
        "power_choose": "—",
        "power_off": "UGASI",
        "power_reboot": "RESETIRAJ",
        "power_exit": "IZAĐI",
        "power_off_confirm": "UGASITI RASPBERRY?",
        "power_reboot_confirm": "RESETIRATI RASPBERRY?",
        "power_exit_confirm": "IZAĆI IZ APLIKACIJE?",
        "report": "IZVJEŠTAJ",
        "report_title": "IZVJEŠTAJ",
        "report_auto": "Automatski hitovi",
        "report_manual": "Ručno upisano",
        "report_corrected": "Automatski prepravljeni",
        "report_top_numbers": "Najčešće prepravljeni",
        "report_top_games": "Najigranija igra",
        "report_games": "Odigrano igara",
        "report_none": "Nema podataka",
        "report_close": "ZATVORI",
    },
    "en": {
        "settings": "SETTINGS",
        "language": "Language",
        "bg_color": "Background color",
        "btn_color": "Button color",
        "start_mode": "Game start",
        "start_mqtt_qr": "MQTT / QR code",
        "start_button": "START button",
        "start_always": "Always on",
        "device": "Device",
        "league_qr": "League QR",
        "league_qr_on": "On",
        "league_qr_off": "Off",
        "tip_detection": "Tip detection",
        "tip_fitline": "FitLine",
        "tip_ml_onnx": "ML (ONNX)",
        "fitline_scale": "FitLine scale",
        "ml_tip_dataset": "ML tip dataset",
        "ml_tip_dataset_off": "Off",
        "ml_tip_dataset_on": "Save hits",
        "manual_mode": "Manual input",
        "manual_mode_off": "Off",
        "manual_mode_on": "On",
        "auto_calibrate": "Auto calibration",
        "auto_calibrate_on": "On",
        "auto_calibrate_off": "Off",
        "saving_ref": "SAVING REF...",
        "ref_not_saved": "Reference not saved — try again",
        "no_saved_cal": "No calibration — use CALIBRATION in settings first",
        "play_mode_auto": "AUTO",
        "play_mode_manual": "MANUAL",
        "switch_player": "SWITCH PLAYER",
        "calibration": "CALIBRATION",
        "calibrate": "CALIBRATE",
        "calibrating": "CALIBRATING...",
        "calibrating_hint": "Finding the board on cameras",
        "starting_game": "STARTING GAME",
        "starting_game_hint": "Calibration and detection setup",
        "please_wait": "Please wait",
        "select_game_sub": "Choose a game mode",
        "cal_incomplete": "Calibration incomplete — try again",
        "ingame_cal_title": "RECALIBRATE?",
        "ingame_cal_s1": "Hits not detecting well?",
        "ingame_cal_s2": "Remove all darts and recalibrate the board for more accurate detection.",
        "ingame_cal_done": "DONE",
        "ingame_cal_retry": "CALIBRATE AGAIN",
        "click_bull": "BULL",
        "click_bull_hint": "Click approximately on the bull",
        "click_ellipse": "ELLIPSE",
        "click_ellipse_hint": "Click up to 4 points on the outer rim",
        "overlay": "ELLIPSE",
        "enter_name": "PLAYER NAME",
        "name_skip": "SKIP",
        "name_remove": "REMOVE",
        "back": "BACK",
        "standby_qr": "Scan QR to start a game!",
        "standby_button": "Press START to begin",
        "start_game": "START",
        "select_game": "SELECT GAME",
        "select_players": "PLAYERS",
        "rules": "RULES",
        "continue": "CONTINUE",
        "tutorial": "TUTORIAL",
        "tutorial_options": "OPTIONS",
        "tutorial_basics": "HOW TO PLAY",
        "tutorial_ex_double_in": "DOUBLE IN\nhit a double first",
        "tutorial_ex_double_out": "DOUBLE OUT\nfinish on a double",
        "tutorial_ex_cut_scorer": "SURPLUS MARKS",
        "tutorial_ex_cut_rivals": "SCORE RIVALS",
        "tutorial_ex_killer_lives": "Lives: last player standing wins",
        "tutorial_ex_around_seg": "S / D / T / ANY = which zone counts",
        "clear_board_title": "CLEAR THE BOARD",
        "clear_board_s1": "Remove all darts from the board",
        "clear_board_s2": "before starting the game",
        "board_empty": "BOARD IS CLEAR",
        "exit_confirm": "EXIT GAME?",
        "yes": "YES",
        "no": "NO",
        "exit_game": "EXIT GAME",
        "idle_title": "GAME IDLE",
        "idle_s1": "The game has been idle for 5 minutes and will shut down.",
        "idle_s2": "The game will shut down.",
        "idle_quit": "QUIT",
        "idle_continue": "CONTINUE",
        "game_mode": "GAME MODE",
        "double_in": "DOUBLE IN",
        "double_out": "DOUBLE OUT",
        "no_limit": "NO LIMIT",
        "rounds_20": "20 ROUNDS",
        "cricket_standard": "STANDARD",
        "cricket_no_score": "NO SCORE",
        "cricket_cut": "CUT THROAT",
        "killer_lives": "LIVES",
        "killer_assign": "NUMBER ASSIGNMENT",
        "killer_assign_random": "RANDOM",
        "killer_assign_throw": "BY THROWING",
        "killer_activation": "ACTIVATION",
        "killer_act_single": "SINGLE",
        "killer_act_double": "DOUBLE",
        "killer_act_triple": "TRIPLE",
        "killer_dead": "DEAD",
        "killer_bull_prompt": "YOU HIT BULL — CHOOSE WHOSE LIFE TO TAKE",
        "killer_bull_false": "I DID NOT HIT BULL",
        "killer_one_player_impossible": "IMPOSSIBLE\nWITH 1 PLAYER",
        "halve_rounds": "NUMBER OF ROUNDS",
        "halve_standard": "STANDARD",
        "halve_extended": "EXTENDED",
        "halve_any_double": "ANY DOUBLE",
        "halve_any_triple": "ANY TRIPLE",
        "around_target_mult": "SEGMENTS",
        "around_mult_single": "SINGLE",
        "around_mult_double": "DOUBLE",
        "around_mult_triple": "TRIPLE",
        "around_mult_any": "ANY",
        "around_number_order": "NUMBER ORDER",
        "around_order_standard": "STANDARD",
        "around_order_random": "RANDOM",
        "around_order_reverse": "REVERSE",
        "around_backstep": "BACKSTEP PENALTY",
        "around_back": "BACK",
        "around_backstep_off": "OFF",
        "around_backstep_1": "BACK 1",
        "around_backstep_2": "BACK 2",
        "board_clear_end": "CLEAR BOARD TO FINISH",
        "board_clear": "CLEAR THE BOARD",
        "board_clean_q": "BOARD CLEAR?",
        "not_advanced_q": "DIDN'T ADVANCE?",
        "next_prefix": "Next:",
        "round": "Round",
        "bust": "BUST",
        "tutorial_x01_body": (
            "• Start at {start}.\n"
            "• 3 darts per turn — subtract from your total.\n"
            "• First to exactly 0 wins.\n"
            "• Below 0 = BUST (turn cancelled)."
        ),
        "tutorial_x01_opts": (
            "• DOUBLE IN OFF: scoring starts immediately.\n"
            "• DOUBLE IN ON: hit a double (or B50) to start.\n"
            "• DOUBLE OUT OFF: any hit to exactly 0.\n"
            "• DOUBLE OUT ON: finish on a double; left on 1 = BUST."
        ),
        "tutorial_cricket_body": (
            "• Close 15–20 and bull (3 marks each).\n"
            "• Single=1, double=2, triple=3 marks.\n"
            "• Close everything and meet the mode win condition."
        ),
        "tutorial_cricket_opts": (
            "• NO LIMIT / 20 ROUNDS — game length.\n"
            "• STANDARD: surplus marks score for you.\n"
            "• NO SCORE: marks only, no points.\n"
            "• CUT THROAT: surplus scores rivals; fewer points is better."
        ),
        "tutorial_around_body": (
            "• Hit numbers in order (then bull, depending on options).\n"
            "• Hit required number + segment to advance.\n"
            "• First to finish the sequence wins."
        ),
        "tutorial_around_opts": (
            "• SEGMENTS: SINGLE / DOUBLE / TRIPLE / ANY.\n"
            "• ORDER: STANDARD 1→20→bull, REVERSE, or RANDOM.\n"
            "• BACKSTEP: OFF / BACK 1 / BACK 2 if you miss the whole turn."
        ),
        "tutorial_killer_body": (
            "• Each player has a number and lives.\n"
            "• Become Killer with your number + activation multiplier.\n"
            "• ACTIVATION (Single/Double/Triple): you must hit that exact segment "
            "to become Killer and to take lives from rivals.\n"
            "• As Killer, hit rivals to take a life.\n"
            "• Hitting your own number as Killer costs a life.\n"
            "• Last player with lives wins."
        ),
        "tutorial_killer_opts": (
            "• LIVES: 3 / 5 / 7 / 10.\n"
            "• RANDOM or BY THROWING — number assignment.\n"
            "• ACTIVATION: SINGLE / DOUBLE / TRIPLE — required multiplier "
            "to become Killer and to remove lives."
        ),
        "tutorial_halve_body": (
            "• Start at 0; each round has one target (e.g. 20, ANY DOUBLE, BULL).\n"
            "• Add valid hits on that target with 3 darts "
            "(S/D/T of the number; on DOUBLE/TRIPLE rounds — any double/triple).\n"
            "• Score 0 on the round → your total is halved (floor).\n"
            "• After all rounds, highest score wins."
        ),
        "tutorial_halve_opts": (
            "• STANDARD: 20→15 and BULL (7 rounds).\n"
            "• EXTENDED: 20→15, ANY DOUBLE, 14→7, ANY TRIPLE, 6→1, BULL."
        ),
        "bg_red": "Red",
        "bg_blue": "Blue",
        "bg_green": "Green",
        "bg_yellow": "Yellow",
        "bg_gray": "Gray",
        "bg_purple": "Purple",
        "bg_white": "White",
        "bg_black": "Black",
        "btn_yellow": "Yellow",
        "btn_green": "Green",
        "btn_blue": "Blue",
        "btn_orange": "Orange",
        "btn_white": "White",
        "btn_gray": "Gray",
        "btn_black": "Black",
        "btn_cyan": "Cyan",
        "btn_pink": "Pink",
        "winner_title": "WINNER!",
        "league_qr_hint": "Scan for league",
        "change_password": "CHANGE PASSWORD",
        "change_password_short": "Change",
        "password": "Password",
        "application": "Application",
        "pin_enter": "Enter password",
        "pin_new": "New password",
        "pin_confirm": "Confirm password",
        "pin_wrong": "Wrong password",
        "pin_mismatch": "Passwords do not match",
        "pin_saved": "Password saved",
        "pin_cancel": "CANCEL",
        "pin_ok": "OK",
        "quit_app": "EXIT",
        "quit_confirm": "EXIT THE APPLICATION?",
        "power_choose": "—",
        "power_off": "SHUT DOWN",
        "power_reboot": "REBOOT",
        "power_exit": "EXIT",
        "power_off_confirm": "SHUT DOWN THE RASPBERRY?",
        "power_reboot_confirm": "REBOOT THE RASPBERRY?",
        "power_exit_confirm": "EXIT THE APPLICATION?",
        "report": "REPORT",
        "report_title": "REPORT",
        "report_auto": "Automatic hits",
        "report_manual": "Entered manually",
        "report_corrected": "Automatic hits corrected",
        "report_top_numbers": "Most corrected numbers",
        "report_top_games": "Most played game",
        "report_games": "Games played",
        "report_none": "No data yet",
        "report_close": "CLOSE",
    },
    "de": {
        "settings": "EINSTELLUNGEN",
        "language": "Sprache",
        "bg_color": "Hintergrundfarbe",
        "btn_color": "Tastenfarbe",
        "start_mode": "Spielstart",
        "start_mqtt_qr": "MQTT / QR-Code",
        "start_button": "START-Taste",
        "start_always": "Immer an",
        "device": "Gerät",
        "league_qr": "Liga-QR",
        "league_qr_on": "An",
        "league_qr_off": "Aus",
        "tip_detection": "Tipp-Erkennung",
        "tip_fitline": "FitLine",
        "tip_ml_onnx": "ML (ONNX)",
        "fitline_scale": "FitLine-Skala",
        "ml_tip_dataset": "ML-Tipp-Dataset",
        "ml_tip_dataset_off": "Aus",
        "ml_tip_dataset_on": "Treffer speichern",
        "manual_mode": "Manuelle Eingabe",
        "manual_mode_off": "Aus",
        "manual_mode_on": "An",
        "auto_calibrate": "Auto-Kalibrierung",
        "auto_calibrate_on": "An",
        "auto_calibrate_off": "Aus",
        "saving_ref": "REF SPEICHERN...",
        "ref_not_saved": "Referenz nicht gespeichert — erneut versuchen",
        "no_saved_cal": "Keine Kalibrierung — zuerst KALIBRIERUNG in den Einstellungen",
        "play_mode_auto": "AUTO",
        "play_mode_manual": "MANUAL",
        "switch_player": "SPIELER WECHSELN",
        "calibration": "KALIBRIERUNG",
        "calibrate": "KALIBRIEREN",
        "calibrating": "KALIBRIERE...",
        "calibrating_hint": "Suche das Board auf den Kameras",
        "starting_game": "SPIEL STARTET",
        "starting_game_hint": "Kalibrierung und Vorbereitung",
        "please_wait": "Bitte warten",
        "select_game_sub": "Spielmodus wählen",
        "cal_incomplete": "Kalibrierung unvollständig — erneut versuchen",
        "ingame_cal_title": "NEU KALIBRIEREN?",
        "ingame_cal_s1": "Treffer werden schlecht erkannt?",
        "ingame_cal_s2": "Darts entfernen und die Scheibe neu kalibrieren für präzisere Erkennung.",
        "ingame_cal_done": "FERTIG",
        "ingame_cal_retry": "ERNEUT KALIBRIEREN",
        "click_bull": "BULL",
        "click_bull_hint": "Ungefähr auf den Bull klicken",
        "click_ellipse": "ELLIPSE",
        "click_ellipse_hint": "Bis zu 4 Punkte am Aussenrand klicken",
        "overlay": "ELLIPSE",
        "enter_name": "SPIELERNAME",
        "name_skip": "SKIP",
        "name_remove": "ENTFERNEN",
        "back": "ZURÜCK",
        "standby_qr": "QR scannen zum Start!",
        "standby_button": "START drücken zum Beginnen",
        "start_game": "START",
        "select_game": "SPIEL WÄHLEN",
        "select_players": "SPIELER",
        "rules": "REGELN",
        "continue": "WEITER",
        "tutorial": "ANLEITUNG",
        "tutorial_options": "OPTIONEN",
        "tutorial_basics": "SO GEHT'S",
        "tutorial_ex_double_in": "DOUBLE IN\nzuerst Double",
        "tutorial_ex_double_out": "DOUBLE OUT\nmit Double beenden",
        "tutorial_ex_cut_scorer": "ÜBERSCHUSS",
        "tutorial_ex_cut_rivals": "PUNKTE AN RIVALEN",
        "tutorial_ex_killer_lives": "Leben: der Letzte gewinnt",
        "tutorial_ex_around_seg": "S / D / T / ANY = welche Zone zählt",
        "clear_board_title": "SCHEIBE LEEREN",
        "clear_board_s1": "Alle Darts von der Scheibe nehmen",
        "clear_board_s2": "vor Spielbeginn",
        "board_empty": "SCHEIBE IST LEER",
        "exit_confirm": "SPIEL BEENDEN?",
        "yes": "JA",
        "no": "NEIN",
        "exit_game": "SPIEL BEENDEN",
        "idle_title": "SPIEL INAKTIV",
        "idle_s1": "Das Spiel ist seit 5 Minuten inaktiv und wird beendet.",
        "idle_s2": "Das Spiel wird beendet.",
        "idle_quit": "BEENDEN",
        "idle_continue": "WEITER",
        "game_mode": "GAME MODE",
        "double_in": "DOUBLE IN",
        "double_out": "DOUBLE OUT",
        "no_limit": "OHNE LIMIT",
        "rounds_20": "20 RUNDEN",
        "cricket_standard": "STANDARD",
        "cricket_no_score": "NO SCORE",
        "cricket_cut": "CUT THROAT",
        "killer_lives": "LEBEN",
        "killer_assign": "NUMMERNVERGABE",
        "killer_assign_random": "ZUFALL",
        "killer_assign_throw": "DURCH WURF",
        "killer_activation": "AKTIVIERUNG",
        "killer_act_single": "SINGLE",
        "killer_act_double": "DOUBLE",
        "killer_act_triple": "TRIPLE",
        "killer_dead": "TOT",
        "killer_bull_prompt": "BULL GETROFFEN — WÄHLE WESSEN LEBEN DU NIMMST",
        "killer_bull_false": "KEIN BULL GETROFFEN",
        "killer_one_player_impossible": "UNMÖGLICH\nMIT 1 SPIELER",
        "halve_rounds": "ANZAHL DER RUNDEN",
        "halve_standard": "STANDARD",
        "halve_extended": "ERWEITERT",
        "halve_any_double": "BELIEBIGES DOUBLE",
        "halve_any_triple": "BELIEBIGES TRIPLE",
        "around_target_mult": "SEGMENTE",
        "around_mult_single": "SINGLE",
        "around_mult_double": "DOUBLE",
        "around_mult_triple": "TRIPLE",
        "around_mult_any": "ANY",
        "around_number_order": "ZAHLENREIHENFOLGE",
        "around_order_standard": "STANDARD",
        "around_order_random": "ZUFALL",
        "around_order_reverse": "REVERSE",
        "around_backstep": "RÜCKSCHRITT-STRAFE",
        "around_back": "ZURÜCK",
        "around_backstep_off": "AUS",
        "around_backstep_1": "ZURÜCK 1",
        "around_backstep_2": "ZURÜCK 2",
        "board_clear_end": "SCHEIBE LEEREN ZUM ENDE",
        "board_clear": "SCHEIBE LEEREN",
        "board_clean_q": "SCHEIBE LEER?",
        "not_advanced_q": "NICHT WEITER?",
        "next_prefix": "Nächste:",
        "round": "Runde",
        "bust": "BUST",
        "tutorial_x01_body": (
            "• Start mit {start}.\n"
            "• 3 Darts pro Runde — vom Rest abziehen.\n"
            "• Wer zuerst genau 0 erreicht, gewinnt.\n"
            "• Unter 0 = BUST (Wurf verworfen)."
        ),
        "tutorial_x01_opts": (
            "• DOUBLE IN AUS: Punkte zählen sofort.\n"
            "• DOUBLE IN AN: zuerst Double (oder B50).\n"
            "• DOUBLE OUT AUS: jeder Treffer auf 0.\n"
            "• DOUBLE OUT AN: mit Double beenden; Rest 1 = BUST."
        ),
        "tutorial_cricket_body": (
            "• 15–20 und Bull schließen (3 Marken).\n"
            "• Single=1, Double=2, Triple=3 Marken.\n"
            "• Alles schließen und Modus-Siegbedingung erfüllen."
        ),
        "tutorial_cricket_opts": (
            "• OHNE LIMIT / 20 RUNDEN — Spieldauer.\n"
            "• STANDARD: Überschuss = Punkte für dich.\n"
            "• NO SCORE: nur Marken, keine Punkte.\n"
            "• CUT THROAT: Überschuss an Rivalen; weniger ist besser."
        ),
        "tutorial_around_body": (
            "• Zahlen der Reihe nach (dann Bull, je nach Option).\n"
            "• Geforderte Zahl + Segment zum Weiterschalten.\n"
            "• Wer zuerst fertig ist, gewinnt."
        ),
        "tutorial_around_opts": (
            "• SEGMENTE: SINGLE / DOUBLE / TRIPLE / ANY.\n"
            "• REIHENFOLGE: STANDARD 1→20→Bull, REVERSE oder ZUFALL.\n"
            "• RÜCKSCHRITT: AUS / ZURÜCK 1 / ZURÜCK 2 bei Fehlrunde."
        ),
        "tutorial_killer_body": (
            "• Jeder hat eine Zahl und Leben.\n"
            "• Werde Killer mit Zahl + Aktivierungs-Multiplikator.\n"
            "• AKTIVIERUNG (Single/Double/Triple): genau dieses Segment treffen, "
            "um Killer zu werden und Rivalen Leben zu nehmen.\n"
            "• Als Killer Rivalen treffen und Leben nehmen.\n"
            "• Eigene Zahl als Killer = Leben verloren.\n"
            "• Der Letzte mit Leben gewinnt."
        ),
        "tutorial_killer_opts": (
            "• LEBEN: 3 / 5 / 7 / 10.\n"
            "• ZUFALL oder DURCH WURF — Zahlenvergabe.\n"
            "• AKTIVIERUNG: SINGLE / DOUBLE / TRIPLE — Pflicht-Multiplikator "
            "für Killer-Status und Leben abziehen."
        ),
        "tutorial_halve_body": (
            "• Start bei 0; jede Runde hat ein Ziel (z. B. 20, BELIEBIGES DOUBLE, BULL).\n"
            "• Zähle gültige Treffer auf das Ziel mit 3 Darts "
            "(S/D/T der Zahl; bei DOUBLE/TRIPLE-Runden — beliebiges Double/Triple).\n"
            "• 0 Punkte in der Runde → Gesamtpunktzahl wird halbiert (abgerundet).\n"
            "• Nach allen Runden gewinnt die höchste Punktzahl."
        ),
        "tutorial_halve_opts": (
            "• STANDARD: 20→15 und BULL (7 Runden).\n"
            "• ERWEITERT: 20→15, BELIEBIGES DOUBLE, 14→7, "
            "BELIEBIGES TRIPLE, 6→1, BULL."
        ),
        "bg_red": "Rot",
        "bg_blue": "Blau",
        "bg_green": "Grün",
        "bg_yellow": "Gelb",
        "bg_gray": "Grau",
        "bg_purple": "Lila",
        "bg_white": "Weiß",
        "bg_black": "Schwarz",
        "btn_yellow": "Gelb",
        "btn_green": "Grün",
        "btn_blue": "Blau",
        "btn_orange": "Orange",
        "btn_white": "Weiß",
        "btn_gray": "Grau",
        "btn_black": "Schwarz",
        "btn_cyan": "Cyan",
        "btn_pink": "Rosa",
        "winner_title": "SIEGER!",
        "league_qr_hint": "Für die Liga scannen",
        "change_password": "PASSWORT ÄNDERN",
        "change_password_short": "Ändern",
        "password": "Passwort",
        "application": "Anwendung",
        "pin_enter": "Passwort eingeben",
        "pin_new": "Neues Passwort",
        "pin_confirm": "Passwort bestätigen",
        "pin_wrong": "Falsches Passwort",
        "pin_mismatch": "Passwörter stimmen nicht",
        "pin_saved": "Passwort gespeichert",
        "pin_cancel": "ABBRECHEN",
        "pin_ok": "OK",
        "quit_app": "BEENDEN",
        "quit_confirm": "ANWENDUNG BEENDEN?",
        "power_choose": "—",
        "power_off": "AUS",
        "power_reboot": "NEUSTART",
        "power_exit": "BEENDEN",
        "power_off_confirm": "RASPBERRY AUSSCHALTEN?",
        "power_reboot_confirm": "RASPBERRY NEUSTARTEN?",
        "power_exit_confirm": "ANWENDUNG BEENDEN?",
        "report": "BERICHT",
        "report_title": "BERICHT",
        "report_auto": "Automatische Hits",
        "report_manual": "Manuell eingegeben",
        "report_corrected": "Automatik korrigiert",
        "report_top_numbers": "Am häufigsten korrigiert",
        "report_top_games": "Meistgespieltes Spiel",
        "report_games": "Gespielte Spiele",
        "report_none": "Keine Daten",
        "report_close": "SCHLIESSEN",
    },
    "fr": {
        "settings": "PARAMÈTRES",
        "language": "Langue",
        "bg_color": "Couleur de fond",
        "btn_color": "Couleur des boutons",
        "start_mode": "Démarrage",
        "start_mqtt_qr": "MQTT / QR code",
        "start_button": "Bouton DÉMARRER",
        "start_always": "Toujours allumé",
        "device": "Appareil",
        "league_qr": "QR ligue",
        "league_qr_on": "Activé",
        "league_qr_off": "Désactivé",
        "tip_detection": "Détection pointe",
        "tip_fitline": "FitLine",
        "tip_ml_onnx": "ML (ONNX)",
        "fitline_scale": "Échelle FitLine",
        "ml_tip_dataset": "Dataset pointe ML",
        "ml_tip_dataset_off": "Désactivé",
        "ml_tip_dataset_on": "Enregistrer",
        "manual_mode": "Saisie manuelle",
        "manual_mode_off": "Désactivé",
        "manual_mode_on": "Activé",
        "auto_calibrate": "Calibration auto",
        "auto_calibrate_on": "Activé",
        "auto_calibrate_off": "Désactivé",
        "saving_ref": "ENREG. REF...",
        "ref_not_saved": "Référence non enregistrée — réessayer",
        "no_saved_cal": "Pas de calibrage — d'abord CALIBRAGE dans les réglages",
        "play_mode_auto": "AUTO",
        "play_mode_manual": "MANUAL",
        "switch_player": "JOUEUR SUIVANT",
        "calibration": "CALIBRAGE",
        "calibrate": "CALIBRER",
        "calibrating": "CALIBRAGE...",
        "calibrating_hint": "Recherche de la cible sur les caméras",
        "starting_game": "LANCEMENT DU JEU",
        "starting_game_hint": "Calibrage et préparation",
        "please_wait": "Veuillez patienter",
        "select_game_sub": "Choisissez un mode",
        "cal_incomplete": "Calibration incomplète — réessayez",
        "ingame_cal_title": "RECALIBRER ?",
        "ingame_cal_s1": "Les touches sont mal détectées ?",
        "ingame_cal_s2": "Retirez les fléchettes et recalibrez la cible pour une détection plus précise.",
        "ingame_cal_done": "TERMINÉ",
        "ingame_cal_retry": "RECALIBRER",
        "click_bull": "BULL",
        "click_bull_hint": "Cliquez approximativement sur le bull",
        "click_ellipse": "ELLIPSE",
        "click_ellipse_hint": "Cliquez jusqu'à 4 points sur le bord extérieur",
        "overlay": "ELLIPSE",
        "enter_name": "NOM DU JOUEUR",
        "name_skip": "PASSER",
        "name_remove": "RETIRER",
        "back": "RETOUR",
        "standby_qr": "Scannez le QR pour commencer !",
        "standby_button": "Appuyez sur DÉMARRER",
        "start_game": "DÉMARRER",
        "select_game": "CHOISIR JEU",
        "select_players": "JOUEURS",
        "rules": "RÈGLES",
        "continue": "CONTINUER",
        "tutorial": "TUTORIEL",
        "tutorial_options": "OPTIONS",
        "tutorial_basics": "COMMENT JOUER",
        "tutorial_ex_double_in": "DOUBLE IN\nd'abord un double",
        "tutorial_ex_double_out": "DOUBLE OUT\nfinir sur un double",
        "tutorial_ex_cut_scorer": "MARQUES EN TROP",
        "tutorial_ex_cut_rivals": "POINTS AUX RIVAUX",
        "tutorial_ex_killer_lives": "Vies: le dernier restant gagne",
        "tutorial_ex_around_seg": "S / D / T / ANY = zone qui compte",
        "clear_board_title": "VIDEZ LA CIBLE",
        "clear_board_s1": "Retirez toutes les fléchettes",
        "clear_board_s2": "avant de commencer",
        "board_empty": "CIBLE VIDE",
        "exit_confirm": "QUITTER LA PARTIE ?",
        "yes": "OUI",
        "no": "NON",
        "exit_game": "QUITTER",
        "idle_title": "PARTIE INACTIVE",
        "idle_s1": "La partie est inactive depuis 5 minutes et va s'arrêter.",
        "idle_s2": "La partie va s'arrêter.",
        "idle_quit": "QUITTER",
        "idle_continue": "CONTINUER",
        "game_mode": "GAME MODE",
        "double_in": "DOUBLE IN",
        "double_out": "DOUBLE OUT",
        "no_limit": "SANS LIMITE",
        "rounds_20": "20 MANCHES",
        "cricket_standard": "STANDARD",
        "cricket_no_score": "NO SCORE",
        "cricket_cut": "CUT THROAT",
        "killer_lives": "VIES",
        "killer_assign": "ATTRIBUTION DES NUMÉROS",
        "killer_assign_random": "ALÉATOIRE",
        "killer_assign_throw": "EN LANÇANT",
        "killer_activation": "ACTIVATION",
        "killer_act_single": "SINGLE",
        "killer_act_double": "DOUBLE",
        "killer_act_triple": "TRIPLE",
        "killer_dead": "MORT",
        "killer_bull_prompt": "VOUS AVEZ TOUCHÉ LE BULL — CHOISISSEZ À QUI RETIRER UNE VIE",
        "killer_bull_false": "JE N'AI PAS TOUCHÉ LE BULL",
        "killer_one_player_impossible": "IMPOSSIBLE\nAVEC 1 JOUEUR",
        "halve_rounds": "NOMBRE DE MANCHES",
        "halve_standard": "STANDARD",
        "halve_extended": "ÉTENDU",
        "halve_any_double": "N'IMPORTE QUEL DOUBLE",
        "halve_any_triple": "N'IMPORTE QUEL TRIPLE",
        "around_target_mult": "SEGMENTS",
        "around_mult_single": "SINGLE",
        "around_mult_double": "DOUBLE",
        "around_mult_triple": "TRIPLE",
        "around_mult_any": "ANY",
        "around_number_order": "ORDRE DES NUMÉROS",
        "around_order_standard": "STANDARD",
        "around_order_random": "ALÉATOIRE",
        "around_order_reverse": "REVERSE",
        "around_backstep": "PÉNALITÉ DE RETOUR",
        "around_back": "RETOUR",
        "around_backstep_off": "DÉSACTIVÉ",
        "around_backstep_1": "RETOUR 1",
        "around_backstep_2": "RETOUR 2",
        "board_clear_end": "VIDEZ LA CIBLE POUR FINIR",
        "board_clear": "VIDEZ LA CIBLE",
        "board_clean_q": "CIBLE VIDE ?",
        "not_advanced_q": "PAS AVANCÉ ?",
        "next_prefix": "Suivant :",
        "round": "Manche",
        "bust": "BUST",
        "tutorial_x01_body": (
            "• Départ à {start}.\n"
            "• 3 fléchettes par tour — soustrayez du total.\n"
            "• Premier à exactement 0 gagne.\n"
            "• Sous 0 = BUST (tour annulé)."
        ),
        "tutorial_x01_opts": (
            "• DOUBLE IN OFF: score immédiat.\n"
            "• DOUBLE IN ON: d'abord un double (ou B50).\n"
            "• DOUBLE OUT OFF: n'importe quel tir à 0.\n"
            "• DOUBLE OUT ON: finir sur un double; reste 1 = BUST."
        ),
        "tutorial_cricket_body": (
            "• Fermez 15–20 et bull (3 marques).\n"
            "• Simple=1, double=2, triple=3.\n"
            "• Fermez tout et remplissez la condition du mode."
        ),
        "tutorial_cricket_opts": (
            "• SANS LIMITE / 20 MANCHES — durée.\n"
            "• STANDARD: surplus = points pour vous.\n"
            "• NO SCORE: marques seulement.\n"
            "• CUT THROAT: surplus aux rivaux; moins = mieux."
        ),
        "tutorial_around_body": (
            "• Numéros dans l'ordre (puis bull selon options).\n"
            "• Numéro + segment requis pour avancer.\n"
            "• Premier à finir la séquence gagne."
        ),
        "tutorial_around_opts": (
            "• SEGMENTS: SINGLE / DOUBLE / TRIPLE / ANY.\n"
            "• ORDRE: STANDARD 1→20→bull, REVERSE ou ALÉATOIRE.\n"
            "• RETOUR: OFF / RETOUR 1 / RETOUR 2 si tour manqué."
        ),
        "tutorial_killer_body": (
            "• Chaque joueur a un numéro et des vies.\n"
            "• Devenez Killer avec numéro + multiplicateur d'activation.\n"
            "• ACTIVATION (Single/Double/Triple) : il faut toucher ce segment exact "
            "pour devenir Killer et retirer des vies aux rivaux.\n"
            "• En Killer, visez les rivaux pour prendre une vie.\n"
            "• Votre propre numéro en Killer = vie perdue.\n"
            "• Le dernier avec des vies gagne."
        ),
        "tutorial_killer_opts": (
            "• VIES: 3 / 5 / 7 / 10.\n"
            "• ALÉATOIRE ou EN LANÇANT — numéros.\n"
            "• ACTIVATION: SINGLE / DOUBLE / TRIPLE — multiplicateur requis "
            "pour devenir Killer et retirer des vies."
        ),
        "tutorial_halve_body": (
            "• Départ à 0 ; chaque manche a une cible (ex. 20, N'IMPORTE QUEL DOUBLE, BULL).\n"
            "• Additionnez les touches valides sur la cible avec 3 fléchettes "
            "(S/D/T du numéro ; en manche DOUBLE/TRIPLE — n'importe quel double/triple).\n"
            "• 0 point sur la manche → total divisé par 2 (arrondi vers le bas).\n"
            "• Après toutes les manches, le plus haut score gagne."
        ),
        "tutorial_halve_opts": (
            "• STANDARD: 20→15 et BULL (7 manches).\n"
            "• ÉTENDU: 20→15, N'IMPORTE QUEL DOUBLE, 14→7, "
            "N'IMPORTE QUEL TRIPLE, 6→1, BULL."
        ),
        "bg_red": "Rouge",
        "bg_blue": "Bleu",
        "bg_green": "Vert",
        "bg_yellow": "Jaune",
        "bg_gray": "Gris",
        "bg_purple": "Violet",
        "bg_white": "Blanc",
        "bg_black": "Noir",
        "btn_yellow": "Jaune",
        "btn_green": "Vert",
        "btn_blue": "Bleu",
        "btn_orange": "Orange",
        "btn_white": "Blanc",
        "btn_gray": "Gris",
        "btn_black": "Noir",
        "btn_cyan": "Cyan",
        "btn_pink": "Rose",
        "winner_title": "GAGNANT !",
        "league_qr_hint": "Scanner pour la ligue",
        "change_password": "CHANGER MOT DE PASSE",
        "change_password_short": "Changer",
        "password": "Mot de passe",
        "application": "Application",
        "pin_enter": "Entrer le mot de passe",
        "pin_new": "Nouveau mot de passe",
        "pin_confirm": "Confirmer le mot de passe",
        "pin_wrong": "Mot de passe incorrect",
        "pin_mismatch": "Mots de passe différents",
        "pin_saved": "Mot de passe enregistré",
        "pin_cancel": "ANNULER",
        "pin_ok": "OK",
        "quit_app": "QUITTER",
        "quit_confirm": "QUITTER L'APPLICATION ?",
        "power_choose": "—",
        "power_off": "ÉTEINDRE",
        "power_reboot": "REDÉMARRER",
        "power_exit": "QUITTER",
        "power_off_confirm": "ÉTEINDRE LE RASPBERRY ?",
        "power_reboot_confirm": "REDÉMARRER LE RASPBERRY ?",
        "power_exit_confirm": "QUITTER L'APPLICATION ?",
        "report": "RAPPORT",
        "report_title": "RAPPORT",
        "report_auto": "Hits automatiques",
        "report_manual": "Saisie manuelle",
        "report_corrected": "Hits auto corrigés",
        "report_top_numbers": "Numéros les plus corrigés",
        "report_top_games": "Jeu le plus joué",
        "report_games": "Parties jouées",
        "report_none": "Pas de données",
        "report_close": "FERMER",
    },
    "it": {
        "settings": "IMPOSTAZIONI",
        "language": "Lingua",
        "bg_color": "Colore sfondo",
        "btn_color": "Colore pulsanti",
        "start_mode": "Avvio gioco",
        "start_mqtt_qr": "MQTT / QR code",
        "start_button": "Pulsante AVVIA",
        "start_always": "Sempre acceso",
        "device": "Dispositivo",
        "league_qr": "QR lega",
        "league_qr_on": "Attivato",
        "league_qr_off": "Disattivato",
        "tip_detection": "Rilevamento punta",
        "tip_fitline": "FitLine",
        "tip_ml_onnx": "ML (ONNX)",
        "fitline_scale": "Scala FitLine",
        "ml_tip_dataset": "Dataset punta ML",
        "ml_tip_dataset_off": "Disattivato",
        "ml_tip_dataset_on": "Salva colpi",
        "manual_mode": "Inserimento manuale",
        "manual_mode_off": "Disattivato",
        "manual_mode_on": "Attivato",
        "auto_calibrate": "Calibrazione auto",
        "auto_calibrate_on": "Attivato",
        "auto_calibrate_off": "Disattivato",
        "saving_ref": "SALVO REF...",
        "ref_not_saved": "Riferimento non salvato — riprova",
        "no_saved_cal": "Nessuna calibrazione — prima CALIBRAZIONE nelle impostazioni",
        "play_mode_auto": "AUTO",
        "play_mode_manual": "MANUAL",
        "switch_player": "CAMBIA GIOCATORE",
        "calibration": "CALIBRAZIONE",
        "calibrate": "CALIBRA",
        "calibrating": "CALIBRAZIONE...",
        "calibrating_hint": "Cerco il bersaglio sulle telecamere",
        "starting_game": "AVVIO PARTITA",
        "starting_game_hint": "Calibrazione e preparazione",
        "please_wait": "Attendere",
        "select_game_sub": "Scegli una modalità",
        "cal_incomplete": "Calibrazione incompleta — riprova",
        "ingame_cal_title": "RICALIBRARE?",
        "ingame_cal_s1": "I tiri non vengono rilevati bene?",
        "ingame_cal_s2": "Togli le frecce e ricalibra il bersaglio per un rilevamento più preciso.",
        "ingame_cal_done": "FATTO",
        "ingame_cal_retry": "RICALIBRA",
        "click_bull": "BULL",
        "click_bull_hint": "Clicca approssimativamente sul bull",
        "click_ellipse": "ELLISSE",
        "click_ellipse_hint": "Clicca fino a 4 punti sul bordo esterno",
        "overlay": "ELLISSE",
        "enter_name": "NOME GIOCATORE",
        "name_skip": "SALTA",
        "name_remove": "RIMUOVI",
        "back": "INDIETRO",
        "standby_qr": "Scansiona QR per iniziare!",
        "standby_button": "Premi AVVIA per iniziare",
        "start_game": "AVVIA",
        "select_game": "SCEGLI GIOCO",
        "select_players": "GIOCATORI",
        "rules": "REGOLE",
        "continue": "CONTINUA",
        "tutorial": "TUTORIAL",
        "tutorial_options": "OPZIONI",
        "tutorial_basics": "COME SI GIOCA",
        "tutorial_ex_double_in": "DOUBLE IN\nprima un double",
        "tutorial_ex_double_out": "DOUBLE OUT\nchiudi con un double",
        "tutorial_ex_cut_scorer": "MARCHE IN ECCESSO",
        "tutorial_ex_cut_rivals": "PUNTI AI RIVALI",
        "tutorial_ex_killer_lives": "Vite: vince l'ultimo rimasto",
        "tutorial_ex_around_seg": "S / D / T / ANY = zona che conta",
        "clear_board_title": "SVUOTA IL BERSAGLIO",
        "clear_board_s1": "Rimuovi tutte le freccette",
        "clear_board_s2": "prima di iniziare",
        "board_empty": "BERSAGLIO VUOTO",
        "exit_confirm": "ESCI DAL GIOCO?",
        "yes": "SÌ",
        "no": "NO",
        "exit_game": "ESCI",
        "idle_title": "PARTITA INATTIVA",
        "idle_s1": "La partita è inattiva da 5 minuti e sta per chiudersi.",
        "idle_s2": "La partita sta per chiudersi.",
        "idle_quit": "CHIUDI",
        "idle_continue": "CONTINUA",
        "game_mode": "GAME MODE",
        "double_in": "DOUBLE IN",
        "double_out": "DOUBLE OUT",
        "no_limit": "SENZA LIMITE",
        "rounds_20": "20 ROUND",
        "cricket_standard": "STANDARD",
        "cricket_no_score": "NO SCORE",
        "cricket_cut": "CUT THROAT",
        "killer_lives": "VITE",
        "killer_assign": "ASSEGNAZIONE NUMERI",
        "killer_assign_random": "CASUALE",
        "killer_assign_throw": "LANCIANDO",
        "killer_activation": "ATTIVAZIONE",
        "killer_act_single": "SINGLE",
        "killer_act_double": "DOUBLE",
        "killer_act_triple": "TRIPLE",
        "killer_dead": "MORTO",
        "killer_bull_prompt": "HAI COLPITO IL BULL — SCEGLI A CHI TOGLIERE UNA VITA",
        "killer_bull_false": "NON HO COLPITO IL BULL",
        "killer_one_player_impossible": "IMPOSSIBILE\nCON 1 GIOCATORE",
        "halve_rounds": "NUMERO DI ROUND",
        "halve_standard": "STANDARD",
        "halve_extended": "ESTESO",
        "halve_any_double": "QUALSIASI DOUBLE",
        "halve_any_triple": "QUALSIASI TRIPLE",
        "around_target_mult": "SEGMENTI",
        "around_mult_single": "SINGLE",
        "around_mult_double": "DOUBLE",
        "around_mult_triple": "TRIPLE",
        "around_mult_any": "ANY",
        "around_number_order": "ORDINE DEI NUMERI",
        "around_order_standard": "STANDARD",
        "around_order_random": "CASUALE",
        "around_order_reverse": "REVERSE",
        "around_backstep": "PENALITÀ INDIETRO",
        "around_back": "INDIETRO",
        "around_backstep_off": "DISATTIVATO",
        "around_backstep_1": "INDIETRO 1",
        "around_backstep_2": "INDIETRO 2",
        "board_clear_end": "SVUOTA PER FINIRE",
        "board_clear": "SVUOTA IL BERSAGLIO",
        "board_clean_q": "BERSAGLIO VUOTO?",
        "not_advanced_q": "NON AVANZATO?",
        "next_prefix": "Prossimo:",
        "round": "Round",
        "bust": "BUST",
        "tutorial_x01_body": (
            "• Parti da {start}.\n"
            "• 3 freccette per turno — sottrai dal totale.\n"
            "• Vince chi arriva esattamente a 0.\n"
            "• Sotto 0 = BUST (turno annullato)."
        ),
        "tutorial_x01_opts": (
            "• DOUBLE IN OFF: punteggio subito.\n"
            "• DOUBLE IN ON: prima un double (o B50).\n"
            "• DOUBLE OUT OFF: qualsiasi tiro a 0.\n"
            "• DOUBLE OUT ON: chiudi con un double; resto 1 = BUST."
        ),
        "tutorial_cricket_body": (
            "• Chiudi 15–20 e bull (3 marche).\n"
            "• Single=1, double=2, triple=3.\n"
            "• Chiudi tutto e soddisfa la condizione del modo."
        ),
        "tutorial_cricket_opts": (
            "• SENZA LIMITE / 20 ROUND — durata.\n"
            "• STANDARD: eccedenza = punti a te.\n"
            "• NO SCORE: solo marche.\n"
            "• CUT THROAT: eccedenza ai rivali; meno = meglio."
        ),
        "tutorial_around_body": (
            "• Numeri in ordine (poi bull secondo opzioni).\n"
            "• Numero + segmento richiesti per avanzare.\n"
            "• Vince chi finisce la sequenza per primo."
        ),
        "tutorial_around_opts": (
            "• SEGMENTI: SINGLE / DOUBLE / TRIPLE / ANY.\n"
            "• ORDINE: STANDARD 1→20→bull, REVERSE o CASUALE.\n"
            "• INDIETRO: OFF / INDIETRO 1 / INDIETRO 2 se turni a vuoto."
        ),
        "tutorial_killer_body": (
            "• Ognuno ha un numero e delle vite.\n"
            "• Diventa Killer con numero + moltiplicatore di attivazione.\n"
            "• ATTIVAZIONE (Single/Double/Triple): devi colpire esattamente quel "
            "segmento per diventare Killer e togliere vite ai rivali.\n"
            "• Da Killer colpisci i rivali per togliere una vita.\n"
            "• Il tuo numero da Killer = vita persa.\n"
            "• Vince l'ultimo con vite."
        ),
        "tutorial_killer_opts": (
            "• VITE: 3 / 5 / 7 / 10.\n"
            "• CASUALE o LANCIANDO — numeri.\n"
            "• ATTIVAZIONE: SINGLE / DOUBLE / TRIPLE — moltiplicatore obbligatorio "
            "per diventare Killer e togliere vite."
        ),
        "tutorial_halve_body": (
            "• Parti da 0; ogni round ha un bersaglio (es. 20, QUALSIASI DOUBLE, BULL).\n"
            "• Somma i colpi validi sul bersaglio con 3 freccette "
            "(S/D/T del numero; nei round DOUBLE/TRIPLE — qualsiasi double/triple).\n"
            "• 0 punti nel round → totale dimezzato (verso il basso).\n"
            "• Dopo tutti i round vince il punteggio più alto."
        ),
        "tutorial_halve_opts": (
            "• STANDARD: 20→15 e BULL (7 round).\n"
            "• ESTESO: 20→15, QUALSIASI DOUBLE, 14→7, QUALSIASI TRIPLE, 6→1, BULL."
        ),
        "bg_red": "Rosso",
        "bg_blue": "Blu",
        "bg_green": "Verde",
        "bg_yellow": "Giallo",
        "bg_gray": "Grigio",
        "bg_purple": "Viola",
        "bg_white": "Bianco",
        "bg_black": "Nero",
        "btn_yellow": "Giallo",
        "btn_green": "Verde",
        "btn_blue": "Blu",
        "btn_orange": "Arancione",
        "btn_white": "Bianco",
        "btn_gray": "Grigio",
        "btn_black": "Nero",
        "btn_cyan": "Cyan",
        "btn_pink": "Rosa",
        "winner_title": "VINCITORE!",
        "league_qr_hint": "Scansiona per la lega",
        "change_password": "CAMBIA PASSWORD",
        "change_password_short": "Cambia",
        "password": "Password",
        "application": "Applicazione",
        "pin_enter": "Inserisci password",
        "pin_new": "Nuova password",
        "pin_confirm": "Conferma password",
        "pin_wrong": "Password errata",
        "pin_mismatch": "Le password non coincidono",
        "pin_saved": "Password salvata",
        "pin_cancel": "ANNULLA",
        "pin_ok": "OK",
        "quit_app": "ESCI",
        "quit_confirm": "USCIRE DALL'APP?",
        "power_choose": "—",
        "power_off": "SPEGNI",
        "power_reboot": "RIAVVIA",
        "power_exit": "ESCI",
        "power_off_confirm": "SPEGNERE IL RASPBERRY?",
        "power_reboot_confirm": "RIAVVIARE IL RASPBERRY?",
        "power_exit_confirm": "USCIRE DALL'APP?",
        "report": "REPORT",
        "report_title": "REPORT",
        "report_auto": "Hit automatici",
        "report_manual": "Inseriti a mano",
        "report_corrected": "Hit auto corretti",
        "report_top_numbers": "Numeri più corretti",
        "report_top_games": "Gioco più giocato",
        "report_games": "Partite giocate",
        "report_none": "Nessun dato",
        "report_close": "CHIUDI",
    },
    "es": {
        "settings": "AJUSTES",
        "language": "Idioma",
        "bg_color": "Color de fondo",
        "btn_color": "Color de botones",
        "start_mode": "Inicio del juego",
        "start_mqtt_qr": "MQTT / código QR",
        "start_button": "Botón EMPEZAR",
        "start_always": "Siempre encendido",
        "device": "Dispositivo",
        "league_qr": "QR de liga",
        "league_qr_on": "Activado",
        "league_qr_off": "Desactivado",
        "tip_detection": "Detección de punta",
        "tip_fitline": "FitLine",
        "tip_ml_onnx": "ML (ONNX)",
        "fitline_scale": "Escala FitLine",
        "ml_tip_dataset": "Dataset punta ML",
        "ml_tip_dataset_off": "Desactivado",
        "ml_tip_dataset_on": "Guardar tiros",
        "manual_mode": "Entrada manual",
        "manual_mode_off": "Desactivado",
        "manual_mode_on": "Activado",
        "auto_calibrate": "Calibración auto",
        "auto_calibrate_on": "Activado",
        "auto_calibrate_off": "Desactivado",
        "saving_ref": "GUARDANDO REF...",
        "ref_not_saved": "Referencia no guardada — inténtalo de nuevo",
        "no_saved_cal": "Sin calibración — primero CALIBRACIÓN en ajustes",
        "play_mode_auto": "AUTO",
        "play_mode_manual": "MANUAL",
        "switch_player": "CAMBIAR JUGADOR",
        "calibration": "CALIBRACIÓN",
        "calibrate": "CALIBRAR",
        "calibrating": "CALIBRANDO...",
        "calibrating_hint": "Buscando la diana en las cámaras",
        "starting_game": "INICIANDO PARTIDA",
        "starting_game_hint": "Calibración y preparación",
        "please_wait": "Espera un momento",
        "select_game_sub": "Elige un modo de juego",
        "cal_incomplete": "Calibración incompleta — inténtalo de nuevo",
        "ingame_cal_title": "¿RECALIBRAR?",
        "ingame_cal_s1": "¿Los dardos se detectan mal?",
        "ingame_cal_s2": "Quita los dardos y recalibra la diana para una detección más precisa.",
        "ingame_cal_done": "LISTO",
        "ingame_cal_retry": "CALIBRAR DE NUEVO",
        "click_bull": "BULL",
        "click_bull_hint": "Haz clic aproximadamente en el bull",
        "click_ellipse": "ELIPSE",
        "click_ellipse_hint": "Haz clic en hasta 4 puntos del borde exterior",
        "overlay": "ELIPSE",
        "enter_name": "NOMBRE",
        "name_skip": "SALTAR",
        "name_remove": "QUITAR",
        "back": "ATRÁS",
        "standby_qr": "¡Escanea el QR para empezar!",
        "standby_button": "Pulsa EMPEZAR para comenzar",
        "start_game": "EMPEZAR",
        "select_game": "ELEGIR JUEGO",
        "select_players": "JUGADORES",
        "rules": "REGLAS",
        "continue": "CONTINUAR",
        "tutorial": "TUTORIAL",
        "tutorial_options": "OPCIONES",
        "tutorial_basics": "CÓMO SE JUEGA",
        "tutorial_ex_double_in": "DOUBLE IN\nprimero un doble",
        "tutorial_ex_double_out": "DOUBLE OUT\ncierra con un doble",
        "tutorial_ex_cut_scorer": "MARCAS DE MÁS",
        "tutorial_ex_cut_rivals": "PUNTOS A RIVALES",
        "tutorial_ex_killer_lives": "Vidas: gana el último en pie",
        "tutorial_ex_around_seg": "S / D / T / ANY = zona que cuenta",
        "clear_board_title": "LIMPIA LA DIANA",
        "clear_board_s1": "Quita todos los dardos",
        "clear_board_s2": "antes de empezar",
        "board_empty": "DIANA VACÍA",
        "exit_confirm": "¿SALIR DEL JUEGO?",
        "yes": "SÍ",
        "no": "NO",
        "exit_game": "SALIR",
        "idle_title": "PARTIDA INACTIVA",
        "idle_s1": "La partida lleva 5 minutos inactiva y se va a cerrar.",
        "idle_s2": "La partida se va a cerrar.",
        "idle_quit": "CERRAR",
        "idle_continue": "CONTINUAR",
        "game_mode": "GAME MODE",
        "double_in": "DOUBLE IN",
        "double_out": "DOUBLE OUT",
        "no_limit": "SIN LÍMITE",
        "rounds_20": "20 RONDAS",
        "cricket_standard": "STANDARD",
        "cricket_no_score": "NO SCORE",
        "cricket_cut": "CUT THROAT",
        "killer_lives": "VIDAS",
        "killer_assign": "ASIGNACIÓN DE NÚMEROS",
        "killer_assign_random": "ALEATORIO",
        "killer_assign_throw": "LANZANDO",
        "killer_activation": "ACTIVACIÓN",
        "killer_act_single": "SINGLE",
        "killer_act_double": "DOUBLE",
        "killer_act_triple": "TRIPLE",
        "killer_dead": "MUERTO",
        "killer_bull_prompt": "HAS DADO AL BULL — ELIGE A QUIÉN QUITARLE UNA VIDA",
        "killer_bull_false": "NO HE DADO AL BULL",
        "killer_one_player_impossible": "IMPOSIBLE\nCON 1 JUGADOR",
        "halve_rounds": "NÚMERO DE RONDAS",
        "halve_standard": "STANDARD",
        "halve_extended": "EXTENDIDO",
        "halve_any_double": "CUALQUIER DOBLE",
        "halve_any_triple": "CUALQUIER TRIPLE",
        "around_target_mult": "SEGMENTOS",
        "around_mult_single": "SINGLE",
        "around_mult_double": "DOUBLE",
        "around_mult_triple": "TRIPLE",
        "around_mult_any": "ANY",
        "around_number_order": "ORDEN DE NÚMEROS",
        "around_order_standard": "STANDARD",
        "around_order_random": "ALEATORIO",
        "around_order_reverse": "REVERSE",
        "around_backstep": "PENALIZACIÓN ATRÁS",
        "around_back": "ATRÁS",
        "around_backstep_off": "DESACTIVADO",
        "around_backstep_1": "ATRÁS 1",
        "around_backstep_2": "ATRÁS 2",
        "board_clear_end": "LIMPIA LA DIANA PARA TERMINAR",
        "board_clear": "LIMPIA LA DIANA",
        "board_clean_q": "¿DIANA LIMPIA?",
        "not_advanced_q": "¿NO AVANZÓ?",
        "next_prefix": "Siguiente:",
        "round": "Ronda",
        "bust": "BUST",
        "tutorial_x01_body": (
            "• Empiezas en {start}.\n"
            "• 3 dardos por turno — resta del total.\n"
            "• Gana quien llegue exactamente a 0.\n"
            "• Por debajo de 0 = BUST (turno anulado)."
        ),
        "tutorial_x01_opts": (
            "• DOUBLE IN OFF: puntuación inmediata.\n"
            "• DOUBLE IN ON: primero un doble (o B50).\n"
            "• DOUBLE OUT OFF: cualquier tiro a 0.\n"
            "• DOUBLE OUT ON: cierra con un doble; resto 1 = BUST."
        ),
        "tutorial_cricket_body": (
            "• Cierra 15–20 y bull (3 marcas).\n"
            "• Single=1, double=2, triple=3.\n"
            "• Cierra todo y cumple la condición del modo."
        ),
        "tutorial_cricket_opts": (
            "• SIN LÍMITE / 20 RONDAS — duración.\n"
            "• STANDARD: excedente = puntos para ti.\n"
            "• NO SCORE: solo marcas.\n"
            "• CUT THROAT: excedente a rivales; menos = mejor."
        ),
        "tutorial_around_body": (
            "• Números en orden (luego bull según opciones).\n"
            "• Número + segmento requeridos para avanzar.\n"
            "• Gana quien termine la secuencia primero."
        ),
        "tutorial_around_opts": (
            "• SEGMENTOS: SINGLE / DOUBLE / TRIPLE / ANY.\n"
            "• ORDEN: STANDARD 1→20→bull, REVERSE o ALEATORIO.\n"
            "• ATRÁS: OFF / ATRÁS 1 / ATRÁS 2 si fallas el turno."
        ),
        "tutorial_killer_body": (
            "• Cada uno tiene un número y vidas.\n"
            "• Hazte Killer con número + multiplicador de activación.\n"
            "• ACTIVACIÓN (Single/Double/Triple): debes acertar exactamente ese "
            "segmento para ser Killer y quitar vidas a rivales.\n"
            "• Como Killer apunta a rivales para quitar una vida.\n"
            "• Tu propio número como Killer = pierdes una vida.\n"
            "• Gana el último con vidas."
        ),
        "tutorial_killer_opts": (
            "• VIDAS: 3 / 5 / 7 / 10.\n"
            "• ALEATORIO o LANZANDO — números.\n"
            "• ACTIVACIÓN: SINGLE / DOUBLE / TRIPLE — multiplicador obligatorio "
            "para ser Killer y quitar vidas."
        ),
        "tutorial_halve_body": (
            "• Empiezas en 0; cada ronda tiene un objetivo (p. ej. 20, CUALQUIER DOBLE, BULL).\n"
            "• Suma los aciertos válidos al objetivo con 3 dardos "
            "(S/D/T del número; en rondas DOBLE/TRIPLE — cualquier doble/triple).\n"
            "• 0 puntos en la ronda → total se divide a la mitad (hacia abajo).\n"
            "• Tras todas las rondas gana la puntuación más alta."
        ),
        "tutorial_halve_opts": (
            "• STANDARD: 20→15 y BULL (7 rondas).\n"
            "• EXTENDIDO: 20→15, CUALQUIER DOBLE, 14→7, CUALQUIER TRIPLE, 6→1, BULL."
        ),
        "bg_red": "Rojo",
        "bg_blue": "Azul",
        "bg_green": "Verde",
        "bg_yellow": "Amarillo",
        "bg_gray": "Gris",
        "bg_purple": "Morado",
        "bg_white": "Blanco",
        "bg_black": "Negro",
        "btn_yellow": "Amarillo",
        "btn_green": "Verde",
        "btn_blue": "Azul",
        "btn_orange": "Naranja",
        "btn_white": "Blanco",
        "btn_gray": "Gris",
        "btn_black": "Negro",
        "btn_cyan": "Cian",
        "btn_pink": "Rosa",
        "winner_title": "¡GANADOR!",
        "league_qr_hint": "Escanea para la liga",
        "change_password": "CAMBIAR CONTRASEÑA",
        "change_password_short": "Cambiar",
        "password": "Contraseña",
        "application": "Aplicación",
        "pin_enter": "Introduce la contraseña",
        "pin_new": "Nueva contraseña",
        "pin_confirm": "Confirma la contraseña",
        "pin_wrong": "Contraseña incorrecta",
        "pin_mismatch": "Las contraseñas no coinciden",
        "pin_saved": "Contraseña guardada",
        "pin_cancel": "CANCELAR",
        "pin_ok": "OK",
        "quit_app": "SALIR",
        "quit_confirm": "¿CERRAR LA APLICACIÓN?",
        "power_choose": "—",
        "power_off": "APAGAR",
        "power_reboot": "REINICIAR",
        "power_exit": "SALIR",
        "power_off_confirm": "¿APAGAR EL RASPBERRY?",
        "power_reboot_confirm": "¿REINICIAR EL RASPBERRY?",
        "power_exit_confirm": "¿CERRAR LA APLICACIÓN?",
        "report": "INFORME",
        "report_title": "INFORME",
        "report_auto": "Hits automáticos",
        "report_manual": "Introducido a mano",
        "report_corrected": "Hits auto corregidos",
        "report_top_numbers": "Números más corregidos",
        "report_top_games": "Juego más jugado",
        "report_games": "Partidas jugadas",
        "report_none": "Sin datos",
        "report_close": "CERRAR",
    },
}


@dataclass
class KioskSettings:
    language: str = "hr"
    bg_color: str = "#e02020"
    button_color: str = "#ffe000"
    start_mode: str = "button"
    settings_password: str = "1234"
    # Tip (x,y): "fitline" (default) | "ml_onnx"
    tip_detection: str = "fitline"
    # Legacy (unused): FitLine now always uses fixed FITLINE_SIZE. Kept for JSON compat.
    fitline_scale: float = FITLINE_SCALE_DEFAULT
    # When True, save fused-motion 200×200 PNGs under ml_tip/dataset/images/
    ml_tip_save_dataset: bool = False
    # When True, playing screen keeps keypad always visible (manual scoring).
    manual_mode: bool = False
    # When True, recalibrate board on "PLOČA JE PRAZNA" before each game.
    auto_calibrate: bool = True
    # League QR device id (no pipe). Matches Pi: pikado-1 / pikado-2 / pikado-3.
    device_id: str = "pikado-1"
    # When True, show league QR on the winner / leaderboard screen.
    league_qr_enabled: bool = True

    def t(self, key: str) -> str:
        pack = _I18N.get(self.language) or _I18N["hr"]
        return pack.get(key) or _I18N["hr"].get(key, key)


_settings: KioskSettings | None = None


def settings_path() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, SETTINGS_FILENAME)


def _default_device_id() -> str:
    """pikado/pikado1/cmd → pikado-1; fallback pikado-1."""
    try:
        from mqtt import TOPIC

        for part in str(TOPIC or "").split("/"):
            p = part.strip().lower()
            if not p.startswith("pikado") or p == "pikado":
                continue
            rest = p[len("pikado") :].lstrip("-_")
            if rest.isdigit():
                cand = f"pikado-{rest}"
                if cand in DEVICE_IDS:
                    return cand
    except Exception:
        pass
    return "pikado-1"


def sanitize_device_id(raw: Any) -> str:
    text = str(raw or "").replace("|", "").strip()
    if not text:
        return _default_device_id()
    low = text.lower().replace("_", "-")
    if low in DEVICE_IDS:
        return low
    compact = low.replace("-", "")
    for d in DEVICE_IDS:
        if d.replace("-", "") == compact:
            return d
    return text


def start_qr_image_path(device_id: Any = None) -> str:
    """pikado-1 → assets/qr_start_1.png; fallback qr_start.png."""
    base = os.path.dirname(os.path.abspath(__file__))
    assets = os.path.join(base, "assets")
    if device_id is None:
        device_id = get_settings().device_id
    device = sanitize_device_id(device_id)
    num = ""
    if "-" in str(device):
        tail = str(device).rsplit("-", 1)[-1]
        if tail.isdigit():
            num = tail
    if num:
        numbered = os.path.join(assets, f"qr_start_{num}.png")
        if os.path.isfile(numbered):
            return numbered
    return os.path.join(assets, "qr_start.png")


def load_settings() -> KioskSettings:
    global _settings
    path = settings_path()
    data: Dict[str, Any] = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
        except Exception:
            data = {}
    tip_raw = str(data.get("tip_detection", "fitline") or "fitline").strip().lower()
    ds_raw = data.get("ml_tip_save_dataset", False)
    if isinstance(ds_raw, str):
        ds_save = ds_raw.strip().lower() in ("1", "true", "yes", "on")
    else:
        ds_save = bool(ds_raw)
    mm_raw = data.get("manual_mode", False)
    if isinstance(mm_raw, str):
        manual_on = mm_raw.strip().lower() in ("1", "true", "yes", "on")
    else:
        manual_on = bool(mm_raw)
    ac_raw = data.get("auto_calibrate", True)
    if isinstance(ac_raw, str):
        auto_cal = ac_raw.strip().lower() in ("1", "true", "yes", "on")
    else:
        auto_cal = bool(ac_raw)
    lq_raw = data.get("league_qr_enabled", True)
    if isinstance(lq_raw, str):
        league_qr_on = lq_raw.strip().lower() in ("1", "true", "yes", "on")
    else:
        league_qr_on = bool(lq_raw)
    try:
        fl_scale = float(data.get("fitline_scale", FITLINE_SCALE_DEFAULT))
    except (TypeError, ValueError):
        fl_scale = FITLINE_SCALE_DEFAULT
    if fl_scale not in _FITLINE_SCALE_ALLOWED:
        # Nearest allowed (e.g. legacy / typo).
        fl_scale = min(_FITLINE_SCALE_ALLOWED, key=lambda v: abs(v - fl_scale))
    s = KioskSettings(
        language=str(data.get("language", "hr")),
        bg_color=str(data.get("bg_color", "#e02020")),
        button_color=str(data.get("button_color", "#ffe000")),
        start_mode=str(data.get("start_mode", "button")),
        settings_password=str(data.get("settings_password", "1234") or "1234"),
        tip_detection=tip_raw,
        fitline_scale=float(fl_scale),
        ml_tip_save_dataset=ds_save,
        manual_mode=manual_on,
        auto_calibrate=auto_cal,
        device_id=sanitize_device_id(
            data["device_id"] if "device_id" in data else _default_device_id()
        ),
        league_qr_enabled=league_qr_on,
    )
    if s.language not in _I18N:
        s.language = "hr"
    if s.start_mode not in ("mqtt_qr", "button", "always_on"):
        s.start_mode = "button"
    if not s.device_id:
        s.device_id = _default_device_id()
    # ML tip detection uklonjen iz UI — uvijek FitLine.
    s.tip_detection = "fitline"
    s.ml_tip_save_dataset = False
    if s.fitline_scale not in _FITLINE_SCALE_ALLOWED:
        s.fitline_scale = FITLINE_SCALE_DEFAULT
    # PIN: samo znamenke, 4–8 znakova
    pw = "".join(ch for ch in s.settings_password if ch.isdigit())
    if len(pw) < 4:
        pw = "1234"
    s.settings_password = pw[:8]
    legacy = {
        "#440505": "#e02020",
        "#0a1a3a": "#2060e0",
        "#0a2a14": "#20c040",
        "#2a1040": "#a040e0",
        "#3a2208": "#e0c020",
        "#0a2a2a": "#20c040",
    }
    s.bg_color = legacy.get(s.bg_color.lower(), s.bg_color)
    if not s.bg_color.startswith("#") or len(s.bg_color) != 7:
        s.bg_color = "#e02020"
    if not s.button_color.startswith("#") or len(s.button_color) != 7:
        s.button_color = "#ffe000"
    known_bg = {c.lower() for c, _ in BG_PRESETS}
    if s.bg_color.lower() not in known_bg:
        s.bg_color = "#e02020"
    _settings = s
    return s


def get_settings() -> KioskSettings:
    global _settings
    if _settings is None:
        return load_settings()
    return _settings


def save_settings(s: KioskSettings | None = None) -> None:
    global _settings
    if s is not None:
        _settings = s
    cur = get_settings()
    try:
        with open(settings_path(), "w", encoding="utf-8") as f:
            json.dump(asdict(cur), f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[settings] save fail: {e}", flush=True)


def update_settings(**kwargs: Any) -> KioskSettings:
    s = get_settings()
    for k, v in kwargs.items():
        if hasattr(s, k) and v is not None:
            setattr(s, k, v)
    if "fitline_scale" in kwargs and kwargs["fitline_scale"] is not None:
        try:
            fl = float(s.fitline_scale)
        except (TypeError, ValueError):
            fl = FITLINE_SCALE_DEFAULT
        if fl not in _FITLINE_SCALE_ALLOWED:
            fl = min(_FITLINE_SCALE_ALLOWED, key=lambda v: abs(v - fl))
        s.fitline_scale = float(fl)
    if "settings_password" in kwargs and kwargs["settings_password"] is not None:
        pw = "".join(ch for ch in str(s.settings_password) if ch.isdigit())
        if len(pw) < 4:
            pw = "1234"
        s.settings_password = pw[:8]
    if "device_id" in kwargs and kwargs["device_id"] is not None:
        s.device_id = sanitize_device_id(s.device_id)
    save_settings(s)
    return s


def _parse_hex(color: str) -> Tuple[int, int, int]:
    c = color.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def button_fg(button_color: str) -> str:
    """Boja teksta/ikona na Primary gumbiima (kontrast na button_color)."""
    try:
        r, g, b = _parse_hex(button_color)
    except Exception:
        return "#111111"
    return "#111111" if (r + g + b) > 360 else "#f5f5f5"


def _to_hex(r: int, g: int, b: int) -> str:
    return f"#{max(0, min(255, r)):02x}{max(0, min(255, g)):02x}{max(0, min(255, b)):02x}"


def shade(color: str, factor: float) -> str:
    """factor>1 posvjetli, <1 potamni."""
    r, g, b = _parse_hex(color)
    if factor >= 1.0:
        t = min(1.0, factor - 1.0)
        r = int(r + (255 - r) * t)
        g = int(g + (255 - g) * t)
        b = int(b + (255 - b) * t)
    else:
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)
    return _to_hex(r, g, b)


def build_theme_qss(base_qss: str, *, bg_color: str, button_color: str) -> str:
    """Root: crna do polovice, zatim gradijent u boju; gumbi/tipkovnica po button_color."""
    marker = "QMainWindow#Root, QWidget#Root {"
    idx = base_qss.find(marker)
    if idx >= 0:
        depth = 0
        i = idx
        while i < len(base_qss):
            if base_qss[i] == "{":
                depth += 1
            elif base_qss[i] == "}":
                depth -= 1
                if depth == 0:
                    base_qss = base_qss[:idx] + base_qss[i + 1 :]
                    break
            i += 1

    # Tamniji gradijent: crna drži do ~50%, pa se spušta u odabranu boju.
    bg_soft = shade(bg_color, 0.35)
    bg_end = shade(bg_color, 0.72)
    btn = button_color
    btn_hi = shade(button_color, 1.35)
    btn_lo = shade(button_color, 0.82)
    btn_border = shade(button_color, 0.65)
    br, bg_, bb = _parse_hex(button_color)
    btn_fg = "#111111" if (br + bg_ + bb) > 360 else "#f5f5f5"

    def _btn_block(oid: str, font_px: int) -> str:
        return f"""
QPushButton#{oid} {{
    background-color: {btn};
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {btn_hi}, stop:0.4 {btn}, stop:0.6 {btn}, stop:1 {btn_lo});
    color: {btn_fg};
    border: 3px solid {btn_border};
    border-radius: 12px;
    font-size: {font_px}px;
    font-weight: 800;
    min-height: 0px;
    padding: 2px 10px;
}}
QPushButton#{oid}:hover, QPushButton#{oid}:focus {{
    background-color: {btn};
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {btn_hi}, stop:0.4 {btn}, stop:0.6 {btn}, stop:1 {btn_lo});
}}
QPushButton#{oid}:pressed {{
    background-color: {btn_hi};
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {shade(button_color, 1.5)}, stop:0.5 {btn_hi}, stop:1 {btn});
}}
"""

    theme = f"""
QMainWindow#Root, QWidget#Root {{
    background-color: #000000;
    background: qlineargradient(
        x1: 0, y1: 0, x2: 0, y2: 1,
        stop: 0 #000000,
        stop: 0.50 #000000,
        stop: 0.72 {bg_soft},
        stop: 1 {bg_end}
    );
    color: #f2f2f2;
}}
QPushButton#Primary {{
    background-color: {btn};
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {btn_hi}, stop:0.4 {btn}, stop:0.6 {btn}, stop:1 {btn_lo});
    color: {btn_fg};
    border-color: {btn_border};
    font-size: 68px;
    min-height: 104px;
}}
QPushButton#Primary:hover, QPushButton#Primary:focus {{
    background-color: {btn};
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {btn_hi}, stop:0.4 {btn}, stop:0.6 {btn}, stop:1 {btn_lo});
}}
QPushButton#Primary:pressed {{
    background-color: {btn_hi};
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {shade(button_color, 1.5)}, stop:0.5 {btn_hi}, stop:1 {btn});
}}
{_btn_block("PlayAction", 56)}
{_btn_block("PlayNav", 56)}
{_btn_block("Footer", 66)}
{_btn_block("KeyNum", 60)}
{_btn_block("KeyBull", 54)}
{_btn_block("KeyModOn", 58)}
QPushButton#RuleOn {{
    background-color: {btn};
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {btn_hi}, stop:0.4 {btn}, stop:0.6 {btn}, stop:1 {btn_lo});
    color: {btn_fg};
    border: 3px solid {btn_border};
    border-radius: 12px;
    font-size: 62px;
    font-weight: 800;
    min-height: 110px;
}}
QPushButton#RuleOn:hover, QPushButton#RuleOn:focus {{
    background-color: {btn};
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {btn_hi}, stop:0.4 {btn}, stop:0.6 {btn}, stop:1 {btn_lo});
}}
QPushButton#RuleOn:pressed {{
    background-color: {btn_hi};
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {shade(button_color, 1.5)}, stop:0.5 {btn_hi}, stop:1 {btn});
}}
QPushButton#RuleOff {{
    background-color: #484850;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #606068, stop:0.4 #484850, stop:0.6 #484850, stop:1 #383840);
    color: #ebebeb;
    border: 3px solid #5a5a66;
    border-radius: 12px;
    font-size: 62px;
    font-weight: 800;
    min-height: 110px;
}}
QPushButton#RuleOff:hover, QPushButton#RuleOff:focus {{
    background-color: #484850;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #606068, stop:0.4 #484850, stop:0.6 #484850, stop:1 #383840);
}}
QComboBox {{
    background-color: #2a2a32;
    color: #ffffff;
    border: 3px solid #5a5a66;
    border-radius: 12px;
    padding: 10px 14px;
    font-size: 36px;
    font-weight: 700;
    min-height: 56px;
    max-width: 520px;
}}
QComboBox::drop-down {{
    width: 42px;
    border: none;
}}
QComboBox QAbstractItemView {{
    background-color: #1a1a22;
    color: #ffffff;
    selection-background-color: {btn};
    selection-color: {btn_fg};
    font-size: 32px;
    border: 2px solid #5a5a66;
    padding: 6px;
    min-height: 48px;
}}
QLabel#SettingsSection {{
    color: #c8c8d0;
    font-size: 32px;
    font-weight: 700;
}}
QFrame#Card {{
    background-color: rgba(10, 10, 16, 210);
    border: 2px solid {btn_border};
    border-radius: 24px;
}}
QWidget#LoadingOverlay {{
    background-color: rgba(0, 0, 0, 188);
}}
"""
    # Footer themed block uses Primary button fill — force dark gray border back for gray Footer.
    theme += """
QPushButton#Footer {
    border-color: #5a5a66;
}
"""
    return base_qss + "\n" + theme
