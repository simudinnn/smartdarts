"""Qt kiosk UI — OpenCV stays in KioskSession for cameras/detection."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import (
    Qt,
    QTimer,
    QEvent,
    Signal,
    QObject,
    QPropertyAnimation,
    QVariantAnimation,
    QEasingCurve,
    QRect,
    QPoint,
    QSequentialAnimationGroup,
    QParallelAnimationGroup,
)
from PySide6.QtGui import (
    QImage,
    QPixmap,
    QKeySequence,
    QShortcut,
    QFont,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QColor,
    QIcon,
)
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QFrame,
    QSizePolicy,
    QComboBox,
    QSplashScreen,
    QMenu,
    QWidgetAction,
    QGraphicsOpacityEffect,
    QGraphicsColorizeEffect,
    QScrollArea,
)


def _detect_weak_hw() -> bool:
    """Raspberry Pi / ARM kiosk: prefer lighter UI path (desktop stays rich)."""
    env = (os.environ.get("SMARTDARTS_WEAK_HW") or "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    mach = (platform.machine() or "").lower()
    if mach.startswith(("arm", "aarch")):
        return True
    try:
        with open("/proc/device-tree/model", "r", encoding="utf-8", errors="ignore") as f:
            model = f.read().lower()
        if "raspberry pi" in model:
            return True
    except OSError:
        pass
    return False


# Cached once at import — env SMARTDARTS_WEAK_HW=0/1 overrides auto-detect.
_WEAK_HW = _detect_weak_hw()
# UI poll stays snappy; OpenCV detect/cal runs on a background thread.
_UI_TICK_MS = 40 if _WEAK_HW else 33
_SCORE_ANIM_MS = 32
_PIXMAP_TRANSFORM = Qt.FastTransformation if _WEAK_HW else Qt.SmoothTransformation
_TURN_ANIM_MS = 320 if _WEAK_HW else 420
_IDLE_WARN_SEC = 300.0
_IDLE_EXIT_SEC = 60.0
_IDLE_SCREENS = (
    "playing",
    "select_game",
    "select_players",
    "clear_board",
    "exit_confirm",
    "rules_x01",
    "rules_cricket",
    "rules_killer",
    "rules_around",
    "rules_halve",
    "tutorial_x01",
    "tutorial_cricket",
    "tutorial_killer",
    "tutorial_around",
    "tutorial_halve",
)


class SettingsChoiceButton(QWidget):
    """Cijeli gumb klikabilan; sadržaj centriran; izbornik desno."""

    activated = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("SettingsChoice")
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.NoFocus)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._items: List[Tuple[str, Optional[QIcon], object]] = []
        self._index = 0
        self._icon_size = QPixmap(48, 28).size()
        self.setMinimumHeight(64)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 8, 14, 8)
        lay.setSpacing(12)
        lay.addStretch(1)
        self._icon_lbl = QLabel()
        self._icon_lbl.setAlignment(Qt.AlignCenter)
        self._icon_lbl.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        lay.addWidget(self._icon_lbl, 0)
        self._text_lbl = QLabel()
        self._text_lbl.setAlignment(Qt.AlignCenter)
        self._text_lbl.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._text_lbl.setStyleSheet(
            "color: #ffffff; font-size: 32px; font-weight: 700; background: transparent;"
        )
        lay.addWidget(self._text_lbl, 0)
        lay.addStretch(1)

        self.setStyleSheet(
            "QWidget#SettingsChoice {"
            "  background-color: #2a2a32;"
            "  color: #ffffff;"
            "  border: 3px solid #5a5a66;"
            "  border-radius: 12px;"
            "}"
            "QWidget#SettingsChoice:hover {"
            "  background-color: #32323a;"
            "}"
        )

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._open_menu()
            event.accept()
            return
        super().mousePressEvent(event)

    def set_icon_size(self, size) -> None:
        self._icon_size = size

    def clear_items(self) -> None:
        self._items.clear()
        self._index = 0
        self._refresh()

    def add_item(self, text: str, data: object, icon: Optional[QIcon] = None) -> None:
        self._items.append((str(text), icon, data))
        if len(self._items) == 1:
            self._refresh()

    def count(self) -> int:
        return len(self._items)

    def item_data(self, index: int) -> object:
        if 0 <= index < len(self._items):
            return self._items[index][2]
        return None

    def current_data(self) -> object:
        if 0 <= self._index < len(self._items):
            return self._items[self._index][2]
        return None

    def currentData(self) -> object:
        return self.current_data()

    def set_current_data(self, value: object) -> None:
        target = str(value).lower() if value is not None else ""
        for i, (_t, _ic, data) in enumerate(self._items):
            if str(data).lower() == target:
                self._index = i
                self._refresh()
                return

    def set_item_text(self, index: int, text: str) -> None:
        if 0 <= index < len(self._items):
            _t, ic, data = self._items[index]
            self._items[index] = (str(text), ic, data)
            if index == self._index:
                self._refresh()

    def _refresh(self) -> None:
        if not self._items:
            self._icon_lbl.hide()
            self._text_lbl.hide()
            return
        text, icon, _data = self._items[self._index]
        if icon is not None and not icon.isNull():
            self._icon_lbl.setPixmap(icon.pixmap(self._icon_size))
            self._icon_lbl.show()
        else:
            self._icon_lbl.clear()
            self._icon_lbl.hide()
        if text:
            self._text_lbl.setText(text)
            self._text_lbl.show()
        else:
            self._text_lbl.clear()
            self._text_lbl.hide()

    def _open_menu(self) -> None:
        if not self._items:
            return
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu {"
            "  background-color: #1a1a22;"
            "  color: #ffffff;"
            "  border: 3px solid #5a5a66;"
            "  font-size: 40px;"
            "  font-weight: 700;"
            "  padding: 8px;"
            "  min-width: 420px;"
            "}"
            "QMenu::item {"
            "  padding: 4px;"
            "  border-radius: 8px;"
            "}"
            "QMenu::item:selected {"
            "  background-color: #3a3a44;"
            "}"
        )
        for i, (text, icon, _data) in enumerate(self._items):
            if _data == "" or _data is None:
                continue
            row = QWidget()
            row.setStyleSheet("background: transparent;")
            lay = QHBoxLayout(row)
            lay.setContentsMargins(14, 12, 14, 12)
            lay.setSpacing(14)
            lay.addStretch(1)
            if icon is not None and not icon.isNull():
                if text:
                    pix = icon.pixmap(96, 56)
                else:
                    pix = icon.pixmap(340, 48)
                ic = QLabel()
                ic.setPixmap(pix)
                ic.setAlignment(Qt.AlignCenter)
                lay.addWidget(ic, 0)
            if text:
                tl = QLabel(text)
                tl.setAlignment(Qt.AlignCenter)
                tl.setStyleSheet(
                    "color: #ffffff; font-size: 36px; font-weight: 700; background: transparent;"
                )
                lay.addWidget(tl, 0)
            lay.addStretch(1)
            act = QWidgetAction(menu)
            act.setDefaultWidget(row)
            act.setData(i)
            menu.addAction(act)
        pos = self.mapToGlobal(self.rect().topRight())
        pos.setX(pos.x() + 6)
        chosen = menu.exec(pos)
        if chosen is None:
            return
        idx = int(chosen.data())
        if 0 <= idx < len(self._items) and idx != self._index:
            self._index = idx
            self._refresh()
            self.activated.emit(self.current_data())


from kiosk_session import KioskSession
from calibration import CAMERA_INDICES
from kiosk_settings import (
    AUTO_CALIBRATE_OPTIONS,
    BG_PRESETS,
    BTN_PRESETS,
    DEVICE_IDS,
    LANG_OPTIONS,
    LEAGUE_QR_OPTIONS,
    POWER_OPTIONS,
    START_MODES,
    build_theme_qss,
    button_fg,
    get_settings,
    update_settings,
)
import main_manual as mm


def _metal_bg(mid: str, hi: str, lo: str) -> str:
    """Slabi vertikalni metalik gradient — mid je ista boja gumba."""
    return (
        f"background-color: {mid}; "
        f"background: qlineargradient(x1:0, y1:0, x2:0, y2:1, "
        f"stop:0 {hi}, stop:0.4 {mid}, stop:0.6 {mid}, stop:1 {lo});"
    )


APP_QSS = """
QMainWindow#Root, QWidget#Root {
    background-color: #000000;
    background: qlineargradient(
        x1: 0, y1: 0, x2: 0, y2: 1,
        stop: 0 #000000,
        stop: 0.55 #290e0e,
        stop: 1 #440505
    );
    color: #f2f2f2;
    font-family: "Segoe UI", "Montserrat", "DejaVu Sans", "Arial";
    font-size: 56px;
}
QStackedWidget, QStackedWidget > QWidget {
    background: transparent;
}
QLabel#Wallpaper {
    background: transparent;
    border: none;
}
QPushButton {
    background-color: #3a3a42;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #54545e, stop:0.42 #3a3a42, stop:0.58 #3a3a42, stop:1 #2c2c34);
    color: #ffffff;
    border: 3px solid #5a5a66;
    border-radius: 14px;
    padding: 18px 20px;
    font-size: 64px;
    font-weight: 800;
    min-height: 96px;
}
/* Touchscreen: :hover mora biti isti kao normal — kursor ostaje na zadnjem dodiru */
QPushButton:hover {
    background-color: #3a3a42;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #54545e, stop:0.42 #3a3a42, stop:0.58 #3a3a42, stop:1 #2c2c34);
}
QPushButton:focus { outline: none; }
QPushButton:pressed {
    background-color: #1e1e24;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #2a2a32, stop:0.5 #1e1e24, stop:1 #16161c);
}
QPushButton:disabled {
    background-color: #222228;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #2c2c34, stop:0.5 #222228, stop:1 #1a1a20);
    color: #666670;
    border-color: #3a3a40;
}
QPushButton#Primary {
    background-color: #ffe000;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #fff36a, stop:0.4 #ffe000, stop:0.6 #ffe000, stop:1 #e0c400);
    color: #111111;
    border-color: #c9b000;
    font-size: 68px;
    min-height: 104px;
}
QPushButton#Primary:hover, QPushButton#Primary:focus {
    background-color: #ffe000;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #fff36a, stop:0.4 #ffe000, stop:0.6 #ffe000, stop:1 #e0c400);
}
QPushButton#Primary:pressed {
    background-color: #fff066;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #fff9a0, stop:0.5 #fff066, stop:1 #ffe000);
}
QPushButton#Primary:disabled {
    background-color: #222228;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #2c2c34, stop:0.5 #222228, stop:1 #1a1a20);
    color: #666670;
    border-color: #3a3a40;
}
QPushButton#NavArrow {
    font-size: 48px;
    font-weight: 900;
    min-height: 72px;
    max-height: 72px;
    min-width: 72px;
    max-width: 72px;
    padding: 0px;
    border-radius: 12px;
}
QPushButton#NavArrow:hover, QPushButton#NavArrow:focus {
    background-color: #3a3a42;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #54545e, stop:0.42 #3a3a42, stop:0.58 #3a3a42, stop:1 #2c2c34);
}
QPushButton#Danger {
    background-color: #e04040;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #f06060, stop:0.4 #e04040, stop:0.6 #e04040, stop:1 #c03030);
    color: #111111;
    border-color: #a02828;
    font-size: 66px;
    min-height: 100px;
}
QPushButton#Danger:hover, QPushButton#Danger:focus {
    background-color: #e04040;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #f06060, stop:0.4 #e04040, stop:0.6 #e04040, stop:1 #c03030);
}
QPushButton#Accent {
    background-color: #36d700;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #5ae820, stop:0.4 #36d700, stop:0.6 #36d700, stop:1 #2ab000);
    color: #111111;
    border-color: #289f00;
    font-size: 66px;
    min-height: 104px;
}
QPushButton#Accent:hover, QPushButton#Accent:focus {
    background-color: #36d700;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #5ae820, stop:0.4 #36d700, stop:0.6 #36d700, stop:1 #2ab000);
}
QPushButton#Accent:pressed {
    background-color: #4ae810;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #6ef030, stop:0.5 #4ae810, stop:1 #36d700);
}
QPushButton#Skip {
    background-color: #4a4a55;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #626270, stop:0.4 #4a4a55, stop:0.6 #4a4a55, stop:1 #3a3a44);
    color: #f5f5f5;
    border-color: #7a7a88;
    font-size: 66px;
    min-height: 104px;
}
QPushButton#Skip:hover, QPushButton#Skip:focus {
    background-color: #4a4a55;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #626270, stop:0.4 #4a4a55, stop:0.6 #4a4a55, stop:1 #3a3a44);
}
QPushButton#Footer {
    font-size: 66px;
    font-weight: 800;
    min-height: 104px;
    padding: 18px 20px;
    border-color: #5a5a66;
}
QPushButton#Footer:hover, QPushButton#Footer:focus {
    background-color: #3a3a42;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #54545e, stop:0.42 #3a3a42, stop:0.58 #3a3a42, stop:1 #2c2c34);
}
/* Play gumbi: mali padding — inače setFixedHeight reže donji rub bordera */
QPushButton#PlayAction {
    background-color: #2a2830;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #3e3c48, stop:0.4 #2a2830, stop:0.6 #2a2830, stop:1 #1e1c24);
    color: #ffe14a;
    border: 3px solid #c8b84a;
    border-radius: 12px;
    font-size: 56px;
    font-weight: 800;
    min-height: 0px;
    padding: 2px 10px;
}
QPushButton#PlayAction:hover, QPushButton#PlayAction:focus {
    background-color: #2a2830;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #3e3c48, stop:0.4 #2a2830, stop:0.6 #2a2830, stop:1 #1e1c24);
}
QPushButton#PlayAction:pressed {
    background-color: #3a3830;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #4a4838, stop:0.5 #3a3830, stop:1 #2a2830);
}
QPushButton#PlayActionAccent {
    background-color: #36d700;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #5ae820, stop:0.4 #36d700, stop:0.6 #36d700, stop:1 #2ab000);
    color: #111111;
    border: 3px solid #289f00;
    border-radius: 10px;
    font-size: 56px;
    font-weight: 800;
    min-height: 0px;
    padding: 2px 10px;
}
QPushButton#PlayActionAccent:hover, QPushButton#PlayActionAccent:focus {
    background-color: #36d700;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #5ae820, stop:0.4 #36d700, stop:0.6 #36d700, stop:1 #2ab000);
}
QPushButton#PlayActionAccent:pressed {
    background-color: #4ae810;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #6ef030, stop:0.5 #4ae810, stop:1 #36d700);
}
QPushButton#PlayActionWarn {
    background-color: #ff5d00;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #ff7a33, stop:0.4 #ff5d00, stop:0.6 #ff5d00, stop:1 #e05400);
    color: #111111;
    border: 3px solid #cc4a00;
    border-radius: 10px;
    font-size: 56px;
    font-weight: 800;
    min-height: 0px;
    padding: 2px 10px;
}
QPushButton#PlayActionWarn:hover, QPushButton#PlayActionWarn:focus {
    background-color: #ff5d00;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #ff7a33, stop:0.4 #ff5d00, stop:0.6 #ff5d00, stop:1 #e05400);
}
QPushButton#PlayActionWarn:pressed {
    background-color: #ff7a33;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #ff9a5a, stop:0.5 #ff7a33, stop:1 #ff5d00);
}
QPushButton#PlayNav {
    background-color: #3a3a42;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #54545e, stop:0.42 #3a3a42, stop:0.58 #3a3a42, stop:1 #2c2c34);
    color: #ffffff;
    border: 3px solid #5a5a66;
    border-radius: 10px;
    font-size: 46px;
    font-weight: 800;
    min-height: 0px;
    padding: 2px 10px;
}
QPushButton#PlayNav:hover, QPushButton#PlayNav:focus {
    background-color: #3a3a42;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #54545e, stop:0.42 #3a3a42, stop:0.58 #3a3a42, stop:1 #2c2c34);
}
QPushButton#PlayNav:pressed {
    background-color: #505058;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #686870, stop:0.5 #505058, stop:1 #3a3a42);
}
QPushButton#PlayExit {
    background-color: #e04040;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #f06060, stop:0.4 #e04040, stop:0.6 #e04040, stop:1 #c03030);
    color: #111111;
    border: 3px solid #a02828;
    border-radius: 10px;
    font-size: 46px;
    font-weight: 800;
    min-height: 0px;
    padding: 2px 10px;
}
QPushButton#PlayExit:hover, QPushButton#PlayExit:focus {
    background-color: #e04040;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #f06060, stop:0.4 #e04040, stop:0.6 #e04040, stop:1 #c03030);
}
QPushButton#PlayExit:pressed {
    background-color: #ff6060;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #ff8080, stop:0.5 #ff6060, stop:1 #e04040);
}
QPushButton#PlayExitWide {
    background-color: #ffe000;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #fff36a, stop:0.4 #ffe000, stop:0.6 #ffe000, stop:1 #e0c400);
    color: #111111;
    border: 3px solid #c9b000;
    border-radius: 12px;
    font-size: 56px;
    font-weight: 800;
    min-height: 0px;
    padding: 4px 16px;
}
QPushButton#PlayExitWide:hover, QPushButton#PlayExitWide:focus {
    background-color: #ffe000;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #fff36a, stop:0.4 #ffe000, stop:0.6 #ffe000, stop:1 #e0c400);
}
QPushButton#PlayExitWide:pressed {
    background-color: #fff066;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #fff9a0, stop:0.5 #fff066, stop:1 #ffe000);
}
QPushButton#RuleOn {
    background-color: #ffe000;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #fff36a, stop:0.4 #ffe000, stop:0.6 #ffe000, stop:1 #e0c400);
    color: #111111;
    font-size: 62px;
    min-height: 110px;
}
QPushButton#RuleOn:hover, QPushButton#RuleOn:focus {
    background-color: #ffe000;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #fff36a, stop:0.4 #ffe000, stop:0.6 #ffe000, stop:1 #e0c400);
}
QPushButton#RuleOff {
    background-color: #484850;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #606068, stop:0.4 #484850, stop:0.6 #484850, stop:1 #383840);
    color: #ebebeb;
    border-color: #5a5a66;
    font-size: 62px;
    min-height: 110px;
}
QPushButton#RuleOff:hover, QPushButton#RuleOff:focus {
    background-color: #484850;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #606068, stop:0.4 #484850, stop:0.6 #484850, stop:1 #383840);
}
QPushButton#HitSlot {
    background-color: #1c1c22;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #2c2c34, stop:0.4 #1c1c22, stop:0.6 #1c1c22, stop:1 #141418);
    border: 4px solid #5a5a68;
    font-size: 128px;
    font-weight: 800;
    min-height: 72px;
    border-radius: 16px;
    padding: 6px;
}
QPushButton#HitSlot:hover, QPushButton#HitSlot:focus {
    background-color: #1c1c22;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #2c2c34, stop:0.4 #1c1c22, stop:0.6 #1c1c22, stop:1 #141418);
}
QPushButton#HitSlot:pressed {
    background-color: #2a2a32;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #3a3a44, stop:0.5 #2a2a32, stop:1 #1c1c22);
}
QPushButton#KeyNum,
QPushButton#KeyMod,
QPushButton#KeyModOn,
QPushButton#KeyBull,
QPushButton#KeyMiss,
QPushButton#KeyClr,
QPushButton#KeyClose {
    border-radius: 12px;
    font-weight: 800;
    padding: 2px;
    min-height: 0px;
}
QPushButton#KeyNum {
    background-color: #ffe000;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #fff36a, stop:0.4 #ffe000, stop:0.6 #ffe000, stop:1 #e0c400);
    color: #111111;
    border: 3px solid #c9b000;
    font-size: 60px;
}
QPushButton#KeyNum:hover, QPushButton#KeyNum:focus {
    background-color: #ffe000;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #fff36a, stop:0.4 #ffe000, stop:0.6 #ffe000, stop:1 #e0c400);
}
QPushButton#KeyNum:pressed {
    background-color: #fff066;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #fff9a0, stop:0.5 #fff066, stop:1 #ffe000);
}
QPushButton#KeyMod {
    background-color: #484850;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #606068, stop:0.4 #484850, stop:0.6 #484850, stop:1 #383840);
    color: #ebebeb;
    border: 3px solid #7a7a88;
    font-size: 58px;
}
QPushButton#KeyMod:hover, QPushButton#KeyMod:focus {
    background-color: #484850;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #606068, stop:0.4 #484850, stop:0.6 #484850, stop:1 #383840);
}
QPushButton#KeyMod:pressed {
    background-color: #606070;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #787888, stop:0.5 #606070, stop:1 #484850);
}
QPushButton#KeyModOn {
    background-color: #ffe000;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #fff36a, stop:0.4 #ffe000, stop:0.6 #ffe000, stop:1 #e0c400);
    color: #111111;
    border: 3px solid #c9b000;
    font-size: 58px;
}
QPushButton#KeyModOn:hover, QPushButton#KeyModOn:focus {
    background-color: #ffe000;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #fff36a, stop:0.4 #ffe000, stop:0.6 #ffe000, stop:1 #e0c400);
}
QPushButton#KeyModOn:pressed {
    background-color: #fff066;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #fff9a0, stop:0.5 #fff066, stop:1 #ffe000);
}
QPushButton#KeyBull {
    background-color: #ffe000;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #fff36a, stop:0.4 #ffe000, stop:0.6 #ffe000, stop:1 #e0c400);
    color: #111111;
    border: 3px solid #c9b000;
    font-size: 54px;
}
QPushButton#KeyBull:hover, QPushButton#KeyBull:focus {
    background-color: #ffe000;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #fff36a, stop:0.4 #ffe000, stop:0.6 #ffe000, stop:1 #e0c400);
}
QPushButton#KeyBull:pressed {
    background-color: #fff066;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #fff9a0, stop:0.5 #fff066, stop:1 #ffe000);
}
QPushButton#KeyMiss {
    background-color: #e04040;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #f06060, stop:0.4 #e04040, stop:0.6 #e04040, stop:1 #c03030);
    color: #111111;
    border: 3px solid #a02828;
    font-size: 52px;
}
QPushButton#KeyMiss:hover, QPushButton#KeyMiss:focus {
    background-color: #e04040;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #f06060, stop:0.4 #e04040, stop:0.6 #e04040, stop:1 #c03030);
}
QPushButton#KeyMiss:pressed {
    background-color: #ff6060;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #ff8080, stop:0.5 #ff6060, stop:1 #e04040);
}
QPushButton#KeyClr {
    background-color: #3a3a42;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #54545e, stop:0.42 #3a3a42, stop:0.58 #3a3a42, stop:1 #2c2c34);
    color: #ffffff;
    border: 3px solid #5a5a66;
    font-size: 54px;
}
QPushButton#KeyClr:hover, QPushButton#KeyClr:focus {
    background-color: #3a3a42;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #54545e, stop:0.42 #3a3a42, stop:0.58 #3a3a42, stop:1 #2c2c34);
}
QPushButton#KeyClr:pressed {
    background-color: #505058;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #686870, stop:0.5 #505058, stop:1 #3a3a42);
}
QPushButton#KeyClose {
    background-color: #e04040;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #f06060, stop:0.4 #e04040, stop:0.6 #e04040, stop:1 #c03030);
    color: #111111;
    border: 3px solid #a02828;
    font-size: 54px;
}
QPushButton#KeyClose:hover, QPushButton#KeyClose:focus {
    background-color: #e04040;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #f06060, stop:0.4 #e04040, stop:0.6 #e04040, stop:1 #c03030);
}
QPushButton#KeyClose:pressed {
    background-color: #ff6060;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #ff8080, stop:0.5 #ff6060, stop:1 #e04040);
}
QLabel#StatusBanner {
    font-size: 56px;
    font-weight: 800;
    color: #ffe14a;
    padding: 2px 12px;
    background-color: #2a2830;
    border: 3px solid #c8b84a;
    border-radius: 12px;
    min-height: 0px;
}
QLabel#WinnerTitle {
    font-size: 96px;
    font-weight: 800;
    color: #ffffff;
}
QLabel#WinnerSub {
    font-size: 64px;
    font-weight: 700;
    color: #d0d0d8;
}
QLabel#Title {
    font-size: 93px;
    font-weight: 800;
    color: #ffffff;
}
QLabel#SettingsTitle {
    font-size: 56px;
    font-weight: 800;
    color: #ffffff;
    padding: 0 0 6px 0;
}
QLabel#SettingsSection {
    font-size: 30px;
    font-weight: 700;
    color: #d0d0d8;
}
QLabel#Subtitle {
    font-size: 51px;
    color: #b0b0b0;
}
QLabel#PlayTitle {
    font-size: 51px;
    font-weight: 800;
    color: #ffffff;
}
QLabel#ScoreLine {
    font-size: 92px;
    font-weight: 800;
    background: transparent;
    border: none;
}
QLabel#LiveScore {
    font-size: 76px;
    font-weight: 800;
    color: #00e86a;
    min-height: 64px;
    max-height: 96px;
}
QWidget#KillerNumbersLine {
    min-height: 64px;
    max-height: 96px;
    background: transparent;
}
QLabel#KillerLivesRoster {
    font-size: 42px;
    font-weight: 800;
    border: none;
    background: transparent;
}
QFrame#ScoreSuggestSep {
    background-color: #4a4a58;
    border: none;
    max-height: 3px;
    min-height: 3px;
}
QLabel#SectionLabel {
    font-size: 49px;
    font-weight: 700;
    color: #a8a8b0;
}
QLabel#TutorialBody {
    font-size: 38px;
    font-weight: 600;
    color: #e8e8f0;
    line-height: 1.3;
}
QLabel#TutorialSection {
    font-size: 44px;
    font-weight: 800;
    color: #ffe08a;
    padding-top: 10px;
    padding-bottom: 4px;
}
QLabel#TutorialExampleCap {
    font-size: 32px;
    font-weight: 700;
    color: #c8c8d0;
}
QFrame#TutorialExample {
    background-color: rgba(20, 20, 28, 180);
    border: 2px solid #3a3a48;
    border-radius: 14px;
}
QScrollArea#TutorialScroll {
    background: transparent;
    border: none;
}
QScrollArea#TutorialScroll > QWidget > QWidget {
    background: transparent;
}
QLabel#ClearHero {
    font-size: 98px;
    font-weight: 800;
    color: #ffffff;
}
QLabel#ClearSub {
    font-size: 58px;
    font-weight: 600;
    color: #c8c8d0;
}
QWidget#InGameCalOverlay {
    background-color: rgba(0, 0, 0, 210);
}
QLabel#InGameCalStill {
    background-color: #0a0a0e;
    border: 3px solid #5a5a66;
    border-radius: 12px;
    color: #888888;
    font-size: 28px;
    font-weight: 700;
}
QLabel#InGameCalCamName {
    font-size: 28px;
    font-weight: 700;
    color: #d0d0d8;
}
QLabel#RoundLabel {
    font-size: 42px;
    font-weight: 700;
    color: #c8c8d0;
}
QFrame#Card {
    background-color: #121a2a;
    border: 4px solid #ffe000;
    border-radius: 20px;
}
QFrame#QrFrame {
    background-color: #ffffff;
    border: 8px solid #ffffff;
    border-radius: 28px;
}
"""


class AnimButton(QPushButton):
    """Gumb s kratkim click flashom (tipkovnica / touch)."""

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(text, parent)
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._end_flash)
        self._flash_prev = ""

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._start_flash()
        super().mousePressEvent(event)

    def _start_flash(self) -> None:
        role = self.objectName() or ""
        if not role.startswith("Key"):
            return
        if self._flash_timer.isActive():
            return
        self._flash_prev = self.styleSheet()
        # Svjetliji overlay dok je pritisnuto
        self.setStyleSheet(
            self._flash_prev
            + f"\nQPushButton#{role} {{ border-color: #ffffff; }}"
        )
        self._flash_timer.start(120)

    def _end_flash(self) -> None:
        if self._flash_prev is not None:
            self.setStyleSheet(self._flash_prev)
            self._flash_prev = ""


def _rounded_pixmap(pix: QPixmap, radius: int) -> QPixmap:
    """QR / logo: zaobljeni rubovi na samoj slici."""
    if pix.isNull() or radius <= 0:
        return pix
    out = QPixmap(pix.size())
    out.fill(QColor(0, 0, 0, 0))
    painter = QPainter(out)
    painter.setRenderHint(QPainter.Antialiasing, True)
    path = QPainterPath()
    path.addRoundedRect(0.0, 0.0, float(pix.width()), float(pix.height()), float(radius), float(radius))
    painter.setClipPath(path)
    painter.drawPixmap(0, 0, pix)
    painter.end()
    return out


def _asset_path(*parts: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def _logo_pixmap(height: int) -> Optional[QPixmap]:
    path = _asset_path("assets", "logo.png")
    if not os.path.isfile(path):
        return None
    pix = QPixmap(path)
    if pix.isNull():
        return None
    return pix.scaledToHeight(max(1, int(height)), _PIXMAP_TRANSFORM)


def _icon_pixmap(filename: str, height: int) -> Optional[QPixmap]:
    path = _asset_path("assets", filename)
    if not os.path.isfile(path):
        return None
    pix = QPixmap(path)
    if pix.isNull():
        return None
    # Fit inside a square so icon rings / frames stay centered (no wide overflow).
    h = max(1, int(height))
    return pix.scaled(h, h, Qt.KeepAspectRatio, _PIXMAP_TRANSFORM)


_TINT_CACHE: Dict[Tuple, QPixmap] = {}


def _tint_pixmap(pix: QPixmap, color_hex: str) -> QPixmap:
    """Oboji ikonu u boju igrača (zadrži alpha)."""
    if pix.isNull():
        return pix
    c = QColor(color_hex)
    if not c.isValid():
        c = QColor("#ffffff")
    out = QPixmap(pix.size())
    out.fill(QColor(0, 0, 0, 0))
    p = QPainter(out)
    p.setRenderHint(QPainter.SmoothPixmapTransform, True)
    p.drawPixmap(0, 0, pix)
    p.setCompositionMode(QPainter.CompositionMode_SourceIn)
    p.fillRect(out.rect(), c)
    p.end()
    return out


def _pixmap_matte_black(pix: QPixmap) -> QPixmap:
    """Pretvori gotovo-crnu pozadinu u transparentnu (killer.png itd.)."""
    if pix.isNull():
        return pix
    from PySide6.QtGui import QImage

    img = pix.toImage().convertToFormat(QImage.Format_ARGB32)
    for y in range(img.height()):
        for x in range(img.width()):
            c = img.pixelColor(x, y)
            if c.red() + c.green() + c.blue() < 48:
                c.setAlpha(0)
                img.setPixelColor(x, y, c)
    return QPixmap.fromImage(img)


def _heart_pixmap(size: int, color_hex: str) -> QPixmap:
    """heart.png obojen bojom igrača (crna pozadina → transparentno)."""
    s = max(12, int(size))
    fg = color_hex if color_hex else "#ff4040"
    key = ("heart.png", s, fg.lower())
    cached = _TINT_CACHE.get(key)
    if cached is not None and not cached.isNull():
        return cached
    base = _icon_pixmap("heart.png", s)
    if base is not None and not base.isNull():
        out = _tint_pixmap(_pixmap_matte_black(base), fg)
        _TINT_CACHE[key] = out
        return out
    pix = QPixmap(s, s)
    pix.fill(QColor(0, 0, 0, 0))
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing, True)
    c = QColor(fg)
    if not c.isValid():
        c = QColor("#ff4040")
    p.setBrush(c)
    p.setPen(Qt.NoPen)
    r = s * 0.28
    cx1, cx2 = s * 0.32, s * 0.68
    cy = s * 0.38
    p.drawEllipse(QPoint(int(cx1), int(cy)), int(r), int(r))
    p.drawEllipse(QPoint(int(cx2), int(cy)), int(r), int(r))
    path = QPainterPath()
    path.moveTo(s * 0.12, s * 0.40)
    path.lineTo(s * 0.50, s * 0.92)
    path.lineTo(s * 0.88, s * 0.40)
    path.closeSubpath()
    p.drawPath(path)
    p.end()
    _TINT_CACHE[key] = pix
    return pix


def _is_near_black(color_hex: str) -> bool:
    c = QColor(color_hex)
    if not c.isValid():
        return False
    return (c.red() + c.green() + c.blue()) < 60


def _player_fg(color_hex: str) -> str:
    """Readable tint/label color on dark UI (black player > white)."""
    if _is_near_black(color_hex):
        return "#ffffff"
    c = QColor(color_hex)
    return color_hex if c.isValid() else "#ffffff"


def _icon_with_strike(
    pix: QPixmap,
    strike_hex: str = "#8b0000",
    *,
    thin: bool = False,
) -> QPixmap:
    """X (two diagonals) over icon for eliminated / MRTAV players."""
    if pix is None or pix.isNull():
        return pix
    out = QPixmap(pix.size())
    out.fill(QColor(0, 0, 0, 0))
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.drawPixmap(0, 0, pix)
    from PySide6.QtGui import QPen

    pen = QPen(QColor(strike_hex))
    if thin:
        # Thinner / more inset so initials stay readable under the X.
        pen.setWidth(max(3, pix.width() // 14))
        m = max(8, pix.width() // 6)
    else:
        pen.setWidth(max(6, pix.width() // 6))
        m = max(4, pix.width() // 12)
    pen.setCapStyle(Qt.RoundCap)
    p.setPen(pen)
    w, h = pix.width(), pix.height()
    p.drawLine(m, m, w - m, h - m)
    p.drawLine(w - m, m, m, h - m)
    p.end()
    return out


def _player_icon_tinted(height: int, color_hex: str) -> Optional[QPixmap]:
    key = ("player_icon.png", int(height), color_hex.lower())
    cached = _TINT_CACHE.get(key)
    if cached is not None and not cached.isNull():
        return cached
    base = _icon_pixmap("player_icon.png", height)
    if base is None:
        return None
    out = _tint_pixmap(base, color_hex)
    _TINT_CACHE[key] = out
    return out


def _calib_short_label() -> str:
    raw = get_settings().t("calibration") or ""
    letters = "".join(ch for ch in raw.upper() if ch.isalpha())
    return letters[:5] or "CALIB"


def _name_initials(name: str) -> str:
    s = str(name or "").strip()
    up = s.upper()
    if up.startswith("PLAYER"):
        rest = up[6:].strip()
        if rest.isdigit():
            return f"P{int(rest)}"
    letters = "".join(ch for ch in s if ch.isalpha())
    return letters[:2].upper()


def _is_placeholder_player_name(name: str) -> bool:
    """Automatski PLAYER 1–4 — uređivanje kreće od praznog polja."""
    s = str(name or "").strip().upper()
    if not s.startswith("PLAYER"):
        return False
    rest = s[6:].strip()
    return rest.isdigit() and 1 <= int(rest) <= 4


def _display_player_name(name: str) -> str:
    return "".join(ch for ch in str(name or "") if ch.isalnum() or ch == " ")[:8].upper().strip()


_INITIALS_FS_CACHE: Dict[int, int] = {}


def _initials_font_px(text: str, box: int) -> int:
    """Fiksni font: WW (najšira kombinacija) popuni cijeli okvir ikone."""
    del text
    key = max(1, int(box))
    cached = _INITIALS_FS_CACHE.get(key)
    if cached is not None:
        return cached
    pad = max(1, int(round(key * 0.03)))
    inner = max(10, key - 2 * pad)
    font = QFont()
    font.setBold(True)
    try:
        font.setWeight(QFont.Weight.Black)
    except Exception:
        try:
            font.setWeight(900)
        except Exception:
            pass
    lo, hi = 8, max(8, inner * 2)
    best = 8
    while lo <= hi:
        mid = (lo + hi) // 2
        font.setPixelSize(mid)
        br = QFontMetrics(font).tightBoundingRect("WW")
        if br.width() <= inner and br.height() <= inner:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    _INITIALS_FS_CACHE[key] = best
    return best


def _name_fit_font_px(text: str, width: int, height: int) -> int:
    """Ime unutar gumba — čitljivo, ne preko cijele širine."""
    inner_w = max(20, int(width * 0.72))
    inner_h = max(16, int(height * 0.48))
    sample = text or "WWWWWWWW"
    font = QFont()
    font.setBold(True)
    try:
        font.setWeight(QFont.Weight.Black)
    except Exception:
        pass
    lo, hi = 16, max(16, inner_h)
    best = 16
    while lo <= hi:
        mid = (lo + hi) // 2
        font.setPixelSize(mid)
        fm = QFontMetrics(font)
        br = fm.tightBoundingRect(sample)
        if br.width() <= inner_w and br.height() <= inner_h:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _apply_player_avatar(
    lab: QLabel,
    color_hex: str,
    icon_h: int,
    name: str = "",
    *,
    pix: Optional[QPixmap] = None,
) -> None:
    """Ime → 2 slova; inače player ikona."""
    ini = _name_initials(name)
    lab.setAlignment(Qt.AlignCenter)
    if ini:
        lab.setPixmap(QPixmap())
        lab.setText(ini)
        side = max(24, int(icon_h))
        lab.setFixedSize(side, side)
        fg = _player_fg(color_hex)
        fs = _initials_font_px(ini, side)
        lab.setContentsMargins(0, 0, 0, 0)
        lab.setStyleSheet(
            f"border: none; background: transparent; color: {fg}; "
            f"font-size: {fs}px; font-weight: 900; padding: 0px; margin: 0px;"
        )
        return
    lab.setText("")
    lab.setStyleSheet("border: none; background: transparent;")
    if pix is not None and not pix.isNull():
        lab.setPixmap(pix)
        lab.setFixedSize(pix.width(), pix.height())
    else:
        lab.clear()
        side = max(24, int(icon_h))
        lab.setFixedSize(side, side)


def _initials_avatar_pixmap(name: str, size: int, color_hex: str) -> Optional[QPixmap]:
    """Inicijali kao pixmap (za prekrižene mrtve igrače)."""
    ini = _name_initials(name)
    if not ini:
        return None
    s = max(24, int(size))
    out = QPixmap(s, s)
    out.fill(QColor(0, 0, 0, 0))
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setRenderHint(QPainter.TextAntialiasing, True)
    font = QFont()
    font.setBold(True)
    try:
        font.setWeight(QFont.Weight.Black)
    except Exception:
        try:
            font.setWeight(900)
        except Exception:
            pass
    font.setPixelSize(_initials_font_px(ini, s))
    p.setFont(font)
    p.setPen(QColor(_player_fg(color_hex)))
    p.drawText(QRect(0, 0, s, s), int(Qt.AlignCenter), ini)
    p.end()
    return out


def _winner_icon_tinted(height: int, color_hex: str) -> Optional[QPixmap]:
    key = ("winner_icon.png", int(height), color_hex.lower())
    cached = _TINT_CACHE.get(key)
    if cached is not None and not cached.isNull():
        return cached
    base = _icon_pixmap("winner_icon.png", height)
    if base is None:
        return None
    out = _tint_pixmap(base, color_hex)
    _TINT_CACHE[key] = out
    return out


_LEAGUE_QR_PIX_CACHE: Dict[Tuple[str, int], QPixmap] = {}


def _league_qr_pixmap(payload: str, size: int = 232) -> Optional[QPixmap]:
    """Render league QR payload to a pixmap (no Pillow). Cached per payload/size."""
    text = str(payload or "").strip()
    if not text:
        return None
    key = (text, int(size))
    cached = _LEAGUE_QR_PIX_CACHE.get(key)
    if cached is not None and not cached.isNull():
        return cached
    try:
        from league_qr import qr_matrix

        matrix = qr_matrix(text)
    except Exception as e:
        print(f"[ui] league QR encode error: {e}", flush=True)
        return None
    if not matrix:
        print("[ui] league QR matrix empty", flush=True)
        return None
    n = len(matrix)
    scale = max(1, int(size) // n)
    px = n * scale
    pix = QPixmap(px, px)
    pix.fill(QColor("#ffffff"))
    p = QPainter(pix)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("#111111"))
    for y, row in enumerate(matrix):
        for x, dark in enumerate(row):
            if dark:
                p.drawRect(x * scale, y * scale, scale, scale)
    p.end()
    _LEAGUE_QR_PIX_CACHE[key] = pix
    return pix


def _killer_icon_tinted(height: int, color_hex: str) -> Optional[QPixmap]:
    key = ("killer.png", int(height), color_hex.lower(), "matte")
    cached = _TINT_CACHE.get(key)
    if cached is not None and not cached.isNull():
        return cached
    base = _icon_pixmap("killer.png", height)
    if base is None:
        return _player_icon_tinted(height, color_hex)
    base = _pixmap_matte_black(base)
    out = _tint_pixmap(base, color_hex)
    _TINT_CACHE[key] = out
    return out


def _killer_corner_badge_pix(size: int, color_hex: str = "#c41e3a") -> Optional[QPixmap]:
    """White killer.png with a player-colored corner — cached per size/color."""
    s = max(12, int(size))
    fg = _player_fg(color_hex) if color_hex else "#c41e3a"
    key = ("killer_badge_white", s, fg.lower())
    cached = _TINT_CACHE.get(key)
    if cached is not None and not cached.isNull():
        return cached
    out = QPixmap(s, s)
    out.fill(QColor(0, 0, 0, 0))
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing, True)
    corner = QPainterPath()
    corner.moveTo(float(s), float(s) * 0.22)
    corner.lineTo(float(s), float(s))
    corner.lineTo(float(s) * 0.22, float(s))
    corner.closeSubpath()
    p.fillPath(corner, QColor(fg))
    icon = _icon_pixmap("killer.png", max(10, int(round(s * 0.92))))
    if icon is not None and not icon.isNull():
        icon = _pixmap_matte_black(icon)
        icon = _tint_pixmap(icon, "#f4f4f4")
        x = max(0, (s - icon.width()) // 2)
        y = max(0, (s - icon.height()) // 2)
        p.drawPixmap(x, y, icon)
    p.end()
    _TINT_CACHE[key] = out
    return out


def _color_swatch_icon(color_hex: str, width: int = 112, height: int = 30) -> QIcon:
    """Dugi pravokutnik boje (za zatvoreni gumb i meni)."""
    w, h = int(width), int(height)
    pix = QPixmap(w, h)
    pix.fill(QColor(0, 0, 0, 0))
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setBrush(QColor(color_hex))
    p.setPen(QColor("#ffffff"))
    p.drawRoundedRect(1, 1, w - 2, h - 2, 6, 6)
    p.end()
    return QIcon(pix)


def _flag_icon(lang_code: str, height: int = 40) -> QIcon:
    path = _asset_path("assets", "flags", f"{lang_code}.png")
    if not os.path.isfile(path):
        return QIcon()
    pix = QPixmap(path)
    if pix.isNull():
        return QIcon()
    return QIcon(pix.scaledToHeight(max(1, height), Qt.SmoothTransformation))


def _player_icons_row_pixmap(
    count: int,
    icon_h: int,
    colors: Optional[List[str]] = None,
) -> Optional[QPixmap]:
    """N player_icon u nizu; opcionalno obojano po boji igrača."""
    one = _icon_pixmap("player_icon.png", icon_h)
    if one is None or count < 1:
        return one
    gap = max(4, icon_h // 10)
    w = count * one.width() + (count - 1) * gap
    out = QPixmap(w, one.height())
    out.fill(QColor(0, 0, 0, 0))
    p = QPainter(out)
    x = 0
    for i in range(count):
        icon = one
        if colors and i < len(colors):
            icon = _player_icon_tinted(icon_h, colors[i]) or one
        p.drawPixmap(x, 0, icon)
        x += one.width() + gap
    p.end()
    return out


def _make_logo_label(height: int) -> QLabel:
    lab = QLabel()
    lab.setObjectName("AppLogo")
    lab.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
    lab.setFixedHeight(height)
    pix = _logo_pixmap(height)
    if pix is not None:
        lab.setPixmap(pix)
        lab.setFixedSize(pix.width(), height)
    else:
        lab.hide()
    return lab


def _make_icon_label(filename: str, height: int, *, color: Optional[str] = None) -> QLabel:
    lab = QLabel()
    lab.setAlignment(Qt.AlignVCenter | Qt.AlignCenter)
    lab.setFixedHeight(height)
    pix = _icon_pixmap(filename, height)
    if pix is not None:
        if color:
            pix = _tint_pixmap(pix, color)
        lab.setPixmap(pix)
        lab.setFixedSize(pix.width(), height)
    else:
        lab.setText("•")
        lab.setFixedWidth(height)
    return lab


def _btn(text: str, action: str, *, role: str = "", tall: bool = False) -> AnimButton:
    b = AnimButton(text)
    b.setProperty("action", action)
    if tall and not role:
        role = "Footer"
    if role:
        b.setObjectName(role)
    if role in (
        "PlayNav",
        "PlayExit",
        "PlayAction",
        "PlayActionAccent",
        "PlayActionWarn",
        "PlayExitWide",
    ):
        b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    elif tall or role in ("Footer", "Primary", "Accent", "Skip", "Danger"):
        b.setMinimumHeight(100)
        b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    else:
        b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    b.setCursor(Qt.ArrowCursor)
    b.setFocusPolicy(Qt.NoFocus)
    b.setAttribute(Qt.WA_Hover, False)
    b.setAutoDefault(False)
    b.setDefault(False)
    b.setFlat(False)
    return b


def _rules_footer_btn(text: str, action: str, *, role: str = "Footer") -> AnimButton:
    """Medium text footer button (UPUTE) — same height as NASTAVI."""
    b = _btn(text, action, role=role, tall=False)
    b.setMinimumHeight(78)
    b.setMaximumHeight(84)
    b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    b.setStyleSheet(
        "QPushButton#Footer { min-height: 78px; max-height: 84px; "
        "font-size: 48px; font-weight: 800; padding: 6px 14px; "
        "border-color: #5a5a66; }"
    )
    return b


def _rules_continue_btn(text: str, action: str) -> AnimButton:
    """NASTAVI continue — same height as UPUTE."""
    b = _btn(text, action, role="Primary", tall=True)
    b.setMinimumHeight(78)
    b.setMaximumHeight(84)
    b.setMinimumWidth(280)
    b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    b.setStyleSheet(
        "QPushButton#Primary { min-height: 78px; max-height: 84px; "
        "font-size: 52px; font-weight: 800; padding: 6px 18px; }"
    )
    return b


def _rules_back_btn(text: str, action: str) -> AnimButton:
    """Full-width NAZAD under UPUTE/NASTAVI — shorter than the old tall footer."""
    b = _btn(text, action, role="Footer", tall=True)
    b.setMinimumHeight(68)
    b.setMaximumHeight(74)
    b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    b.setStyleSheet(
        "QPushButton#Footer { min-height: 68px; max-height: 74px; "
        "font-size: 48px; font-weight: 800; padding: 6px 16px; "
        "border-color: #5a5a66; }"
    )
    return b


def _nav_arrow_btn(text: str, action: str) -> AnimButton:
    """Small square < / > footer navigation button."""
    b = _btn(text, action, role="NavArrow", tall=False)
    b.setFixedSize(72, 72)
    b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    b.setStyleSheet(
        "QPushButton#NavArrow { min-height: 72px; max-height: 72px; "
        "min-width: 72px; max-width: 72px; font-size: 48px; "
        "font-weight: 900; padding: 0px; border-radius: 12px; }"
    )
    return b


def _cal_footer_btn(text: str, action: str, *, role: str = "Footer") -> AnimButton:
    """Compact calibration footer — fits five buttons + status on kiosk/Pi."""
    b = _btn(text, action, role=role, tall=False)
    b.setMinimumHeight(72)
    b.setMaximumHeight(80)
    b.setMinimumWidth(0)
    b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    # Role-specific compact overrides (global Primary/Danger/Footer are ~100–104px).
    sel = f"QPushButton#{role}"
    b.setStyleSheet(
        f"{sel} {{ min-height: 72px; max-height: 80px; "
        f"font-size: 36px; font-weight: 800; padding: 4px 8px; }}"
    )
    return b


def _rules_footer_row(
    *,
    back_btn: AnimButton,
    continue_btn: Optional[AnimButton] = None,
    middle: Optional[AnimButton] = None,
) -> QHBoxLayout:
    """Legacy single-row footer (select players / clear board): ← | stretch | optional."""
    foot = QHBoxLayout()
    foot.setSpacing(14)
    foot.addWidget(back_btn, 0, Qt.AlignLeft | Qt.AlignVCenter)
    if middle is not None:
        foot.addWidget(middle, 1)
    else:
        foot.addStretch(1)
    if continue_btn is not None:
        foot.addWidget(continue_btn, 2, Qt.AlignRight | Qt.AlignVCenter)
    return foot


def _rules_footer_block(
    *,
    back_btn: AnimButton,
    continue_btn: Optional[AnimButton] = None,
    middle: Optional[AnimButton] = None,
) -> QVBoxLayout:
    """Rules/tutorial footer:
    [ UPUTE ] [ NASTAVI ]
    [      NAZAD (wide)     ]
    """
    col = QVBoxLayout()
    col.setSpacing(12)
    if middle is not None or continue_btn is not None:
        top = QHBoxLayout()
        top.setSpacing(14)
        if middle is not None:
            top.addWidget(middle, 1)
        if continue_btn is not None:
            top.addWidget(continue_btn, 1)
        col.addLayout(top)
    col.addWidget(back_btn)
    return col


class MarkStack(QWidget):
    """3 horizontal bars — ~3x wider marks."""

    COL_W = 72
    ANIM_MS = 220

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._marks = 0
        self._prev_marks = 0
        self._color = "#ff4040"
        self._appear_t = 1.0
        self._appear_anim: Optional[QVariantAnimation] = None
        self.setFixedWidth(self.COL_W)
        self.setMinimumHeight(108)
        self.setMaximumHeight(130)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def _stop_appear_anim(self) -> None:
        if self._appear_anim is not None:
            self._appear_anim.stop()
            self._appear_anim = None
        self._appear_t = 1.0

    def set_marks(self, marks: int, color: str) -> None:
        marks = max(0, min(3, int(marks)))
        color = color or "#ff4040"
        if marks == self._marks and color == self._color and self._appear_t >= 1.0:
            return
        old = self._marks
        self._color = color
        self._marks = marks
        if marks > old:
            self._prev_marks = old
            self._stop_appear_anim()
            anim = QVariantAnimation(self)
            anim.setDuration(self.ANIM_MS)
            anim.setStartValue(0.0)
            anim.setEndValue(1.0)
            anim.setEasingCurve(QEasingCurve.OutBack)

            def _tick(v) -> None:
                self._appear_t = float(v)
                self.update()

            def _done() -> None:
                self._appear_anim = None
                self._appear_t = 1.0
                self._prev_marks = self._marks
                self.update()

            anim.valueChanged.connect(_tick)
            anim.finished.connect(_done)
            self._appear_anim = anim
            self._appear_t = 0.0
            anim.start()
        else:
            self._stop_appear_anim()
            self._prev_marks = marks
            self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        from PySide6.QtGui import QBrush, QPen

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = max(1, self.width()), max(1, self.height())
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        inset = 4
        bar_w = max(20, w - 2 * inset)
        gap = max(5, h // 14)
        bar_h = max(18, (h - 2 * gap) // 3)
        y = max(0, (h - (3 * bar_h + 2 * gap)) // 2)
        fill = QColor(self._color)
        if not fill.isValid():
            fill = QColor(255, 64, 64)
        empty = QColor(36, 36, 42)
        t = max(0.0, min(1.0, float(self._appear_t)))
        # OutBack can overshoot >1 briefly — clamp paint scale.
        scale = max(0.0, min(1.15, t))
        for i in range(3):
            filled_now = i >= (3 - self._marks)
            filled_before = i >= (3 - self._prev_marks)
            appearing = filled_now and not filled_before and t < 1.0
            p.setPen(QPen(QColor(22, 22, 26), 1))
            if appearing:
                # Scale-in + fade from bar center (hit-bounce style).
                cx = inset + bar_w / 2.0
                cy = y + bar_h / 2.0
                bw = bar_w * (0.55 + 0.45 * min(1.0, scale))
                bh = bar_h * (0.55 + 0.45 * min(1.0, scale))
                fill_a = QColor(fill)
                fill_a.setAlpha(max(0, min(255, int(230 * min(1.0, t) + 25))))
                p.setBrush(QBrush(fill_a))
                p.drawRoundedRect(
                    int(round(cx - bw / 2.0)),
                    int(round(cy - bh / 2.0)),
                    max(1, int(round(bw))),
                    max(1, int(round(bh))),
                    5,
                    5,
                )
            else:
                p.setBrush(QBrush(fill if filled_now else empty))
                p.drawRoundedRect(inset, y, bar_w, bar_h, 5, 5)
            y += bar_h + gap
        p.end()


def _fit_text_font_px(
    text: str,
    slot_w: int,
    *,
    base_px: int,
    min_px: int = 12,
    max_px: Optional[int] = None,
) -> int:
    """Largest bold pixel size that fits slot_w. Short text grows, long text shrinks."""
    s = str(text) if str(text) else "0"
    lo = max(8, int(min_px))
    base = max(lo, int(base_px))
    hi = int(max_px) if max_px is not None else max(base, int(round(base * 1.22)))
    if hi < lo:
        hi = lo
    slot = max(8, int(slot_w))
    font = QFont()
    font.setBold(True)
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        font.setPixelSize(mid)
        if QFontMetrics(font).horizontalAdvance(s) <= slot:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


class CricketRow(QFrame):
    """Ikona | 7 mark stacks | pts — puna širina."""

    TARGETS = (15, 16, 17, 18, 19, 20, 25)
    PTS_W = 132
    ICON_H = 72
    ICON_BORDER = 4

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAutoFillBackground(False)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 8, 8, 8)
        lay.setSpacing(0)
        icon_box = self.ICON_H + 2 * self.ICON_BORDER
        self.icon_frame = QFrame()
        self.icon_frame.setObjectName("CricketIconFrame")
        self.icon_frame.setFixedSize(icon_box, icon_box)
        iframe = QVBoxLayout(self.icon_frame)
        iframe.setContentsMargins(0, 0, 0, 0)
        iframe.setSpacing(0)
        iframe.setAlignment(Qt.AlignCenter)
        self.icon_lab = QLabel()
        self.icon_lab.setAlignment(Qt.AlignCenter)
        self.icon_lab.setFixedSize(self.ICON_H, self.ICON_H)
        self.icon_lab.setStyleSheet("border: none; background: transparent;")
        iframe.addWidget(self.icon_lab, 0, Qt.AlignCenter)
        lay.addWidget(self.icon_frame, 0, Qt.AlignVCenter)
        self.stacks = []
        for _ in self.TARGETS:
            lay.addStretch(1)
            st = MarkStack()
            self.stacks.append(st)
            lay.addWidget(st, 0)
        lay.addStretch(1)
        self.pts_lab = QLabel("0")
        self.pts_lab.setFixedWidth(self.PTS_W)
        self.pts_lab.setFixedHeight(72)
        self.pts_lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.addWidget(self.pts_lab, 0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(128)
        self._last_finished = None
        self._last_active = False
        self._row_color = "#ffffff"
        self._border_alpha = 0.0
        self._border_anim: Optional[QVariantAnimation] = None
        self._border_pulse: Optional[QSequentialAnimationGroup] = None

    def _stop_border_anim(self) -> None:
        if self._border_pulse is not None:
            self._border_pulse.stop()
            self._border_pulse = None
        if self._border_anim is not None:
            self._border_anim.stop()
            self._border_anim = None

    def _rgba(self, hex_color: str, alpha: float) -> str:
        c = QColor(hex_color)
        a = max(0.0, min(1.0, float(alpha)))
        return f"rgba({c.red()}, {c.green()}, {c.blue()}, {a:.3f})"

    def _apply_chrome(self, *, border_alpha: float, finished: bool) -> None:
        fg = _player_fg(self._row_color)
        if finished:
            border = fg
            icon_bw = self.ICON_BORDER
            icon_border = fg
            self._border_alpha = 1.0
        elif border_alpha <= 0.02:
            border = "transparent"
            icon_bw = 0
            icon_border = "transparent"
            self._border_alpha = 0.0
        else:
            border = self._rgba(fg, border_alpha)
            icon_bw = self.ICON_BORDER
            icon_border = self._rgba(fg, border_alpha)
            self._border_alpha = float(border_alpha)
        icon_radius = max(10, int(round(14 * self.ICON_H / 100.0)))
        self.setStyleSheet(
            f"QFrame {{ background-color: transparent; border: 3px solid {border};"
            f" border-radius: 10px; }}"
            f" QFrame#CricketIconFrame {{"
            f" background: transparent; border: {icon_bw}px solid {icon_border};"
            f" border-radius: {icon_radius}px; }}"
            f" QLabel {{ border: none; background: transparent; }}"
        )

    def animate_active_border(self, becoming_active: bool) -> None:
        """Fade/pulse active border in or out — no row reorder."""
        if self._last_finished:
            return
        self._stop_border_anim()
        start = float(self._border_alpha)
        if becoming_active:
            fade = QVariantAnimation(self)
            fade.setStartValue(start)
            fade.setEndValue(1.0)
            fade.setDuration(220)
            fade.setEasingCurve(QEasingCurve.OutCubic)
            fade.valueChanged.connect(
                lambda v: self._apply_chrome(border_alpha=float(v), finished=False)
            )
            pulse = QVariantAnimation(self)
            pulse.setStartValue(1.0)
            pulse.setEndValue(0.55)
            pulse.setDuration(160)
            pulse.setEasingCurve(QEasingCurve.InOutQuad)
            pulse.valueChanged.connect(
                lambda v: self._apply_chrome(border_alpha=float(v), finished=False)
            )
            pulse_back = QVariantAnimation(self)
            pulse_back.setStartValue(0.55)
            pulse_back.setEndValue(1.0)
            pulse_back.setDuration(180)
            pulse_back.setEasingCurve(QEasingCurve.InOutQuad)
            pulse_back.valueChanged.connect(
                lambda v: self._apply_chrome(border_alpha=float(v), finished=False)
            )
            seq = QSequentialAnimationGroup(self)
            seq.addAnimation(fade)
            seq.addAnimation(pulse)
            seq.addAnimation(pulse_back)
            self._border_pulse = seq
            seq.start()
        else:
            anim = QVariantAnimation(self)
            anim.setStartValue(max(start, 0.05) if start <= 0.02 else start)
            anim.setEndValue(0.0)
            anim.setDuration(240)
            anim.setEasingCurve(QEasingCurve.InCubic)
            anim.valueChanged.connect(
                lambda v: self._apply_chrome(border_alpha=float(v), finished=False)
            )
            self._border_anim = anim
            anim.start()

    def set_row(
        self,
        label: str,
        marks: dict,
        points: int,
        color: str,
        active: bool,
        *,
        show_pts: bool,
        finished: bool = False,
        rank: Optional[int] = None,
        animate_border: Optional[bool] = None,
        name: str = "",
    ) -> None:
        _ = label
        want_active = bool(active) and not finished
        marks_key = tuple(sorted((marks or {}).items()))
        pts_key = (
            f"#{int(rank)}"
            if show_pts and finished and rank is not None
            else (str(int(points)) if show_pts else "")
        )
        same_content = (
            color == getattr(self, "_last_row_color", None)
            and marks_key == getattr(self, "_last_marks_key", None)
            and pts_key == getattr(self, "_last_pts_key", None)
            and show_pts == getattr(self, "_last_show_pts", None)
            and finished == getattr(self, "_last_finished", None)
            and animate_border is None
            and want_active == getattr(self, "_last_active", None)
            and str(name or "") == getattr(self, "_last_name", None)
        )
        if same_content:
            return
        self._last_row_color = color
        self._last_marks_key = marks_key
        self._last_pts_key = pts_key
        self._last_show_pts = show_pts
        self._last_name = str(name or "")
        self._row_color = color
        fg = _player_fg(color)
        pix = _player_icon_tinted(self.ICON_H, fg)
        _apply_player_avatar(self.icon_lab, color, self.ICON_H, name, pix=pix)
        for st, key in zip(self.stacks, self.TARGETS):
            st.set_marks(_mark_count(marks, key), fg)
        if show_pts:
            # Gotov igrač: pokaži #mjesto umjesto bodova (jasnije tko je završio)
            if finished and rank is not None:
                self.pts_lab.setText(f"#{int(rank)}")
            else:
                self.pts_lab.setText(str(int(points)))
            self.pts_lab.show()
            pts_text = self.pts_lab.text()
            pts_px = _fit_text_font_px(
                pts_text,
                self.PTS_W - 12,
                base_px=56,
                min_px=18,
                max_px=64,
            )
            self.pts_lab.setStyleSheet(
                f"font-size: {pts_px}px; font-weight: 800; color: {fg}; border: none; "
                f"background: transparent; padding-right: 4px;"
            )
        else:
            self.pts_lab.hide()

        if finished:
            self._stop_border_anim()
            self._apply_chrome(border_alpha=1.0, finished=True)
        elif animate_border is True:
            self.animate_active_border(True)
        elif animate_border is False:
            self.animate_active_border(False)
        else:
            self._stop_border_anim()
            self._apply_chrome(
                border_alpha=1.0 if want_active else 0.0, finished=False
            )
        _apply_player_avatar(self.icon_lab, color, self.ICON_H, name, pix=pix)
        self._last_finished = finished
        self._last_active = want_active


def _painter_round_rect_path(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    tl: float = 0.0,
    tr: float = 0.0,
    br: float = 0.0,
    bl: float = 0.0,
) -> QPainterPath:
    """Rect path with independent corner radii (0 = square corner)."""
    path = QPainterPath()
    if w <= 0 or h <= 0:
        return path
    tl = max(0.0, min(float(tl), w / 2.0, h / 2.0))
    tr = max(0.0, min(float(tr), w / 2.0, h / 2.0))
    br = max(0.0, min(float(br), w / 2.0, h / 2.0))
    bl = max(0.0, min(float(bl), w / 2.0, h / 2.0))
    path.moveTo(x + tl, y)
    path.lineTo(x + w - tr, y)
    if tr > 0:
        path.arcTo(x + w - 2 * tr, y, 2 * tr, 2 * tr, 90, -90)
    else:
        path.lineTo(x + w, y)
    path.lineTo(x + w, y + h - br)
    if br > 0:
        path.arcTo(x + w - 2 * br, y + h - 2 * br, 2 * br, 2 * br, 0, -90)
    else:
        path.lineTo(x + w, y + h)
    path.lineTo(x + bl, y + h)
    if bl > 0:
        path.arcTo(x, y + h - 2 * bl, 2 * bl, 2 * bl, 270, -90)
    else:
        path.lineTo(x, y + h)
    path.lineTo(x, y + tl)
    if tl > 0:
        path.arcTo(x, y, 2 * tl, 2 * tl, 180, -90)
    else:
        path.lineTo(x, y)
    path.closeSubpath()
    return path


class KillerLivesBar(QWidget):
    """Segmented lives bar: total = max lives, filled = current (player color)."""

    # Dark empty segments — readable on black cards without looking washed out.
    EMPTY_COLOR = "#1a1a1e"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("KillerLivesBar")
        self._max_lives = 3
        self._lives = 3
        self._fill_color = QColor("#ffffff")
        self._bar_h = 18
        self.setFixedHeight(self._bar_h)
        self.setMinimumWidth(40)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)

    def set_bar_height(self, h: int) -> None:
        self._bar_h = max(8, int(h))
        self.setFixedHeight(self._bar_h)
        self.update()

    def set_state(self, lives: int, max_lives: int, color: str) -> None:
        max_lives = max(1, min(10, int(max_lives)))
        lives = max(0, min(max_lives, int(lives)))
        fg = QColor(_player_fg(color))
        if not fg.isValid():
            fg = QColor("#ffffff")
        if (
            lives == self._lives
            and max_lives == self._max_lives
            and fg == self._fill_color
        ):
            return
        self._lives = lives
        self._max_lives = max_lives
        self._fill_color = fg
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        n = max(1, int(self._max_lives))
        filled = max(0, min(n, int(self._lives)))
        w = self.width()
        h = self.height()
        if w < 2 or h < 2:
            return
        # Gaps between lives; only the first and last segments are rounded.
        gap = max(6, min(12, int(round(h * 0.48))))
        while n > 1 and (n - 1) * gap >= w * 0.40:
            gap = max(3, gap - 1)
        total_gap = (n - 1) * gap
        usable = max(n, w - total_gap)
        seg_w = usable / float(n)
        radius = max(2.0, min(h / 2.0, seg_w / 2.0))
        empty = QColor(self.EMPTY_COLOR)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        x = 0.0
        for i in range(n):
            if i == n - 1:
                rect_w = max(1.0, float(w) - x)
            else:
                rect_w = seg_w
            rx = float(round(x))
            rw = max(1.0, float(round(rect_w)))
            if n == 1:
                tl = tr = br = bl = radius
            elif i == 0:
                tl = bl = radius
                tr = br = 0.0
            elif i == n - 1:
                tl = bl = 0.0
                tr = br = radius
            else:
                tl = tr = br = bl = 0.0
            p.setBrush(self._fill_color if i < filled else empty)
            p.drawPath(
                _painter_round_rect_path(
                    rx, 0.0, rw, float(h), tl=tl, tr=tr, br=br, bl=bl
                )
            )
            x += seg_w + gap
        p.end()


class KillerRow(QFrame):
    """[icon] number — then segmented lives bar (X01 PlayerCard chrome)."""

    ICON_H = 96
    BAR_H = 18
    FONT_PX = 64
    MIN_H = 140
    MAX_H = 180
    ICON_BORDER = 4
    MAX_LIVES = 10
    KILLER_MARK = "#8b1515"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._icon_h = self.ICON_H
        self._font_px = self.FONT_PX
        self._bar_h = self.BAR_H
        root = QVBoxLayout(self)
        self.setAttribute(Qt.WA_StyledBackground, True)
        root.setContentsMargins(14, 8, 14, 8)
        root.setSpacing(6)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(14)
        # Frame = icon + border ring. Margins stay 0 so stylesheet border + AlignCenter
        # keep the pixmap centered (extra layout margins + border caused low/right drift).
        icon_box = self._icon_h + 2 * self.ICON_BORDER
        self.icon_frame = QFrame()
        self.icon_frame.setObjectName("KillerIconFrame")
        self.icon_frame.setFixedSize(icon_box, icon_box)
        iframe = QVBoxLayout(self.icon_frame)
        iframe.setContentsMargins(0, 0, 0, 0)
        iframe.setSpacing(0)
        iframe.setAlignment(Qt.AlignCenter)
        self.icon_lab = QLabel()
        self.icon_lab.setAlignment(Qt.AlignCenter)
        self.icon_lab.setFixedSize(self._icon_h, self._icon_h)
        self.icon_lab.setStyleSheet("border: none; background: transparent;")
        iframe.addWidget(self.icon_lab, 0, Qt.AlignCenter)
        self.k_badge = QLabel("", self.icon_frame)
        self.k_badge.setObjectName("KillerBadge")
        self.k_badge.setAlignment(Qt.AlignCenter)
        self.k_badge.setAttribute(Qt.WA_StyledBackground, True)
        self.k_badge.setScaledContents(False)
        self.k_badge.hide()
        top.addWidget(self.icon_frame, 0, Qt.AlignVCenter)
        self.num_lab = QLabel("")
        self.num_lab.setObjectName("KillerNum")
        self.num_lab.setAlignment(Qt.AlignCenter)
        self._num_slot_w = 0
        top.addWidget(self.num_lab, 0, Qt.AlignVCenter)
        self.lives_wrap = QWidget()
        self.lives_wrap.setObjectName("KillerLivesWrap")
        self.lives_wrap.setStyleSheet("background: transparent; border: none;")
        self.lives_wrap.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        lw = QHBoxLayout(self.lives_wrap)
        lw.setContentsMargins(0, 0, 0, 0)
        lw.setSpacing(4)
        self.heart_lab = QLabel()
        self.heart_lab.setObjectName("KillerLifeHeart")
        self.heart_lab.setAlignment(Qt.AlignCenter)
        self.heart_lab.setStyleSheet("border: none; background: transparent;")
        lw.addWidget(self.heart_lab, 0, Qt.AlignVCenter)
        self.lives_count = QLabel("0")
        self.lives_count.setObjectName("KillerLivesCount")
        self.lives_count.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.lives_count.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        lw.addWidget(self.lives_count, 0, Qt.AlignVCenter)
        self.head_frame = QFrame(self)
        self.head_frame.setObjectName("KillerHeadFrame")
        self.head_frame.setAttribute(Qt.WA_StyledBackground, True)
        self.head_frame.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.head_frame.hide()
        self._head_lay = QHBoxLayout(self.head_frame)
        self._head_lay.setContentsMargins(6, 4, 8, 4)
        self._head_lay.setSpacing(4)
        top.addStretch(1)
        top.addWidget(self.lives_wrap, 0, Qt.AlignVCenter)
        self._top_lay = top
        root.addLayout(top, 0)

        lives_row = QHBoxLayout()
        lives_row.setContentsMargins(0, 0, 0, 0)
        lives_row.setSpacing(0)
        self._lives_row = lives_row
        root.addLayout(lives_row, 0)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(self.MAX_H)
        self._compact = False
        self._waiting_fixed = False
        self._last_is_killer = None
        self._last_color = ""
        self._last_lives = None
        self._last_max_lives = None
        self._last_dead = None
        self._last_number_key = None
        self._last_active = None
        self._icon_anim: Optional[QPropertyAnimation] = None
        self._icon_pulse_lay = None
        self._card_life_anim: Optional[QParallelAnimationGroup] = None
        self._life_flash_timer: Optional[QTimer] = None
        self._life_flash_step = 0
        self._life_flash_base_ss = ""
        self._style_color = "#ffffff"
        self._style_active = False
        self._style_finished = False
        self._style_is_killer = False
        self._stack_mode = "active"
        self._badge_anim: Optional[QParallelAnimationGroup] = None
        self._refresh_num_slot()

    @staticmethod
    def bar_h_for_icon(icon_h: int) -> int:
        """Segmented bar under icon+number — keep compact so the kiosk size stays fixed."""
        return max(12, min(28, int(round(icon_h * 0.145))))

    def set_metrics(self, icon_h: int, font_px: int, min_h: int, max_h: int) -> None:
        """Prilagodi veličinu kartice (aktivni vs waiting) kao X01 PlayerCard."""
        _ = min_h
        card_h = int(max_h)
        if (
            icon_h == self._icon_h
            and font_px == self._font_px
            and self.maximumHeight() == card_h
            and self.minimumHeight() == card_h
        ):
            return
        icon_changed = int(icon_h) != int(self._icon_h)
        font_changed = int(font_px) != int(self._font_px)
        self._icon_h = int(icon_h)
        self._font_px = int(font_px)
        self.setFixedHeight(card_h)
        scale = self._icon_h / float(self.ICON_H)
        m = max(6, int(round(8 * scale)))
        hpad = max(14, int(round(14 * scale)))
        root = self.layout()
        if root is not None:
            root.setContentsMargins(hpad, m, hpad, m)
            root.setSpacing(max(4, int(round(6 * scale))))
        if self._top_lay is not None:
            self._top_lay.setSpacing(max(12, int(round(14 * scale))))
        icon_box = self._icon_h + 2 * self.ICON_BORDER
        self.icon_frame.setFixedSize(icon_box, icon_box)
        self.icon_lab.setFixedSize(self._icon_h, self._icon_h)
        self._refresh_num_slot()
        if icon_changed:
            self._layout_killer_badge()
            self._last_color = ""
            self._last_active = None
        if icon_changed or font_changed:
            self._chrome_key = None
        self._refresh_life_heart(self._style_color or "#ffffff")
        if self._compact:
            self._relayout_stack("compact")
            self.apply_compact_width(True)
        elif self._waiting_fixed:
            self._relayout_stack("waiting")
            self.apply_waiting_fixed_width(True)
        else:
            self._relayout_stack("active")

    def _refresh_num_slot(self) -> None:
        font = QFont()
        font.setBold(True)
        font.setPixelSize(max(12, int(self._font_px)))
        self._num_slot_w = QFontMetrics(font).horizontalAdvance("T20") + 8
        self.num_lab.setFixedWidth(self._num_slot_w)
        self.num_lab.setFixedHeight(max(28, int(round(self._font_px * 1.18))))

    def _sync_num_font(self, text: str, fg: str) -> None:
        slot = int(getattr(self, "_num_slot_w", 0) or self.num_lab.width() or 80)
        px = _fit_text_font_px(
            text,
            slot,
            base_px=self._font_px,
            min_px=16,
            max_px=max(self._font_px, int(round(self._font_px * 1.18))),
        )
        self.num_lab.setStyleSheet(
            f"font-size: {px}px; font-weight: 800; color: {fg};"
            f" border: none; background: transparent;"
        )

    def _relayout_stack(self, mode: str) -> None:
        """active: ikona | broj | životi.
        waiting: ikona + životi gore, veliki broj ispod.
        compact: samo ikona."""
        if getattr(self, "_stack_mode", None) == mode:
            return
        self._stack_mode = mode
        top = self._top_lay
        bot = self._lives_row
        head = getattr(self, "_head_lay", None)
        if top is not None:
            while top.count():
                top.takeAt(0)
        if bot is not None:
            while bot.count():
                bot.takeAt(0)
        if head is not None:
            while head.count():
                head.takeAt(0)
        hf = getattr(self, "head_frame", None)
        if hf is not None:
            hf.show()
        if mode == "waiting":
            self._head_lay.addWidget(self.icon_frame, 0, Qt.AlignVCenter)
            self._head_lay.addWidget(self.lives_wrap, 0, Qt.AlignVCenter)
            self.lives_wrap.show()
            top.addWidget(self.head_frame, 0, Qt.AlignLeft)
            self.num_lab.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            bot.addStretch(1)
            bot.addWidget(self.num_lab, 0)
            bot.addStretch(1)
        elif mode == "compact":
            self.lives_wrap.hide()
            self.head_frame.hide()
            self.num_lab.hide()
            top.addWidget(self.icon_frame, 0, Qt.AlignVCenter)
        else:
            self.head_frame.hide()
            top.addWidget(self.icon_frame, 0, Qt.AlignVCenter)
            self.num_lab.setAlignment(Qt.AlignCenter)
            top.addWidget(self.num_lab, 0, Qt.AlignVCenter)
            top.addStretch(1)
            top.addWidget(self.lives_wrap, 0, Qt.AlignVCenter)
            self.lives_wrap.show()
        if (
            getattr(self, "k_badge", None) is not None
            and self.k_badge.isVisible()
        ):
            self._layout_killer_badge()

    def apply_compact_width(self, compact: bool, *, sample: str = "") -> None:
        """Fixed širina za finished (samo ikona) — layout ne skače."""
        del sample
        self._compact = bool(compact)
        if compact:
            self._waiting_fixed = False
            self._relayout_stack("compact")
            scale = self._icon_h / float(self.ICON_H)
            hpad = max(10, int(round(14 * scale)))
            w = hpad + self._icon_h + 2 * self.ICON_BORDER + hpad
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self.setFixedWidth(w)
        else:
            if not self._waiting_fixed:
                self._relayout_stack("active")
            self.setMinimumWidth(0)
            self.setMaximumWidth(16777215)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def apply_waiting_fixed_width(
        self, fixed: bool = True, *, sample: str = "D20"
    ) -> None:
        """Waiting kartice dijele širinu glavne kartice (isti slot i kad netko umre)."""
        del sample
        self._waiting_fixed = bool(fixed)
        if not fixed:
            if not self._compact:
                self._relayout_stack("active")
                self.setMinimumWidth(0)
                self.setMaximumWidth(16777215)
                self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            return
        self._compact = False
        self._relayout_stack("waiting")
        self.setMinimumWidth(0)
        self.setMaximumWidth(16777215)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

    def _refresh_life_heart(self, color: str) -> None:
        """Heart glyph next to the lives count."""
        if not hasattr(self, "heart_lab"):
            return
        waiting = getattr(self, "_stack_mode", "active") == "waiting"
        ratio = 0.36 if waiting else 0.40
        sz = max(20, int(round(self._icon_h * ratio)))
        pix = _heart_pixmap(sz, _player_fg(color) if color else "#ffffff")
        if pix is not None and not pix.isNull():
            self.heart_lab.setPixmap(pix)
            self.heart_lab.setFixedSize(pix.width(), pix.height())
        else:
            self.heart_lab.clear()

    def _stop_card_life_anim(self) -> None:
        if self._card_life_anim is not None:
            self._card_life_anim.stop()
            self._card_life_anim = None
        if self._life_flash_timer is not None:
            self._life_flash_timer.stop()
            self._life_flash_timer = None
        self.setGraphicsEffect(None)
        # Restore stable chrome (flash must not leave thicker border / layout churn).
        if self._style_color:
            self._chrome_key = None
            self._apply_card_chrome(
                self._style_color, self._style_active, self._style_finished
            )

    def _layout_killer_badge(self) -> None:
        """White killer.png with red corner — on waiting cards, the head-frame corner by lives."""
        badge = getattr(self, "k_badge", None)
        if badge is None:
            return
        waiting = getattr(self, "_stack_mode", "active") == "waiting"
        host = self.head_frame if waiting else self.icon_frame
        if badge.parentWidget() is not host:
            badge.setParent(host)
        # Opponent cards: slightly larger corner + knife; active keeps the compact badge.
        bs = max(22, int(round(self._icon_h * (0.58 if waiting else 0.42))))
        fg = self._style_color or "#ffffff"
        pix = _killer_corner_badge_pix(bs, fg)
        badge.setText("")
        if pix is not None and not pix.isNull():
            badge.setPixmap(pix)
        else:
            badge.setPixmap(QPixmap())
        if waiting:
            hw = host.width() if host.width() > 4 else host.sizeHint().width()
            hh = host.height() if host.height() > 4 else host.sizeHint().height()
            box_w = max(hw, 8)
            box_h = max(hh, 8)
        else:
            box_w = host.width() if host.width() > 0 else (
                self._icon_h + 2 * self.ICON_BORDER
            )
            box_h = box_w
        badge.setFixedSize(bs, bs)
        badge.move(max(0, box_w - bs + 2), max(0, box_h - bs + 2))
        badge.raise_()
        self._badge_icon_h = self._icon_h
        self._badge_host = "head" if waiting else "icon"
        self._badge_color = fg

    def _sync_killer_badge(self, visible: bool, *, animate: bool = False) -> None:
        badge = getattr(self, "k_badge", None)
        if badge is None:
            return
        if not visible:
            self._stop_badge_anim()
            badge.hide()
            return
        need_layout = (
            animate
            or not badge.isVisible()
            or getattr(self, "_badge_icon_h", None) != self._icon_h
            or getattr(self, "_badge_host", None)
            != ("head" if getattr(self, "_stack_mode", "active") == "waiting" else "icon")
            or getattr(self, "_badge_color", None) != (self._style_color or "")
        )
        if need_layout:
            self._layout_killer_badge()
        badge.show()
        if animate:
            self._animate_killer_badge()
        else:
            badge.raise_()
        if getattr(self, "_stack_mode", "active") == "waiting":
            QTimer.singleShot(0, self._layout_killer_badge)

    def _stop_badge_anim(self) -> None:
        anim = getattr(self, "_badge_anim", None)
        if anim is not None:
            try:
                anim.finished.disconnect()
            except (TypeError, RuntimeError):
                pass
            anim.stop()
            self._badge_anim = None
        badge = getattr(self, "k_badge", None)
        if badge is not None:
            badge.setGraphicsEffect(None)
            self._layout_killer_badge()

    def _animate_killer_badge(self) -> None:
        """Pop-in kad igrač postane Killer."""
        badge = getattr(self, "k_badge", None)
        if badge is None:
            return
        self._stop_badge_anim()
        self._layout_killer_badge()
        badge.show()
        badge.raise_()
        end = badge.geometry()
        if end.width() < 4 or end.height() < 4:
            return
        cx, cy = end.center().x(), end.center().y()
        sw = max(6, int(end.width() * 0.2))
        sh = max(6, int(end.height() * 0.2))
        start = QRect(cx - sw // 2, cy - sh // 2, sw, sh)
        badge.setGeometry(start)
        eff = QGraphicsOpacityEffect(badge)
        eff.setOpacity(0.0)
        badge.setGraphicsEffect(eff)
        opac = QPropertyAnimation(eff, b"opacity")
        opac.setDuration(260)
        opac.setStartValue(0.0)
        opac.setEndValue(1.0)
        opac.setEasingCurve(QEasingCurve.OutCubic)
        geo = QPropertyAnimation(badge, b"geometry")
        geo.setDuration(320)
        geo.setStartValue(start)
        geo.setEndValue(end)
        geo.setEasingCurve(QEasingCurve.OutCubic)
        group = QParallelAnimationGroup(self)
        group.addAnimation(opac)
        group.addAnimation(geo)

        def _done() -> None:
            self._badge_anim = None
            badge.setGraphicsEffect(None)
            self._layout_killer_badge()

        group.finished.connect(_done)
        self._badge_anim = group
        group.start()

    def _killer_badge_css(self) -> str:
        return (
            " QLabel#KillerBadge { background: transparent; border: none;"
            " padding: 0px; }"
        )

    def _apply_card_chrome(
        self,
        color: str,
        active: bool,
        finished: bool,
        is_killer: Optional[bool] = None,
    ) -> None:
        """X01-style frame: colored border only on the active (current-turn) player."""
        fg = _player_fg(color)
        if is_killer is None:
            is_killer = bool(getattr(self, "_style_is_killer", False))
        is_killer_eff = bool(is_killer) and not finished
        waiting = getattr(self, "_stack_mode", "active") == "waiting"
        lives_px = max(22, int(round(self._icon_h * (0.48 if waiting else 0.54))))
        self._lives_px = lives_px
        if hasattr(self, "lives_count"):
            lf = QFont()
            lf.setBold(True)
            lf.setPixelSize(lives_px)
            lives_slot = QFontMetrics(lf).horizontalAdvance("00") + 4
            self.lives_count.setFixedWidth(lives_slot)
        chrome_key = (
            fg, bool(active), bool(finished), is_killer_eff, waiting,
            int(self._font_px), int(self._icon_h), int(lives_px),
        )
        if chrome_key == getattr(self, "_chrome_key", None):
            self._style_color = fg
            self._style_active = active
            self._style_finished = finished
            self._style_is_killer = is_killer_eff
            return
        self._chrome_key = chrome_key
        self._style_color = fg
        self._style_active = active
        self._style_finished = finished
        self._style_is_killer = is_killer_eff
        # Dead / finished: no outer border (icon may still show X).
        if active and not finished:
            border = fg
            bg = "#1c1218"
            border_w = 4
            icon_border = fg
            icon_bw = self.ICON_BORDER
        else:
            border = "transparent"
            bg = "transparent"
            border_w = 4
            icon_border = "transparent"
            icon_bw = 0
        if self._style_is_killer and not waiting:
            icon_border = fg
            icon_bw = self.ICON_BORDER
        if waiting and not finished:
            head_border = fg
            head_bw = 3
            icon_border = "transparent"
            icon_bw = 0
        else:
            head_border = "transparent"
            head_bw = 0
        radius = 16
        icon_radius = max(12, int(round(16 * self._icon_h / float(PlayerCard.ICON_H))))
        head_radius = max(10, int(round(14 * self._icon_h / float(PlayerCard.ICON_H))))
        self.setStyleSheet(
            f"QFrame {{ background-color: {bg}; border: {border_w}px solid {border};"
            f" border-radius: {radius}px; }}"
            f" QFrame#KillerIconFrame {{"
            f" background: transparent; border: {icon_bw}px solid {icon_border};"
            f" border-radius: {icon_radius}px; }}"
            f" QFrame#KillerHeadFrame {{"
            f" background: transparent; border: {head_bw}px solid {head_border};"
            f" border-radius: {head_radius}px; }}"
            f" QWidget#KillerLivesWrap {{ background: transparent; border: none; }}"
            f" QLabel#KillerNum {{ font-size: {self._font_px}px; font-weight: 800;"
            f" color: {fg}; border: none; background: transparent; }}"
            f" QLabel#KillerLivesCount {{ font-size: {lives_px}px; font-weight: 800;"
            f" color: {fg}; border: none; background: transparent; }}"
            f" QLabel#KillerLifeHeart {{ border: none; background: transparent; }}"
            f" QLabel {{ border: none; background: transparent; }}"
            f"{self._killer_badge_css()}"
        )

    def _pulse_icon(self) -> None:
        """Kratki scale flash kad postane Killer / smrt — centrirano na ikonu."""
        if self._icon_anim is not None:
            try:
                self._icon_anim.finished.disconnect()
            except (TypeError, RuntimeError):
                pass
            self._icon_anim.stop()
            self._icon_anim = None
            self._restore_icon_after_pulse()

        frame = self.icon_frame
        lab = self.icon_lab
        lay = frame.layout()
        if lay is not None:
            lay.activate()
        # Prefer pixmap size; fall back to current label size.
        pix = lab.pixmap()
        if pix is not None and not pix.isNull():
            sw, sh = pix.width(), pix.height()
        else:
            sw, sh = lab.width(), lab.height()
        if sw < 4 or sh < 4:
            return

        # Take label out of the layout so geometry anim isn't reset to (0,0).
        self._icon_pulse_lay = lay
        if lay is not None:
            lay.removeWidget(lab)
        lab.setParent(frame)
        # Fixed size would clamp the pulse — allow free geometry during anim.
        lab.setMinimumSize(0, 0)
        lab.setMaximumSize(16777215, 16777215)
        lab.setScaledContents(True)

        if frame.width() < 4 or frame.height() < 4:
            if self.layout() is not None:
                self.layout().activate()
            if lay is not None:
                lay.activate()
        fc = frame.rect().center()
        if frame.width() < 4 or frame.height() < 4:
            # Fallback: center on expected icon box before first layout pass.
            box = self._icon_h + 2 * self.ICON_BORDER
            fc = QPoint(box // 2, box // 2)

        def _centered(w: int, h: int) -> QRect:
            return QRect(fc.x() - w // 2, fc.y() - h // 2, w, h)

        start = _centered(sw, sh)
        bw, bh = int(sw * 1.28), int(sh * 1.28)
        big = _centered(bw, bh)
        lab.setGeometry(start)
        lab.raise_()
        if getattr(self, "k_badge", None) is not None:
            self.k_badge.raise_()

        grow = QPropertyAnimation(lab, b"geometry")
        grow.setDuration(160)
        grow.setStartValue(start)
        grow.setEndValue(big)
        grow.setEasingCurve(QEasingCurve.OutCubic)
        shrink = QPropertyAnimation(lab, b"geometry")
        shrink.setDuration(180)
        shrink.setStartValue(big)
        shrink.setEndValue(start)
        shrink.setEasingCurve(QEasingCurve.InCubic)
        seq = QSequentialAnimationGroup(self)
        seq.addAnimation(grow)
        seq.addAnimation(shrink)

        def _done() -> None:
            self._icon_anim = None
            self._restore_icon_after_pulse()

        seq.finished.connect(_done)
        self._icon_anim = seq  # type: ignore[assignment]
        seq.start()

    def _restore_icon_after_pulse(self) -> None:
        lab = self.icon_lab
        frame = self.icon_frame
        lab.setScaledContents(False)
        pix = lab.pixmap()
        if pix is not None and not pix.isNull():
            lab.setFixedSize(pix.width(), pix.height())
        else:
            lab.setFixedSize(self._icon_h, self._icon_h)
        lay = self._icon_pulse_lay if self._icon_pulse_lay is not None else frame.layout()
        self._icon_pulse_lay = None
        if lay is not None and lay.indexOf(lab) < 0:
            lay.addWidget(lab, 0, Qt.AlignCenter)
        lab.setAlignment(Qt.AlignCenter)
        if getattr(self, "k_badge", None) is not None:
            self.k_badge.raise_()

    def _animate_card_life_loss(self) -> None:
        """One slower red wash/flash when a life is lost (no layout size change)."""
        self._stop_card_life_anim()
        self._life_flash_step = 0
        self._life_flash_base_ss = self.styleSheet()
        # Keep the same border width as normal chrome so card min/max height and
        # the Expanding lives bar width stay stable (no jump/grow on waiting cards).
        border_w = 4
        # Killer ring stays even on life-loss flash; non-killers keep turn ring only.
        # Killer ring on the active card only; waiting uses the head-frame corner badge.
        if getattr(self, "_style_is_killer", False) and getattr(
            self, "_stack_mode", "active"
        ) != "waiting":
            icon_bw = self.ICON_BORDER
            icon_border = self._style_color
        elif self._style_active and not self._style_finished:
            icon_bw = self.ICON_BORDER
            icon_border = self._style_color
        else:
            icon_bw = 0
            icon_border = "transparent"

        def _flash_tick() -> None:
            self._life_flash_step += 1
            if self._life_flash_step == 1:
                icon_radius = max(
                    12, int(round(16 * self._icon_h / float(PlayerCard.ICON_H)))
                )
                waiting = getattr(self, "_stack_mode", "active") == "waiting"
                head_bw = 3 if waiting and not self._style_finished else 0
                head_border = self._style_color if head_bw else "transparent"
                head_radius = max(10, int(round(14 * self._icon_h / float(PlayerCard.ICON_H))))
                lives_px = int(getattr(self, "_lives_px", 0)) or max(
                    22, int(round(self._icon_h * (0.48 if waiting else 0.54)))
                )
                self.setStyleSheet(
                    f"QFrame {{ background-color: #3a1010; border: {border_w}px solid #ff3030;"
                    f" border-radius: 16px; }}"
                    f" QFrame#KillerIconFrame {{"
                    f" background: transparent; border: {icon_bw}px solid {icon_border};"
                    f" border-radius: {icon_radius}px; }}"
                    f" QFrame#KillerHeadFrame {{"
                    f" background: transparent; border: {head_bw}px solid {head_border};"
                    f" border-radius: {head_radius}px; }}"
                    f" QWidget#KillerLivesWrap {{ background: transparent; border: none; }}"
                    f" QLabel#KillerNum {{ font-size: {self._font_px}px; font-weight: 800;"
                    f" color: #ff6060; border: none; background: transparent; }}"
                    f" QLabel#KillerLivesCount {{ font-size: {lives_px}px; font-weight: 800;"
                    f" color: #ff6060; border: none; background: transparent; }}"
                    f" QLabel#KillerLifeHeart {{ border: none; background: transparent; }}"
                    f" QLabel {{ border: none; background: transparent; }}"
                    f"{self._killer_badge_css()}"
                )
                return
            self._chrome_key = None
            self._apply_card_chrome(
                self._style_color, self._style_active, self._style_finished
            )
            if self._life_flash_timer is not None:
                self._life_flash_timer.stop()
                self._life_flash_timer = None

        self._life_flash_timer = QTimer(self)
        self._life_flash_timer.setInterval(420)
        self._life_flash_timer.timeout.connect(_flash_tick)
        self._life_flash_timer.start()
        _flash_tick()

        # Color wash only — never animate child geometry (layout-managed widgets
        # like the Expanding lives bar grow/jump when geometry is shaken).
        group = QParallelAnimationGroup(self)
        colorize = QGraphicsColorizeEffect(self)
        colorize.setColor(QColor("#ff2020"))
        colorize.setStrength(0.0)
        self.setGraphicsEffect(colorize)
        c_up = QPropertyAnimation(colorize, b"strength")
        c_up.setDuration(220)
        c_up.setStartValue(0.0)
        c_up.setEndValue(0.65)
        c_up.setEasingCurve(QEasingCurve.OutQuad)
        c_down = QPropertyAnimation(colorize, b"strength")
        c_down.setDuration(380)
        c_down.setStartValue(0.65)
        c_down.setEndValue(0.0)
        c_down.setEasingCurve(QEasingCurve.InQuad)
        c_seq = QSequentialAnimationGroup(self)
        c_seq.addAnimation(c_up)
        c_seq.addAnimation(c_down)
        group.addAnimation(c_seq)

        def _done() -> None:
            self.setGraphicsEffect(None)
            self._card_life_anim = None

        group.finished.connect(_done)
        self._card_life_anim = group
        group.start()

    def _set_lives_display(
        self,
        lives: int,
        max_lives: int,
        color: str,
        *,
        finished: bool,
    ) -> None:
        """Heart + lives count next to initials; hidden when dead."""
        max_lives = max(1, min(self.MAX_LIVES, int(max_lives)))
        lives = max(0, min(max_lives, int(lives)))
        wrap = getattr(self, "lives_wrap", None)
        count = getattr(self, "lives_count", None)
        if finished:
            if wrap is not None:
                wrap.hide()
            self._last_lives = lives
            self._last_max_lives = max_lives
            return
        if count is not None:
            count.setText(str(lives))
            slot = max(8, int(count.width() or 24))
            lives_px = int(getattr(self, "_lives_px", 0)) or max(22, int(round(self._icon_h * 0.54)))
            px = _fit_text_font_px(
                str(lives),
                slot,
                base_px=lives_px,
                min_px=16,
                max_px=max(lives_px, int(round(lives_px * 1.15))),
            )
            fg = _player_fg(color)
            count.setStyleSheet(
                f"font-size: {px}px; font-weight: 800; color: {fg};"
                f" border: none; background: transparent;"
            )
        self._refresh_life_heart(color)
        if wrap is not None and getattr(self, "_stack_mode", "active") != "compact":
            wrap.show()
        self._last_lives = lives
        self._last_max_lives = max_lives

    def set_row(
        self,
        number: Optional[object],
        lives: int,
        color: str,
        active: bool,
        *,
        is_killer: bool = False,
        finished: bool = False,
        act_prefix: str = "D",
        max_lives: int = 3,
        name: str = "",
    ) -> None:
        # Colored border only while this row is the active turn (not when dead).
        self._apply_card_chrome(
            color, active and not finished, finished, is_killer=is_killer
        )

        became_killer = (
            is_killer
            and not finished
            and self._last_is_killer is False
            and self._last_is_killer is not None
        )
        became_dead = finished and self._last_dead is False
        animate_loss = (
            self._last_lives is not None and int(lives) < int(self._last_lives)
        )

        show_killer = bool(is_killer) and not finished
        fg = _player_fg(color)
        name_s = str(name or "")
        if (
            color != self._last_color
            or finished != bool(self._last_dead)
            or self._icon_h != getattr(self, "_drawn_icon_h", None)
            or name_s != getattr(self, "_last_name", None)
        ):
            ini = _name_initials(name_s)
            if finished:
                pix = _initials_avatar_pixmap(name_s, self._icon_h, color)
                if pix is None:
                    pix = _player_icon_tinted(self._icon_h, fg)
                if pix is not None:
                    pix = _icon_with_strike(pix, thin=True)
                self.icon_lab.setText("")
                self.icon_lab.setStyleSheet("border: none; background: transparent;")
                if pix is not None and not pix.isNull():
                    self.icon_lab.setPixmap(pix)
                    self.icon_lab.setFixedSize(pix.width(), pix.height())
                else:
                    self.icon_lab.clear()
                    self.icon_lab.setFixedSize(self._icon_h, self._icon_h)
            elif ini:
                _apply_player_avatar(self.icon_lab, color, self._icon_h, name_s)
            else:
                pix = _player_icon_tinted(self._icon_h, fg)
                self.icon_lab.setText("")
                self.icon_lab.setStyleSheet("border: none; background: transparent;")
                if pix is not None and not pix.isNull():
                    self.icon_lab.setPixmap(pix)
                    self.icon_lab.setFixedSize(pix.width(), pix.height())
                else:
                    self.icon_lab.clear()
                    self.icon_lab.setFixedSize(self._icon_h, self._icon_h)
            self.icon_lab.setAlignment(Qt.AlignCenter)
            self._drawn_icon_h = self._icon_h
            self._last_name = name_s

        self._sync_killer_badge(show_killer, animate=became_killer)
        self._last_is_killer = bool(is_killer)

        if finished:
            self.num_lab.clear()
            self.num_lab.hide()
        else:
            if number is None:
                self.num_lab.setText("—")
            else:
                self.num_lab.setText(f"{act_prefix}{int(number)}")
            self._sync_num_font(self.num_lab.text() or "—", fg)
            self.num_lab.show()
        self._set_lives_display(
            int(lives),
            int(max_lives),
            color,
            finished=finished,
        )
        self._last_color = color
        self._last_number_key = None if number is None else int(number)

        self._last_dead = finished
        self._last_active = active
        if became_dead:
            QTimer.singleShot(0, self._pulse_icon)
        # Flash on life loss for live/main cards. Compact finished cards use
        # the death ghost-slide instead (geometry shake must never run on them).
        if animate_loss and not (finished and self._compact):
            QTimer.singleShot(0, self._animate_card_life_loss)


def _mark_count(marks: dict, key: int) -> int:
    if key in marks:
        return int(marks[key] or 0)
    if str(key) in marks:
        return int(marks[str(key)] or 0)
    if key == 25:
        for alt in ("bull", "B", 50):
            if alt in marks:
                return int(marks[alt] or 0)
            if str(alt) in marks:
                return int(marks[str(alt)] or 0)
    return 0


class AroundProgressBar(QWidget):
    """Continuous 0–21 progress under Around player cards (steps completed)."""

    EMPTY_COLOR = "#1a1a1e"
    STEPS = 21

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("AroundProgressBar")
        self._filled = 0
        self._fill_color = QColor("#ffffff")
        self.setFixedHeight(10)
        self.setMinimumWidth(40)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

    def set_bar_height(self, h: int) -> None:
        self.setFixedHeight(max(6, int(h)))
        self.update()

    def set_progress(self, completed: int, color: str) -> None:
        filled = max(0, min(self.STEPS, int(completed)))
        fg = QColor(_player_fg(color))
        if not fg.isValid():
            fg = QColor("#ffffff")
        if filled == self._filled and fg == self._fill_color:
            return
        self._filled = filled
        self._fill_color = fg
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        w = self.width()
        h = self.height()
        if w < 2 or h < 2:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        radius = max(2.0, h / 2.0)
        p.setBrush(QColor(self.EMPTY_COLOR))
        p.drawPath(
            _painter_round_rect_path(
                0.0, 0.0, float(w), float(h), tl=radius, tr=radius, br=radius, bl=radius
            )
        )
        if self._filled > 0:
            frac = self._filled / float(self.STEPS)
            fw = max(float(h), float(w) * frac)
            if self._filled >= self.STEPS:
                fw = float(w)
            p.setBrush(self._fill_color)
            p.drawPath(
                _painter_round_rect_path(
                    0.0, 0.0, fw, float(h), tl=radius, tr=radius, br=radius, bl=radius
                )
            )
        p.end()


class PlayerCard(QFrame):
    """Ikona | rezultat u istom redu (veća ikona, boja igrača)."""

    # Baseline metrike; aktivni = 2.85×, čekajući ≈0.92× (max 3 u redu).
    ICON_H = 100
    FONT_PX = 92
    MIN_H = 118
    MAX_H = 140
    ICON_BORDER = 4
    # Count up/down: ±1 po ticku; sporije radi čitljivosti (≈32ms → 20 bodova ≈ 640ms).
    # On Pi use fewer timer fires (same ±1 step, slightly longer total).
    SCORE_ANIM_MS = _SCORE_ANIM_MS
    # Distinct Halve It "cut in half" (scale-X), not countdown.
    HALVE_ANIM_MS = 420 if _WEAK_HW else 560
    AROUND_STEPS = 21

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._icon_h = self.ICON_H
        self._font_px = self.FONT_PX
        self._score_anim_enabled = True
        self._around_mode = False
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 8, 14, 8)
        root.setSpacing(0)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(14)
        row.setDirection(QHBoxLayout.LeftToRight)
        self._row_lay = row

        icon_box = self._icon_h + 2 * self.ICON_BORDER
        self.icon_frame = QFrame()
        self.icon_frame.setObjectName("PlayerIconFrame")
        self.icon_frame.setFixedSize(icon_box, icon_box)
        iframe = QVBoxLayout(self.icon_frame)
        iframe.setContentsMargins(0, 0, 0, 0)
        iframe.setSpacing(0)
        iframe.setAlignment(Qt.AlignCenter)
        self.icon_lab = QLabel()
        self.icon_lab.setAlignment(Qt.AlignCenter)
        self.icon_lab.setFixedSize(self._icon_h, self._icon_h)
        self.icon_lab.setStyleSheet("border: none; background: transparent;")
        iframe.addWidget(self.icon_lab, 0, Qt.AlignCenter)
        row.addWidget(self.icon_frame, 0, Qt.AlignVCenter)

        self.line = QLabel("")
        self.line.setObjectName("ScoreLine")
        self.line.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        row.addWidget(self.line, 0, Qt.AlignVCenter)
        self._tail_stretch = row.addStretch(1)
        root.addLayout(row)

        self.progress = AroundProgressBar()
        self.progress.hide()
        self.progress.setMaximumHeight(0)
        root.addWidget(self.progress)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(self.MAX_H)
        self._compact = False
        self._score_slot_w = 0
        self._player_index: Optional[int] = None
        self._last_color = ""
        self._last_score = ""
        self._last_active = None
        self._last_finished = None
        self._last_font_px_used: Optional[int] = None
        self._anim_target: Optional[int] = None
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(self.SCORE_ANIM_MS)
        self._anim_timer.timeout.connect(self._tick_score_anim)
        self._halve_anim: Optional[QVariantAnimation] = None
        self._halve_overlay: Optional[QLabel] = None
        self._score_fg = "#ffffff"
        self._refresh_score_slot()

    def set_score_anim_enabled(self, enabled: bool) -> None:
        self._score_anim_enabled = bool(enabled)
        if not self._score_anim_enabled:
            self._stop_score_anim()
            self._stop_halve_anim(commit=True)

    def set_around_progress(self, completed: Optional[int], color: str) -> None:
        """Show 0–21 bar for Around; pass None to hide (X01)."""
        root = self.layout()
        scale = self._icon_h / float(self.ICON_H)
        bar_h = max(7, int(round(10 * scale)))
        if completed is None:
            self._around_mode = False
            self.progress.hide()
            self.progress.setMaximumHeight(0)
            if root is not None:
                root.setSpacing(0)
            return
        self._around_mode = True
        self.progress.set_bar_height(bar_h)
        self.progress.setMaximumHeight(bar_h)
        self.progress.show()
        self.progress.set_progress(int(completed), color)
        if root is not None:
            root.setSpacing(max(4, int(round(6 * scale))))

    def set_metrics(self, icon_h: int, font_px: int, min_h: int, max_h: int) -> None:
        """Prilagodi veličinu kartice (ikona/font/visina) prema broju igrača."""
        extra = 0
        if self._around_mode:
            scale0 = max(1, int(icon_h)) / float(self.ICON_H)
            extra = max(7, int(round(10 * scale0))) + max(4, int(round(6 * scale0)))
        adj_h = int(max_h) + extra
        _ = min_h
        if (
            icon_h == self._icon_h
            and font_px == self._font_px
            and self.maximumHeight() == adj_h
            and self.minimumHeight() == adj_h
        ):
            return
        self._icon_h = int(icon_h)
        self._font_px = int(font_px)
        self.setFixedHeight(adj_h)
        scale = self._icon_h / float(self.ICON_H)
        m = max(8, int(round(8 * scale)))
        hpad = max(14, int(round(14 * scale)))
        root = self.layout()
        if root is not None:
            root.setContentsMargins(hpad, m, hpad, m)
            if self._around_mode:
                root.setSpacing(max(4, int(round(6 * scale))))
            else:
                root.setSpacing(0)
        if self._row_lay is not None:
            self._row_lay.setSpacing(max(14, int(round(14 * scale))))
        icon_box = self._icon_h + 2 * self.ICON_BORDER
        self.icon_frame.setFixedSize(icon_box, icon_box)
        self.icon_lab.setFixedSize(self._icon_h, self._icon_h)
        self._refresh_score_slot()
        if self._around_mode:
            bar_h = max(7, int(round(10 * scale)))
            self.progress.set_bar_height(bar_h)
            self.progress.setMaximumHeight(bar_h)
        # Forsiraj redraw ikone i stylesheeta pri sljedećem set_player.
        self._last_color = ""
        self._last_active = None
        self._last_finished = None
        self._last_font_px_used = None
        if self._compact:
            self.apply_compact_width(True)

    def _refresh_score_slot(self) -> None:
        font = QFont()
        font.setBold(True)
        font.setPixelSize(max(12, int(self._font_px)))
        self._score_slot_w = QFontMetrics(font).horizontalAdvance("000") + 6
        self.line.setFixedWidth(self._score_slot_w)
        self.line.setFixedHeight(max(24, int(round(self._font_px * 1.18))))

    def _sync_score_font(self, text: str) -> None:
        px = self._font_px_for_score(str(text))
        fg = getattr(self, "_score_fg", "#ffffff")
        self.line.setStyleSheet(
            f"QLabel#ScoreLine {{ color: {fg}; background: transparent; border: none;"
            f" font-size: {px}px; font-weight: 800; }}"
        )
        self._last_font_px_used = px

    def _set_line_text(self, text: str) -> None:
        self.line.setText(str(text))
        self._sync_score_font(str(text))

    def apply_compact_width(self, compact: bool, *, sample: str = "000") -> None:
        """Fixed širina za sample tekst (npr. '000' ili '#1') — layout ne skače."""
        self._compact = bool(compact)
        lay = self._row_lay
        if compact:
            if lay is not None:
                for i in range(lay.count() - 1, -1, -1):
                    item = lay.itemAt(i)
                    if item is not None and item.spacerItem() is not None:
                        lay.removeItem(item)
            if sample.startswith("#"):
                font = QFont()
                font.setBold(True)
                font.setPixelSize(max(12, int(self._font_px)))
                fm_w = QFontMetrics(font).horizontalAdvance(sample) + 10
                self.line.setFixedWidth(max(24, fm_w))
            else:
                fm_w = int(self._score_slot_w or 0)
                if fm_w <= 0:
                    self._refresh_score_slot()
                    fm_w = self._score_slot_w
                self.line.setFixedWidth(fm_w)
            scale = self._icon_h / float(self.ICON_H)
            pad_scale = 0.88 if sample.startswith("#") else 1.0
            hpad = max(10, int(round(14 * scale * pad_scale)))
            gap = max(12, int(round(16 * scale * pad_scale)))
            w = hpad + self._icon_h + 2 * self.ICON_BORDER + gap + fm_w + hpad + 8
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self.setFixedWidth(w)
        else:
            self._refresh_score_slot()
            self.setMinimumWidth(0)
            self.setMaximumWidth(16777215)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            if lay is not None:
                has_stretch = any(
                    lay.itemAt(i) is not None and lay.itemAt(i).spacerItem() is not None
                    for i in range(lay.count())
                )
                if not has_stretch:
                    lay.addStretch(1)

    def _stop_score_anim(self) -> None:
        self._anim_timer.stop()
        self._anim_target = None

    def reset_score_display(self) -> None:
        """Cancel in-flight score anims and drop cached display so a new match cannot tween from leftovers."""
        self._stop_score_anim()
        self._stop_halve_anim(commit=False)
        self._last_score = ""
        self._player_index = None
        self.line.setText("")

    def _cleanup_halve_overlay(self) -> None:
        ov = self._halve_overlay
        self._halve_overlay = None
        if ov is not None:
            ov.hide()
            ov.deleteLater()

    def _stop_halve_anim(self, *, commit: bool = False, target: Optional[int] = None) -> None:
        anim = self._halve_anim
        self._halve_anim = None
        if anim is not None:
            try:
                anim.stop()
            except Exception:
                pass
            anim.deleteLater()
        self._cleanup_halve_overlay()
        if self.line.graphicsEffect() is not None:
            self.line.setGraphicsEffect(None)
        self.line.setVisible(True)
        if commit and target is not None:
            self._set_line_text(str(target))

    def _tick_score_anim(self) -> None:
        if self._anim_target is None:
            self._stop_score_anim()
            return
        try:
            cur = int(self.line.text())
        except ValueError:
            self._set_line_text(str(self._anim_target))
            self._stop_score_anim()
            return
        if cur == self._anim_target:
            self._set_line_text(str(self._anim_target))
            self._stop_score_anim()
            return
        nxt = cur + 1 if cur < self._anim_target else cur - 1
        self._set_line_text(str(nxt))

    def _start_halve_anim(self, target: int) -> None:
        """Scale-X shrink of the old score, then reveal the halved value (~560ms)."""
        self._stop_score_anim()
        self._stop_halve_anim(commit=False)
        if self.line.width() < 4 or self.line.height() < 4:
            self._set_line_text(str(target))
            return
        if _WEAK_HW:
            self._start_halve_fade(target)
            return
        pix = self.line.grab()
        if pix.isNull() or pix.width() < 2:
            self._set_line_text(str(target))
            return
        # Keep layout stable: fade real label under overlay, swap text at mid-point.
        fade = QGraphicsOpacityEffect(self.line)
        fade.setOpacity(0.0)
        self.line.setGraphicsEffect(fade)
        self._set_line_text(str(target))

        ov = QLabel(self)
        ov.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        ov.setStyleSheet("background: transparent; border: none;")
        ov.setAlignment(Qt.AlignCenter)
        top_left = self.line.mapTo(self, QPoint(0, 0))
        full = QRect(top_left, self.line.size())
        ov.setGeometry(full)
        ov.setPixmap(pix)
        ov.show()
        ov.raise_()
        self._halve_overlay = ov

        anim = QVariantAnimation(self)
        anim.setDuration(self.HALVE_ANIM_MS)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.InOutCubic)
        mid = 0.45
        pw, ph = pix.width(), pix.height()
        revealed = {"done": False}

        def _reveal() -> None:
            if revealed["done"]:
                return
            revealed["done"] = True
            self._cleanup_halve_overlay()
            if self.line.graphicsEffect() is not None:
                self.line.setGraphicsEffect(None)
            self._set_line_text(str(target))

        def _on_value(v: object) -> None:
            t = float(v)
            if self._halve_overlay is None:
                return
            if t <= mid:
                sx = max(0.04, 1.0 - (t / mid))
                w = max(2, int(round(pw * sx)))
                scaled = pix.scaled(
                    w, ph, Qt.IgnoreAspectRatio, _PIXMAP_TRANSFORM
                )
                self._halve_overlay.setPixmap(scaled)
                x = full.center().x() - w // 2
                self._halve_overlay.setGeometry(x, full.y(), w, full.height())
            else:
                _reveal()

        def _on_finished() -> None:
            self._halve_anim = None
            _reveal()

        anim.valueChanged.connect(_on_value)
        anim.finished.connect(_on_finished)
        self._halve_anim = anim
        anim.start()

    def _start_halve_fade(self, target: int) -> None:
        """Pi: fade score text — bez grab()/rescale svakog framea."""
        fade = QGraphicsOpacityEffect(self.line)
        fade.setOpacity(1.0)
        self.line.setGraphicsEffect(fade)
        anim = QVariantAnimation(self)
        anim.setDuration(self.HALVE_ANIM_MS)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.InOutCubic)
        revealed = {"done": False}

        def _on_value(v: object) -> None:
            t = float(v)
            if t < 0.45:
                fade.setOpacity(max(0.0, 1.0 - t / 0.45))
                return
            if not revealed["done"]:
                revealed["done"] = True
                self._set_line_text(str(target))
            fade.setOpacity(min(1.0, (t - 0.45) / 0.55))

        def _on_finished() -> None:
            self._halve_anim = None
            if self.line.graphicsEffect() is not None:
                self.line.setGraphicsEffect(None)
            self._set_line_text(str(target))

        anim.valueChanged.connect(_on_value)
        anim.finished.connect(_on_finished)
        self._halve_anim = anim
        anim.start()

    def _font_px_for_score(self, score: str) -> int:
        """Scale score into the fixed 3-digit slot so 99 vs 501 ne mijenja karticu."""
        s = str(score)
        slot = int(self.line.width() or getattr(self, "_score_slot_w", 0) or 0)
        if slot <= 0:
            self._refresh_score_slot()
            slot = int(self._score_slot_w)
        base = max(12, int(self._font_px))
        n = len(s)
        if s.upper() == "BULL":
            max_px = min(base, int(round(base * 0.78)))
        elif n <= 1:
            max_px = int(round(base * 1.28))
        elif n == 2:
            max_px = int(round(base * 1.18))
        else:
            max_px = base
        return _fit_text_font_px(s, slot, base_px=base, min_px=12, max_px=max_px)

    def set_player(
        self,
        label: str,
        score: str,
        color: str,
        active: bool,
        player_index: Optional[int] = None,
        finished: bool = False,
        score_halved: bool = False,
        name: str = "",
    ) -> None:
        _ = label
        # Promjena igrača na istoj kartici — score odmah, bez countdown animacije.
        if player_index is not None and player_index != self._player_index:
            self._stop_score_anim()
            self._stop_halve_anim(commit=True)
            self._player_index = int(player_index)
            self._set_line_text(str(score))
            self._last_score = str(score)
            self._last_color = ""
            self._last_active = None
            self._last_finished = None
            self._last_font_px_used = None
            self._last_name = None
        if not hasattr(self, "_last_finished"):
            self._last_finished = None
        score_s = str(score)
        name_s = str(name or "")
        font_px = self._font_px_for_score(score_s)
        if (
            color != self._last_color
            or active != self._last_active
            or score_s != self._last_score
            or finished != self._last_finished
            or font_px != self._last_font_px_used
            or name_s != getattr(self, "_last_name", None)
        ):
            fg = _player_fg(color)
            self._score_fg = fg
            if finished:
                border = fg
                bg = "#181818"
                border_w = 3
                icon_border = fg
                icon_bw = self.ICON_BORDER
            elif active:
                border = fg
                bg = "#1c1218"
                border_w = 4 if self._icon_h <= 130 else 5
                icon_border = fg
                icon_bw = self.ICON_BORDER
            else:
                border = "transparent"
                bg = "#141418" if self._around_mode else "transparent"
                border_w = 4 if self._icon_h <= 130 else 5
                icon_border = "transparent"
                icon_bw = 0
            radius = max(12, int(round(16 * self._icon_h / float(self.ICON_H))))
            icon_radius = max(12, int(round(16 * self._icon_h / float(self.ICON_H))))
            self.setStyleSheet(
                f"QFrame {{ background-color: {bg}; border: {border_w}px solid {border};"
                f" border-radius: {radius}px; }}"
                f" QFrame#PlayerIconFrame {{"
                f" background: transparent; border: {icon_bw}px solid {icon_border};"
                f" border-radius: {icon_radius}px; }}"
                f" QLabel#ScoreLine {{ color: {fg}; background: transparent; border: none;"
                f" font-weight: 800; }}"
                f" QLabel {{ border: none; background: transparent; }}"
                f" QWidget#AroundProgressBar {{ background: transparent; border: none; }}"
            )
            self._last_active = active
            self._last_finished = finished
            self._sync_score_font(self.line.text() or score_s)
        if score_s != self._last_score:
            self._last_score = score_s
            self._apply_score_text(score_s, halved=bool(score_halved))
        if color != self._last_color or name_s != getattr(self, "_last_name", None):
            pix = _player_icon_tinted(self._icon_h, _player_fg(color))
            _apply_player_avatar(self.icon_lab, color, self._icon_h, name_s, pix=pix)
            self._last_color = color
            self._last_name = name_s

    def _apply_score_text(self, score: str, *, halved: bool = False) -> None:
        """±1 count anim; Halve miss uses distinct scale-X cut; else set immediately."""
        if not self._score_anim_enabled:
            self._stop_score_anim()
            self._stop_halve_anim(commit=True)
            self._set_line_text(score)
            return
        try:
            target = int(score)
        except ValueError:
            self._stop_score_anim()
            self._stop_halve_anim(commit=True)
            self._set_line_text(score)
            return
        try:
            current = int(self.line.text())
        except ValueError:
            current = target
        if self.line.text() == "" or current == target:
            self._stop_score_anim()
            self._stop_halve_anim(commit=True)
            self._set_line_text(str(target))
            return
        # Halve It miss: distinct scale-X cut (flag from game / mid-turn preview).
        if target < current and halved:
            self._start_halve_anim(target)
            return
        # Count up or down (±1 per tick) — Halve adds, X01 remaining, undo, etc.
        if target != current:
            self._stop_halve_anim(commit=False)
            self._anim_target = target
            if not self._anim_timer.isActive():
                self._anim_timer.start()
            return
        self._stop_score_anim()
        self._set_line_text(str(target))


class MqttBridge(QObject):
    command = Signal(str)


class MainWindow(QMainWindow):
    def __init__(self, session: KioskSession) -> None:
        super().__init__()
        self.session = session
        self.setWindowTitle("Pikado")
        self.setObjectName("Root")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._hit_tags_prev: List[str] = ["-", "-", "-"]
        self._hit_anims: List[Optional[QSequentialAnimationGroup]] = [None, None, None]
        self._hit_geo_restore: List[Optional[QRect]] = [None, None, None]
        self._player_cards: List[PlayerCard] = []
        self._x01_cards_by_pi: Dict[int, PlayerCard] = {}
        self._x01_waiting_wrap: Optional[QWidget] = None
        self._x01_waiting_layout: Optional[QHBoxLayout] = None
        self._x01_last_active_pi: Optional[int] = None
        self._x01_turn_anim: Optional[QParallelAnimationGroup] = None
        self._x01_turn_ghosts: List[QWidget] = []
        self._cricket_rows_by_pi: Dict[int, CricketRow] = {}
        self._cricket_hdr: Optional[QWidget] = None
        self._cricket_round_lab: Optional[QLabel] = None
        self._cricket_last_active_pi: Optional[int] = None
        self._cricket_turn_anim: Optional[QParallelAnimationGroup] = None
        self._cricket_turn_ghosts: List[QWidget] = []
        self._killer_rows_by_pi: Dict[int, KillerRow] = {}
        self._killer_hdr: Optional[QWidget] = None
        self._killer_waiting_wrap: Optional[QWidget] = None
        self._killer_waiting_layout: Optional[QHBoxLayout] = None
        self._killer_last_active_pi: Optional[int] = None
        self._killer_finished_pis: set = set()
        self._killer_turn_anim: Optional[QParallelAnimationGroup] = None
        self._killer_turn_ghosts: List[QWidget] = []
        self._winner_title: Optional[QLabel] = None
        self._last_cal_view_ms: float = 0.0
        self._i18n_lang: Optional[str] = None
        self._i18n_screen: Optional[str] = None
        self._mqtt_client = None
        self._pin_mode: str = ""
        self._pin_buf: str = ""
        self._pin_pending: str = ""
        self._locking_geo = False

        root = QWidget()
        root.setObjectName("Root")
        root.setAttribute(Qt.WA_StyledBackground, True)
        root.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setCentralWidget(root)

        # Wallpaper overlay (transparent PNG preko gradient pozadine)
        self._wallpaper = QLabel(root)
        self._wallpaper.setObjectName("Wallpaper")
        self._wallpaper.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._wallpaper.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self._wallpaper_pix: Optional[QPixmap] = None
        self._wallpaper_laid_size: Optional[Tuple[int, int]] = None
        self._wallpaper_scaled: Optional[QPixmap] = None
        wp = QPixmap(_asset_path("assets", "background.png"))
        if not wp.isNull():
            self._wallpaper_pix = wp
            self._wallpaper.show()
            self._wallpaper.lower()
        else:
            self._wallpaper.hide()

        layout = QVBoxLayout(root)
        # Gornji/donji rub — HDMI overscan + zračnost na vrhu
        layout.setContentsMargins(36, 52, 36, 72)
        layout.setSpacing(0)

        self.stack = QStackedWidget()
        # Expanding fills the window; window min/max lock stops pixmap sizeHints from stretching kiosk.
        self.stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.stack, 1)

        self.pages = {
            "standby": self._build_standby(),
            "select_game": self._build_select_game(),
            "rules_x01": self._build_rules_x01(),
            "rules_cricket": self._build_rules_cricket(),
            "rules_killer": self._build_rules_killer(),
            "rules_around": self._build_rules_around(),
            "rules_halve": self._build_rules_halve(),
            "tutorial_x01": self._build_tutorial("x01"),
            "tutorial_cricket": self._build_tutorial("cricket"),
            "tutorial_killer": self._build_tutorial("killer"),
            "tutorial_around": self._build_tutorial("around"),
            "tutorial_halve": self._build_tutorial("halve"),
            "select_players": self._build_select_players(),
            "clear_board": self._build_clear_board(),
            "playing": self._build_playing(),
            "exit_confirm": self._build_exit_confirm(),
            "calibration": self._build_calibration(),
            "settings": self._build_settings(),
        }
        for _name, w in self.pages.items():
            w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            self.stack.addWidget(w)

        # Keep wallpaper under the page stack (never over turn ghosts).
        if self._wallpaper.isVisible() or self._wallpaper_pix is not None:
            self._wallpaper.lower()
            self._wallpaper.stackUnder(self.stack)

        # Plutajući gumb postavki — lijevi kut, nevidljiv ali klikabilan (samo tko zna gdje)
        self.settings_fab = QPushButton("", root)
        self.settings_fab.setObjectName("SettingsFab")
        self.settings_fab.setFixedSize(100, 100)
        self.settings_fab.setFocusPolicy(Qt.NoFocus)
        self.settings_fab.setCursor(Qt.ArrowCursor)
        self.settings_fab.setFlat(True)
        self.settings_fab.setText("")
        self.settings_fab.setStyleSheet(
            "QPushButton#SettingsFab {"
            "  background: transparent;"
            "  background-color: rgba(0,0,0,0);"
            "  color: transparent;"
            "  border: none;"
            "  border-radius: 0;"
            "  font-size: 1px;"
            "  padding: 0;"
            "  min-height: 0;"
            "  min-width: 0;"
            "  outline: none;"
            "}"
            "QPushButton#SettingsFab:hover, QPushButton#SettingsFab:focus,"
            "QPushButton#SettingsFab:pressed {"
            "  background: transparent;"
            "  background-color: rgba(0,0,0,0);"
            "  color: transparent;"
            "  border: none;"
            "  outline: none;"
            "}"
        )
        self.settings_fab.clicked.connect(self._request_settings)

        self._build_pin_overlay(root)
        self._build_quit_overlay(root)
        self._build_idle_overlay(root)
        self._build_killer_bull_overlay(root)
        self._build_ingame_cal_overlay(root)
        self._build_name_overlay(root)

        self._wire_actions(root)
        self._last_ui_sig: Optional[str] = None
        self._apply_theme()
        self._refresh_player_count_icons()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start(_UI_TICK_MS)
        if _WEAK_HW:
            print(
                f"[ui] weak-hw path on (tick={_UI_TICK_MS}ms, anim={_TURN_ANIM_MS}ms, vision=bg)",
                flush=True,
            )

        QShortcut(QKeySequence("2"), self, activated=self._request_settings)
        QShortcut(QKeySequence("3"), self, activated=self._mqtt_on)
        QShortcut(QKeySequence("4"), self, activated=self._save_kiosk_screenshot)
        QShortcut(QKeySequence("Q"), self, activated=self.close)
        QShortcut(QKeySequence("Escape"), self, activated=self._on_escape)
        QShortcut(QKeySequence("D"), self, activated=self._force_detect)
        QShortcut(QKeySequence("T"), self, activated=self._toggle_cal_topdown)
        QShortcut(QKeySequence("O"), self, activated=self._toggle_cal_overlay)
        QShortcut(QKeySequence("B"), self, activated=self._cal_detect_hotkey)
        QShortcut(QKeySequence("J"), self, activated=self._fake_cams_next_hit)
        QShortcut(QKeySequence("L"), self, activated=self._fake_cams_next_player)

        self.refresh()
        self._place_chrome()
        self._idle_last = time.monotonic()
        self._idle_deadline: Optional[float] = None
        self._idle_hits_sig = None
        self._power_pending = ""

    def _apply_theme(self) -> None:
        s = get_settings()
        qss = build_theme_qss(APP_QSS, bg_color=s.bg_color, button_color=s.button_color)
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(qss)
        # Forsiraj repaint root pozadine (inače ostane stari gradijent).
        self.setStyleSheet("")
        root = self.centralWidget()
        if root is not None:
            root.setAttribute(Qt.WA_StyledBackground, True)
            root.update()
        self.update()
        self._refresh_player_count_icons()

    def _kiosk_screen_geometry(self):
        screen = self.screen()
        if screen is None:
            app = QApplication.instance()
            if app is not None:
                screen = app.primaryScreen()
        if screen is None:
            return None
        return screen.geometry()

    def _lock_kiosk_display(self, *, force: bool = False) -> None:
        """Keep the kiosk window glued to the native screen — no pinch/drag/resize."""
        if not bool(getattr(self.session, "kiosk", False)):
            return
        if getattr(self, "_locking_geo", False):
            return
        geo = self._kiosk_screen_geometry()
        if geo is None or geo.width() < 2 or geo.height() < 2:
            return
        want_w, want_h = geo.width(), geo.height()
        cur = self.geometry()
        already = (
            self.isFullScreen()
            and cur.x() == geo.x()
            and cur.y() == geo.y()
            and cur.width() == want_w
            and cur.height() == want_h
            and self.minimumWidth() == want_w
            and self.minimumHeight() == want_h
            and self.maximumWidth() == want_w
            and self.maximumHeight() == want_h
        )
        if already and not force:
            return
        self._locking_geo = True
        try:
            if not self.isFullScreen():
                self.showFullScreen()
            if self.minimumWidth() != want_w or self.minimumHeight() != want_h:
                self.setMinimumSize(want_w, want_h)
            if self.maximumWidth() != want_w or self.maximumHeight() != want_h:
                self.setMaximumSize(want_w, want_h)
            if self.geometry() != geo:
                self.setGeometry(geo)
            st = self.windowState()
            if not (st & Qt.WindowFullScreen):
                self.setWindowState(st | Qt.WindowFullScreen)
        finally:
            self._locking_geo = False

    def _swallow_kiosk_gesture(self, event) -> bool:
        if not bool(getattr(self.session, "kiosk", False)) or event is None:
            return False
        et = getattr(QEvent, "Type", QEvent)
        for name in ("NativeGesture", "Gesture", "GestureOverride"):
            val = getattr(et, name, None)
            if val is not None and event.type() == val:
                return True
        return False

    def event(self, event):  # noqa: N802
        if self._swallow_kiosk_gesture(event):
            self._lock_kiosk_display()
            return True
        return super().event(event)

    def eventFilter(self, watched, event):  # noqa: N802
        if self._swallow_kiosk_gesture(event):
            self._lock_kiosk_display()
            return True
        if self._is_user_pointer_event(event):
            idle_up = hasattr(self, "idle_overlay") and self.idle_overlay.isVisible()
            if not idle_up:
                self._idle_note_activity()
        return super().eventFilter(watched, event)

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if event is None:
            return
        ws = getattr(QEvent.Type, "WindowStateChange", None)
        if ws is None:
            ws = getattr(QEvent, "WindowStateChange", None)
        if ws is not None and event.type() == ws:
            self._lock_kiosk_display(force=True)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._lock_kiosk_display(force=True)

    def moveEvent(self, event) -> None:  # noqa: N802
        super().moveEvent(event)
        self._lock_kiosk_display()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._lock_kiosk_display()
        self._place_chrome()
        if getattr(self.session, "screen", "") == "select_players":
            self._refresh_player_count_icons()

    def _layout_wallpaper(self) -> None:
        root = self.centralWidget()
        if root is None or not hasattr(self, "_wallpaper"):
            return
        pix = getattr(self, "_wallpaper_pix", None)
        if pix is None or pix.isNull():
            self._wallpaper.hide()
            self._wallpaper_laid_size = None
            self._wallpaper_scaled = None
            return
        w = max(1, root.width())
        h = max(1, root.height())
        self._wallpaper.setGeometry(0, 0, w, h)
        # Scale once per size — never SmoothTransformation-rescale on every UI tick.
        if self._wallpaper_laid_size == (w, h) and self._wallpaper_scaled is not None:
            if self._wallpaper.pixmap() is None or self._wallpaper.pixmap().isNull():
                self._wallpaper.setPixmap(self._wallpaper_scaled)
            self._wallpaper.show()
            self._wallpaper.lower()
            if hasattr(self, "stack") and self.stack is not None:
                self._wallpaper.stackUnder(self.stack)
            return
        scaled = pix.scaled(w, h, Qt.KeepAspectRatioByExpanding, _PIXMAP_TRANSFORM)
        # Centriraj crop ako je veći od prozora
        x = max(0, (scaled.width() - w) // 2)
        y = max(0, (scaled.height() - h) // 2)
        cropped = scaled.copy(x, y, w, h)
        self._wallpaper_laid_size = (w, h)
        self._wallpaper_scaled = cropped
        self._wallpaper.setPixmap(cropped)
        self._wallpaper.show()
        # Always keep wallpaper behind UI (never raise during turn anims).
        self._wallpaper.lower()
        if hasattr(self, "stack") and self.stack is not None:
            self._wallpaper.stackUnder(self.stack)

    def _place_chrome(self, *, wallpaper: bool = True) -> None:
        root = self.centralWidget()
        if root is None:
            return
        if wallpaper:
            self._layout_wallpaper()
        if hasattr(self, "ingame_cal_overlay") and self.ingame_cal_overlay.isVisible():
            self.ingame_cal_overlay.setGeometry(0, 0, root.width(), root.height())
            self.ingame_cal_overlay.raise_()
            self._layout_ingame_cal_stills()
        if hasattr(self, "pin_overlay") and self.pin_overlay.isVisible():
            self.pin_overlay.setGeometry(0, 0, root.width(), root.height())
            self.pin_overlay.raise_()
        if hasattr(self, "quit_overlay") and self.quit_overlay.isVisible():
            self.quit_overlay.setGeometry(0, 0, root.width(), root.height())
            self.quit_overlay.raise_()
        if hasattr(self, "name_overlay") and self.name_overlay.isVisible():
            self.name_overlay.setGeometry(0, 0, root.width(), root.height())
            self.name_overlay.raise_()
        if hasattr(self, "killer_bull_overlay") and self.killer_bull_overlay.isVisible():
            self.killer_bull_overlay.setGeometry(0, 0, root.width(), root.height())
            self.killer_bull_overlay.raise_()
        m = 18
        if hasattr(self, "settings_fab"):
            self.settings_fab.move(m, m)
            self.settings_fab.raise_()
        if hasattr(self, "ingame_cal_overlay") and self.ingame_cal_overlay.isVisible():
            self.ingame_cal_overlay.raise_()
        if hasattr(self, "name_overlay") and self.name_overlay.isVisible():
            self.name_overlay.raise_()
        if hasattr(self, "idle_overlay") and self.idle_overlay.isVisible():
            self.idle_overlay.setGeometry(0, 0, root.width(), root.height())
            self.idle_overlay.raise_()

    def _update_settings_fab_visibility(self) -> None:
        if not hasattr(self, "settings_fab"):
            return
        screen = self.session.screen
        pin_up = hasattr(self, "pin_overlay") and self.pin_overlay.isVisible()
        quit_up = hasattr(self, "quit_overlay") and self.quit_overlay.isVisible()
        idle_up = hasattr(self, "idle_overlay") and self.idle_overlay.isVisible()
        name_up = hasattr(self, "name_overlay") and self.name_overlay.isVisible()
        cal_up = bool(self.session.is_in_game_calibrating())
        show = (
            screen not in ("settings",)
            and not pin_up
            and not quit_up
            and not idle_up
            and not cal_up
            and not name_up
        )
        self.settings_fab.setVisible(show)
        if show:
            self.settings_fab.raise_()

    def _wire_actions(self, root: QWidget) -> None:
        for btn in root.findChildren(QPushButton):
            action = btn.property("action")
            if action:
                btn.clicked.connect(lambda _=False, a=action: self._act(a))

    def _act(self, action: str) -> None:
        idle_up = hasattr(self, "idle_overlay") and self.idle_overlay.isVisible()
        if action in ("idle_quit", "idle_continue", "confirm_quit_yes", "confirm_quit_no"):
            return
        if self._idle_active() and not idle_up:
            self._idle_note_activity()
        if action == "settings_open":
            self._request_settings()
            return
        if action == "settings_change_password":
            self._show_pin("set_new")
            return
        if action == "settings_quit":
            self._show_quit_confirm("exit")
            return
        if str(action).startswith("player_slot:"):
            try:
                slot = int(action.split(":", 1)[1])
            except ValueError:
                return
            self._open_name_editor(slot)
            return
        if action in ("ingame_cal_confirm_empty", "ingame_cal_retry"):
            self.session.handle_action(action)
            self._last_ui_sig = None
            self._sync_ingame_cal_overlay()
            QApplication.processEvents()
            QTimer.singleShot(200, self._run_deferred_ingame_cal)
            return
        self.session.handle_action(action)
        self._last_ui_sig = None
        if str(action).startswith("settings_"):
            self._apply_theme()
        self.refresh()
        if action == "clear_board_confirm":
            self.clear_status.setText(get_settings().t("calibrating"))
            self.clear_status.update()
            self.clear_status.repaint()
            self.repaint()
            QApplication.processEvents()
            QTimer.singleShot(600, self._run_deferred_clear_board_calibrate)

    def _on_settings_combo(self, kind: str) -> None:
        s = get_settings()
        pending_power = ""
        if kind == "lang":
            code = self.settings_lang.currentData()
            if code and code != s.language:
                update_settings(language=str(code))
        elif kind == "bg":
            color = self.settings_bg.currentData()
            if color and str(color).lower() != s.bg_color.lower():
                update_settings(bg_color=str(color))
                self._apply_theme()
        elif kind == "btn":
            color = self.settings_btn.currentData()
            if color and str(color).lower() != s.button_color.lower():
                update_settings(button_color=str(color))
                self._apply_theme()
        elif kind == "start":
            mode = self.settings_start.currentData()
            if mode and mode != s.start_mode:
                update_settings(start_mode=str(mode))
                self._ensure_mqtt_client()
                self.session._apply_start_mode_home()
        elif kind == "device":
            device = self.settings_device.currentData()
            if device and str(device) != s.device_id:
                update_settings(device_id=str(device))
        elif kind == "league_qr":
            mode = self.settings_league_qr.currentData()
            want = str(mode) != "off"
            if want != bool(s.league_qr_enabled):
                update_settings(league_qr_enabled=want)
        elif kind == "auto_calibrate":
            mode = self.settings_auto_cal.currentData()
            want = str(mode) != "off"
            if want != bool(s.auto_calibrate):
                update_settings(auto_calibrate=want)
        elif kind == "power":
            pending_power = str(self.settings_power.currentData() or "")
            self.settings_power.set_current_data("")
        self._last_ui_sig = None
        if kind == "lang":
            self._i18n_lang = None
            self._apply_i18n()
        self._sync_settings_combos()
        self.refresh()
        if pending_power in ("shutdown", "reboot", "exit"):
            self._show_quit_confirm(pending_power)

    def _open_settings(self) -> None:
        self.session.enter_settings()
        self._last_ui_sig = None
        self._i18n_screen = None
        self._sync_settings_combos()
        self.refresh()

    def _request_settings(self) -> None:
        if self.session.screen == "settings":
            return
        if self.session.is_in_game_calibrating():
            return
        if hasattr(self, "name_overlay") and self.name_overlay.isVisible():
            return
        if hasattr(self, "pin_overlay") and self.pin_overlay.isVisible():
            return
        if hasattr(self, "quit_overlay") and self.quit_overlay.isVisible():
            return
        self._show_pin("unlock")

    def _build_pin_overlay(self, root: QWidget) -> None:
        self.pin_overlay = QWidget(root)
        self.pin_overlay.setObjectName("PinOverlay")
        self.pin_overlay.hide()
        self.pin_overlay.setStyleSheet(
            "QWidget#PinOverlay { background-color: rgba(0,0,0,190); }"
        )
        ol = QVBoxLayout(self.pin_overlay)
        ol.setContentsMargins(40, 40, 40, 40)
        ol.setAlignment(Qt.AlignCenter)

        card = QFrame()
        card.setObjectName("Card")
        card.setFixedWidth(520)
        cl = QVBoxLayout(card)
        cl.setSpacing(14)
        cl.setContentsMargins(28, 24, 28, 24)

        self.pin_title = QLabel("")
        self.pin_title.setObjectName("Title")
        self.pin_title.setAlignment(Qt.AlignCenter)
        self.pin_title.setStyleSheet("font-size: 42px; font-weight: 800;")
        cl.addWidget(self.pin_title)

        self.pin_display = QLabel("")
        self.pin_display.setAlignment(Qt.AlignCenter)
        self.pin_display.setStyleSheet(
            "font-size: 56px; font-weight: 800; color: #ffe14a; letter-spacing: 10px; "
            "min-height: 72px;"
        )
        cl.addWidget(self.pin_display)

        self.pin_status = QLabel("")
        self.pin_status.setAlignment(Qt.AlignCenter)
        self.pin_status.setStyleSheet("font-size: 28px; color: #ff6a6a; min-height: 36px;")
        cl.addWidget(self.pin_status)

        pad = QGridLayout()
        pad.setSpacing(10)
        for i, digit in enumerate("123456789"):
            b = QPushButton(digit)
            b.setObjectName("KeyNum")
            b.setFixedSize(120, 72)
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(lambda _=False, d=digit: self._pin_digit(d))
            pad.addWidget(b, i // 3, i % 3)
        btn_clr = QPushButton("⌫")
        btn_clr.setObjectName("KeyNum")
        btn_clr.setFixedSize(120, 72)
        btn_clr.setFocusPolicy(Qt.NoFocus)
        btn_clr.clicked.connect(self._pin_backspace)
        pad.addWidget(btn_clr, 3, 0)
        btn0 = QPushButton("0")
        btn0.setObjectName("KeyNum")
        btn0.setFixedSize(120, 72)
        btn0.setFocusPolicy(Qt.NoFocus)
        btn0.clicked.connect(lambda _=False: self._pin_digit("0"))
        pad.addWidget(btn0, 3, 1)
        btn_ok = QPushButton("OK")
        btn_ok.setObjectName("KeyNum")
        btn_ok.setFixedSize(120, 72)
        btn_ok.setFocusPolicy(Qt.NoFocus)
        btn_ok.clicked.connect(self._pin_submit)
        self.pin_ok_btn = btn_ok
        pad.addWidget(btn_ok, 3, 2)
        cl.addLayout(pad)

        self.pin_cancel_btn = QPushButton("ODUSTANI")
        self.pin_cancel_btn.setObjectName("Danger")
        self.pin_cancel_btn.setMinimumHeight(72)
        self.pin_cancel_btn.setFocusPolicy(Qt.NoFocus)
        self.pin_cancel_btn.clicked.connect(self._hide_pin)
        cl.addWidget(self.pin_cancel_btn)

        ol.addWidget(card, 0, Qt.AlignCenter)

    def _build_quit_overlay(self, root: QWidget) -> None:
        self.quit_overlay = QWidget(root)
        self.quit_overlay.setObjectName("QuitOverlay")
        self.quit_overlay.hide()
        self.quit_overlay.setStyleSheet(
            "QWidget#QuitOverlay { background-color: rgba(0,0,0,190); }"
        )
        ol = QVBoxLayout(self.quit_overlay)
        ol.setContentsMargins(40, 40, 40, 40)
        ol.setAlignment(Qt.AlignCenter)

        card = QFrame()
        card.setObjectName("Card")
        card.setMinimumWidth(720)
        card.setMaximumWidth(900)
        cl = QVBoxLayout(card)
        cl.setSpacing(28)
        cl.setContentsMargins(36, 32, 36, 32)

        self.quit_title = QLabel("")
        self.quit_title.setObjectName("SettingsTitle")
        self.quit_title.setAlignment(Qt.AlignCenter)
        self.quit_title.setWordWrap(True)
        cl.addWidget(self.quit_title)

        row = QHBoxLayout()
        row.setSpacing(16)
        self.quit_yes_btn = _btn("DA", "confirm_quit_yes", role="Danger", tall=True)
        self.quit_no_btn = _btn("NE", "confirm_quit_no", role="Primary", tall=True)
        self.quit_yes_btn.setFocusPolicy(Qt.NoFocus)
        self.quit_no_btn.setFocusPolicy(Qt.NoFocus)
        self.quit_yes_btn.clicked.connect(self._confirm_power_action)
        self.quit_no_btn.clicked.connect(self._hide_quit_confirm)
        row.addWidget(self.quit_yes_btn, 1)
        row.addWidget(self.quit_no_btn, 1)
        cl.addLayout(row)
        ol.addWidget(card, 0, Qt.AlignCenter)

    def _build_idle_overlay(self, root: QWidget) -> None:
        self.idle_overlay = QWidget(root)
        self.idle_overlay.setObjectName("IdleOverlay")
        self.idle_overlay.hide()
        self.idle_overlay.setAttribute(Qt.WA_StyledBackground, True)
        self.idle_overlay.setStyleSheet(
            "QWidget#IdleOverlay { background-color: rgba(0,0,0,190); }"
        )
        ol = QVBoxLayout(self.idle_overlay)
        ol.setContentsMargins(40, 40, 40, 40)
        ol.setAlignment(Qt.AlignCenter)

        card = QFrame()
        card.setObjectName("Card")
        card.setMinimumWidth(1080)
        card.setMaximumWidth(1280)
        cl = QVBoxLayout(card)
        cl.setSpacing(22)
        cl.setContentsMargins(40, 36, 40, 36)

        self.idle_title = QLabel("")
        self.idle_title.setObjectName("SettingsTitle")
        self.idle_title.setAlignment(Qt.AlignCenter)
        self.idle_title.setWordWrap(True)
        cl.addWidget(self.idle_title)

        self.idle_s1 = QLabel("")
        self.idle_s1.setObjectName("Subtitle")
        self.idle_s1.setAlignment(Qt.AlignCenter)
        self.idle_s1.setWordWrap(True)
        cl.addWidget(self.idle_s1)

        self.idle_s2 = QLabel("")
        self.idle_s2.hide()

        row = QHBoxLayout()
        row.setSpacing(20)
        self.idle_quit_btn = _btn("UGASI", "idle_quit", role="Danger", tall=True)
        self.idle_continue_btn = _btn("NASTAVI", "idle_continue", role="Primary", tall=True)
        self.idle_quit_btn.setFocusPolicy(Qt.NoFocus)
        self.idle_continue_btn.setFocusPolicy(Qt.NoFocus)
        self.idle_quit_btn.setMinimumHeight(96)
        self.idle_continue_btn.setMinimumHeight(96)
        self.idle_continue_btn.setMinimumWidth(420)
        self.idle_quit_btn.setStyleSheet(
            "QPushButton#Danger { font-size: 40px; min-height: 96px; "
            "padding: 10px 22px; }"
        )
        self.idle_continue_btn.setStyleSheet(
            "QPushButton#Primary { font-size: 40px; min-height: 96px; "
            "padding: 10px 22px; }"
        )
        self.idle_quit_btn.clicked.connect(self._idle_quit_game)
        self.idle_continue_btn.clicked.connect(self._idle_continue_game)
        row.addWidget(self.idle_quit_btn, 1)
        row.addWidget(self.idle_continue_btn, 1)
        cl.addLayout(row)
        ol.addWidget(card, 0, Qt.AlignCenter)

    def _build_name_overlay(self, root: QWidget) -> None:
        self._name_slot: Optional[int] = None
        self._name_buf = ""
        self._name_was_selected = False
        self.name_overlay = QWidget(root)
        self.name_overlay.setObjectName("NameOverlay")
        self.name_overlay.hide()
        self.name_overlay.setAttribute(Qt.WA_StyledBackground, True)
        self.name_overlay.setStyleSheet(
            "QWidget#NameOverlay { background-color: rgba(0,0,0,200); }"
        )
        ol = QVBoxLayout(self.name_overlay)
        ol.setContentsMargins(24, 20, 24, 20)
        ol.setAlignment(Qt.AlignCenter)

        card = QFrame()
        card.setObjectName("Card")
        card.setMinimumWidth(1280)
        card.setMaximumWidth(1480)
        cl = QVBoxLayout(card)
        cl.setSpacing(14)
        cl.setContentsMargins(24, 20, 24, 20)

        self.name_title = QLabel("")
        self.name_title.setObjectName("Title")
        self.name_title.setAlignment(Qt.AlignCenter)
        self.name_title.setStyleSheet("font-size: 42px; font-weight: 800;")
        cl.addWidget(self.name_title)

        self.name_display = QLabel("")
        self.name_display.setAlignment(Qt.AlignCenter)
        self.name_display.setStyleSheet(
            "font-size: 56px; font-weight: 800; color: #ffe14a; min-height: 72px;"
        )
        cl.addWidget(self.name_display)

        key_w, key_h = 112, 88
        key_gap = 10

        def _letter_key(ch: str) -> QPushButton:
            b = QPushButton(ch)
            b.setObjectName("KeyNum")
            b.setFixedSize(key_w, key_h)
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(lambda _=False, c=ch: self._name_type(c))
            return b

        def _style_action(b: QPushButton, obj: str, font_px: int) -> None:
            b.setObjectName(obj)
            b.setFixedSize(key_w, key_h)
            b.setFocusPolicy(Qt.NoFocus)
            b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            b.setStyleSheet(
                f"QPushButton#{obj} {{ min-width: {key_w}px; max-width: {key_w}px; "
                f"min-height: {key_h}px; max-height: {key_h}px; padding: 0px; "
                f"font-size: {font_px}px; font-weight: 800; }}"
            )

        # Row 1: QWERTYUIOP
        r1 = QHBoxLayout()
        r1.setSpacing(key_gap)
        r1.addStretch(1)
        for ch in "QWERTYUIOP":
            r1.addWidget(_letter_key(ch), 0)
        r1.addStretch(1)
        cl.addLayout(r1)

        # Row 2: ASDFGHJKL + backspace (10 keys, same width as row 1)
        r2 = QHBoxLayout()
        r2.setSpacing(key_gap)
        r2.addStretch(1)
        for ch in "ASDFGHJKL":
            r2.addWidget(_letter_key(ch), 0)
        self.name_bs_btn = QPushButton("⌫")
        _style_action(self.name_bs_btn, "KeyNum", 40)
        self.name_bs_btn.clicked.connect(self._name_backspace)
        r2.addWidget(self.name_bs_btn, 0)
        r2.addStretch(1)
        cl.addLayout(r2)

        # Row 3: ZXCVBNM only
        r3 = QHBoxLayout()
        r3.setSpacing(key_gap)
        r3.addStretch(1)
        for ch in "ZXCVBNM":
            r3.addWidget(_letter_key(ch), 0)
        r3.addStretch(1)
        cl.addLayout(r3)

        actions = QHBoxLayout()
        actions.setSpacing(key_gap)
        self.name_ok_btn = QPushButton("OK")
        self.name_cancel_btn = QPushButton("NAZAD")
        self.name_remove_btn = QPushButton("OBRIŠI")
        for b, obj, fs in (
            (self.name_ok_btn, "KeyNum", 48),
            (self.name_cancel_btn, "Footer", 44),
            (self.name_remove_btn, "Danger", 44),
        ):
            b.setObjectName(obj)
            b.setMinimumHeight(key_h)
            b.setMaximumHeight(key_h)
            b.setFocusPolicy(Qt.NoFocus)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setStyleSheet(
                f"QPushButton#{obj} {{ min-height: {key_h}px; max-height: {key_h}px; "
                f"padding: 0px 12px; font-size: {fs}px; font-weight: 800; }}"
            )
        self.name_ok_btn.clicked.connect(self._name_ok)
        self.name_cancel_btn.clicked.connect(self._hide_name_editor)
        self.name_remove_btn.clicked.connect(self._name_remove)
        actions.addWidget(self.name_cancel_btn, 1)
        actions.addWidget(self.name_remove_btn, 1)
        actions.addWidget(self.name_ok_btn, 1)
        cl.addLayout(actions)

        ol.addWidget(card, 0, Qt.AlignCenter)

    def _open_name_editor(self, slot: int) -> None:
        if self.session.screen != "select_players":
            return
        slots = self.session.player_slots()
        cur = slots[slot] if 0 <= slot < len(slots) else None
        self._name_slot = int(slot)
        self._name_was_selected = isinstance(cur, dict)
        raw = str((cur or {}).get("name") or "") if isinstance(cur, dict) else ""
        self._name_buf = "" if _is_placeholder_player_name(raw) else raw
        t = get_settings().t
        self.name_title.setText(t("enter_name"))
        self.name_ok_btn.setText("OK")
        self.name_remove_btn.setText(t("name_remove"))
        self.name_cancel_btn.setText(t("back"))
        self.name_remove_btn.setEnabled(True)
        self.name_remove_btn.setVisible(self._name_was_selected)
        self._name_refresh_display()
        self.name_overlay.show()
        self.name_overlay.raise_()
        self._place_chrome()
        self._update_settings_fab_visibility()

    def _hide_name_editor(self) -> None:
        self.name_overlay.hide()
        self._name_slot = None
        self._name_buf = ""
        self._update_settings_fab_visibility()

    def _name_refresh_display(self) -> None:
        shown = self._name_buf.upper() if self._name_buf else "—"
        self.name_display.setText(shown)

    def _name_type(self, ch: str) -> None:
        letter = "".join(c for c in ch if c.isalpha())
        if not letter or len(self._name_buf) >= 8:
            return
        self._name_buf += letter[:1].upper()
        self._name_refresh_display()

    def _name_backspace(self) -> None:
        if self._name_buf:
            self._name_buf = self._name_buf[:-1]
            self._name_refresh_display()

    def _name_ok(self) -> None:
        if self._name_slot is None:
            return
        self.session.set_player_slot(self._name_slot, self._name_buf)
        self._hide_name_editor()
        self._last_ui_sig = None
        self.refresh()

    def _name_remove(self) -> None:
        if self._name_slot is None:
            return
        self.session.clear_player_slot(self._name_slot)
        self._hide_name_editor()
        self._last_ui_sig = None
        self.refresh()

    def _build_killer_bull_overlay(self, root: QWidget) -> None:
        self.killer_bull_overlay = QWidget(root)
        self.killer_bull_overlay.setObjectName("KillerBullOverlay")
        self.killer_bull_overlay.hide()
        self.killer_bull_overlay.setStyleSheet(
            "QWidget#KillerBullOverlay { background-color: rgba(0,0,0,200); }"
        )
        ol = QVBoxLayout(self.killer_bull_overlay)
        ol.setContentsMargins(28, 24, 28, 24)
        ol.setAlignment(Qt.AlignCenter)

        card = QFrame()
        card.setObjectName("Card")
        card.setMinimumWidth(1560)
        card.setMaximumWidth(1820)
        card.setMinimumHeight(780)
        cl = QVBoxLayout(card)
        cl.setSpacing(16)
        cl.setContentsMargins(40, 32, 40, 32)

        self.killer_bull_title = QLabel("")
        self.killer_bull_title.setObjectName("Title")
        self.killer_bull_title.setAlignment(Qt.AlignCenter)
        self.killer_bull_title.setWordWrap(True)
        self.killer_bull_title.setStyleSheet(
            "font-size: 40px; font-weight: 800; color: #ffffff;"
        )
        cl.addWidget(self.killer_bull_title)

        self.killer_bull_targets_host = QWidget()
        self.killer_bull_targets_lay = QHBoxLayout(self.killer_bull_targets_host)
        self.killer_bull_targets_lay.setContentsMargins(0, 14, 0, 14)
        self.killer_bull_targets_lay.setSpacing(18)
        self.killer_bull_targets_lay.setAlignment(Qt.AlignCenter)
        cl.addWidget(self.killer_bull_targets_host, 1)

        # Manji, manje dominantan — ne tall Primary/Danger.
        self.killer_bull_false_btn = _btn(
            "NISAM POGODIO BULL", "killer_bull_false", role="Danger", tall=False
        )
        self.killer_bull_false_btn.setMinimumHeight(56)
        self.killer_bull_false_btn.setMaximumHeight(64)
        self.killer_bull_false_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.killer_bull_false_btn.setStyleSheet(
            "QPushButton#Danger { min-height: 56px; font-size: 36px; padding: 6px 16px; }"
        )
        cl.addWidget(self.killer_bull_false_btn)

        ol.addWidget(card, 0, Qt.AlignCenter)

    def _build_ingame_cal_overlay(self, root: QWidget) -> None:
        self._ingame_cal_pixmaps: Dict[int, QPixmap] = {}
        self._ingame_cal_stills_loaded = False
        self.ingame_cal_overlay = QWidget(root)
        self.ingame_cal_overlay.setObjectName("InGameCalOverlay")
        self.ingame_cal_overlay.hide()
        self.ingame_cal_overlay.setAttribute(Qt.WA_StyledBackground, True)
        ol = QVBoxLayout(self.ingame_cal_overlay)
        ol.setContentsMargins(20, 20, 20, 20)
        ol.setSpacing(0)

        self.ingame_cal_stack = QStackedWidget()
        self.ingame_cal_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        ol.addWidget(self.ingame_cal_stack, 1)

        prompt = QWidget()
        pl = QVBoxLayout(prompt)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setAlignment(Qt.AlignCenter)
        pcard = QFrame()
        pcard.setObjectName("Card")
        pcard.setMinimumWidth(900)
        pcard.setMaximumWidth(1200)
        pcl = QVBoxLayout(pcard)
        pcl.setSpacing(14)
        pcl.setContentsMargins(40, 32, 40, 32)
        self.ingame_cal_title = QLabel("")
        self.ingame_cal_title.setObjectName("ClearHero")
        self.ingame_cal_title.setAlignment(Qt.AlignCenter)
        self.ingame_cal_title.setWordWrap(True)
        self.ingame_cal_title.setStyleSheet(
            "QLabel#ClearHero { font-size: 64px; font-weight: 800; }"
        )
        self.ingame_cal_s1 = QLabel("")
        self.ingame_cal_s1.setObjectName("ClearSub")
        self.ingame_cal_s1.setAlignment(Qt.AlignCenter)
        self.ingame_cal_s1.setWordWrap(True)
        self.ingame_cal_s1.setStyleSheet(
            "QLabel#ClearSub { font-size: 40px; font-weight: 600; }"
        )
        self.ingame_cal_s2 = QLabel("")
        self.ingame_cal_s2.setObjectName("ClearSub")
        self.ingame_cal_s2.setAlignment(Qt.AlignCenter)
        self.ingame_cal_s2.setWordWrap(True)
        self.ingame_cal_s2.setStyleSheet(
            "QLabel#ClearSub { font-size: 40px; font-weight: 600; }"
        )
        self.ingame_cal_status = QLabel("")
        self.ingame_cal_status.setAlignment(Qt.AlignCenter)
        self.ingame_cal_status.setWordWrap(True)
        self.ingame_cal_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.ingame_cal_status.setStyleSheet(
            "color: #ffd24a; font-size: 32px; font-weight: 800; padding: 0 12px;"
        )
        pcl.addStretch(1)
        pcl.addWidget(self.ingame_cal_title)
        pcl.addWidget(self.ingame_cal_s1)
        pcl.addWidget(self.ingame_cal_s2)
        pcl.addWidget(self.ingame_cal_status)
        pcl.addStretch(1)
        self.ingame_cal_empty_btn = _btn(
            "PLOČA JE PRAZNA", "ingame_cal_confirm_empty", role="Accent", tall=True
        )
        self.ingame_cal_back_btn = _btn("NAZAD", "ingame_cal_back", role="Footer", tall=True)
        self.ingame_cal_back_btn.setMinimumHeight(72)
        self.ingame_cal_back_btn.setMaximumHeight(80)
        pcl.addWidget(self.ingame_cal_empty_btn)
        pcl.addWidget(self.ingame_cal_back_btn)
        pl.addWidget(pcard, 0, Qt.AlignCenter)

        result = QWidget()
        rl = QVBoxLayout(result)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(12)
        rcard = QFrame()
        rcard.setObjectName("Card")
        rcard.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        rcl = QVBoxLayout(rcard)
        rcl.setContentsMargins(20, 16, 20, 16)
        rcl.setSpacing(10)
        self.ingame_cal_result_status = QLabel("")
        self.ingame_cal_result_status.setAlignment(Qt.AlignCenter)
        self.ingame_cal_result_status.setWordWrap(True)
        self.ingame_cal_result_status.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred
        )
        self.ingame_cal_result_status.setStyleSheet(
            "color: #ffd24a; font-size: 28px; font-weight: 800;"
        )
        rcl.addWidget(self.ingame_cal_result_status, 0)
        cams_row = QHBoxLayout()
        cams_row.setSpacing(12)
        self.ingame_cal_cam_labs: List[QLabel] = []
        self.ingame_cal_cam_names: List[QLabel] = []
        for idx in CAMERA_INDICES:
            col = QWidget()
            col.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            cl = QVBoxLayout(col)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(6)
            still = QLabel("—")
            still.setObjectName("InGameCalStill")
            still.setAlignment(Qt.AlignCenter)
            still.setMinimumSize(220, 160)
            still.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            still.setScaledContents(False)
            name = QLabel(f"CAM {idx}")
            name.setObjectName("InGameCalCamName")
            name.setAlignment(Qt.AlignCenter)
            cl.addWidget(still, 1)
            cl.addWidget(name, 0)
            self.ingame_cal_cam_labs.append(still)
            self.ingame_cal_cam_names.append(name)
            cams_row.addWidget(col, 1)
        rcl.addLayout(cams_row, 1)
        btns = QHBoxLayout()
        btns.setSpacing(12)
        self.ingame_cal_done_btn = _btn("GOTOVO", "ingame_cal_done", role="Primary", tall=True)
        self.ingame_cal_retry_btn = _btn(
            "PONOVNO KALIBRIRAJ", "ingame_cal_retry", role="Footer", tall=True
        )
        self.ingame_cal_done_btn.setMinimumHeight(72)
        self.ingame_cal_retry_btn.setMinimumHeight(72)
        btns.addWidget(self.ingame_cal_done_btn, 1)
        btns.addWidget(self.ingame_cal_retry_btn, 1)
        rcl.addLayout(btns, 0)
        rl.addWidget(rcard, 1)

        self.ingame_cal_stack.addWidget(prompt)
        self.ingame_cal_stack.addWidget(result)

    def _bgr_to_pixmap(self, img) -> Optional[QPixmap]:
        if img is None:
            return None
        try:
            rgb = img[:, :, ::-1].copy()
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            return QPixmap.fromImage(qimg.copy())
        except Exception:
            return None

    def _load_ingame_cal_stills(self) -> None:
        stills = self.session.ctx.get("_ingame_cal_stills") or {}
        self._ingame_cal_pixmaps = {}
        for i, idx in enumerate(CAMERA_INDICES):
            pix = self._bgr_to_pixmap(stills.get(idx))
            if pix is not None:
                self._ingame_cal_pixmaps[int(idx)] = pix
            lab = self.ingame_cal_cam_labs[i]
            if pix is None:
                lab.setPixmap(QPixmap())
                lab.setText("NEMA SIGNALA")
        self._ingame_cal_stills_loaded = True
        self._layout_ingame_cal_stills()

    def _layout_ingame_cal_stills(self) -> None:
        if not getattr(self, "_ingame_cal_stills_loaded", False):
            return
        mode = (
            Qt.FastTransformation if _WEAK_HW else Qt.SmoothTransformation
        )
        for i, idx in enumerate(CAMERA_INDICES):
            lab = self.ingame_cal_cam_labs[i]
            pix = self._ingame_cal_pixmaps.get(int(idx))
            if pix is None or pix.isNull():
                continue
            lw, lh = max(1, lab.width()), max(1, lab.height())
            lab.setPixmap(pix.scaled(lw, lh, Qt.KeepAspectRatio, mode))

    def _sync_ingame_cal_overlay(self, snap: Optional[dict] = None) -> None:
        if not hasattr(self, "ingame_cal_overlay"):
            return
        if snap is None:
            snap = {
                "ingame_cal_open": self.session.is_in_game_calibrating(),
                "ingame_cal_phase": str(self.session.ctx.get("_ingame_cal_phase") or ""),
                "ingame_cal_status": str(self.session.ctx.get("_ingame_cal_status") or ""),
                "screen": self.session.screen,
            }
        open_ = bool(snap.get("ingame_cal_open")) and snap.get("screen") == "playing"
        if not open_:
            self.ingame_cal_overlay.hide()
            self._ingame_cal_stills_loaded = False
            self._ingame_cal_pixmaps = {}
            self._update_settings_fab_visibility()
            return
        phase = str(snap.get("ingame_cal_phase") or "prompt")
        status = str(snap.get("ingame_cal_status") or "")
        busy = phase == "busy"
        self.ingame_cal_status.setText(status)
        self.ingame_cal_result_status.setText(status)
        self.ingame_cal_empty_btn.setEnabled(not busy)
        self.ingame_cal_back_btn.setEnabled(not busy)
        self.ingame_cal_done_btn.setEnabled(not busy)
        self.ingame_cal_retry_btn.setEnabled(not busy)
        if phase == "result":
            self.ingame_cal_stack.setCurrentIndex(1)
            if not self._ingame_cal_stills_loaded:
                self._load_ingame_cal_stills()
            QTimer.singleShot(0, self._layout_ingame_cal_stills)
        elif phase == "busy" and self.ingame_cal_stack.currentIndex() == 1:
            self.ingame_cal_result_status.setText(status)
        else:
            self.ingame_cal_stack.setCurrentIndex(0)
            self._ingame_cal_stills_loaded = False
        self.ingame_cal_overlay.show()
        self.ingame_cal_overlay.raise_()
        root = self.centralWidget()
        if root is not None:
            self.ingame_cal_overlay.setGeometry(0, 0, root.width(), root.height())
        self._update_settings_fab_visibility()

    def _run_deferred_ingame_cal(self) -> None:
        if not self.session.is_in_game_calibrating():
            return
        if str(self.session.ctx.get("_ingame_cal_phase") or "") != "busy":
            return
        self.session.run_in_game_calibrate()
        self._last_ui_sig = None
        self._ingame_cal_stills_loaded = False
        self._sync_ingame_cal_overlay()
        self.refresh()

    def _show_pin(self, mode: str) -> None:
        t = get_settings().t
        self._pin_mode = mode
        self._pin_buf = ""
        self._pin_pending = ""
        self.pin_status.setText("")
        if mode == "unlock":
            self.pin_title.setText(t("pin_enter"))
        elif mode == "set_new":
            self.pin_title.setText(t("pin_new"))
        else:
            self.pin_title.setText(t("pin_confirm"))
        self.pin_ok_btn.setText(t("pin_ok"))
        self.pin_cancel_btn.setText(t("pin_cancel"))
        self._pin_refresh_display()
        self.pin_overlay.show()
        self.pin_overlay.raise_()
        self._place_chrome()
        self._update_settings_fab_visibility()

    def _hide_pin(self) -> None:
        self.pin_overlay.hide()
        self._pin_mode = ""
        self._pin_buf = ""
        self._pin_pending = ""
        self._update_settings_fab_visibility()

    def _show_quit_confirm(self, kind: str = "exit") -> None:
        self._power_pending = kind if kind in ("shutdown", "reboot", "exit") else "exit"
        t = get_settings().t
        key = {
            "shutdown": "power_off_confirm",
            "reboot": "power_reboot_confirm",
            "exit": "power_exit_confirm",
        }.get(self._power_pending, "power_exit_confirm")
        self.quit_title.setText(t(key))
        self.quit_yes_btn.setText(t("yes"))
        self.quit_no_btn.setText(t("no"))
        self.quit_overlay.show()
        self.quit_overlay.raise_()
        self._place_chrome()
        self._update_settings_fab_visibility()

    def _hide_quit_confirm(self) -> None:
        self._power_pending = ""
        if hasattr(self, "quit_overlay"):
            self.quit_overlay.hide()
        self._update_settings_fab_visibility()

    def _confirm_power_action(self) -> None:
        kind = str(getattr(self, "_power_pending", "") or "exit")
        self._hide_quit_confirm()
        if kind == "shutdown":
            self._system_power("shutdown")
            return
        if kind == "reboot":
            self._system_power("reboot")
            return
        self._quit_application()

    def _idle_playing(self) -> bool:
        return self._idle_active()

    def _idle_active(self) -> bool:
        screen = self.session.screen
        if screen not in _IDLE_SCREENS:
            return False
        if screen == "playing":
            g = self.session.ctx.get("game")
            if not g:
                return False
            if g.get("winner_player_idx") is not None:
                return False
            if self.session.is_in_game_calibrating():
                return False
        return True

    def _idle_note_activity(self) -> None:
        self._idle_last = time.monotonic()
        if self._idle_deadline is not None:
            self._hide_idle_overlay()

    def _hide_idle_overlay(self) -> None:
        self._idle_deadline = None
        if hasattr(self, "idle_overlay"):
            self.idle_overlay.hide()
        self._update_settings_fab_visibility()

    def _idle_continue_game(self) -> None:
        self._hide_idle_overlay()
        self._idle_last = time.monotonic()

    def _idle_quit_game(self) -> None:
        self._hide_idle_overlay()
        self.session.clear_to_standby(mqtt_off=True)
        self._last_ui_sig = None
        self.refresh()

    def _show_idle_overlay(self) -> None:
        t = get_settings().t
        self.idle_title.setText(t("idle_title"))
        self.idle_s1.setText(t("idle_s1"))
        self.idle_quit_btn.setText(t("idle_quit"))
        self.idle_continue_btn.setText(t("idle_continue"))
        self.idle_overlay.show()
        self.idle_overlay.raise_()
        self._place_chrome()
        self._update_settings_fab_visibility()

    def _idle_tick(self) -> None:
        if not self._idle_playing():
            if self._idle_deadline is not None or (
                hasattr(self, "idle_overlay") and self.idle_overlay.isVisible()
            ):
                self._hide_idle_overlay()
            self._idle_last = time.monotonic()
            self._idle_hits_sig = None
            return
        snap_hits = None
        g = self.session.ctx.get("game")
        if g:
            snap_hits = tuple(
                None if h is None else str(h)
                for h in (g.get("turn_hits") or [None, None, None])
            )
        status = str(self.session.ctx.get("_dart_status") or "")
        if snap_hits != self._idle_hits_sig:
            self._idle_hits_sig = snap_hits
            if snap_hits is not None and any(h is not None for h in snap_hits):
                self._idle_note_activity()
        prev_status = getattr(self, "_idle_last_status", "")
        self._idle_last_status = status
        if status != prev_status and "HIT" in status.upper():
            self._idle_note_activity()
        now = time.monotonic()
        if self._idle_deadline is not None:
            if now >= self._idle_deadline:
                self._idle_quit_game()
            return
        if now - self._idle_last >= _IDLE_WARN_SEC:
            self._idle_deadline = now + _IDLE_EXIT_SEC
            self._show_idle_overlay()

    def _is_user_pointer_event(self, event) -> bool:
        try:
            et = event.type()
        except Exception:
            return False
        names = (
            "MouseButtonPress",
            "MouseButtonDblClick",
            "TouchBegin",
            "TabletPress",
        )
        for name in names:
            want = getattr(QEvent.Type, name, None)
            if want is None:
                want = getattr(QEvent, name, None)
            if want is not None and et == want:
                return True
        return False

    def _quit_application(self) -> None:
        self._hide_quit_confirm()
        try:
            self.session.shutdown()
        except Exception:
            pass
        app = QApplication.instance()
        if app is not None:
            app.quit()
        else:
            self.close()

    def _system_power(self, kind: str) -> None:
        """Raspberry: ugasi ili reboot. Na Windowsu samo izlaz iz programa."""
        try:
            self.session.shutdown()
        except Exception:
            pass
        if sys.platform.startswith("linux"):
            if kind == "reboot":
                cmds = (
                    [
                        "busctl",
                        "call",
                        "org.freedesktop.login1",
                        "/org/freedesktop/login1",
                        "org.freedesktop.login1.Manager",
                        "Reboot",
                        "b",
                        "true",
                    ],
                    ["sudo", "-n", "systemctl", "reboot"],
                    ["sudo", "-n", "shutdown", "-r", "now"],
                )
            else:
                cmds = (
                    [
                        "busctl",
                        "call",
                        "org.freedesktop.login1",
                        "/org/freedesktop/login1",
                        "org.freedesktop.login1.Manager",
                        "PowerOff",
                        "b",
                        "true",
                    ],
                    ["sudo", "-n", "systemctl", "poweroff"],
                    ["sudo", "-n", "shutdown", "-h", "now"],
                )
            ok = False
            for cmd in cmds:
                try:
                    r = subprocess.run(cmd, timeout=3.0, check=False)
                    if r.returncode == 0:
                        print(f"[power] {kind}: {' '.join(cmd)}", flush=True)
                        ok = True
                        break
                except Exception as exc:
                    print(f"[power] {kind} fail {cmd[0]}: {exc}", flush=True)
            if not ok:
                print(f"[power] {kind} nije uspio — izlazim iz programa", flush=True)
        else:
            print(f"[power] {kind} samo na Raspberryju — izlazim iz programa", flush=True)
        app = QApplication.instance()
        if app is not None:
            app.quit()
        else:
            self.close()

    def _pin_refresh_display(self) -> None:
        n = len(self._pin_buf)
        self.pin_display.setText("•" * n if n else "—")

    def _pin_digit(self, d: str) -> None:
        if len(self._pin_buf) >= 8:
            return
        self._pin_buf += d
        self.pin_status.setText("")
        self._pin_refresh_display()
        # Auto-submit unlock kad dužina == lozinka
        if self._pin_mode == "unlock":
            expect = get_settings().settings_password
            if len(self._pin_buf) >= len(expect):
                self._pin_submit()

    def _pin_backspace(self) -> None:
        if self._pin_buf:
            self._pin_buf = self._pin_buf[:-1]
            self._pin_refresh_display()

    def _pin_submit(self) -> None:
        t = get_settings().t
        buf = self._pin_buf
        if self._pin_mode == "unlock":
            if buf == get_settings().settings_password:
                self._hide_pin()
                self._open_settings()
            else:
                self.pin_status.setText(t("pin_wrong"))
                self._pin_buf = ""
                self._pin_refresh_display()
            return
        if self._pin_mode == "set_new":
            if len(buf) < 4:
                self.pin_status.setText(t("pin_new") + " (4+)")
                return
            self._pin_pending = buf
            self._pin_buf = ""
            self._pin_mode = "set_confirm"
            self.pin_title.setText(t("pin_confirm"))
            self.pin_status.setText("")
            self._pin_refresh_display()
            return
        if self._pin_mode == "set_confirm":
            if buf != self._pin_pending:
                self.pin_status.setText(t("pin_mismatch"))
                self._pin_buf = ""
                self._pin_pending = ""
                self._pin_mode = "set_new"
                self.pin_title.setText(t("pin_new"))
                self._pin_refresh_display()
                return
            update_settings(settings_password=buf)
            self.pin_status.setStyleSheet(
                "font-size: 28px; color: #6dff8a; min-height: 36px;"
            )
            self.pin_status.setText(t("pin_saved"))
            QTimer.singleShot(700, self._hide_pin)
            QTimer.singleShot(
                750,
                lambda: self.pin_status.setStyleSheet(
                    "font-size: 28px; color: #ff6a6a; min-height: 36px;"
                ),
            )

    def _run_deferred_clear_board_calibrate(self) -> None:
        if self.session.screen != "clear_board":
            return
        if not self.session.ctx.get("_clear_board_pending_run"):
            return
        self.session.ctx["_clear_board_pending_run"] = False
        self.session.ctx["_clear_board_pending_at"] = None
        self.session.confirm_clear_board_and_start()
        self._last_ui_sig = None
        self.refresh()

    def _ensure_mqtt_client(self) -> None:
        """Pokreni MQTT subscriber ako je način mqtt_qr (npr. nakon promjene u postavkama)."""
        if get_settings().start_mode != "mqtt_qr":
            return
        if getattr(self, "_mqtt_client", None) is not None:
            return
        try:
            import ssl
            import paho.mqtt.client as mqtt
            from mqtt import BROKER_HOST, BROKER_PORT, PASSWORD, TOPIC, USERNAME

            def _on_message(_mqttc, _userdata, msg):
                try:
                    payload = msg.payload.decode().strip().lower()
                except Exception:
                    payload = ""
                if payload in ("on", "off"):
                    # Signal iz paho threada > Qt slot (isti pattern kao u run_app).
                    bridge = getattr(self, "_mqtt_bridge", None)
                    if bridge is not None:
                        bridge.command.emit(payload)

            def _on_connect(mqttc, _userdata, _flags, rc, properties=None):
                try:
                    if rc == 0:
                        mqttc.subscribe(TOPIC)
                except Exception:
                    pass

            client = mqtt.Client()
            client.username_pw_set(USERNAME, PASSWORD)
            client.tls_set(
                ca_certs=None,
                certfile=None,
                keyfile=None,
                cert_reqs=ssl.CERT_REQUIRED,
                tls_version=ssl.PROTOCOL_TLS_CLIENT,
            )
            client.on_message = _on_message
            client.on_connect = _on_connect
            client.connect(BROKER_HOST, BROKER_PORT, 60)
            client.loop_start()
            self._mqtt_client = client
            print("[MQTT] GUI subscriber started (ensure)", flush=True)
        except Exception as e:
            print(f"[MQTT] GUI subscriber failed (ensure): {e}", flush=True)

    def _mqtt_on(self) -> None:
        """Tipka 3: objavi MQTT 'on' (LED ring + wake preko subscribera)."""
        if get_settings().start_mode == "always_on":
            print("[MQTT] tipka 3 skipped (always_on)", flush=True)
            return
        try:
            from mqtt import TOPIC

            client = getattr(self, "_mqtt_client", None)
            if client is not None:
                # Ne čekaj PUBACK na UI threadu — loop već radi u pozadini.
                client.publish(TOPIC, "on", qos=1)
                print(f"[MQTT] tipka 3: published 'on' -> {TOPIC}", flush=True)
                return
        except Exception as e:
            print(f"[MQTT] tipka 3 persistent publish failed: {e}", flush=True)
        try:
            import main_manual as mm

            mm.send_mqtt_on()
        except Exception as e:
            print(f"[MQTT] tipka 3 fallback send_mqtt_on failed: {e}", flush=True)

    def _toggle_cal(self) -> None:
        if self.session.screen == "calibration":
            self.session.exit_calibration()
        else:
            self.session.enter_calibration()
        self._last_ui_sig = None
        self.refresh()

    def _toggle_cal_topdown(self) -> None:
        if self.session.screen != "calibration":
            return
        self._act("calibration_capture")

    def _toggle_cal_overlay(self) -> None:
        if self.session.screen != "calibration":
            return
        self._act("calibration_overlay")

    def _cal_detect_hotkey(self) -> None:
        if self.session.screen != "calibration":
            return
        self._act("calibration_detect")

    def _on_escape(self) -> None:
        if hasattr(self, "quit_overlay") and self.quit_overlay.isVisible():
            self._hide_quit_confirm()
            return
        if hasattr(self, "pin_overlay") and self.pin_overlay.isVisible():
            self._hide_pin()
            return
        if hasattr(self, "name_overlay") and self.name_overlay.isVisible():
            self._hide_name_editor()
            return
        if self.session.is_in_game_calibrating():
            phase = str(self.session.ctx.get("_ingame_cal_phase") or "")
            if phase == "busy":
                return
            if phase != "result":
                self.session.close_in_game_calibrate()
                self._last_ui_sig = None
                self._sync_ingame_cal_overlay()
                self.refresh()
            return
        if self.session.screen == "calibration":
            self.session.exit_calibration()
            self._last_ui_sig = None
            self.refresh()
        elif self.session.screen == "settings":
            self.session.exit_settings()
            self._last_ui_sig = None
            self.refresh()
        elif not self.session.kiosk:
            self.close()

    def _force_detect(self) -> None:
        if self.session.is_in_game_calibrating():
            return
        if self.session.screen == "playing":
            self.session.try_auto_detect_dart(force_log=True)
            self._last_ui_sig = None
            self.refresh()
            self._flush_dart_commit()

    def _fake_cams_next_hit(self) -> None:
        """J: ciklički ubaci sljedeći hit triplet (samo --fakecams)."""
        if self.session.cycle_fake_dart_hit():
            self._last_ui_sig = None

    def _fake_cams_next_player(self) -> None:
        """L: next player / end turn — isto kao UI '>' (samo --fakecams)."""
        if self.session.fake_cams_advance_next_player():
            self._last_ui_sig = None
            self.refresh()

    def _save_kiosk_screenshot(self) -> None:
        """Tipka 4: screenshot cijelog kiosk GUI u debug_edges/."""
        import time

        try:
            from board_calibration import _debug_dir_path, _ensure_debug_dir_exists

            _ensure_debug_dir_exists()
            pix = self.grab()
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(_debug_dir_path(), f"kiosk_ui_{stamp}.png")
            if pix.save(path):
                print(f"[debug] kiosk screenshot: {path}", flush=True)
            else:
                print(f"[debug] kiosk screenshot FAIL: {path}", flush=True)
        except Exception as e:
            print(f"[debug] kiosk screenshot error: {e}", flush=True)

    def _on_tick(self) -> None:
        if bool(getattr(self.session, "kiosk", False)):
            self._lock_kiosk_display()
        try:
            self.session.tick()
        except Exception as exc:
            print(f"[ui] tick error (ignored): {exc}", flush=True)
        if self.session.screen == "calibration":
            import time as _time

            now = _time.monotonic() * 1000.0
            # Preview 3 kamera: max ~10 FPS (60 FPS + overlay smrzava USB)
            if now - self._last_cal_view_ms >= 100.0:
                self._last_cal_view_ms = now
                self._update_calibration_view()
            self._flush_dart_commit()
            self._idle_tick()
            return
        self.refresh()
        dart_det = self.session.ctx.get("dart_detector")
        if dart_det is not None and getattr(dart_det, "_pending_ui_commit", False):
            self.repaint()
        self._flush_dart_commit()
        self._idle_tick()

    def _flush_dart_commit(self) -> None:
        dart_det = self.session.ctx.get("dart_detector")
        if dart_det is None or not getattr(dart_det, "_pending_ui_commit", False):
            return
        board_cal = self.session.ctx.get("board_calibrator")
        dart_det.flush_commit_after_ui(board_cal)

    def _ui_signature(self, snap: dict) -> str:
        """Stable, cheap signature — avoid repr() of nested dicts (expensive every tick)."""
        players = snap.get("players") or []
        players_sig = tuple(
            (
                p.get("index"),
                p.get("name"),
                p.get("score_text"),
                p.get("active"),
                p.get("rank"),
                p.get("points"),
                p.get("remaining"),
                p.get("lives"),
                p.get("is_killer"),
                p.get("number"),
                p.get("completed"),
                p.get("next_target_idx"),
                p.get("score"),
                p.get("score_halved"),
                tuple(sorted((p.get("marks") or {}).items())),
            )
            for p in players
        )
        settings = snap.get("settings") or {}
        bull = snap.get("killer_bull_pending")
        bull_sig = None
        if isinstance(bull, dict):
            bull_sig = (bull.get("attacker"), bull.get("victim"), bull.get("hit"))
        flash = snap.get("killer_life_flash")
        if isinstance(flash, (list, tuple)):
            flash_sig = tuple(flash)
        else:
            flash_sig = flash
        targets = snap.get("killer_bull_targets") or []
        targets_sig = tuple(
            (t.get("index"), t.get("number"), t.get("lives"), t.get("is_killer"))
            for t in targets
            if isinstance(t, dict)
        )
        leaderboard = snap.get("leaderboard") or []
        lb_sig = tuple(
            (int(e[0]), int(e[1]))
            if isinstance(e, (list, tuple)) and len(e) >= 2
            else e
            for e in leaderboard
        )
        undo_tags = snap.get("undo_pending_tags")
        return repr(
            (
                snap.get("screen"),
                snap.get("selected_game_mode"),
                snap.get("player_slots"),
                tuple(sorted((snap.get("rules_pending") or {}).items())),
                snap.get("current_player_idx"),
                tuple(snap.get("turn_hit_tags") or ()),
                tuple(snap.get("slot_enabled") or ()),
                snap.get("editing_hit_slot"),
                snap.get("input_multiplier"),
                snap.get("manual_mode"),
                snap.get("can_submit"),
                snap.get("is_early_finish"),
                snap.get("is_bust"),
                snap.get("block_adv"),
                snap.get("show_manual_advance"),
                snap.get("advance_label"),
                snap.get("board_clear_message"),
                snap.get("has_undo"),
                snap.get("turn_anim_reverse"),
                snap.get("winner"),
                lb_sig,
                snap.get("live_score_line"),
                snap.get("clear_board_status"),
                snap.get("ingame_cal_open"),
                snap.get("ingame_cal_phase"),
                snap.get("ingame_cal_status"),
                players_sig,
                snap.get("round_number"),
                tuple(undo_tags) if undo_tags is not None else None,
                snap.get("game_title"),
                snap.get("killer_phase"),
                bull_sig,
                targets_sig,
                flash_sig,
                (
                    settings.get("language"),
                    settings.get("bg_color"),
                    settings.get("button_color"),
                    settings.get("start_mode"),
                    settings.get("manual_mode"),
                    settings.get("auto_calibrate"),
                    settings.get("device_id"),
                    settings.get("league_qr_enabled"),
                ),
                snap.get("league_qr_payload"),
                bool(snap.get("cal_click_bull_mode")),
                bool(snap.get("cal_click_ellipse_mode")),
                bool(snap.get("cal_show_topdown")),
                bool(snap.get("cal_show_overlay")),
                tuple(
                    sorted(
                        (int(k), round(float(v[0]), 1), round(float(v[1]), 1))
                        for k, v in (snap.get("bull_hints") or {}).items()
                    )
                ),
                tuple(
                    sorted(
                        (
                            int(k),
                            tuple(
                                (round(float(p[0]), 1), round(float(p[1]), 1))
                                for p in (v or [])[:4]
                                if p is not None and len(p) >= 2
                            ),
                        )
                        for k, v in (snap.get("ellipse_hints") or {}).items()
                    )
                ),
            )
        )

    def _animate_hit(self, idx: int) -> None:
        """Popup bounce na hit slotu (3 hita lijevo)."""
        if idx < 0 or idx >= len(self.hit_btns):
            return
        btn = self.hit_btns[idx]
        tag = (btn.text() or "").strip()
        if not tag or tag == "-":
            return

        prev = self._hit_anims[idx]
        if prev is not None:
            prev.stop()
            self._hit_anims[idx] = None
            if self._hit_geo_restore[idx] is not None:
                btn.setGeometry(self._hit_geo_restore[idx])
                self._hit_geo_restore[idx] = None

        # Layout geometry trenutnog slota (prije bouncea).
        base = QRect(btn.geometry())
        self._hit_geo_restore[idx] = QRect(base)

        # Skok van (~12%) pa povratak — "popup" osjećaj na samom gumbu.
        dx = max(10, base.width() // 16)
        dy = max(8, base.height() // 14)
        popped = QRect(
            base.x() - dx,
            base.y() - dy,
            base.width() + 2 * dx,
            base.height() + 2 * dy,
        )

        grow = QPropertyAnimation(btn, b"geometry")
        grow.setDuration(140)
        grow.setStartValue(QRect(base))
        grow.setEndValue(popped)
        grow.setEasingCurve(QEasingCurve.OutBack)

        shrink = QPropertyAnimation(btn, b"geometry")
        shrink.setDuration(180)
        shrink.setStartValue(popped)
        shrink.setEndValue(QRect(base))
        shrink.setEasingCurve(QEasingCurve.OutCubic)

        seq = QSequentialAnimationGroup(self)
        seq.addAnimation(grow)
        seq.addAnimation(shrink)

        def _restore() -> None:
            saved = self._hit_geo_restore[idx]
            if saved is not None:
                btn.setGeometry(saved)
            self._hit_geo_restore[idx] = None
            self._hit_anims[idx] = None

        seq.finished.connect(_restore)
        self._hit_anims[idx] = seq
        seq.start()

    def _start_winner_pulse(self, lab: QLabel) -> None:
        # Bez fade pulse
        self._winner_title = lab

    def _stop_winner_pulse(self) -> None:
        self._winner_title = None

    # --- page builders ---

    def _build_standby(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setAlignment(Qt.AlignCenter)

        title_h = 108
        title_row = QHBoxLayout()
        title_row.setSpacing(18)
        title_row.setAlignment(Qt.AlignCenter)
        standby_logo = _make_logo_label(title_h)
        title = QLabel("SMARTDARTS")
        title.setObjectName("Title")
        title.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        title.setStyleSheet("font-size: 108px; font-weight: 800;")
        title_row.addWidget(standby_logo, 0, Qt.AlignVCenter)
        title_row.addWidget(title, 0, Qt.AlignVCenter)

        self.standby_sub = QLabel("Skeniraj QR za početak igre!")
        self.standby_sub.setObjectName("Subtitle")
        self.standby_sub.setAlignment(Qt.AlignCenter)
        self.standby_sub.setStyleSheet("font-size: 54px; color: #c0c0c8;")
        self.qr_frame = QFrame()
        self.qr_frame.setObjectName("QrFrame")
        self.qr_frame.setAttribute(Qt.WA_StyledBackground, True)
        qr_lay = QVBoxLayout(self.qr_frame)
        qr_lay.setContentsMargins(10, 10, 10, 10)
        self.qr_label = QLabel()
        self.qr_label.setAlignment(Qt.AlignCenter)
        self.qr_label.setMinimumSize(520, 520)
        self.qr_label.setStyleSheet("background: transparent; border: none;")
        qr_lay.addWidget(self.qr_label)
        self.standby_start_btn = _btn("POKRENI", "standby_start", role="Primary", tall=True)
        self.standby_start_btn.setMinimumHeight(120)
        self.standby_start_btn.hide()
        lay.addStretch(1)
        lay.addLayout(title_row)
        lay.addSpacing(12)
        lay.addWidget(self.standby_sub)
        lay.addSpacing(28)
        lay.addWidget(self.qr_frame, 0, Qt.AlignCenter)
        lay.addWidget(self.standby_start_btn, 0, Qt.AlignCenter)
        lay.addStretch(1)
        return w

    def _build_select_game(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(16, 16, 16, 16)
        self.select_game_title = QLabel("ODABERI IGRU")
        self.select_game_title.setObjectName("Title")
        self.select_game_title.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.select_game_title)
        grid = QGridLayout()
        modes = [
            ("301", "game:301"),
            ("501", "game:501"),
            ("AROUND", "game:around"),
            ("CRICKET", "game:cricket"),
            ("KILLER", "game:killer"),
            ("HALVE IT", "game:halve"),
        ]
        for i, (lab, act) in enumerate(modes):
            b = _btn(lab, act, role="Primary")
            grid.addWidget(b, i // 2, i % 2)
        lay.addLayout(grid, 1)
        return w

    def _build_rules_x01(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(18)
        self.rules_x01_title = QLabel("PRAVILA")
        self.rules_x01_title.setObjectName("Title")
        self.rules_x01_title.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.rules_x01_title)
        lay.addStretch(1)
        lbl_mode = QLabel("GAME MODE")
        lbl_mode.setObjectName("SectionLabel")
        lbl_mode.setAlignment(Qt.AlignLeft)
        self.lbl_game_mode = lbl_mode
        lay.addWidget(lbl_mode)
        row = QHBoxLayout()
        row.setSpacing(20)
        row.setAlignment(Qt.AlignCenter)
        self.btn_double_in = _btn("DOUBLE IN: NE", "rule:double_in", role="RuleOff")
        self.btn_double_out = _btn("DOUBLE OUT: NE", "rule:double_out", role="RuleOff")
        self.btn_double_in.setMinimumWidth(340)
        self.btn_double_out.setMinimumWidth(340)
        row.addWidget(self.btn_double_in)
        row.addWidget(self.btn_double_out)
        lay.addLayout(row)
        lay.addStretch(1)
        self.rules_x01_tutorial = _rules_footer_btn("UPUTE", "tutorial_open")
        self.rules_x01_continue = _rules_continue_btn("NASTAVI", "rules_continue")
        self.rules_x01_back = _rules_back_btn("NAZAD", "nav_back")
        lay.addLayout(
            _rules_footer_block(
                back_btn=self.rules_x01_back,
                middle=self.rules_x01_tutorial,
                continue_btn=self.rules_x01_continue,
            )
        )
        return w

    def _build_rules_cricket(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        # Compact like Killer/Around — round limit + mode option rows.
        lay.setContentsMargins(24, 12, 24, 12)
        lay.setSpacing(8)
        title = QLabel("CRICKET PRAVILA")
        title.setObjectName("Title")
        title.setAlignment(Qt.AlignCenter)
        self.rules_cricket_title = title
        lay.addWidget(title)
        lay.addStretch(1)

        def _rule_btn(text: str, action: str) -> AnimButton:
            b = _btn(text, action, role="RuleOff")
            b.setMinimumHeight(72)
            b.setMaximumHeight(78)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setStyleSheet(
                "QPushButton#RuleOn, QPushButton#RuleOff {"
                " min-height: 72px; font-size: 48px; padding: 4px 12px; }"
            )
            return b

        lbl_rounds = QLabel("BROJ RUNDI")
        lbl_rounds.setObjectName("SectionLabel")
        lbl_rounds.setAlignment(Qt.AlignLeft)
        lay.addWidget(lbl_rounds)
        row1 = QHBoxLayout()
        row1.setSpacing(16)
        self.btn_max_off = _rule_btn("BEZ LIMITA", "rule:max_rounds_off")
        self.btn_max_20 = _rule_btn("20 RUNDI", "rule:max_rounds_20")
        row1.addWidget(self.btn_max_off)
        row1.addWidget(self.btn_max_20)
        lay.addLayout(row1)

        lay.addSpacing(10)
        lbl_mode = QLabel("GAME MODE")
        lbl_mode.setObjectName("SectionLabel")
        lbl_mode.setAlignment(Qt.AlignLeft)
        lay.addWidget(lbl_mode)
        row2 = QHBoxLayout()
        row2.setSpacing(16)
        self.btn_c_std = _rule_btn("STANDARD", "rule:cricket_standard")
        self.btn_c_nos = _rule_btn("NO SCORE", "rule:cricket_no_score")
        self.btn_c_cut = _rule_btn("CUT THROAT", "rule:cricket_cut_throat")
        row2.addWidget(self.btn_c_std)
        row2.addWidget(self.btn_c_nos)
        row2.addWidget(self.btn_c_cut)
        lay.addLayout(row2)

        lay.addStretch(1)
        self.rules_cricket_tutorial = _rules_footer_btn("UPUTE", "tutorial_open")
        self.rules_cricket_continue = _rules_continue_btn("NASTAVI", "rules_continue")
        self.rules_cricket_back = _rules_back_btn("NAZAD", "nav_back")
        lay.addLayout(
            _rules_footer_block(
                back_btn=self.rules_cricket_back,
                middle=self.rules_cricket_tutorial,
                continue_btn=self.rules_cricket_continue,
            )
        )
        return w

    def _build_rules_killer(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        # Usklađeno s Cricket/X01, ali zbijenije — 3 sekcije + horizontalni footer.
        lay.setContentsMargins(24, 12, 24, 12)
        lay.setSpacing(8)
        title = QLabel("KILLER PRAVILA")
        title.setObjectName("Title")
        title.setAlignment(Qt.AlignCenter)
        self.rules_killer_title = title
        lay.addWidget(title)

        def _rule_btn(text: str, action: str) -> AnimButton:
            b = _btn(text, action, role="RuleOff")
            b.setMinimumHeight(72)
            b.setMaximumHeight(78)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setStyleSheet(
                "QPushButton#RuleOn, QPushButton#RuleOff {"
                " min-height: 72px; font-size: 48px; padding: 4px 12px; }"
            )
            return b

        self.lbl_killer_lives = QLabel("BROJ ŽIVOTA")
        self.lbl_killer_lives.setObjectName("SectionLabel")
        self.lbl_killer_lives.setAlignment(Qt.AlignLeft)
        lay.addWidget(self.lbl_killer_lives)
        row_lives = QHBoxLayout()
        row_lives.setSpacing(12)
        self.btn_killer_lives: Dict[int, QPushButton] = {}
        for lives in (3, 5, 7, 10):
            b = _rule_btn(str(lives), f"rule:killer_lives:{lives}")
            self.btn_killer_lives[lives] = b
            row_lives.addWidget(b)
        lay.addLayout(row_lives)

        self.lbl_killer_assign = QLabel("DODJELA BROJEVA")
        self.lbl_killer_assign.setObjectName("SectionLabel")
        self.lbl_killer_assign.setAlignment(Qt.AlignLeft)
        lay.addWidget(self.lbl_killer_assign)
        row_assign = QHBoxLayout()
        row_assign.setSpacing(12)
        self.btn_killer_assign_random = _rule_btn(
            "RANDOM", "rule:killer_assign:random"
        )
        self.btn_killer_assign_throw = _rule_btn(
            "GAĐANJEM", "rule:killer_assign:throw"
        )
        row_assign.addWidget(self.btn_killer_assign_random)
        row_assign.addWidget(self.btn_killer_assign_throw)
        lay.addLayout(row_assign)

        self.lbl_killer_activation = QLabel("AKTIVACIJA")
        self.lbl_killer_activation.setObjectName("SectionLabel")
        self.lbl_killer_activation.setAlignment(Qt.AlignLeft)
        lay.addWidget(self.lbl_killer_activation)
        row_act = QHBoxLayout()
        row_act.setSpacing(12)
        self.btn_killer_act_single = _rule_btn(
            "SINGLE", "rule:killer_activation:single"
        )
        self.btn_killer_act_double = _rule_btn(
            "DOUBLE", "rule:killer_activation:double"
        )
        self.btn_killer_act_triple = _rule_btn(
            "TRIPLE", "rule:killer_activation:triple"
        )
        row_act.addWidget(self.btn_killer_act_single)
        row_act.addWidget(self.btn_killer_act_double)
        row_act.addWidget(self.btn_killer_act_triple)
        lay.addLayout(row_act)

        lay.addStretch(1)
        self.rules_killer_tutorial = _rules_footer_btn("UPUTE", "tutorial_open")
        self.rules_killer_continue = _rules_continue_btn("NASTAVI", "rules_continue")
        self.rules_killer_back = _rules_back_btn("NAZAD", "nav_back")
        lay.addLayout(
            _rules_footer_block(
                back_btn=self.rules_killer_back,
                middle=self.rules_killer_tutorial,
                continue_btn=self.rules_killer_continue,
            )
        )
        return w

    def _build_rules_around(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        # Compact like Killer — 3 rule rows + UPUTE|NASTAVI footer.
        lay.setContentsMargins(24, 12, 24, 12)
        lay.setSpacing(8)
        title = QLabel("AROUND PRAVILA")
        title.setObjectName("Title")
        title.setAlignment(Qt.AlignCenter)
        self.rules_around_title = title
        lay.addWidget(title)

        def _rule_btn(text: str, action: str) -> AnimButton:
            b = _btn(text, action, role="RuleOff")
            b.setMinimumHeight(72)
            b.setMaximumHeight(78)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setStyleSheet(
                "QPushButton#RuleOn, QPushButton#RuleOff {"
                " min-height: 72px; font-size: 40px; padding: 4px 8px; }"
            )
            return b

        self.lbl_around_mult = QLabel("SEGMENTI")
        self.lbl_around_mult.setObjectName("SectionLabel")
        self.lbl_around_mult.setAlignment(Qt.AlignLeft)
        lay.addWidget(self.lbl_around_mult)
        row_mult = QHBoxLayout()
        row_mult.setSpacing(10)
        self.btn_around_mult_single = _rule_btn("SINGLE", "rule:around_mult:single")
        self.btn_around_mult_double = _rule_btn("DOUBLE", "rule:around_mult:double")
        self.btn_around_mult_triple = _rule_btn("TRIPLE", "rule:around_mult:triple")
        self.btn_around_mult_any = _rule_btn("ANY", "rule:around_mult:any")
        row_mult.addWidget(self.btn_around_mult_single)
        row_mult.addWidget(self.btn_around_mult_double)
        row_mult.addWidget(self.btn_around_mult_triple)
        row_mult.addWidget(self.btn_around_mult_any)
        lay.addLayout(row_mult)

        self.lbl_around_order = QLabel("REDOSLIJED BROJEVA")
        self.lbl_around_order.setObjectName("SectionLabel")
        self.lbl_around_order.setAlignment(Qt.AlignLeft)
        lay.addWidget(self.lbl_around_order)
        row_order = QHBoxLayout()
        row_order.setSpacing(10)
        self.btn_around_order_standard = _rule_btn(
            "STANDARD", "rule:around_order:standard"
        )
        self.btn_around_order_random = _rule_btn(
            "RANDOM", "rule:around_order:random"
        )
        self.btn_around_order_reverse = _rule_btn(
            "REVERSE", "rule:around_order:reverse"
        )
        row_order.addWidget(self.btn_around_order_standard)
        row_order.addWidget(self.btn_around_order_random)
        row_order.addWidget(self.btn_around_order_reverse)
        lay.addLayout(row_order)

        self.lbl_around_backstep = QLabel("KAZNA UNAZAD")
        self.lbl_around_backstep.setObjectName("SectionLabel")
        self.lbl_around_backstep.setAlignment(Qt.AlignLeft)
        lay.addWidget(self.lbl_around_backstep)
        row_bs = QHBoxLayout()
        row_bs.setSpacing(10)
        self.btn_around_backstep_off = _rule_btn("ISKLJUČENO", "rule:around_backstep:0")
        self.btn_around_backstep_1 = _rule_btn("NATRAG 1", "rule:around_backstep:1")
        self.btn_around_backstep_2 = _rule_btn("NATRAG 2", "rule:around_backstep:2")
        row_bs.addWidget(self.btn_around_backstep_off)
        row_bs.addWidget(self.btn_around_backstep_1)
        row_bs.addWidget(self.btn_around_backstep_2)
        lay.addLayout(row_bs)

        lay.addStretch(1)
        self.rules_around_tutorial = _rules_footer_btn("UPUTE", "tutorial_open")
        self.rules_around_continue = _rules_continue_btn("NASTAVI", "rules_continue")
        self.rules_around_back = _rules_back_btn("NAZAD", "nav_back")
        lay.addLayout(
            _rules_footer_block(
                back_btn=self.rules_around_back,
                middle=self.rules_around_tutorial,
                continue_btn=self.rules_around_continue,
            )
        )
        return w

    def _build_rules_halve(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(18)
        title = QLabel("HALVE IT PRAVILA")
        title.setObjectName("Title")
        title.setAlignment(Qt.AlignCenter)
        self.rules_halve_title = title
        lay.addWidget(title)
        lay.addStretch(1)

        def _rule_btn(text: str, action: str) -> AnimButton:
            b = _btn(text, action, role="RuleOff")
            b.setMinimumHeight(88)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            return b

        self.lbl_halve_mode = QLabel("BROJ RUNDI")
        self.lbl_halve_mode.setObjectName("SectionLabel")
        self.lbl_halve_mode.setAlignment(Qt.AlignLeft)
        lay.addWidget(self.lbl_halve_mode)
        row = QHBoxLayout()
        row.setSpacing(20)
        self.btn_halve_standard = _rule_btn("STANDARD", "rule:halve_mode:standard")
        self.btn_halve_extended = _rule_btn("PRODUŽENO", "rule:halve_mode:extended")
        self.btn_halve_standard.setMinimumWidth(340)
        self.btn_halve_extended.setMinimumWidth(340)
        row.addWidget(self.btn_halve_standard)
        row.addWidget(self.btn_halve_extended)
        lay.addLayout(row)
        lay.addStretch(1)
        self.rules_halve_tutorial = _rules_footer_btn("UPUTE", "tutorial_open")
        self.rules_halve_continue = _rules_continue_btn("NASTAVI", "rules_continue")
        self.rules_halve_back = _rules_back_btn("NAZAD", "nav_back")
        lay.addLayout(
            _rules_footer_block(
                back_btn=self.rules_halve_back,
                middle=self.rules_halve_tutorial,
                continue_btn=self.rules_halve_continue,
            )
        )
        return w

    def _build_tutorial(self, kind: str) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(24, 16, 24, 16)
        lay.setSpacing(12)
        title = QLabel("UPUTE")
        title.setObjectName("Title")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 72px;")
        setattr(self, f"tutorial_{kind}_title", title)
        lay.addWidget(title)

        scroll = QScrollArea()
        scroll.setObjectName("TutorialScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        body_host = QWidget()
        body_lay = QVBoxLayout(body_host)
        body_lay.setContentsMargins(8, 4, 8, 12)
        body_lay.setSpacing(10)
        setattr(self, f"tutorial_{kind}_body_lay", body_lay)
        setattr(self, f"tutorial_{kind}_body_host", body_host)
        # Keep a body label ref for legacy callers; content is rebuilt in _fill_tutorial.
        body = QLabel("")
        body.setObjectName("TutorialBody")
        body.setWordWrap(True)
        body.hide()
        setattr(self, f"tutorial_{kind}_body", body)
        scroll.setWidget(body_host)
        lay.addWidget(scroll, 1)
        back = _rules_back_btn("NAZAD", "nav_back")
        setattr(self, f"tutorial_{kind}_back", back)
        lay.addLayout(_rules_footer_block(back_btn=back))
        return w

    def _build_select_players(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(16, 16, 16, 16)
        self.select_players_title = QLabel("BROJ IGRAČA")
        self.select_players_title.setObjectName("Title")
        self.select_players_title.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.select_players_title)
        grid = QGridLayout()
        grid.setSpacing(16)
        self._player_count_btns: Dict[int, QPushButton] = {}
        self._player_count_nums: Dict[int, QLabel] = {}
        self._player_count_icons: Dict[int, QLabel] = {}
        for n in range(1, 5):
            slot = n - 1
            b = AnimButton("")
            b.setProperty("action", f"player_slot:{slot}")
            b.setObjectName("Primary")
            b.setMinimumHeight(140)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            b.setCursor(Qt.ArrowCursor)
            b.setFocusPolicy(Qt.NoFocus)
            inner = QVBoxLayout(b)
            inner.setContentsMargins(12, 14, 12, 14)
            inner.setSpacing(8)
            num = QLabel("")
            num.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            num.setWordWrap(True)
            num.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            icons = QLabel()
            icons.setStyleSheet("background: transparent; border: none;")
            icons.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            icons.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            inner.addStretch(1)
            inner.addWidget(icons, 0, Qt.AlignHCenter)
            inner.addWidget(num, 0, Qt.AlignHCenter)
            inner.addStretch(1)
            self._player_count_btns[n] = b
            self._player_count_nums[n] = num
            self._player_count_icons[n] = icons
            grid.addWidget(b, (n - 1) // 2, (n - 1) % 2)
        lay.addLayout(grid, 1)
        self.select_players_continue = _rules_continue_btn("NASTAVI", "players_start")
        self.select_players_back = _rules_back_btn("NAZAD", "nav_back")
        for b in (self.select_players_back, self.select_players_continue):
            b.setMinimumHeight(78)
            b.setMaximumHeight(84)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.select_players_back.setStyleSheet(
            "QPushButton#Footer { min-height: 78px; max-height: 84px; "
            "font-size: 52px; font-weight: 800; padding: 6px 16px; "
            "border-color: #5a5a66; }"
        )
        foot = QHBoxLayout()
        foot.setSpacing(16)
        foot.addWidget(self.select_players_back, 1)
        foot.addWidget(self.select_players_continue, 1)
        lay.addLayout(foot)
        return w

    def _refresh_player_count_icons(self) -> None:
        if not hasattr(self, "_player_count_icons"):
            return
        slots = []
        try:
            slots = list(self.session.player_slots())
        except Exception:
            slots = [None, None, None, None]
        while len(slots) < 4:
            slots.append(None)
        n_sel = sum(1 for s in slots if isinstance(s, dict))
        mode = ""
        try:
            mode = str(self.session.state.get("selected_game_mode") or "").lower()
        except Exception:
            mode = ""
        min_n = 2 if mode == "killer" else 1
        if hasattr(self, "select_players_continue"):
            self.select_players_continue.setEnabled(n_sel >= min_n)
        for n in range(1, 5):
            slot = n - 1
            btn = self._player_count_btns.get(n)
            num = self._player_count_nums.get(n)
            icons = self._player_count_icons.get(n)
            color = _bgr_hex(slot)
            fg = _player_fg(color)
            data = slots[slot] if slot < len(slots) else None
            selected = isinstance(data, dict)
            name = str((data or {}).get("name") or "") if selected else ""
            if btn is not None:
                btn.setEnabled(True)
                if selected:
                    btn.setStyleSheet(
                        f"QPushButton#Primary, QPushButton#Primary:hover, QPushButton#Primary:focus {{"
                        f" background-color: #3a3a42;"
                        f" background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
                        f"  stop:0 #54545e, stop:0.42 #3a3a42, stop:0.58 #3a3a42, stop:1 #2c2c34);"
                        f" border: 5px solid {fg}; color: #ffffff; }}"
                        f" QPushButton#Primary:pressed {{"
                        f" background-color: #1e1e24;"
                        f" background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
                        f"  stop:0 #3a3a44, stop:0.5 #2a2a32, stop:1 #1c1c22); }}"
                    )
                else:
                    btn.setStyleSheet(
                        "QPushButton#Primary, QPushButton#Primary:hover, QPushButton#Primary:focus {"
                        " background-color: #3a3a42;"
                        " background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
                        "  stop:0 #54545e, stop:0.42 #3a3a42, stop:0.58 #3a3a42, stop:1 #2c2c34);"
                        " border: 3px solid #5a5a66; color: #ffffff; }"
                        " QPushButton#Primary:pressed {"
                        " background-color: #1e1e24;"
                        " background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
                        "  stop:0 #3a3a44, stop:0.5 #2a2a32, stop:1 #1c1c22); }"
                    )
            if selected and _name_initials(name):
                if icons is not None:
                    icons.hide()
                if num is not None:
                    num.show()
                    shown = name.upper()
                    num.setText(shown)
                    bw = max(80, btn.width() if btn is not None else 400)
                    bh = max(60, btn.height() if btn is not None else 200)
                    fs = _name_fit_font_px(shown, bw, bh)
                    num.setStyleSheet(
                        f"font-size: {fs}px; font-weight: 900; color: {fg}; "
                        "background: transparent; border: none; padding: 0px;"
                    )
            else:
                if num is not None:
                    num.setText("+" if not selected else "")
                    num.setStyleSheet(
                        f"font-size: 48px; font-weight: 800; color: {fg}; "
                        "background: transparent; border: none;"
                    )
                    num.setVisible(not selected)
                if icons is not None:
                    pix = _player_icon_tinted(72, fg)
                    if pix is not None:
                        icons.setPixmap(pix)
                    icons.show()


    def _build_clear_board(self) -> QWidget:
        w = QWidget()
        w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(16, 16, 16, 16)
        card = QFrame()
        card.setObjectName("Card")
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cl = QVBoxLayout(card)
        cl.setSpacing(18)
        self.clear_title = QLabel("OČISTITE PLOČU")
        self.clear_title.setObjectName("ClearHero")
        self.clear_title.setAlignment(Qt.AlignCenter)
        self.clear_title.setWordWrap(True)
        self.clear_s1 = QLabel("Uklonite sve strelice s ploče")
        self.clear_s1.setObjectName("ClearSub")
        self.clear_s1.setAlignment(Qt.AlignCenter)
        self.clear_s1.setWordWrap(True)
        self.clear_s2 = QLabel("prije početka igre")
        self.clear_s2.setObjectName("ClearSub")
        self.clear_s2.setAlignment(Qt.AlignCenter)
        self.clear_s2.setWordWrap(True)
        self.clear_status = QLabel("")
        self.clear_status.setAlignment(Qt.AlignCenter)
        self.clear_status.setWordWrap(True)
        # Ignored horizontal: dug tekst ne smije širiti QStackedWidget / cijeli kiosk.
        self.clear_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.clear_status.setMinimumWidth(0)
        self.clear_status.setStyleSheet(
            "color: #ffd24a; font-size: 36px; font-weight: 800; padding: 0 12px;"
        )
        for lab in (self.clear_title, self.clear_s1, self.clear_s2):
            lab.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            lab.setMinimumWidth(0)
        cl.addStretch(1)
        cl.addWidget(self.clear_title)
        cl.addWidget(self.clear_s1)
        cl.addWidget(self.clear_s2)
        cl.addWidget(self.clear_status)
        cl.addStretch(1)
        self.clear_confirm_btn = _btn("PLOČA JE PRAZNA", "clear_board_confirm", role="Accent", tall=True)
        cl.addWidget(self.clear_confirm_btn)
        lay.addWidget(card, 1)
        self.clear_board_back = _rules_back_btn("NAZAD", "nav_back")
        lay.addLayout(_rules_footer_block(back_btn=self.clear_board_back))
        return w

    def _build_exit_confirm(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(24)
        lay.addStretch(1)
        self.exit_title = QLabel("IZAĐI IZ IGRE?")
        self.exit_title.setObjectName("Title")
        self.exit_title.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.exit_title)
        row = QHBoxLayout()
        row.setSpacing(16)
        self.exit_yes = _btn("DA", "confirm_exit_yes", role="Danger", tall=True)
        self.exit_no = _btn("NE", "confirm_exit_no", role="Primary", tall=True)
        row.addWidget(self.exit_yes)
        row.addWidget(self.exit_no)
        lay.addLayout(row)
        lay.addStretch(1)
        return w

    def _build_calibration(self) -> QWidget:
        w = QWidget()
        w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)
        self.cal_click_banner = QLabel("")
        self.cal_click_banner.setAlignment(Qt.AlignCenter)
        self.cal_click_banner.setWordWrap(True)
        self.cal_click_banner.setStyleSheet(
            "color: #ffd24a; font-size: 26px; font-weight: 700; padding: 6px 8px;"
        )
        self.cal_click_banner.setVisible(False)
        lay.addWidget(self.cal_click_banner, 0)
        self.cal_label = QLabel("Kalibracija…")
        self.cal_label.setAlignment(Qt.AlignCenter)
        self.cal_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.cal_label.setMinimumSize(0, 0)
        self.cal_label.setScaledContents(False)
        self.cal_label.mousePressEvent = self._on_cal_click  # type: ignore
        lay.addWidget(self.cal_label, 1)
        row = QHBoxLayout()
        row.setSpacing(10)
        row.setContentsMargins(0, 6, 0, 0)
        self.cal_back_btn = _cal_footer_btn("NAZAD", "calibration_back", role="Danger")
        self.cal_click_bull_btn = _cal_footer_btn("BULL", "calibration_click_bull", role="Footer")
        self.cal_click_ellipse_btn = _cal_footer_btn("ELIPSA", "calibration_click_ellipse", role="Footer")
        self.cal_topdown_btn = _cal_footer_btn("TOPDOWN", "calibration_capture", role="Footer")
        self.cal_detect_btn = _cal_footer_btn("KALIBRIRAJ", "calibration_detect", role="Primary")
        row.addWidget(self.cal_back_btn, 1)
        row.addWidget(self.cal_click_bull_btn, 1)
        row.addWidget(self.cal_click_ellipse_btn, 1)
        row.addWidget(self.cal_topdown_btn, 1)
        row.addWidget(self.cal_detect_btn, 1)
        lay.addLayout(row)
        return w

    def _build_settings(self) -> QWidget:
        w = QWidget()
        w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(48, 20, 48, 20)
        lay.setSpacing(16)

        self.settings_title = QLabel("POSTAVKE")
        self.settings_title.setObjectName("SettingsTitle")
        self.settings_title.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.settings_title, 0)

        choice_w = 300

        def _row(label: QLabel, widget: QWidget) -> QHBoxLayout:
            row = QHBoxLayout()
            row.setSpacing(16)
            label.setWordWrap(True)
            label.setFixedWidth(200)
            label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            widget.setFixedWidth(choice_w)
            widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            widget.setMinimumHeight(64)
            widget.setMaximumHeight(72)
            row.addWidget(label, 0)
            row.addWidget(widget, 0)
            row.addStretch(1)
            return row

        left = QVBoxLayout()
        left.setSpacing(36)
        right = QVBoxLayout()
        right.setSpacing(36)

        self.settings_lang_lbl = QLabel("Jezik")
        self.settings_lang_lbl.setObjectName("SettingsSection")
        self.settings_lang = SettingsChoiceButton()
        self.settings_lang.set_icon_size(QPixmap(72, 42).size())
        for code, name, _flag in LANG_OPTIONS:
            self.settings_lang.add_item(f"  {name}", code, _flag_icon(code, 42))
        self.settings_lang.activated.connect(lambda _=None: self._on_settings_combo("lang"))
        left.addLayout(_row(self.settings_lang_lbl, self.settings_lang))

        self.settings_bg_lbl = QLabel("Boja pozadine")
        self.settings_bg_lbl.setObjectName("SettingsSection")
        self.settings_bg = SettingsChoiceButton()
        self.settings_bg.set_icon_size(QPixmap(160, 36).size())
        for hex_c, _key in BG_PRESETS:
            self.settings_bg.add_item("", hex_c, _color_swatch_icon(hex_c, 160, 36))
        self.settings_bg.activated.connect(lambda _=None: self._on_settings_combo("bg"))
        left.addLayout(_row(self.settings_bg_lbl, self.settings_bg))

        self.settings_btn_lbl = QLabel("Boja gumba")
        self.settings_btn_lbl.setObjectName("SettingsSection")
        self.settings_btn = SettingsChoiceButton()
        self.settings_btn.set_icon_size(QPixmap(160, 36).size())
        for hex_c, _key in BTN_PRESETS:
            self.settings_btn.add_item("", hex_c, _color_swatch_icon(hex_c, 160, 36))
        self.settings_btn.activated.connect(lambda _=None: self._on_settings_combo("btn"))
        left.addLayout(_row(self.settings_btn_lbl, self.settings_btn))

        self.settings_start_lbl = QLabel("Pokretanje igre")
        self.settings_start_lbl.setObjectName("SettingsSection")
        self.settings_start = SettingsChoiceButton()
        for mode, key in START_MODES:
            self.settings_start.add_item(get_settings().t(key), mode)
        self.settings_start.activated.connect(lambda _=None: self._on_settings_combo("start"))
        left.addLayout(_row(self.settings_start_lbl, self.settings_start))

        self.settings_quit_lbl = QLabel("Aplikacija")
        self.settings_quit_lbl.setObjectName("SettingsSection")
        self.settings_power = SettingsChoiceButton()
        for mode, key in POWER_OPTIONS:
            self.settings_power.add_item(get_settings().t(key), mode)
        self.settings_power.set_current_data("")
        self.settings_power.activated.connect(lambda _=None: self._on_settings_combo("power"))
        left.addLayout(_row(self.settings_quit_lbl, self.settings_power))
        left.addStretch(1)

        self.settings_device_lbl = QLabel("Uređaj")
        self.settings_device_lbl.setObjectName("SettingsSection")
        self.settings_device = SettingsChoiceButton()
        for did in DEVICE_IDS:
            self.settings_device.add_item(did, did)
        self.settings_device.activated.connect(
            lambda _=None: self._on_settings_combo("device")
        )
        right.addLayout(_row(self.settings_device_lbl, self.settings_device))

        self.settings_league_qr_lbl = QLabel("Ligaški QR")
        self.settings_league_qr_lbl.setObjectName("SettingsSection")
        self.settings_league_qr = SettingsChoiceButton()
        for mode, key in LEAGUE_QR_OPTIONS:
            self.settings_league_qr.add_item(get_settings().t(key), mode)
        self.settings_league_qr.activated.connect(
            lambda _=None: self._on_settings_combo("league_qr")
        )
        right.addLayout(_row(self.settings_league_qr_lbl, self.settings_league_qr))

        self.settings_auto_cal_lbl = QLabel("Auto kalibracija")
        self.settings_auto_cal_lbl.setObjectName("SettingsSection")
        self.settings_auto_cal = SettingsChoiceButton()
        for mode, key in AUTO_CALIBRATE_OPTIONS:
            self.settings_auto_cal.add_item(get_settings().t(key), mode)
        self.settings_auto_cal.activated.connect(
            lambda _=None: self._on_settings_combo("auto_calibrate")
        )
        right.addLayout(_row(self.settings_auto_cal_lbl, self.settings_auto_cal))

        self.settings_pw_lbl = QLabel("Lozinka")
        self.settings_pw_lbl.setObjectName("SettingsSection")
        self.settings_pw_btn = _btn(
            "Promijeni", "settings_change_password", role="", tall=False
        )
        self.settings_pw_btn.setObjectName("SettingsPw")
        self.settings_pw_btn.setStyleSheet(
            "QPushButton#SettingsPw {"
            "  background-color: #2a2a32;"
            "  color: #ffffff;"
            "  border: 3px solid #5a5a66;"
            "  border-radius: 12px;"
            "  font-size: 32px;"
            "  font-weight: 700;"
            "  padding: 0px;"
            "  min-height: 64px;"
            "  max-height: 72px;"
            "}"
            "QPushButton#SettingsPw:hover, QPushButton#SettingsPw:focus {"
            "  background-color: #32323a;"
            "}"
        )
        right.addLayout(_row(self.settings_pw_lbl, self.settings_pw_btn))
        right.addStretch(1)

        cols = QHBoxLayout()
        cols.setSpacing(64)
        cols.addLayout(left, 1)
        cols.addLayout(right, 1)
        lay.addLayout(cols, 1)

        foot = QHBoxLayout()
        foot.setSpacing(16)
        self.settings_back_btn = _btn("NATRAG", "settings_back", role="Footer", tall=True)
        self.settings_cal_btn = _btn("KALIBRACIJA", "settings_calibration", role="Accent", tall=True)
        foot.addWidget(self.settings_back_btn, 1)
        foot.addWidget(self.settings_cal_btn, 1)
        lay.addLayout(foot)
        return w

    def _sync_settings_combos(self) -> None:
        s = get_settings()
        if not hasattr(self, "settings_lang"):
            return
        self.settings_lang.set_current_data(s.language)
        self.settings_bg.set_current_data(s.bg_color)
        self.settings_btn.set_current_data(s.button_color)
        self.settings_start.set_current_data(s.start_mode)
        for i, (_mode, key) in enumerate(START_MODES):
            if i < self.settings_start.count():
                self.settings_start.set_item_text(i, s.t(key))
        if hasattr(self, "settings_device"):
            known = {
                str(self.settings_device.item_data(i))
                for i in range(self.settings_device.count())
            }
            if s.device_id not in known:
                self.settings_device.add_item(s.device_id, s.device_id)
            self.settings_device.set_current_data(s.device_id)
        if hasattr(self, "settings_league_qr"):
            self.settings_league_qr.set_current_data(
                "on" if s.league_qr_enabled else "off"
            )
            for i, (_mode, key) in enumerate(LEAGUE_QR_OPTIONS):
                if i < self.settings_league_qr.count():
                    self.settings_league_qr.set_item_text(i, s.t(key))
        if hasattr(self, "settings_auto_cal"):
            self.settings_auto_cal.set_current_data("on" if s.auto_calibrate else "off")
            for i, (_mode, key) in enumerate(AUTO_CALIBRATE_OPTIONS):
                if i < self.settings_auto_cal.count():
                    self.settings_auto_cal.set_item_text(i, s.t(key))
        if hasattr(self, "settings_power"):
            for i, (_mode, key) in enumerate(POWER_OPTIONS):
                if i < self.settings_power.count():
                    self.settings_power.set_item_text(i, s.t(key))
            self.settings_power.set_current_data("")

    def _apply_i18n(self) -> None:
        s = get_settings()
        t = s.t
        if hasattr(self, "settings_title"):
            self.settings_title.setText(t("settings"))
            self.settings_lang_lbl.setText(t("language"))
            self.settings_bg_lbl.setText(t("bg_color"))
            self.settings_btn_lbl.setText(t("btn_color"))
            self.settings_start_lbl.setText(t("start_mode"))
            if hasattr(self, "settings_device_lbl"):
                self.settings_device_lbl.setText(t("device"))
            if hasattr(self, "settings_league_qr_lbl"):
                self.settings_league_qr_lbl.setText(t("league_qr"))
            self.settings_cal_btn.setText(t("calibration"))
            self.settings_back_btn.setText(t("back"))
            if hasattr(self, "settings_auto_cal_lbl"):
                self.settings_auto_cal_lbl.setText(t("auto_calibrate"))
            if hasattr(self, "settings_pw_lbl"):
                self.settings_pw_lbl.setText(t("password"))
            if hasattr(self, "settings_pw_btn"):
                self.settings_pw_btn.setText(t("change_password_short"))
            if hasattr(self, "settings_quit_lbl"):
                self.settings_quit_lbl.setText(t("application"))
        if hasattr(self, "select_game_title"):
            self.select_game_title.setText(t("select_game"))
        if hasattr(self, "select_players_title"):
            self.select_players_title.setText(t("select_players"))
        if hasattr(self, "select_players_continue"):
            self.select_players_continue.setText(t("continue"))
        if hasattr(self, "clear_title"):
            self.clear_title.setText(t("clear_board_title"))
            self.clear_s1.setText(t("clear_board_s1"))
            self.clear_s2.setText(t("clear_board_s2"))
            self.clear_confirm_btn.setText(t("board_empty"))
        if hasattr(self, "exit_title"):
            self.exit_title.setText(t("exit_confirm"))
            self.exit_yes.setText(t("yes"))
            self.exit_no.setText(t("no"))
        if hasattr(self, "quit_title"):
            self.quit_title.setText(t("power_exit_confirm"))
            if hasattr(self, "quit_yes_btn"):
                self.quit_yes_btn.setText(t("yes"))
            if hasattr(self, "quit_no_btn"):
                self.quit_no_btn.setText(t("no"))
        if hasattr(self, "cal_detect_btn"):
            self.cal_detect_btn.setText(t("calibrate"))
            self.cal_back_btn.setText(t("back"))
            if hasattr(self, "cal_click_bull_btn"):
                self.cal_click_bull_btn.setText(t("click_bull"))
            if hasattr(self, "cal_click_ellipse_btn"):
                self.cal_click_ellipse_btn.setText(t("click_ellipse"))
            if hasattr(self, "cal_click_banner"):
                self.cal_click_banner.setText(t("click_bull_hint"))
        if hasattr(self, "standby_start_btn"):
            self.standby_start_btn.setText(t("start_game"))
        if hasattr(self, "rules_x01_tutorial"):
            self.rules_x01_tutorial.setText(t("tutorial").upper())
        if hasattr(self, "rules_x01_continue"):
            self.rules_x01_continue.setText(t("continue").upper())
        if hasattr(self, "rules_x01_back"):
            self.rules_x01_back.setText(t("back").upper())
        if hasattr(self, "rules_cricket_tutorial"):
            self.rules_cricket_tutorial.setText(t("tutorial").upper())
        if hasattr(self, "rules_cricket_continue"):
            self.rules_cricket_continue.setText(t("continue").upper())
        if hasattr(self, "rules_cricket_back"):
            self.rules_cricket_back.setText(t("back").upper())
        if hasattr(self, "rules_cricket_title"):
            self.rules_cricket_title.setText(f"CRICKET {t('rules')}")
        if hasattr(self, "rules_killer_tutorial"):
            self.rules_killer_tutorial.setText(t("tutorial").upper())
        if hasattr(self, "rules_killer_continue"):
            self.rules_killer_continue.setText(t("continue").upper())
        if hasattr(self, "rules_killer_back"):
            self.rules_killer_back.setText(t("back").upper())
        if hasattr(self, "rules_killer_title"):
            self.rules_killer_title.setText(f"KILLER {t('rules')}".upper())
        if hasattr(self, "rules_around_tutorial"):
            self.rules_around_tutorial.setText(t("tutorial").upper())
            self.rules_around_title.setText(f"AROUND {t('rules')}".upper())
        if hasattr(self, "rules_around_continue"):
            self.rules_around_continue.setText(t("continue").upper())
        if hasattr(self, "rules_around_back"):
            self.rules_around_back.setText(t("back").upper())
        if hasattr(self, "rules_halve_tutorial"):
            self.rules_halve_tutorial.setText(t("tutorial").upper())
            self.rules_halve_title.setText(f"HALVE IT {t('rules')}".upper())
        if hasattr(self, "rules_halve_continue"):
            self.rules_halve_continue.setText(t("continue").upper())
        if hasattr(self, "rules_halve_back"):
            self.rules_halve_back.setText(t("back").upper())
        if hasattr(self, "lbl_halve_mode"):
            self.lbl_halve_mode.setText(t("halve_rounds").upper())
            self.btn_halve_standard.setText(t("halve_standard").upper())
            self.btn_halve_extended.setText(t("halve_extended").upper())
        for kind in ("x01", "cricket", "killer", "around", "halve"):
            title = getattr(self, f"tutorial_{kind}_title", None)
            back = getattr(self, f"tutorial_{kind}_back", None)
            if title is not None:
                title.setText(t("tutorial").upper())
            if back is not None:
                back.setText(t("back").upper())
            self._fill_tutorial(kind)
        if hasattr(self, "lbl_killer_lives"):
            self.lbl_killer_lives.setText(t("killer_lives").upper())
            self.lbl_killer_assign.setText(t("killer_assign").upper())
            self.lbl_killer_activation.setText(t("killer_activation").upper())
            self.btn_killer_assign_random.setText(t("killer_assign_random").upper())
            self.btn_killer_assign_throw.setText(t("killer_assign_throw").upper())
            self.btn_killer_act_single.setText(t("killer_act_single").upper())
            self.btn_killer_act_double.setText(t("killer_act_double").upper())
            self.btn_killer_act_triple.setText(t("killer_act_triple").upper())
        if hasattr(self, "lbl_around_mult"):
            self.lbl_around_mult.setText(t("around_target_mult").upper())
            self.lbl_around_order.setText(t("around_number_order").upper())
            self.lbl_around_backstep.setText(t("around_backstep").upper())
            self.btn_around_mult_single.setText(t("around_mult_single").upper())
            self.btn_around_mult_double.setText(t("around_mult_double").upper())
            self.btn_around_mult_triple.setText(t("around_mult_triple").upper())
            self.btn_around_mult_any.setText(t("around_mult_any").upper())
            self.btn_around_order_standard.setText(t("around_order_standard").upper())
            self.btn_around_order_random.setText(t("around_order_random").upper())
            self.btn_around_order_reverse.setText(t("around_order_reverse").upper())
            self.btn_around_backstep_off.setText(t("around_backstep_off").upper())
            back_lab = t("around_back").upper()
            self.btn_around_backstep_1.setText(f"{back_lab} 1")
            self.btn_around_backstep_2.setText(f"{back_lab} 2")
        if hasattr(self, "killer_bull_title"):
            self.killer_bull_title.setText(t("killer_bull_prompt").upper())
            self.killer_bull_false_btn.setText(t("killer_bull_false").upper())
        if hasattr(self, "btn_max_off"):
            self.btn_max_off.setText(t("no_limit"))
            self.btn_max_20.setText(t("rounds_20"))
            self.btn_c_std.setText(t("cricket_standard"))
            self.btn_c_nos.setText(t("cricket_no_score"))
            self.btn_c_cut.setText(t("cricket_cut"))
        if hasattr(self, "lbl_game_mode"):
            self.lbl_game_mode.setText(t("game_mode"))
        if hasattr(self, "undo_btn") and self.undo_btn.isVisible():
            # Ne diraj wide exit na winner ekranu.
            if self.exit_btn.objectName() != "PlayExitWide":
                self.undo_btn.setText(t("back"))
        if hasattr(self, "play_cal_btn"):
            self.play_cal_btn.setText(_calib_short_label())
        if hasattr(self, "ingame_cal_title"):
            self.ingame_cal_title.setText(t("ingame_cal_title"))
            self.ingame_cal_s1.setText(t("ingame_cal_s1"))
            self.ingame_cal_s2.setText(t("ingame_cal_s2"))
            self.ingame_cal_empty_btn.setText(t("board_empty"))
            self.ingame_cal_back_btn.setText(t("back"))
            self.ingame_cal_done_btn.setText(t("ingame_cal_done"))
            self.ingame_cal_retry_btn.setText(t("ingame_cal_retry"))
        if hasattr(self, "idle_title"):
            self.idle_title.setText(t("idle_title"))
            self.idle_s1.setText(t("idle_s1"))
            self.idle_quit_btn.setText(t("idle_quit"))
            self.idle_continue_btn.setText(t("idle_continue"))
        self._refresh_player_count_icons()

    def _on_cal_click(self, event) -> None:
        if self.cal_label.pixmap() is None:
            return
        pix = self.cal_label.pixmap()
        lw, lh = self.cal_label.width(), self.cal_label.height()
        # Prefer device-independent size so HiDPI letterbox math matches widget coords.
        try:
            dis = pix.deviceIndependentSize()
            pw, ph = int(dis.width()), int(dis.height())
        except Exception:
            pw, ph = pix.width(), pix.height()
        if pw <= 0 or ph <= 0:
            return
        scale = min(lw / float(pw), lh / float(ph))
        if scale <= 0:
            return
        dw, dh = pw * scale, ph * scale
        ox = (lw - dw) * 0.5
        oy = (lh - dh) * 0.5
        lx = float(event.position().x())
        ly = float(event.position().y())
        if lx < ox or ly < oy or lx >= ox + dw or ly >= oy + dh:
            return
        x = int((lx - ox) / scale)
        y = int((ly - oy) / scale)
        x = max(0, min(pw - 1, x))
        y = max(0, min(ph - 1, y))
        self.session.handle_calibration_click(x, y, pw, ph)
        self._last_ui_sig = None
        self.refresh()

    def _build_playing(self) -> QWidget:
        w = QWidget()
        w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root = QVBoxLayout(w)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        body = QHBoxLayout()
        body.setSpacing(12)

        left = QVBoxLayout()
        left.setSpacing(6)

        title_h = 52
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(12)
        self.play_logo = _make_logo_label(title_h)
        self.play_title = QLabel("")
        self.play_title.setObjectName("PlayTitle")
        self.play_title.setFixedHeight(title_h)
        self.play_title.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        title_row.addWidget(self.play_logo, 0, Qt.AlignVCenter)
        title_row.addWidget(self.play_title, 1, Qt.AlignVCenter)
        left.addLayout(title_row)

        self.left_stack = QStackedWidget()
        self.left_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.hits_wrap = QWidget()
        hits_l = QVBoxLayout(self.hits_wrap)
        hits_l.setContentsMargins(0, 0, 0, 0)
        hits_l.setSpacing(10)
        self.hit_btns = []
        for i in range(3):
            b = _btn("-", f"hit_slot:{i}", role="HitSlot")
            b.setMinimumHeight(64)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.hit_btns.append(b)
            hits_l.addWidget(b, 1)

        # Ručni način: hitovi u fiksnom redu iznad tipkovnice (ne mijenja desni layout).
        self.manual_hits_wrap = QWidget()
        self.manual_hits_wrap.setFixedHeight(118)
        self.manual_hits_wrap.hide()
        mh = QHBoxLayout(self.manual_hits_wrap)
        mh.setContentsMargins(0, 0, 0, 4)
        mh.setSpacing(8)
        self.manual_hit_btns = []
        for i in range(3):
            b = _btn("-", f"hit_slot:{i}", role="HitSlot")
            b.setMinimumHeight(0)
            b.setMaximumHeight(110)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            b.setStyleSheet(
                "QPushButton#HitSlot { font-size: 72px; font-weight: 800; "
                "min-height: 0px; padding: 2px; }"
            )
            self.manual_hit_btns.append(b)
            mh.addWidget(b, 1)

        self.keypad_wrap = QWidget()
        kp = QVBoxLayout(self.keypad_wrap)
        kp.setContentsMargins(0, 0, 0, 0)
        kp.setSpacing(4)
        grid = QGridLayout()
        grid.setSpacing(4)
        grid.setContentsMargins(0, 0, 0, 0)
        self.mult2 = _btn("D", "mult:2", role="KeyMod")
        self.mult3 = _btn("T", "mult:3", role="KeyMod")
        b25 = _btn("B25", "bull:25", role="KeyBull")
        b50 = _btn("B50", "bull:50", role="KeyBull")
        miss = _btn("MISS", "hit:miss", role="KeyMiss")
        for b in (self.mult2, self.mult3, b25, b50, miss):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            b.setMinimumSize(0, 0)
        # Prvi red: D T 25 50 MISS
        grid.addWidget(self.mult2, 0, 0)
        grid.addWidget(self.mult3, 0, 1)
        grid.addWidget(b25, 0, 2)
        grid.addWidget(b50, 0, 3)
        grid.addWidget(miss, 0, 4)
        for n in range(1, 21):
            r, c = divmod(n - 1, 5)
            nb = _btn(str(n), f"num:{n}", role="KeyNum")
            nb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            nb.setMinimumSize(0, 0)
            grid.addWidget(nb, r + 1, c)
        clr = _btn("CLR", "clear_one", role="KeyClr")
        key_x = _btn("X", "keypad_close", role="KeyClose")
        self.key_clr_btn = clr
        self.key_close_btn = key_x
        clr.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        key_x.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        clr.setMinimumSize(0, 0)
        key_x.setMinimumSize(0, 0)
        # Zadnji red: CLR (4) + X (1)
        self.keypad_grid = grid
        grid.addWidget(clr, 5, 0, 1, 4)
        grid.addWidget(key_x, 5, 4, 1, 1)
        for r in range(6):
            grid.setRowStretch(r, 1)
        for c in range(5):
            grid.setColumnStretch(c, 1)
        kp.addLayout(grid, 1)

        self.left_stack.addWidget(self.hits_wrap)
        self.left_stack.addWidget(self.keypad_wrap)
        left.addWidget(self.manual_hits_wrap, 0)
        left.addWidget(self.left_stack, 1)

        right = QVBoxLayout()
        right.setSpacing(6)
        self.right_layout = right

        # Scoreboard bez stretch-a — live ostaje odmah ispod (winner: stretch=1)
        self.score_box = QVBoxLayout()
        self.score_box.setSpacing(6)
        self.score_box.setContentsMargins(0, 0, 0, 0)
        self.score_box.setAlignment(Qt.AlignTop)
        right.addLayout(self.score_box, 0)

        self.score_suggest_sep = QFrame()
        self.score_suggest_sep.setObjectName("ScoreSuggestSep")
        self.score_suggest_sep.setFixedHeight(3)
        self.score_suggest_sep.hide()
        right.addWidget(self.score_suggest_sep, 0)

        # X01: green LiveScore suggestions. Killer: hidden (lives are on the cards).
        self.live_stack = QStackedWidget()
        self.live_stack.setMinimumHeight(80)
        self.live_line = QLabel(" ")
        self.live_line.setObjectName("LiveScore")
        self.live_line.setWordWrap(True)
        self.live_line.setMinimumHeight(80)
        self.live_line.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.killer_numbers_line = QWidget()
        self.killer_numbers_line.setObjectName("KillerNumbersLine")
        self.killer_numbers_line.setMinimumHeight(0)
        self.killer_numbers_lay = QHBoxLayout(self.killer_numbers_line)
        self.killer_numbers_lay.setContentsMargins(0, 0, 0, 0)
        self.killer_numbers_lay.setSpacing(22)
        self.killer_numbers_lay.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.live_stack.addWidget(self.live_line)
        self.live_stack.addWidget(self.killer_numbers_line)
        self.live_stack.setCurrentWidget(self.live_line)
        right.addWidget(self.live_stack, 0)

        right.addStretch(1)

        # Donji blok: OČISTI iznad NAZAD+X — fiksna visina da se ne reže
        self.footer_block = QWidget()
        self.footer_block.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        footer_l = QVBoxLayout(self.footer_block)
        footer_l.setContentsMargins(0, 4, 0, 4)
        footer_l.setSpacing(12)

        FOOT_H = 140
        NAV_H = 84
        self.action_stack = QStackedWidget()
        self.action_stack.setFixedHeight(FOOT_H)
        self.action_empty = QWidget()
        self.board_msg = QLabel("")
        self.board_msg.setObjectName("StatusBanner")
        self.board_msg.setAlignment(Qt.AlignCenter)
        self.board_msg.setWordWrap(True)
        self.board_msg.setFixedHeight(FOOT_H)
        self.advance_btn = _btn(">", "manual_advance_confirm", role="PlayAction")
        self.advance_btn.setFixedHeight(FOOT_H)
        self.action_stack.addWidget(self.action_empty)
        self.action_stack.addWidget(self.board_msg)
        self.action_stack.addWidget(self.advance_btn)
        footer_l.addWidget(self.action_stack)

        self.bottom_row = QHBoxLayout()
        self.bottom_row.setSpacing(12)
        self.undo_btn = _btn("NAZAD", "undo_turn", role="PlayNav")
        self.mode_toggle_btn = _btn("AUTO", "toggle_manual_mode", role="PlayNav")
        self.play_cal_btn = _btn(_calib_short_label(), "ingame_cal_open", role="PlayNav")
        self.exit_btn = _btn("X", "request_exit_game", role="PlayExit")
        self.undo_btn.setFixedHeight(NAV_H)
        self.mode_toggle_btn.setFixedHeight(NAV_H)
        self.mode_toggle_btn.setMinimumWidth(140)
        self.play_cal_btn.setFixedHeight(NAV_H)
        self.play_cal_btn.setMinimumWidth(160)
        self.exit_btn.setFixedHeight(NAV_H)
        self.exit_btn.setMinimumWidth(96)
        self.exit_btn.setMaximumWidth(120)
        # Stretch lijevo/desno = 0 u igri; na winner ekranu = 1 za centriranje
        self.bottom_row.addStretch(0)
        self.bottom_row.addWidget(self.undo_btn, 1)
        self.bottom_row.addWidget(self.mode_toggle_btn, 1)
        self.bottom_row.addWidget(self.play_cal_btn, 1)
        self.bottom_row.addWidget(self.exit_btn, 0)
        self.bottom_row.addStretch(0)
        footer_l.addLayout(self.bottom_row)
        self.footer_block.setFixedHeight(FOOT_H + NAV_H + 12 + 8)

        right.addWidget(self.footer_block, 0)

        self.left_w = QWidget()
        self.left_w.setLayout(left)
        self.left_w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.right_w = QWidget()
        self.right_w.setLayout(right)
        self.right_w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.play_body = body
        body.addWidget(self.left_w, 5)
        body.addWidget(self.right_w, 5)
        root.addLayout(body, 1)
        return w

    def _update_calibration_view(self) -> None:
        # Match pixmap to label (Qt footer buttons are outside the label).
        img = self.session.get_calibration_frame(
            max(800, self.cal_label.width()),
            max(450, self.cal_label.height()),
        )
        if img is None:
            return
        rgb = img[:, :, ::-1].copy()
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.cal_label.setPixmap(QPixmap.fromImage(qimg.copy()))

    def refresh(self) -> None:
        snap = self.session.snapshot_for_ui()
        sig = self._ui_signature(snap)
        if sig == self._last_ui_sig and snap.get("screen") != "calibration":
            self._update_settings_fab_visibility()
            return
        self._last_ui_sig = sig
        screen = snap["screen"]
        prev_screen = getattr(self, "_ui_screen", None)
        self._ui_screen = screen
        page = self.pages.get(screen)
        if page is not None:
            self.stack.setCurrentWidget(page)
        lang = get_settings().language
        if self._i18n_lang != lang or self._i18n_screen != screen:
            self._i18n_lang = lang
            self._i18n_screen = screen
            self._apply_i18n()
        self._update_settings_fab_visibility()
        # Wallpaper only scales on resize; refresh just repositions overlays/fab.
        self._place_chrome(wallpaper=False)
        self._sync_ingame_cal_overlay(snap)
        if screen != "playing" and hasattr(self, "killer_bull_overlay"):
            self.killer_bull_overlay.hide()
        if screen != "select_players" and hasattr(self, "name_overlay"):
            self.name_overlay.hide()
        # Exit / menu: cancel score anims and drop cached card scores.
        # Enter playing: also rebuild so a new match never tweens from leftovers.
        if screen != "playing" or prev_screen != "playing":
            self._teardown_scoreboard_ui()

        if screen == "standby":
            s = get_settings()
            mode = s.start_mode
            if mode == "button":
                self.standby_sub.setText(s.t("standby_button"))
                self.qr_frame.hide()
                self.standby_start_btn.setText(s.t("start_game"))
                self.standby_start_btn.show()
            else:
                self.standby_sub.setText(s.t("standby_qr"))
                self.standby_start_btn.hide()
                self.qr_frame.show()
                qr = snap.get("qr_path")
                if qr:
                    pix = QPixmap(qr)
                    self.qr_label.setPixmap(
                        _rounded_pixmap(
                            pix.scaled(560, 560, Qt.KeepAspectRatio, _PIXMAP_TRANSFORM),
                            20,
                        )
                    )

        if screen == "settings":
            self._sync_settings_combos()

        if screen == "rules_x01":
            mode = snap.get("selected_game_mode") or "X01"
            self.rules_x01_title.setText(f"{str(mode).upper()} {get_settings().t('rules')}")
            rp = snap["rules_pending"]
            di_on = bool(rp.get("double_in"))
            do_on = bool(rp.get("double_out"))
            yn = (get_settings().t("yes"), get_settings().t("no"))
            self.btn_double_in.setText(f"{get_settings().t('double_in')}: {yn[0] if di_on else yn[1]}")
            self.btn_double_out.setText(f"{get_settings().t('double_out')}: {yn[0] if do_on else yn[1]}")
            self.btn_double_in.setObjectName("RuleOn" if di_on else "RuleOff")
            self.btn_double_out.setObjectName("RuleOn" if do_on else "RuleOff")
            for b in (self.btn_double_in, self.btn_double_out):
                b.style().unpolish(b)
                b.style().polish(b)
                b.update()
                b.repaint()

        if screen == "rules_cricket":
            rp = snap["rules_pending"]
            self.btn_max_off.setObjectName("RuleOn" if not rp.get("max_rounds_20") else "RuleOff")
            self.btn_max_20.setObjectName("RuleOn" if rp.get("max_rounds_20") else "RuleOff")
            cm = str(rp.get("cricket_mode", "standard"))
            self.btn_c_std.setObjectName("RuleOn" if cm == "standard" else "RuleOff")
            self.btn_c_nos.setObjectName("RuleOn" if cm == "no_score" else "RuleOff")
            self.btn_c_cut.setObjectName("RuleOn" if cm == "cut_throat" else "RuleOff")
            for b in (
                self.btn_max_off,
                self.btn_max_20,
                self.btn_c_std,
                self.btn_c_nos,
                self.btn_c_cut,
            ):
                # Forsiraj ponovno učitavanje QSS nakon promjene objectName.
                b.style().unpolish(b)
                b.style().polish(b)
                b.update()
                b.repaint()

        if screen == "rules_killer":
            rp = snap["rules_pending"]
            t = get_settings().t
            lives = int(rp.get("killer_lives", 3) or 3)
            for n, b in self.btn_killer_lives.items():
                b.setObjectName("RuleOn" if n == lives else "RuleOff")
            assign = str(rp.get("killer_assign", "random")).lower()
            self.btn_killer_assign_random.setObjectName(
                "RuleOn" if assign == "random" else "RuleOff"
            )
            self.btn_killer_assign_throw.setObjectName(
                "RuleOn" if assign == "throw" else "RuleOff"
            )
            act = str(rp.get("killer_activation", "double")).lower()
            self.btn_killer_act_single.setObjectName("RuleOn" if act == "single" else "RuleOff")
            self.btn_killer_act_double.setObjectName("RuleOn" if act == "double" else "RuleOff")
            self.btn_killer_act_triple.setObjectName("RuleOn" if act == "triple" else "RuleOff")
            for b in list(self.btn_killer_lives.values()) + [
                self.btn_killer_assign_random,
                self.btn_killer_assign_throw,
                self.btn_killer_act_single,
                self.btn_killer_act_double,
                self.btn_killer_act_triple,
            ]:
                b.style().unpolish(b)
                b.style().polish(b)
                b.update()
                b.repaint()

        if screen == "rules_around":
            rp = snap["rules_pending"]
            mult = str(rp.get("around_mult", "any")).lower()
            self.btn_around_mult_single.setObjectName(
                "RuleOn" if mult == "single" else "RuleOff"
            )
            self.btn_around_mult_double.setObjectName(
                "RuleOn" if mult == "double" else "RuleOff"
            )
            self.btn_around_mult_triple.setObjectName(
                "RuleOn" if mult == "triple" else "RuleOff"
            )
            self.btn_around_mult_any.setObjectName(
                "RuleOn" if mult == "any" else "RuleOff"
            )
            order = str(rp.get("around_order", "standard")).lower()
            self.btn_around_order_standard.setObjectName(
                "RuleOn" if order == "standard" else "RuleOff"
            )
            self.btn_around_order_random.setObjectName(
                "RuleOn" if order == "random" else "RuleOff"
            )
            self.btn_around_order_reverse.setObjectName(
                "RuleOn" if order == "reverse" else "RuleOff"
            )
            bs = int(rp.get("around_backstep", 0) or 0)
            self.btn_around_backstep_off.setObjectName(
                "RuleOn" if bs == 0 else "RuleOff"
            )
            self.btn_around_backstep_1.setObjectName(
                "RuleOn" if bs == 1 else "RuleOff"
            )
            self.btn_around_backstep_2.setObjectName(
                "RuleOn" if bs == 2 else "RuleOff"
            )
            for b in (
                self.btn_around_mult_single,
                self.btn_around_mult_double,
                self.btn_around_mult_triple,
                self.btn_around_mult_any,
                self.btn_around_order_standard,
                self.btn_around_order_random,
                self.btn_around_order_reverse,
                self.btn_around_backstep_off,
                self.btn_around_backstep_1,
                self.btn_around_backstep_2,
            ):
                b.style().unpolish(b)
                b.style().polish(b)
                b.update()
                b.repaint()

        if screen == "rules_halve":
            rp = snap["rules_pending"]
            hm = str(rp.get("halve_mode", "standard")).lower()
            self.btn_halve_standard.setObjectName(
                "RuleOn" if hm == "standard" else "RuleOff"
            )
            self.btn_halve_extended.setObjectName(
                "RuleOn" if hm == "extended" else "RuleOff"
            )
            for b in (self.btn_halve_standard, self.btn_halve_extended):
                b.style().unpolish(b)
                b.style().polish(b)
                b.update()
                b.repaint()

        if screen == "select_players":
            self._refresh_player_count_icons()

        if screen.startswith("tutorial_"):
            t = get_settings().t
            mode = str(snap.get("selected_game_mode") or "").upper()
            kind = screen.replace("tutorial_", "", 1)
            title_w = getattr(self, f"tutorial_{kind}_title", None)
            if title_w is not None:
                if kind == "x01":
                    title_w.setText(f"{mode or 'X01'} {t('tutorial')}".upper())
                elif kind == "cricket":
                    title_w.setText(f"CRICKET {t('tutorial')}".upper())
                elif kind == "killer":
                    title_w.setText(f"KILLER {t('tutorial')}".upper())
                elif kind == "around":
                    title_w.setText(f"AROUND {t('tutorial')}".upper())
                elif kind == "halve":
                    title_w.setText(f"HALVE IT {t('tutorial')}".upper())
            self._fill_tutorial(kind, snap)

        if screen == "clear_board":
            status = str(snap.get("clear_board_status") or "")
            self.clear_status.setText(status)
            # Reset size hint kad se status skraćuje / briše (sprječi "zalijepljeni" stretch).
            self.clear_status.adjustSize()
            page = self.pages.get("clear_board")
            if page is not None:
                page.updateGeometry()
            self.stack.updateGeometry()

        if screen == "calibration":
            self._sync_cal_toolbar_ui(snap)

        if screen == "playing":
            self._refresh_playing(snap)

    def _style_cal_toggle(self, btn, on: bool, *, enabled: bool = True) -> None:
        role = "Accent" if (on and enabled) else "Footer"
        btn.setObjectName(role)
        btn.setEnabled(enabled)
        btn.setStyleSheet(
            f"QPushButton#{role} {{ min-height: 72px; max-height: 80px; "
            f"font-size: 32px; font-weight: 800; padding: 4px 6px; }}"
            f"QPushButton#{role}:disabled {{ background-color: #222228; "
            f"color: #666670; border-color: #3a3a40; }}"
        )
        btn.style().unpolish(btn)
        btn.style().polish(btn)
        btn.update()

    def _sync_cal_toolbar_ui(self, snap: dict) -> None:
        if not hasattr(self, "cal_click_banner"):
            return
        t = get_settings().t
        click_on = bool(snap.get("cal_click_bull_mode"))
        ellipse_on = bool(snap.get("cal_click_ellipse_mode"))
        topdown_on = bool(snap.get("cal_show_topdown"))
        click_enabled = not topdown_on
        if click_on:
            self.cal_click_banner.setText(t("click_bull_hint"))
        elif ellipse_on:
            self.cal_click_banner.setText(t("click_ellipse_hint"))
        else:
            self.cal_click_banner.setText("")
        self.cal_click_banner.setVisible(click_on or ellipse_on)
        if hasattr(self, "cal_click_bull_btn"):
            self.cal_click_bull_btn.setText(t("click_bull"))
            self._style_cal_toggle(self.cal_click_bull_btn, click_on, enabled=click_enabled)
        if hasattr(self, "cal_click_ellipse_btn"):
            self.cal_click_ellipse_btn.setText(t("click_ellipse"))
            self._style_cal_toggle(self.cal_click_ellipse_btn, ellipse_on, enabled=click_enabled)
        if hasattr(self, "cal_topdown_btn"):
            self._style_cal_toggle(self.cal_topdown_btn, topdown_on)
        if hasattr(self, "cal_label"):
            self.cal_label.setCursor(
                Qt.CrossCursor if (click_on or ellipse_on) else Qt.ArrowCursor
            )

    def _sync_cal_click_bull_ui(self, snap: dict) -> None:
        self._sync_cal_toolbar_ui(snap)

    def _tutorial_section_label(self, text: str) -> QLabel:
        lab = QLabel(text.upper())
        lab.setObjectName("TutorialSection")
        lab.setWordWrap(True)
        return lab

    def _tutorial_body_label(self, text: str) -> QLabel:
        lab = QLabel(text)
        lab.setObjectName("TutorialBody")
        lab.setWordWrap(True)
        lab.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lab.setTextInteractionFlags(Qt.NoTextInteraction)
        return lab

    def _clear_layout(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
            child = item.layout()
            if child is not None:
                self._clear_layout(child)

    def _fill_tutorial(self, kind: str, snap: Optional[dict] = None) -> None:
        lay = getattr(self, f"tutorial_{kind}_body_lay", None)
        if lay is None:
            return
        self._clear_layout(lay)
        t = get_settings().t
        # Basics (text-only — no icon/example widgets)
        lay.addWidget(self._tutorial_section_label(t("tutorial_basics")))
        if kind == "x01":
            body_txt = self._tutorial_x01_body_text(snap)
        else:
            body_txt = t(f"tutorial_{kind}_body")
        lay.addWidget(self._tutorial_body_label(body_txt))
        # Options
        opts = t(f"tutorial_{kind}_opts")
        if opts and opts != f"tutorial_{kind}_opts":
            lay.addWidget(self._tutorial_section_label(t("tutorial_options")))
            lay.addWidget(self._tutorial_body_label(opts))
        lay.addStretch(1)
        # Keep hidden body in sync for any legacy reads
        body = getattr(self, f"tutorial_{kind}_body", None)
        if body is not None:
            body.setText(body_txt)

    def _tutorial_x01_body_text(self, snap: Optional[dict] = None) -> str:
        """X01 tutorial with the actual selected start score (301 or 501)."""
        mode = ""
        if snap is not None:
            mode = str(snap.get("selected_game_mode") or "")
        elif hasattr(self, "session") and self.session is not None:
            try:
                mode = str(
                    (self.session.snapshot_for_ui() or {}).get("selected_game_mode")
                    or ""
                )
            except Exception:
                mode = ""
        start = mode.strip() if str(mode).strip() in ("301", "501") else "501"
        return get_settings().t("tutorial_x01_body").format(start=start)

    def _clear_score_box(self) -> None:
        self._stop_x01_turn_anim()
        self._stop_cricket_turn_anim()
        self._stop_killer_turn_anim()
        # Stop score countdown/halve anims before deleteLater (deferred destroy).
        for card in list(self._x01_cards_by_pi.values()):
            card.reset_score_display()
        for card in list(self._player_cards):
            if isinstance(card, PlayerCard):
                card.reset_score_display()
        while self.score_box.count():
            item = self.score_box.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._player_cards = []
        self._x01_cards_by_pi = {}
        self._x01_waiting_wrap = None
        self._x01_waiting_layout = None
        self._x01_last_active_pi = None
        self._x01_last_waiting_pis = None
        self._x01_last_finished_pis = None
        self._cricket_rows_by_pi = {}
        self._cricket_hdr = None
        self._cricket_round_lab = None
        self._cricket_last_active_pi = None
        self._killer_rows_by_pi = {}
        self._killer_hdr = None
        self._killer_waiting_wrap = None
        self._killer_waiting_layout = None
        self._killer_last_active_pi = None
        self._killer_finished_pis = set()
        self._hit_tags_prev = ["-", "-", "-"]
        if hasattr(self, "killer_numbers_lay") and self.killer_numbers_lay is not None:
            while self.killer_numbers_lay.count():
                item = self.killer_numbers_lay.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.deleteLater()
        if hasattr(self, "live_stack") and hasattr(self, "live_line"):
            self.live_stack.setCurrentWidget(self.live_line)

    def _teardown_scoreboard_ui(self) -> None:
        """Leave/enter match: drop reused score widgets so leftover scores cannot animate."""
        if (
            self._x01_cards_by_pi
            or self._player_cards
            or self._cricket_rows_by_pi
            or self._killer_rows_by_pi
            or self.score_box.count()
        ):
            self._clear_score_box()

    def _stop_cricket_turn_anim(self) -> None:
        if self._cricket_turn_anim is not None:
            self._cricket_turn_anim.stop()
            self._cricket_turn_anim = None
        for g in self._cricket_turn_ghosts:
            g.deleteLater()
        self._cricket_turn_ghosts = []
        for row in self._cricket_rows_by_pi.values():
            row.setGraphicsEffect(None)
            row.setVisible(True)

    def _stop_x01_turn_anim(self) -> None:
        if self._x01_turn_anim is not None:
            self._x01_turn_anim.stop()
            self._x01_turn_anim = None
        for g in self._x01_turn_ghosts:
            g.deleteLater()
        self._x01_turn_ghosts = []
        for card in self._x01_cards_by_pi.values():
            card.setGraphicsEffect(None)
            card.setVisible(True)

    @staticmethod
    def _x01_hide_card_keep_layout(card: QWidget) -> None:
        """Sakrij vizualno, ali zadrži mjesto u layoutu (za mjerenje end geometrije)."""
        eff = QGraphicsOpacityEffect(card)
        eff.setOpacity(0.0)
        card.setGraphicsEffect(eff)

    @staticmethod
    def _x01_card_geom_in(card: QWidget, host: QWidget) -> QRect:
        top_left = card.mapTo(host, QPoint(0, 0))
        return QRect(top_left, card.size())

    @staticmethod
    def _x01_player_finished(p: dict) -> bool:
        return p.get("rank") is not None or bool(p.get("completed"))

    @staticmethod
    def _turn_change_is_reverse(
        players: list,
        prev_pi: int,
        new_pi: int,
        finished_predicate,
    ) -> bool:
        """True ako new_pi nije forward-next među još-igrajućima nakon prev_pi (undo)."""
        n = len(players)
        if n < 2 or prev_pi == new_pi:
            return False
        prev_i = None
        for i, p in enumerate(players):
            if int(p.get("index", i)) == prev_pi:
                prev_i = i
                break
        if prev_i is None:
            return False
        forward_next = None
        for k in range(1, n + 1):
            p = players[(prev_i + k) % n]
            if finished_predicate(p):
                continue
            forward_next = int(p.get("index", (prev_i + k) % n))
            break
        if forward_next is None:
            return False
        return new_pi != forward_next

    def _x01_ensure_waiting_row(self) -> QHBoxLayout:
        """Donji horizontalni red: waiting lijevo + finished desno."""
        if self._x01_waiting_wrap is None:
            self._x01_waiting_wrap = QWidget()
            self._x01_waiting_wrap.setSizePolicy(
                QSizePolicy.Expanding, QSizePolicy.Fixed
            )
            self._x01_waiting_layout = QHBoxLayout(self._x01_waiting_wrap)
            self._x01_waiting_layout.setContentsMargins(0, 0, 0, 0)
            self._x01_waiting_layout.setSpacing(10)
            self.score_box.addWidget(self._x01_waiting_wrap, 0)
        assert self._x01_waiting_layout is not None
        return self._x01_waiting_layout

    @staticmethod
    def _reuse_or_live_grab(widget, fallback: Optional[QPixmap]) -> Optional[QPixmap]:
        """Grab after relayout so the ghost matches the new card size."""
        if widget is None or widget.width() < 2 or widget.height() < 2:
            return fallback
        pix = widget.grab()
        if pix is None or pix.isNull():
            return fallback
        return pix

    def _x01_add_ghost_slide(
        self,
        group: QParallelAnimationGroup,
        ghosts: List[QWidget],
        host: QWidget,
        pix: QPixmap,
        start: QRect,
        end: QRect,
        *,
        fade_out: bool = False,
        fade_in: bool = False,
        duration: Optional[int] = None,
        curve: QEasingCurve.Type = QEasingCurve.InOutCubic,
        stack_behind: bool = False,
    ) -> None:
        if pix.isNull() or start.width() < 2 or end.width() < 2:
            return
        if duration is None:
            duration = _TURN_ANIM_MS
        # Parent to score host (right_w) — never root under Wallpaper.
        ghost = QLabel(host)
        ghost.setPixmap(pix)
        ghost.setScaledContents(True)
        ghost.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        ghost.setGeometry(start)
        ghost.show()
        if stack_behind:
            ghost.lower()
        else:
            ghost.raise_()
        ghosts.append(ghost)

        geo = QPropertyAnimation(ghost, b"geometry")
        geo.setDuration(duration)
        geo.setStartValue(QRect(start))
        geo.setEndValue(QRect(end))
        geo.setEasingCurve(curve)
        group.addAnimation(geo)

        if fade_out or fade_in:
            eff = QGraphicsOpacityEffect(ghost)
            ghost.setGraphicsEffect(eff)
            op = QPropertyAnimation(eff, b"opacity")
            op.setDuration(duration)
            op.setStartValue(0.0 if fade_in else 1.0)
            op.setEndValue(1.0 if fade_in else 0.0)
            op.setEasingCurve(QEasingCurve.InOutQuad)
            group.addAnimation(op)

    def _ensure_turn_anim_z_order(self, host: QWidget) -> None:
        """Keep turn ghosts/score rows above Wallpaper during anim."""
        wp = getattr(self, "_wallpaper", None)
        if wp is not None:
            wp.lower()
            if hasattr(self, "stack") and self.stack is not None:
                wp.stackUnder(self.stack)
        host.raise_()
        for row in getattr(self, "_cricket_rows_by_pi", {}).values():
            if row is not None:
                row.raise_()
        for g in (
            list(getattr(self, "_cricket_turn_ghosts", []) or [])
            + list(getattr(self, "_x01_turn_ghosts", []) or [])
            + list(getattr(self, "_killer_turn_ghosts", []) or [])
        ):
            if g is not None:
                g.raise_()

    def _x01_play_turn_anim(
        self,
        starts: Dict[int, QRect],
        pixmaps: Dict[int, QPixmap],
        pis: List[int],
        incoming_pi: int,
        outgoing_pi: Optional[int],
        waiting_pis: List[int],
        finished_pis: List[int],
        *,
        reverse: bool = False,
    ) -> None:
        """Horizontalni slide.

        Forward: iduci nestaje lijevo; glavni odlazi desno; novi ulazi slijeva.
        Reverse (undo): smjerovi obrnuti.
        """
        host = getattr(self, "right_w", None)
        if host is None or not starts:
            return
        if host.layout() is not None:
            host.layout().activate()
        host.update()

        ends: Dict[int, QRect] = {}
        for pi in pis:
            card = self._x01_cards_by_pi.get(pi)
            if card is None or card.width() < 2 or card.height() < 2:
                continue
            ends[pi] = self._x01_card_geom_in(card, host)

        for g in self._x01_turn_ghosts:
            g.deleteLater()
        self._x01_turn_ghosts = []
        if self._x01_turn_anim is not None:
            self._x01_turn_anim.stop()
            self._x01_turn_anim = None

        group = QParallelAnimationGroup(self)
        ghosts: List[QWidget] = []
        animated: set = set()
        main_end = ends.get(incoming_pi)
        out_start = starts.get(outgoing_pi) if outgoing_pi is not None else None
        in_wait_start = starts.get(incoming_pi)
        # Forward: out > right, in < left. Reverse: out < left, in > right.
        out_sign = -1 if reverse else 1
        in_sign = 1 if reverse else -1

        # 1) Stari aktivni > finished desno u donjem redu, ili waiting
        if outgoing_pi is not None and out_start is not None:
            out_pix = pixmaps.get(outgoing_pi)
            if out_pix is not None and not out_pix.isNull():
                card = self._x01_cards_by_pi.get(outgoing_pi)
                if card is not None:
                    self._x01_hide_card_keep_layout(card)
                if outgoing_pi in finished_pis and outgoing_pi in ends:
                    # Lock na desnu stranu donjeg reda
                    self._x01_add_ghost_slide(
                        group,
                        ghosts,
                        host,
                        out_pix,
                        out_start,
                        ends[outgoing_pi],
                        curve=QEasingCurve.InOutCubic,
                    )
                else:
                    exit_end = QRect(
                        out_start.x() + out_sign * max(160, out_start.width() // 2),
                        out_start.y(),
                        out_start.width(),
                        out_start.height(),
                    )
                    self._x01_add_ghost_slide(
                        group,
                        ghosts,
                        host,
                        out_pix,
                        out_start,
                        exit_end,
                        fade_out=True,
                        curve=QEasingCurve.InOutCubic,
                    )
                    wait_end = ends.get(outgoing_pi)
                    if wait_end is not None and outgoing_pi in waiting_pis:
                        end_pix = self._reuse_or_live_grab(card, out_pix)
                        if end_pix is None or end_pix.isNull():
                            end_pix = out_pix
                        enter_start = QRect(
                            wait_end.x() + out_sign * max(160, wait_end.width()),
                            wait_end.y(),
                            wait_end.width(),
                            wait_end.height(),
                        )
                        self._x01_add_ghost_slide(
                            group,
                            ghosts,
                            host,
                            end_pix,
                            enter_start,
                            wait_end,
                            fade_in=True,
                            curve=QEasingCurve.OutCubic,
                        )
                animated.add(outgoing_pi)

        # 2) Novi aktivni: waiting > nestane; ulazi u glavnu
        if main_end is not None:
            card = self._x01_cards_by_pi.get(incoming_pi)
            in_pix = self._reuse_or_live_grab(card, pixmaps.get(incoming_pi))
            if in_pix is None or in_pix.isNull():
                in_pix = pixmaps.get(incoming_pi)
            if in_pix is not None and not in_pix.isNull():
                if card is not None:
                    self._x01_hide_card_keep_layout(card)
                if in_wait_start is not None and (
                    out_start is None
                    or in_wait_start.y() > (out_start.y() + out_start.height() // 3)
                ):
                    w_pix = pixmaps.get(incoming_pi)
                    if w_pix is not None and not w_pix.isNull():
                        wait_exit = QRect(
                            in_wait_start.x()
                            + in_sign * max(160, in_wait_start.width()),
                            in_wait_start.y(),
                            in_wait_start.width(),
                            in_wait_start.height(),
                        )
                        self._x01_add_ghost_slide(
                            group,
                            ghosts,
                            host,
                            w_pix,
                            in_wait_start,
                            wait_exit,
                            fade_out=True,
                            curve=QEasingCurve.InOutCubic,
                        )
                enter_start = QRect(
                    main_end.x() + in_sign * (main_end.width() + 48),
                    main_end.y(),
                    main_end.width(),
                    main_end.height(),
                )
                self._x01_add_ghost_slide(
                    group,
                    ghosts,
                    host,
                    in_pix,
                    enter_start,
                    main_end,
                    fade_in=True,
                    curve=QEasingCurve.OutCubic,
                )
                animated.add(incoming_pi)

        # 3) Ostali waiting (ne finished): horizontalni shift
        for pi in waiting_pis:
            if pi in animated:
                continue
            start = starts.get(pi)
            end = ends.get(pi)
            pix = pixmaps.get(pi)
            if start is None or end is None or pix is None or pix.isNull():
                continue
            if start == end:
                continue
            card = self._x01_cards_by_pi.get(pi)
            if card is not None:
                self._x01_hide_card_keep_layout(card)
            self._x01_add_ghost_slide(
                group,
                ghosts,
                host,
                pix,
                start,
                end,
                curve=QEasingCurve.InOutCubic,
            )
            animated.add(pi)

        # Finished locked — bez animacije (osim outgoing>finished gore)
        for pi in pis:
            if pi in animated:
                continue
            card = self._x01_cards_by_pi.get(pi)
            if card is not None:
                card.setGraphicsEffect(None)

        self._x01_turn_ghosts = ghosts

        def _done() -> None:
            for g in list(self._x01_turn_ghosts):
                g.deleteLater()
            self._x01_turn_ghosts = []
            for pi in pis:
                card = self._x01_cards_by_pi.get(pi)
                if card is not None:
                    card.setGraphicsEffect(None)
                    card.setVisible(True)
            self._x01_turn_anim = None

        if group.animationCount() == 0:
            _done()
            return
        group.finished.connect(_done)
        self._x01_turn_anim = group
        group.start()

    @staticmethod
    def _x01_active_metrics() -> tuple:
        """Aktivni igrač — uvijek veličina kao za 1 igrača (2.85× baseline)."""
        scale = 2.85
        icon = int(round(PlayerCard.ICON_H * scale))
        font = int(round(PlayerCard.FONT_PX * scale))
        min_h = int(round(PlayerCard.MIN_H * scale))
        max_h = int(round(PlayerCard.MAX_H * scale))
        return icon, font, min_h, max_h

    @staticmethod
    def _x01_waiting_metrics() -> tuple:
        """Čekajući / finished — skalirano da stanu max 3 horizontalno."""
        scale = 0.92
        icon = int(round(PlayerCard.ICON_H * scale))
        font = int(round(PlayerCard.FONT_PX * 0.72))
        min_h = int(round(PlayerCard.MIN_H * scale))
        max_h = int(round(PlayerCard.MAX_H * scale))
        return icon, font, min_h, max_h

    def _x01_card_for(self, pi: int) -> PlayerCard:
        """Jedna kartica po indexu igrača — score ostaje na igraču pri premještanju."""
        card = self._x01_cards_by_pi.get(pi)
        if card is None:
            card = PlayerCard()
            self._x01_cards_by_pi[pi] = card
        return card

    def _x01_detach_card(self, card: PlayerCard) -> None:
        if self.score_box.indexOf(card) >= 0:
            self.score_box.removeWidget(card)
        if (
            self._x01_waiting_layout is not None
            and self._x01_waiting_layout.indexOf(card) >= 0
        ):
            self._x01_waiting_layout.removeWidget(card)
        card.setParent(None)

    @staticmethod
    def _around_card_progress(p: dict) -> int:
        """Steps completed 0–21 for Around progress bar."""
        if bool(p.get("completed")) or MainWindow._x01_player_finished(p):
            return PlayerCard.AROUND_STEPS
        score = str(p.get("score_text") or "").upper()
        if score in ("DONE",) or score.startswith("#"):
            return PlayerCard.AROUND_STEPS
        idx = p.get("next_target_idx")
        if idx is None:
            return 0
        return max(0, min(PlayerCard.AROUND_STEPS, int(idx)))

    def _sync_x01_player_cards(
        self, players: list, *, reverse: bool = False, around_mode: bool = False
    ) -> None:
        """Aktivni gore (puna širina); dolje waiting lijevo + finished desno (lock)."""
        if not players:
            self._clear_score_box()
            return

        n = len(players)
        active_i = next((i for i, p in enumerate(players) if p.get("active")), 0)
        active_p = players[active_i]
        active_pi = int(active_p.get("index", active_i))

        waiting_display = []
        for k in range(1, n):
            p = players[(active_i + k) % n]
            if self._x01_player_finished(p):
                continue
            if bool(p.get("active")):
                continue
            waiting_display.append(p)
            if len(waiting_display) >= 3:
                break

        finished_display = [p for p in players if self._x01_player_finished(p)]
        finished_display.sort(key=lambda p: int(p.get("rank") or 99))

        self.score_box.setSpacing(14)
        self.score_box.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        a_icon, a_font, a_min, a_max = self._x01_active_metrics()
        w_icon, w_font, w_min, w_max = self._x01_waiting_metrics()

        keep_pis = {int(p.get("index", i)) for i, p in enumerate(players)}
        for pi in list(self._x01_cards_by_pi.keys()):
            if pi not in keep_pis:
                card = self._x01_cards_by_pi.pop(pi)
                self._x01_detach_card(card)
                card.deleteLater()

        prev_active = self._x01_last_active_pi
        will_anim = (
            prev_active is not None
            and prev_active != active_pi
            and n > 1
            and prev_active in self._x01_cards_by_pi
        )
        if will_anim and not reverse:
            reverse = self._turn_change_is_reverse(
                players, prev_active, active_pi, self._x01_player_finished
            )
        starts: Dict[int, QRect] = {}
        pixmaps: Dict[int, QPixmap] = {}
        host = getattr(self, "right_w", None)
        if will_anim and host is not None:
            self._stop_x01_turn_anim()
            for pi, card in self._x01_cards_by_pi.items():
                if card.width() < 2 or card.height() < 2:
                    continue
                starts[pi] = self._x01_card_geom_in(card, host)
                pixmaps[pi] = card.grab()
                self._x01_hide_card_keep_layout(card)

        waiting_pis_expected = [
            int(p.get("index", -1)) for p in waiting_display if int(p.get("index", -1)) >= 0
        ]
        finished_pis_expected = [
            int(p.get("index", -1)) for p in finished_display if int(p.get("index", -1)) >= 0
        ]
        # Mid-turn score updates: keep layout, only refresh card contents.
        structure_stable = (
            not will_anim
            and prev_active == active_pi
            and prev_active is not None
            and set(self._x01_cards_by_pi.keys()) == keep_pis
            and getattr(self, "_x01_last_waiting_pis", None) == waiting_pis_expected
            and getattr(self, "_x01_last_finished_pis", None) == finished_pis_expected
            and self.score_box.indexOf(self._x01_cards_by_pi.get(active_pi)) == 0
        )
        if structure_stable:
            active_card = self._x01_cards_by_pi[active_pi]
            active_card.set_score_anim_enabled(not around_mode)
            if around_mode:
                active_card.set_around_progress(
                    self._around_card_progress(active_p), active_p.get("color", "#fff")
                )
            else:
                active_card.set_around_progress(None, active_p.get("color", "#fff"))
            active_card.set_metrics(a_icon, a_font, a_min, a_max)
            active_card.apply_compact_width(False)
            active_card.set_player(
                active_p["label"],
                str(active_p["score_text"]),
                active_p.get("color", "#fff"),
                True,
                player_index=active_pi,
                finished=False,
                score_halved=bool(active_p.get("score_halved")),
                name=str(active_p.get("name") or ""),
            )
            for p in waiting_display:
                pi = int(p.get("index", -1))
                if pi < 0:
                    continue
                card = self._x01_cards_by_pi.get(pi)
                if card is None:
                    continue
                card.set_score_anim_enabled(not around_mode)
                if around_mode:
                    card.set_around_progress(
                        self._around_card_progress(p), p.get("color", "#fff")
                    )
                else:
                    card.set_around_progress(None, p.get("color", "#fff"))
                card.set_metrics(w_icon, w_font, w_min, w_max)
                card.apply_compact_width(False)
                card.set_player(
                    p["label"],
                    str(p["score_text"]),
                    p.get("color", "#fff"),
                    False,
                    player_index=pi,
                    finished=False,
                    score_halved=bool(p.get("score_halved")),
                    name=str(p.get("name") or ""),
                )
            for p in finished_display:
                pi = int(p.get("index", -1))
                if pi < 0:
                    continue
                card = self._x01_cards_by_pi.get(pi)
                if card is None:
                    continue
                card.set_score_anim_enabled(not around_mode)
                if around_mode:
                    card.set_around_progress(
                        self._around_card_progress(p), p.get("color", "#fff")
                    )
                else:
                    card.set_around_progress(None, p.get("color", "#fff"))
                card.set_metrics(w_icon, w_font, w_min, w_max)
                card.apply_compact_width(True, sample="#1")
                card.set_player(
                    p["label"],
                    str(p["score_text"]),
                    p.get("color", "#fff"),
                    False,
                    player_index=pi,
                    finished=True,
                    score_halved=bool(p.get("score_halved")),
                    name=str(p.get("name") or ""),
                )
            self._player_cards = [
                self._x01_cards_by_pi[pi] for pi in sorted(self._x01_cards_by_pi.keys())
            ]
            self._x01_last_active_pi = active_pi
            self._x01_last_waiting_pis = waiting_pis_expected
            self._x01_last_finished_pis = finished_pis_expected
            return

        # Makni sve kartice iz layouta (score_box + waiting)
        for card in list(self._x01_cards_by_pi.values()):
            self._x01_detach_card(card)
        if self._x01_waiting_layout is not None:
            while self._x01_waiting_layout.count():
                item = self._x01_waiting_layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.setParent(None)

        # Aktivni gore — uvijek puna širina scoreboarda (ne dira ostatak ekrana)
        active_card = self._x01_card_for(active_pi)
        self._x01_detach_card(active_card)
        self.score_box.insertWidget(0, active_card, 0)
        active_card.set_score_anim_enabled(not around_mode)
        if around_mode:
            active_card.set_around_progress(
                self._around_card_progress(active_p), active_p.get("color", "#fff")
            )
        else:
            active_card.set_around_progress(None, active_p.get("color", "#fff"))
        active_card.set_metrics(a_icon, a_font, a_min, a_max)
        active_card.apply_compact_width(False)
        active_card.set_player(
            active_p["label"],
            str(active_p["score_text"]),
            active_p.get("color", "#fff"),
            True,
            player_index=active_pi,
            finished=False,
            score_halved=bool(active_p.get("score_halved")),
            name=str(active_p.get("name") or ""),
        )

        # Donji red: waiting (cirkulacija) | stretch | finished (lock desno)
        waiting_pis: List[int] = []
        finished_pis: List[int] = []
        row_n = len(waiting_display) + len(finished_display)
        if row_n == 0:
            if self._x01_waiting_wrap is not None:
                self._x01_waiting_wrap.hide()
        else:
            wait_lay = self._x01_ensure_waiting_row()
            # Osiguraj da je waiting ispod aktivnog
            if self.score_box.indexOf(self._x01_waiting_wrap) < 0:
                self.score_box.addWidget(self._x01_waiting_wrap, 0)
            elif self.score_box.indexOf(active_card) == 0:
                # waiting već u boxu
                pass
            self._x01_waiting_wrap.show()

            for p in waiting_display:
                pi = int(p.get("index", -1))
                if pi < 0:
                    continue
                waiting_pis.append(pi)
                card = self._x01_card_for(pi)
                self._x01_detach_card(card)
                wait_lay.addWidget(card, 1)
                card.set_score_anim_enabled(not around_mode)
                if around_mode:
                    card.set_around_progress(
                        self._around_card_progress(p), p.get("color", "#fff")
                    )
                else:
                    card.set_around_progress(None, p.get("color", "#fff"))
                card.set_metrics(w_icon, w_font, w_min, w_max)
                card.apply_compact_width(False)
                card.set_player(
                    p["label"],
                    str(p["score_text"]),
                    p.get("color", "#fff"),
                    False,
                    player_index=pi,
                    finished=False,
                    score_halved=bool(p.get("score_halved")),
                    name=str(p.get("name") or ""),
                )

            if finished_display:
                wait_lay.addStretch(1)

            for p in finished_display:
                pi = int(p.get("index", -1))
                if pi < 0:
                    continue
                finished_pis.append(pi)
                card = self._x01_card_for(pi)
                self._x01_detach_card(card)
                wait_lay.addWidget(card, 0)
                card.set_score_anim_enabled(not around_mode)
                if around_mode:
                    card.set_around_progress(
                        self._around_card_progress(p), p.get("color", "#fff")
                    )
                else:
                    card.set_around_progress(None, p.get("color", "#fff"))
                card.set_metrics(w_icon, w_font, w_min, w_max)
                card.apply_compact_width(True, sample="#1")
                card.set_player(
                    p["label"],
                    str(p["score_text"]),
                    p.get("color", "#fff"),
                    False,
                    player_index=pi,
                    finished=True,
                    score_halved=bool(p.get("score_halved")),
                    name=str(p.get("name") or ""),
                )

        self._player_cards = [
            self._x01_cards_by_pi[pi] for pi in sorted(self._x01_cards_by_pi.keys())
        ]
        self._x01_last_active_pi = active_pi
        self._x01_last_waiting_pis = list(waiting_pis)
        self._x01_last_finished_pis = list(finished_pis)

        if will_anim and starts and host is not None:
            pis = list(keep_pis)
            out_pi = prev_active
            in_pi = active_pi
            rev = reverse

            def _go() -> None:
                self._x01_play_turn_anim(
                    starts,
                    pixmaps,
                    pis,
                    in_pi,
                    out_pi,
                    waiting_pis,
                    finished_pis,
                    reverse=rev,
                )

            QTimer.singleShot(0, _go)

    def _cricket_row_for(self, pi: int) -> CricketRow:
        row = self._cricket_rows_by_pi.get(pi)
        if row is None:
            row = CricketRow()
            self._cricket_rows_by_pi[pi] = row
        return row

    def _cricket_ensure_header(self, *, show_pts: bool, snap: dict, rules: dict) -> None:
        if rules.get("max_rounds_20"):
            lim = int(rules.get("max_rounds_limit", 20))
            txt = f"{get_settings().t('round')}: {snap.get('round_number', 1)}/{lim}"
            if self._cricket_round_lab is None:
                self._cricket_round_lab = QLabel(txt)
                self._cricket_round_lab.setObjectName("RoundLabel")
                self._cricket_round_lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.score_box.insertWidget(0, self._cricket_round_lab)
            else:
                self._cricket_round_lab.setText(txt)
                if self.score_box.indexOf(self._cricket_round_lab) < 0:
                    self.score_box.insertWidget(0, self._cricket_round_lab)
        elif self._cricket_round_lab is not None:
            self.score_box.removeWidget(self._cricket_round_lab)
            self._cricket_round_lab.deleteLater()
            self._cricket_round_lab = None

        if self._cricket_hdr is None:
            hdr_row = QHBoxLayout()
            hdr_row.setContentsMargins(6, 0, 8, 0)
            hdr_row.setSpacing(0)
            spacer = QLabel("")
            spacer.setFixedWidth(
                CricketRow.ICON_H + 2 * CricketRow.ICON_BORDER
            )
            hdr_row.addWidget(spacer)
            for lab in ("15", "16", "17", "18", "19", "20", "B"):
                hdr_row.addStretch(1)
                h = QLabel(lab)
                h.setAlignment(Qt.AlignCenter)
                h.setFixedWidth(MarkStack.COL_W)
                h.setStyleSheet(
                    "font-size: 40px; font-weight: 800; color: #d0d0d8; padding-right: 4px;"
                )
                hdr_row.addWidget(h, 0)
            hdr_row.addStretch(1)
            pts_h = QLabel("PTS" if show_pts else "")
            pts_h.setObjectName("CricketPtsHdr")
            pts_h.setFixedWidth(CricketRow.PTS_W)
            pts_h.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            pts_h.setStyleSheet(
                "font-size: 40px; font-weight: 700; color: #a0a0a8; padding-right: 4px;"
            )
            hdr_row.addWidget(pts_h)
            self._cricket_hdr = QWidget()
            self._cricket_hdr.setLayout(hdr_row)
            self._cricket_hdr.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            insert_at = 1 if self._cricket_round_lab is not None else 0
            self.score_box.insertWidget(insert_at, self._cricket_hdr)
        else:
            pts_h = self._cricket_hdr.findChild(QLabel, "CricketPtsHdr")
            if pts_h is not None:
                pts_h.setText("PTS" if show_pts else "")
            if self.score_box.indexOf(self._cricket_hdr) < 0:
                insert_at = 1 if self._cricket_round_lab is not None else 0
                self.score_box.insertWidget(insert_at, self._cricket_hdr)

    def _cricket_play_turn_anim(
        self,
        starts: Dict[int, QRect],
        pixmaps: Dict[int, QPixmap],
        order_pis: List[int],
        incoming_pi: int,
        outgoing_pi: Optional[int],
        *,
        reverse: bool = False,
    ) -> None:
        """Turn swap: novi top ulazi horizontalno; stari izlazi horizontalno;
        svi ostali (uklj. 2. red) samo vertikalno.

        Forward: outgoing > desno (fade); incoming < slijeva u top; waiting shift gore.
        Reverse (undo): horizontalni smjerovi obrnuti.
        """
        host = getattr(self, "right_w", None)
        if host is None or not starts:
            return
        if host.layout() is not None:
            host.layout().activate()
        host.update()
        self._ensure_turn_anim_z_order(host)

        ends: Dict[int, QRect] = {}
        for pi in order_pis:
            row = self._cricket_rows_by_pi.get(pi)
            if row is None or row.width() < 2:
                continue
            ends[pi] = self._x01_card_geom_in(row, host)

        for g in self._cricket_turn_ghosts:
            g.deleteLater()
        self._cricket_turn_ghosts = []
        if self._cricket_turn_anim is not None:
            self._cricket_turn_anim.stop()
            self._cricket_turn_anim = None

        group = QParallelAnimationGroup(self)
        ghosts: List[QWidget] = []
        animated: set = set()
        main_end = ends.get(incoming_pi)
        out_start = starts.get(outgoing_pi) if outgoing_pi is not None else None
        in_wait_start = starts.get(incoming_pi)
        # Forward: out > right, in < left. Reverse: flipped.
        out_sign = -1 if reverse else 1
        in_sign = 1 if reverse else -1
        dur = 400

        def _grab_end_pix(pi: int, fallback: Optional[QPixmap]) -> Optional[QPixmap]:
            """Grab after layout update; clear opacity hide so pixmap isn't empty."""
            row = self._cricket_rows_by_pi.get(pi)
            if row is None or row.width() < 2:
                return fallback
            row.setGraphicsEffect(None)
            pix = self._reuse_or_live_grab(row, fallback) or fallback
            self._x01_hide_card_keep_layout(row)
            if pix is None or pix.isNull():
                return fallback
            return pix

        def _vertical_end(start: QRect, end: QRect) -> QRect:
            """Same X/width as start — only Y (and end height) may change."""
            return QRect(start.x(), end.y(), start.width(), end.height())

        # 1) Stari aktivni: SAMO horizontalni izlaz s vrha.
        #    Ulazak u waiting slot = vertikalno (ili fade na mjestu) — nikad sideways.
        if outgoing_pi is not None and out_start is not None:
            out_pix = pixmaps.get(outgoing_pi)
            if out_pix is not None and not out_pix.isNull():
                row = self._cricket_rows_by_pi.get(outgoing_pi)
                if row is not None:
                    self._x01_hide_card_keep_layout(row)
                exit_end = QRect(
                    out_start.x() + out_sign * max(160, out_start.width() // 2),
                    out_start.y(),
                    out_start.width(),
                    out_start.height(),
                )
                self._x01_add_ghost_slide(
                    group,
                    ghosts,
                    host,
                    out_pix,
                    out_start,
                    exit_end,
                    fade_out=True,
                    duration=dur,
                    curve=QEasingCurve.InOutCubic,
                )
                wait_end = ends.get(outgoing_pi)
                if wait_end is not None and outgoing_pi != incoming_pi:
                    end_pix = _grab_end_pix(outgoing_pi, out_pix)
                    if end_pix is not None and not end_pix.isNull():
                        # Vertical settle into wait slot (X locked to wait_end).
                        vert_start = QRect(
                            wait_end.x(),
                            out_start.y(),
                            wait_end.width(),
                            wait_end.height(),
                        )
                        if abs(vert_start.y() - wait_end.y()) > 2:
                            self._x01_add_ghost_slide(
                                group,
                                ghosts,
                                host,
                                end_pix,
                                vert_start,
                                wait_end,
                                fade_in=True,
                                duration=dur,
                                curve=QEasingCurve.OutCubic,
                            )
                        else:
                            self._x01_add_ghost_slide(
                                group,
                                ghosts,
                                host,
                                end_pix,
                                wait_end,
                                wait_end,
                                fade_in=True,
                                duration=dur,
                                curve=QEasingCurve.OutCubic,
                            )
                animated.add(outgoing_pi)

        # 2) Novi aktivni: horizontalni izlaz iz wait (samo incoming) + ulaz u top.
        if main_end is not None:
            in_pix = _grab_end_pix(incoming_pi, pixmaps.get(incoming_pi))
            if in_pix is not None and not in_pix.isNull():
                row = self._cricket_rows_by_pi.get(incoming_pi)
                if row is not None:
                    self._x01_hide_card_keep_layout(row)
                if in_wait_start is not None and (
                    out_start is None
                    or in_wait_start.y() > (out_start.y() + out_start.height() // 3)
                ):
                    w_pix = pixmaps.get(incoming_pi)
                    if w_pix is not None and not w_pix.isNull():
                        wait_exit = QRect(
                            in_wait_start.x()
                            + in_sign * max(160, in_wait_start.width()),
                            in_wait_start.y(),
                            in_wait_start.width(),
                            in_wait_start.height(),
                        )
                        self._x01_add_ghost_slide(
                            group,
                            ghosts,
                            host,
                            w_pix,
                            in_wait_start,
                            wait_exit,
                            fade_out=True,
                            duration=dur,
                            curve=QEasingCurve.InOutCubic,
                        )
                enter_start = QRect(
                    main_end.x() + in_sign * (main_end.width() + 48),
                    main_end.y(),
                    main_end.width(),
                    main_end.height(),
                )
                self._x01_add_ghost_slide(
                    group,
                    ghosts,
                    host,
                    in_pix,
                    enter_start,
                    main_end,
                    fade_in=True,
                    duration=dur,
                    curve=QEasingCurve.OutCubic,
                )
                animated.add(incoming_pi)

        # 3) Ostali waiting: SAMO vertikalni shift (nikad sideways).
        for pi in order_pis:
            if pi in animated:
                continue
            start = starts.get(pi)
            end = ends.get(pi)
            pix = pixmaps.get(pi)
            if start is None or end is None or pix is None or pix.isNull():
                continue
            end_v = _vertical_end(start, end)
            if start == end_v:
                continue
            row = self._cricket_rows_by_pi.get(pi)
            if row is not None:
                self._x01_hide_card_keep_layout(row)
            self._x01_add_ghost_slide(
                group,
                ghosts,
                host,
                pix,
                start,
                end_v,
                duration=dur,
                curve=QEasingCurve.InOutCubic,
            )
            animated.add(pi)

        for pi in order_pis:
            if pi in animated:
                continue
            row = self._cricket_rows_by_pi.get(pi)
            if row is not None:
                row.setGraphicsEffect(None)

        self._cricket_turn_ghosts = ghosts
        self._ensure_turn_anim_z_order(host)
        for g in ghosts:
            g.raise_()

        def _done() -> None:
            for g in list(self._cricket_turn_ghosts):
                g.deleteLater()
            self._cricket_turn_ghosts = []
            # Restore every row — avoid stuck invisible/duplicated cards.
            for row in self._cricket_rows_by_pi.values():
                row.setGraphicsEffect(None)
                row.setVisible(True)
            self._cricket_turn_anim = None

        if group.animationCount() == 0:
            _done()
            return
        group.finished.connect(_done)
        self._cricket_turn_anim = group
        group.start()

    def _sync_cricket_rows(self, players: list, rules: dict, snap: dict) -> None:
        """Stable player-order cricket rows; animate active border on turn change."""
        # Očisti X01 / winner podium ostatke — inače POBJEDNIK ostane iznad cricketa
        foreign = False
        if self._x01_cards_by_pi or self._x01_waiting_wrap is not None:
            foreign = True
        elif self._cricket_hdr is None and self.score_box.count() > 0:
            foreign = True
        else:
            known = set(self._cricket_rows_by_pi.values())
            if self._cricket_hdr is not None:
                known.add(self._cricket_hdr)
            if self._cricket_round_lab is not None:
                known.add(self._cricket_round_lab)
            for i in range(self.score_box.count()):
                w = self.score_box.itemAt(i).widget()
                if w is not None and w not in known:
                    foreign = True
                    break
        if foreign:
            self._clear_score_box()

        self._stop_cricket_turn_anim()
        self.score_box.setSpacing(6)
        show_pts = str(rules.get("cricket_mode", "standard")).lower() != "no_score"
        self._cricket_ensure_header(show_pts=show_pts, snap=snap, rules=rules)

        n = len(players)
        if n == 0:
            return
        active_i = next((i for i, p in enumerate(players) if p.get("active")), 0)
        active_pi = int(players[active_i].get("index", active_i))
        prev_active = self._cricket_last_active_pi
        active_changed = (
            prev_active is not None
            and int(prev_active) != int(active_pi)
            and not bool(snap.get("turn_anim_reverse"))
        )
        # Undo: also animate border (incoming = prev, outgoing = current).
        undo_border = (
            prev_active is not None
            and int(prev_active) != int(active_pi)
            and bool(snap.get("turn_anim_reverse"))
        )
        border_anim = active_changed or undo_border

        # Fixed order by player index — do not rotate active to top.
        order = sorted(players, key=lambda p: int(p.get("index", 0)))
        order_pis = [int(p.get("index", -1)) for p in order if int(p.get("index", -1)) >= 0]

        keep = set(order_pis)
        for pi in list(self._cricket_rows_by_pi.keys()):
            if pi not in keep:
                row = self._cricket_rows_by_pi.pop(pi)
                if self.score_box.indexOf(row) >= 0:
                    self.score_box.removeWidget(row)
                row.deleteLater()

        # Avoid remove/re-add thrash when player order is already correct (mid-turn updates).
        hdr_offset = 0
        if self._cricket_round_lab is not None and self.score_box.indexOf(self._cricket_round_lab) >= 0:
            hdr_offset += 1
        if self._cricket_hdr is not None and self.score_box.indexOf(self._cricket_hdr) >= 0:
            hdr_offset += 1
        need_repack = self.score_box.count() != hdr_offset + len(order_pis)
        if not need_repack:
            for i, pi in enumerate(order_pis):
                item = self.score_box.itemAt(hdr_offset + i)
                w = item.widget() if item is not None else None
                if w is not self._cricket_rows_by_pi.get(pi):
                    need_repack = True
                    break

        if need_repack:
            for pi in order_pis:
                row = self._cricket_row_for(pi)
                if self.score_box.indexOf(row) >= 0:
                    self.score_box.removeWidget(row)

        for p in order:
            pi = int(p.get("index", -1))
            if pi < 0:
                continue
            row = self._cricket_row_for(pi)
            rank = p.get("rank")
            finished_p = rank is not None
            name = f"#{rank}" if rank else p["label"]
            is_active = bool(p.get("active")) and not finished_p
            anim_border = None
            if border_anim and not finished_p:
                if pi == active_pi:
                    anim_border = True
                elif prev_active is not None and pi == int(prev_active):
                    anim_border = False
            row.set_row(
                name,
                p.get("marks") or {},
                int(p.get("points", 0)),
                p.get("color", "#fff"),
                is_active,
                show_pts=show_pts,
                finished=finished_p,
                rank=int(rank) if rank is not None else None,
                animate_border=anim_border,
                name=str(p.get("name") or ""),
            )
            row.setGraphicsEffect(None)
            row.setVisible(True)
            if need_repack:
                self.score_box.addWidget(row)

        self._cricket_last_active_pi = active_pi
        _ = snap

    def _killer_row_for(self, pi: int) -> KillerRow:
        row = self._killer_rows_by_pi.get(pi)
        if row is None:
            row = KillerRow()
            self._killer_rows_by_pi[pi] = row
        return row

    def _killer_detach_row(self, row: KillerRow) -> None:
        if self.score_box.indexOf(row) >= 0:
            self.score_box.removeWidget(row)
        if (
            self._killer_waiting_layout is not None
            and self._killer_waiting_layout.indexOf(row) >= 0
        ):
            self._killer_waiting_layout.removeWidget(row)
        row.setParent(None)

    def _killer_align_waiting_row(self, wait_lay: QHBoxLayout) -> None:
        """Živi zbijeni lijevo, mrtvi zbijeni desno."""
        for i in range(wait_lay.count() - 1, -1, -1):
            item = wait_lay.itemAt(i)
            if item is None:
                continue
            if item.spacerItem() is not None:
                wait_lay.takeAt(i)
            else:
                wait_lay.setStretch(i, 0)
        insert_at = wait_lay.count()
        for i in range(wait_lay.count()):
            it = wait_lay.itemAt(i)
            w = it.widget() if it is not None else None
            if isinstance(w, KillerRow) and getattr(w, "_compact", False):
                insert_at = i
                break
        wait_lay.insertStretch(insert_at, 1)

    def _killer_lock_waiting_row_height(self, h: int) -> None:
        """Fiksna visina reda — smrt ne smije micati tipkovnicu / footer."""
        wrap = getattr(self, "_killer_waiting_wrap", None)
        if wrap is None:
            return
        hh = max(1, int(h))
        wrap.setMinimumHeight(hh)
        wrap.setMaximumHeight(hh)

    def _killer_ensure_waiting_row(self) -> QHBoxLayout:
        """Donji horizontalni red: waiting lijevo + finished (prekrižena ikona) desno."""
        if self._killer_waiting_wrap is None:
            self._killer_waiting_wrap = QWidget()
            self._killer_waiting_wrap.setSizePolicy(
                QSizePolicy.Expanding, QSizePolicy.Fixed
            )
            self._killer_waiting_layout = QHBoxLayout(self._killer_waiting_wrap)
            self._killer_waiting_layout.setContentsMargins(0, 0, 0, 0)
            self._killer_waiting_layout.setSpacing(8)
            self.score_box.addWidget(self._killer_waiting_wrap, 0)
        assert self._killer_waiting_layout is not None
        return self._killer_waiting_layout

    @staticmethod
    def _killer_card_height(icon: int, scale: float) -> tuple:
        """Height for one row: initials | number | lives."""
        pad = max(6, int(round(8 * scale)))
        body = icon + 2 * KillerRow.ICON_BORDER
        min_h = pad * 2 + body
        max_h = min_h + max(8, int(round(10 * scale)))
        return min_h, max_h

    @staticmethod
    def _killer_waiting_card_height(icon: int, font: int, scale: float) -> tuple:
        """Waiting: initials+lives on top, large number below."""
        pad = max(6, int(round(8 * scale)))
        gap = max(4, int(round(6 * scale)))
        head_pad = 14
        top = icon + 2 * KillerRow.ICON_BORDER + head_pad
        min_h = pad * 2 + top + gap + int(font)
        max_h = min_h + max(6, int(round(8 * scale)))
        return min_h, max_h

    @classmethod
    def _killer_active_metrics(cls) -> tuple:
        """Aktivni Killer — stane u fiksni kiosk layout."""
        scale = 2.15
        icon = int(round(KillerRow.ICON_H * scale))
        font = int(round(KillerRow.FONT_PX * scale))
        min_h, max_h = cls._killer_card_height(icon, scale)
        return icon, font, min_h, max_h

    @classmethod
    def _killer_waiting_metrics(cls) -> tuple:
        """Tri protivničke kartice = širina glavne; stane u fiksni kiosk layout."""
        scale = 1.22
        icon = int(round(KillerRow.ICON_H * scale))
        font = int(round(KillerRow.FONT_PX * 1.12))
        min_h, max_h = cls._killer_waiting_card_height(icon, font, scale)
        return icon, font, min_h, max_h

    @classmethod
    def _killer_dead_metrics(cls, w_icon: int, w_font: int) -> tuple:
        """Veći inicijali za mrtve — X ostaje čitljiv preko slova; visina <= waiting."""
        d_icon = int(round(w_icon * 1.42))
        f_pad = max(6, int(round(8 * 0.92)))
        f_min = f_pad * 2 + d_icon + 2 * KillerRow.ICON_BORDER
        f_max = f_min + 8
        return d_icon, w_font, f_min, f_max

    @staticmethod
    def _killer_player_finished(p: dict) -> bool:
        # Dead = eliminated (0 lives). Winner keeps lives > 0 and must NOT get
        # strike-X / finished chrome even when placements assign them rank #1.
        return int(p.get("lives") or 0) <= 0

    def _stop_killer_turn_anim(self) -> None:
        if self._killer_turn_anim is not None:
            self._killer_turn_anim.stop()
            self._killer_turn_anim = None
        for g in self._killer_turn_ghosts:
            g.deleteLater()
        self._killer_turn_ghosts = []
        for row in self._killer_rows_by_pi.values():
            row.setGraphicsEffect(None)
            row.setVisible(True)

    def _killer_play_turn_anim(
        self,
        starts: Dict[int, QRect],
        pixmaps: Dict[int, QPixmap],
        pis: List[int],
        incoming_pi: int,
        outgoing_pi: Optional[int],
        waiting_pis: List[int],
        finished_pis: List[int],
        *,
        reverse: bool = False,
        death_pis: Optional[set] = None,
    ) -> None:
        """Horizontalni slide kao X01 (aktivni gore, waiting/finished dolje)."""
        host = getattr(self, "right_w", None)
        if host is None or not starts:
            return
        if host.layout() is not None:
            host.layout().activate()
        host.update()

        ends: Dict[int, QRect] = {}
        for pi in pis:
            row = self._killer_rows_by_pi.get(pi)
            if row is None or row.width() < 2 or row.height() < 2:
                continue
            ends[pi] = self._x01_card_geom_in(row, host)

        for g in self._killer_turn_ghosts:
            g.deleteLater()
        self._killer_turn_ghosts = []
        if self._killer_turn_anim is not None:
            self._killer_turn_anim.stop()
            self._killer_turn_anim = None

        group = QParallelAnimationGroup(self)
        ghosts: List[QWidget] = []
        animated: set = set()
        death_pis = set(death_pis or ())
        main_end = ends.get(incoming_pi)
        out_start = starts.get(outgoing_pi) if outgoing_pi is not None else None
        in_wait_start = starts.get(incoming_pi)
        out_sign = -1 if reverse else 1
        in_sign = 1 if reverse else -1

        if outgoing_pi is not None and out_start is not None:
            out_pix = pixmaps.get(outgoing_pi)
            if out_pix is not None and not out_pix.isNull():
                row = self._killer_rows_by_pi.get(outgoing_pi)
                if row is not None:
                    row.setGraphicsEffect(None)
                end_pix = self._reuse_or_live_grab(row, out_pix)
                if end_pix is None or end_pix.isNull():
                    end_pix = out_pix
                if row is not None:
                    self._x01_hide_card_keep_layout(row)
                if outgoing_pi in finished_pis and outgoing_pi in ends:
                    # Dying / finished: slide to compact dead slot (no teleport).
                    self._x01_add_ghost_slide(
                        group,
                        ghosts,
                        host,
                        out_pix,
                        out_start,
                        ends[outgoing_pi],
                        fade_out=True,
                        curve=QEasingCurve.InOutCubic,
                        duration=480,
                    )
                    self._x01_add_ghost_slide(
                        group,
                        ghosts,
                        host,
                        end_pix,
                        out_start,
                        ends[outgoing_pi],
                        fade_in=True,
                        curve=QEasingCurve.InOutCubic,
                        duration=480,
                    )
                else:
                    exit_end = QRect(
                        out_start.x() + out_sign * max(160, out_start.width() // 2),
                        out_start.y(),
                        out_start.width(),
                        out_start.height(),
                    )
                    self._x01_add_ghost_slide(
                        group,
                        ghosts,
                        host,
                        out_pix,
                        out_start,
                        exit_end,
                        fade_out=True,
                        curve=QEasingCurve.InOutCubic,
                    )
                    wait_end = ends.get(outgoing_pi)
                    if wait_end is not None and outgoing_pi in waiting_pis:
                        enter_start = QRect(
                            wait_end.x() + out_sign * max(160, wait_end.width()),
                            wait_end.y(),
                            wait_end.width(),
                            wait_end.height(),
                        )
                        self._x01_add_ghost_slide(
                            group,
                            ghosts,
                            host,
                            end_pix,
                            enter_start,
                            wait_end,
                            fade_in=True,
                            curve=QEasingCurve.OutCubic,
                        )
                animated.add(outgoing_pi)

        if main_end is not None:
            row = self._killer_rows_by_pi.get(incoming_pi)
            if row is not None:
                row.setGraphicsEffect(None)
            in_pix = self._reuse_or_live_grab(row, pixmaps.get(incoming_pi))
            if in_pix is None or in_pix.isNull():
                in_pix = pixmaps.get(incoming_pi)
            if in_pix is not None and not in_pix.isNull():
                if row is not None:
                    self._x01_hide_card_keep_layout(row)
                if in_wait_start is not None and (
                    out_start is None
                    or in_wait_start.y() > (out_start.y() + out_start.height() // 3)
                ):
                    w_pix = pixmaps.get(incoming_pi)
                    if w_pix is not None and not w_pix.isNull():
                        wait_exit = QRect(
                            in_wait_start.x()
                            + in_sign * max(160, in_wait_start.width()),
                            in_wait_start.y(),
                            in_wait_start.width(),
                            in_wait_start.height(),
                        )
                        self._x01_add_ghost_slide(
                            group,
                            ghosts,
                            host,
                            w_pix,
                            in_wait_start,
                            wait_exit,
                            fade_out=True,
                            curve=QEasingCurve.InOutCubic,
                        )
                enter_start = QRect(
                    main_end.x() + in_sign * (main_end.width() + 48),
                    main_end.y(),
                    main_end.width(),
                    main_end.height(),
                )
                self._x01_add_ghost_slide(
                    group,
                    ghosts,
                    host,
                    in_pix,
                    enter_start,
                    main_end,
                    fade_in=True,
                    curve=QEasingCurve.OutCubic,
                )
                animated.add(incoming_pi)

        for pi in waiting_pis:
            if pi in animated:
                continue
            start = starts.get(pi)
            end = ends.get(pi)
            pix = pixmaps.get(pi)
            if start is None or end is None or pix is None or pix.isNull():
                continue
            # Living waiters stay fixed-size; only allow a horizontal shift ghost.
            if start.width() != end.width() or start.height() != end.height():
                end = QRect(end.x(), end.y(), start.width(), start.height())
            if start == end:
                continue
            row = self._killer_rows_by_pi.get(pi)
            if row is not None:
                self._x01_hide_card_keep_layout(row)
            self._x01_add_ghost_slide(
                group,
                ghosts,
                host,
                pix,
                start,
                end,
                curve=QEasingCurve.InOutCubic,
            )
            animated.add(pi)

        # Extra deaths this frame (e.g. waiting player killed) — slide to finished.
        for pi in death_pis:
            if pi in animated:
                continue
            start = starts.get(pi)
            end = ends.get(pi)
            pix = pixmaps.get(pi)
            if start is None or end is None or pix is None or pix.isNull():
                continue
            row = self._killer_rows_by_pi.get(pi)
            if row is not None:
                row.setGraphicsEffect(None)
            end_pix = self._reuse_or_live_grab(row, pix)
            if end_pix is None or end_pix.isNull():
                end_pix = pix
            if row is not None:
                self._x01_hide_card_keep_layout(row)
            self._x01_add_ghost_slide(
                group,
                ghosts,
                host,
                pix,
                start,
                end,
                fade_out=True,
                curve=QEasingCurve.InOutCubic,
                duration=480,
            )
            self._x01_add_ghost_slide(
                group,
                ghosts,
                host,
                end_pix,
                start,
                end,
                fade_in=True,
                curve=QEasingCurve.InOutCubic,
                duration=480,
            )
            animated.add(pi)

        for pi in pis:
            if pi in animated:
                continue
            row = self._killer_rows_by_pi.get(pi)
            if row is not None:
                row.setGraphicsEffect(None)

        self._killer_turn_ghosts = ghosts

        def _done() -> None:
            for g in list(self._killer_turn_ghosts):
                g.deleteLater()
            self._killer_turn_ghosts = []
            for pi in pis:
                row = self._killer_rows_by_pi.get(pi)
                if row is not None:
                    row.setGraphicsEffect(None)
                    row.setVisible(True)
            self._killer_turn_anim = None

        if group.animationCount() == 0:
            _done()
            return
        group.finished.connect(_done)
        self._killer_turn_anim = group
        group.start()

    def _killer_play_death_anim(
        self,
        starts: Dict[int, QRect],
        pixmaps: Dict[int, QPixmap],
        pis: List[int],
        death_pis: set,
        waiting_pis: List[int],
    ) -> None:
        """Slide newly-dead cards into the finished (right) slot; shift waiting."""
        host = getattr(self, "right_w", None)
        if host is None or not starts or not death_pis:
            return
        if host.layout() is not None:
            host.layout().activate()
        host.update()

        ends: Dict[int, QRect] = {}
        for pi in pis:
            row = self._killer_rows_by_pi.get(pi)
            if row is None or row.width() < 2 or row.height() < 2:
                continue
            ends[pi] = self._x01_card_geom_in(row, host)

        for g in self._killer_turn_ghosts:
            g.deleteLater()
        self._killer_turn_ghosts = []
        if self._killer_turn_anim is not None:
            self._killer_turn_anim.stop()
            self._killer_turn_anim = None

        group = QParallelAnimationGroup(self)
        ghosts: List[QWidget] = []
        animated: set = set()

        for pi in death_pis:
            start = starts.get(pi)
            end = ends.get(pi)
            pix = pixmaps.get(pi)
            if start is None or end is None or pix is None or pix.isNull():
                continue
            row = self._killer_rows_by_pi.get(pi)
            if row is not None:
                row.setGraphicsEffect(None)
            end_pix = self._reuse_or_live_grab(row, pix)
            if end_pix is None or end_pix.isNull():
                end_pix = pix
            if row is not None:
                self._x01_hide_card_keep_layout(row)
            # Cross-fade live card > compact dead icon while sliding right.
            self._x01_add_ghost_slide(
                group,
                ghosts,
                host,
                pix,
                start,
                end,
                fade_out=True,
                curve=QEasingCurve.InOutCubic,
                duration=480,
            )
            self._x01_add_ghost_slide(
                group,
                ghosts,
                host,
                end_pix,
                start,
                end,
                fade_in=True,
                curve=QEasingCurve.InOutCubic,
                duration=480,
            )
            animated.add(pi)

        for pi in waiting_pis:
            if pi in animated:
                continue
            start = starts.get(pi)
            end = ends.get(pi)
            pix = pixmaps.get(pi)
            if start is None or end is None or pix is None or pix.isNull():
                continue
            # Living waiters stay fixed-size; only allow a horizontal shift ghost.
            if start.width() != end.width() or start.height() != end.height():
                end = QRect(end.x(), end.y(), start.width(), start.height())
            if start == end:
                continue
            row = self._killer_rows_by_pi.get(pi)
            if row is not None:
                self._x01_hide_card_keep_layout(row)
            self._x01_add_ghost_slide(
                group,
                ghosts,
                host,
                pix,
                start,
                end,
                curve=QEasingCurve.InOutCubic,
            )
            animated.add(pi)

        for pi in pis:
            if pi in animated:
                continue
            row = self._killer_rows_by_pi.get(pi)
            if row is not None:
                row.setGraphicsEffect(None)

        self._killer_turn_ghosts = ghosts

        def _done() -> None:
            for g in list(self._killer_turn_ghosts):
                g.deleteLater()
            self._killer_turn_ghosts = []
            for pi in pis:
                row = self._killer_rows_by_pi.get(pi)
                if row is not None:
                    row.setGraphicsEffect(None)
                    row.setVisible(True)
            self._killer_turn_anim = None

        if group.animationCount() == 0:
            _done()
            return
        group.finished.connect(_done)
        self._killer_turn_anim = group
        group.start()

    def _sync_killer_rows(self, players: list, snap: dict) -> None:
        """Killer scoreboard: aktivni gore; waiting lijevo + finished desno (kao X01)."""
        foreign = False
        if self._x01_cards_by_pi or self._x01_waiting_wrap is not None:
            foreign = True
        elif self._cricket_rows_by_pi or self._cricket_hdr is not None:
            foreign = True
        elif self.score_box.count() > 0 and not self._killer_rows_by_pi:
            foreign = True
        else:
            known = set(self._killer_rows_by_pi.values())
            if self._killer_hdr is not None:
                known.add(self._killer_hdr)
            if self._killer_waiting_wrap is not None:
                known.add(self._killer_waiting_wrap)
            for i in range(self.score_box.count()):
                w = self.score_box.itemAt(i).widget()
                if w is not None and w not in known:
                    foreign = True
                    break
        if foreign:
            self._clear_score_box()

        # Remove column headers (Broj / Životi) if leftover
        if self._killer_hdr is not None:
            if self.score_box.indexOf(self._killer_hdr) >= 0:
                self.score_box.removeWidget(self._killer_hdr)
            self._killer_hdr.deleteLater()
            self._killer_hdr = None

        if not players:
            self._clear_score_box()
            return

        self.score_box.setSpacing(14)
        self.score_box.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        rules = snap.get("rules") or {}
        act_prefix = mm._killer_activation_prefix(rules)
        max_lives = int(rules.get("killer_lives", 3) or 3)
        max_lives = max(1, min(KillerRow.MAX_LIVES, max_lives))

        n = len(players)
        active_i = next((i for i, p in enumerate(players) if p.get("active")), 0)
        active_p = players[active_i]
        active_pi = int(active_p.get("index", active_i))
        active_finished = self._killer_player_finished(active_p)

        waiting_display = []
        for k in range(1, n):
            p = players[(active_i + k) % n]
            if self._killer_player_finished(p):
                continue
            if bool(p.get("active")):
                continue
            waiting_display.append(p)
            if len(waiting_display) >= 3:
                break

        finished_display = [p for p in players if self._killer_player_finished(p)]
        # Active who just died mid-turn stays on main card until turn flips;
        # still list them among finished for the bottom-right lock once not active.
        if active_finished:
            finished_display = [
                p
                for p in finished_display
                if int(p.get("index", -1)) != active_pi
            ]
        finished_display.sort(key=lambda p: int(p.get("rank") or 99))

        a_icon, a_font, a_min, a_max = self._killer_active_metrics()
        w_icon, w_font, w_min, w_max = self._killer_waiting_metrics()

        keep_pis = {int(p.get("index", i)) for i, p in enumerate(players)}
        for pi in list(self._killer_rows_by_pi.keys()):
            if pi not in keep_pis:
                row = self._killer_rows_by_pi.pop(pi)
                self._killer_detach_row(row)
                row.deleteLater()

        prev_active = self._killer_last_active_pi
        will_anim = (
            prev_active is not None
            and prev_active != active_pi
            and n > 1
            and prev_active in self._killer_rows_by_pi
        )
        reverse = bool(snap.get("turn_anim_reverse"))
        if will_anim and not reverse and prev_active is not None:
            reverse = self._turn_change_is_reverse(
                players, prev_active, active_pi, self._killer_player_finished
            )

        # Newly finished players that move into the bottom-right finished area.
        prev_finished = set(self._killer_finished_pis or set())
        finished_now = {
            int(p.get("index", -1))
            for p in finished_display
            if int(p.get("index", -1)) >= 0
        }
        if active_finished and active_pi >= 0:
            finished_now.add(active_pi)
        # Only animate deaths that land in the finished row this frame
        # (active who dies mid-turn stays on main — animate when turn flips).
        death_move_pis = {
            int(p.get("index", -1))
            for p in finished_display
            if int(p.get("index", -1)) >= 0
            and int(p.get("index", -1)) not in prev_finished
        }

        starts: Dict[int, QRect] = {}
        pixmaps: Dict[int, QPixmap] = {}
        host = getattr(self, "right_w", None)
        need_capture = (will_anim or bool(death_move_pis)) and host is not None
        if need_capture:
            self._stop_killer_turn_anim()
            for pi, row in self._killer_rows_by_pi.items():
                if row.width() < 2 or row.height() < 2:
                    continue
                starts[pi] = self._x01_card_geom_in(row, host)
                pixmaps[pi] = row.grab()
                self._x01_hide_card_keep_layout(row)

        waiting_pis_expected = [
            int(p.get("index", -1)) for p in waiting_display if int(p.get("index", -1)) >= 0
        ]
        finished_pis_expected = [
            int(p.get("index", -1)) for p in finished_display if int(p.get("index", -1)) >= 0
        ]
        structure_stable = (
            not need_capture
            and prev_active == active_pi
            and prev_active is not None
            and set(self._killer_rows_by_pi.keys()) == keep_pis
            and getattr(self, "_killer_last_waiting_pis", None) == waiting_pis_expected
            and getattr(self, "_killer_last_finished_pis", None) == finished_pis_expected
            and self.score_box.indexOf(self._killer_rows_by_pi.get(active_pi)) == 0
        )

        def _apply_row(row: KillerRow, p: dict, *, active: bool, finished: bool) -> None:
            lives = int(p.get("lives") or 0)
            # Winner keeps lives > 0 — never dead chrome / strike-X.
            finished_eff = bool(finished) and lives <= 0
            row.set_row(
                p.get("number"),
                lives,
                p.get("color", "#fff"),
                active and not finished_eff,
                is_killer=bool(p.get("is_killer")),
                finished=finished_eff,
                act_prefix=act_prefix,
                max_lives=max_lives,
                name=str(p.get("name") or ""),
            )

        if structure_stable:
            active_row = self._killer_rows_by_pi[active_pi]
            active_row.set_metrics(a_icon, a_font, a_min, a_max)
            active_row.apply_waiting_fixed_width(False)
            active_row.apply_compact_width(False)
            _apply_row(active_row, active_p, active=True, finished=active_finished)
            wait_sample = f"{act_prefix}20"
            for p in waiting_display:
                pi = int(p.get("index", -1))
                if pi < 0:
                    continue
                row = self._killer_rows_by_pi.get(pi)
                if row is None:
                    continue
                row.set_metrics(w_icon, w_font, w_min, w_max)
                row.apply_waiting_fixed_width(True, sample=wait_sample)
                _apply_row(row, p, active=False, finished=False)
            for p in finished_display:
                pi = int(p.get("index", -1))
                if pi < 0:
                    continue
                row = self._killer_rows_by_pi.get(pi)
                if row is None:
                    continue
                row.set_metrics(w_icon, w_font, w_min, w_max)
                row.apply_waiting_fixed_width(False)
                row.apply_compact_width(True)
                _apply_row(row, p, active=False, finished=True)
            wait_lay = self._killer_waiting_layout
            if wait_lay is not None:
                self._killer_align_waiting_row(wait_lay)
            self._killer_lock_waiting_row_height(w_max)
            self._killer_last_active_pi = active_pi
            self._killer_finished_pis = finished_now
            self._killer_last_waiting_pis = waiting_pis_expected
            self._killer_last_finished_pis = finished_pis_expected
            return

        for row in list(self._killer_rows_by_pi.values()):
            self._killer_detach_row(row)
        if self._killer_waiting_layout is not None:
            while self._killer_waiting_layout.count():
                item = self._killer_waiting_layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.setParent(None)

        # Aktivni gore — puna širina
        active_row = self._killer_row_for(active_pi)
        self._killer_detach_row(active_row)
        self.score_box.insertWidget(0, active_row, 0)
        active_row.set_metrics(a_icon, a_font, a_min, a_max)
        active_row.apply_waiting_fixed_width(False)
        active_row.apply_compact_width(False)
        _apply_row(active_row, active_p, active=True, finished=active_finished)

        waiting_pis: List[int] = []
        finished_pis: List[int] = []
        row_n = len(waiting_display) + len(finished_display)
        if row_n == 0:
            if self._killer_waiting_wrap is not None:
                self._killer_waiting_wrap.hide()
        else:
            wait_lay = self._killer_ensure_waiting_row()
            if self.score_box.indexOf(self._killer_waiting_wrap) < 0:
                self.score_box.addWidget(self._killer_waiting_wrap, 0)
            self._killer_waiting_wrap.show()

            wait_sample = f"{act_prefix}20"
            for p in waiting_display:
                pi = int(p.get("index", -1))
                if pi < 0:
                    continue
                waiting_pis.append(pi)
                row = self._killer_row_for(pi)
                self._killer_detach_row(row)
                wait_lay.addWidget(row, 0)
                row.set_metrics(w_icon, w_font, w_min, w_max)
                row.apply_waiting_fixed_width(True, sample=wait_sample)
                _apply_row(row, p, active=False, finished=False)

            for p in finished_display:
                pi = int(p.get("index", -1))
                if pi < 0:
                    continue
                finished_pis.append(pi)
                row = self._killer_row_for(pi)
                self._killer_detach_row(row)
                wait_lay.addWidget(row, 0)
                row.set_metrics(w_icon, w_font, w_min, w_max)
                row.apply_waiting_fixed_width(False)
                row.apply_compact_width(True)
                _apply_row(row, p, active=False, finished=True)

            self._killer_align_waiting_row(wait_lay)
            self._killer_lock_waiting_row_height(w_max)

        self._killer_last_active_pi = active_pi if active_pi >= 0 else None
        self._killer_finished_pis = finished_now
        self._killer_last_waiting_pis = list(waiting_pis)
        self._killer_last_finished_pis = list(finished_pis)

        if starts and host is not None:
            pis = list(keep_pis)
            death_pis = set(death_move_pis)
            if will_anim:
                out_pi = prev_active
                in_pi = active_pi
                rev = reverse

                def _go_turn() -> None:
                    self._killer_play_turn_anim(
                        starts,
                        pixmaps,
                        pis,
                        in_pi,
                        out_pi,
                        waiting_pis,
                        finished_pis,
                        reverse=rev,
                        death_pis=death_pis,
                    )

                QTimer.singleShot(0, _go_turn)
            elif death_pis:

                def _go_death() -> None:
                    self._killer_play_death_anim(
                        starts,
                        pixmaps,
                        pis,
                        death_pis,
                        waiting_pis,
                    )

                QTimer.singleShot(0, _go_death)

    def _refresh_killer_bull_overlay(self, snap: dict) -> None:
        if not hasattr(self, "killer_bull_overlay"):
            return
        pending = snap.get("killer_bull_pending")
        if (
            not pending
            or snap.get("screen") != "playing"
            or self.session.is_in_game_calibrating()
        ):
            self.killer_bull_overlay.hide()
            return
        t = get_settings().t
        self.killer_bull_title.setText(t("killer_bull_prompt").upper())
        self.killer_bull_false_btn.setText(t("killer_bull_false").upper())

        while self.killer_bull_targets_lay.count():
            item = self.killer_bull_targets_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        targets = list(snap.get("killer_bull_targets") or [])
        # Fallback: pull lives from players snap if targets omit them.
        players_by_i = {
            int(p.get("index", i)): p
            for i, p in enumerate(snap.get("players") or [])
        }
        n_tgt = max(1, len(targets))
        for tgt in targets:
            pi = int(tgt.get("index", -1))
            if pi < 0:
                continue
            psnap = players_by_i.get(pi) or {}
            color = str(tgt.get("color") or psnap.get("color") or "#ffffff")
            fg = _player_fg(color)
            lives = int(
                tgt.get("lives")
                if tgt.get("lives") is not None
                else (psnap.get("lives") or 0)
            )
            if lives <= 0:
                continue
            # Neutral dark chrome — do not tint whole button with player color
            # (poor contrast on Primary / theme blue).
            btn = AnimButton("")
            btn.setProperty("action", f"killer_bull_steal:{pi}")
            btn.setObjectName("KillerBullTarget")
            btn.setMinimumHeight(340)
            btn.setMinimumWidth(300)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            btn.setCursor(Qt.ArrowCursor)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.setStyleSheet(
                "QPushButton#KillerBullTarget {"
                "  background-color: #2a2a32;"
                "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
                "    stop:0 #3a3a44, stop:0.45 #2a2a32, stop:0.55 #2a2a32, stop:1 #1e1e26);"
                "  color: #ffffff;"
                "  border: 3px solid #5a5a66;"
                "  border-radius: 18px;"
                "  padding: 0px;"
                "}"
                "QPushButton#KillerBullTarget:hover, QPushButton#KillerBullTarget:focus {"
                "  background-color: #32323c;"
                "  border-color: #787888;"
                "}"
                "QPushButton#KillerBullTarget:pressed {"
                "  background-color: #1e1e26;"
                "}"
            )
            content = QWidget(btn)
            content.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            content.setStyleSheet("background: transparent; border: none;")
            inner = QVBoxLayout(content)
            inner.setContentsMargins(20, 28, 20, 28)
            inner.setSpacing(18)
            inner.setAlignment(Qt.AlignCenter)

            pname = str(tgt.get("name") or psnap.get("name") or "").strip()
            shown = _display_player_name(pname)
            if shown:
                name_lab = QLabel(shown)
                name_lab.setAlignment(Qt.AlignCenter)
                name_lab.setWordWrap(False)
                name_px = 58 if n_tgt <= 3 else 48
                if len(shown) >= 7:
                    name_px = 50 if n_tgt <= 3 else 40
                name_lab.setStyleSheet(
                    f"font-size: {name_px}px; font-weight: 900; color: {fg};"
                    f" background: transparent; border: none;"
                )
                inner.addWidget(name_lab, 0, Qt.AlignHCenter)
            else:
                icon = QLabel()
                icon_h = 120 if n_tgt <= 3 else 96
                icon.setFixedSize(icon_h, icon_h)
                icon.setAlignment(Qt.AlignCenter)
                icon.setStyleSheet("background: transparent; border: none;")
                pix = _player_icon_tinted(icon_h, fg)
                _apply_player_avatar(icon, color, icon_h, "", pix=pix)
                inner.addWidget(icon, 0, Qt.AlignHCenter)

            lives_row = QWidget()
            lives_row.setStyleSheet("background: transparent; border: none;")
            lr = QHBoxLayout(lives_row)
            lr.setContentsMargins(0, 0, 0, 0)
            lr.setSpacing(12)
            lr.setAlignment(Qt.AlignCenter)
            heart_lab = QLabel()
            heart_lab.setAlignment(Qt.AlignCenter)
            heart_lab.setStyleSheet("background: transparent; border: none;")
            heart_sz = 56 if n_tgt <= 3 else 48
            heart = _heart_pixmap(heart_sz, fg)
            if heart is not None and not heart.isNull():
                heart_lab.setPixmap(heart)
                heart_lab.setFixedSize(heart.width(), heart.height())
            lr.addWidget(heart_lab, 0, Qt.AlignVCenter)
            count = QLabel(str(max(0, lives)))
            count.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            lives_px = 64 if n_tgt <= 3 else 52
            count.setStyleSheet(
                f"font-size: {lives_px}px; font-weight: 800; color: {fg};"
                f" background: transparent; border: none;"
            )
            lr.addWidget(count, 0, Qt.AlignVCenter)
            inner.addWidget(lives_row, 0, Qt.AlignHCenter)

            outer = QVBoxLayout(btn)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(0)
            outer.addWidget(content)

            btn.clicked.connect(lambda _=False, a=f"killer_bull_steal:{pi}": self._act(a))
            self.killer_bull_targets_lay.addWidget(btn, 1)

        root = self.centralWidget()
        if root is not None:
            self.killer_bull_overlay.setGeometry(0, 0, root.width(), root.height())
        self.killer_bull_overlay.show()
        self.killer_bull_overlay.raise_()

    def _add_player_card(self, label: str, score: str, color: str, active: bool) -> None:
        card = PlayerCard()
        card.set_player(label, score, color, active)
        self.score_box.addWidget(card, 0, Qt.AlignLeft)
        self._player_cards.append(card)

    def _refresh_playing(self, snap: dict) -> None:
        self.play_title.setText(snap.get("game_title") or "IGRA")
        editing = snap.get("editing_hit_slot")
        manual_mode = bool(snap.get("manual_mode"))
        show_title = (
            editing is None
            and not manual_mode
            and snap.get("winner") is None
        )
        self.play_title.setVisible(show_title)
        has_logo = self.play_logo.pixmap() is not None and not self.play_logo.pixmap().isNull()
        self.play_logo.setVisible(show_title and has_logo)
        if hasattr(self, "manual_hits_wrap"):
            show_manual_hits = manual_mode and snap.get("winner") is None
            self.manual_hits_wrap.setVisible(show_manual_hits)
        self.left_stack.setCurrentWidget(
            self.keypad_wrap
            if (editing is not None or manual_mode)
            else self.hits_wrap
        )

        tags = snap.get("turn_hit_tags") or ["-", "-", "-"]
        enabled = snap.get("slot_enabled") or [True, True, True]

        def _style_hit_btn(b: QPushButton, new_tag: str, *, font_px: int) -> None:
            if new_tag == "MISS":
                col, border = "#ff5050", "#ff5050"
                metal = _metal_bg("#2a1518", "#3a2228", "#1a1014")
            elif new_tag == "-" or not new_tag:
                col, border = "#777780", "#5a5a68"
                metal = _metal_bg("#1c1c22", "#2c2c34", "#141418")
            elif new_tag.startswith("T"):
                col, border = "#d090ff", "#c070f0"
                metal = _metal_bg("#1a1224", "#2a1c38", "#120c1c")
            elif new_tag.startswith("D") or new_tag == "B50":
                col, border = "#ffd040", "#e0b020"
                metal = _metal_bg("#2a2410", "#3a3420", "#1c1808")
            else:
                col, border = "#00ff78", "#00c860"
                metal = _metal_bg("#122018", "#1e3028", "#0c1810")
            b.setStyleSheet(
                f"QPushButton#HitSlot {{ color: {col}; border-color: {border}; "
                f"{metal} font-size: {font_px}px; font-weight: 800; min-height: 0px; }}"
                f"QPushButton#HitSlot:hover, QPushButton#HitSlot:focus {{ "
                f"color: {col}; border-color: {border}; {metal} }}"
            )

        for i, b in enumerate(self.hit_btns):
            new_tag = tags[i]
            old_tag = self._hit_tags_prev[i] if i < len(self._hit_tags_prev) else "-"
            b.setText(new_tag)
            b.setEnabled(bool(enabled[i]))
            if new_tag != old_tag:
                _style_hit_btn(b, new_tag, font_px=128)
                b.repaint()
                if new_tag != "-":
                    self._animate_hit(i)
        if hasattr(self, "manual_hit_btns"):
            editing_slot = snap.get("editing_hit_slot")
            for i, b in enumerate(self.manual_hit_btns):
                new_tag = tags[i] if i < len(tags) else "-"
                b.setText(new_tag)
                b.setEnabled(bool(enabled[i]) if i < len(enabled) else True)
                _style_hit_btn(b, new_tag, font_px=72)
                if (
                    editing_slot is not None
                    and int(editing_slot) == i
                    and manual_mode
                ):
                    b.setStyleSheet(
                        b.styleSheet()
                        + "QPushButton#HitSlot { border-width: 5px; }"
                    )
        if hasattr(self, "keypad_grid") and hasattr(self, "key_clr_btn") and hasattr(
            self, "key_close_btn"
        ):
            if manual_mode:
                self.keypad_grid.addWidget(self.key_clr_btn, 5, 0, 1, 5)
                self.key_close_btn.hide()
            else:
                self.keypad_grid.addWidget(self.key_clr_btn, 5, 0, 1, 4)
                self.keypad_grid.addWidget(self.key_close_btn, 5, 4, 1, 1)
                self.key_close_btn.show()
        self._hit_tags_prev = list(tags)

        mult = int(snap.get("input_multiplier", 1))
        self.mult2.setObjectName("KeyModOn" if mult == 2 else "KeyMod")
        self.mult3.setObjectName("KeyModOn" if mult == 3 else "KeyMod")
        for b in (self.mult2, self.mult3):
            b.style().unpolish(b)
            b.style().polish(b)

        if snap.get("winner") is not None:
            self._clear_score_box()
            self.play_title.hide()
            if hasattr(self, "manual_hits_wrap"):
                self.manual_hits_wrap.hide()
            self.left_w.hide()
            self.play_body.setStretch(0, 0)
            self.play_body.setStretch(1, 1)
            self.live_line.hide()
            if hasattr(self, "live_stack"):
                self.live_stack.hide()
            if hasattr(self, "score_suggest_sep"):
                self.score_suggest_sep.hide()
            self.undo_btn.hide()
            if hasattr(self, "mode_toggle_btn"):
                self.mode_toggle_btn.hide()
            if hasattr(self, "play_cal_btn"):
                self.play_cal_btn.hide()
            self.action_stack.hide()
            # Scoreboard popuni prostor radi vertikalnog centriranja
            self.right_layout.setStretch(0, 1)
            self.right_layout.setStretch(3, 0)
            entries = snap.get("leaderboard") or []
            by_rank = {int(rank): int(pi) for rank, pi in entries}
            names_by_pi = {
                int(p.get("index", i)): str(p.get("name") or "")
                for i, p in enumerate(snap.get("players") or [])
            }
            mode = str(snap.get("mode") or "").lower()
            is_killer_win = mode == "killer"

            n_players = len(snap.get("players") or [])
            # (icon_h, font_px, name_px, bottom_pad)
            size_by_rank = {
                1: (168, 96, 40, 20),
                2: (124, 70, 32, 10),
                3: (112, 64, 30, 4),
                4: (104, 58, 28, 0),
            }
            col_w = 236

            def _podium_col(rank: int, pi: int) -> QWidget:
                color = _bgr_hex(pi)
                fg = _player_fg(color)
                icon_h, font_px, name_px, lift = size_by_rank.get(rank, (104, 58, 28, 0))
                col = QWidget()
                col.setMinimumWidth(col_w)
                col.setMaximumWidth(280)
                cl = QVBoxLayout(col)
                cl.setContentsMargins(6, 0, 6, lift)
                cl.setSpacing(10 if rank == 1 else 8)
                cl.setAlignment(Qt.AlignHCenter | Qt.AlignBottom)

                icon = QLabel()
                icon.setAlignment(Qt.AlignCenter)
                pname = names_by_pi.get(int(pi), "")
                if rank == 1:
                    if is_killer_win:
                        pix = _killer_icon_tinted(icon_h, fg)
                    else:
                        pix = _winner_icon_tinted(icon_h, fg)
                else:
                    pix = _initials_avatar_pixmap(pname, icon_h, color)
                    if pix is None:
                        pix = _player_icon_tinted(icon_h, fg)
                    if is_killer_win and pix is not None:
                        pix = _icon_with_strike(pix, thin=True)
                if pix is not None:
                    icon.setPixmap(pix)
                    icon.setFixedSize(pix.width(), pix.height())

                lab = QLabel(f"#{rank}")
                lab.setAlignment(Qt.AlignCenter)
                if rank == 1:
                    lab.setObjectName("WinnerTitle")
                    lab.setStyleSheet(
                        f"font-size: {font_px}px; font-weight: 900; color: {fg};"
                    )
                    self._start_winner_pulse(lab)
                else:
                    lab.setObjectName("WinnerSub")
                    lab.setStyleSheet(
                        f"font-size: {font_px}px; font-weight: 800; color: {fg};"
                    )

                cl.addWidget(icon, 0, Qt.AlignHCenter)
                cl.addWidget(lab, 0, Qt.AlignHCenter)
                shown_name = _display_player_name(pname)
                if shown_name:
                    nl = QLabel(shown_name)
                    nl.setAlignment(Qt.AlignCenter)
                    nl.setWordWrap(False)
                    nl.setFixedWidth(col_w)
                    nl.setStyleSheet(
                        f"font-size: {name_px}px; font-weight: 800; color: {fg};"
                    )
                    cl.addWidget(nl, 0, Qt.AlignHCenter)
                return col

            wrap = QWidget()
            wrap.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            wl = QVBoxLayout(wrap)
            wl.setAlignment(Qt.AlignCenter)
            qr_payload = (
                str(snap.get("league_qr_payload") or "").strip()
                if n_players >= 2 and get_settings().league_qr_enabled
                else ""
            )
            wl.setSpacing(18 if qr_payload else 28)
            wl.setContentsMargins(16, 8, 16, 8)

            title = QLabel(get_settings().t("winner_title"))
            title.setObjectName("WinnerTitle")
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet(
                "font-size: 84px; font-weight: 900; color: #ffe14a; letter-spacing: 2px;"
            )
            wl.addWidget(title, 0, Qt.AlignHCenter)

            qr_col_w = 400
            qr_col: Optional[QWidget] = None
            if qr_payload:
                qr_col = QWidget()
                qr_col.setFixedWidth(qr_col_w)
                qcl = QVBoxLayout(qr_col)
                qcl.setContentsMargins(4, 0, 4, 4)
                qcl.setSpacing(12)
                qcl.setAlignment(Qt.AlignHCenter | Qt.AlignBottom)
                frame = QFrame()
                frame.setObjectName("QrFrame")
                frame.setAttribute(Qt.WA_StyledBackground, True)
                fl = QVBoxLayout(frame)
                fl.setContentsMargins(12, 12, 12, 12)
                ql = QLabel()
                ql.setAlignment(Qt.AlignCenter)
                qr_pix = _league_qr_pixmap(qr_payload, 360)
                if qr_pix is not None and not qr_pix.isNull():
                    ql.setPixmap(qr_pix)
                    ql.setFixedSize(qr_pix.width(), qr_pix.height())
                else:
                    ql.setText(qr_payload)
                    ql.setWordWrap(True)
                    ql.setFixedWidth(340)
                    ql.setStyleSheet(
                        "font-size: 16px; font-weight: 700; color: #111111; "
                        "background: #ffffff;"
                    )
                fl.addWidget(ql)
                qcl.addWidget(frame, 0, Qt.AlignHCenter)
                hint = QLabel(get_settings().t("league_qr_hint"))
                hint.setAlignment(Qt.AlignCenter)
                hint.setWordWrap(True)
                hint.setStyleSheet(
                    "font-size: 30px; font-weight: 800; color: #ffe14a;"
                )
                qcl.addWidget(hint, 0, Qt.AlignHCenter)

            # Lijevo | #1 | desno s jednakim stretchom > #1 uvijek u sredini ekrana
            podium_row = QWidget()
            podium_row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            pr = QHBoxLayout(podium_row)
            pr.setContentsMargins(0, 20, 0, 0)
            pr.setSpacing(20 if qr_col is not None else 36)

            left_side = QHBoxLayout()
            left_side.setContentsMargins(0, 0, 0, 0)
            left_side.setSpacing(36)
            left_side.setAlignment(Qt.AlignRight | Qt.AlignBottom)
            for rank in (4, 2):
                if rank in by_rank:
                    left_side.addWidget(
                        _podium_col(rank, by_rank[rank]), 0, Qt.AlignBottom
                    )

            right_side = QHBoxLayout()
            right_side.setContentsMargins(0, 0, 0, 0)
            right_side.setSpacing(36)
            right_side.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
            for rank in (3,):
                if rank in by_rank:
                    right_side.addWidget(
                        _podium_col(rank, by_rank[rank]), 0, Qt.AlignBottom
                    )

            if qr_col is not None:
                left_pad = QWidget()
                left_pad.setFixedWidth(qr_col_w)
                pr.addWidget(left_pad, 0)
            if 1 in by_rank:
                pr.addLayout(left_side, 1)
                pr.addWidget(_podium_col(1, by_rank[1]), 0, Qt.AlignBottom)
                pr.addLayout(right_side, 1)
            else:
                # Fallback: svi unosi centrirani ako nema #1
                pr.addStretch(1)
                for rank, pi in ((int(r), int(pi)) for r, pi in entries):
                    pr.addWidget(_podium_col(rank, pi), 0, Qt.AlignBottom)
                pr.addStretch(1)
            if qr_col is not None:
                pr.addWidget(qr_col, 0, Qt.AlignBottom)

            wl.addWidget(podium_row, 1)
            self.score_box.setAlignment(Qt.AlignVCenter)
            self.score_box.addStretch(1)
            self.score_box.addWidget(wrap, 1)
            self.score_box.addStretch(1)
            self.exit_btn.setText(get_settings().t("exit_game"))
            self.exit_btn.setObjectName("PlayExitWide")
            self.exit_btn.style().unpolish(self.exit_btn)
            self.exit_btn.style().polish(self.exit_btn)
            self.exit_btn.setFixedHeight(100)
            self.exit_btn.setMinimumWidth(420)
            self.exit_btn.setMaximumWidth(560)
            self.exit_btn.show()
            self.footer_block.show()
            # Centriraj exit: stretch lijevo/desno, bez undo/mode/cal
            # layout: stretch | undo | mode | cal | exit | stretch
            self.bottom_row.setStretch(0, 1)
            self.bottom_row.setStretch(1, 0)
            self.bottom_row.setStretch(2, 0)
            self.bottom_row.setStretch(3, 0)
            self.bottom_row.setStretch(4, 0)
            self.bottom_row.setStretch(5, 1)
            if hasattr(self, "killer_bull_overlay"):
                self.killer_bull_overlay.hide()
            if hasattr(self, "ingame_cal_overlay"):
                self.ingame_cal_overlay.hide()
            return

        self._stop_winner_pulse()
        self.left_w.show()
        self.play_body.setStretch(0, 5)
        self.play_body.setStretch(1, 5)
        self.right_layout.setStretch(0, 0)
        self.right_layout.setStretch(3, 1)
        self.score_box.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.left_stack.show()
        self.live_line.show()
        if hasattr(self, "live_stack"):
            self.live_stack.show()
        self.undo_btn.show()
        if hasattr(self, "mode_toggle_btn"):
            self.mode_toggle_btn.show()
        if hasattr(self, "play_cal_btn"):
            self.play_cal_btn.show()
        self.action_stack.show()
        self.exit_btn.setText("X")
        self.exit_btn.setObjectName("PlayExit")
        self.exit_btn.style().unpolish(self.exit_btn)
        self.exit_btn.style().polish(self.exit_btn)
        self.exit_btn.setFixedHeight(84)
        self.exit_btn.setMinimumWidth(96)
        self.exit_btn.setMaximumWidth(120)
        self.undo_btn.setFixedHeight(84)
        if hasattr(self, "mode_toggle_btn"):
            self.mode_toggle_btn.setFixedHeight(84)
        if hasattr(self, "play_cal_btn"):
            self.play_cal_btn.setFixedHeight(84)
        self.bottom_row.setStretch(0, 0)
        self.bottom_row.setStretch(1, 1)
        self.bottom_row.setStretch(2, 1)
        self.bottom_row.setStretch(3, 1)
        self.bottom_row.setStretch(4, 0)
        self.bottom_row.setStretch(5, 0)
        mode = snap.get("mode") or ""
        players = snap.get("players") or []
        rules = snap.get("rules") or {}

        if mode == "cricket":
            self._sync_cricket_rows(players, rules, snap)
        elif mode == "killer":
            self._sync_killer_rows(players, snap)
        else:
            # Cricket/Killer/winner ostaci u score_boxu > full rebuild
            known = set(self._x01_cards_by_pi.values())
            if self._x01_waiting_wrap is not None:
                known.add(self._x01_waiting_wrap)
            for i in range(self.score_box.count()):
                w = self.score_box.itemAt(i).widget()
                if w is not None and w not in known:
                    self._clear_score_box()
                    break
            self._sync_x01_player_cards(
                players,
                reverse=bool(snap.get("turn_anim_reverse")),
                around_mode=mode in ("around", "around_the_world"),
            )

        if mode == "killer":
            # Lives are on the cards — hide under-card roster / suggest area.
            self.live_line.setText(" ")
            self.live_line.setStyleSheet("")
            if hasattr(self, "live_stack"):
                self.live_stack.hide()
            self.score_suggest_sep.hide()
        elif mode == "halve":
            if hasattr(self, "live_stack"):
                self.live_stack.show()
                self.live_stack.setCurrentWidget(self.live_line)
            live = snap.get("live_score_line") or ""
            self.live_line.setText(live if str(live).strip() else " ")
            self.live_line.setStyleSheet(
                "QLabel#LiveScore { color: #ffe14a; font-size: 64px; font-weight: 800; }"
            )
            self.score_suggest_sep.setVisible(bool(str(live).strip()))
        elif snap.get("is_bust"):
            if hasattr(self, "live_stack"):
                self.live_stack.show()
                self.live_stack.setCurrentWidget(self.live_line)
            self.live_line.setText(get_settings().t("bust"))
            self.live_line.setStyleSheet(
                "QLabel#LiveScore { color: #ff4040; font-size: 80px; font-weight: 800; }"
            )
            self.score_suggest_sep.show()
        else:
            if hasattr(self, "live_stack"):
                self.live_stack.show()
                self.live_stack.setCurrentWidget(self.live_line)
            live = snap.get("live_score_line")
            pending = snap.get("undo_pending_tags")
            live_txt = live or ""
            if pending:
                filled = [t for t in pending if t and t != "-"]
                if filled:
                    live_txt = get_settings().t("next_prefix") + " " + ", ".join(filled)
            self.live_line.setText(live_txt if live_txt else " ")
            self.live_line.setStyleSheet("")
            has_sug = bool(str(live_txt).strip())
            self.score_suggest_sep.setVisible(has_sug)

        msg = (snap.get("board_clear_message") or "").strip()
        show_banner = bool(msg) and not snap.get("show_manual_advance")
        if snap.get("show_manual_advance"):
            label = snap.get("advance_label") or ">"
            self.advance_btn.setText(label)
            if snap.get("block_adv") or snap.get("manual_mode"):
                role = "PlayAction"
            elif self.session.ctx.get("_dart_manual_advance_confirm"):
                role = "PlayActionAccent"
            else:
                role = "PlayActionWarn"
            self.advance_btn.setObjectName(role)
            self.advance_btn.style().unpolish(self.advance_btn)
            self.advance_btn.style().polish(self.advance_btn)
            self.action_stack.setCurrentWidget(self.advance_btn)
        elif show_banner:
            self.board_msg.setText(msg)
            self.action_stack.setCurrentWidget(self.board_msg)
        else:
            self.board_msg.clear()
            self.action_stack.setCurrentWidget(self.action_empty)

        self.undo_btn.setEnabled(bool(snap.get("has_undo")))
        if hasattr(self, "mode_toggle_btn"):
            t = get_settings().t
            if bool(snap.get("manual_mode")):
                self.mode_toggle_btn.setText(t("play_mode_manual"))
            else:
                self.mode_toggle_btn.setText(t("play_mode_auto"))
            # Uvijek sivi (PlayNav) — i AUTO i MANUAL.
            self.mode_toggle_btn.setObjectName("PlayNav")
            self.mode_toggle_btn.style().unpolish(self.mode_toggle_btn)
            self.mode_toggle_btn.style().polish(self.mode_toggle_btn)
        self._refresh_killer_bull_overlay(snap)



def _bgr_hex(pi: int) -> str:
    b, g, r = mm._player_color(pi)
    return f"#{r:02x}{g:02x}{b:02x}"


def _find_splash_pixmap() -> Optional[QPixmap]:
    """Splash dok se UI diže — assets/splash.png, pa starije lokacije."""
    root = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(root, "assets", "splash.png"),
        os.path.join(root, "splash.png"),
        os.path.expanduser("~/splash.png"),
        "/home/user2/smartdarts/assets/splash.png",
        "/home/user2/splash.png",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            pm = QPixmap(path)
            if not pm.isNull():
                return pm
    return None


def run_qt_app(*, kiosk: bool = False, fake_cams_dir: Optional[str] = None) -> int:
    # Must be set before QApplication on Pi — GLES is faster than full GL software path.
    if _WEAK_HW:
        QApplication.setAttribute(Qt.AA_UseOpenGLES, True)
        # Avoid unnecessary high-DPI pixmap thrash on fixed HDMI panels.
        try:
            QApplication.setAttribute(Qt.AA_Use96Dpi, True)
        except AttributeError:
            pass
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # Touchscreen: bez sticky hover highlighta
    app.setAttribute(Qt.AA_SynthesizeMouseForUnhandledTouchEvents, True)
    # Prefer fonts that support Croatian diacritics
    font = QFont("Segoe UI")
    if not font.exactMatch():
        font = QFont("DejaVu Sans")
    font.setPointSize(28)
    app.setFont(font)
    app.setStyleSheet(APP_QSS)

    # Odmah fullscreen splash dok se učitava MainWindow / kamere.
    splash: Optional[QSplashScreen] = None
    splash_pm = _find_splash_pixmap()
    if splash_pm is None:
        screen = app.primaryScreen()
        if screen is not None:
            geo = screen.geometry()
            splash_pm = QPixmap(geo.size())
            splash_pm.fill(QColor("#0a0a0e"))
    if splash_pm is not None:
        screen = app.primaryScreen()
        if screen is not None:
            geo = screen.geometry()
            splash_pm = splash_pm.scaled(
                geo.size(),
                Qt.KeepAspectRatioByExpanding,
                _PIXMAP_TRANSFORM,
            )
        splash = QSplashScreen(splash_pm)
        splash.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        if kiosk:
            splash.showFullScreen()
        else:
            splash.show()
        app.processEvents()

    session = KioskSession(kiosk=kiosk, fake_cams_dir=fake_cams_dir)
    bridge = MqttBridge()

    def _on_mqtt_message(_mqttc, _userdata, msg):
        try:
            payload = msg.payload.decode().strip().lower()
        except Exception:
            payload = ""
        if payload in ("on", "off"):
            bridge.command.emit(payload)

    def _on_mqtt_connect(mqttc, _userdata, _flags, rc, properties=None):
        try:
            if rc == 0:
                from mqtt import TOPIC

                mqttc.subscribe(TOPIC)
        except Exception:
            pass

    mqtt_client = None
    start_mode = get_settings().start_mode
    # button/always_on: ne blokiraj boot na MQTT (do ~2.5s).
    if start_mode == "mqtt_qr":
        try:
            import ssl
            import paho.mqtt.client as mqtt
            from mqtt import BROKER_HOST, BROKER_PORT, PASSWORD, USERNAME

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
    else:
        print(f"[MQTT] GUI subscriber skipped (start_mode={start_mode})", flush=True)

    if splash is not None:
        app.processEvents()

    win = MainWindow(session)
    win._mqtt_client = mqtt_client
    win._mqtt_bridge = bridge

    # mqtt_qr: ne šalji 'on' pri bootu — čekaj pravi QR sken (inače odmah odabir igre).
    if start_mode == "mqtt_qr":
        print("[MQTT] Waiting for QR / 'on' (no startup publish)", flush=True)
    else:
        print(f"[MQTT] Startup skipped (start_mode={start_mode})", flush=True)

    def _on_cmd(cmd: str) -> None:
        # MQTT 'on' budi select_game samo u mqtt_qr načinu.
        if cmd == "on":
            if get_settings().start_mode == "mqtt_qr":
                session.mqtt_wake()
        elif cmd == "off":
            session.clear_to_standby(mqtt_off=False)
        win._last_ui_sig = None
        win.refresh()

    bridge.command.connect(_on_cmd)

    if kiosk:
        win.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        win.showFullScreen()
        win._lock_kiosk_display(force=True)
        app.installEventFilter(win)
    else:
        win.resize(1280, 720)
        win.show()

    if splash is not None:
        splash.finish(win)

    # Osiguraj točan home ekran (button > standby).
    session._apply_start_mode_home()
    win._last_ui_sig = None
    win.refresh()

    code = app.exec()

    try:
        if mqtt_client is not None:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
    except Exception:
        pass
    session.shutdown()
    return int(code)
