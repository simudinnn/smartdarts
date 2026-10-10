"""Stable TV player grid, using the original game card widgets unchanged."""
from __future__ import annotations

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSizePolicy,
)


class SpectatorCards(QGraphicsView):
    """Scale complete cards around fixed cell centres, never rotate players."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setInteractive(False)
        self.setFocusPolicy(Qt.NoFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(0, 0)
        self.setStyleSheet("QGraphicsView { background: transparent; border: none; }")
        # SmoothPixmapTransform forces costly resampling of every proxy-widget
        # repaint on Raspberry Pi. Text/icons are already rendered antialiased.
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setViewportUpdateMode(QGraphicsView.MinimalViewportUpdate)
        self.setOptimizationFlags(
            QGraphicsView.DontSavePainterState
            | QGraphicsView.DontAdjustForAntialiasing
        )
        self.setScene(QGraphicsScene(self))
        self.entries = {}
        self.order = []
        self.mode = None

    def clear_cards(self):
        # Scene owns proxies and widgets (including their local animations).
        self.scene().clear()
        self.entries.clear()
        self.order = []

    @staticmethod
    def _cricket_panel(row):
        # Same header widths/order as the main Cricket scoreboard. Repeat it
        # per card so every player's markers remain readable in a 2x2 grid.
        from qt_ui import CricketRow, MarkStack
        panel = QWidget()
        panel.setAttribute(Qt.WA_StyledBackground, True)
        panel.setStyleSheet("background: #000000; border-radius: 16px;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        header = QHBoxLayout()
        header.setContentsMargins(6, 0, 8, 0)
        header.setSpacing(0)
        avatar_space = QLabel("")
        avatar_space.setFixedWidth(CricketRow.ICON_H + 2 * CricketRow.ICON_BORDER)
        header.addWidget(avatar_space)
        for target in ("15", "16", "17", "18", "19", "20", "B"):
            header.addStretch(1)
            label = QLabel(target)
            label.setFixedWidth(MarkStack.COL_W)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet(
                "font-size: 40px; font-weight: 800; color: #d0d0d8; padding: 0px;"
            )
            header.addWidget(label)
        header.addStretch(1)
        points = QLabel("PTS")
        points.setFixedWidth(CricketRow.PTS_W)
        points.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        points.setStyleSheet("font-size: 36px; font-weight: 700; color: #d0d0d8; padding-right: 4px;")
        header.addWidget(points)
        layout.addLayout(header)
        layout.addWidget(row)
        panel.setFixedSize(880, 184)
        return panel, points

    def _create_entry(self, pi, mode):
        from qt_ui import PlayerCard, CricketRow, KillerRow, MainWindow
        points_header = None
        if mode == "cricket":
            card = CricketRow()
            widget, points_header = self._cricket_panel(card)
        elif mode == "killer":
            card = KillerRow()
            icon_h = 164
            font_px = 92
            min_h, max_h = MainWindow._killer_card_height(icon_h, 1.46)
            card.set_metrics(icon_h, font_px, min_h, max_h)
            card.apply_waiting_fixed_width(False)
            card.apply_compact_width(False)
            card.setFixedWidth(820)
            widget = card
        else:
            card = PlayerCard()
            card.set_around_progress(0 if mode == "around" else None, "#ffffff")
            card.set_metrics(160, 147, 200, 224)
            card.apply_compact_width(False)
            card.setFixedWidth(680)
            card.set_score_anim_enabled(False)
            widget = card
        card.setProperty("spectatorCard", True)
        if hasattr(card, "icon_lab"):
            card.icon_lab.setProperty("spectatorInitials", True)
        proxy = self.scene().addWidget(widget)
        rect = proxy.boundingRect()
        proxy.setTransformOriginPoint(rect.center())
        entry = dict(card=card, widget=widget, proxy=proxy,
                     active=None, points_header=points_header, centre=None)
        self.entries[pi] = entry
        return entry

    def update_snapshot(self, snap):
        from qt_ui import MainWindow, KillerRow
        import main_manual as mm
        mode = str(snap.get("mode") or "x01").lower()
        mode = {"around_the_world": "around", "halve_it": "halve"}.get(mode, mode)
        if mode != self.mode:
            self.clear_cards()
            self.mode = mode
        players = sorted(snap.get("players") or [], key=lambda p: int(p.get("index", 0)))[:4]
        order = [int(p.get("index", i)) for i, p in enumerate(players)]
        for pi in list(self.entries):
            if pi not in order:
                entry = self.entries.pop(pi)
                self.scene().removeItem(entry["proxy"])
                entry["proxy"].deleteLater()
        self.order = order
        rules = snap.get("rules") or {}
        for pi, player in zip(order, players):
            entry = self.entries.get(pi) or self._create_entry(pi, mode)
            card = entry["card"]
            color = player.get("color") or "#ffffff"
            name = str(player.get("name") or "")
            finished = (MainWindow._killer_player_finished(player) if mode == "killer"
                        else MainWindow._x01_player_finished(player))
            active = bool(player.get("active")) and not finished
            entry["active"] = active
            if mode == "cricket":
                show_pts = str(rules.get("cricket_mode", "standard")).lower() != "no_score"
                entry["points_header"].setText("PTS" if show_pts else "")
                card.set_row(player.get("label", f"P{pi+1}"), player.get("marks") or {},
                             int(player.get("points") or 0), color, active,
                             show_pts=show_pts, finished=finished, rank=player.get("rank"),
                             animate_border=None,
                             name=name)
            elif mode == "killer":
                card.set_row(player.get("number"), int(player.get("lives") or 0), color, active,
                             is_killer=bool(player.get("is_killer")), finished=finished,
                             act_prefix=mm._killer_activation_prefix(rules),
                             max_lives=max(1, min(KillerRow.MAX_LIVES, int(rules.get("killer_lives", 3) or 3))),
                             name=name)
            else:
                card.set_around_progress(MainWindow._around_card_progress(player) if mode == "around" else None, color)
                card.set_player(player.get("label", f"P{pi+1}"), str(player.get("score_text", "")),
                                color, active, player_index=pi, finished=finished,
                                score_halved=bool(player.get("score_halved")), name=name)
        self._layout_cards()

    def _layout_cards(self):
        width, height = self.viewport().width(), self.viewport().height()
        self.scene().setSceneRect(QRectF(0, 0, width, height))
        n = len(self.order)
        if not n or width < 2 or height < 2:
            return
        cols, rows = (1, 1) if n == 1 else (2, 1) if n == 2 else (2, 2)
        gap = 16
        cell_w = (width - gap * (cols - 1)) / cols
        cell_h = (height - gap * (rows - 1)) / rows
        for index, pi in enumerate(self.order):
            entry = self.entries[pi]
            proxy = entry["proxy"]
            rect = proxy.boundingRect()
            cx = (index % cols) * (cell_w + gap) + cell_w / 2
            cy = (index // cols) * (cell_h + gap) + cell_h / 2
            centre = (cx, cy)
            fit = min((cell_w - 24) / rect.width(),
                      (cell_h - 18) / rect.height())
            scale = max(.05, fit)
            proxy.setPos(cx - rect.width() / 2, cy - rect.height() / 2)
            proxy.setScale(scale)
            entry["centre"] = centre

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.scene() is not None:
            self._layout_cards()
