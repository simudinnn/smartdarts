"""Run with QT_QPA_PLATFORM=offscreen for display-independent UI checks."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QRect, Qt
from PySide6.QtWidgets import QApplication, QPushButton, QWidget
from spectator_display import external_screen, hdmi1_screen


class Screen:
    def __init__(self, x, y=0, name="HDMI-1"):
        self._name = name
        self.rect = QRect(x, y, 1920, 1080)

    def name(self):
        return self._name

    def geometry(self):
        return self.rect


class SpectatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_monitor_selection(self):
        touch, tv, mirror = Screen(0), Screen(1920), Screen(0)
        self.assertIsNone(external_screen([touch], touch))
        self.assertIs(external_screen([tv, touch], touch), tv)
        self.assertIsNone(external_screen([tv], touch))
        self.assertIsNone(external_screen([touch, mirror], touch))
        self.assertIs(external_screen([touch, mirror], touch, wayland=True), mirror)
        self.assertIsNone(external_screen([touch], touch, wayland=True))
        self.assertIsNone(external_screen([tv], touch, wayland=True))
        self.assertIs(external_screen([touch, Screen(-1920)], touch).__class__, Screen)

    def test_disconnected_tv_does_not_render_snapshots(self):
        from spectator_display import SpectatorDisplay
        display = SpectatorDisplay.__new__(SpectatorDisplay)
        display.window = Mock()
        display.target = None
        display.update_snapshot({"screen": "playing"})
        display.window.render_spectator.assert_not_called()

    def test_shared_read_only_rendering(self):
        from qt_ui import MainWindow, PlayerCard
        from unittest.mock import Mock
        session = SimpleNamespace(kiosk=False, state={}, ctx={}, screen="standby",
                                  player_slots=lambda: [], is_in_game_calibrating=lambda: False,
                                  snapshot_for_ui=Mock(side_effect=AssertionError("second snapshot")))
        window = MainWindow(session, spectator=True)
        window.resize(1280, 720)
        window.show()
        self.assertFalse(window.timer.isActive())
        self.assertTrue(window.windowFlags() & Qt.WindowDoesNotAcceptFocus)
        players = [dict(index=i, label=f"P{i + 1}", name=f"Igrač {i + 1}", score_text="301",
                        active=i == 0, color="#00ed84", marks={20: 3, 19: 2}) for i in range(4)]
        for mode in ["x01", "cricket", "killer", "around", "halve", "count_up"]:
            snap = dict(screen="playing", players=players, mode=mode, game_title=mode.upper(),
                        turn_hit_tags=["T20", "D20", "MISS"], manual_mode=True,
                        editing_hit_slot=0, league_qr_payload="private")
            window.render_spectator(snap)
            self.app.processEvents()
            self.assertTrue(window.hits_wrap.isVisible())
            self.assertFalse(window.keypad_wrap.isVisible())
            self.assertFalse(window.footer_block.isVisible())
            self.assertTrue(all(not b.isVisible() or b in window.hit_btns
                                for b in window.findChildren(QPushButton)))
            self.assertTrue(snap["manual_mode"])
            self.assertEqual(snap["league_qr_payload"], "private")
            self.assertFalse(window.grab().isNull())
        for screen in ["standby", "select_game", "select_players", "settings"]:
            window.render_spectator(dict(screen=screen))
            self.assertIs(window.stack.currentWidget(), window.pages["standby"])
            self.assertFalse(window.qr_frame.isVisible())
            self.assertFalse(window.standby_start_btn.isVisible())
        window.render_spectator(dict(screen="playing", mode="x01", players=players,
                                     winner=0, leaderboard=[(1,0),(2,1),(3,2),(4,3)],
                                     league_qr_payload="private"))
        self.assertFalse(window.left_w.isVisible())
        self.assertFalse(window.footer_block.isVisible())
        session.snapshot_for_ui.assert_not_called()
        window.close()

    def test_scaled_tv_uses_shared_widgets(self):
        from spectator_display import SpectatorWindow
        from qt_ui import MainWindow
        session = SimpleNamespace(kiosk=False, state={}, ctx={}, screen="standby",
                                  player_slots=lambda: [], is_in_game_calibrating=lambda: False)
        window = SpectatorWindow(session)
        self.assertIsInstance(window.content, MainWindow)
        self.assertFalse(window.isInteractive())
        for width, height in [(1280, 720), (1920, 1080)]:
            window.resize(width, height)
            window.show()
            self.app.processEvents()
            self.assertEqual(window.content.width(), 1920)
            self.assertEqual(window.content.height(), 1080)
            self.assertFalse(window.grab().isNull())
        window.close()

    def test_tv_card_grid_is_stable_and_active_only_borders(self):
        from spectator_cards import SpectatorCards
        from qt_ui import CricketRow, KillerRow, PlayerCard

        players = [dict(index=i, label=f"P{i + 1}", name=f"Igrač {i + 1}",
                        score_text=str(501 - i * 40), active=i == 0,
                        color="#00ed84", lives=3, number=20 - i,
                        marks={15: i, 20: 3}, points=i * 20,
                        next_target_idx=i) for i in range(4)]
        grid = SpectatorCards()
        grid.resize(1800, 440)
        grid.show()
        expected = {"x01": PlayerCard, "halve": PlayerCard, "count_up": PlayerCard,
                    "around": PlayerCard, "cricket": CricketRow,
                    "killer": KillerRow}
        for mode, card_type in expected.items():
            snap = dict(mode=mode, players=players,
                        rules={"killer_lives": 3, "cricket_mode": "standard"})
            grid.update_snapshot(snap)
            self.app.processEvents()
            self.assertTrue(all(isinstance(entry["card"], card_type)
                                for entry in grid.entries.values()))
            before = {pi: entry["centre"] for pi, entry in grid.entries.items()}
            before_scales = {pi: entry["proxy"].scale()
                             for pi, entry in grid.entries.items()}
            changed = [dict(player, active=player["index"] == 2)
                       for player in players]
            grid.update_snapshot(dict(snap, players=changed))
            after = {pi: entry["centre"] for pi, entry in grid.entries.items()}
            self.assertEqual(before, after)
            self.assertEqual(before_scales,
                             {pi: entry["proxy"].scale()
                              for pi, entry in grid.entries.items()})
            self.assertTrue(all(entry["card"].property("spectatorCard")
                                for entry in grid.entries.values()))
            self.assertTrue(all(entry["card"].icon_lab.property("spectatorInitials")
                                for entry in grid.entries.values()))
            self.assertEqual(len({round(c[0]) for c in after.values()}), 2)
            self.assertEqual(len({round(c[1]) for c in after.values()}), 2)
        grid.clear_cards()
        grid.close()

    def test_one_two_three_player_grid_shapes(self):
        from spectator_cards import SpectatorCards

        grid = SpectatorCards()
        grid.resize(1800, 440)
        grid.show()
        base = [dict(index=i, label=f"P{i + 1}", name=f"P{i + 1}",
                     score_text="301", active=i == 0, color="#ff3038")
                for i in range(4)]
        for count, columns, rows in [(1, 1, 1), (2, 2, 1),
                                     (3, 2, 2), (4, 2, 2)]:
            grid.update_snapshot(dict(mode="x01", players=base[:count]))
            centres = [entry["centre"] for entry in grid.entries.values()]
            self.assertEqual(len({round(c[0]) for c in centres}), columns)
            self.assertEqual(len({round(c[1]) for c in centres}), rows)
        grid.clear_cards()
        grid.close()

    def test_hdmi1_is_selected_regardless_of_order(self):
        for name in ("HDMI-1", "HDMI-A-1", "HDMI1"):
            touch, tv = Screen(0, name=name), Screen(1920, name="HDMI-A-2")
            self.assertIs(hdmi1_screen([tv, touch]), touch)
            self.assertIs(hdmi1_screen([touch, tv]), touch)
            self.assertIsNone(hdmi1_screen([tv]))


if __name__ == "__main__":
    unittest.main()
