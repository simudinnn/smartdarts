"""End-to-end manual turn, undo and main-screen rendering regression tests."""
import os
import threading
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

import main_manual as mm
from kiosk_session import KioskSession
from kiosk_settings import KioskSettings
from qt_ui import MainWindow


class FullGameUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _session(self, mode: str) -> KioskSession:
        session = KioskSession.__new__(KioskSession)
        session.kiosk = False
        session.state = {
            "screen": "select_players",
            "selected_game_mode": mode,
            "player_slots": [{"name": "Ana"}, {"name": "Ivan"}, None, None],
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
                "count_up_rounds": 8,
                "count_up_mode": "normal",
            },
        }
        session.ctx = {"camera_mgr": Mock(active=False)}
        session.all_hits = []
        session._vision_lock = threading.RLock()
        return session

    def test_every_mode_survives_turn_switch_undo_and_recommit_without_tv(self):
        settings = KioskSettings(input_mode="manual")
        for mode in ("301", "501", "cricket", "killer", "around", "halve", "count_up"):
            with self.subTest(mode=mode), \
                    patch("kiosk_session.get_settings", return_value=settings), \
                    patch("kiosk_session.record_game"), \
                    patch("main_manual._click_sound"):
                session = self._session(mode)
                session.start_from_player_slots()
                window = MainWindow(session)
                window.timer.stop()
                window.resize(1280, 720)
                window.show()
                window.refresh()
                self.app.processEvents()

                for _ in range(3):
                    session.handle_keypad("hit:miss")
                session.advance_to_next_player()
                window._last_ui_sig = None
                window.refresh()
                self.app.processEvents()

                game = session.ctx["game"]
                self.assertEqual(len(game["players"]), 2)
                self.assertEqual(session.state["current_player_idx"], 1)
                self.assertTrue(game["undo_stack"])
                self.assertTrue(window.undo_btn.isEnabled())

                session.handle_action("undo_turn")
                window._last_ui_sig = None
                window.refresh()
                self.app.processEvents()
                self.assertEqual(session.state["current_player_idx"], 0)
                self.assertTrue(session.ctx["_dart_block_auto_advance"])

                session.confirm_undo_advance_to_next()
                window._last_ui_sig = None
                window.refresh()
                self.app.processEvents()
                self.assertEqual(len(game["players"]), 2)
                self.assertEqual(session.state["current_player_idx"], 1)
                self.assertFalse(session.ctx["_dart_block_auto_advance"])

                if mode == "cricket":
                    self.assertEqual(len(window._cricket_rows_by_pi), 2)
                elif mode == "killer":
                    self.assertEqual(len(window._killer_rows_by_pi), 2)
                    self.assertTrue(all(int(p["lives"]) == 3 for p in game["players"]))
                else:
                    self.assertEqual(len(window._x01_cards_by_pi), 2)

                window.close()
                window.deleteLater()
                self.app.processEvents()

    def test_every_mode_keeps_main_and_tv_scoreboards_during_full_turn(self):
        settings = KioskSettings(input_mode="manual")
        for mode in ("301", "501", "cricket", "killer", "around", "halve", "count_up"):
            with self.subTest(mode=mode), \
                    patch("kiosk_session.get_settings", return_value=settings), \
                    patch("kiosk_session.record_game"), \
                    patch("main_manual._click_sound"):
                session = self._session(mode)
                session.start_from_player_slots()
                main = MainWindow(session)
                main.timer.stop()
                tv = MainWindow(session, spectator=True)
                main.resize(1280, 720)
                tv.resize(1920, 1080)
                main.show()
                tv.show()

                main.refresh()
                tv.render_spectator(session.snapshot_for_ui())
                for _ in range(3):
                    session.handle_keypad("hit:miss")
                    main._last_ui_sig = None
                    main.refresh()
                    tv.render_spectator(session.snapshot_for_ui())
                    self.app.processEvents()

                session.advance_to_next_player()
                main._last_ui_sig = None
                main.refresh()
                tv.render_spectator(session.snapshot_for_ui())
                self.app.processEvents()

                self.assertEqual(len(session.ctx["game"]["players"]), 2)
                self.assertEqual(session.state["current_player_idx"], 1)
                self.assertTrue(main.advance_btn.isEnabled())
                self.assertIsNotNone(tv._tv_cards)
                self.assertEqual(len(tv._tv_cards.entries), 2)
                self.assertTrue(all(entry["proxy"].scale() > 0
                                    for entry in tv._tv_cards.entries.values()))

                main.close()
                tv.close()
                main.deleteLater()
                tv.deleteLater()
                self.app.processEvents()

    def test_auto_mode_turns_render_safely_with_main_and_tv(self):
        settings = KioskSettings(input_mode="auto")
        for mode in ("301", "501", "cricket", "killer", "around", "halve", "count_up"):
            with self.subTest(mode=mode), \
                    patch("kiosk_session.get_settings", return_value=settings), \
                    patch("kiosk_session.record_game"), \
                    patch("main_manual._click_sound"):
                session = self._session(mode)
                session.start_from_player_slots()
                session.start_playing_after_clear_board()
                main = MainWindow(session)
                main.timer.stop()
                tv = MainWindow(session, spectator=True)
                main.show()
                tv.show()
                main.refresh()
                tv.render_spectator(session.snapshot_for_ui())

                for slot in range(3):
                    session.set_turn_hit(slot, mm._make_miss_hit(), source="auto")
                    main._last_ui_sig = None
                    main.refresh()
                    tv.render_spectator(session.snapshot_for_ui())
                session.advance_to_next_player(auto=True, skip_idle_wait=True)
                main._last_ui_sig = None
                main.refresh()
                tv.render_spectator(session.snapshot_for_ui())
                self.app.processEvents()

                self.assertEqual(session.state["current_player_idx"], 1)
                self.assertEqual(len(session.ctx["game"]["players"]), 2)
                self.assertIsNotNone(tv._tv_cards)
                self.assertEqual(len(tv._tv_cards.entries), 2)

                main.close()
                tv.close()
                main.deleteLater()
                tv.deleteLater()
                self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
