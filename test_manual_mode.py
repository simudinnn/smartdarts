"""Camera-free game startup and locked manual input regression tests."""
import threading
import unittest
from unittest.mock import Mock, patch

import main_manual as mm
from kiosk_session import KioskSession
from kiosk_settings import KioskSettings


class ManualModeTests(unittest.TestCase):
    def setUp(self):
        self.settings = KioskSettings(input_mode="manual")
        self.settings_patch = patch("kiosk_session.get_settings", return_value=self.settings)
        self.settings_patch.start()
        self.addCleanup(self.settings_patch.stop)
        self.session = KioskSession.__new__(KioskSession)
        self.session.kiosk = False
        self.session.state = dict(screen="select_players", selected_game_mode="x01",
                                  player_slots=[{"name": "Ana"}, {"name": "Ivan"}, None, None],
                                  rules_pending={})
        self.camera = Mock(active=False)
        self.session.ctx = {"camera_mgr": self.camera}
        self.session.all_hits = []
        self.session._vision_lock = threading.RLock()

    def start_game(self):
        with patch("kiosk_session.record_game"), patch("main_manual._click_sound"):
            self.session.start_from_player_slots()

    def test_manual_starts_without_cameras_and_opens_keypad(self):
        self.start_game()
        self.assertEqual(self.session.screen, "playing")
        self.assertEqual(self.session.ctx["editing_hit_slot"], 0)
        self.session.ensure_cameras()
        self.camera.start.assert_not_called()
        self.assertTrue(self.session.is_manual_mode())

    def test_count_up_scores_round_limit_and_reverse_winner(self):
        high = mm._make_number_hit(20, 1)
        low = mm._make_number_hit(5, 1)
        for game_mode, expected_winner in (("normal", 0), ("reverse", 1)):
            game = mm._create_game(
                "count_up", 2,
                {"count_up_rounds": 8, "count_up_mode": game_mode},
            )
            for _round in range(8):
                for player in (0, 1):
                    hit = high if player == 0 else low
                    mm._apply_turn_score(game, player, [hit, hit, hit])
                    mm._advance_round_if_needed(game, player, 1 - player)
            self.assertEqual([p["score"] for p in game["players"]], [480, 120])
            self.assertEqual(game["winner_player_idx"], expected_winner)
            self.assertEqual(game["placements"][0], expected_winner)

    def test_manual_is_locked_for_entire_game(self):
        self.start_game()
        self.settings.input_mode = "auto"
        with patch("kiosk_session.update_settings") as save:
            self.session.toggle_manual_mode()
            save.assert_not_called()
        self.assertTrue(self.session.is_manual_mode())
        self.assertIsNone(self.session._auto_detect_vision(undo_mode=True))

    def test_auto_keeps_clear_board_and_camera_start(self):
        self.settings.input_mode = "auto"
        self.start_game()
        self.assertEqual(self.session.screen, "clear_board")
        self.session.ensure_cameras()
        self.camera.start.assert_called_once()
        self.settings.input_mode = "manual"
        self.assertFalse(self.session.is_manual_only())

    def test_repeated_clear_walks_back_through_manual_hits(self):
        self.start_game()
        game = self.session.ctx["game"]
        hits = [mm._make_number_hit(20, 3), mm._make_number_hit(19, 2),
                mm._make_bull_hit(True)]
        game["turn_hits"] = list(hits)
        self.session.ctx["_hit_sources"] = ["manual", "manual", "manual"]
        self.session.ctx["editing_hit_slot"] = 2
        with patch("main_manual._click_sound"):
            self.session.handle_keypad("clear_one")
            self.assertEqual(game["turn_hits"], [hits[0], hits[1], None])
            self.assertEqual(self.session.ctx["editing_hit_slot"], 2)
            self.session.handle_keypad("clear_one")
            self.assertEqual(game["turn_hits"], [hits[0], None, None])
            self.assertEqual(self.session.ctx["editing_hit_slot"], 1)
            self.session.handle_keypad("clear_one")
            self.assertEqual(game["turn_hits"], [None, None, None])
            self.assertEqual(self.session.ctx["editing_hit_slot"], 0)

    def test_manual_clear_removes_newest_hit_without_selected_slot(self):
        self.start_game()
        game = self.session.ctx["game"]
        hits = [mm._make_number_hit(20, 1), mm._make_number_hit(18, 1), None]
        game["turn_hits"] = list(hits)
        self.session.ctx["editing_hit_slot"] = None
        with patch("main_manual._click_sound"):
            self.session.handle_keypad("clear_one")
        self.assertEqual(game["turn_hits"], [hits[0], None, None])
        self.assertEqual(self.session.ctx["editing_hit_slot"], 1)

    def test_killer_hit_clear_and_advance_do_not_end_game(self):
        self.session.state["selected_game_mode"] = "killer"
        self.session.state["rules_pending"] = {
            "killer_lives": 3,
            "killer_assign": "random",
            "killer_activation": "double",
        }
        self.start_game()
        game = self.session.ctx["game"]
        starting_players = [dict(player) for player in game["players"]]

        with patch("main_manual._click_sound"):
            self.session.handle_keypad("num:1")
            self.assertIsNone(game["winner_player_idx"])
            self.assertTrue(all(int(p["lives"]) == 3 for p in game["players"]))

            self.session.handle_keypad("clear_one")
            self.assertIsNone(game["winner_player_idx"])
            self.assertEqual(game["turn_hits"], [None, None, None])

            self.session.advance_to_next_player()
            self.assertIsNone(game["winner_player_idx"])
            self.assertEqual([p["lives"] for p in game["players"]],
                             [p["lives"] for p in starting_players])

    def test_killer_winner_requires_confirmed_elimination(self):
        game = mm._create_game("killer", 2, {
            "killer_lives": 3,
            "killer_assign": "random",
            "killer_activation": "double",
        })
        game["players"][1].pop("lives")
        mm._killer_set_winner_if_any(game)
        self.assertIsNone(game["winner_player_idx"])

    def test_killer_activation_is_visible_immediately_mid_turn(self):
        self.session.state["selected_game_mode"] = "killer"
        self.session.state["rules_pending"] = {
            "killer_lives": 3,
            "killer_assign": "random",
            "killer_activation": "double",
        }
        self.start_game()
        game = self.session.ctx["game"]
        own_number = int(game["players"][0]["number"])
        self.session.ctx["input_multiplier"] = 2
        with patch("main_manual._click_sound"):
            self.session.handle_keypad(f"num:{own_number}")

        self.assertTrue(game["players"][0]["is_killer"])
        snapshot = self.session.snapshot_for_ui()
        self.assertTrue(snapshot["players"][0]["is_killer"])
        self.assertEqual(snapshot["turn_hit_tags"][0], f"D{own_number}")

    def test_other_game_previews_keep_all_players_and_expected_scores(self):
        cases = [
            ("301", mm._make_number_hit(20, 1), "remaining", 281),
            ("501", mm._make_number_hit(20, 1), "remaining", 481),
            ("cricket", mm._make_number_hit(20, 1), "points", 0),
            ("around", mm._make_number_hit(1, 1), "next_target_idx", 1),
            ("halve", mm._make_number_hit(20, 1), "score", 20),
        ]
        for mode, hit, key, expected in cases:
            with self.subTest(mode=mode):
                game = mm._create_game(mode, 2, {})
                preview = mm._preview_players_after_turn(game, 0, [hit, None, None])
                self.assertEqual(len(preview), 2)
                self.assertEqual(preview[0].get(key), expected)
                self.assertIsNone(game["winner_player_idx"])
                self.assertEqual(len(game["players"]), 2)


if __name__ == "__main__":
    unittest.main()
