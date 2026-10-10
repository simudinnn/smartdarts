"""Wi-Fi kiosk modal parsing and UI regression tests."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QListWidgetItem

from qt_ui import MainWindow


class WifiUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_nmcli_parser_handles_escaped_ssids_and_deduplicates(self):
        output = (
            r":Cafe\:Guest:51:WPA2" "\n"
            r"*:Home\\Net:72:WPA2" "\n"
            r":Home\\Net:90:WPA2" "\n"
            r":Open:34:--" "\n"
        )
        rows = MainWindow._wifi_parse_scan(output)
        self.assertEqual([row["ssid"] for row in rows], ["Home\\Net", "Cafe:Guest", "Open"])
        self.assertTrue(rows[0]["active"])
        self.assertEqual(rows[0]["signal"], 72)

    def test_wifi_modal_opens_inside_main_window(self):
        session = SimpleNamespace(
            kiosk=False,
            state={},
            ctx={},
            screen="settings",
            player_slots=lambda: [],
            is_in_game_calibrating=lambda: False,
            snapshot_for_ui=lambda: {"screen": "settings", "settings": {}},
        )
        window = MainWindow(session)
        window.timer.stop()
        window.resize(1280, 720)
        window.show()
        with patch.object(window, "_wifi_scan") as scan:
            window._show_wifi_overlay()
            self.app.processEvents()
            self.assertTrue(window.wifi_overlay.isVisible())
            self.assertFalse(window.wifi_keyboard_overlay.isVisible())
            self.assertEqual(window.wifi_password_btn.text(), "UPIŠI ZAPORKU")
            self.assertFalse(window.wifi_password_btn.isVisible())
            item = QListWidgetItem("Test mreža")
            item.setData(Qt.UserRole, {
                "ssid": "Test mreža", "active": False, "signal": 80, "security": "WPA2"
            })
            window.wifi_list.addItem(item)
            window.wifi_list.setCurrentItem(item)
            self.app.processEvents()
            self.assertTrue(window.wifi_password_btn.isVisible())
            window._show_wifi_keyboard()
            self.app.processEvents()
            self.assertTrue(window.wifi_keyboard_overlay.isVisible())
            self.assertIs(window.wifi_keyboard_stack.currentWidget(),
                          window.wifi_keyboard_alpha)
            window._wifi_toggle_keyboard()
            self.assertIs(window.wifi_keyboard_stack.currentWidget(),
                          window.wifi_keyboard_symbols)
            self.assertEqual(window.wifi_mode_btn.text(), "ABC")
            window._wifi_toggle_keyboard()
            self.assertEqual(window.wifi_mode_btn.text(), "123#?")
            window._wifi_keyboard_append("Test 1!")
            self.assertEqual(window.wifi_keyboard_display.text(), "●" * 7)
            window._wifi_toggle_password_visibility()
            self.assertEqual(window.wifi_keyboard_display.text(), "Test 1!")
            self.assertEqual(window.wifi_password_visibility_btn.text(), "SAKRIJ")
            window._wifi_keyboard_accept()
            self.assertEqual(window._wifi_password_value, "Test 1!")
            self.assertFalse(window.wifi_keyboard_overlay.isVisible())
            self.assertTrue(window.wifi_password_btn.text().startswith("ZAPORKA"))

            open_item = QListWidgetItem("Otvorena mreža")
            open_item.setData(Qt.UserRole, {
                "ssid": "Otvorena mreža", "active": False, "signal": 70, "security": "--"
            })
            window.wifi_list.addItem(open_item)
            window.wifi_list.setCurrentItem(open_item)
            self.app.processEvents()
            self.assertFalse(window.wifi_password_btn.isVisible())
            self.assertEqual(window._wifi_password_value, "")
            scan.assert_called_once()
        window._hide_wifi_overlay()
        self.assertFalse(window.wifi_overlay.isVisible())
        window.close()


if __name__ == "__main__":
    unittest.main()
