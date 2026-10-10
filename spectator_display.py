"""Read-only TV display. Uses the main UI snapshot, never a second session."""
from __future__ import annotations

from PySide6.QtCore import Qt, QObject, QTimer, QEvent
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QApplication, QGraphicsView, QGraphicsScene


def hdmi1_screen(screens):
    """HDMI-1 / HDMI-A-1 / HDMI1: fixed control output, never list order."""
    for screen in screens:
        name = screen.name().upper().replace("-", "").replace("_", "")
        if name in ("HDMI1", "HDMIA1"):
            return screen
    return None


def control_screen_name():
    app = QApplication.instance()
    screen = hdmi1_screen(app.screens()) if app else None
    return screen.name() if screen else ""


def external_screen(screens, control, *, wayland=False):
    """Wayland outputs are selected by identity, not desktop coordinates."""
    if control not in screens:
        return None
    return next((s for s in screens if s is not control
                 and (wayland or not s.geometry().intersects(control.geometry()))), None)


class SpectatorWindow(QGraphicsView):
    """Scale the shared 1080p kiosk layout as one surface on any TV."""
    def __init__(self, session):
        super().__init__()
        from qt_ui import MainWindow

        self.setWindowTitle("SmartDarts — TV")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint
                            | Qt.WindowDoesNotAcceptFocus | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setCursor(Qt.BlankCursor)
        self.setFocusPolicy(Qt.NoFocus)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("QGraphicsView { background: black; border: none; }")
        # The transform is static; avoid full-screen smooth resampling on every
        # child animation frame on the Raspberry Pi raster backend.
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setViewportUpdateMode(QGraphicsView.MinimalViewportUpdate)
        self.setOptimizationFlags(
            QGraphicsView.DontSavePainterState
            | QGraphicsView.DontAdjustForAntialiasing
        )
        scene = QGraphicsScene(self)
        self.setScene(scene)
        self.content = MainWindow(session, spectator=True)
        self.content.setWindowFlags(Qt.Widget)
        self.content.resize(1920, 1080)
        scene.addWidget(self.content)
        scene.setSceneRect(0, 0, 1920, 1080)
        self.setInteractive(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.scene() is not None:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def render_spectator(self, snapshot):
        self.content.render_spectator(snapshot)


class SpectatorDisplay(QObject):
    """HDMI1 owns controls. A different output owns the read-only TV window."""
    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.app = QApplication.instance()
        self.wayland = self.app.platformName().startswith("wayland")
        # Do not even build/render the TV MainWindow on a single-monitor kiosk.
        # This keeps the original main-screen path completely isolated.
        self.window = None
        self.target = None
        self._target_size = None
        self._last_status = None
        self._placing = False
        main.installEventFilter(self)
        self.app.screenAdded.connect(self.sync)
        self.app.screenRemoved.connect(self.screen_removed)
        self.app.aboutToQuit.connect(self._drop_window)
        main.destroyed.connect(self._drop_window)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.sync)
        self.timer.start()
        self.sync()

    def eventFilter(self, watched, event):
        if watched is self.main and event.type() in (QEvent.Hide, QEvent.Close):
            if self.window is not None:
                self.window.hide()
        return super().eventFilter(watched, event)

    def screen_removed(self, screen):
        self._drop_window()
        self.target = None
        QTimer.singleShot(0, self.sync)

    def _ensure_window(self):
        if self.window is None:
            self.window = SpectatorWindow(self.main.session)
        return self.window

    def _drop_window(self, *_):
        window = self.window
        self.window = None
        if window is not None:
            window.close()
            window.deleteLater()

    def sync(self, *_):
        if self._placing:
            return
        screens = self.app.screens()
        self.control = hdmi1_screen(screens)
        status = tuple((s.name(), s.geometry().getRect()) for s in screens)
        if status != self._last_status:
            print(f"[spectator] fixed HDMI1 controls; screens={status}", flush=True)
            self._last_status = status
        if self.control is None or not self.main.session.kiosk:
            self._drop_window()
            self.target = None
            return
        self.main._control_screen = self.control
        self.main.winId()
        if self.main.windowHandle().screen() is not self.control:
            if self.window is not None:
                self.window.hide()
            self._placing = True
            self.main._locking_geo = True
            try:
                self.main.hide()
                self.main.setWindowState(Qt.WindowNoState)
                self.main.windowHandle().setScreen(self.control)
                self.main.setMinimumSize(0, 0)
                self.main.setMaximumSize(16777215, 16777215)
                if self.wayland:
                    self.main.resize(self.control.size())
                else:
                    self.main.setGeometry(self.control.geometry())
                self.main.showFullScreen()
            finally:
                self.main._locking_geo = False
                self._placing = False
        target = external_screen(screens, self.control, wayland=self.wayland)
        if target is None or not self.main.isVisible():
            self._drop_window()
            self.target = None
            return
        window = self._ensure_window()
        geo = target.geometry()
        if (target is not self.target or self._target_size != geo.size()
                or not window.isVisible()):
            window.hide()
            window.setWindowState(Qt.WindowNoState)
            window.winId()
            window.windowHandle().setScreen(target)
            if self.wayland:
                window.resize(geo.size())
            else:
                window.setGeometry(geo)
            self.target = target
            self._target_size = geo.size()
            window.showFullScreen()
            print(f"[spectator] controls={self.control.name()!r}; TV={target.name()!r}", flush=True)

    def update_snapshot(self, snap):
        # A disconnected TV must do zero UI work and cannot affect the main UI.
        if self.window is not None and self.target is not None:
            # The TV is read-only. A rendering problem there must never abort
            # the button handler or navigation on the control touchscreen.
            try:
                self.window.render_spectator(snap)
            except Exception as exc:
                print(f"[spectator] render error (ignored): {exc}", flush=True)
