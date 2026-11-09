"""Точка входа приложения Optimization Toolkit."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.styles import apply_application_font


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Optimization Toolkit")
    apply_application_font(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
