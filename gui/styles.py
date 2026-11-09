"""Единые размеры шрифтов для всего приложения."""

from __future__ import annotations

from PyQt6.QtGui import QFont, QFontDatabase
from PyQt6.QtWidgets import QApplication

# Базовый шрифт приложения увеличивается на эту величину относительно системного.
BASE_FONT_DELTA = 5

TITLE_FONT_PX = 24
SUMMARY_FONT_PX = 20
LOG_FONT_DELTA = 5
DOCUMENT_FONT_POINT_SIZE = 16
DIALOG_FONT_POINT_SIZE = 15


def apply_application_font(app: QApplication) -> None:
    """Увеличивает базовый шрифт приложения, сохраняя системный шрифт."""

    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
    point_size = font.pointSize()
    if point_size <= 0:
        font = app.font()
        point_size = font.pointSize()
    if point_size <= 0:
        point_size = 10
    font.setPointSize(point_size + BASE_FONT_DELTA)
    app.setFont(font)


def title_style() -> str:
    return f"font-size: {TITLE_FONT_PX}px; font-weight: bold;"


def summary_style() -> str:
    return f"font-size: {SUMMARY_FONT_PX}px; font-weight: bold;"


def enlarged_fixed_font() -> QFont:
    """Моноширинный шрифт для логов и результатов."""

    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    point_size = font.pointSize()
    if point_size <= 0:
        point_size = 10
    font.setPointSize(point_size + LOG_FONT_DELTA)
    return font


def document_font() -> QFont:
    """Шрифт для встроенного просмотра документов."""

    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
    font.setPointSize(DOCUMENT_FONT_POINT_SIZE)
    return font
