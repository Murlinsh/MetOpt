"""Встроенный просмотр документов с крупным шрифтом."""

from __future__ import annotations

from PyQt6.QtGui import QTextCharFormat
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QTextBrowser,
    QVBoxLayout,
)

from .styles import document_font


class HelpDialog(QDialog):
    """Показывает Markdown-документ внутри приложения."""

    def __init__(
        self,
        title: str,
        markdown_text: str,
        parent=None,
        search_text: str | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(980, 760)

        layout = QVBoxLayout(self)

        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(True)
        self.view.setOpenLinks(True)
        self.view.setFont(document_font())
        self.view.document().setDefaultFont(document_font())
        self.view.setMarkdown(markdown_text)
        self._anchor: str | None = None
        if search_text:
            cursor = self.view.document().find(search_text)
            if not cursor.isNull():
                anchor_format = QTextCharFormat()
                anchor_format.setAnchor(True)
                anchor_format.setAnchorNames(["section_start"])
                cursor.mergeCharFormat(anchor_format)
                self._anchor = "section_start"
        layout.addWidget(self.view, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_button = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close_button is not None:
            close_button.setText("Закрыть")
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def showEvent(self, event) -> None:  # noqa: N802 - имя из Qt API
        super().showEvent(event)
        if self._anchor:
            self.view.scrollToAnchor(self._anchor)
