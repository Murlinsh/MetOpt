"""Страница 3: выбор режима вывода."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from .styles import title_style


class ModePage(QWidget):
    back_requested = pyqtSignal()
    next_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        title = QLabel("Выбор режима")
        title.setStyleSheet(title_style())
        layout.addWidget(title)

        question = QLabel("Как вы хотите получить результат?")
        layout.addWidget(question)

        self.answer_radio = QRadioButton("Готовый ответ")
        self.explain_radio = QRadioButton("Объяснение по итерациям")
        self.explain_radio.setChecked(True)

        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.answer_radio)
        self.button_group.addButton(self.explain_radio)

        layout.addWidget(self.answer_radio)
        layout.addWidget(self.explain_radio)

        explanation = QLabel(
            "«Готовый ответ» показывает только точку минимума, значение функции "
            "и число итераций. «Объяснение по итерациям» выводит пошаговый лог "
            "со всеми промежуточными величинами."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        layout.addStretch(1)

        buttons = QHBoxLayout()
        self.back_button = QPushButton("Назад")
        self.next_button = QPushButton("Далее")
        buttons.addWidget(self.back_button)
        buttons.addStretch(1)
        buttons.addWidget(self.next_button)
        layout.addLayout(buttons)

        self.back_button.clicked.connect(self.back_requested)
        self.next_button.clicked.connect(self.next_requested)

    def prepare(self, mode: str | None) -> None:
        if mode == "answer":
            self.answer_radio.setChecked(True)
        else:
            self.explain_radio.setChecked(True)

    def selected_mode(self) -> str:
        return "answer" if self.answer_radio.isChecked() else "explain"
