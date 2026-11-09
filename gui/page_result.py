"""Страница 4: результат расчёта."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .styles import enlarged_fixed_font, summary_style, title_style


class ResultPage(QWidget):
    same_method_requested = pyqtSignal()
    change_method_requested = pyqtSignal()
    exit_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        title = QLabel("Результат")
        title.setStyleSheet(title_style())
        layout.addWidget(title)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(summary_style())
        layout.addWidget(self.summary_label)

        self.warning_label = QLabel("")
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: #a33;")
        self.warning_label.hide()
        layout.addWidget(self.warning_label)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.log_edit.setPlaceholderText(
            "В режиме «Готовый ответ» пошаговый лог не выводится."
        )
        self.log_edit.setFont(enlarged_fixed_font())
        layout.addWidget(self.log_edit, 1)

        self.file_label = QLabel("")
        self.file_label.setWordWrap(True)
        self.file_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.file_label)

        buttons = QHBoxLayout()
        self.same_method_button = QPushButton("Тот же метод, новая задача")
        self.change_method_button = QPushButton("Сменить метод")
        self.exit_button = QPushButton("Выход")
        buttons.addWidget(self.same_method_button)
        buttons.addWidget(self.change_method_button)
        buttons.addStretch(1)
        buttons.addWidget(self.exit_button)
        layout.addLayout(buttons)

        self.same_method_button.clicked.connect(self.same_method_requested)
        self.change_method_button.clicked.connect(self.change_method_requested)
        self.exit_button.clicked.connect(self.exit_requested)

    def show_result(self, result, log_text: str, file_path: str) -> None:
        names = ("x", "y", "z", "w")
        coordinates = ", ".join(
            f"{names[index] if index < len(names) else f'x{index}'} = {value:.6f}"
            for index, value in enumerate(result.x_min)
        )
        self.summary_label.setText(
            f"Минимум: {coordinates}, f = {result.f_min:.6f}, "
            f"итераций {result.iterations}"
        )

        if result.converged:
            self.warning_label.hide()
        else:
            self.warning_label.setText(
                getattr(result, "message", "")
                or "Достигнут предел итераций: сходимость не подтверждена."
            )
            self.warning_label.show()

        self.log_edit.setPlainText(log_text)
        if file_path.startswith("не удалось"):
            self.file_label.setText(f"Файл не сохранён: {file_path}")
        else:
            self.file_label.setText(f"Файл сохранён: {file_path}")
