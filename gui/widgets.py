"""Вспомогательные виджеты и диалог ручного ввода градиента."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QTextBrowser,
    QVBoxLayout,
)

from methods.types import FieldSpec


class InputError(Exception):
    """Ошибка ввода на странице данных."""

    def __init__(self, field: str, message: str):
        super().__init__(message)
        self.field = field
        self.message = message


def parse_float(text: str, label: str) -> float:
    """Разбирает число из поля ввода, поддерживая десятичную запятую."""

    value = text.strip().replace(",", ".")
    if not value:
        raise InputError(label, f"Поле «{label}» не заполнено.")
    try:
        return float(value)
    except ValueError as exc:
        raise InputError(
            label,
            f"Поле «{label}» должно содержать число. Получено: «{text.strip()}».",
        ) from exc


def format_default(value) -> str:
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


class FieldEditor:
    """Редактор одного специфического поля метода."""

    def __init__(self, spec: FieldSpec):
        self.spec = spec
        self.checkbox: QCheckBox | None = None
        self.line_edit: QLineEdit | None = None

        if spec.type == "bool":
            self.checkbox = QCheckBox()
            self.checkbox.setChecked(bool(spec.default))
            self.widget = self.checkbox
        else:
            self.line_edit = QLineEdit(format_default(spec.default))
            self.line_edit.setPlaceholderText(format_default(spec.default))
            self.widget = self.line_edit

        if spec.description:
            self.widget.setToolTip(spec.description)

    def value(self):
        if self.spec.type == "bool":
            return bool(self.checkbox.isChecked()) if self.checkbox else False

        assert self.line_edit is not None
        number = parse_float(self.line_edit.text(), self.spec.label)
        if self.spec.type == "int":
            if not float(number).is_integer():
                raise InputError(
                    self.spec.label,
                    f"Поле «{self.spec.label}» должно содержать целое число.",
                )
            return int(number)
        return number

    def set_value(self, value) -> None:
        if self.spec.type == "bool":
            if self.checkbox is not None:
                self.checkbox.setChecked(bool(value))
            return
        if self.line_edit is not None:
            self.line_edit.setText(format_default(value))


class GradientDialog(QDialog):
    """Диалог ручного ввода градиента с подробной подсказкой."""

    def __init__(self, parent=None, initial_text: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Ручной ввод градиента")
        self.setMinimumSize(780, 640)

        layout = QVBoxLayout(self)

        help_view = QTextBrowser()
        help_view.setOpenExternalLinks(False)
        help_view.setHtml(
            """
            <h3>Не удалось построить градиент автоматически.</h3>
            <p>Введите градиент вручную в формате:</p>
            <p><code>[∂f/∂x, ∂f/∂y]</code></p>

            <h4>Что такое градиент</h4>
            <p>Для функции двух переменных <code>f(x, y)</code> градиент — это
            вектор из двух частных производных:</p>
            <ul>
              <li><b>∂f/∂x</b> — производная по <code>x</code>, при этом
                  <code>y</code> считается константой;</li>
              <li><b>∂f/∂y</b> — производная по <code>y</code>, при этом
                  <code>x</code> считается константой.</li>
            </ul>

            <h4>Пример</h4>
            <p>Для <code>f(x, y) = sin(x) + cos(y) + 0.1*(x**2 + y**2)</code>:</p>
            <ul>
              <li><b>∂f/∂x = cos(x) + 0.2*x</b> — дифференцируем по x:
                  <code>sin(x)</code> → <code>cos(x)</code>,
                  <code>0.1*x**2</code> → <code>0.2*x</code>,
                  <code>cos(y)</code> даёт 0;</li>
              <li><b>∂f/∂y = -sin(y) + 0.2*y</b> — дифференцируем по y:
                  <code>cos(y)</code> → <code>-sin(y)</code>,
                  <code>0.1*y**2</code> → <code>0.2*y</code>,
                  <code>sin(x)</code> даёт 0;</li>
              <li>В поле ввода пишем:
                  <code>[cos(x) + 0.2*x, -sin(y) + 0.2*y]</code></li>
            </ul>

            <h4>Синтаксис и пробелы</h4>
            <ul>
              <li>Степень — <b>**</b> (не <code>^</code>). Умножение — явное
                  (<code>2*x</code>). Функции: <code>sin, cos, tan, asin, acos,
                  atan, exp, log, sqrt, abs</code>. Константы: <code>pi</code>,
                  <code>E</code>.</li>
              <li>Пробелы внутри выражения допускаются в любом количестве.</li>
              <li>Квадратные скобки вокруг всего списка — <b>обязательны</b>,
                  внутри — два выражения через запятую.</li>
              <li>Полное руководство — кнопка «Синтаксис» в верхней панели.</li>
            </ul>
            """
        )
        layout.addWidget(help_view, 1)

        label = QLabel("Градиент:")
        layout.addWidget(label)

        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText("[cos(x) + 0.2*x, -sin(y) + 0.2*y]")
        self.editor.setPlainText(initial_text)
        self.editor.setMinimumHeight(80)
        layout.addWidget(self.editor)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.editor.setFocus()

    def text(self) -> str:
        return self.editor.toPlainText().strip()
