"""Страница 2: ввод функции, начальной точки и параметров метода."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from methods import registry
from .widgets import FieldEditor, InputError, format_default, parse_float


class InputPage(QWidget):
    back_requested = pyqtSignal()
    next_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.spec = None
        self.field_editors: dict[str, FieldEditor] = {}

        layout = QVBoxLayout(self)

        self.title_label = QLabel("Ввод данных")
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(self.title_label)

        form = QFormLayout()
        self.f_edit = QLineEdit()
        self.f_edit.setPlaceholderText("Например: sin(x) + cos(y) + 0.1*(x**2 + y**2)")
        self.x0_edit = QLineEdit()
        self.x0_edit.setPlaceholderText("Например: 1.0")
        self.y0_edit = QLineEdit()
        self.y0_edit.setPlaceholderText("Например: 1.0")
        self.epsilon_edit = QLineEdit("1e-4")
        self.epsilon_edit.setPlaceholderText("Например: 1e-4")

        form.addRow("Функция f(x, y):", self.f_edit)
        form.addRow("x₀:", self.x0_edit)
        form.addRow("y₀:", self.y0_edit)
        form.addRow("Epsilon:", self.epsilon_edit)
        layout.addLayout(form)

        hint = QLabel(
            "Степень записывается как ** (не ^), умножение — явное (2*x). "
            "Подробная шпаргалка — кнопка «Синтаксис» в верхней панели."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.spec_group = QGroupBox("Параметры метода")
        self.spec_form = QFormLayout(self.spec_group)
        layout.addWidget(self.spec_group)

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

    def prepare(self, spec, last_input: dict) -> None:
        self.spec = spec
        self.title_label.setText(f"Ввод данных — {registry.display_name(spec)}")

        common = last_input.get("common", {}) if isinstance(last_input, dict) else {}
        if not isinstance(common, dict):
            common = {}

        self.f_edit.setText(str(common.get("f_expr", "") or ""))
        self.x0_edit.setText(
            format_default(common["x0"]) if "x0" in common else ""
        )
        self.y0_edit.setText(
            format_default(common["y0"]) if "y0" in common else ""
        )
        epsilon = common.get("epsilon", 1e-4)
        self.epsilon_edit.setText(format_default(epsilon) if epsilon is not None else "1e-4")

        while self.spec_form.rowCount():
            self.spec_form.removeRow(0)
        self.field_editors.clear()

        per_method = last_input.get("per_method", {}) if isinstance(last_input, dict) else {}
        saved = per_method.get(spec.id, {}) if isinstance(per_method, dict) else {}
        if not isinstance(saved, dict):
            saved = {}

        for field_spec in spec.input_fields:
            editor = FieldEditor(field_spec)
            if field_spec.name in saved:
                editor.set_value(saved[field_spec.name])
            self.spec_form.addRow(field_spec.label + ":", editor.widget)
            self.field_editors[field_spec.name] = editor

        self.spec_group.setVisible(bool(spec.input_fields))
        self.f_edit.setFocus()

    def collect(self) -> dict:
        f_text = self.f_edit.text().strip()
        if not f_text:
            raise InputError(
                "Функция f(x, y)",
                "Поле «Функция f(x, y)» не заполнено. Введите выражение, "
                "например: sin(x) + cos(y) + 0.1*(x**2 + y**2).",
            )

        x0 = parse_float(self.x0_edit.text(), "x₀")
        y0 = parse_float(self.y0_edit.text(), "y₀")
        epsilon = parse_float(self.epsilon_edit.text(), "Epsilon")
        if epsilon <= 0:
            raise InputError(
                "Epsilon",
                "Epsilon должно быть положительным числом (например, 1e-4).",
            )

        spec_values: dict = {}
        for name, editor in self.field_editors.items():
            value = editor.value()
            if editor.spec.type != "bool":
                if name in {"h", "step", "inner_eps", "alpha0"} and value <= 0:
                    raise InputError(
                        editor.spec.label,
                        f"Поле «{editor.spec.label}» должно быть положительным.",
                    )
                if name in {"alpha", "beta", "gamma"} and value <= 0:
                    raise InputError(
                        editor.spec.label,
                        f"Поле «{editor.spec.label}» должно быть положительным.",
                    )
                if name == "c" and value < 0:
                    raise InputError(
                        editor.spec.label,
                        "Скорость затухания c не может быть отрицательной.",
                    )
            spec_values[name] = value

        if self.spec is not None:
            self._validate_method_specific(spec_values)

        return {
            "f_text": f_text,
            "x0": x0,
            "y0": y0,
            "epsilon": epsilon,
            "spec": spec_values,
        }

    def _validate_method_specific(self, values: dict) -> None:
        if self.spec.id == "hooke_jeeves":
            alpha = values.get("alpha")
            if alpha is not None and not (0 < alpha < 1):
                raise InputError(
                    "Коэффициент уменьшения шага alpha",
                    "Для метода Хука-Дживса коэффициент alpha должен быть "
                    "в интервале (0, 1), например 0.5.",
                )
        elif self.spec.id == "nelder_mead":
            beta = values.get("beta")
            gamma = values.get("gamma")
            if beta is not None and not (0 < beta < 1):
                raise InputError(
                    "Коэффициент сжатия beta",
                    "Для метода Нелдера-Мида коэффициент beta должен быть "
                    "в интервале (0, 1), например 0.5.",
                )
            if gamma is not None and gamma <= 1:
                raise InputError(
                    "Коэффициент растяжения gamma",
                    "Для метода Нелдера-Мида коэффициент gamma должен быть "
                    "больше 1, например 2.0.",
                )
