"""Реестр методов оптимизации.

Через этот модуль GUI узнаёт, какие методы доступны, какие поля ввода им
нужны и какую функцию запускать.
"""

from __future__ import annotations

from . import (
    coordinate_descent,
    gradient_descent,
    hooke_jeeves,
    nelder_mead,
    newton_classic,
    quasi_newton,
    ravine,
)
from .types import FieldSpec, MethodSpec, RunResult

GROUP_TITLES = {
    "no_derivative": "Методы без производных",
    "gradient": "Градиентные методы",
    "newton": "Методы Ньютона",
}

SUBGROUP_TITLES = {
    "gradient_descent": "Градиентный спуск",
    "quasi_newton": "Квазиньютоновские",
}


METHODS: list[MethodSpec] = [
    MethodSpec(
        id="hooke_jeeves",
        group="no_derivative",
        name="Хук-Дживс",
        dim=2,
        input_fields=[
            FieldSpec("h", "float", 0.5, "Начальный шаг h", "Шаг исследующего поиска по каждой оси."),
            FieldSpec(
                "alpha",
                "float",
                0.5,
                "Коэффициент уменьшения шага alpha",
                "Во сколько раз уменьшается шаг, если улучшения нет (0 < alpha < 1).",
            ),
        ],
        run=hooke_jeeves.run,
        needs_gradient=False,
        needs_hessian=False,
    ),
    MethodSpec(
        id="nelder_mead",
        group="no_derivative",
        name="Нелдер-Мид",
        dim=2,
        input_fields=[
            FieldSpec("step", "float", 1.0, "Начальный размер симплекса step", "Смещение вершин симплекса от начальной точки."),
            FieldSpec("alpha", "float", 1.0, "Коэффициент отражения alpha", "Обычно alpha = 1."),
            FieldSpec("beta", "float", 0.5, "Коэффициент сжатия beta", "Обычно 0 < beta < 1."),
            FieldSpec("gamma", "float", 2.0, "Коэффициент растяжения gamma", "Обычно gamma > 1."),
        ],
        run=nelder_mead.run,
        needs_gradient=False,
        needs_hessian=False,
    ),
    MethodSpec(
        id="ravine",
        group="no_derivative",
        name="Метод оврагов",
        dim=2,
        input_fields=[
            FieldSpec("h", "float", 0.5, "Начальный шаг h", "Внешний шаг поперёк оврага."),
            FieldSpec(
                "inner_eps",
                "float",
                1e-4,
                "Внутренний Epsilon (спуск)",
                "Точность внутреннего градиентного спуска к дну оврага.",
            ),
        ],
        run=ravine.run,
        needs_gradient=False,
        needs_hessian=False,
    ),
    MethodSpec(
        id="coordinate_descent",
        group="no_derivative",
        name="Покоординатный спуск",
        dim=2,
        input_fields=[],
        run=coordinate_descent.run,
        needs_gradient=False,
        needs_hessian=False,
    ),
    MethodSpec(
        id="steepest",
        group="gradient",
        subgroup="gradient_descent",
        name="Наискорейший спуск",
        dim=2,
        input_fields=[],
        run=gradient_descent.run_steepest,
        needs_gradient=True,
        needs_hessian=False,
    ),
    MethodSpec(
        id="conjugate",
        group="gradient",
        subgroup="gradient_descent",
        name="Сопряжённые градиенты",
        dim=2,
        input_fields=[],
        run=gradient_descent.run_conjugate,
        needs_gradient=True,
        needs_hessian=False,
    ),
    MethodSpec(
        id="sgd",
        group="gradient",
        subgroup="gradient_descent",
        name="Стохастический (SGD)",
        dim=2,
        input_fields=[
            FieldSpec("alpha0", "float", 0.05, "Начальная скорость обучения alpha0", "alpha_k = alpha0 / (1 + c·k)."),
            FieldSpec("c", "float", 0.01, "Скорость затухания c", "Чем больше c, тем быстрее уменьшается alpha_k."),
            FieldSpec(
                "normalize",
                "bool",
                False,
                "Нормализовать приближение градиента",
                "Перед шагом приближение градиента приводится к единичной длине.",
            ),
        ],
        run=gradient_descent.run_sgd,
        needs_gradient=True,
        needs_hessian=False,
    ),
    MethodSpec(
        id="newton",
        group="newton",
        name="Классический Ньютон",
        dim=2,
        input_fields=[],
        run=newton_classic.run,
        needs_gradient=True,
        needs_hessian=True,
    ),
    MethodSpec(
        id="broyden",
        group="newton",
        subgroup="quasi_newton",
        name="Бройден",
        dim=2,
        input_fields=[],
        run=quasi_newton.run_broyden,
        needs_gradient=True,
        needs_hessian=False,
    ),
    MethodSpec(
        id="bfgs",
        group="newton",
        subgroup="quasi_newton",
        name="BFGS",
        dim=2,
        input_fields=[],
        run=quasi_newton.run_bfgs,
        needs_gradient=True,
        needs_hessian=False,
    ),
]

_BY_ID = {method.id: method for method in METHODS}


def get_method(method_id: str) -> MethodSpec | None:
    return _BY_ID.get(method_id)


def methods_by_group(group: str) -> list[MethodSpec]:
    return [method for method in METHODS if method.group == group]


def display_name(method: MethodSpec) -> str:
    """Человекочитаемое имя для метаданных и заголовков."""

    if method.subgroup:
        return f"{SUBGROUP_TITLES.get(method.subgroup, method.subgroup)} → {method.name}"
    if method.group == "newton":
        return f"Ньютон → {method.name}"
    return method.name


__all__ = [
    "METHODS",
    "GROUP_TITLES",
    "SUBGROUP_TITLES",
    "FieldSpec",
    "MethodSpec",
    "RunResult",
    "get_method",
    "methods_by_group",
    "display_name",
]
