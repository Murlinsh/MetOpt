"""Вспомогательные функции для численных методов."""

from __future__ import annotations

import math
from typing import Callable, Sequence

MAX_ITER = 5000


def norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def add(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def sub(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def mul(vector: Sequence[float], scalar: float) -> list[float]:
    return [value * scalar for value in vector]


def format_point(point: Sequence[float], digits: int = 6) -> str:
    return "(" + ", ".join(f"{value:.{digits}f}" for value in point) + ")"


def format_vector(vector: Sequence[float], digits: int = 6) -> str:
    return "[" + ", ".join(f"{value:.{digits}f}" for value in vector) + "]"


def format_number(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def line_search(
    f_callable: Callable[..., float],
    x: Sequence[float],
    direction: Sequence[float],
    gradient: Sequence[float],
    alpha: float = 1.0,
    c: float = 1e-4,
    rho: float = 0.5,
    max_steps: int = 200,
) -> float:
    """Дробный шаг (backtracking) с условием Армихо."""

    current = f_callable(*x)
    slope = dot(gradient, direction)

    for _ in range(max_steps):
        candidate = [x[i] + alpha * direction[i] for i in range(len(x))]
        if f_callable(*candidate) <= current + c * alpha * slope:
            return alpha
        alpha *= rho
        if alpha < 1e-14:
            break

    return max(alpha, 1e-14)


def safe_eval(f_callable: Callable[..., float], point: Sequence[float]) -> float:
    try:
        value = float(f_callable(*point))
    except Exception as exc:
        raise ValueError(
            f"Не удалось вычислить функцию в точке {format_point(point)}: {exc}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Функция вернула нечисловое значение в точке {format_point(point)}."
        )
    return value
