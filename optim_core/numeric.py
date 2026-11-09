"""Численные производные центральными разностями.

Используются как запасной вариант, если символьное дифференцирование
невозможно, а также как «вторая линия защиты» для гессиана в методе Ньютона.
"""

from __future__ import annotations

from typing import Callable, Sequence


def numeric_gradient(
    f_callable: Callable[..., float], x: Sequence[float], h: float = 1e-6
) -> list[float]:
    """Численный градиент в точке ``x`` центральными разностями."""

    point = list(x)
    gradient: list[float] = []
    for index in range(len(point)):
        plus = point[:]
        minus = point[:]
        plus[index] += h
        minus[index] -= h
        gradient.append((f_callable(*plus) - f_callable(*minus)) / (2.0 * h))
    return gradient


def numeric_hessian(
    f_callable: Callable[..., float], x: Sequence[float], h: float = 1e-5
) -> list[list[float]]:
    """Численная матрица Гессе в точке ``x``."""

    point = list(x)
    n = len(point)
    value = f_callable(*point)
    hessian = [[0.0] * n for _ in range(n)]

    for i in range(n):
        plus = point[:]
        minus = point[:]
        plus[i] += h
        minus[i] -= h
        hessian[i][i] = (f_callable(*plus) - 2.0 * value + f_callable(*minus)) / (h * h)

    for i in range(n):
        for j in range(i + 1, n):
            pp = point[:]
            pm = point[:]
            mp = point[:]
            mm = point[:]
            pp[i] += h
            pp[j] += h
            pm[i] += h
            pm[j] -= h
            mp[i] -= h
            mp[j] += h
            mm[i] -= h
            mm[j] -= h
            mixed = (
                f_callable(*pp)
                - f_callable(*pm)
                - f_callable(*mp)
                + f_callable(*mm)
            ) / (4.0 * h * h)
            hessian[i][j] = mixed
            hessian[j][i] = mixed

    return hessian
