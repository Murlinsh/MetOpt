"""Метод покоординатного спуска (циклический покоординатный спуск)."""

from __future__ import annotations

import math

from .helpers import MAX_ITER, format_point, safe_eval
from .types import RunResult

_PHI = (math.sqrt(5.0) - 1.0) / 2.0


def _bracket_minimum(func, x0: float, step: float, max_expansions: int = 60) -> tuple[float, float]:
    """Находит отрезок, внутри которого лежит минимум одномерной функции."""

    left = x0 - step
    right = x0 + step
    f0 = func(x0)
    f_left = func(left)
    f_right = func(right)

    if f_left >= f0 and f_right >= f0:
        return left, right

    if f_right < f_left:
        previous = x0
        current, f_current = right, f_right
        step_size = step
        for _ in range(max_expansions):
            step_size *= 2.0
            candidate = current + step_size
            f_candidate = func(candidate)
            if f_candidate >= f_current:
                return previous, candidate
            previous = current
            current, f_current = candidate, f_candidate
        return previous, current

    previous = x0
    current, f_current = left, f_left
    step_size = step
    for _ in range(max_expansions):
        step_size *= 2.0
        candidate = current - step_size
        f_candidate = func(candidate)
        if f_candidate >= f_current:
            return candidate, previous
        previous = current
        current, f_current = candidate, f_candidate
    return current, previous


def _golden_section(func, left: float, right: float, tolerance: float) -> float:
    """Минимизирует унимодальную функцию на отрезке методом золотого сечения."""

    a, b = left, right
    c = b - _PHI * (b - a)
    d = a + _PHI * (b - a)
    f_c = func(c)
    f_d = func(d)

    while abs(b - a) > tolerance:
        if f_c < f_d:
            b, d, f_d = d, c, f_c
            c = b - _PHI * (b - a)
            f_c = func(c)
        else:
            a, c, f_c = c, d, f_d
            d = a + _PHI * (b - a)
            f_d = func(d)

    return (a + b) / 2.0


def _minimize_along_axis(
    f_callable,
    point: list[float],
    axis: int,
    initial_step: float,
    tolerance: float,
) -> tuple[float, float]:
    """Находит минимум вдоль одной координатной оси."""

    def one_dimensional(value: float) -> float:
        candidate = point[:]
        candidate[axis] = value
        return safe_eval(f_callable, candidate)

    left, right = _bracket_minimum(one_dimensional, point[axis], initial_step)
    best = _golden_section(one_dimensional, left, right, tolerance)
    return best, one_dimensional(best)


def run(f_callable, grad_callable, hess_callable, params: dict, mode: str) -> RunResult:
    x = [float(params["x0"]), float(params["y0"])]
    epsilon = float(params["epsilon"])

    explain = mode == "explain"
    log: list[str] = []

    initial_step = max(1.0, 0.25 * max(abs(x[0]), abs(x[1])))
    axis_names = ("x", "y")

    if explain:
        log.append("Метод покоординатного спуска")
        log.append(f"Начальная точка: x = {format_point(x)}, f = {f_callable(*x):.6f}")
        log.append(
            f"Epsilon = {epsilon:g}, начальный шаг для одномерного поиска = "
            f"{initial_step:g}"
        )
        log.append(
            "На каждой итерации по очереди минимизируем функцию вдоль осей x и y "
            "методом золотого сечения."
        )
        log.append("-" * 60)

    iterations = 0
    converged = False

    while iterations < MAX_ITER:
        iterations += 1
        previous = x[:]

        if explain:
            log.append(f"\n=== Итерация {iterations} ===")
            log.append(
                f"  Начало итерации: x = {format_point(x)}, "
                f"f = {f_callable(*x):.6f}"
            )

        for axis in range(2):
            old_value = x[axis]
            old_function = safe_eval(f_callable, x)
            best_value, best_function = _minimize_along_axis(
                f_callable, x, axis, initial_step, epsilon
            )

            if best_function < old_function:
                x[axis] = best_value
                if explain:
                    log.append(
                        f"  Ось {axis_names[axis]}: {old_value:.8f} → "
                        f"{best_value:.8f}, f = {best_function:.6f}"
                    )
            elif explain:
                log.append(
                    f"  Ось {axis_names[axis]}: улучшения нет, "
                    f"координата остаётся {old_value:.8f}"
                )

        shift = max(abs(x[index] - previous[index]) for index in range(2))
        if explain:
            log.append(
                f"  Новое x = {format_point(x)}, f = {f_callable(*x):.6f}, "
                f"смещение = {shift:.8f}"
            )

        if shift <= epsilon:
            converged = True
            if explain:
                log.append("  Смещение ≤ Epsilon — критерий сходимости выполнен, стоп")
            break

    if explain:
        if not converged:
            log.append("\nДостигнут предел итераций. Сходимость не подтверждена.")
        log.append(f"Результат: x* = {format_point(x)}, f(x*) = {f_callable(*x):.6f}")

    return RunResult(
        x_min=tuple(x),
        f_min=float(f_callable(*x)),
        iterations=iterations,
        log_lines=log,
        converged=converged,
    )
