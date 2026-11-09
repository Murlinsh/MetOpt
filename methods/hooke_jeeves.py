"""Метод Хука-Дживса (Hooke-Jeeves)."""

from __future__ import annotations

from .helpers import MAX_ITER, format_point, format_vector
from .types import RunResult


def _explore(
    f_callable,
    x: list[float],
    h: list[float],
    log: list[str] | None,
) -> list[float]:
    """Исследующий поиск по каждой координате."""

    axis_names = ("x", "y")
    for index in range(len(x)):
        candidate = x[:]
        candidate[index] += h[index]
        if f_callable(*candidate) < f_callable(*x):
            x = candidate
            if log is not None:
                log.append(
                    f"    Шаг + по {axis_names[index]}: {format_point(x)}, "
                    f"f = {f_callable(*x):.6f} — улучшение"
                )
            continue

        candidate = x[:]
        candidate[index] -= h[index]
        if f_callable(*candidate) < f_callable(*x):
            x = candidate
            if log is not None:
                log.append(
                    f"    Шаг − по {axis_names[index]}: {format_point(x)}, "
                    f"f = {f_callable(*x):.6f} — улучшение"
                )
        elif log is not None:
            log.append(
                f"    Шаги по {axis_names[index]} в + и −: улучшения нет"
            )
    return x


def run(f_callable, grad_callable, hess_callable, params: dict, mode: str) -> RunResult:
    x = [float(params["x0"]), float(params["y0"])]
    h = [float(params["h"]), float(params["h"])]
    alpha = float(params["alpha"])
    epsilon = float(params["epsilon"])

    explain = mode == "explain"
    log: list[str] = []

    if explain:
        log.append("Метод Хука-Дживса")
        log.append(f"Начальная точка: x = {format_point(x)}, f = {f_callable(*x):.6f}")
        log.append(f"Начальный шаг: h = {format_vector(h)}, alpha = {alpha:g}, epsilon = {epsilon:g}")
        log.append("-" * 60)

    iterations = 0
    converged = False
    steps_left = MAX_ITER

    while max(abs(value) for value in h) > epsilon and steps_left > 0:
        iterations += 1
        steps_left -= 1
        x_previous = x[:]

        if explain:
            log.append(f"\n=== Итерация {iterations} ===")
            log.append(
                f"  Текущая точка: {format_point(x)}, "
                f"f = {f_callable(*x):.6f}, h = {format_vector(h)}"
            )
            log.append("  Исследующий поиск:")

        x = _explore(f_callable, x, h, log if explain else None)

        if f_callable(*x) < f_callable(*x_previous):
            if explain:
                log.append(
                    f"  Исследующий поиск успешен: {format_point(x)}, "
                    f"f = {f_callable(*x):.6f}"
                )
                log.append("  Поиск по образцу:")

            while steps_left > 0:
                steps_left -= 1
                pattern_point = [
                    2.0 * x[index] - x_previous[index] for index in range(len(x))
                ]
                candidate = _explore(
                    f_callable, pattern_point, h, log if explain else None
                )
                if f_callable(*candidate) < f_callable(*x):
                    x_previous = x[:]
                    x = candidate[:]
                    if explain:
                        log.append(
                            f"    Успешный шаг по образцу: {format_point(x)}, "
                            f"f = {f_callable(*x):.6f}"
                        )
                else:
                    if explain:
                        log.append("    Шаг по образцу без улучшения — стоп")
                    break
        else:
            h = [value * alpha for value in h]
            if explain:
                log.append(f"  Уменьшаем шаг: h = {format_vector(h)}")

        if max(abs(value) for value in h) <= epsilon:
            converged = True
            break

    if max(abs(value) for value in h) <= epsilon:
        converged = True

    if explain:
        if converged:
            log.append("\nКритерий остановки выполнен: max(|h_x|, |h_y|) ≤ Epsilon.")
        else:
            log.append("\nДостигнут предел итераций. Сходимость не подтверждена.")
        log.append(f"Результат: x* = {format_point(x)}, f(x*) = {f_callable(*x):.6f}")

    return RunResult(
        x_min=tuple(x),
        f_min=float(f_callable(*x)),
        iterations=iterations,
        log_lines=log,
        converged=converged,
    )
