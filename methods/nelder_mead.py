"""Метод Нелдера-Мида (симплексный метод)."""

from __future__ import annotations

import math

from .helpers import MAX_ITER, format_point
from .types import RunResult


def run(f_callable, grad_callable, hess_callable, params: dict, mode: str) -> RunResult:
    x0 = [float(params["x0"]), float(params["y0"])]
    step = float(params["step"])
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    gamma = float(params["gamma"])
    epsilon = float(params["epsilon"])

    explain = mode == "explain"
    log: list[str] = []

    simplex = [x0[:]]
    for index in range(2):
        point = x0[:]
        point[index] += step
        simplex.append(point)

    if explain:
        log.append("Метод Нелдера-Мида")
        log.append("Начальный симплекс:")
        for label, point in zip(("P0", "P1", "P2"), simplex):
            log.append(f"  {label} = {format_point(point)}, f = {f_callable(*point):.6f}")
        log.append(
            f"alpha = {alpha:g} (отражение), beta = {beta:g} (сжатие), "
            f"gamma = {gamma:g} (растяжение), epsilon = {epsilon:g}"
        )
        log.append("-" * 60)

    iterations = 0
    converged = False
    sigma = float("inf")

    while iterations < MAX_ITER:
        iterations += 1
        simplex.sort(key=lambda point: f_callable(*point))
        p_best, p_good, p_worst = simplex[0], simplex[1], simplex[2]
        f_best = f_callable(*p_best)
        f_good = f_callable(*p_good)
        f_worst = f_callable(*p_worst)

        centroid = [
            (p_best[index] + p_good[index]) / 2.0 for index in range(2)
        ]
        f_centroid = f_callable(*centroid)
        sigma = math.sqrt(
            (
                (f_best - f_centroid) ** 2
                + (f_good - f_centroid) ** 2
                + (f_worst - f_centroid) ** 2
            )
            / 3.0
        )

        if explain:
            log.append(f"\n=== Итерация {iterations} ===")
            log.append(f"  Лучшая вершина:  {format_point(p_best)}, f = {f_best:.6f}")
            log.append(f"  Средняя вершина: {format_point(p_good)}, f = {f_good:.6f}")
            log.append(f"  Худшая вершина:  {format_point(p_worst)}, f = {f_worst:.6f}")
            log.append(
                f"  Центроид лучших двух: {format_point(centroid)}, "
                f"f = {f_centroid:.6f}"
            )
            log.append(f"  sigma = {sigma:.8f}")

        if sigma <= epsilon:
            converged = True
            if explain:
                log.append("  sigma ≤ Epsilon — критерий сходимости выполнен, стоп")
            break

        # Отражение
        reflected = [
            centroid[index] + alpha * (centroid[index] - p_worst[index])
            for index in range(2)
        ]
        f_reflected = f_callable(*reflected)
        if explain:
            log.append(
                f"  Отражение: {format_point(reflected)}, f = {f_reflected:.6f}"
            )

        if f_best <= f_reflected < f_good:
            simplex[2] = reflected
            if explain:
                log.append("  Отражение принято")
            continue

        if f_reflected < f_best:
            # Растяжение
            expanded = [
                centroid[index] + gamma * (reflected[index] - centroid[index])
                for index in range(2)
            ]
            f_expanded = f_callable(*expanded)
            if explain:
                log.append(
                    f"  Растяжение: {format_point(expanded)}, f = {f_expanded:.6f}"
                )
            if f_expanded < f_reflected:
                simplex[2] = expanded
                if explain:
                    log.append("  Принята точка растяжения")
            else:
                simplex[2] = reflected
                if explain:
                    log.append("  Принята точка отражения")
            continue

        # Сжатие
        if f_reflected < f_worst:
            contracted = [
                centroid[index] + beta * (reflected[index] - centroid[index])
                for index in range(2)
            ]
            if explain:
                log.append(
                    f"  Внешнее сжатие: {format_point(contracted)}"
                )
        else:
            contracted = [
                centroid[index] + beta * (p_worst[index] - centroid[index])
                for index in range(2)
            ]
            if explain:
                log.append(
                    f"  Внутреннее сжатие: {format_point(contracted)}"
                )

        f_contracted = f_callable(*contracted)
        if explain:
            log.append(f"  f(сжатой точки) = {f_contracted:.6f}")

        if f_contracted < min(f_reflected, f_worst):
            simplex[2] = contracted
            if explain:
                log.append("  Сжатие принято")
        else:
            if explain:
                log.append("  Редукция симплекса к лучшей вершине")
            for index in (1, 2):
                simplex[index] = [
                    p_best[coordinate]
                    + 0.5 * (simplex[index][coordinate] - p_best[coordinate])
                    for coordinate in range(2)
                ]
            if explain:
                for label, point in zip(("P0", "P1", "P2"), simplex):
                    log.append(
                        f"    {label} = {format_point(point)}, "
                        f"f = {f_callable(*point):.6f}"
                    )

    simplex.sort(key=lambda point: f_callable(*point))
    best = simplex[0]

    if explain:
        if converged:
            log.append("\nКритерий остановки выполнен: sigma ≤ Epsilon.")
        else:
            log.append("\nДостигнут предел итераций. Сходимость не подтверждена.")
        log.append(
            f"Результат: x* = {format_point(best)}, "
            f"f(x*) = {f_callable(*best):.6f}, sigma = {sigma:.8f}"
        )

    return RunResult(
        x_min=tuple(best),
        f_min=float(f_callable(*best)),
        iterations=iterations,
        log_lines=log,
        converged=converged,
    )
