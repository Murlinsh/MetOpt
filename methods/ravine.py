"""Метод оврагов (Гельфанда-Цетлина)."""

from __future__ import annotations

from optim_core.numeric import numeric_gradient

from .helpers import MAX_ITER, format_point, format_vector, norm
from .types import RunResult


def _descend(
    f_callable,
    point: list[float],
    epsilon: float,
    max_iter: int = 500,
) -> list[float]:
    """Спуск к дну оврага: градиентный спуск с дробным шагом."""

    x = point[:]
    for _ in range(max_iter):
        gradient = numeric_gradient(f_callable, x)
        gradient_norm = norm(gradient)
        if gradient_norm < epsilon:
            break

        alpha = 0.1
        c = 1e-4
        rho = 0.5
        current = f_callable(*x)
        while True:
            candidate = [
                x[index] - alpha * gradient[index] for index in range(len(x))
            ]
            if f_callable(*candidate) <= current - c * alpha * gradient_norm**2:
                break
            alpha *= rho
            if alpha < 1e-12:
                break
        x = [x[index] - alpha * gradient[index] for index in range(len(x))]

    return x


def run(f_callable, grad_callable, hess_callable, params: dict, mode: str) -> RunResult:
    x = [float(params["x0"]), float(params["y0"])]
    h = float(params["h"])
    inner_epsilon = float(params["inner_eps"])
    epsilon = float(params["epsilon"])

    explain = mode == "explain"
    log: list[str] = []

    x = _descend(f_callable, x, inner_epsilon)

    if explain:
        log.append("Метод оврагов (Гельфанда-Цетлина)")
        log.append(
            f"Начальная точка: {format_point([float(params['x0']), float(params['y0'])])}"
        )
        log.append(
            f"После спуска к дну оврага: x = {format_point(x)}, "
            f"f = {f_callable(*x):.6f}"
        )
        log.append(
            f"Внешний шаг h = {h:g}, epsilon = {epsilon:g}, "
            f"внутренний epsilon = {inner_epsilon:g}"
        )
        log.append("-" * 60)

    iterations = 0
    direction_index = 0
    converged = False

    while h > epsilon and iterations < MAX_ITER:
        iterations += 1
        direction = [0.0, 0.0]
        direction[direction_index] = 1.0

        if explain:
            log.append(f"\n=== Итерация {iterations} ===")
            log.append(
                f"  Дно оврага: {format_point(x)}, f = {f_callable(*x):.6f}"
            )
            log.append(
                f"  Шаг h = {h:g}, направление d = {format_vector(direction)}"
            )

        trial = _descend(
            f_callable,
            [x[0] + h * direction[0], x[1] + h * direction[1]],
            inner_epsilon,
        )
        if explain:
            log.append(
                f"  Шаг +h·d и спуск: {format_point(trial)}, "
                f"f = {f_callable(*trial):.6f}"
            )

        if f_callable(*trial) < f_callable(*x):
            x = trial
            if explain:
                log.append("  Улучшение — продолжаем в том же направлении")
            continue

        trial = _descend(
            f_callable,
            [x[0] - h * direction[0], x[1] - h * direction[1]],
            inner_epsilon,
        )
        if explain:
            log.append(
                f"  Шаг −h·d и спуск: {format_point(trial)}, "
                f"f = {f_callable(*trial):.6f}"
            )

        if f_callable(*trial) < f_callable(*x):
            x = trial
            if explain:
                log.append("  Улучшение в обратную сторону")
        else:
            h /= 2.0
            direction_index = (direction_index + 1) % 2
            if explain:
                log.append(
                    f"  Улучшения нет — уменьшаем шаг: h = {h:g}, "
                    f"меняем направление на ось {'xy'[direction_index]}"
                )

        if h <= epsilon:
            converged = True

    if h <= epsilon:
        converged = True

    if explain:
        if converged:
            log.append("\nКритерий остановки выполнен: h ≤ Epsilon.")
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
