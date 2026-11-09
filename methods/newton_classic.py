"""Классический метод Ньютона."""

from __future__ import annotations

from optim_core.logging_utils import format_matrix
from optim_core.matrices import mat_add, mat_identity, mat_inv, mat_vec, mat_scale

from .helpers import MAX_ITER, format_point, format_vector, norm, dot, line_search
from .types import RunResult


def run(f_callable, grad_callable, hess_callable, params: dict, mode: str) -> RunResult:
    x = [float(params["x0"]), float(params["y0"])]
    epsilon = float(params["epsilon"])

    explain = mode == "explain"
    log: list[str] = []

    if explain:
        log.append("Классический метод Ньютона")
        log.append(f"Начальная точка: x = {format_point(x)}, f = {f_callable(*x):.6f}")
        log.append(f"Epsilon = {epsilon:g}")
        log.append("-" * 60)

    iterations = 0
    converged = False

    while iterations < MAX_ITER:
        gradient = list(grad_callable(*x))
        gradient_norm = norm(gradient)

        if explain:
            log.append(f"\n=== Итерация {iterations + 1} ===")
            log.append(
                f"  x = {format_point(x)}, f = {f_callable(*x):.6f}"
            )
            log.append(
                f"  grad = {format_vector(gradient)}, "
                f"‖grad‖ = {gradient_norm:.8f}"
            )

        if gradient_norm <= epsilon:
            converged = True
            if explain:
                log.append("  ‖grad‖ ≤ Epsilon — критерий сходимости выполнен, стоп")
            break

        iterations += 1
        hessian = [list(row) for row in hess_callable(*x)]
        if explain:
            log.append("  Матрица Гессе H:")
            log.append(format_matrix(hessian))

        try:
            inverse = mat_inv(hessian)
        except ValueError:
            damping = max(epsilon, 1e-8)
            damped = mat_add(hessian, mat_scale(mat_identity(2), damping))
            if explain:
                log.append(
                    f"  Матрица Гессе вырождена. Пробуем демпфирование: "
                    f"H + {damping:g}·I"
                )
                log.append(format_matrix(damped))
            try:
                inverse = mat_inv(damped)
            except ValueError as exc:
                raise ValueError(
                    "Матрица Гессе вырождена даже после демпфирования. "
                    "Попробуйте другую начальную точку."
                ) from exc

        direction = [-value for value in mat_vec(inverse, gradient)]
        if dot(gradient, direction) >= 0:
            if explain:
                log.append(
                    "  Направление Ньютона не является направлением убывания — "
                    "используем направление наискорейшего спуска -grad"
                )
            direction = [-value for value in gradient]

        alpha = line_search(f_callable, x, direction, gradient)
        x = [x[index] + alpha * direction[index] for index in range(2)]

        if explain:
            log.append("  Обратная матрица Гессе H⁻¹:")
            log.append(format_matrix(inverse))
            log.append(f"  Шаг d = -H⁻¹·grad = {format_vector(direction)}")
            if alpha != 1.0:
                log.append(f"  Шаг по направлению уменьшен: alpha = {alpha:.8f}")
            log.append(
                f"  Новое x = {format_point(x)}, f = {f_callable(*x):.6f}"
            )

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
