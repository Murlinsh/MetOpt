"""Квазиньютоновские методы: Бройден и BFGS."""

from __future__ import annotations

from optim_core.logging_utils import format_matrix
from optim_core.matrices import (
    mat_add,
    mat_identity,
    mat_inv,
    mat_mul,
    mat_scale,
    mat_vec,
)

from .helpers import (
    MAX_ITER,
    add,
    dot,
    format_point,
    format_vector,
    line_search,
    mul,
    norm,
    sub,
)
from .types import RunResult


def _inverse_with_damping(matrix: list[list[float]], epsilon: float, log: list[str] | None):
    try:
        return mat_inv(matrix)
    except ValueError:
        damping = max(epsilon, 1e-8)
        damped = mat_add(matrix, mat_scale(mat_identity(2), damping))
        if log is not None:
            log.append(
                f"  Матрица вырождена. Демпфирование: M + {damping:g}·I"
            )
        try:
            return mat_inv(damped)
        except ValueError as exc:
            raise ValueError(
                "Матрица квазиньютоновского приближения вырождена даже после "
                "демпфирования. Попробуйте другую начальную точку."
            ) from exc


def run_bfgs(f_callable, grad_callable, hess_callable, params: dict, mode: str) -> RunResult:
    x = [float(params["x0"]), float(params["y0"])]
    epsilon = float(params["epsilon"])

    explain = mode == "explain"
    log: list[str] = []

    inverse_hessian = mat_identity(2)
    gradient = list(grad_callable(*x))

    if explain:
        log.append("Квазиньютоновский метод BFGS")
        log.append(f"Начальная точка: x = {format_point(x)}, f = {f_callable(*x):.6f}")
        log.append("Начальное приближение обратного гессиана H₀ = I")
        log.append(f"Epsilon = {epsilon:g}")
        log.append("-" * 60)

    iterations = 0
    converged = False

    while iterations < MAX_ITER:
        gradient_norm = norm(gradient)

        if explain:
            log.append(f"\n=== Итерация {iterations + 1} ===")
            log.append(
                f"  x = {format_point(x)}, f = {f_callable(*x):.6f}, "
                f"‖grad‖ = {gradient_norm:.8f}"
            )
            log.append("  Текущее приближение обратного гессиана H:")
            log.append(format_matrix(inverse_hessian))

        if gradient_norm <= epsilon:
            converged = True
            if explain:
                log.append("  ‖grad‖ ≤ Epsilon — критерий сходимости выполнен, стоп")
            break

        iterations += 1
        direction = [-value for value in mat_vec(inverse_hessian, gradient)]
        if dot(gradient, direction) >= 0:
            if explain:
                log.append(
                    "  Направление BFGS не является направлением убывания — "
                    "сбрасываем H к I и используем -grad"
                )
            inverse_hessian = mat_identity(2)
            direction = [-value for value in gradient]
        alpha = line_search(f_callable, x, direction, gradient)
        x_new = add(x, mul(direction, alpha))
        gradient_new = list(grad_callable(*x_new))

        s = sub(x_new, x)
        y = sub(gradient_new, gradient)

        if explain:
            log.append(f"  Направление d = -H·grad = {format_vector(direction)}")
            log.append(f"  alpha = {alpha:.8f}")
            log.append(
                f"  Новое x = {format_point(x_new)}, f = {f_callable(*x_new):.6f}"
            )
            log.append(f"  s = {format_vector(s)}, y = {format_vector(y)}")

        ys = dot(y, s)
        if ys > 1e-12:
            rho = 1.0 / ys
            n = 2
            v = [
                [
                    (1.0 if i == j else 0.0) - rho * s[i] * y[j]
                    for j in range(n)
                ]
                for i in range(n)
            ]
            vh = mat_mul(v, inverse_hessian)
            vhvt = mat_mul(vh, [[v[j][i] for j in range(n)] for i in range(n)])
            inverse_hessian = [
                [
                    vhvt[i][j] + rho * s[i] * s[j]
                    for j in range(n)
                ]
                for i in range(n)
            ]
            if explain:
                log.append("  BFGS-обновление H выполнено (rank-2)")
                log.append("  Новое приближение обратного гессиана H:")
                log.append(format_matrix(inverse_hessian))
        elif explain:
            log.append("  y·s ≤ 0 — обновление H пропущено")

        x, gradient = x_new, gradient_new

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


def run_broyden(f_callable, grad_callable, hess_callable, params: dict, mode: str) -> RunResult:
    x = [float(params["x0"]), float(params["y0"])]
    epsilon = float(params["epsilon"])

    explain = mode == "explain"
    log: list[str] = []

    approximation = mat_identity(2)
    gradient = list(grad_callable(*x))

    if explain:
        log.append("Квазиньютоновский метод Бройдена (rank-1)")
        log.append(f"Начальная точка: x = {format_point(x)}, f = {f_callable(*x):.6f}")
        log.append("Начальное приближение матрицы Гессе B₀ = I")
        log.append(f"Epsilon = {epsilon:g}")
        log.append("-" * 60)

    iterations = 0
    converged = False

    while iterations < MAX_ITER:
        gradient_norm = norm(gradient)

        if explain:
            log.append(f"\n=== Итерация {iterations + 1} ===")
            log.append(
                f"  x = {format_point(x)}, f = {f_callable(*x):.6f}, "
                f"‖grad‖ = {gradient_norm:.8f}"
            )
            log.append("  Текущее приближение матрицы Гессе B:")
            log.append(format_matrix(approximation))

        if gradient_norm <= epsilon:
            converged = True
            if explain:
                log.append("  ‖grad‖ ≤ Epsilon — критерий сходимости выполнен, стоп")
            break

        iterations += 1
        inverse = _inverse_with_damping(approximation, epsilon, log if explain else None)
        direction = [-value for value in mat_vec(inverse, gradient)]
        if dot(gradient, direction) >= 0:
            if explain:
                log.append(
                    "  Направление Бройдена не является направлением убывания — "
                    "сбрасываем B к I и используем -grad"
                )
            approximation = mat_identity(2)
            inverse = mat_identity(2)
            direction = [-value for value in gradient]
        alpha = line_search(f_callable, x, direction, gradient)
        x_new = add(x, mul(direction, alpha))
        gradient_new = list(grad_callable(*x_new))

        s = sub(x_new, x)
        y = sub(gradient_new, gradient)

        if explain:
            log.append(f"  Обратная матрица B⁻¹:")
            log.append(format_matrix(inverse))
            log.append(f"  Направление d = -B⁻¹·grad = {format_vector(direction)}")
            log.append(f"  alpha = {alpha:.8f}")
            log.append(
                f"  Новое x = {format_point(x_new)}, f = {f_callable(*x_new):.6f}"
            )
            log.append(f"  s = {format_vector(s)}, y = {format_vector(y)}")

        ss = dot(s, s)
        if ss > 1e-16:
            bs = mat_vec(approximation, s)
            difference = [y[i] - bs[i] for i in range(2)]
            approximation = [
                [
                    approximation[i][j] + difference[i] * s[j] / ss
                    for j in range(2)
                ]
                for i in range(2)
            ]
            if explain:
                log.append("  Rank-1 обновление B выполнено")
                log.append("  Новое приближение матрицы Гессе B:")
                log.append(format_matrix(approximation))
        elif explain:
            log.append("  ‖s‖² слишком мал — обновление B пропущено")

        x, gradient = x_new, gradient_new

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
