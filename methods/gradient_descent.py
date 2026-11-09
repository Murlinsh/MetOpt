"""Градиентные методы: наискорейший спуск, сопряжённые градиенты, SGD."""

from __future__ import annotations

import random

from .helpers import (
    MAX_ITER,
    add,
    dot,
    format_number,
    format_point,
    format_vector,
    line_search,
    mul,
    norm,
    sub,
)
from .types import RunResult


def run_steepest(f_callable, grad_callable, hess_callable, params: dict, mode: str) -> RunResult:
    x = [float(params["x0"]), float(params["y0"])]
    epsilon = float(params["epsilon"])

    explain = mode == "explain"
    log: list[str] = []

    if explain:
        log.append("Метод наискорейшего градиентного спуска")
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
        direction = mul(gradient, -1.0)
        alpha = line_search(f_callable, x, direction, gradient)
        x = add(x, mul(direction, alpha))

        if explain:
            log.append(f"  Направление d = {format_vector(direction)}")
            log.append(f"  alpha = {alpha:.8f}")
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


def run_conjugate(f_callable, grad_callable, hess_callable, params: dict, mode: str) -> RunResult:
    x = [float(params["x0"]), float(params["y0"])]
    epsilon = float(params["epsilon"])

    explain = mode == "explain"
    log: list[str] = []

    gradient = list(grad_callable(*x))
    direction = mul(gradient, -1.0)

    if explain:
        log.append("Метод сопряжённых градиентов (Флетчер-Ривс)")
        log.append(f"Начальная точка: x = {format_point(x)}, f = {f_callable(*x):.6f}")
        log.append(f"Epsilon = {epsilon:g}")
        log.append("-" * 60)

    iterations = 0
    converged = False

    while iterations < MAX_ITER:
        gradient_norm = norm(gradient)

        if explain:
            log.append(f"\n=== Итерация {iterations + 1} ===")
            log.append(
                f"  x = {format_point(x)}, f = {f_callable(*x):.6f}"
            )
            log.append(
                f"  g = {format_vector(gradient)}, ‖g‖ = {gradient_norm:.8f}"
            )
            log.append(f"  d = {format_vector(direction)}")

        if gradient_norm <= epsilon:
            converged = True
            if explain:
                log.append("  ‖g‖ ≤ Epsilon — критерий сходимости выполнен, стоп")
            break

        iterations += 1
        alpha = line_search(f_callable, x, direction, gradient)
        x_new = add(x, mul(direction, alpha))
        gradient_new = list(grad_callable(*x_new))

        old_norm_squared = dot(gradient, gradient)
        beta = (
            dot(gradient_new, gradient_new) / old_norm_squared
            if old_norm_squared > 1e-16
            else 0.0
        )
        direction = add(mul(gradient_new, -1.0), mul(direction, beta))

        if explain:
            log.append(f"  alpha = {alpha:.8f}, beta = {beta:.8f}")
            log.append(
                f"  Новое x = {format_point(x_new)}, f = {f_callable(*x_new):.6f}"
            )
            log.append(f"  Новое направление d = {format_vector(direction)}")

        x = x_new
        gradient = gradient_new

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


def run_sgd(f_callable, grad_callable, hess_callable, params: dict, mode: str) -> RunResult:
    x = [float(params["x0"]), float(params["y0"])]
    epsilon = float(params["epsilon"])
    alpha0 = float(params["alpha0"])
    decay = float(params["c"])
    normalize = bool(params.get("normalize", False))

    terms = params.get("_terms") or []
    n_terms = len(terms)

    explain = mode == "explain"
    log: list[str] = []

    if explain:
        log.append("Стохастический градиентный спуск (SGD)")
        log.append(f"Начальная точка: x = {format_point(x)}, f = {f_callable(*x):.6f}")
        log.append(
            f"alpha0 = {alpha0:g}, c = {decay:g}, "
            f"нормализация = {'да' if normalize else 'нет'}, epsilon = {epsilon:g}"
        )
        if n_terms > 1:
            log.append(
                f"Функция разложена на {n_terms} слагаемых. На каждой итерации "
                "случайно выбирается одно слагаемое."
            )
        else:
            log.append(
                "Функция не разложена на несколько слагаемых: SGD вырождается "
                "в обычный градиентный спуск."
            )
        log.append("-" * 60)

    iterations = 0
    converged = False
    message = ""
    if n_terms <= 1:
        message = "SGD вырожден в обычный градиентный спуск: у функции одно слагаемое."

    best_point = x[:]
    best_gradient_norm = float("inf")

    while iterations < MAX_ITER:
        gradient = list(grad_callable(*x))
        gradient_norm = norm(gradient)

        if gradient_norm < best_gradient_norm:
            best_gradient_norm = gradient_norm
            best_point = x[:]

        if explain:
            log.append(f"\n=== Итерация {iterations + 1} ===")
            log.append(
                f"  x = {format_point(x)}, f = {f_callable(*x):.6f}"
            )
            log.append(
                f"  Полный градиент = {format_vector(gradient)}, "
                f"‖grad‖ = {gradient_norm:.8f}"
            )

        if gradient_norm <= epsilon:
            converged = True
            if explain:
                log.append("  ‖grad‖ ≤ Epsilon (по чистому градиенту) — стоп")
            break

        iterations += 1
        learning_rate = alpha0 / (1.0 + decay * iterations)

        if n_terms > 1:
            chosen = random.randrange(n_terms)
            _, term_gradient = terms[chosen]
            approximation = [n_terms * value for value in term_gradient(*x)]
            if explain:
                log.append(f"  Случайно выбрано слагаемое №{chosen + 1} из {n_terms}")
                log.append(
                    f"  Градиент слагаемого (с множителем n = {n_terms}): "
                    f"{format_vector(approximation)}"
                )
        else:
            chosen = None
            approximation = gradient[:]
            if explain:
                log.append("  Используется полный градиент (одно слагаемое)")

        if normalize:
            approximation_norm = norm(approximation)
            if approximation_norm > 1e-15:
                approximation = [value / approximation_norm for value in approximation]
                if explain:
                    log.append(
                        "  Приближение градиента нормализовано к единичной длине"
                    )

        x = [x[index] - learning_rate * approximation[index] for index in range(2)]

        if explain:
            log.append(f"  alpha_{iterations} = {format_number(learning_rate, 8)}")
            log.append(
                f"  Новое x = {format_point(x)}, f = {f_callable(*x):.6f}"
            )

    if not converged:
        # Возвращаем лучшую найденную точку, а не последнюю зашумлённую.
        x = best_point

    if explain:
        if not converged:
            log.append("\nДостигнут предел итераций. Сходимость не подтверждена.")
            log.append(
                "  Возвращена лучшая найденная точка "
                "(с минимальной нормой полного градиента)."
            )
        log.append(f"Результат: x* = {format_point(x)}, f(x*) = {f_callable(*x):.6f}")

    if not converged and n_terms > 1:
        message = (
            "Достигнут предел итераций: из-за случайного выбора слагаемых "
            "критерий по чистому градиенту не достигнут. Показана лучшая "
            "найденная точка."
        )

    return RunResult(
        x_min=tuple(x),
        f_min=float(f_callable(*x)),
        iterations=iterations,
        log_lines=log,
        converged=converged,
        message=message,
    )
