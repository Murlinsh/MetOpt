"""Вспомогательные функции для тестов методов оптимизации."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from methods import registry
from methods.types import MethodSpec
from optim_core.numeric import numeric_gradient, numeric_hessian
from optim_core.parser import (
    build_gradient,
    build_hessian,
    compile_function,
    compile_to_callable,
    parse_function,
    split_additive_terms,
)

VARIABLES = ["x", "y"]


@dataclass
class Problem:
    """Набор вычислимых функций для одной тестовой задачи."""

    f: Callable
    gradient: Callable | None
    hessian: Callable | None
    terms: list[tuple[Callable, Callable]]


def build_problem(expression: str) -> Problem:
    """Строит функцию, градиент, гессиан и слагаемые для SGD."""

    expr = parse_function(expression, VARIABLES)
    f = compile_function(expr, VARIABLES)

    gradient_exprs = build_gradient(expr, VARIABLES)
    gradient = (
        compile_to_callable(gradient_exprs, VARIABLES)
        if gradient_exprs is not None
        else None
    )

    hessian_exprs = build_hessian(expr, VARIABLES)
    if hessian_exprs is not None:
        flat = [
            hessian_exprs[0][0],
            hessian_exprs[0][1],
            hessian_exprs[1][0],
            hessian_exprs[1][1],
        ]
        compiled_hessian = compile_to_callable(flat, VARIABLES)

        def hessian(*args):
            values = compiled_hessian(*args)
            return [[values[0], values[1]], [values[2], values[3]]]
    else:
        def hessian(*args):
            return numeric_hessian(f, list(args))

    terms: list[tuple[Callable, Callable]] = []
    for term in split_additive_terms(expr):
        term_function = compile_function(term, VARIABLES)
        term_gradient_exprs = build_gradient(term, VARIABLES)
        if term_gradient_exprs is None:
            def term_gradient(*args, _function=term_function):
                return numeric_gradient(_function, list(args))
        else:
            term_gradient = compile_to_callable(term_gradient_exprs, VARIABLES)
        terms.append((term_function, term_gradient))

    return Problem(f=f, gradient=gradient, hessian=hessian, terms=terms)


def make_params(
    spec: MethodSpec,
    problem: Problem,
    start: tuple[float, float],
    epsilon: float = 1e-4,
) -> dict:
    """Собирает словарь параметров так, как это делает GUI."""

    params = {
        "x0": start[0],
        "y0": start[1],
        "epsilon": epsilon,
    }
    for field in spec.input_fields:
        params[field.name] = field.default
    if spec.id == "sgd":
        params["_terms"] = problem.terms
        params["_n_terms"] = len(problem.terms)
    return params


ALL_METHOD_IDS = [spec.id for spec in registry.METHODS]
