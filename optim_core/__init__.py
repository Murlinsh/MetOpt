"""Пакет с математическим ядром приложения."""

from .parser import (
    ParserError,
    build_gradient,
    build_hessian,
    build_hessian_from_gradient,
    compile_function,
    compile_to_callable,
    parse_function,
    parse_gradient_text,
    split_additive_terms,
)
from .numeric import numeric_gradient, numeric_hessian
from .matrices import (
    det,
    mat_add,
    mat_identity,
    mat_inv,
    mat_mul,
    mat_vec,
    solve_linear,
)

__all__ = [
    "ParserError",
    "parse_function",
    "build_gradient",
    "build_hessian",
    "build_hessian_from_gradient",
    "parse_gradient_text",
    "compile_to_callable",
    "compile_function",
    "split_additive_terms",
    "numeric_gradient",
    "numeric_hessian",
    "mat_mul",
    "mat_vec",
    "mat_inv",
    "solve_linear",
    "mat_add",
    "mat_identity",
    "det",
]
