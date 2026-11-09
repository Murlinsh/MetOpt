"""Тесты разбора выражений и символьных производных."""

from __future__ import annotations

import pytest

from optim_core.parser import (
    ParserError,
    build_gradient,
    build_hessian,
    build_hessian_from_gradient,
    compile_to_callable,
    parse_function,
    parse_gradient_text,
    split_additive_terms,
)

VARIABLES = ["x", "y"]


@pytest.mark.parametrize(
    "expression",
    [
        "sin(x) + cos(y) + 0.1*(x**2 + y**2)",
        "(x - 1)**2 + (y + 2)**2",
        "exp(-x)*sin(x) + 0.1*x**2",
        "sqrt(x**2 + y**2) + abs(x)",
        "log(x**2 + 1) + atan(y)",
        "x**2 + 4*y**2 + x*y",
    ],
)
def test_parse_valid_function(expression: str) -> None:
    assert parse_function(expression, VARIABLES) is not None


@pytest.mark.parametrize(
    "expression",
    [
        "x^2",
        "2x + y",
        "2(x + 1)",
        "e**x",
        "foo(x)",
        "sin x",
        "x(y)",
        "x + z",
        "sin",
        "sqrt",
        "x, y",
    ],
)
def test_parse_rejects_invalid_function(expression: str) -> None:
    with pytest.raises(ParserError):
        parse_function(expression, VARIABLES)


def test_caret_has_explanatory_message() -> None:
    with pytest.raises(ParserError, match="\\*\\*"):
        parse_function("x^2", VARIABLES)


def test_lowercase_e_has_explanatory_message() -> None:
    with pytest.raises(ParserError, match="E"):
        parse_function("e**x", VARIABLES)


def test_gradient_and_hessian_are_built_symbolically() -> None:
    expr = parse_function("sin(x) + cos(y) + 0.1*(x**2 + y**2)", VARIABLES)

    gradient = build_gradient(expr, VARIABLES)
    assert gradient is not None
    assert len(gradient) == 2

    hessian = build_hessian(expr, VARIABLES)
    assert hessian is not None
    assert len(hessian) == 2
    assert all(len(row) == 2 for row in hessian)

    from_gradient = build_hessian_from_gradient(gradient, VARIABLES)
    assert from_gradient is not None
    assert from_gradient == hessian


def test_abs_gradient_compiles_but_hessian_uses_numeric_fallback() -> None:
    expr = parse_function("abs(x) + y**2", VARIABLES)

    gradient = build_gradient(expr, VARIABLES)
    assert gradient is not None
    gradient_callable = compile_to_callable(gradient, VARIABLES)
    assert gradient_callable(1.0, 2.0) == pytest.approx((1.0, 4.0))
    assert gradient_callable(-1.0, 2.0) == pytest.approx((-1.0, 4.0))

    # Вторая производная Abs содержит DiracDelta, поэтому символьный
    # гессиан намеренно не строится — GUI перейдёт к численному.
    assert build_hessian(expr, VARIABLES) is None


@pytest.mark.parametrize(
    "text",
    [
        "[cos(x) + 0.2*x, -sin(y) + 0.2*y]",
        "[cos(x)+0.2*x,-sin(y)+0.2*y]",
        "[cos(x), -sin(y)]",
    ],
)
def test_parse_manual_gradient(text: str) -> None:
    gradient = parse_gradient_text(text, VARIABLES)
    assert len(gradient) == 2


@pytest.mark.parametrize(
    "text",
    ["cos(x), -sin(y)", "[cos(x)]", "[]", "[cos(x), ]", "[cos(x), -sin(y), 1]"],
)
def test_parse_manual_gradient_rejects_bad_format(text: str) -> None:
    with pytest.raises(ParserError):
        parse_gradient_text(text, VARIABLES)


def test_split_additive_terms_expands_expression() -> None:
    expr = parse_function("sin(x) + cos(y) + 0.1*(x**2 + y**2)", VARIABLES)
    terms = split_additive_terms(expr)
    assert len(terms) == 4
