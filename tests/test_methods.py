"""Логические тесты для каждого метода оптимизации."""

from __future__ import annotations

import math

import pytest

from methods import registry
from tests.helpers import ALL_METHOD_IDS, Problem, make_params


EXPECTED_METHODS = {
    "hooke_jeeves",
    "nelder_mead",
    "ravine",
    "coordinate_descent",
    "steepest",
    "conjugate",
    "sgd",
    "newton",
    "broyden",
    "bfgs",
}


def _run(method_id: str, problem: Problem, start, mode: str, epsilon: float = 1e-4):
    spec = registry.get_method(method_id)
    assert spec is not None
    params = make_params(spec, problem, start, epsilon)
    return spec.run(problem.f, problem.gradient, problem.hessian, params, mode)


def test_registry_contains_all_expected_methods() -> None:
    assert {spec.id for spec in registry.METHODS} == EXPECTED_METHODS


def test_coordinate_descent_is_in_no_derivative_group() -> None:
    ids = [spec.id for spec in registry.methods_by_group("no_derivative")]
    assert "coordinate_descent" in ids


@pytest.mark.parametrize("method_id", ALL_METHOD_IDS)
def test_method_converges_on_quadratic(method_id: str, quadratic: Problem) -> None:
    start = (3.0, -2.0)
    result = _run(method_id, quadratic, start, "answer")

    assert result.converged, (
        f"{method_id} не сошёлся: x={result.x_min}, "
        f"f={result.f_min}, iterations={result.iterations}"
    )
    assert abs(result.x_min[0]) < 2e-2
    assert abs(result.x_min[1]) < 2e-2
    assert result.f_min < 1e-3
    assert result.f_min <= quadratic.f(*start) + 1e-9
    assert abs(result.f_min - quadratic.f(*result.x_min)) < 1e-9
    assert len(result.x_min) == 2
    assert result.iterations >= 0


@pytest.mark.parametrize("method_id", ALL_METHOD_IDS)
def test_method_finds_nonconvex_minimum(method_id: str, nonconvex: Problem) -> None:
    start = (1.0, 1.0)
    result = _run(method_id, nonconvex, start, "answer")

    assert result.f_min < nonconvex.f(*start) - 0.5
    assert result.f_min < -0.97
    assert abs(result.x_min[0] - (-1.30644)) < 5e-2
    assert abs(result.x_min[1] - 2.59574) < 5e-2


@pytest.mark.parametrize("method_id", ALL_METHOD_IDS)
def test_explain_mode_produces_iteration_log(method_id: str, quadratic: Problem) -> None:
    result = _run(method_id, quadratic, (3.0, -2.0), "explain")

    assert result.log_lines, f"{method_id} не сформировал лог итераций"
    assert any("Результат" in line for line in result.log_lines)


@pytest.mark.parametrize("method_id", ALL_METHOD_IDS)
def test_answer_mode_has_no_iteration_log(method_id: str, quadratic: Problem) -> None:
    result = _run(method_id, quadratic, (3.0, -2.0), "answer")

    assert result.log_lines == []


def test_coordinate_descent_solves_rotated_quadratic(
    rotated_quadratic: Problem,
) -> None:
    result = _run(
        "coordinate_descent",
        rotated_quadratic,
        (2.0, 1.0),
        "answer",
        epsilon=1e-5,
    )

    assert result.converged
    assert abs(result.x_min[0]) < 1e-4
    assert abs(result.x_min[1]) < 1e-4
    assert result.f_min < 1e-8


def test_coordinate_descent_works_without_derivatives(quadratic: Problem) -> None:
    spec = registry.get_method("coordinate_descent")
    params = make_params(spec, quadratic, (3.0, -2.0), epsilon=1e-5)
    result = spec.run(quadratic.f, None, None, params, "answer")

    assert result.converged
    assert abs(result.x_min[0]) < 1e-3
    assert abs(result.x_min[1]) < 1e-3


def test_sgd_reports_best_point_on_nonconvex_function(nonconvex: Problem) -> None:
    result = _run("sgd", nonconvex, (1.0, 1.0), "explain")

    assert not result.converged
    assert "лучшая" in result.message.lower()
    assert result.f_min < -0.97


# ----------------------------------------------------------------------
# Проверка формул отдельных методов
# ----------------------------------------------------------------------


def test_hooke_jeeves_pattern_point_formula() -> None:
    from methods.hooke_jeeves import _pattern_point

    x = [2.0, 3.0]
    x_previous = [1.0, 1.0]

    # p = x + λ·(x − x_previous)
    assert _pattern_point(x, x_previous, 1.0) == pytest.approx([3.0, 5.0])
    assert _pattern_point(x, x_previous, 0.5) == pytest.approx([2.5, 4.0])
    assert _pattern_point(x, x_previous, 0.0) == pytest.approx(x)


def test_hooke_jeeves_has_pattern_lambda_parameter() -> None:
    spec = registry.get_method("hooke_jeeves")
    assert spec is not None
    fields = {field.name: field for field in spec.input_fields}

    assert "lam" in fields
    assert fields["lam"].default == 1.0


@pytest.mark.parametrize("lam", [0.5, 1.0, 2.0])
def test_hooke_jeeves_converges_for_different_lambda(
    quadratic: Problem,
    lam: float,
) -> None:
    spec = registry.get_method("hooke_jeeves")
    params = make_params(spec, quadratic, (3.0, -2.0), epsilon=1e-4)
    params["lam"] = lam
    result = spec.run(quadratic.f, None, None, params, "answer")

    assert result.converged
    assert abs(result.x_min[0]) < 2e-2
    assert abs(result.x_min[1]) < 2e-2


def test_nelder_mead_sigma_is_standard_deviation() -> None:
    from methods.nelder_mead import _function_sigma

    assert _function_sigma([1.0, 1.0, 1.0]) == 0.0
    assert _function_sigma([0.0, 2.0, 4.0]) == pytest.approx(math.sqrt(8.0 / 3.0))


def test_golden_section_finds_quadratic_minimum() -> None:
    from methods.coordinate_descent import _golden_section

    best = _golden_section(lambda t: (t - 3.0) ** 2, -5.0, 10.0, 1e-8)

    assert best == pytest.approx(3.0, abs=1e-6)


@pytest.mark.parametrize("minimum", [-10.0, 10.0])
def test_bracket_minimum_contains_minimum(minimum: float) -> None:
    from methods.coordinate_descent import _bracket_minimum

    left, right = _bracket_minimum(lambda t: (t - minimum) ** 2, 0.0, 1.0)

    assert left < minimum < right


def test_coordinate_descent_is_stationary_along_each_axis(
    rotated_quadratic: Problem,
) -> None:
    result = _run(
        "coordinate_descent",
        rotated_quadratic,
        (2.0, 1.0),
        "answer",
        epsilon=1e-7,
    )

    assert result.converged
    gradient = rotated_quadratic.gradient(*result.x_min)
    assert gradient is not None
    assert abs(gradient[0]) < 1e-4
    assert abs(gradient[1]) < 1e-4
