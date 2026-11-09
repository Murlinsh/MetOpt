"""Логические тесты для каждого метода оптимизации."""

from __future__ import annotations

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
