"""Общие фикстуры для тестов."""

from __future__ import annotations

import random

import pytest

from tests.helpers import Problem, build_problem


@pytest.fixture(autouse=True)
def deterministic_random() -> None:
    """Фиксирует генератор случайных чисел для воспроизводимости SGD."""

    random.seed(0)


@pytest.fixture(scope="session")
def quadratic() -> Problem:
    return build_problem("x**2 + 4*y**2")


@pytest.fixture(scope="session")
def rotated_quadratic() -> Problem:
    return build_problem("x**2 + y**2 + x*y")


@pytest.fixture(scope="session")
def nonconvex() -> Problem:
    return build_problem("sin(x) + cos(y) + 0.1*(x**2 + y**2)")
