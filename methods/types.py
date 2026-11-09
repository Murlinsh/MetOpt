"""Общие типы данных для методов оптимизации."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class FieldSpec:
    """Описание дополнительного поля ввода для конкретного метода."""

    name: str
    type: str  # "float" | "int" | "bool"
    default: Any
    label: str
    description: str = ""


@dataclass
class RunResult:
    """Результат работы метода."""

    x_min: tuple[float, ...]
    f_min: float
    iterations: int
    log_lines: list[str] = field(default_factory=list)
    converged: bool = True
    message: str = ""


@dataclass
class MethodSpec:
    """Описание метода в реестре."""

    id: str
    group: str  # "no_derivative" | "gradient" | "newton"
    name: str
    dim: int
    input_fields: list[FieldSpec]
    run: Callable
    needs_gradient: bool = False
    needs_hessian: bool = False
    subgroup: str | None = None
