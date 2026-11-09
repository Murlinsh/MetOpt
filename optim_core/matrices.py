"""Небольшие матричные операции без NumPy.

Поддерживаются матрицы размерности не выше 3×3 — этого достаточно для
двумерных методов оптимизации.
"""

from __future__ import annotations

from typing import Sequence

Matrix = list[list[float]]
Vector = list[float]

_EPS = 1e-14


def mat_identity(n: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def mat_add(a: Matrix, b: Matrix) -> Matrix:
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def mat_sub(a: Matrix, b: Matrix) -> Matrix:
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def mat_scale(a: Matrix, scalar: float) -> Matrix:
    return [[value * scalar for value in row] for row in a]


def mat_mul(a: Matrix, b: Matrix) -> Matrix:
    rows = len(a)
    inner = len(b)
    cols = len(b[0])
    return [
        [sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(cols)]
        for i in range(rows)
    ]


def mat_vec(a: Matrix, v: Sequence[float]) -> Vector:
    return [sum(row[j] * v[j] for j in range(len(v))) for row in a]


def mat_transpose(a: Matrix) -> Matrix:
    return [list(row) for row in zip(*a)]


def det(a: Matrix) -> float:
    n = len(a)
    if n == 1:
        return float(a[0][0])
    if n == 2:
        return float(a[0][0] * a[1][1] - a[0][1] * a[1][0])
    if n == 3:
        return float(
            a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
        )
    raise ValueError("Поддерживаются матрицы размерности не выше 3×3.")


def mat_inv(a: Matrix) -> Matrix:
    """Обращает матрицу 2×2 или 3×3.

    :raises ValueError: если матрица вырождена (определитель близок к нулю).
    """

    n = len(a)
    if n == 2:
        determinant = det(a)
        if abs(determinant) < _EPS:
            raise ValueError("Матрица вырождена: определитель близок к нулю.")
        return [
            [a[1][1] / determinant, -a[0][1] / determinant],
            [-a[1][0] / determinant, a[0][0] / determinant],
        ]

    if n == 3:
        determinant = det(a)
        if abs(determinant) < _EPS:
            raise ValueError("Матрица вырождена: определитель близок к нулю.")

        cofactors = [
            [
                ((-1) ** (i + j))
                * det(
                    [
                        [a[r][c] for c in range(3) if c != j]
                        for r in range(3)
                        if r != i
                    ]
                )
                for j in range(3)
            ]
            for i in range(3)
        ]
        # Присоединённая (adjugate) матрица — транспонированная матрица алгебраических дополнений.
        adjugate = mat_transpose(cofactors)
        return [[value / determinant for value in row] for row in adjugate]

    if n == 1:
        if abs(a[0][0]) < _EPS:
            raise ValueError("Матрица вырождена: элемент равен нулю.")
        return [[1.0 / a[0][0]]]

    raise ValueError("Поддерживаются матрицы размерности не выше 3×3.")


def solve_linear(a: Matrix, b: Sequence[float]) -> Vector:
    """Решает систему ``A·x = b`` через обратную матрицу."""

    return mat_vec(mat_inv(a), list(b))
