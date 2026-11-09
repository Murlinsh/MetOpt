"""Форматирование матриц и сборка текстовых файлов результатов."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


def _format_number(value: float, digits: int = 6) -> str:
    number = float(value)
    if abs(number) < 0.5 * 10 ** (-digits):
        number = 0.0
    return f"{number:.{digits}f}"


def format_vector(vector: Sequence[float], digits: int = 6) -> str:
    """Форматирует вектор в виде ``[a, b]``."""

    return "[" + ", ".join(_format_number(value, digits) for value in vector) + "]"


def format_matrix(matrix: Sequence[Sequence[float]], digits: int = 6) -> str:
    """Форматирует матрицу с выравниванием по столбцам."""

    if not matrix:
        return "[]"

    rows = [[_format_number(value, digits) for value in row] for row in matrix]
    widths = [
        max(len(row[column]) for row in rows)
        for column in range(len(rows[0]))
    ]
    lines = []
    for row in rows:
        cells = "  ".join(
            row[column].rjust(widths[column]) for column in range(len(row))
        )
        lines.append(f"[ {cells} ]")
    return "\n".join(lines)


def format_point(point: Sequence[float], digits: int = 6) -> str:
    return "(" + ", ".join(_format_number(value, digits) for value in point) + ")"


def _format_spec_params(params: Mapping[str, Any] | None) -> str:
    if not params:
        return "нет"
    visible = {
        key: value
        for key, value in params.items()
        if not str(key).startswith("_")
    }
    if not visible:
        return "нет"
    return ", ".join(f"{key} = {value}" for key, value in visible.items())


def build_result_file(
    meta: Mapping[str, Any],
    iterations_log: Iterable[str],
    result: Any,
) -> str:
    """Собирает полный текст файла результата.

    ``meta`` — словарь метаданных, ``iterations_log`` — строки лога,
    ``result`` — объект ``RunResult`` (или совместимый по полям объект).
    """

    mode = meta.get("mode", "answer")
    mode_label = (
        "Объяснение по итерациям"
        if mode == "explain"
        else "Готовый ответ"
    )

    lines: list[str] = ["=== МЕТАДАННЫЕ ==="]
    lines.append(f"Дата и время: {meta.get('datetime', '')}")
    lines.append(f"Метод: {meta.get('method', 'не выбран')}")
    lines.append(f"Режим: {mode_label}")
    lines.append(f"Функция: {meta.get('function', '')}")

    gradient = meta.get("gradient")
    if gradient:
        source = meta.get("gradient_source")
        suffix = f"   ({source})" if source else ""
        lines.append(f"Градиент: {gradient}{suffix}")

    hessian = meta.get("hessian")
    if hessian:
        source = meta.get("hessian_source")
        suffix = f"  ({source})" if source else ""
        lines.append(f"Гессиан: {hessian}{suffix}")

    lines.append(f"x0 = {meta.get('x0', 0.0)}, y0 = {meta.get('y0', 0.0)}")
    lines.append(f"Epsilon = {meta.get('epsilon', 0.0)}")
    lines.append(f"Спец. параметры: {_format_spec_params(meta.get('spec_params'))}")

    if not getattr(result, "converged", True):
        lines.append(
            "Предупреждение: достигнут предел итераций, сходимость не подтверждена."
        )
    warning = getattr(result, "message", "") or meta.get("warning")
    if warning:
        lines.append(f"Примечание: {warning}")

    lines.append("")
    lines.append("=== ИТЕРАЦИИ ===")
    log_text = "\n".join(iterations_log)
    if log_text.strip():
        lines.append(log_text)
    else:
        lines.append("<лог итераций не выводился: выбран режим «Готовый ответ»>")

    lines.append("")
    lines.append("=== РЕЗУЛЬТАТ ===")
    point = getattr(result, "x_min")
    lines.append("x* = " + format_point(point))
    lines.append(f"f(x*) = {_format_number(getattr(result, 'f_min'))}")
    lines.append(f"Итераций: {getattr(result, 'iterations')}")
    lines.append("")

    return "\n".join(lines)


def build_error_file(
    datetime_str: str,
    method: str,
    function_text: str,
    field: str,
    message: str,
) -> str:
    """Собирает текст файла для случая ошибки."""

    lines = [
        "=== ОШИБКА ===",
        f"Дата и время: {datetime_str}",
        f"Метод: {method or 'не выбран'}",
        f"Функция (введённая как есть): {function_text}",
        f"Поле, в котором ошибка: {field}",
        f"Сообщение: {message}",
        "",
    ]
    return "\n".join(lines)
