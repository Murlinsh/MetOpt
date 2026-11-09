"""Тесты формирования файлов результатов и ошибок."""

from __future__ import annotations

from types import SimpleNamespace

from optim_core.logging_utils import build_error_file, build_result_file


def _result(**overrides):
    data = {
        "x_min": (-1.25, 2.5),
        "f_min": -0.5,
        "iterations": 7,
        "converged": True,
        "message": "",
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def _meta(**overrides):
    data = {
        "datetime": "2026-01-01T12-00-00",
        "method": "Тестовый метод",
        "mode": "explain",
        "function": "x**2 + y**2",
        "gradient": "[2*x, 2*y]",
        "gradient_source": "символьный",
        "x0": 1.0,
        "y0": 2.0,
        "epsilon": 1e-4,
        "spec_params": {},
    }
    data.update(overrides)
    return data


def test_result_file_contains_metadata_and_answer() -> None:
    text = build_result_file(_meta(), ["=== Итерация 1 ==="], _result())

    assert "=== МЕТАДАННЫЕ ===" in text
    assert "Метод: Тестовый метод" in text
    assert "Режим: Объяснение по итерациям" in text
    assert "=== ИТЕРАЦИИ ===" in text
    assert "=== Итерация 1 ===" in text
    assert "=== РЕЗУЛЬТАТ ===" in text
    assert "x* = (-1.250000, 2.500000)" in text
    assert "f(x*) = -0.500000" in text
    assert "Итераций: 7" in text


def test_answer_mode_notes_that_log_was_not_printed() -> None:
    text = build_result_file(_meta(mode="answer"), [], _result())

    assert "Режим: Готовый ответ" in text
    assert "лог итераций не выводился" in text


def test_result_file_contains_warning_for_nonconverged_run() -> None:
    text = build_result_file(
        _meta(),
        [],
        _result(converged=False, message="Показана лучшая точка."),
    )

    assert "Предупреждение: достигнут предел итераций" in text
    assert "Примечание: Показана лучшая точка." in text


def test_error_file_contains_input_context() -> None:
    text = build_error_file(
        "2026-01-01T12-00-00",
        "Метод Хука-Дживса",
        "x^2 + y^2",
        "Функция f(x, y)",
        "Символ «^» не допускается.",
    )

    assert "=== ОШИБКА ===" in text
    assert "Метод: Метод Хука-Дживса" in text
    assert "Функция (введённая как есть): x^2 + y^2" in text
    assert "Поле, в котором ошибка: Функция f(x, y)" in text
    assert "Символ «^» не допускается." in text
