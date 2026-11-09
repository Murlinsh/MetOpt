"""Сохранение состояния и файлов результатов."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import LAST_INPUT_FILE, RESULTS_DIR


def make_timestamp() -> str:
    """Возвращает дату-время в формате, пригодном для имени файла."""

    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def load_last_input() -> dict[str, Any]:
    """Читает ``last_input.json``. При любой проблеме возвращает пустой словарь."""

    try:
        with LAST_INPUT_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}

    if not isinstance(data, dict):
        return {}

    common = data.get("common")
    if not isinstance(common, dict):
        data["common"] = {}
    per_method = data.get("per_method")
    if not isinstance(per_method, dict):
        data["per_method"] = {}
    if not isinstance(data.get("last_method_id"), str):
        data["last_method_id"] = None
    return data


def save_last_input(data: dict[str, Any]) -> None:
    """Записывает ``last_input.json`` в корень проекта."""

    LAST_INPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LAST_INPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def write_result_text(text: str, timestamp: str | None = None) -> Path:
    """Создаёт ``results/<timestamp>.txt`` и возвращает путь к файлу."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base = timestamp or make_timestamp()
    path = RESULTS_DIR / f"{base}.txt"
    counter = 2
    while path.exists():
        path = RESULTS_DIR / f"{base}-{counter}.txt"
        counter += 1
    path.write_text(text, encoding="utf-8")
    return path
