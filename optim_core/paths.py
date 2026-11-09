"""Пути проекта.

Все пути вычисляются относительно корня проекта, поэтому приложение можно
запускать из любого каталога.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
LAST_INPUT_FILE = PROJECT_ROOT / "last_input.json"
SYNTAX_FILE = PROJECT_ROOT / "SYNTAX.md"
