"""Парсинг пользовательских выражений и символьное дифференцирование.

Модуль отвечает за:

* разбор строки с функцией ``f(x, y)``;
* проверку допустимого синтаксиса (без ``^``, без неявного умножения и т.п.);
* построение градиента и матрицы Гессе через SymPy;
* компиляцию символьных выражений в обычные Python-функции.

Все ошибки ввода поднимаются как :class:`ParserError` с человекочитаемым
текстом на русском языке.
"""

from __future__ import annotations

import io
import token
import tokenize
from typing import Iterable, Sequence

import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

# ---------------------------------------------------------------------------
# Допустимые имена
# ---------------------------------------------------------------------------

ALLOWED_FUNCTIONS: dict[str, object] = {
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "asin": sympy.asin,
    "acos": sympy.acos,
    "atan": sympy.atan,
    "exp": sympy.exp,
    "log": sympy.log,
    "sqrt": sympy.sqrt,
    "abs": sympy.Abs,
}

ALLOWED_CONSTANTS: dict[str, object] = {
    "pi": sympy.pi,
    "E": sympy.E,
}

ALLOWED_OPERATORS = {"+", "-", "*", "/", "**", "(", ")"}

_IGNORED_TOKEN_TYPES = {
    token.ENDMARKER,
    token.NEWLINE,
    token.NL,
    token.INDENT,
    token.DEDENT,
}


class ParserError(Exception):
    """Ошибка разбора пользовательского выражения."""


# ---------------------------------------------------------------------------
# Проверка синтаксиса
# ---------------------------------------------------------------------------


def _tokenize(expression: str) -> list[tokenize.TokenInfo]:
    try:
        return list(tokenize.generate_tokens(io.StringIO(expression).readline))
    except (tokenize.TokenError, IndentationError) as exc:
        raise ParserError(
            "Не удалось разобрать выражение. Проверьте расстановку скобок "
            "и правильность записи. Подробная шпаргалка — кнопка «Синтаксис»."
        ) from exc


def _significant_tokens(expression: str) -> list[tokenize.TokenInfo]:
    return [
        tok
        for tok in _tokenize(expression)
        if tok.type not in _IGNORED_TOKEN_TYPES
    ]


def _is_operand_end(tok: tokenize.TokenInfo) -> bool:
    return tok.type in (token.NUMBER, token.NAME) or (
        tok.type == token.OP and tok.string == ")"
    )


def _is_operand_start(tok: tokenize.TokenInfo) -> bool:
    return tok.type in (token.NUMBER, token.NAME) or (
        tok.type == token.OP and tok.string == "("
    )


def _validate_syntax(expression: str, variables: Sequence[str]) -> None:
    """Проверяет имена и неявное умножение до вызова SymPy."""

    if "^" in expression:
        raise ParserError(
            "Символ «^» не допускается. Для степени используйте «**»: "
            "например, x**2 вместо x^2. Шпаргалка — кнопка «Синтаксис»."
        )

    if not expression.strip():
        raise ParserError("Выражение пустое. Введите функцию, например: sin(x) + cos(y).")

    tokens = _significant_tokens(expression)
    allowed_names = set(ALLOWED_FUNCTIONS) | set(ALLOWED_CONSTANTS) | set(variables)

    for index, tok in enumerate(tokens):
        if tok.type == token.NAME:
            name = tok.string
            if name not in allowed_names:
                if name == "e":
                    raise ParserError(
                        "Символ «e» не распознан. Константа Эйлера записывается "
                        "заглавной буквой E. Если нужна экспонента, используйте exp(x)."
                    )
                raise ParserError(
                    f"Неизвестный символ «{name}». Разрешены переменные "
                    f"{', '.join(variables)}, функции sin, cos, tan, asin, acos, "
                    "atan, exp, log, sqrt, abs и константы pi, E. "
                    "Шпаргалка — кнопка «Синтаксис»."
                )

        elif tok.type == token.OP:
            if tok.string not in ALLOWED_OPERATORS:
                raise ParserError(
                    f"Оператор «{tok.string}» не допускается. Используйте "
                    "обычные арифметические операции +, -, *, /, ** и круглые скобки."
                )

        elif tok.type == token.NUMBER:
            continue

        else:
            raise ParserError(
                f"Не удалось распознать фрагмент «{tok.string}». "
                "Проверьте запись выражения. Шпаргалка — кнопка «Синтаксис»."
            )

        # Проверка неявного умножения и вызова не-функции.
        if index == 0:
            continue
        prev = tokens[index - 1]

        if prev.type == token.NAME and tok.type == token.OP and tok.string == "(":
            if prev.string not in ALLOWED_FUNCTIONS:
                raise ParserError(
                    f"«{prev.string}» не является функцией. Для умножения "
                    "используйте явный знак «*», например: 2*x."
                )
            continue

        if _is_operand_end(prev) and _is_operand_start(tok):
            raise ParserError(
                "Похоже, пропущен знак умножения. Умножение должно быть явным: "
                "пишите 2*x, а не 2x; 2*(x + 1), а не 2(x + 1). "
                "Шпаргалка — кнопка «Синтаксис»."
            )


# ---------------------------------------------------------------------------
# Парсинг
# ---------------------------------------------------------------------------


def _symbol(name: str) -> sympy.Symbol:
    """Создаёт вещественный символ: производные от abs/sqrt получаются проще."""

    return sympy.Symbol(name, real=True)


def _local_dict(variables: Iterable[str]) -> dict[str, object]:
    local: dict[str, object] = {name: _symbol(name) for name in variables}
    local.update(ALLOWED_FUNCTIONS)
    local.update(ALLOWED_CONSTANTS)
    return local


def parse_function(s: str, variables: list[str]) -> sympy.Expr:
    """Разбирает строку с функцией и возвращает символьное выражение.

    :raises ParserError: если выражение пустое, содержит недопустимые символы
        или не может быть разобрано.
    """

    expression = s.strip()
    _validate_syntax(expression, variables)

    try:
        expr = parse_expr(
            expression,
            local_dict=_local_dict(variables),
            transformations=standard_transformations,
            evaluate=True,
        )
    except (sympy.SympifyError, SyntaxError, TypeError, ValueError) as exc:
        raise ParserError(
            "Не удалось разобрать выражение. Проверьте запись: степень — **, "
            "умножение — явное (2*x), функции записываются со скобками "
            "(sin(x)). Шпаргалка — кнопка «Синтаксис»."
        ) from exc

    if isinstance(expr, sympy.FunctionClass) or callable(expr):
        raise ParserError(
            "Функция должна быть применена к аргументу, например sin(x), "
            "а не записана отдельным именем."
        )

    if isinstance(expr, sympy.Tuple):
        raise ParserError(
            "Ожидалось одно выражение, а не список. Для градиента используется "
            "отдельное поле ввода формата [∂f/∂x, ∂f/∂y]."
        )

    unknown = expr.free_symbols - {_symbol(name) for name in variables}
    if unknown:
        names = ", ".join(sorted(str(sym) for sym in unknown))
        raise ParserError(
            f"Неизвестные символы: {names}. Разрешены только переменные "
            f"{', '.join(variables)} и константы pi, E."
        )

    undefined_functions = expr.atoms(sympy.core.function.AppliedUndef)
    if undefined_functions:
        names = ", ".join(sorted(str(fn.func) for fn in undefined_functions))
        raise ParserError(
            f"Неизвестные функции: {names}. Разрешены sin, cos, tan, asin, acos, "
            "atan, exp, log, sqrt, abs."
        )

    return expr


def _contains_unsupported(exprs: Sequence[sympy.Expr]) -> bool:
    """Проверяет, есть ли в выражениях объекты, которые не компилируются."""

    unsupported = (sympy.Derivative, sympy.DiracDelta, sympy.Heaviside)
    return any(expr.has(*unsupported) for expr in exprs)


def build_gradient(expr: sympy.Expr, variables: list[str]) -> list[sympy.Expr] | None:
    """Пытается построить градиент символьно. При неудаче возвращает ``None``."""

    try:
        gradient = [sympy.diff(expr, _symbol(name)) for name in variables]
    except Exception:
        return None
    if _contains_unsupported(gradient):
        return None
    return gradient


def build_hessian(expr: sympy.Expr, variables: list[str]) -> list[list[sympy.Expr]] | None:
    """Пытается построить матрицу Гессе символьно. При неудаче — ``None``."""

    try:
        symbols = [_symbol(name) for name in variables]
        hessian = [
            [sympy.diff(expr, first, second) for second in symbols]
            for first in symbols
        ]
    except Exception:
        return None
    if _contains_unsupported([value for row in hessian for value in row]):
        return None
    return hessian


def build_hessian_from_gradient(
    gradient: list[sympy.Expr], variables: list[str]
) -> list[list[sympy.Expr]] | None:
    """Строит гессиан, дифференцируя уже готовый градиент."""

    try:
        symbols = [_symbol(name) for name in variables]
        hessian = [
            [sympy.diff(component, var) for var in symbols]
            for component in gradient
        ]
    except Exception:
        return None
    if _contains_unsupported([value for row in hessian for value in row]):
        return None
    return hessian


def _split_top_level(text: str) -> list[str]:
    """Делит строку по запятым верхнего уровня (вне скобок)."""

    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    parts.append("".join(current).strip())
    return parts


def parse_gradient_text(s: str, variables: list[str]) -> list[sympy.Expr]:
    """Разбирает ручной ввод градиента вида ``[∂f/∂x, ∂f/∂y]``."""

    text = s.strip()
    if not text:
        raise ParserError("Поле градиента пустое. Введите, например: [cos(x), -sin(y)].")

    if not (text.startswith("[") and text.endswith("]")):
        raise ParserError(
            "Градиент должен быть заключён в квадратные скобки: "
            "[∂f/∂x, ∂f/∂y]. Пример: [cos(x) + 0.2*x, -sin(y) + 0.2*y]."
        )

    inner = text[1:-1].strip()
    if not inner:
        raise ParserError("Внутри квадратных скобок нет выражений. Введите два выражения через запятую.")

    parts = _split_top_level(inner)
    if len(parts) != len(variables):
        raise ParserError(
            f"Ожидалось {len(variables)} выражения через запятую, "
            f"а получено {len(parts)}. Формат: [∂f/∂x, ∂f/∂y]."
        )

    if any(not part for part in parts):
        raise ParserError("Одно из выражений градиента пустое. Проверьте запятые.")

    return [parse_function(part, variables) for part in parts]


# ---------------------------------------------------------------------------
# Компиляция в callable
# ---------------------------------------------------------------------------


def compile_to_callable(
    exprs: Sequence[sympy.Expr] | sympy.Expr, variables: list[str]
) -> callable:
    """Компилирует выражение или список выражений в функцию.

    Возвращаемая функция принимает координаты как отдельные аргументы
    (``fn(x, y)``) и всегда возвращает кортеж чисел.
    """

    if isinstance(exprs, (list, tuple)):
        expressions = list(exprs)
    else:
        expressions = [exprs]

    try:
        compiled = sympy.lambdify(variables, expressions, modules=["math"])
    except Exception as exc:  # pragma: no cover - защита от экзотических выражений
        raise ParserError(f"Не удалось скомпилировать выражение: {exc}") from exc

    def wrapper(*args: float) -> tuple[float, ...]:
        if len(args) != len(variables):
            raise TypeError(
                f"Ожидалось {len(variables)} аргумента, получено {len(args)}."
            )
        try:
            values = compiled(*args)
        except Exception as exc:
            raise ValueError(
                f"Не удалось вычислить функцию в точке {tuple(args)}: {exc}"
            ) from exc
        if not isinstance(values, (list, tuple)):
            values = [values]
        return tuple(float(value) for value in values)

    return wrapper


def compile_function(expr: sympy.Expr, variables: list[str]) -> callable:
    """Компилирует скалярное выражение в функцию, возвращающую число."""

    compiled = compile_to_callable([expr], variables)

    def wrapper(*args: float) -> float:
        return compiled(*args)[0]

    return wrapper


def split_additive_terms(expr: sympy.Expr) -> list[sympy.Expr]:
    """Раскладывает выражение на сумму слагаемых (для SGD).

    Выражение предварительно раскрывается, чтобы конструкции вида
    ``0.1*(x**2 + y**2)`` распались на отдельные слагаемые.
    """

    try:
        expanded = sympy.expand(expr)
        terms = list(sympy.Add.make_args(expanded))
    except Exception:
        terms = [expr]
    return terms or [expr]
