"""Главное окно приложения: тулбар, статусная строка и четыре страницы."""

from __future__ import annotations

from pathlib import Path

import sympy
from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QAction, QDesktopServices
from PyQt6.QtWidgets import (
    QDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QToolBar,
)

from optim_core.logging_utils import build_error_file, build_result_file
from optim_core.numeric import numeric_gradient, numeric_hessian
from optim_core.parser import (
    ParserError,
    build_gradient,
    build_hessian,
    build_hessian_from_gradient,
    compile_function,
    compile_to_callable,
    parse_function,
    parse_gradient_text,
    split_additive_terms,
)
from optim_core.paths import ALGORITHMS_FILE, RESULTS_DIR, SYNTAX_FILE
from optim_core import storage

from methods import registry

from .help_dialog import HelpDialog
from .page_input import InputPage
from .page_method import MethodPage
from .page_mode import ModePage
from .page_result import ResultPage
from .widgets import GradientDialog, InputError

VARIABLES = ["x", "y"]

ALGORITHM_HEADINGS = {
    "hooke_jeeves": "1. Метод Хука-Дживса",
    "nelder_mead": "2. Метод Нелдера-Мида",
    "ravine": "3. Метод оврагов",
    "coordinate_descent": "4. Покоординатный спуск",
    "steepest": "5. Наискорейший градиентный спуск",
    "conjugate": "6. Метод сопряжённых градиентов",
    "sgd": "7. Стохастический градиентный спуск",
    "newton": "8. Классический метод Ньютона",
    "broyden": "9. Метод Бройдена",
    "bfgs": "10. Метод BFGS",
}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optimization Toolkit — методы оптимизации")
        self.resize(1050, 780)

        self.last_input = storage.load_last_input()
        self.current_spec = None
        self.current_context: dict | None = None

        self._build_toolbar()
        self._build_pages()
        self._build_status_bar()
        self._connect_signals()

        last_method_id = self.last_input.get("last_method_id")
        self.method_page.select_method(last_method_id)
        self.stack.setCurrentWidget(self.method_page)

    # ------------------------------------------------------------------
    # Построение интерфейса
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Инструменты", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        syntax_action = QAction("Синтаксис", self)
        syntax_action.setToolTip("Открыть шпаргалку по синтаксису")
        syntax_action.triggered.connect(self._open_syntax)
        toolbar.addAction(syntax_action)

        algorithms_action = QAction("Алгоритмы", self)
        algorithms_action.setToolTip(
            "Открыть подробное пояснение того, как работает каждый метод"
        )
        algorithms_action.triggered.connect(lambda: self._open_algorithms())
        toolbar.addAction(algorithms_action)

        results_action = QAction("Папка результатов", self)
        results_action.setToolTip("Открыть папку с результатами")
        results_action.triggered.connect(self._open_results)
        toolbar.addAction(results_action)

        toolbar.addSeparator()

        about_action = QAction("О программе", self)
        about_action.triggered.connect(self._show_about)
        toolbar.addAction(about_action)

    def _build_pages(self) -> None:
        self.stack = QStackedWidget()
        self.method_page = MethodPage()
        self.input_page = InputPage()
        self.mode_page = ModePage()
        self.result_page = ResultPage()

        self.stack.addWidget(self.method_page)
        self.stack.addWidget(self.input_page)
        self.stack.addWidget(self.mode_page)
        self.stack.addWidget(self.result_page)
        self.setCentralWidget(self.stack)

    def _build_status_bar(self) -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        label = QLabel(f"Результаты сохраняются в: {RESULTS_DIR}")
        self.statusBar().addPermanentWidget(label)

    def _connect_signals(self) -> None:
        self.method_page.next_requested.connect(self._method_next)
        self.method_page.exit_requested.connect(self.close)
        self.method_page.help_requested.connect(self._open_method_help)

        self.input_page.back_requested.connect(
            lambda: self.stack.setCurrentWidget(self.method_page)
        )
        self.input_page.next_requested.connect(self._input_next)

        self.mode_page.back_requested.connect(
            lambda: self.stack.setCurrentWidget(self.input_page)
        )
        self.mode_page.next_requested.connect(self._run)

        self.result_page.same_method_requested.connect(self._same_method)
        self.result_page.change_method_requested.connect(self._change_method)
        self.result_page.exit_requested.connect(self.close)

    # ------------------------------------------------------------------
    # Тулбар
    # ------------------------------------------------------------------

    def _open_syntax(self) -> None:
        self._open_document(SYNTAX_FILE, "Синтаксис")

    def _open_algorithms(self, search_text: str | None = None) -> None:
        self._open_document(
            ALGORITHMS_FILE,
            "Как работают методы оптимизации",
            search_text=search_text,
        )

    def _open_method_help(self) -> None:
        method_id = self.method_page.selected_method_id()
        search_text = ALGORITHM_HEADINGS.get(method_id) if method_id else None
        self._open_algorithms(search_text=search_text)

    def _open_document(
        self,
        path: Path,
        title: str,
        search_text: str | None = None,
    ) -> None:
        if not path.exists():
            QMessageBox.warning(
                self,
                "Файл не найден",
                f"Не удалось найти файл документа:\n{path}",
            )
            return

        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(
                self,
                "Не удалось открыть документ",
                f"Не удалось прочитать файл:\n{path}\n\n{exc}",
            )
            return

        dialog = HelpDialog(title, text, self, search_text=search_text)
        dialog.exec()

    def _open_results(self) -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(RESULTS_DIR)))
        if not opened:
            QMessageBox.warning(
                self,
                "Не удалось открыть папку",
                f"Папка результатов:\n{RESULTS_DIR}",
            )

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "О программе",
            "<h3>Optimization Toolkit</h3>"
            "<p>Приложение для решения задач оптимизации функций двух переменных.</p>"
            "<p>Реализованы методы без производных, градиентные методы и методы "
            "Ньютона. Градиент и матрица Гессе строятся автоматически через SymPy.</p>"
            "<p>Результаты сохраняются в папку <code>results/</code> рядом с "
            "программой.</p>",
        )

    # ------------------------------------------------------------------
    # Навигация и расчёт
    # ------------------------------------------------------------------

    def _method_next(self) -> None:
        method_id = self.method_page.selected_method_id()
        if not method_id:
            QMessageBox.warning(
                self,
                "Метод не выбран",
                "Выберите метод в дереве, затем нажмите «Далее».",
            )
            return

        spec = registry.get_method(method_id)
        if spec is None:
            QMessageBox.critical(self, "Ошибка", f"Неизвестный метод: {method_id}")
            return

        self.current_spec = spec
        self.input_page.prepare(spec, self.last_input)
        self.stack.setCurrentWidget(self.input_page)

    def _input_next(self) -> None:
        if self.current_spec is None:
            return

        try:
            raw = self.input_page.collect()
        except InputError as exc:
            self._report_input_error(
                exc.field, exc.message, self.input_page.f_edit.text()
            )
            return

        try:
            f_expr = parse_function(raw["f_text"], VARIABLES)
        except ParserError as exc:
            self._report_input_error("Функция f(x, y)", str(exc), raw["f_text"])
            return

        try:
            f_callable = compile_function(f_expr, VARIABLES)
        except ParserError as exc:
            self._report_input_error("Функция f(x, y)", str(exc), raw["f_text"])
            return
        except Exception as exc:
            self._report_input_error("Функция f(x, y)", str(exc), raw["f_text"])
            return

        grad_exprs = None
        grad_callable = None
        grad_source = None
        manual_text = None

        if self.current_spec.needs_gradient:
            candidate = build_gradient(f_expr, VARIABLES)
            if candidate is not None:
                try:
                    grad_callable = compile_to_callable(candidate, VARIABLES)
                except (ParserError, ValueError, TypeError):
                    grad_callable = None
                else:
                    grad_exprs = candidate
                    grad_source = "символьный"

            if grad_callable is None:
                gradient_result = self._ask_gradient(raw)
                if gradient_result is None:
                    return
                grad_exprs, manual_text = gradient_result
                try:
                    grad_callable = compile_to_callable(grad_exprs, VARIABLES)
                except (ParserError, ValueError, TypeError) as exc:
                    self._report_input_error("Градиент", str(exc), raw["f_text"])
                    return
                grad_source = "ручной"

        hess_exprs = None
        hess_source = None
        hess_callable = None
        if self.current_spec.needs_hessian:
            direct_hessian = None
            if grad_source == "символьный":
                direct_hessian = build_hessian(f_expr, VARIABLES)

            if direct_hessian is not None:
                try:
                    hess_callable = self._make_hessian_callable(
                        direct_hessian, f_callable
                    )
                except (ParserError, ValueError, TypeError):
                    hess_callable = None
                else:
                    hess_exprs = direct_hessian
                    hess_source = "символьный"

            if hess_callable is None and grad_exprs is not None:
                hess_from_grad = build_hessian_from_gradient(grad_exprs, VARIABLES)
                if hess_from_grad is not None:
                    try:
                        hess_callable = self._make_hessian_callable(
                            hess_from_grad, f_callable
                        )
                    except (ParserError, ValueError, TypeError):
                        hess_callable = None
                    else:
                        hess_exprs = hess_from_grad
                        hess_source = "символьный (из градиента)"

            if hess_callable is None:
                hess_source = "численный"

        if hess_callable is None and self.current_spec.needs_hessian:
            hess_callable = lambda *args: numeric_hessian(f_callable, list(args))

        # Проверяем, что функция и градиент вычислимы в начальной точке.
        try:
            f_callable(raw["x0"], raw["y0"])
            if grad_callable is not None:
                grad_callable(raw["x0"], raw["y0"])
        except Exception as exc:
            self._report_input_error(
                "Начальная точка",
                f"Не удалось вычислить функцию или градиент в начальной точке "
                f"({raw['x0']}, {raw['y0']}): {exc}",
                raw["f_text"],
            )
            return

        extra_params: dict = {}
        if self.current_spec.id == "sgd":
            try:
                terms = split_additive_terms(f_expr)
                prepared_terms = []
                for term in terms:
                    term_function = compile_function(term, VARIABLES)
                    term_gradient_exprs = build_gradient(term, VARIABLES)
                    if term_gradient_exprs is None:
                        def term_gradient(*args, _function=term_function):
                            return numeric_gradient(_function, list(args))
                    else:
                        term_gradient = compile_to_callable(
                            term_gradient_exprs, VARIABLES
                        )
                    prepared_terms.append((term_function, term_gradient))
                extra_params["_terms"] = prepared_terms
                extra_params["_n_terms"] = len(prepared_terms)
            except Exception as exc:
                self._report_input_error(
                    "Функция f(x, y)",
                    f"Не удалось подготовить слагаемые для SGD: {exc}",
                    raw["f_text"],
                )
                return

        raw["gradient_manual"] = manual_text
        self._save_last_input(raw)

        self.current_context = {
            "raw": raw,
            "f_expr": f_expr,
            "grad_exprs": grad_exprs,
            "grad_source": grad_source,
            "hess_exprs": hess_exprs,
            "hess_source": hess_source,
            "f_callable": f_callable,
            "grad_callable": grad_callable,
            "hess_callable": hess_callable,
            "extra_params": extra_params,
        }

        common = self.last_input.get("common", {})
        mode = common.get("mode", "explain") if isinstance(common, dict) else "explain"
        self.mode_page.prepare(mode)
        self.stack.setCurrentWidget(self.mode_page)

    def _ask_gradient(self, raw: dict):
        common = self.last_input.get("common", {})
        initial = ""
        if isinstance(common, dict):
            initial = common.get("gradient_manual") or ""

        while True:
            dialog = GradientDialog(self, initial_text=initial)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return None

            text = dialog.text()
            if not text:
                self._report_input_error(
                    "Градиент",
                    "Поле градиента пустое. Введите два выражения в квадратных "
                    "скобках, например: [cos(x), -sin(y)].",
                    raw["f_text"],
                )
                initial = text
                continue

            try:
                exprs = parse_gradient_text(text, VARIABLES)
                compiled = compile_to_callable(exprs, VARIABLES)
                compiled(raw["x0"], raw["y0"])
            except (ParserError, ValueError, TypeError) as exc:
                self._report_input_error("Градиент", str(exc), raw["f_text"])
                initial = text
                continue

            return exprs, text

    def _make_hessian_callable(self, hess_exprs, f_callable):
        flat = [
            hess_exprs[0][0],
            hess_exprs[0][1],
            hess_exprs[1][0],
            hess_exprs[1][1],
        ]
        compiled = compile_to_callable(flat, VARIABLES)

        def hess_callable(*args):
            try:
                values = compiled(*args)
                return [
                    [values[0], values[1]],
                    [values[2], values[3]],
                ]
            except Exception:
                # «Вторая линия защиты»: если символьно посчитать не удалось,
                # используем численный гессиан.
                return numeric_hessian(f_callable, list(args))

        return hess_callable

    def _run(self) -> None:
        if self.current_spec is None or self.current_context is None:
            return

        spec = self.current_spec
        context = self.current_context
        raw = context["raw"]
        mode = self.mode_page.selected_mode()

        params = {
            "x0": raw["x0"],
            "y0": raw["y0"],
            "epsilon": raw["epsilon"],
        }
        params.update(raw["spec"])
        params.update(context["extra_params"])

        try:
            result = spec.run(
                context["f_callable"],
                context["grad_callable"],
                context["hess_callable"],
                params,
                mode,
            )
        except Exception as exc:
            timestamp = storage.make_timestamp()
            text = build_error_file(
                timestamp,
                registry.display_name(spec),
                raw["f_text"],
                "Расчёт",
                str(exc),
            )
            try:
                path = storage.write_result_text(text, timestamp)
                details = f"\n\nПодробности сохранены в файл:\n{path}"
            except OSError as save_error:
                details = f"\n\nНе удалось сохранить файл: {save_error}"
            QMessageBox.critical(
                self,
                "Ошибка расчёта",
                f"Во время расчёта произошла ошибка:\n{exc}{details}",
            )
            return

        timestamp = storage.make_timestamp()
        meta = {
            "datetime": timestamp,
            "method": registry.display_name(spec),
            "mode": mode,
            "function": raw["f_text"],
            "gradient": self._gradient_text(context["grad_exprs"]),
            "gradient_source": context["grad_source"],
            "hessian": self._hessian_text(context["hess_exprs"]),
            "hessian_source": context["hess_source"],
            "x0": raw["x0"],
            "y0": raw["y0"],
            "epsilon": raw["epsilon"],
            "spec_params": raw["spec"],
        }
        log_for_file = result.log_lines if mode == "explain" else []
        text = build_result_file(meta, log_for_file, result)

        try:
            path = storage.write_result_text(text, timestamp)
            file_path_text = str(path)
        except OSError as exc:
            file_path_text = f"не удалось сохранить ({exc})"
            QMessageBox.warning(
                self,
                "Не удалось сохранить результат",
                f"Результат показан на экране, но файл не создан:\n{exc}",
            )

        common = self.last_input.setdefault("common", {})
        if isinstance(common, dict):
            common["mode"] = mode
        try:
            storage.save_last_input(self.last_input)
        except OSError:
            pass

        log_text = "\n".join(result.log_lines) if mode == "explain" else ""
        self.result_page.show_result(result, log_text, file_path_text)
        self.stack.setCurrentWidget(self.result_page)

    def _save_last_input(self, raw: dict) -> None:
        common_previous = self.last_input.get("common", {})
        if not isinstance(common_previous, dict):
            common_previous = {}

        common = {
            "f_expr": raw["f_text"],
            "x0": raw["x0"],
            "y0": raw["y0"],
            "epsilon": raw["epsilon"],
            "gradient_manual": raw.get("gradient_manual"),
            "mode": common_previous.get("mode", "explain"),
        }

        per_method = self.last_input.get("per_method", {})
        if not isinstance(per_method, dict):
            per_method = {}
        per_method = dict(per_method)
        per_method[self.current_spec.id] = raw["spec"]

        data = {
            "last_method_id": self.current_spec.id,
            "common": common,
            "per_method": per_method,
        }
        try:
            storage.save_last_input(data)
        except OSError:
            pass
        self.last_input = data

    def _same_method(self) -> None:
        if self.current_spec is None:
            return
        self.input_page.prepare(self.current_spec, self.last_input)
        self.stack.setCurrentWidget(self.input_page)

    def _change_method(self) -> None:
        self.method_page.select_method(self.last_input.get("last_method_id"))
        self.stack.setCurrentWidget(self.method_page)

    # ------------------------------------------------------------------
    # Ошибки
    # ------------------------------------------------------------------

    def _report_input_error(self, field: str, message: str, function_text: str) -> None:
        method_name = (
            registry.display_name(self.current_spec)
            if self.current_spec is not None
            else "не выбран"
        )
        timestamp = storage.make_timestamp()
        text = build_error_file(timestamp, method_name, function_text, field, message)
        details = ""
        try:
            path = storage.write_result_text(text, timestamp)
            details = f"\n\nПодробности сохранены в файл:\n{path}"
        except OSError as exc:
            details = f"\n\nНе удалось сохранить файл с ошибкой: {exc}"

        QMessageBox.critical(
            self,
            "Ошибка ввода",
            f"{message}{details}",
        )

    @staticmethod
    def _gradient_text(exprs) -> str | None:
        if not exprs:
            return None
        return "[" + ", ".join(sympy.sstr(expr) for expr in exprs) + "]"

    @staticmethod
    def _hessian_text(hessian) -> str | None:
        if not hessian:
            return None
        rows = [
            "[" + ", ".join(sympy.sstr(expr) for expr in row) + "]"
            for row in hessian
        ]
        return "[" + ", ".join(rows) + "]"
