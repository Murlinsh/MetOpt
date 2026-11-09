"""Страница 1: выбор метода."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from methods import registry


class MethodPage(QWidget):
    next_requested = pyqtSignal()
    exit_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: dict[str, QTreeWidgetItem] = {}

        layout = QVBoxLayout(self)
        title = QLabel("Выбор метода оптимизации")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        subtitle = QLabel("Выберите метод в дереве и нажмите «Далее».")
        subtitle.setWordWrap(True)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(1)
        self.tree.itemDoubleClicked.connect(self._on_double_click)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.tree, 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.next_button = QPushButton("Далее")
        self.exit_button = QPushButton("Выход")
        buttons.addWidget(self.next_button)
        buttons.addWidget(self.exit_button)
        layout.addLayout(buttons)

        self.next_button.clicked.connect(self.next_requested)
        self.exit_button.clicked.connect(self.exit_requested)

        self._build_tree()

    def _group_item(self, title: str, parent: QTreeWidgetItem | None = None) -> QTreeWidgetItem:
        item = QTreeWidgetItem([title])
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        if parent is None:
            self.tree.addTopLevelItem(item)
        else:
            parent.addChild(item)
        return item

    def _leaf_item(self, method, parent: QTreeWidgetItem) -> QTreeWidgetItem:
        item = QTreeWidgetItem([method.name])
        item.setData(0, Qt.ItemDataRole.UserRole, method.id)
        item.setToolTip(0, registry.display_name(method))
        parent.addChild(item)
        self._items[method.id] = item
        return item

    def _build_tree(self) -> None:
        self.tree.clear()
        self._items.clear()

        no_derivative = self._group_item(registry.GROUP_TITLES["no_derivative"])
        for method in registry.methods_by_group("no_derivative"):
            self._leaf_item(method, no_derivative)

        gradient = self._group_item(registry.GROUP_TITLES["gradient"])
        gradient_subgroup = self._group_item(
            registry.SUBGROUP_TITLES["gradient_descent"], gradient
        )
        for method in registry.methods_by_group("gradient"):
            self._leaf_item(method, gradient_subgroup)

        newton = self._group_item(registry.GROUP_TITLES["newton"])
        quasi = self._group_item(registry.SUBGROUP_TITLES["quasi_newton"], newton)
        for method in registry.methods_by_group("newton"):
            parent = quasi if method.subgroup == "quasi_newton" else newton
            self._leaf_item(method, parent)

        self.tree.expandAll()

    def select_method(self, method_id: str | None) -> None:
        if not method_id:
            return
        item = self._items.get(method_id)
        if item is not None:
            self.tree.setCurrentItem(item)
            self.tree.scrollToItem(item)

    def selected_method_id(self) -> str | None:
        item = self.tree.currentItem()
        if item is None:
            return None
        value = item.data(0, Qt.ItemDataRole.UserRole)
        return str(value) if value else None

    def _on_double_click(self, item: QTreeWidgetItem, column: int) -> None:
        if item.data(0, Qt.ItemDataRole.UserRole):
            self.next_requested.emit()
