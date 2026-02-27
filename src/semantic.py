#!/usr/bin/env python3.14
# SPDX-License-Identifier: Apache-2.0

"""
Semantic analysis for Blaze.
Phase 5: variable declarations, type checking (all ints are i32).
"""

from typing import Dict, Optional, Set

from src.blaze_ast import (
    Assign,
    BinaryOp,
    Call,
    IntLiteral,
    Let,
    Module,
    Name,
    Node,
    Var,
)


class SemanticError(Exception):
    """Raised when a semantic error is detected."""

    def __init__(self, message: str, line: int = 0, col: int = 0):
        self.message = message
        self.line = line
        self.col = col
        super().__init__(self._format())

    def _format(self) -> str:
        if self.line:
            return f"{self.line}:{self.col}: error: {self.message}"
        return f"error: {self.message}"


class Symbol:
    """Information about a declared variable."""

    __slots__ = ("name", "type", "mutable", "line", "col")

    def __init__(self, name: str, type: str, mutable: bool, line: int, col: int):
        self.name = name
        self.type = type
        self.mutable = mutable
        self.line = line
        self.col = col


class SemanticAnalyzer:
    """Performs type checking and other semantic validation."""

    def __init__(self):
        self.symbols: Dict[str, Symbol] = {}
        self.current_line = 0
        self.current_col = 0

    def analyze(self, node: Node) -> Node:
        """Walk the AST and perform semantic checks."""
        self.visit(node)
        return node

    def visit(self, node: Node):
        """Dispatch to appropriate visitor method."""
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method:
            method(node)
        else:
            # Default: walk children
            for child in self._children(node):
                self.visit(child)

    def _children(self, node: Node) -> list:
        """Return list of child nodes."""
        if isinstance(node, Module):
            return node.body
        elif isinstance(node, Let):
            return [node.value] if node.value else []
        elif isinstance(node, Var):
            return [node.value] if node.value else []
        elif isinstance(node, Assign):
            return [node.value]
        elif isinstance(node, BinaryOp):
            return [node.left, node.right]
        elif isinstance(node, Call):
            return node.args
        elif isinstance(node, Name):
            return []
        elif isinstance(node, IntLiteral):
            return []
        else:
            return []

    def visit_Module(self, node: Module):
        """Check module body."""
        self.symbols.clear()
        for stmt in node.body:
            self.visit(stmt)

    def visit_Let(self, node: Let):
        """Check let binding."""
        if node.name in self.symbols:
            raise SemanticError(
                f"variable `{node.name}` already declared", node.line, node.col
            )
        self.visit(node.value)
        value_type = self._infer_type(node.value)
        if node.type_ann:
            if value_type != node.type_ann:
                raise SemanticError(
                    f"type mismatch: expected {node.type_ann}, found {value_type}",
                    node.line,
                    node.col,
                )
        # Register symbol
        self.symbols[node.name] = Symbol(
            node.name, value_type, mutable=False, line=node.line, col=node.col
        )

    def visit_Var(self, node: Var):
        """Check var binding."""
        if node.name in self.symbols:
            raise SemanticError(
                f"variable `{node.name}` already declared", node.line, node.col
            )
        self.visit(node.value)
        value_type = self._infer_type(node.value)
        if node.type_ann:
            if value_type != node.type_ann:
                raise SemanticError(
                    f"type mismatch: expected {node.type_ann}, found {value_type}",
                    node.line,
                    node.col,
                )
        # Register symbol
        self.symbols[node.name] = Symbol(
            node.name, value_type, mutable=True, line=node.line, col=node.col
        )

    def visit_Assign(self, node: Assign):
        """Check assignment to mutable variable."""
        if node.name not in self.symbols:
            raise SemanticError(
                f"use of undeclared variable `{node.name}`", node.line, node.col
            )
        sym = self.symbols[node.name]
        if not sym.mutable:
            raise SemanticError(
                f"cannot assign to immutable variable `{node.name}`",
                node.line,
                node.col,
            )
        self.visit(node.value)
        value_type = self._infer_type(node.value)
        if value_type != sym.type:
            raise SemanticError(
                f"type mismatch: expected {sym.type}, found {value_type}",
                node.line,
                node.col,
            )

    def visit_Name(self, node: Name):
        """Check variable reference."""
        if node.id not in self.symbols:
            raise SemanticError(
                f"use of undeclared variable `{node.id}`", node.line, node.col
            )

    def visit_BinaryOp(self, node: BinaryOp):
        """Check binary operation types."""
        self.visit(node.left)
        self.visit(node.right)
        left_type = self._infer_type(node.left)
        right_type = self._infer_type(node.right)
        if left_type != right_type:
            raise SemanticError(
                f"type mismatch in binary op: {left_type} vs {right_type}",
                node.line,
                node.col,
            )
        # For now, only support integer operations
        if left_type != "i32":
            raise SemanticError(
                f"binary operation not supported on type {left_type}",
                node.line,
                node.col,
            )

    def visit_Call(self, node: Call):
        """Check function calls."""
        if node.func != "println":
            raise SemanticError(f"unknown function `{node.func}`", node.line, node.col)
        for arg in node.args:
            self.visit(arg)

    def _infer_type(self, node: Node) -> str:
        """Infer the type of an expression node."""
        if isinstance(node, IntLiteral):
            return "i32"
        elif isinstance(node, Name):
            sym = self.symbols.get(node.id)
            if not sym:
                raise SemanticError(
                    f"use of undeclared variable `{node.id}`", node.line, node.col
                )
            return sym.type
        elif isinstance(node, BinaryOp):
            left_type = self._infer_type(node.left)
            right_type = self._infer_type(node.right)
            if left_type != right_type:
                raise SemanticError(
                    f"type mismatch: {left_type} vs {right_type}", node.line, node.col
                )
            return left_type
        elif isinstance(node, Call):
            # For now, println returns nothing (unit)
            return "unit"
        else:
            raise SemanticError(
                f"cannot infer type for {type(node).__name__}", node.line, node.col
            )
