#!/usr/bin/env python3.14
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional

from src.blaze_ast import (
    Assign,
    BinaryOp,
    Call,
    Function,
    IntLiteral,
    Let,
    Module,
    Name,
    Node,
    Param,
    Return,
    Var,
)


class SemanticError(Exception):
    def __init__(self, message: str, line: int = 0, col: int = 0):
        self.message = message
        self.line = line
        self.col = col
        super().__init__(self._format())

    def _format(self) -> str:
        return (
            f"{self.line}:{self.col}: error: {self.message}"
            if self.line
            else f"error: {self.message}"
        )


class Symbol:
    __slots__ = ("name", "type", "mutable", "line", "col")

    def __init__(self, name: str, type: str, mutable: bool, line: int, col: int):
        self.name = name
        self.type = type
        self.mutable = mutable
        self.line = line
        self.col = col


class FuncInfo:
    __slots__ = ("name", "params", "return_type", "line", "col")

    def __init__(
        self,
        name: str,
        params: List[Param],
        return_type: Optional[str],
        line: int,
        col: int,
    ):
        self.name = name
        self.params = params
        self.return_type = return_type
        self.line = line
        self.col = col


class SemanticAnalyzer:
    def __init__(self):
        self.symbols: Dict[str, Symbol] = {}
        self.functions: Dict[str, FuncInfo] = {}
        self.current_function: Optional[FuncInfo] = None
        self.in_function = False

    def analyze(self, node: Node) -> Node:
        self.visit(node)
        return node

    def visit(self, node: Node):
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method:
            method(node)
        else:
            for child in self._children(node):
                self.visit(child)

    def _children(self, node: Node) -> list:
        if isinstance(node, Module):
            return node.body
        if isinstance(node, Function):
            return node.body
        if isinstance(node, Let) or isinstance(node, Var) or isinstance(node, Assign):
            return [node.value] if node.value else []
        if isinstance(node, BinaryOp):
            return [node.left, node.right]
        if isinstance(node, Call):
            return node.args
        if isinstance(node, Return):
            return [node.value] if node.value else []
        return []

    def visit_Module(self, node: Module):
        self.functions.clear()
        for stmt in node.body:
            if isinstance(stmt, Function):
                if stmt.name in self.functions:
                    raise SemanticError(
                        f"function `{stmt.name}` already defined", stmt.line, stmt.col
                    )
                self.functions[stmt.name] = FuncInfo(
                    stmt.name, stmt.params, stmt.return_type, stmt.line, stmt.col
                )
        self.symbols.clear()
        for stmt in node.body:
            self.visit(stmt)

    def visit_Function(self, node: Function):
        old_symbols = self.symbols
        self.symbols = {}
        old_func = self.current_function
        self.current_function = self.functions[node.name]
        self.in_function = True

        for param in node.params:
            if param.name in self.symbols:
                raise SemanticError(
                    f"duplicate parameter name `{param.name}`", param.line, param.col
                )
            param_type = param.type_ann if param.type_ann else "i32"
            self.symbols[param.name] = Symbol(
                param.name, param_type, False, param.line, param.col
            )

        for stmt in node.body:
            self.visit(stmt)

        if node.return_type:
            last = node.body[-1] if node.body else None
            if not isinstance(last, Return):
                raise SemanticError(
                    f"function `{node.name}` must return a value of type {node.return_type}",
                    node.line,
                    node.col,
                )

        self.symbols = old_symbols
        self.current_function = old_func
        self.in_function = bool(old_func)

    def visit_Return(self, node: Return):
        if not self.in_function:
            raise SemanticError("return outside function", node.line, node.col)
        if node.value:
            self.visit(node.value)
            ret_type = self._infer_type(node.value)
        else:
            ret_type = "unit"
        if (
            self.current_function.return_type
            and ret_type != self.current_function.return_type
        ):
            raise SemanticError(
                f"return type mismatch: expected {self.current_function.return_type}, found {ret_type}",
                node.line,
                node.col,
            )

    def visit_Call(self, node: Call):
        # Special case: println is a built-in function
        if node.func == "println":
            if len(node.args) != 1:
                raise SemanticError(
                    "println expects exactly one argument", node.line, node.col
                )
            for arg in node.args:
                self.visit(arg)
            return

        # Regular function call
        if node.func not in self.functions:
            raise SemanticError(f"unknown function `{node.func}`", node.line, node.col)
        func = self.functions[node.func]
        if len(node.args) != len(func.params):
            raise SemanticError(
                f"function `{node.func}` expects {len(func.params)} arguments, got {len(node.args)}",
                node.line,
                node.col,
            )
        for arg, param in zip(node.args, func.params):
            self.visit(arg)
            arg_type = self._infer_type(arg)
            param_type = param.type_ann if param.type_ann else "i32"
            if arg_type != param_type:
                raise SemanticError(
                    f"argument type mismatch: expected {param_type}, found {arg_type}",
                    arg.line,
                    arg.col,
                )

    def visit_Let(self, node: Let):
        if node.name in self.symbols:
            raise SemanticError(
                f"variable `{node.name}` already declared", node.line, node.col
            )
        self.visit(node.value)
        value_type = self._infer_type(node.value)
        if node.type_ann and value_type != node.type_ann:
            raise SemanticError(
                f"type mismatch: expected {node.type_ann}, found {value_type}",
                node.line,
                node.col,
            )
        self.symbols[node.name] = Symbol(
            node.name, value_type, False, node.line, node.col
        )

    def visit_Var(self, node: Var):
        if node.name in self.symbols:
            raise SemanticError(
                f"variable `{node.name}` already declared", node.line, node.col
            )
        self.visit(node.value)
        value_type = self._infer_type(node.value)
        if node.type_ann and value_type != node.type_ann:
            raise SemanticError(
                f"type mismatch: expected {node.type_ann}, found {value_type}",
                node.line,
                node.col,
            )
        self.symbols[node.name] = Symbol(
            node.name, value_type, True, node.line, node.col
        )

    def visit_Assign(self, node: Assign):
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
        if node.id not in self.symbols:
            raise SemanticError(
                f"use of undeclared variable `{node.id}`", node.line, node.col
            )

    def visit_BinaryOp(self, node: BinaryOp):
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
        if left_type != "i32":
            raise SemanticError(
                f"binary operation not supported on type {left_type}",
                node.line,
                node.col,
            )

    def _infer_type(self, node: Node) -> str:
        if isinstance(node, IntLiteral):
            return "i32"
        if isinstance(node, Name):
            sym = self.symbols.get(node.id)
            if not sym:
                raise SemanticError(
                    f"use of undeclared variable `{node.id}`", node.line, node.col
                )
            return sym.type
        if isinstance(node, BinaryOp):
            left = self._infer_type(node.left)
            right = self._infer_type(node.right)
            if left != right:
                raise SemanticError(
                    f"type mismatch: {left} vs {right}", node.line, node.col
                )
            return left
        if isinstance(node, Call):
            if node.func == "println":
                # println is a statement, not an expression; if it appears in expression context, that's an error,
                # but we return "unit" to allow it to type-check (the error will be caught elsewhere).
                return "unit"
            if node.func not in self.functions:
                raise SemanticError(
                    f"unknown function `{node.func}`", node.line, node.col
                )
            func = self.functions[node.func]
            return func.return_type if func.return_type else "unit"
        raise SemanticError(
            f"cannot infer type for {type(node).__name__}", node.line, node.col
        )
