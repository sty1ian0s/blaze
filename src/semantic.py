#!/usr/bin/env python3.14
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Dict, List, Optional

from src.blaze_ast import (
    ArrayLiteral,
    Assign,
    BinaryOp,
    Break,
    Call,
    Continue,
    Function,
    If,
    Index,
    IntLiteral,
    Let,
    Loop,
    Module,
    Name,
    Node,
    Param,
    Return,
    Var,
    While,
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
        self.loop_stack = []

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
        if isinstance(node, If):
            return [node.cond] + node.then_body + node.else_body
        if isinstance(node, While):
            return [node.cond] + node.body
        if isinstance(node, Loop):
            return node.body
        if isinstance(node, Let) or isinstance(node, Var):
            return [node.value] if node.value else []
        if isinstance(node, Assign):
            return [node.target, node.value]
        if isinstance(node, BinaryOp):
            return [node.left, node.right]
        if isinstance(node, Call):
            return node.args
        if isinstance(node, Return):
            return [node.value] if node.value else []
        if isinstance(node, Index):
            return [node.target, node.index]
        if isinstance(node, ArrayLiteral):
            return node.elements
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
        func_info = self.functions[node.name]
        self.current_function = func_info
        self.in_function = True

        for param in node.params:
            if param.name in self.symbols:
                raise SemanticError(
                    f"duplicate parameter name `{param.name}`", param.line, param.col
                )
            if not param.type_ann:
                raise SemanticError(
                    f"parameter `{param.name}` must have a type annotation",
                    param.line,
                    param.col,
                )
            self.symbols[param.name] = Symbol(
                param.name, param.type_ann, False, param.line, param.col
            )

        last_expr_type = None
        for stmt in node.body:
            self.visit(stmt)
            if not isinstance(
                stmt, (Let, Var, Assign, Return, If, While, Loop, Break, Continue)
            ):
                last_expr_type = self._infer_type(stmt)

        if func_info.return_type:
            if last_expr_type and last_expr_type != func_info.return_type:
                raise SemanticError(
                    f"function `{node.name}` returns {last_expr_type} but expected {func_info.return_type}",
                    node.line,
                    node.col,
                )
        else:
            func_info.return_type = (
                last_expr_type if last_expr_type is not None else "unit"
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
        expected = self.current_function.return_type
        if expected and ret_type != expected:
            raise SemanticError(
                f"return type mismatch: expected {expected}, found {ret_type}",
                node.line,
                node.col,
            )

    def visit_Call(self, node: Call):
        if node.func == "println":
            if len(node.args) != 1:
                raise SemanticError(
                    "println expects exactly one argument", node.line, node.col
                )
            for arg in node.args:
                self.visit(arg)
                arg_type = self._infer_type(arg)
                if arg_type != "i32":
                    raise SemanticError(
                        f"println argument must be i32, found {arg_type}",
                        arg.line,
                        arg.col,
                    )
            return
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
        if node.type_ann:
            if value_type != node.type_ann:
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
        if node.type_ann:
            if value_type != node.type_ann:
                raise SemanticError(
                    f"type mismatch: expected {node.type_ann}, found {value_type}",
                    node.line,
                    node.col,
                )
        self.symbols[node.name] = Symbol(
            node.name, value_type, True, node.line, node.col
        )

    def visit_Assign(self, node: Assign):
        if isinstance(node.target, Name):
            if node.target.id not in self.symbols:
                raise SemanticError(
                    f"use of undeclared variable `{node.target.id}`",
                    node.target.line,
                    node.target.col,
                )
            sym = self.symbols[node.target.id]
            if not sym.mutable:
                raise SemanticError(
                    f"cannot assign to immutable variable `{node.target.id}`",
                    node.target.line,
                    node.target.col,
                )
            expected_type = sym.type
        elif isinstance(node.target, Index):
            target_type = self._infer_type(node.target)
            expected_type = target_type
        else:
            raise SemanticError(
                f"cannot assign to {type(node.target).__name__}",
                node.target.line,
                node.target.col,
            )

        self.visit(node.value)
        value_type = self._infer_type(node.value)
        if value_type != expected_type:
            raise SemanticError(
                f"type mismatch: expected {expected_type}, found {value_type}",
                node.value.line,
                node.value.col,
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

    def visit_If(self, node: If):
        self.visit(node.cond)
        cond_type = self._infer_type(node.cond)
        if cond_type != "i32":
            raise SemanticError(
                f"if condition must be integer, found {cond_type}",
                node.cond.line,
                node.cond.col,
            )
        for stmt in node.then_body:
            self.visit(stmt)
        for stmt in node.else_body:
            self.visit(stmt)

    def visit_While(self, node: While):
        self.visit(node.cond)
        cond_type = self._infer_type(node.cond)
        if cond_type != "i32":
            raise SemanticError(
                f"while condition must be integer, found {cond_type}",
                node.cond.line,
                node.cond.col,
            )
        self.loop_stack.append(node.label)
        for stmt in node.body:
            self.visit(stmt)
        self.loop_stack.pop()

    def visit_Loop(self, node: Loop):
        self.loop_stack.append(node.label)
        for stmt in node.body:
            self.visit(stmt)
        self.loop_stack.pop()

    def visit_Break(self, node: Break):
        if not self.loop_stack:
            raise SemanticError("break outside loop", node.line, node.col)

    def visit_Continue(self, node: Continue):
        if not self.loop_stack:
            raise SemanticError("continue outside loop", node.line, node.col)

    def visit_Index(self, node: Index):
        self.visit(node.target)
        self.visit(node.index)

    def visit_ArrayLiteral(self, node: ArrayLiteral):
        for elem in node.elements:
            self.visit(elem)

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
                return "unit"
            if node.func not in self.functions:
                raise SemanticError(
                    f"unknown function `{node.func}`", node.line, node.col
                )
            func = self.functions[node.func]
            return func.return_type if func.return_type else "unit"
        if isinstance(node, If):
            then_type = "unit"
            for stmt in node.then_body:
                if not isinstance(
                    stmt, (Let, Var, Assign, Return, Break, Continue, If, While, Loop)
                ):
                    then_type = self._infer_type(stmt)
            else_type = "unit"
            for stmt in node.else_body:
                if not isinstance(
                    stmt, (Let, Var, Assign, Return, Break, Continue, If, While, Loop)
                ):
                    else_type = self._infer_type(stmt)
            if then_type != else_type:
                raise SemanticError(
                    f"if branches have mismatched types: {then_type} vs {else_type}",
                    node.line,
                    node.col,
                )
            return then_type
        if isinstance(node, ArrayLiteral):
            if not node.elements:
                raise SemanticError(
                    "empty array literals not supported", node.line, node.col
                )
            elem_type = self._infer_type(node.elements[0])
            for elem in node.elements[1:]:
                t = self._infer_type(elem)
                if t != elem_type:
                    raise SemanticError(
                        f"array elements must have same type: {elem_type} vs {t}",
                        elem.line,
                        elem.col,
                    )
            return f"[{elem_type}; {len(node.elements)}]"
        if isinstance(node, Index):
            target_type = self._infer_type(node.target)
            if not target_type.startswith("["):
                raise SemanticError(
                    f"cannot index non-array type `{target_type}`",
                    node.target.line,
                    node.target.col,
                )
            match = re.match(r"\[([^;]+);\s*(\d+)\]", target_type)
            if not match:
                raise SemanticError(
                    f"invalid array type format: `{target_type}`",
                    node.target.line,
                    node.target.col,
                )
            elem_type = match.group(1).strip()
            size = int(match.group(2))
            index_type = self._infer_type(node.index)
            if index_type != "i32":
                raise SemanticError(
                    f"array index must be i32, found `{index_type}`",
                    node.index.line,
                    node.index.col,
                )
            if isinstance(node.index, IntLiteral):
                idx = node.index.value
                if idx < 0 or idx >= size:
                    raise SemanticError(
                        f"index {idx} out of bounds for array of size {size}",
                        node.index.line,
                        node.index.col,
                    )
            return elem_type
        raise SemanticError(
            f"cannot infer type for {type(node).__name__}", node.line, node.col
        )
