#!/usr/bin/env python3.14
# SPDX-License-Identifier: Apache-2.0

"""
Abstract Syntax Tree (AST) node definitions for Blaze.
All nodes store source location (line, col) for error reporting.
"""

from typing import Any, List, Optional


class Node:
    """Base class for all AST nodes."""

    __slots__ = ("line", "col")

    def __init__(self, line: int = 0, col: int = 0):
        self.line = line
        self.col = col

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Module(Node):
    """Root node of a Blaze program."""

    __slots__ = ("body",)

    def __init__(self, body: List[Node], line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.body = body

    def __repr__(self):
        return f"Module(body={self.body!r})"


class IntLiteral(Node):
    """Integer literal (e.g., 42)."""

    __slots__ = ("value",)

    def __init__(self, value: int, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.value = value

    def __repr__(self):
        return f"IntLiteral({self.value})"


class Name(Node):
    """Variable reference."""

    __slots__ = ("id",)

    def __init__(self, id: str, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.id = id

    def __repr__(self):
        return f"Name({self.id!r})"


class BinaryOp(Node):
    """Binary operation: left op right."""

    __slots__ = ("op", "left", "right")

    def __init__(self, op: str, left: Node, right: Node, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self):
        return f"BinaryOp({self.op!r}, {self.left!r}, {self.right!r})"


class Let(Node):
    """Immutable let binding."""

    __slots__ = ("name", "type_ann", "value")

    def __init__(
        self,
        name: str,
        type_ann: Optional[str],
        value: Node,
        line: int = 0,
        col: int = 0,
    ):
        super().__init__(line, col)
        self.name = name
        self.type_ann = type_ann
        self.value = value

    def __repr__(self):
        return f"Let({self.name!r}, {self.type_ann!r}, {self.value!r})"


class Var(Node):
    """Mutable var binding."""

    __slots__ = ("name", "type_ann", "value")

    def __init__(
        self,
        name: str,
        type_ann: Optional[str],
        value: Node,
        line: int = 0,
        col: int = 0,
    ):
        super().__init__(line, col)
        self.name = name
        self.type_ann = type_ann
        self.value = value

    def __repr__(self):
        return f"Var({self.name!r}, {self.type_ann!r}, {self.value!r})"


class Assign(Node):
    """Assignment to a mutable variable or array element."""

    __slots__ = ("target", "value")

    def __init__(self, target: Node, value: Node, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.target = target
        self.value = value

    def __repr__(self):
        return f"Assign({self.target!r}, {self.value!r})"


class Call(Node):
    """Function call: name(args)."""

    __slots__ = ("func", "args")

    def __init__(self, func: str, args: List[Node], line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.func = func
        self.args = args

    def __repr__(self):
        return f"Call({self.func!r}, {self.args!r})"


class Param(Node):
    """Function parameter."""

    __slots__ = ("name", "type_ann")

    def __init__(self, name: str, type_ann: Optional[str], line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.name = name
        self.type_ann = type_ann

    def __repr__(self):
        return f"Param({self.name!r}, {self.type_ann!r})"


class Function(Node):
    """Function definition."""

    __slots__ = ("name", "params", "return_type", "body")

    def __init__(
        self,
        name: str,
        params: List[Param],
        return_type: Optional[str],
        body: List[Node],
        line: int = 0,
        col: int = 0,
    ):
        super().__init__(line, col)
        self.name = name
        self.params = params
        self.return_type = return_type
        self.body = body

    def __repr__(self):
        return f"Function({self.name!r}, {self.params!r}, {self.return_type!r}, body=[...])"


class Return(Node):
    """Return statement."""

    __slots__ = ("value",)

    def __init__(self, value: Optional[Node], line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.value = value

    def __repr__(self):
        return f"Return({self.value!r})"


class If(Node):
    """If expression."""

    __slots__ = ("cond", "then_body", "else_body")

    def __init__(
        self,
        cond: Node,
        then_body: List[Node],
        else_body: List[Node],
        line: int = 0,
        col: int = 0,
    ):
        super().__init__(line, col)
        self.cond = cond
        self.then_body = then_body
        self.else_body = else_body

    def __repr__(self):
        return f"If({self.cond!r}, then=[...], else=[...])"


class While(Node):
    """While loop."""

    __slots__ = ("cond", "body", "label")

    def __init__(
        self,
        cond: Node,
        body: List[Node],
        label: Optional[str] = None,
        line: int = 0,
        col: int = 0,
    ):
        super().__init__(line, col)
        self.cond = cond
        self.body = body
        self.label = label

    def __repr__(self):
        return f"While({self.cond!r}, body=[...], label={self.label!r})"


class Loop(Node):
    """Infinite loop."""

    __slots__ = ("body", "label")

    def __init__(
        self, body: List[Node], label: Optional[str] = None, line: int = 0, col: int = 0
    ):
        super().__init__(line, col)
        self.body = body
        self.label = label

    def __repr__(self):
        return f"Loop(body=[...], label={self.label!r})"


class Break(Node):
    """Break statement."""

    __slots__ = ("label",)

    def __init__(self, label: Optional[str] = None, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.label = label

    def __repr__(self):
        return f"Break({self.label!r})"


class Continue(Node):
    """Continue statement."""

    __slots__ = ("label",)

    def __init__(self, label: Optional[str] = None, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.label = label

    def __repr__(self):
        return f"Continue({self.label!r})"


class ArrayLiteral(Node):
    """Array literal: [expr, expr, ...] (trailing comma allowed)."""

    __slots__ = ("elements",)

    def __init__(self, elements: List[Node], line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.elements = elements

    def __repr__(self):
        return f"ArrayLiteral({self.elements!r})"


class Index(Node):
    """Index expression: target[index]."""

    __slots__ = ("target", "index")

    def __init__(self, target: Node, index: Node, line: int = 0, col: int = 0):
        super().__init__(line, col)
        self.target = target
        self.index = index

    def __repr__(self):
        return f"Index({self.target!r}, {self.index!r})"
