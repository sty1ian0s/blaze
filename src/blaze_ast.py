#!/usr/bin/env python3.14
# SPDX-License-Identifier: Apache-2.0

"""
Abstract Syntax Tree (AST) node definitions for Blaze.
"""

from typing import Any, List, Optional


class Node:
    """Base class for all AST nodes."""

    __slots__ = ()

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Module(Node):
    """Root node of a Blaze program. Contains a list of top-level items (functions, structs, etc.)."""

    __slots__ = ("body",)

    def __init__(self, body: List[Node]):
        self.body = body

    def __repr__(self):
        return f"Module(body={self.body!r})"


class IntLiteral(Node):
    """Integer literal (e.g., 42)."""

    __slots__ = ("value",)

    def __init__(self, value: int):
        self.value = value

    def __repr__(self):
        return f"IntLiteral({self.value})"


class Call(Node):
    """Function call: name(args)."""

    __slots__ = ("func", "args")

    def __init__(self, func: str, args: List[Node]):
        self.func = func
        self.args = args

    def __repr__(self):
        return f"Call({self.func!r}, {self.args!r})"
