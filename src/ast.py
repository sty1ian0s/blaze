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
