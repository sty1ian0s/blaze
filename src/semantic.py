#!/usr/bin/env python3.14
# SPDX-License-Identifier: Apache-2.0

"""
Semantic analysis for Blaze.
For Phase 4, this is a placeholder that just returns the AST unchanged.
"""

from src.blaze_ast import Node


class SemanticError(Exception):
    """Raised when a semantic error is detected."""

    pass


class SemanticAnalyzer:
    """Performs type checking and other semantic validation."""

    def __init__(self):
        pass

    def analyze(self, node: Node) -> Node:
        """Walk the AST and perform semantic checks. For now, just return node."""
        # In future phases, we'll traverse and check types, ownership, etc.
        return node
