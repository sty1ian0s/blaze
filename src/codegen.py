#!/usr/bin/env python3.14
# SPDX-License-Identifier: Apache-2.0

"""
Blaze code generator – translates AST to LLVM IR.
"""

from src.blaze_ast import Module


class CodeGenError(Exception):
    """Raised when code generation fails."""

    pass


class CodeGen:
    """Generates LLVM IR from an AST."""

    def __init__(self):
        self.output = []
        self.indent_level = 0

    def indent(self):
        self.indent_level += 1

    def dedent(self):
        self.indent_level -= 1

    def emit(self, line: str = ""):
        """Emit a line of IR with current indentation."""
        if line:
            self.output.append("  " * self.indent_level + line)
        else:
            self.output.append("")

    def generate(self, node) -> str:
        """Generate IR for the given AST node and return as string."""
        if isinstance(node, Module):
            self.gen_module(node)
        else:
            raise CodeGenError(f"Unsupported node type: {type(node).__name__}")
        return "\n".join(self.output)

    def gen_module(self, node: Module):
        """Generate IR for a module."""
        # Emit module header
        self.emit("; ModuleID = 'blaze_module'")
        self.emit(
            'target triple = "x86_64-unknown-linux-gnu"'
        )  # FIXME: make configurable
        self.emit()

        # For an empty module, we still need a main function returning 0
        self.gen_main()

    def gen_main(self):
        """Generate the main function."""
        self.emit("define i32 @main() {")
        self.indent()
        self.emit("ret i32 0")
        self.dedent()
        self.emit("}")
