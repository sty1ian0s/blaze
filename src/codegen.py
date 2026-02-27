#!/usr/bin/env python3.14
# SPDX-License-Identifier: Apache-2.0

"""
Blaze code generator – translates AST to LLVM IR.
"""

from src.blaze_ast import Call, IntLiteral, Module


class CodeGenError(Exception):
    """Raised when code generation fails."""

    pass


class CodeGen:
    """Generates LLVM IR from an AST."""

    def __init__(self):
        self.output = []
        self.indent_level = 0
        self.println_idx = 0  # index for current println being generated

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

    def escape_string(self, s: str) -> str:
        """Escape a string for use in LLVM IR constant."""
        result = []
        for ch in s:
            if ch == "\n":
                result.append("\\0A")
            elif ch == "\r":
                result.append("\\0D")
            elif ch == "\t":
                result.append("\\09")
            elif ch == '"':
                result.append('\\"')
            elif ch == "\\":
                result.append("\\\\")
            elif 32 <= ord(ch) < 127:
                result.append(ch)
            else:
                result.append(f"\\{ord(ch):02X}")
        return "".join(result)

    def generate(self, node) -> str:
        """Generate IR for the given AST node and return as string."""
        self.output = []
        # Emit module header
        self.emit("; ModuleID = 'blaze_module'")
        self.emit('target triple = "x86_64-unknown-linux-gnu"')
        self.emit()
        # Declare external printf
        self.emit("declare i32 @printf(i8*, ...)")
        self.emit()
        # Generate the program (includes collecting format strings)
        self.gen_module(node)
        return "\n".join(self.output)

    def gen_module(self, node: Module):
        """Generate IR for a module."""
        # Count printlns to generate format strings
        println_count = 0
        for stmt in node.body:
            if isinstance(stmt, Call) and stmt.func == "println":
                println_count += 1

        # Emit format strings for each println (all are "%d\n" for now)
        for i in range(println_count):
            fmt_name = f"fmt.{i}"
            fmt_content = "%d\n"
            escaped = self.escape_string(fmt_content)
            # Length includes null terminator
            fmt_len = len(fmt_content) + 1
            self.emit(
                f'@{fmt_name} = private unnamed_addr constant [{fmt_len} x i8] c"{escaped}\\00", align 1'
            )
        if println_count > 0:
            self.emit()

        # Emit main function
        self.emit("define i32 @main() {")
        self.indent()
        self.println_idx = 0
        for stmt in node.body:
            self.gen_statement(stmt)
        self.emit("ret i32 0")
        self.dedent()
        self.emit("}")

    def gen_statement(self, node):
        """Generate IR for a statement."""
        if isinstance(node, Call) and node.func == "println":
            self.gen_println(node)
        else:
            raise CodeGenError(f"Unsupported statement: {type(node).__name__}")

    def gen_println(self, node: Call):
        """Generate a call to printf for println."""
        if len(node.args) != 1:
            raise CodeGenError("println expects exactly one argument")
        arg = node.args[0]
        if not isinstance(arg, IntLiteral):
            raise CodeGenError("println argument must be integer literal (for now)")

        fmt_name = f"fmt.{self.println_idx}"
        self.println_idx += 1
        fmt_len = len("%d\n") + 1  # includes null terminator

        # Load format string using GEP (no parentheses)
        self.emit(
            f"%fmt = getelementptr inbounds [{fmt_len} x i8], [{fmt_len} x i8]* @{fmt_name}, i32 0, i32 0"
        )
        # Prepare argument
        self.emit(f"%val = add i32 {arg.value}, 0")
        # Call printf
        self.emit(f"call i32 (i8*, ...) @printf(i8* %fmt, i32 %val)")
