#!/usr/bin/env python3.14
# SPDX-License-Identifier: Apache-2.0

"""
Blaze code generator – translates AST to LLVM IR.
"""

from src.blaze_ast import Assign, BinaryOp, Call, IntLiteral, Let, Module, Name, Var


class CodeGenError(Exception):
    """Raised when code generation fails."""

    pass


class CodeGen:
    """Generates LLVM IR from an AST."""

    def __init__(self):
        self.output = []
        self.indent_level = 0
        self.println_idx = 0
        self.vars = {}  # name -> (alloca_name, type)
        self.alloca_counter = 0

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

    def fresh_alloca(self) -> str:
        """Return a fresh alloca name."""
        name = f"alloca.{self.alloca_counter}"
        self.alloca_counter += 1
        return name

    def generate(self, node) -> str:
        """Generate IR for the given AST node and return as string."""
        self.output = []
        self.println_idx = 0
        self.vars = {}
        self.alloca_counter = 0

        # Emit module header
        self.emit("; ModuleID = 'blaze_module'")
        self.emit('target triple = "x86_64-unknown-linux-gnu"')
        self.emit()
        # Declare external printf
        self.emit("declare i32 @printf(i8*, ...)")
        self.emit()

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
        self.gen_module(node)
        self.emit("ret i32 0")
        self.dedent()
        self.emit("}")
        return "\n".join(self.output)

    def gen_module(self, node: Module):
        """Generate IR for a module body."""
        for stmt in node.body:
            self.gen_statement(stmt)

    def gen_statement(self, node):
        """Generate IR for a statement."""
        if isinstance(node, Let):
            self.gen_let(node)
        elif isinstance(node, Var):
            self.gen_var(node)
        elif isinstance(node, Assign):
            self.gen_assign(node)
        elif isinstance(node, Call) and node.func == "println":
            self.gen_println(node)
        else:
            raise CodeGenError(f"Unsupported statement: {type(node).__name__}")

    def gen_let(self, node: Let):
        """Generate IR for an immutable let binding."""
        # Allocate space
        alloca_name = self.fresh_alloca()
        self.emit(f"%{alloca_name} = alloca i32, align 4")
        # Compute value
        val_reg = self.gen_expression(node.value)
        # Store
        self.emit(f"store i32 %{val_reg}, i32* %{alloca_name}, align 4")
        # Record variable
        self.vars[node.name] = (alloca_name, "i32")

    def gen_var(self, node: Var):
        """Generate IR for a mutable var binding (same as let for now)."""
        # For now, immutable and mutable are handled the same (no extra checks)
        alloca_name = self.fresh_alloca()
        self.emit(f"%{alloca_name} = alloca i32, align 4")
        val_reg = self.gen_expression(node.value)
        self.emit(f"store i32 %{val_reg}, i32* %{alloca_name}, align 4")
        self.vars[node.name] = (alloca_name, "i32")

    def gen_assign(self, node: Assign):
        """Generate IR for an assignment."""
        if node.name not in self.vars:
            raise CodeGenError(f"undeclared variable `{node.name}`")
        alloca_name, typ = self.vars[node.name]
        val_reg = self.gen_expression(node.value)
        self.emit(f"store i32 %{val_reg}, i32* %{alloca_name}, align 4")

    def gen_expression(self, node) -> str:
        """Generate IR for an expression, returning the name of the register holding the result."""
        if isinstance(node, IntLiteral):
            # Return a constant value as an immediate
            # We'll create a register by adding 0
            reg = f"tmp.{self.alloca_counter}"
            self.alloca_counter += 1
            self.emit(f"%{reg} = add i32 {node.value}, 0")
            return reg
        elif isinstance(node, Name):
            if node.id not in self.vars:
                raise CodeGenError(f"undeclared variable `{node.id}`")
            alloca_name, typ = self.vars[node.id]
            reg = f"tmp.{self.alloca_counter}"
            self.alloca_counter += 1
            self.emit(f"%{reg} = load i32, i32* %{alloca_name}, align 4")
            return reg
        elif isinstance(node, BinaryOp):
            left_reg = self.gen_expression(node.left)
            right_reg = self.gen_expression(node.right)
            reg = f"tmp.{self.alloca_counter}"
            self.alloca_counter += 1
            if node.op == "+":
                self.emit(f"%{reg} = add i32 %{left_reg}, %{right_reg}")
            elif node.op == "-":
                self.emit(f"%{reg} = sub i32 %{left_reg}, %{right_reg}")
            elif node.op == "*":
                self.emit(f"%{reg} = mul i32 %{left_reg}, %{right_reg}")
            elif node.op == "/":
                self.emit(f"%{reg} = sdiv i32 %{left_reg}, %{right_reg}")
            else:
                raise CodeGenError(f"unsupported binary operator {node.op}")
            return reg
        else:
            raise CodeGenError(f"unsupported expression: {type(node).__name__}")

    def gen_println(self, node: Call):
        """Generate a call to printf for println."""
        if len(node.args) != 1:
            raise CodeGenError("println expects exactly one argument")
        arg = node.args[0]
        arg_reg = self.gen_expression(arg)
        fmt_name = f"fmt.{self.println_idx}"
        self.println_idx += 1
        fmt_len = len("%d\n") + 1
        # Load format string
        fmt_reg = f"fmtp.{self.alloca_counter}"
        self.alloca_counter += 1
        self.emit(
            f"%{fmt_reg} = getelementptr inbounds [{fmt_len} x i8], [{fmt_len} x i8]* @{fmt_name}, i32 0, i32 0"
        )
        # Call printf
        self.emit(f"call i32 (i8*, ...) @printf(i8* %{fmt_reg}, i32 %{arg_reg})")
