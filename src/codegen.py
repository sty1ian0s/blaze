#!/usr/bin/env python3.14
# SPDX-License-Identifier: Apache-2.0

"""
Blaze code generator – translates AST to LLVM IR.
"""

from typing import List, Optional

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


class CodeGenError(Exception):
    """Raised when code generation fails."""

    pass


class CodeGen:
    """Generates LLVM IR from an AST."""

    def __init__(self):
        self.output = []
        self.indent_level = 0
        self.println_idx = 0
        self.vars = {}
        self.functions = set()
        self.alloca_counter = 0
        self.label_counter = 0

    def indent(self):
        self.indent_level += 1

    def dedent(self):
        self.indent_level -= 1

    def emit(self, line: str = ""):
        if line:
            self.output.append("  " * self.indent_level + line)
        else:
            self.output.append("")

    def escape_string(self, s: str) -> str:
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
        name = f"alloca.{self.alloca_counter}"
        self.alloca_counter += 1
        return name

    def fresh_label(self, base: str = "label") -> str:
        name = f"{base}.{self.label_counter}"
        self.label_counter += 1
        return name

    def count_prints(self, node: Node) -> int:
        count = 0
        if isinstance(node, Call) and node.func == "println":
            count += 1
        elif isinstance(node, Module):
            for child in node.body:
                count += self.count_prints(child)
        elif isinstance(node, Function):
            for stmt in node.body:
                count += self.count_prints(stmt)
        elif hasattr(node, "args") and isinstance(node.args, list):
            for arg in node.args:
                if isinstance(arg, Node):
                    count += self.count_prints(arg)
        elif hasattr(node, "left") and isinstance(node.left, Node):
            count += self.count_prints(node.left)
            if isinstance(node.right, Node):
                count += self.count_prints(node.right)
        elif hasattr(node, "value") and isinstance(node.value, Node):
            count += self.count_prints(node.value)
        return count

    def generate(self, node) -> str:
        self.output = []
        self.println_idx = 0
        self.vars = {}
        self.functions = set()
        self.alloca_counter = 0
        self.label_counter = 0

        self.emit("; ModuleID = 'blaze_module'")
        self.emit('target triple = "x86_64-unknown-linux-gnu"')
        self.emit()
        self.emit("declare i32 @printf(i8*, ...)")
        self.emit()

        if not isinstance(node, Module):
            raise CodeGenError("Root node must be Module")

        # Emit format strings for all println calls
        println_count = self.count_prints(node)
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

        # Separate functions from top-level statements
        functions = []
        top_level_stmts = []
        for item in node.body:
            if isinstance(item, Function):
                functions.append(item)
            else:
                top_level_stmts.append(item)

        for func in functions:
            self.gen_function(func)

        self.gen_main_with_stmts(top_level_stmts)
        return "\n".join(self.output)

    def gen_main_with_stmts(self, stmts: List[Node]):
        self.emit("define i32 @main() {")
        self.indent()
        old_vars = self.vars
        self.vars = {}
        for stmt in stmts:
            self.gen_statement(stmt)
        self.emit("ret i32 0")
        self.dedent()
        self.emit("}")
        self.vars = old_vars

    def gen_function(self, node: Function):
        self.functions.add(node.name)
        ret_type = node.return_type if node.return_type else "i32"
        llvm_ret_type = "i32" if ret_type == "i32" else "void"
        param_list = ", ".join(f"i32 %{p.name}" for p in node.params)
        self.emit(f"define {llvm_ret_type} @{node.name}({param_list}) {{")
        self.indent()

        old_vars = self.vars
        self.vars = {}

        for param in node.params:
            alloca = self.fresh_alloca()
            self.emit(f"%{alloca} = alloca i32, align 4")
            self.emit(f"store i32 %{param.name}, i32* %{alloca}, align 4")
            self.vars[param.name] = (alloca, "i32")

        last_expr_reg = None
        has_return = False
        for stmt in node.body:
            reg = self.gen_statement(stmt)
            if isinstance(stmt, Return):
                has_return = True
            if reg is not None:
                last_expr_reg = reg

        if not has_return:
            if ret_type == "i32":
                if last_expr_reg is not None:
                    self.emit(f"ret i32 %{last_expr_reg}")
                else:
                    self.emit(
                        "ret i32 0"
                    )  # fallback (should not happen if semantic correct)
            else:
                self.emit("ret void")

        self.dedent()
        self.emit("}")
        self.vars = old_vars

    def gen_statement(self, node) -> Optional[str]:
        if isinstance(node, Let):
            self.gen_let(node)
            return None
        elif isinstance(node, Var):
            self.gen_var(node)
            return None
        elif isinstance(node, Assign):
            self.gen_assign(node)
            return None
        elif isinstance(node, Return):
            self.gen_return(node)
            return None
        elif isinstance(node, Call) and node.func == "println":
            self.gen_println(node)
            return None
        elif isinstance(node, Call):
            # Regular call that may return a value
            return self.gen_call(node)
        else:
            # Any other expression (BinaryOp, IntLiteral, Name)
            return self.gen_expression(node)

    def gen_return(self, node: Return):
        if node.value:
            val = self.gen_expression(node.value)
            self.emit(f"ret i32 %{val}")
        else:
            self.emit("ret void")

    def gen_call(self, node: Call) -> str:
        args = [f"i32 %{self.gen_expression(arg)}" for arg in node.args]
        arg_str = ", ".join(args)
        reg = f"tmp.{self.alloca_counter}"
        self.alloca_counter += 1
        self.emit(f"%{reg} = call i32 @{node.func}({arg_str})")
        return reg

    def gen_let(self, node: Let):
        alloca = self.fresh_alloca()
        self.emit(f"%{alloca} = alloca i32, align 4")
        val = self.gen_expression(node.value)
        self.emit(f"store i32 %{val}, i32* %{alloca}, align 4")
        self.vars[node.name] = (alloca, "i32")

    def gen_var(self, node: Var):
        alloca = self.fresh_alloca()
        self.emit(f"%{alloca} = alloca i32, align 4")
        val = self.gen_expression(node.value)
        self.emit(f"store i32 %{val}, i32* %{alloca}, align 4")
        self.vars[node.name] = (alloca, "i32")

    def gen_assign(self, node: Assign):
        if node.name not in self.vars:
            raise CodeGenError(f"undeclared variable `{node.name}`")
        alloca, typ = self.vars[node.name]
        val = self.gen_expression(node.value)
        self.emit(f"store i32 %{val}, i32* %{alloca}, align 4")

    def gen_println(self, node: Call):
        if len(node.args) != 1:
            raise CodeGenError("println expects exactly one argument")
        arg = self.gen_expression(node.args[0])
        fmt_name = f"fmt.{self.println_idx}"
        self.println_idx += 1
        fmt_len = len("%d\n") + 1
        fmt_reg = f"fmtp.{self.alloca_counter}"
        self.alloca_counter += 1
        self.emit(
            f"%{fmt_reg} = getelementptr inbounds [{fmt_len} x i8], [{fmt_len} x i8]* @{fmt_name}, i32 0, i32 0"
        )
        self.emit(f"call i32 (i8*, ...) @printf(i8* %{fmt_reg}, i32 %{arg})")

    def gen_expression(self, node) -> str:
        if isinstance(node, IntLiteral):
            reg = f"tmp.{self.alloca_counter}"
            self.alloca_counter += 1
            self.emit(f"%{reg} = add i32 {node.value}, 0")
            return reg
        elif isinstance(node, Name):
            if node.id not in self.vars:
                raise CodeGenError(f"undeclared variable `{node.id}`")
            alloca, typ = self.vars[node.id]
            reg = f"tmp.{self.alloca_counter}"
            self.alloca_counter += 1
            self.emit(f"%{reg} = load i32, i32* %{alloca}, align 4")
            return reg
        elif isinstance(node, BinaryOp):
            left = self.gen_expression(node.left)
            right = self.gen_expression(node.right)
            reg = f"tmp.{self.alloca_counter}"
            self.alloca_counter += 1
            if node.op == "+":
                self.emit(f"%{reg} = add i32 %{left}, %{right}")
            elif node.op == "-":
                self.emit(f"%{reg} = sub i32 %{left}, %{right}")
            elif node.op == "*":
                self.emit(f"%{reg} = mul i32 %{left}, %{right}")
            elif node.op == "/":
                self.emit(f"%{reg} = sdiv i32 %{left}, %{right}")
            else:
                raise CodeGenError(f"unsupported binary operator {node.op}")
            return reg
        elif isinstance(node, Call):
            reg = self.gen_call(node)
            return reg
        else:
            raise CodeGenError(f"unsupported expression: {type(node).__name__}")
