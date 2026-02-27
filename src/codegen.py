#!/usr/bin/env python3.14
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

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


class CodeGenError(Exception):
    pass


class CodeGen:
    def __init__(self):
        self.output = []
        self.indent_level = 0
        self.println_idx = 0
        self.vars = {}
        self.functions = set()
        self.alloca_counter = 0
        self.label_counter = 0
        self.blocks = []

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
        elif isinstance(node, If):
            for stmt in node.then_body:
                count += self.count_prints(stmt)
            for stmt in node.else_body:
                count += self.count_prints(stmt)
        elif isinstance(node, While) or isinstance(node, Loop):
            for stmt in node.body:
                count += self.count_prints(stmt)
        elif isinstance(node, ArrayLiteral):
            for elem in node.elements:
                count += self.count_prints(elem)
        elif isinstance(node, Index):
            count += self.count_prints(node.target)
            count += self.count_prints(node.index)
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
        self.blocks = []

        self.emit("; ModuleID = 'blaze_module'")
        self.emit('target triple = "x86_64-unknown-linux-gnu"')
        self.emit()
        self.emit("declare i32 @printf(i8*, ...)")
        self.emit()

        if not isinstance(node, Module):
            raise CodeGenError("Root node must be Module")

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
                    self.emit("ret i32 0")
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
            return self.gen_call(node)
        elif isinstance(node, If):
            self.gen_if_stmt(node)
            return None
        elif isinstance(node, While):
            self.gen_while(node)
            return None
        elif isinstance(node, Loop):
            self.gen_loop(node)
            return None
        elif isinstance(node, Break):
            self.gen_break(node)
            return None
        elif isinstance(node, Continue):
            self.gen_continue(node)
            return None
        else:
            return self.gen_expression(node)

    def gen_block_expr(self, stmts: List[Node]) -> Optional[str]:
        last_reg = None
        for stmt in stmts:
            reg = self.gen_statement(stmt)
            if reg is not None:
                last_reg = reg
        return last_reg

    def gen_if_expr(self, node: If) -> str:
        cond_reg = self.gen_expression(node.cond)
        then_label = self.fresh_label("then")
        else_label = self.fresh_label("else")
        merge_label = self.fresh_label("merge")

        cmp_reg = f"cmp.{self.alloca_counter}"
        self.alloca_counter += 1
        self.emit(f"%{cmp_reg} = icmp ne i32 %{cond_reg}, 0")
        self.emit(f"br i1 %{cmp_reg}, label %{then_label}, label %{else_label}")

        self.emit(f"{then_label}:")
        self.indent()
        then_reg = self.gen_block_expr(node.then_body)
        if then_reg is None:
            then_reg = f"tmp.{self.alloca_counter}"
            self.alloca_counter += 1
            self.emit(f"%{then_reg} = add i32 0, 0")
        self.emit(f"br label %{merge_label}")
        self.dedent()

        self.emit(f"{else_label}:")
        self.indent()
        else_reg = self.gen_block_expr(node.else_body)
        if else_reg is None:
            else_reg = f"tmp.{self.alloca_counter}"
            self.alloca_counter += 1
            self.emit(f"%{else_reg} = add i32 0, 0")
        self.emit(f"br label %{merge_label}")
        self.dedent()

        self.emit(f"{merge_label}:")
        self.indent()
        phi_reg = f"phi.{self.alloca_counter}"
        self.alloca_counter += 1
        self.emit(
            f"%{phi_reg} = phi i32 [ %{then_reg}, %{then_label} ], [ %{else_reg}, %{else_label} ]"
        )
        self.dedent()
        return phi_reg

    def gen_if_stmt(self, node: If):
        cond_reg = self.gen_expression(node.cond)
        then_label = self.fresh_label("then")
        else_label = self.fresh_label("else")
        merge_label = self.fresh_label("merge")

        cmp_reg = f"cmp.{self.alloca_counter}"
        self.alloca_counter += 1
        self.emit(f"%{cmp_reg} = icmp ne i32 %{cond_reg}, 0")
        self.emit(f"br i1 %{cmp_reg}, label %{then_label}, label %{else_label}")

        self.emit(f"{then_label}:")
        self.indent()
        for stmt in node.then_body:
            self.gen_statement(stmt)
        if not self._block_terminated():
            self.emit(f"br label %{merge_label}")
        self.dedent()

        self.emit(f"{else_label}:")
        self.indent()
        for stmt in node.else_body:
            self.gen_statement(stmt)
        if not self._block_terminated():
            self.emit(f"br label %{merge_label}")
        self.dedent()

        self.emit(f"{merge_label}:")
        # if both branches terminated, merge is unreachable
        if self._block_terminated():
            self.emit("unreachable")

    def _block_terminated(self) -> bool:
        if not self.output:
            return False
        last = self.output[-1].strip()
        return last.startswith("ret ") or last.startswith("br label %")

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
        val_reg = self.gen_expression(node.value)
        if isinstance(node.value, ArrayLiteral):
            ptr_alloca = self.fresh_alloca()
            self.emit(f"%{ptr_alloca} = alloca i32*, align 8")
            self.emit(f"store i32* %{val_reg}, i32** %{ptr_alloca}, align 8")
            self.vars[node.name] = (ptr_alloca, "i32*")
        else:
            alloca = self.fresh_alloca()
            self.emit(f"%{alloca} = alloca i32, align 4")
            self.emit(f"store i32 %{val_reg}, i32* %{alloca}, align 4")
            self.vars[node.name] = (alloca, "i32")

    def gen_var(self, node: Var):
        val_reg = self.gen_expression(node.value)
        if isinstance(node.value, ArrayLiteral):
            ptr_alloca = self.fresh_alloca()
            self.emit(f"%{ptr_alloca} = alloca i32*, align 8")
            self.emit(f"store i32* %{val_reg}, i32** %{ptr_alloca}, align 8")
            self.vars[node.name] = (ptr_alloca, "i32*")
        else:
            alloca = self.fresh_alloca()
            self.emit(f"%{alloca} = alloca i32, align 4")
            self.emit(f"store i32 %{val_reg}, i32* %{alloca}, align 4")
            self.vars[node.name] = (alloca, "i32")

    def gen_assign(self, node: Assign):
        if isinstance(node.target, Name):
            if node.target.id not in self.vars:
                raise CodeGenError(f"undeclared variable `{node.target.id}`")
            alloca, typ = self.vars[node.target.id]
            if typ == "i32*":
                # assign to array pointer variable? Not allowed? For now, only i32 variables can be assigned.
                raise CodeGenError(
                    f"cannot assign to array variable `{node.target.id}` directly"
                )
            val = self.gen_expression(node.value)
            self.emit(f"store i32 %{val}, i32* %{alloca}, align 4")
        elif isinstance(node.target, Index):
            # array element assignment
            base_reg = self.gen_expression(node.target.target)
            index_reg = self.gen_expression(node.target.index)
            ptr_reg = f"ptr.{self.alloca_counter}"
            self.alloca_counter += 1
            self.emit(
                f"%{ptr_reg} = getelementptr inbounds i32, i32* %{base_reg}, i32 %{index_reg}"
            )
            val_reg = self.gen_expression(node.value)
            self.emit(f"store i32 %{val_reg}, i32* %{ptr_reg}, align 4")
        else:
            raise CodeGenError(
                f"unsupported assignment target: {type(node.target).__name__}"
            )

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
            if typ == "i32*":
                self.emit(f"%{reg} = load i32*, i32** %{alloca}, align 8")
            else:
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
            elif node.op in ("==", "!=", "<", ">", "<=", ">="):
                cmp_reg = f"cmp.{self.alloca_counter}"
                self.alloca_counter += 1
                if node.op == "==":
                    self.emit(f"%{cmp_reg} = icmp eq i32 %{left}, %{right}")
                elif node.op == "!=":
                    self.emit(f"%{cmp_reg} = icmp ne i32 %{left}, %{right}")
                elif node.op == "<":
                    self.emit(f"%{cmp_reg} = icmp slt i32 %{left}, %{right}")
                elif node.op == ">":
                    self.emit(f"%{cmp_reg} = icmp sgt i32 %{left}, %{right}")
                elif node.op == "<=":
                    self.emit(f"%{cmp_reg} = icmp sle i32 %{left}, %{right}")
                elif node.op == ">=":
                    self.emit(f"%{cmp_reg} = icmp sge i32 %{left}, %{right}")
                self.emit(f"%{reg} = zext i1 %{cmp_reg} to i32")
            else:
                raise CodeGenError(f"unsupported binary operator {node.op}")
            return reg
        elif isinstance(node, Call):
            return self.gen_call(node)
        elif isinstance(node, If):
            return self.gen_if_expr(node)
        elif isinstance(node, ArrayLiteral):
            elem_count = len(node.elements)
            alloca_name = self.fresh_alloca()
            self.emit(f"%{alloca_name} = alloca [{elem_count} x i32], align 4")
            for i, elem in enumerate(node.elements):
                elem_reg = self.gen_expression(elem)
                ptr_reg = f"ptr.{self.alloca_counter}"
                self.alloca_counter += 1
                self.emit(
                    f"%{ptr_reg} = getelementptr inbounds [{elem_count} x i32], [{elem_count} x i32]* %{alloca_name}, i32 0, i32 {i}"
                )
                self.emit(f"store i32 %{elem_reg}, i32* %{ptr_reg}, align 4")
            ptr_cast = f"cast.{self.alloca_counter}"
            self.alloca_counter += 1
            self.emit(
                f"%{ptr_cast} = bitcast [{elem_count} x i32]* %{alloca_name} to i32*"
            )
            return ptr_cast
        elif isinstance(node, Index):
            target_reg = self.gen_expression(node.target)
            index_reg = self.gen_expression(node.index)
            ptr_reg = f"ptr.{self.alloca_counter}"
            self.alloca_counter += 1
            self.emit(
                f"%{ptr_reg} = getelementptr inbounds i32, i32* %{target_reg}, i32 %{index_reg}"
            )
            val_reg = f"tmp.{self.alloca_counter}"
            self.alloca_counter += 1
            self.emit(f"%{val_reg} = load i32, i32* %{ptr_reg}, align 4")
            return val_reg
        else:
            raise CodeGenError(f"unsupported expression: {type(node).__name__}")

    def gen_while(self, node: While):
        cond_label = self.fresh_label("while.cond")
        body_label = self.fresh_label("while.body")
        exit_label = self.fresh_label("while.exit")
        self.blocks.append((cond_label, exit_label, node.label))
        self.emit(f"br label %{cond_label}")
        self.emit(f"{cond_label}:")
        self.indent()
        cond_reg = self.gen_expression(node.cond)
        cmp_reg = f"cmp.{self.alloca_counter}"
        self.alloca_counter += 1
        self.emit(f"%{cmp_reg} = icmp ne i32 %{cond_reg}, 0")
        self.emit(f"br i1 %{cmp_reg}, label %{body_label}, label %{exit_label}")
        self.dedent()
        self.emit(f"{body_label}:")
        self.indent()
        for stmt in node.body:
            self.gen_statement(stmt)
        self.emit(f"br label %{cond_label}")
        self.dedent()
        self.emit(f"{exit_label}:")
        self.blocks.pop()

    def gen_loop(self, node: Loop):
        body_label = self.fresh_label("loop.body")
        exit_label = self.fresh_label("loop.exit")
        self.blocks.append((body_label, exit_label, node.label))
        self.emit(f"br label %{body_label}")
        self.emit(f"{body_label}:")
        self.indent()
        for stmt in node.body:
            self.gen_statement(stmt)
        self.emit(f"br label %{body_label}")
        self.dedent()
        self.emit(f"{exit_label}:")
        self.blocks.pop()

    def gen_break(self, node: Break):
        if not self.blocks:
            raise CodeGenError("break outside loop")
        _, exit_label, _ = self.blocks[-1]
        self.emit(f"br label %{exit_label}")

    def gen_continue(self, node: Continue):
        if not self.blocks:
            raise CodeGenError("continue outside loop")
        header_label, _, _ = self.blocks[-1]
        self.emit(f"br label %{header_label}")
