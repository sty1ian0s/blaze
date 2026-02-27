#!/usr/bin/env python3.14
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

from src.blaze_ast import (
    ArrayLiteral,  # new nodes
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
from src.lexer import Lexer, LexerError, Token, TokenType


class ParseError(Exception):
    def __init__(self, message: str, token: Token):
        self.message = message
        self.token = token
        super().__init__(self._format())

    def _format(self) -> str:
        return f"{self.token.line}:{self.token.col}: error: {self.message} near '{self.token.raw}'"


class Parser:
    PRECEDENCE = {
        "==": 5,
        "!=": 5,
        "<": 5,
        ">": 5,
        "<=": 5,
        ">=": 5,
        "+": 10,
        "-": 10,
        "*": 20,
        "/": 20,
    }

    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self.tokens = list(lexer.tokenize())
        self.pos = 0
        self.current = self.tokens[0] if self.tokens else None

    def _advance(self):
        self.pos += 1
        self.current = self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def peek(self) -> Optional[TokenType]:
        return self.current.type if self.current else None

    def peek_token(self, offset: int = 0) -> Optional[Token]:
        idx = self.pos + offset
        return self.tokens[idx] if idx < len(self.tokens) else None

    def consume(self, expected_type: TokenType) -> Token:
        if self.current and self.current.type == expected_type:
            token = self.current
            self._advance()
            return token
        self._error(
            f"Expected {expected_type.name}, got {self.current.type.name if self.current else 'EOF'}"
        )

    def _error(self, message: str):
        if self.current:
            raise ParseError(message, self.current)
        last = self.tokens[-1] if self.tokens else Token(TokenType.EOF, "", 1, 1)
        raise ParseError(message, last)

    def parse_program(self) -> Module:
        items = []
        self._skip_newlines()
        while self.current and self.current.type != TokenType.EOF:
            if self.current.type == TokenType.KEYWORD and self.current.value == "fn":
                items.append(self.parse_function())
            else:
                items.append(self.parse_statement())
            self._skip_newlines()
        return Module(body=items, line=1, col=1)

    def _skip_newlines(self):
        while self.current and self.current.type == TokenType.NEWLINE:
            self._advance()

    def parse_function(self) -> Function:
        start = self.consume(TokenType.KEYWORD)  # 'fn'
        name = self.consume(TokenType.IDENT).value
        self.consume(TokenType.DELIMITER)  # '('
        params = self.parse_params()
        self.consume(TokenType.DELIMITER)  # ')'
        return_type = None
        if (
            self.current
            and self.current.type == TokenType.OPERATOR
            and self.current.value == "->"
        ):
            self.consume(TokenType.OPERATOR)
            return_type = self.consume(TokenType.IDENT).value
        body = self.parse_block()
        return Function(name, params, return_type, body, line=start.line, col=start.col)

    def parse_params(self) -> List[Param]:
        params = []
        if self.current and not (
            self.current.type == TokenType.DELIMITER and self.current.value == ")"
        ):
            params.append(self.parse_param())
            while (
                self.current
                and self.current.type == TokenType.DELIMITER
                and self.current.value == ","
            ):
                self.consume(TokenType.DELIMITER)  # ','
                params.append(self.parse_param())
        return params

    def parse_param(self) -> Param:
        name_tok = self.consume(TokenType.IDENT)
        name = name_tok.value
        type_ann = None
        if (
            self.current
            and self.current.type == TokenType.OPERATOR
            and self.current.value == ":"
        ):
            self.consume(TokenType.OPERATOR)
            type_ann = self.consume(TokenType.IDENT).value
        return Param(name, type_ann, line=name_tok.line, col=name_tok.col)

    def parse_block(self) -> List[Node]:
        self.consume(TokenType.DELIMITER)  # '{'
        stmts = []
        self._skip_newlines()
        while self.current and not (
            self.current.type == TokenType.DELIMITER and self.current.value == "}"
        ):
            stmts.append(self.parse_statement())
            self._skip_newlines()
        self.consume(TokenType.DELIMITER)  # '}'
        return stmts

    def parse_statement(self) -> Node:
        # Check for control flow keywords first
        if self.current and self.current.type == TokenType.KEYWORD:
            kw = self.current.value
            if kw == "if":
                return self.parse_if()
            if kw == "while":
                return self.parse_while()
            if kw == "loop":
                return self.parse_loop()
            if kw == "break":
                return self.parse_break()
            if kw == "continue":
                return self.parse_continue()
            if kw == "return":
                return self.parse_return()
            if kw == "let":
                return self.parse_let()
            if kw == "var":
                return self.parse_var()
        # Check for assignment: left-hand side expression followed by '='
        # We'll parse an expression, then if the next token is '=', treat as assignment.
        # Save position
        saved_pos = self.pos
        saved_current = self.current
        # Parse a primary/postfix expression as potential left-hand side
        lhs = self.parse_postfix()
        # Look ahead for '=' after newlines
        offset = 0
        nxt = self.peek_token(offset)
        while nxt and nxt.type == TokenType.NEWLINE:
            offset += 1
            nxt = self.peek_token(offset)
        if nxt and nxt.type == TokenType.OPERATOR and nxt.value == "=":
            # It's an assignment
            # Consume the left-hand side we already parsed? We've advanced the parser.
            # We need to backtrack to before parsing lhs, then parse assignment properly.
            # Simpler: rewind and call parse_assign directly.
            self.pos = saved_pos
            self.current = saved_current
            return self.parse_assign()
        # Otherwise, it's an expression statement; we have already parsed lhs and consumed it.
        # Continue parsing remaining postfix/binary ops? Actually lhs is just primary+postfix.
        # We need to continue parsing binary ops after lhs. We'll re-parse fully.
        # Rewind and parse full expression.
        self.pos = saved_pos
        self.current = saved_current
        expr = self.parse_expression()
        return expr

    def parse_if(self) -> If:
        start = self.consume(TokenType.KEYWORD)  # 'if'
        cond = self.parse_expression()
        then_body = self.parse_block()
        else_body = []
        if (
            self.current
            and self.current.type == TokenType.KEYWORD
            and self.current.value == "else"
        ):
            self.consume(TokenType.KEYWORD)  # 'else'
            else_body = self.parse_block()
        return If(cond, then_body, else_body, line=start.line, col=start.col)

    def parse_while(self) -> While:
        start = self.consume(TokenType.KEYWORD)  # 'while'
        cond = self.parse_expression()
        body = self.parse_block()
        return While(cond, body, line=start.line, col=start.col)

    def parse_loop(self) -> Loop:
        start = self.consume(TokenType.KEYWORD)  # 'loop'
        body = self.parse_block()
        return Loop(body, line=start.line, col=start.col)

    def parse_break(self) -> Break:
        start = self.consume(TokenType.KEYWORD)  # 'break'
        return Break(line=start.line, col=start.col)

    def parse_continue(self) -> Continue:
        start = self.consume(TokenType.KEYWORD)  # 'continue'
        return Continue(line=start.line, col=start.col)

    def parse_return(self) -> Return:
        ret = self.consume(TokenType.KEYWORD)  # 'return'
        if self.current and (
            self.current.type == TokenType.NEWLINE
            or (self.current.type == TokenType.DELIMITER and self.current.value == "}")
        ):
            return Return(None, line=ret.line, col=ret.col)
        expr = self.parse_expression()
        return Return(expr, line=ret.line, col=ret.col)

    def parse_let(self) -> Let:
        start = self.consume(TokenType.KEYWORD)  # 'let'
        name = self.consume(TokenType.IDENT).value
        type_ann = None
        if (
            self.current
            and self.current.type == TokenType.OPERATOR
            and self.current.value == ":"
        ):
            self.consume(TokenType.OPERATOR)
            type_ann = self.consume(TokenType.IDENT).value
        self.consume(TokenType.OPERATOR)  # '='
        value = self.parse_expression()
        return Let(name, type_ann, value, line=start.line, col=start.col)

    def parse_var(self) -> Var:
        start = self.consume(TokenType.KEYWORD)  # 'var'
        name = self.consume(TokenType.IDENT).value
        type_ann = None
        if (
            self.current
            and self.current.type == TokenType.OPERATOR
            and self.current.value == ":"
        ):
            self.consume(TokenType.OPERATOR)
            type_ann = self.consume(TokenType.IDENT).value
        self.consume(TokenType.OPERATOR)  # '='
        value = self.parse_expression()
        return Var(name, type_ann, value, line=start.line, col=start.col)

    def parse_assign(self) -> Assign:
        # Left-hand side can be a name or an index expression
        lhs = self.parse_postfix()
        # Skip newlines before '='
        while self.current and self.current.type == TokenType.NEWLINE:
            self._advance()
        self.consume(TokenType.OPERATOR)  # '='
        value = self.parse_expression()
        return Assign(lhs, value, line=lhs.line, col=lhs.col)

    def parse_expression(self) -> Node:
        """Entry point for expression parsing."""
        return self.parse_binary(0)

    def parse_binary(self, min_prec: int) -> Node:
        """Parse binary expressions with precedence climbing."""
        lhs = self.parse_postfix()
        while True:
            tok = self.current
            if not tok or tok.type != TokenType.OPERATOR:
                break
            op = tok.value
            if op not in self.PRECEDENCE:
                break
            prec = self.PRECEDENCE[op]
            if prec < min_prec:
                break
            self.consume(TokenType.OPERATOR)
            rhs = self.parse_binary(prec + 1)
            lhs = BinaryOp(op, lhs, rhs, line=lhs.line, col=lhs.col)
        return lhs

    def parse_postfix(self) -> Node:
        """Parse a primary expression followed by postfix operators (indexing)."""
        lhs = self.parse_primary()
        while True:
            if (
                self.current
                and self.current.type == TokenType.DELIMITER
                and self.current.value == "["
            ):
                self.consume(TokenType.DELIMITER)  # '['
                index = self.parse_expression()
                self.consume(TokenType.DELIMITER)  # ']'
                lhs = Index(lhs, index, line=lhs.line, col=lhs.col)
            else:
                break
        return lhs

    def parse_primary(self) -> Node:
        # Keyword expressions (if)
        if self.current and self.current.type == TokenType.KEYWORD:
            if self.current.value == "if":
                return self.parse_if()

        if self.current and self.current.type == TokenType.NUMBER:
            tok = self.consume(TokenType.NUMBER)
            try:
                value = int(tok.value, 0)
            except ValueError:
                self._error(f"Invalid integer literal: {tok.value}")
            return IntLiteral(value, line=tok.line, col=tok.col)

        elif self.current and self.current.type == TokenType.IDENT:
            tok = self.consume(TokenType.IDENT)
            if (
                self.current
                and self.current.type == TokenType.DELIMITER
                and self.current.value == "("
            ):
                args = self.parse_call_args()
                return Call(tok.value, args, line=tok.line, col=tok.col)
            else:
                return Name(tok.value, line=tok.line, col=tok.col)

        elif (
            self.current
            and self.current.type == TokenType.DELIMITER
            and self.current.value == "["
        ):
            start = self.current
            self.consume(TokenType.DELIMITER)  # '['
            elements = []
            if self.current and not (
                self.current.type == TokenType.DELIMITER and self.current.value == "]"
            ):
                elements.append(self.parse_expression())
                while (
                    self.current
                    and self.current.type == TokenType.DELIMITER
                    and self.current.value == ","
                ):
                    self.consume(TokenType.DELIMITER)  # ','
                    if (
                        self.current
                        and self.current.type == TokenType.DELIMITER
                        and self.current.value == "]"
                    ):
                        break
                    elements.append(self.parse_expression())
            self.consume(TokenType.DELIMITER)  # ']'
            return ArrayLiteral(elements, line=start.line, col=start.col)

        elif (
            self.current
            and self.current.type == TokenType.DELIMITER
            and self.current.value == "("
        ):
            open_tok = self.consume(TokenType.DELIMITER)
            expr = self.parse_expression()
            self.consume(TokenType.DELIMITER)  # ')'
            expr.line = open_tok.line
            expr.col = open_tok.col
            return expr

        self._error("Expected expression")

    def parse_call_args(self) -> List[Node]:
        self.consume(TokenType.DELIMITER)  # '('
        args = []
        if self.current and not (
            self.current.type == TokenType.DELIMITER and self.current.value == ")"
        ):
            args.append(self.parse_expression())
            while (
                self.current
                and self.current.type == TokenType.DELIMITER
                and self.current.value == ","
            ):
                self.consume(TokenType.DELIMITER)
                args.append(self.parse_expression())
        self.consume(TokenType.DELIMITER)  # ')'
        return args
