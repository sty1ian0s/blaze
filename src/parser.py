#!/usr/bin/env python3.14
# SPDX-License-Identifier: Apache-2.0

"""
Blaze parser – recursive descent parser that consumes tokens from the lexer
and produces an AST with source locations.
"""

from typing import List, Optional

from src.blaze_ast import (
    Assign,
    BinaryOp,
    Call,
    IntLiteral,
    Let,
    Module,
    Name,
    Node,
    Var,
)
from src.lexer import Lexer, LexerError, Token, TokenType


class ParseError(Exception):
    """Raised when the parser encounters a syntax error."""

    def __init__(self, message: str, token: Token):
        self.message = message
        self.token = token
        super().__init__(self._format())

    def _format(self) -> str:
        return f"{self.token.line}:{self.token.col}: error: {self.message} near '{self.token.raw}'"


class Parser:
    """Recursive descent parser for Blaze."""

    # Precedence levels for binary operators (higher = tighter)
    PRECEDENCE = {
        "+": 10,
        "-": 10,
        "*": 20,
        "/": 20,
    }

    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self.tokens = list(lexer.tokenize())  # load all tokens for easy lookahead
        self.pos = 0
        self.current = self.tokens[0] if self.tokens else None

    def _advance(self) -> None:
        """Move to the next token."""
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current = self.tokens[self.pos]
        else:
            self.current = None

    def peek(self) -> Optional[TokenType]:
        """Return the type of the current token, or None if at EOF."""
        return self.current.type if self.current else None

    def peek_token(self, offset: int = 0) -> Optional[Token]:
        """Peek ahead without consuming."""
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return None

    def consume(self, expected_type: TokenType) -> Token:
        """If the current token is of the expected type, consume it and return it; otherwise raise ParseError."""
        if self.current and self.current.type == expected_type:
            token = self.current
            self._advance()
            return token
        self._error(
            f"Expected {expected_type.name}, got {self.current.type.name if self.current else 'EOF'}"
        )

    def expect(self, expected_type: TokenType) -> Token:
        """Like consume, but does not advance (just checks)."""
        if self.current and self.current.type == expected_type:
            return self.current
        self._error(
            f"Expected {expected_type.name}, got {self.current.type.name if self.current else 'EOF'}"
        )

    def _error(self, message: str) -> None:
        """Raise a ParseError with the current token (or last token if at EOF)."""
        if self.current:
            raise ParseError(message, self.current)
        else:
            # At EOF, use the last token if available
            last_token = (
                self.tokens[-1] if self.tokens else Token(TokenType.EOF, "", 1, 1)
            )
            raise ParseError(message, last_token)

    def parse_program(self) -> Module:
        """Parse a whole Blaze program."""
        items: List[Node] = []
        # Skip leading newlines
        self._skip_newlines()
        # Parse statements until EOF
        while self.current is not None and self.current.type != TokenType.EOF:
            stmt = self.parse_statement()
            items.append(stmt)
            # After a statement, skip newlines (including multiple)
            self._skip_newlines()
        return Module(body=items, line=1, col=1)

    def _skip_newlines(self) -> None:
        """Consume all consecutive NEWLINE tokens."""
        while self.current and self.current.type == TokenType.NEWLINE:
            self._advance()

    def parse_statement(self) -> Node:
        """Parse a single statement."""
        # Check for let/var keywords
        if self.current and self.current.type == TokenType.KEYWORD:
            kw = self.current.value
            if kw == "let":
                return self.parse_let()
            elif kw == "var":
                return self.parse_var()
        # Check for assignment: identifier followed by '=' (skip newlines)
        if self.current and self.current.type == TokenType.IDENT:
            # Look ahead to see if the next non-newline token is '='
            offset = 1
            next_tok = self.peek_token(offset)
            while next_tok and next_tok.type == TokenType.NEWLINE:
                offset += 1
                next_tok = self.peek_token(offset)
            if (
                next_tok
                and next_tok.type == TokenType.OPERATOR
                and next_tok.value == "="
            ):
                return self.parse_assign()
        # Otherwise, parse as expression (must be println)
        expr = self.parse_expression()
        if isinstance(expr, Call) and expr.func == "println":
            return expr
        self._error("Expected statement (let, var, assignment, or println)")

    def parse_let(self) -> Let:
        """Parse 'let name [: type] = expr'."""
        start_token = self.current  # 'let'
        self.consume(TokenType.KEYWORD)  # 'let'
        name_token = self.consume(TokenType.IDENT)
        name = name_token.value
        type_ann = None
        if (
            self.current
            and self.current.type == TokenType.OPERATOR
            and self.current.value == ":"
        ):
            self.consume(TokenType.OPERATOR)  # ':'
            type_token = self.consume(TokenType.IDENT)
            type_ann = type_token.value
        self.consume(TokenType.OPERATOR)  # '='
        value = self.parse_expression()
        return Let(name, type_ann, value, line=start_token.line, col=start_token.col)

    def parse_var(self) -> Var:
        """Parse 'var name [: type] = expr'."""
        start_token = self.current
        self.consume(TokenType.KEYWORD)  # 'var'
        name_token = self.consume(TokenType.IDENT)
        name = name_token.value
        type_ann = None
        if (
            self.current
            and self.current.type == TokenType.OPERATOR
            and self.current.value == ":"
        ):
            self.consume(TokenType.OPERATOR)  # ':'
            type_token = self.consume(TokenType.IDENT)
            type_ann = type_token.value
        self.consume(TokenType.OPERATOR)  # '='
        value = self.parse_expression()
        return Var(name, type_ann, value, line=start_token.line, col=start_token.col)

    def parse_assign(self) -> Assign:
        """Parse 'name = expr'."""
        name_token = self.consume(TokenType.IDENT)
        name = name_token.value
        # Skip newlines before '='
        while self.current and self.current.type == TokenType.NEWLINE:
            self._advance()
        self.consume(TokenType.OPERATOR)  # '='
        value = self.parse_expression()
        return Assign(name, value, line=name_token.line, col=name_token.col)

    def parse_expression(self) -> Node:
        """Parse an expression with precedence."""
        return self.parse_binary(0)

    def parse_binary(self, min_prec: int) -> Node:
        """Parse binary expressions using precedence climbing."""
        lhs = self.parse_primary()
        while True:
            tok = self.current
            if tok is None or tok.type != TokenType.OPERATOR:
                break
            op = tok.value
            if op not in self.PRECEDENCE:
                break
            prec = self.PRECEDENCE[op]
            if prec < min_prec:
                break
            self.consume(TokenType.OPERATOR)  # consume operator
            rhs = self.parse_binary(prec + 1)
            lhs = BinaryOp(
                op, lhs, rhs, line=lhs.line, col=lhs.col
            )  # use left's location
        return lhs

    def parse_primary(self) -> Node:
        """Parse a primary expression: integer literal, identifier, or parenthesized expression."""
        if self.current and self.current.type == TokenType.NUMBER:
            token = self.consume(TokenType.NUMBER)
            try:
                value = int(token.value)
            except ValueError:
                self._error(f"Invalid integer literal: {token.value}")
            return IntLiteral(value, line=token.line, col=token.col)
        elif self.current and self.current.type == TokenType.IDENT:
            token = self.consume(TokenType.IDENT)
            # Check if it's a function call (next token is '(')
            if (
                self.current
                and self.current.type == TokenType.DELIMITER
                and self.current.value == "("
            ):
                args = self.parse_call_args()
                return Call(token.value, args, line=token.line, col=token.col)
            else:
                return Name(token.value, line=token.line, col=token.col)
        elif (
            self.current
            and self.current.type == TokenType.DELIMITER
            and self.current.value == "("
        ):
            open_token = self.consume(TokenType.DELIMITER)  # '('
            expr = self.parse_expression()
            self.consume(TokenType.DELIMITER)  # ')'
            expr.line = open_token.line  # propagate location
            expr.col = open_token.col
            return expr
        self._error("Expected expression")

    def parse_call_args(self) -> List[Node]:
        """Parse arguments inside parentheses."""
        self.consume(TokenType.DELIMITER)  # '('
        args = []
        if self.current and not (
            self.current.type == TokenType.DELIMITER and self.current.value == ")"
        ):
            # Parse first argument
            args.append(self.parse_expression())
            # Parse remaining arguments separated by commas
            while (
                self.current
                and self.current.type == TokenType.OPERATOR
                and self.current.value == ","
            ):
                self.consume(TokenType.OPERATOR)  # ','
                args.append(self.parse_expression())
        self.consume(TokenType.DELIMITER)  # ')'
        return args
