#!/usr/bin/env python3.14
# SPDX-License-Identifier: Apache-2.0

"""
Blaze parser – recursive descent parser that consumes tokens from the lexer
and produces an AST.
"""

from typing import List, Optional

from src.blaze_ast import Call, IntLiteral, Module, Node
from src.lexer import Lexer, LexerError, Token, TokenType


class ParseError(Exception):

    def __init__(self, message: str, token: Token):
        self.message = message
        self.token = token
        super().__init__(self._format())

    def _format(self) -> str:
        return f"{self.token.line}:{self.token.col}: error: {self.message} near '{self.token.raw}'"


class Parser:
    """Recursive descent parser for Blaze."""

    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self.tokens = iter(lexer.tokenize())
        self.current: Optional[Token] = None
        self._advance()

    def _advance(self) -> None:
        """Move to the next token."""
        try:
            self.current = next(self.tokens)
        except StopIteration:
            self.current = None

    def peek(self) -> Optional[TokenType]:
        """Return the type of the current token, or None if at EOF."""
        return self.current.type if self.current else None

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
            # At EOF, we don't have a token; use a dummy token with last known position
            raise ParseError(
                message, Token(TokenType.EOF, "", self.lexer.line, self.lexer.col)
            )
            raise ParseError(message, Token(TokenType.EOF, "", self.lexer.line, self.lexer.col))

    def parse_program(self) -> Module:
        """Parse a whole Blaze program."""
        items: List[Node] = []
        # Skip leading newlines
        while self.peek() == TokenType.NEWLINE:
            self.consume(TokenType.NEWLINE)
        # Parse statements until EOF
        while self.peek() is not None and self.peek() != TokenType.EOF:
            stmt = self.parse_statement()
            items.append(stmt)
            # Expect newline or EOF after statement
            if self.peek() == TokenType.NEWLINE:
                self.consume(TokenType.NEWLINE)
                # Skip any additional newlines before next statement
                while self.peek() == TokenType.NEWLINE:
                    self.consume(TokenType.NEWLINE)
            elif self.peek() is not None and self.peek() != TokenType.EOF:
                self._error("Expected newline after statement")
        return Module(body=items)

    def parse_statement(self) -> Node:
        """Parse a single statement. For Phase 4, only 'println(integer)' is supported."""
        # Check if it's a println call (println is an identifier, not a keyword)
        if self.peek() == TokenType.IDENT and self.current.value == "println":
            return self.parse_println()
        self._error("Expected statement (println)")

    def parse_println(self) -> Call:
        """Parse 'println(integer)'. Returns a Call node."""
        self.consume(TokenType.IDENT)  # 'println'
        self.consume(TokenType.DELIMITER)  # '('
        # Parse argument – for now only integer literal
        arg = self.parse_expression()
        self.consume(TokenType.DELIMITER)  # ')'
        return Call(func="println", args=[arg])

    def parse_expression(self) -> Node:
        """Parse an expression. For Phase 4, only integer literals."""
        if self.peek() == TokenType.NUMBER:
            token = self.consume(TokenType.NUMBER)
            # Convert to integer (handles decimal only for now)
            try:
                value = int(token.value)
            except ValueError:
                self._error(f"Invalid integer literal: {token.value}")
            return IntLiteral(value)
        self._error("Expected integer literal")
