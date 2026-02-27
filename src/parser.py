#!/usr/bin/env python3.14
# SPDX-License-Identifier: Apache-2.0

"""
Blaze parser – recursive descent parser that consumes tokens from the lexer
and produces an AST.
"""

from typing import List, Optional

from src.ast import Module, Node
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

    def parse_program(self) -> Module:
        """Parse a whole Blaze program."""
        items: List[Node] = []
        # Skip leading newlines (empty lines)
        while self.peek() == TokenType.NEWLINE:
            self.consume(TokenType.NEWLINE)
        # After skipping, if we are at EOF, it's an empty program
        if self.peek() is None or self.peek() == TokenType.EOF:
            return Module(body=items)
        # Otherwise, error
        self._error(
            "Unexpected token at top level (only empty programs are supported in Phase 2)"
        )
