#!/usr/bin/env python3.14
# SPDX-License-Identifier: Apache-2.0

"""
Blaze lexer – converts source text into a stream of tokens.
Handles line continuation, nested comments, escapes, and error reporting.
"""

from enum import IntEnum, auto
from typing import Generator, List, Optional, Tuple


class TokenType(IntEnum):
    """All token kinds produced by the lexer."""

    IDENT = auto()
    NUMBER = auto()
    FLOAT = auto()
    STRING = auto()
    CHAR = auto()
    KEYWORD = auto()
    OPERATOR = auto()
    DELIMITER = auto()
    NEWLINE = auto()
    EOF = auto()


class Token:
    """A single token with source location."""

    __slots__ = ("type", "value", "line", "col", "raw")

    def __init__(
        self,
        type: TokenType,
        value: str,
        line: int,
        col: int,
        raw: Optional[str] = None,
    ):
        self.type = type
        self.value = value  # semantic value (e.g., string contents without quotes)
        self.line = line  # 1‑based line number
        self.col = col  # 1‑based column of the first character
        self.raw = raw if raw is not None else value  # original source text

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.col})"


class LexerError(Exception):
    """Raised when the lexer encounters an invalid character or malformed literal."""

    def __init__(self, message: str, line: int, col: int, source_line: str = ""):
        self.message = message
        self.line = line
        self.col = col
        self.source_line = source_line
        super().__init__(self._format())

    def _format(self) -> str:
        snippet = self.source_line.strip()
        pointer = " " * (self.col - 1) + "^"
        return f"{self.line}:{self.col}: error: {self.message}\n{snippet}\n{pointer}"


class Lexer:
    """Blaze lexer. Produces tokens via the tokenize() generator."""

    # Keywords (from reference)
    KEYWORDS = {
        "as",
        "async",
        "break",
        "comptime",
        "const",
        "continue",
        "else",
        "ensure",
        "enum",
        "false",
        "fn",
        "for",
        "if",
        "import",
        "let",
        "macro",
        "match",
        "mut",
        "parallel",
        "pub",
        "require",
        "return",
        "spawn",
        "struct",
        "true",
        "type",
        "ui",
        "unsafe",
        "var",
        "while",
    }

    # Operators (single and multi‑character)
    OPERATORS = {
        # single char
        "+",
        "-",
        "*",
        "/",
        "%",
        "=",
        "<",
        ">",
        "!",
        "&",
        "|",
        "^",
        "?",
        "@",
        "_",
        ":",
        ".",
        # multi‑char (longest first for matching)
        "..=",
        "..",
        "=>",
        "->",
        "?.",
        "??",
        "<<=",
        ">>=",
        "<=",
        ">=",
        "==",
        "!=",
        "&&",
        "||",
        "+=",
        "-=",
        "*=",
        "/=",
        "%=",
        "&=",
        "|=",
        "^=",
        "<<",
        ">>",
    }
    OPERATORS_SORTED = sorted(OPERATORS, key=len, reverse=True)

    # Delimiters (single characters)
    DELIMITERS = {"(", ")", "[", "]", "{", "}", ",", ";"}

    # Whitespace (skipped except newline)
    WHITESPACE = {" ", "\t", "\r"}

    def __init__(self, source: str, filename: str = "<input>"):
        self.source = source
        self.filename = filename
        self.pos = 0  # current character index
        self.line = 1  # current line (1‑based)
        self.col = 1  # current column (1‑based)
        self.len = len(source)

        # Continuation state
        self._bracket_stack: List[str] = []  # stack of opening brackets '(', '[', '{'
        self._last_non_whitespace_token: Optional[Token] = (
            None  # last token that wasn't whitespace or comment
        )

    def _current(self) -> Optional[str]:
        """Return the current character or None if at EOF."""
        if self.pos >= self.len:
            return None
        return self.source[self.pos]

    def _advance(self, n: int = 1) -> None:
        """Advance the position by n characters, updating line/col."""
        for _ in range(n):
            if self.pos >= self.len:
                return
            ch = self.source[self.pos]
            self.pos += 1
            if ch == "\n":
                self.line += 1
                self.col = 1
            else:
                self.col += 1

    def _peek(self, offset: int = 1) -> Optional[str]:
        """Look ahead without advancing."""
        peek_pos = self.pos + offset
        if peek_pos >= self.len:
            return None
        return self.source[peek_pos]

    def _get_source_line(self, line_no: Optional[int] = None) -> str:
        """Return the source line at the given line number (or current line)."""
        if line_no is None:
            line_no = self.line
        lines = self.source.splitlines()
        if 1 <= line_no <= len(lines):
            return lines[line_no - 1]
        return ""

    @staticmethod
    def _is_hex_digit(ch: str) -> bool:
        return ch.isdigit() or ch in "abcdefABCDEF"

    def _skip_whitespace(self) -> None:
        """Skip over spaces, tabs, carriage returns, but not newlines."""
        while (ch := self._current()) is not None and ch in self.WHITESPACE:
            self._advance()

    def _skip_line_comment(self) -> None:
        """Skip from // to the end of the line."""
        self._advance(2)  # skip the '//'
        while (ch := self._current()) is not None and ch != "\n":
            self._advance()
        # Do NOT advance over the newline – it will be handled by the main loop

    def _skip_block_comment(self) -> None:
        """Skip a nested block comment /* ... */."""
        self._advance(2)  # skip '/*'
        depth = 1
        while depth > 0:
            ch = self._current()
            if ch is None:
                raise LexerError(
                    "Unterminated block comment",
                    self.line,
                    self.col,
                    self._get_source_line(),
                )
            if ch == "/" and self._peek() == "*":
                self._advance(2)
                depth += 1
            elif ch == "*" and self._peek() == "/":
                self._advance(2)
                depth -= 1
            else:
                self._advance()

    def _read_number(self) -> Token:
        """Read a numeric literal (integer or float) with support for underscores."""
        start_line, start_col = self.line, self.col
        start_pos = self.pos
        is_float = False

        ch = self._current()
        # Check for binary/hex/octal prefix
        if ch == "0":
            nch = self._peek()
            if nch in ("x", "X"):  # hexadecimal
                self._advance(2)
                while (ch := self._current()) is not None and (
                    ch.isdigit() or ch in "abcdefABCDEF" or ch == "_"
                ):
                    if ch != "_":
                        pass  # just consume
                    self._advance()
                # After consuming, we have the whole literal; no need to validate further.
                return Token(
                    TokenType.NUMBER,
                    self.source[start_pos : self.pos],
                    start_line,
                    start_col,
                )
            elif nch in ("b", "B"):  # binary
                self._advance(2)
                while (ch := self._current()) is not None and (
                    ch in "01_" or ch == "_"
                ):
                    self._advance()
                return Token(
                    TokenType.NUMBER,
                    self.source[start_pos : self.pos],
                    start_line,
                    start_col,
                )
            elif nch in ("o", "O"):  # octal
                self._advance(2)
                while (ch := self._current()) is not None and (
                    ch in "01234567_" or ch == "_"
                ):
                    self._advance()
                return Token(
                    TokenType.NUMBER,
                    self.source[start_pos : self.pos],
                    start_line,
                    start_col,
                )

        # Decimal integer or float
        while (ch := self._current()) is not None and (ch.isdigit() or ch == "_"):
            self._advance()
        if self._current() == "." and self._peek() and self._peek().isdigit():
            is_float = True
            self._advance()  # consume '.'
            while (ch := self._current()) is not None and (ch.isdigit() or ch == "_"):
                self._advance()
        # Check for exponent
        if (ch := self._current()) is not None and ch in "eE":
            is_float = True
            self._advance()
            if self._current() in "+-":
                self._advance()
            if (ch := self._current()) is None or (not ch.isdigit() and ch != "_"):
                raise LexerError(
                    "Expected exponent digits",
                    self.line,
                    self.col,
                    self._get_source_line(),
                )
            while (ch := self._current()) is not None and (ch.isdigit() or ch == "_"):
                self._advance()
        # Optional float suffix f32/f64
        # Optional float suffix f32/f64
        if (ch := self._current()) is not None and ch == "f":
            if self._peek() == "3" and self._peek(2) == "2":
                is_float = True
                self._advance(3)  # consume f32
            elif self._peek() == "6" and self._peek(2) == "4":
                is_float = True
                self._advance(3)  # consume f64
        value = self.source[start_pos : self.pos]
        token_type = TokenType.FLOAT if is_float else TokenType.NUMBER
        return Token(token_type, value, start_line, start_col, raw=value)

    def _read_string(self, quote_char: str) -> Token:
        """Read a string literal (supports escapes and triple quotes)."""
        start_line, start_col = self.line, self.col
        start_pos = self.pos
        triple = False

        # Check for triple quotes
        if self._current() == quote_char and self._peek() == quote_char:
            triple = True
            self._advance(3)  # skip opening triple quotes
        else:
            self._advance()  # skip opening quote

        content = []
        while True:
            ch = self._current()
            if ch is None:
                raise LexerError(
                    "Unterminated string literal",
                    start_line,
                    start_col,
                    self._get_source_line(start_line),
                )
            if triple:
                if (
                    ch == quote_char
                    and self._peek() == quote_char
                    and self._peek(2) == quote_char
                ):
                    self._advance(3)  # skip closing triple quotes
                    break
            else:
                if ch == quote_char:
                    self._advance()  # skip closing quote
                    break
                if ch == "\n" and not triple:
                    raise LexerError(
                        "Unterminated string literal",
                        start_line,
                        start_col,
                        self._get_source_line(start_line),
                    )

            if ch == "\\":
                # Escape sequence
                self._advance()
                esc = self._current()
                if esc is None:
                    raise LexerError(
                        "Unterminated escape sequence",
                        self.line,
                        self.col,
                        self._get_source_line(),
                    )
                self._advance()
                if esc == "n":
                    content.append("\n")
                elif esc == "t":
                    content.append("\t")
                elif esc == "r":
                    content.append("\r")
                elif esc == "\\":
                    content.append("\\")
                elif esc == '"':
                    content.append('"')
                elif esc == "'":
                    content.append("'")
                elif esc == "u" and self._current() == "{":
                    # Unicode escape: \u{...}
                    self._advance()  # skip '{'
                    hex_digits = []
                    while (ch := self._current()) is not None and ch != "}":
                        if not self._is_hex_digit(ch):
                            raise LexerError(
                                "Invalid hexadecimal digit in Unicode escape",
                                self.line,
                                self.col,
                                self._get_source_line(),
                            )
                        hex_digits.append(ch)
                        self._advance()
                    if self._current() != "}":
                        raise LexerError(
                            "Unclosed Unicode escape",
                            self.line,
                            self.col,
                            self._get_source_line(),
                        )
                    self._advance()  # skip '}'
                    if not hex_digits:
                        raise LexerError(
                            "Empty Unicode escape",
                            self.line,
                            self.col,
                            self._get_source_line(),
                        )
                    code = int("".join(hex_digits), 16)
                    try:
                        content.append(chr(code))
                    except ValueError:
                        raise LexerError(
                            f"Invalid Unicode code point U+{code:04X}",
                            self.line,
                            self.col,
                            self._get_source_line(),
                        )
                else:
                    raise LexerError(
                        f"Unknown escape sequence '\\{esc}'",
                        self.line,
                        self.col,
                        self._get_source_line(),
                    )
            else:
                content.append(ch)
                self._advance()

        value = "".join(content)
        raw = self.source[start_pos : self.pos]
        return Token(TokenType.STRING, value, start_line, start_col, raw=raw)

    def _read_char(self) -> Token:
        """Read a character literal: 'a' or '\n'."""
        start_line, start_col = self.line, self.col
        start_pos = self.pos
        self._advance()  # skip opening '
        ch = self._current()
        if ch is None:
            raise LexerError(
                "Unterminated character literal",
                start_line,
                start_col,
                self._get_source_line(start_line),
            )
        if ch == "\\":
            # Escape
            self._advance()
            esc = self._current()
            if esc is None:
                raise LexerError(
                    "Unterminated escape sequence",
                    self.line,
                    self.col,
                    self._get_source_line(),
                )
            self._advance()
            if esc == "n":
                char_value = "\n"
            elif esc == "t":
                char_value = "\t"
            elif esc == "r":
                char_value = "\r"
            elif esc == "\\":
                char_value = "\\"
            elif esc == "'":
                char_value = "'"
            elif esc == "u" and self._current() == "{":
                # Unicode escape: \u{...}
                self._advance()  # skip '{'
                hex_digits = []
                while (ch := self._current()) is not None and ch != "}":
                    if not self._is_hex_digit(ch):
                        raise LexerError(
                            "Invalid hexadecimal digit in Unicode escape",
                            self.line,
                            self.col,
                            self._get_source_line(),
                        )
                    hex_digits.append(ch)
                    self._advance()
                if self._current() != "}":
                    raise LexerError(
                        "Unclosed Unicode escape",
                        self.line,
                        self.col,
                        self._get_source_line(),
                    )
                self._advance()  # skip '}'
                if not hex_digits:
                    raise LexerError(
                        "Empty Unicode escape",
                        self.line,
                        self.col,
                        self._get_source_line(),
                    )
                code = int("".join(hex_digits), 16)
                try:
                    char_value = chr(code)
                except ValueError:
                    raise LexerError(
                        f"Invalid Unicode code point U+{code:04X}",
                        self.line,
                        self.col,
                        self._get_source_line(),
                    )
            else:
                raise LexerError(
                    f"Unknown escape sequence '\\{esc}'",
                    self.line,
                    self.col,
                    self._get_source_line(),
                )
        else:
            char_value = ch
            self._advance()
        if self._current() != "'":
            raise LexerError(
                "Expected closing quote for character literal",
                self.line,
                self.col,
                self._get_source_line(),
            )
        self._advance()  # skip closing '
        raw = self.source[start_pos : self.pos]
        return Token(TokenType.CHAR, char_value, start_line, start_col, raw=raw)

    def _read_identifier_or_keyword(self) -> Token:
        """Read an identifier (or keyword if it matches)."""
        start_line, start_col = self.line, self.col
        start_pos = self.pos
        while (ch := self._current()) is not None and (ch.isalnum() or ch == "_"):
            self._advance()
        value = self.source[start_pos : self.pos]
        # Special case: standalone underscore is an operator, not an identifier
        if value == "_":
            token_type = TokenType.OPERATOR
        else:
            token_type = (
                TokenType.KEYWORD if value in self.KEYWORDS else TokenType.IDENT
            )
        return Token(token_type, value, start_line, start_col)

    def _read_operator(self) -> Optional[Token]:
        """Read an operator (multi‑character if possible)."""
        start_line, start_col = self.line, self.col
        start_pos = self.pos
        for op in self.OPERATORS_SORTED:
            if self.source.startswith(op, self.pos):
                self._advance(len(op))
                return Token(TokenType.OPERATOR, op, start_line, start_col)
        return None

    def _read_delimiter(self) -> Optional[Token]:
        """Read a single‑character delimiter."""
        ch = self._current()
        if ch in self.DELIMITERS:
            start_line, start_col = self.line, self.col
            self._advance()
            return Token(TokenType.DELIMITER, ch, start_line, start_col)
        return None

    def tokenize(self) -> Generator[Token, None, None]:
        """Main lexer entry point: yields tokens until EOF."""
        while True:
            # Skip whitespace (but not newline)
            self._skip_whitespace()

            ch = self._current()
            if ch is None:
                break

            # Handle newline
            if ch == "\n":
                # Check if we are in a continuation context
                is_continuation = False
                if self._bracket_stack:
                    is_continuation = True
                elif self._last_non_whitespace_token:
                    last = self._last_non_whitespace_token
                    if (
                        last.type in (TokenType.OPERATOR, TokenType.DELIMITER)
                        and last.value == ","
                    ):
                        is_continuation = True
                    elif last.type == TokenType.OPERATOR:
                        is_continuation = True

                if is_continuation:
                    # Continuation: treat newline as whitespace, do not emit NEWLINE
                    self._advance()
                    continue
                else:
                    # End of logical line: emit NEWLINE token
                    self._advance()
                    yield Token(TokenType.NEWLINE, "\n", self.line - 1, self.col)
                    continue

            # Handle comments
            if ch == "/":
                next_ch = self._peek()
                if next_ch == "/":
                    self._skip_line_comment()
                    continue
                elif next_ch == "*":
                    self._skip_block_comment()
                    continue

            # Handle delimiters (before numbers, because '.' is ambiguous)
            delim_token = self._read_delimiter()
            if delim_token is not None:
                # Update bracket stack
                if delim_token.value in ("(", "[", "{"):
                    self._bracket_stack.append(delim_token.value)
                elif delim_token.value in (")", "]", "}"):
                    if self._bracket_stack:
                        matching = {"(": ")", "[": "]", "{": "}"}
                        expected = matching[self._bracket_stack[-1]]
                        if delim_token.value == expected:
                            self._bracket_stack.pop()
                        # else: mismatched – parser will catch
                yield delim_token
                self._last_non_whitespace_token = delim_token
                continue

            # Handle numbers (including those starting with '.')
            if ch.isdigit() or (ch == "." and self._peek() and self._peek().isdigit()):
                token = self._read_number()
                yield token
                self._last_non_whitespace_token = token
                continue

            # Handle strings and characters
            if ch in ('"', "'"):
                if ch == '"':
                    token = self._read_string('"')
                else:
                    token = self._read_char()
                yield token
                self._last_non_whitespace_token = token
                continue

            # Handle identifiers and keywords (letters or underscore)
            if ch.isalpha() or ch == "_":
                token = self._read_identifier_or_keyword()
                yield token
                self._last_non_whitespace_token = token
                continue

            # Handle operators
            op_token = self._read_operator()
            if op_token is not None:
                yield op_token
                self._last_non_whitespace_token = op_token
                continue

            # If we reach here, it's an invalid character
            raise LexerError(
                f"Invalid character '{ch}'",
                self.line,
                self.col,
                self._get_source_line(),
            )

        # End of file
        yield Token(TokenType.EOF, "", self.line, self.col)

    def tokenize_all(self) -> List[Token]:
        """Return a list of all tokens (convenience for testing)."""
        return list(self.tokenize())
