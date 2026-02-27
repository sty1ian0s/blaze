#!/usr/bin/env python3.14
# SPDX-License-Identifier: Apache-2.0

import unittest

from src.lexer import Lexer, LexerError, TokenType


class TestLexer(unittest.TestCase):
    def token_types(self, source, include_eof=False):
        lexer = Lexer(source)
        types = [tok.type for tok in lexer.tokenize()]
        if not include_eof:
            types = [t for t in types if t != TokenType.EOF]
        return types

    def tokens(self, source, include_eof=False):
        lexer = Lexer(source)
        toks = list(lexer.tokenize())
        if not include_eof:
            toks = [t for t in toks if t.type != TokenType.EOF]
        return toks

    def test_empty(self):
        tokens = self.tokens("", include_eof=True)
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].type, TokenType.EOF)

    def test_newline(self):
        tokens = self.tokens("\n")
        self.assertEqual([t.type for t in tokens], [TokenType.NEWLINE])
        self.assertEqual(tokens[0].line, 1)
        self.assertEqual(tokens[0].col, 1)

    def test_identifiers(self):
        tokens = self.tokens("foo bar _baz")
        types = [t.type for t in tokens]
        self.assertEqual(types, [TokenType.IDENT, TokenType.IDENT, TokenType.IDENT])
        self.assertEqual(tokens[0].value, "foo")
        self.assertEqual(tokens[1].value, "bar")
        self.assertEqual(tokens[2].value, "_baz")

    def test_keywords(self):
        keywords = [
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
        ]
        source = " ".join(keywords)
        tokens = self.tokens(source)
        self.assertEqual(len(tokens), len(keywords))
        for t, kw in zip(tokens, keywords):
            self.assertEqual(t.type, TokenType.KEYWORD)
            self.assertEqual(t.value, kw)

    def test_numbers(self):
        cases = [
            ("123", TokenType.NUMBER, "123"),
            ("0x7B", TokenType.NUMBER, "0x7B"),
            ("0b1101", TokenType.NUMBER, "0b1101"),
            ("0o17", TokenType.NUMBER, "0o17"),
            ("1_000", TokenType.NUMBER, "1_000"),
            ("3.14", TokenType.FLOAT, "3.14"),
            ("2.5e-10", TokenType.FLOAT, "2.5e-10"),
            ("1f32", TokenType.FLOAT, "1f32"),
            (".5", TokenType.FLOAT, ".5"),
        ]
        for src, typ, val in cases:
            with self.subTest(src=src):
                tokens = self.tokens(src)
                self.assertEqual(len(tokens), 1)
                self.assertEqual(tokens[0].type, typ)
                self.assertEqual(tokens[0].value, val)

    def test_operators(self):
        ops = [
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
            "..",
            "..=",
            "=>",
            "->",
            "?.",
            "??",
            "<<",
            ">>",
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
        ]
        source = " ".join(ops)
        tokens = self.tokens(source)
        self.assertEqual(len(tokens), len(ops))
        for t, op in zip(tokens, ops):
            self.assertEqual(t.type, TokenType.OPERATOR)
            self.assertEqual(t.value, op)

    def test_delimiters(self):
        delims = ["(", ")", "[", "]", "{", "}", ",", ";"]
        source = " ".join(delims)
        tokens = self.tokens(source)
        self.assertEqual(len(tokens), len(delims))
        for t, delim in zip(tokens, delims):
            self.assertEqual(t.type, TokenType.DELIMITER)
            self.assertEqual(t.value, delim)

    def test_strings(self):
        cases = [
            ('"hello"', "hello"),
            ('"hello\\nworld"', "hello\nworld"),
            ('"\\t\\r\\"\\\\"', '\t\r"\\'),
            ('"\\u{7B}"', "{"),
            ('"""multi\nline"""', "multi\nline"),
        ]
        for src, expected in cases:
            with self.subTest(src=src):
                tokens = self.tokens(src)
                self.assertEqual(len(tokens), 1)
                self.assertEqual(tokens[0].type, TokenType.STRING)
                self.assertEqual(tokens[0].value, expected)

    def test_chars(self):
        cases = [
            ("'a'", "a"),
            ("'\\n'", "\n"),
            ("'\\u{7B}'", "{"),
        ]
        for src, expected in cases:
            with self.subTest(src=src):
                tokens = self.tokens(src)
                self.assertEqual(len(tokens), 1)
                self.assertEqual(tokens[0].type, TokenType.CHAR)
                self.assertEqual(tokens[0].value, expected)

    def test_line_continuation_operator(self):
        source = "let x = 1 +\n2"
        tokens = self.tokens(source)
        types = [t.type for t in tokens]
        # Expected: let, x, =, 1, +, 2, (no NEWLINE)
        self.assertEqual(
            types,
            [
                TokenType.KEYWORD,
                TokenType.IDENT,
                TokenType.OPERATOR,
                TokenType.NUMBER,
                TokenType.OPERATOR,
                TokenType.NUMBER,
            ],
        )

    def test_line_continuation_comma(self):
        source = "let arr = [1,\n2]"
        tokens = self.tokens(source)
        types = [t.type for t in tokens]
        # Expected: let, arr, =, [, 1, ,, 2, ]
        self.assertEqual(
            types,
            [
                TokenType.KEYWORD,
                TokenType.IDENT,
                TokenType.OPERATOR,
                TokenType.DELIMITER,
                TokenType.NUMBER,
                TokenType.DELIMITER,
                TokenType.NUMBER,
                TokenType.DELIMITER,
            ],
        )

    def test_line_continuation_unclosed_bracket(self):
        source = "let x = (\n1 + 2\n)"
        tokens = self.tokens(source)
        types = [t.type for t in tokens]
        # Expected: let, x, =, (, 1, +, 2, )
        self.assertEqual(
            types,
            [
                TokenType.KEYWORD,
                TokenType.IDENT,
                TokenType.OPERATOR,
                TokenType.DELIMITER,
                TokenType.NUMBER,
                TokenType.OPERATOR,
                TokenType.NUMBER,
                TokenType.DELIMITER,
            ],
        )

    def test_line_continuation_not_applied(self):
        source = "let x = 1\nlet y = 2"
        tokens = self.tokens(source)
        types = [t.type for t in tokens]
        # Should have NEWLINE between 1 and let
        self.assertEqual(types[4], TokenType.NEWLINE)  # after 1
        self.assertEqual(types[5], TokenType.KEYWORD)  # let

    def test_nested_comments(self):
        source = "/* outer /* inner */ outer */ let x = 5"
        tokens = self.tokens(source)
        types = [t.type for t in tokens]
        self.assertEqual(
            types,
            [TokenType.KEYWORD, TokenType.IDENT, TokenType.OPERATOR, TokenType.NUMBER],
        )

    def test_line_comment(self):
        source = "let x = 5 // this is a comment\ny = 6"
        tokens = self.tokens(source)
        values = [t.value for t in tokens if t.type != TokenType.NEWLINE]
        self.assertEqual(values, ["let", "x", "=", "5", "y", "=", "6"])

    def test_invalid_character(self):
        source = "let x = $"
        with self.assertRaises(LexerError) as cm:
            list(Lexer(source).tokenize())
        self.assertIn("Invalid character '$'", str(cm.exception))

    def test_unterminated_string(self):
        source = '"hello'
        with self.assertRaises(LexerError):
            list(Lexer(source).tokenize())

    def test_unterminated_char(self):
        source = "'a"
        with self.assertRaises(LexerError):
            list(Lexer(source).tokenize())

    def test_unterminated_block_comment(self):
        source = "/* comment"
        with self.assertRaises(LexerError):
            list(Lexer(source).tokenize())

    def test_unknown_escape(self):
        source = '"\\q"'
        with self.assertRaises(LexerError):
            list(Lexer(source).tokenize())


if __name__ == "__main__":
    unittest.main()
