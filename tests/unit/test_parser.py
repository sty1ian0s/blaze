#!/usr/bin/env python3.14
# SPDX-License-Identifier: Apache-2.0

import unittest

from src.ast import Module
from src.lexer import Lexer
from src.parser import ParseError, Parser


class TestParser(unittest.TestCase):
    def parse(self, source: str):
        """Parse source and return the AST module."""
        lexer = Lexer(source)
        parser = Parser(lexer)
        return parser.parse_program()

    def test_empty_program(self):
        module = self.parse("")
        self.assertIsInstance(module, Module)
        self.assertEqual(module.body, [])

    def test_empty_with_whitespace(self):
        module = self.parse("   \n  \t  ")
        self.assertIsInstance(module, Module)
        self.assertEqual(module.body, [])

    def test_empty_with_comments(self):
        module = self.parse("// just a comment\n/* block comment */")
        self.assertIsInstance(module, Module)
        self.assertEqual(module.body, [])

    def test_stray_token_error(self):
        with self.assertRaises(ParseError) as cm:
            self.parse("let")
        self.assertIn("Unexpected token", str(cm.exception))

    def test_stray_number_error(self):
        with self.assertRaises(ParseError):
            self.parse("123")

    def test_stray_operator_error(self):
        with self.assertRaises(ParseError):
            self.parse("+")

    def test_multiple_lines_without_items(self):
        module = self.parse("\n\n\n")
        self.assertEqual(module.body, [])


if __name__ == "__main__":
    unittest.main()
