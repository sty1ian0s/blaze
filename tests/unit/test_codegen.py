#!/usr/bin/env python3.14
# SPDX-License-Identifier: Apache-2.0

import re
import unittest

from src.blaze_ast import Module
from src.codegen import CodeGen


class TestCodeGen(unittest.TestCase):
    def assert_ir_contains(self, ir: str, pattern: str):
        self.assertIsNotNone(
            re.search(pattern, ir, re.MULTILINE),
            f"Pattern {pattern!r} not found in IR:\n{ir}",
        )

    def test_empty_module(self):
        module = Module(body=[])
        codegen = CodeGen()
        ir = codegen.generate(module)
        # Check that main function is defined and returns 0
        self.assert_ir_contains(ir, r"define\s+i32\s+@main\s*\(\s*\)\s*\{")
        self.assert_ir_contains(ir, r"ret\s+i32\s+0")
        # Should not have any other functions
        self.assertEqual(len(re.findall(r"define\s+", ir)), 1)


if __name__ == "__main__":
    unittest.main()
