#!/usr/bin/env python3.14
# SPDX-License-Identifier: Apache-2.0

"""
Blaze compiler – driver that orchestrates lexing, parsing, semantic analysis,
code generation, and invocation of clang to produce an executable.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path so that imports from src work when running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.codegen import CodeGen, CodeGenError
from src.lexer import Lexer, LexerError
from src.parser import ParseError, Parser
from src.semantic import SemanticAnalyzer, SemanticError


class CompileError(Exception):
    """Wrapper for compilation errors."""

    pass


def run_clang(ir_path: Path, output_path: Path):
    """Run clang to compile the IR file directly to an executable."""
    clang_cmd = ["clang", str(ir_path), "-o", str(output_path)]
    proc = subprocess.run(clang_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise CompileError(f"clang failed:\n{proc.stderr}")


def main():
    parser = argparse.ArgumentParser(description="Blaze compiler")
    parser.add_argument("input", help="Input .blz file")
    parser.add_argument("-o", "--output", required=True, help="Output executable")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input file {input_path} not found", file=sys.stderr)
        sys.exit(1)

    source = input_path.read_text(encoding="utf-8")

    try:
        # Lexing
        lexer = Lexer(source, filename=str(input_path))
        # Parsing
        parser = Parser(lexer)
        ast = parser.parse_program()
        # Semantic analysis (placeholder)
        semantic = SemanticAnalyzer()
        ast = semantic.analyze(ast)
        # Code generation
        codegen = CodeGen()
        ir = codegen.generate(ast)
        # Write IR to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ll", delete=False) as f:
            f.write(ir)
            ir_path = Path(f.name)
        # Compile to executable using clang directly
        run_clang(ir_path, output_path)
        # Clean up IR file
        ir_path.unlink(missing_ok=True)
        print(f"Successfully compiled {input_path} to {output_path}", file=sys.stderr)

    except (LexerError, ParseError, SemanticError, CodeGenError, CompileError) as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
