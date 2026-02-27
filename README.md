# Blaze Compiler (Bootstrap Stage 0)

This is the Python bootstrap compiler for the Blaze programming language.  
It follows the multi‑stage bootstrapping plan described in `AGENTS.md`.

## Current Status: Phase 1 – Lexer

- Project structure created.
- Test harness (`test.py`) and minimal compiler stub (`src/compiler.py`) added.
- **Lexer** implemented with full support for:
  - All token types (identifiers, numbers, floats, strings, chars, keywords, operators, delimiters, newline, EOF)
  - Line continuation (operators, comma, unclosed brackets)
  - Nested block comments and line comments
  - Integer literals (decimal, hex, binary, octal with underscores)
  - Float literals (with optional exponent)
  - String and character escapes (including `\u{XXXX}` Unicode)
  - Precise error reporting with filename, line, column, and source snippet
- Unit tests for lexer pass.

Next phase: **Parser** (empty program).

## License

Copyright 2025 The Blaze Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
