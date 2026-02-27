#!/usr/bin/env python3.14
# SPDX-License-Identifier: Apache-2.0

import os
import re
import subprocess
import sys
from pathlib import Path

COMPILER = ["python3.14", "src/compiler.py"]
TESTS_DIR = Path("tests/integration")


def parse_test_file(path: Path):
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    expected = {
        "exit_code": None,
        "output": None,
        "compile_error": False,
        "error_pattern": None,
    }
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("//"):
            break
        line = line[2:].strip()
        # EXPECTED: exit_code=...
        m = re.match(r"EXPECTED:\s*exit_code=(\d+)", line, re.IGNORECASE)
        if m:
            expected["exit_code"] = int(m.group(1))
            i += 1
            continue
        m = re.match(r"OUTPUT:\s*(.*)", line, re.IGNORECASE)
        if m:
            output_str = m.group(1).strip()
            # Replace literal '\n' with actual newline
            output_str = output_str.replace("\\n", "\n")
            expected["output"] = output_str
            i += 1
            continue
        m = re.match(r"EXPECTED:\s*compile_error", line, re.IGNORECASE)
        if m:
            expected["compile_error"] = True
            i += 1
            continue
        m = re.match(r"ERROR:\s*(.*)", line, re.IGNORECASE)
        if m:
            expected["error_pattern"] = m.group(1).strip()
            i += 1
            continue
        i += 1
    return expected


def run_test(test_path: Path):
    expected = parse_test_file(test_path)
    tmp_exe = test_path.with_suffix(".out")
    cmd = COMPILER + [str(test_path), "-o", str(tmp_exe)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

    if expected["compile_error"]:
        if proc.returncode == 0:
            return False, f"Expected compilation error, but compiler succeeded"
        if expected["error_pattern"]:
            if expected["error_pattern"] not in proc.stderr:
                return (
                    False,
                    f"Expected error pattern {expected['error_pattern']!r} not found in stderr:\n{proc.stderr}",
                )
        return True, "Compilation failed as expected"

    if proc.returncode != 0:
        return False, f"Compilation failed (exit {proc.returncode}):\n{proc.stderr}"

    if not tmp_exe.exists():
        return False, f"Executable {tmp_exe} not created"

    run_proc = subprocess.run([str(tmp_exe)], capture_output=True, text=True)
    tmp_exe.unlink(missing_ok=True)

    if (
        expected["exit_code"] is not None
        and run_proc.returncode != expected["exit_code"]
    ):
        return (
            False,
            f"Exit code {run_proc.returncode} != expected {expected['exit_code']}",
        )

    if expected["output"] is not None:
        got = run_proc.stdout.rstrip("\n")
        want = expected["output"].rstrip("\n")
        if got != want:
            return False, f"Output mismatch:\n  got:  {got!r}\n  want: {want!r}"

    return True, "OK"


def main():
    tests = sorted(TESTS_DIR.rglob("*.blz"))
    if not tests:
        print("No integration tests found.")
        return 1

    failed = 0
    for test in tests:
        rel = os.path.relpath(str(test), start=str(Path.cwd()))
        print(f"TEST {rel} ... ", end="", flush=True)
        ok, msg = run_test(test)
        if ok:
            print("PASS")
        else:
            print("FAIL")
            print(f"  {msg}")
            failed += 1

    if failed:
        print(f"\n{len(tests) - failed} passed, {failed} failed")
        return 1
    print(f"\nAll {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
