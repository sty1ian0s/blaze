#!/usr/bin/env python3.14
# SPDX-License-Identifier: Apache-2.0

"""
Blaze compiler – Phase 0 stub.
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Blaze compiler (bootstrap)")
    parser.add_argument("input", help="Input .blz file")
    parser.add_argument("-o", "--output", required=True, help="Output executable")
    args = parser.parse_args()

    # Phase 0: just print that compilation is not implemented and exit with error
    print("Blaze compiler: not implemented (Phase 0)", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
