#!/usr/bin/env python3
"""Fail when a shared liblecore exports an undeclared or missing ABI-0 symbol."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPOSITORY_ROOT / "native/liblecore/ABI0_SYMBOLS.txt"


def expected_symbols(format_enabled: bool) -> set[str]:
    symbols = {
        line.strip()
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not format_enabled:
        symbols = {symbol for symbol in symbols if not symbol.startswith("lecore_format_")}
    return symbols


def exported_symbols(library: Path, nm_command: str) -> set[str]:
    command = shlex.split(nm_command) + ["-g", str(library)]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    symbols: set[str] = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        # GNU and LLVM nm put a one-letter symbol class immediately before the name.
        symbol_class, name = fields[-2:]
        if len(symbol_class) != 1 or symbol_class.upper() not in {"T", "D", "B", "R"}:
            continue
        if name.startswith("_lecore_"):  # Mach-O's conventional C prefix.
            name = name[1:]
        if name.startswith("lecore_"):
            symbols.add(name)
    return symbols


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--nm", default=os.environ.get("NM", "nm"))
    parser.add_argument("--format", choices=("on", "off"), default="on")
    arguments = parser.parse_args()

    if not arguments.library.is_file():
        parser.error(f"shared library does not exist: {arguments.library}")
    expected = expected_symbols(arguments.format == "on")
    actual = exported_symbols(arguments.library, arguments.nm)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        if missing:
            print("missing ABI-0 exports: " + ", ".join(missing), file=sys.stderr)
        if unexpected:
            print("unexpected liblecore exports: " + ", ".join(unexpected), file=sys.stderr)
        return 1
    print(f"liblecore ABI-0 exports: {len(actual)} symbols match the manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
