#!/usr/bin/env python3
"""Emit the production Qwen runner envelope without model work for ilxyr CI."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "experiments" / "qwen35_acceptance" / "run.py"


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "qwen_executor_contract_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    # The generated experiment passes its normal runner and model arguments to
    # this executable. Requiring that first argument keeps the rehearsal tied
    # to the production entrypoint while deliberately avoiding model work.
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or Path(args[0]).resolve() != RUNNER.resolve():
        raise SystemExit("expected the generated Qwen runner as argv[1]")
    min_tokens = 4096
    if "--min-tokens" in args:
        index = args.index("--min-tokens")
        try:
            min_tokens = int(args[index + 1])
        except (IndexError, ValueError) as exc:
            raise SystemExit("invalid --min-tokens in generated runner arguments") from exc
    source_commit = None
    if "--expected-source-commit" in args:
        index = args.index("--expected-source-commit")
        try:
            source_commit = args[index + 1]
        except IndexError as exc:
            raise SystemExit(
                "missing --expected-source-commit value in generated arguments") from exc
    envelope = load_runner().executor_contract_fixture(
        min_tokens=min_tokens, source_commit=source_commit)
    print(json.dumps(envelope, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
