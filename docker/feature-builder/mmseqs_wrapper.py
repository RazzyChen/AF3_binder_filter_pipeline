#!/usr/bin/env python3
"""Delegate to MMseqs2 while enforcing a shared-node memory ceiling."""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    real_binary = os.environ.get("AERITH_MMSEQS_BINARY", "mmseqs")
    arguments = list(sys.argv[1:])
    if arguments and arguments[0] == "search" and "--split-memory-limit" not in arguments:
        arguments.extend(
            [
                "--split-memory-limit",
                os.environ.get("AERITH_MMSEQS_SPLIT_MEMORY_LIMIT", "32G"),
            ]
        )
    if arguments and arguments[0] == "search":
        requested_iterations = os.environ.get("AERITH_MMSEQS_NUM_ITERATIONS")
        if requested_iterations and "--num-iterations" in arguments:
            index = arguments.index("--num-iterations")
            arguments[index + 1] = requested_iterations
    return subprocess.call([real_binary, *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
