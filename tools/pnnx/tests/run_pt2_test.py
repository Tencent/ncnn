#!/usr/bin/env python3

# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
from pathlib import Path
import subprocess
import sys

import torch

from pnnx_test_utils import PT2_SKIP_RETURN_CODE
from pnnx_test_utils import SUPPORTED
from pnnx_test_utils import pt2_producer_status


def main():
    parser = argparse.ArgumentParser(description="Run a pnnx PT2 test with its CTest environment")
    parser.add_argument("--pnnx", required=True, help="pnnx executable for PNNX_TEST_PNNX")
    parser.add_argument("--build-dir", required=True, help="directory prepended to PYTHONPATH")
    parser.add_argument("script", help="PT2 test script to execute")
    arguments = parser.parse_args()

    producer_status = pt2_producer_status()
    if producer_status != SUPPORTED:
        print(
            "%s: pt2 producer gate: %s (torch %s)"
            % (Path(arguments.script).stem, producer_status, torch.__version__)
        )
        return PT2_SKIP_RETURN_CODE

    environment = os.environ.copy()
    environment["PNNX_TEST_FORMAT"] = "pt2"
    environment["PNNX_TEST_PNNX"] = str(Path(arguments.pnnx).resolve())
    build_dir = str(Path(arguments.build_dir).resolve())
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        build_dir
        if not existing_pythonpath
        else build_dir + os.pathsep + existing_pythonpath
    )

    return subprocess.call([sys.executable, str(Path(arguments.script).resolve())], env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
