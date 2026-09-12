# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os
import subprocess
import sys

# Generated *_pnnx.py is written to the test working directory.
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())


def run_pnnx(model, extra=None):
    candidates = [
        os.path.join("..", "..", "src", "pnnx"),
        os.path.join("..", "..", "src", "pnnx.exe"),
        os.path.join("..", "..", "src", "Release", "pnnx.exe"),
    ]
    exe = None
    for c in candidates:
        if os.path.isfile(c):
            exe = c
            break
    if exe is None:
        exe = candidates[0]
    cmd = [exe, model]
    if extra:
        cmd.append(extra)
    subprocess.check_call(cmd)
