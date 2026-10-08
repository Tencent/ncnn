# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_pnnx_reshape_semantics import test


if __name__ == "__main__":
    if test("../../src/pnnx", test_ncnn=True):
        exit(0)
    else:
        exit(1)
