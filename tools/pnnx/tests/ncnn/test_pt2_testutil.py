# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from testutil_pt2 import (
    _as_output_tuple,
    _prepare_ncnn_input,
    _restore_ncnn_output,
)


def main():
    failed = []

    def check(cond, msg):
        print(("ok   " if cond else "FAIL ") + msg)
        if not cond:
            failed.append(msg)

    # batch_index 0/233: preserve/strip the explicit leading batch axis.
    check(len(_as_output_tuple(np.zeros((2, 3), dtype=np.float32))) == 1,
          "outputs: single tensor becomes one-item tuple")
    check(len(_as_output_tuple((np.zeros(1), np.zeros(1)))) == 2,
          "outputs: tuple arity is preserved")
    check(len(_as_output_tuple([np.zeros(1), np.zeros(1)])) == 2,
          "outputs: list arity is preserved")

    out = _restore_ncnn_output(np.zeros((2, 3), dtype=np.float32), np.zeros((1, 2, 3), dtype=np.float32), 0)
    check(out.shape == (1, 2, 3), "restore: batch_index=0 strips leading 1 and reshapes back")

    try:
        _restore_ncnn_output(np.zeros((3, 2), dtype=np.float32), np.zeros((2, 3), dtype=np.float32), 0)
        check(False, "restore: transposed shape with same numel rejected")
    except ValueError:
        check(True, "restore: transposed shape with same numel rejected")

    try:
        _restore_ncnn_output(np.zeros((2, 3), dtype=np.float32), np.zeros((4, 2, 3), dtype=np.float32), 0)
        check(False, "restore: batch_index=0 with reference leading dim != 1 rejected")
    except ValueError:
        check(True, "restore: batch_index=0 with reference leading dim != 1 rejected")

    out = _restore_ncnn_output(np.zeros((2, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.float32), 233)
    check(out.shape == (2, 3), "restore: batch_index=233 passthrough when shapes equal")

    out = _restore_ncnn_output(np.zeros((4, 5), dtype=np.float32), np.zeros((1, 4, 5), dtype=np.float32), 233)
    check(out.shape == (1, 4, 5), "restore: batch_index=233 strips size-1 leading dim of reference")

    try:
        _restore_ncnn_output(np.zeros((2, 3), dtype=np.float32), np.zeros((1, 2, 4), dtype=np.float32), 233)
        check(False, "restore: batch_index=233 mismatched shapes rejected")
    except ValueError:
        check(True, "restore: batch_index=233 mismatched shapes rejected")

    inp = _prepare_ncnn_input(torch.zeros(1, 4, 5), 233)
    check(inp.shape == (4, 5), "prepare: batch_index=233 strips size-1 leading dim")

    inp = _prepare_ncnn_input(torch.zeros(2, 4, 5), 233)
    check(inp.shape == (2, 4, 5), "prepare: batch_index=233 keeps input with leading dim != 1")

    inp = _prepare_ncnn_input(torch.zeros(1, 4, 5), 0)
    check(inp.shape == (1, 4, 5), "prepare: batch_index=0 passthrough")

    if failed:
        print("RESULT: %d failed" % len(failed))
        return 1
    print("RESULT: all pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
