# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

# testutil_pt2 对拍 helper 的形状语义回归(直接 python 运行,无需 pyncnn):
#   python test_pt2_testutil.py
# 覆盖 docs/15 P2.3:helper 只允许 ncnn wrapper 明确声明的 batch 轴剥离
# 关系,拒绝仅凭元素数相等的任意 reshape——错序但 numel 相同的形状必须失败。

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from testutil_pt2 import _prepare_ncnn_input, _restore_ncnn_output


def main():
    failed = []

    def check(cond, msg):
        print(("ok   " if cond else "FAIL ") + msg)
        if not cond:
            failed.append(msg)

    # ---- _restore_ncnn_output ----

    # 正例:batch_index=0,ncnn 输出 (2,3) → 还原为 (1,2,3)
    out = _restore_ncnn_output(np.zeros((2, 3), dtype=np.float32), np.zeros((1, 2, 3), dtype=np.float32), 0)
    check(out.shape == (1, 2, 3), "restore: batch_index=0 strips leading 1 and reshapes back")

    # 负例 1:错序但 numel 相同((3,2) vs (2,3))→ 必须拒绝
    try:
        _restore_ncnn_output(np.zeros((3, 2), dtype=np.float32), np.zeros((2, 3), dtype=np.float32), 0)
        check(False, "restore: transposed shape with same numel rejected")
    except ValueError:
        check(True, "restore: transposed shape with same numel rejected")

    # 负例 2:batch_index=0 但 reference 首维非 1 → 拒绝
    try:
        _restore_ncnn_output(np.zeros((2, 3), dtype=np.float32), np.zeros((4, 2, 3), dtype=np.float32), 0)
        check(False, "restore: batch_index=0 with reference leading dim != 1 rejected")
    except ValueError:
        check(True, "restore: batch_index=0 with reference leading dim != 1 rejected")

    # 正例:batch_index=233 且形状一致 → 直通
    out = _restore_ncnn_output(np.zeros((2, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.float32), 233)
    check(out.shape == (2, 3), "restore: batch_index=233 passthrough when shapes equal")

    # 正例:batch_index=233,torch 参考首维 size-1、ncnn 剥掉该维 → 还原
    # (test_pt2_weights 场景:ref (1,4,5) vs ncnn (4,5))
    out = _restore_ncnn_output(np.zeros((4, 5), dtype=np.float32), np.zeros((1, 4, 5), dtype=np.float32), 233)
    check(out.shape == (1, 4, 5), "restore: batch_index=233 strips size-1 leading dim of reference")

    # 负例 3:batch_index=233 且形状不等 → 拒绝
    try:
        _restore_ncnn_output(np.zeros((2, 3), dtype=np.float32), np.zeros((1, 2, 4), dtype=np.float32), 233)
        check(False, "restore: batch_index=233 mismatched shapes rejected")
    except ValueError:
        check(True, "restore: batch_index=233 mismatched shapes rejected")

    # ---- _prepare_ncnn_input ----

    # 正例:batch_index=233,首维 size-1 的 3D 输入 → 剥离 batch 维
    inp = _prepare_ncnn_input(torch.zeros(1, 4, 5), 233)
    check(inp.shape == (4, 5), "prepare: batch_index=233 strips size-1 leading dim")

    # 边界:batch_index=233 但首维非 1 → 不剥离,保持原样
    inp = _prepare_ncnn_input(torch.zeros(2, 4, 5), 233)
    check(inp.shape == (2, 4, 5), "prepare: batch_index=233 keeps input with leading dim != 1")

    # 正例:batch_index=0 → 不做任何剥离
    inp = _prepare_ncnn_input(torch.zeros(1, 4, 5), 0)
    check(inp.shape == (1, 4, 5), "prepare: batch_index=0 passthrough")

    if failed:
        print("RESULT: %d failed" % len(failed))
        return 1
    print("RESULT: all pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
