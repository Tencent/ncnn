# Copyright 2026 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import sys

import numpy as np
import pytest

import ncnn


def test_mat_item_index_checks():
    mat = ncnn.Mat(4)
    for i in range(4):
        mat[i] = float(i)
    # a for loop stops at the end of the data instead of reading past it
    assert [v for v in mat] == [0.0, 1.0, 2.0, 3.0]
    with pytest.raises(IndexError):
        mat[4]
    with pytest.raises(IndexError):
        mat[4] = 1.0
    # huge indices must not wrap around in the bound check
    for i in (sys.maxsize, 2 * sys.maxsize + 1):
        with pytest.raises(IndexError):
            mat[i]
        with pytest.raises(IndexError):
            mat[i] = 1.0


def test_mat_accessor_index_checks():
    mat = ncnn.Mat((4, 3, 2))  # w=4, h=3, c=2
    mat.fill(1.0)
    assert mat.channel(1).w == 4 and mat.channel_range(0, 2).c == 2
    assert len(mat.row(2)) == 4 and mat.row_range(1, 2).h == 2
    with pytest.raises(IndexError):
        mat.channel(2)
    with pytest.raises(IndexError):
        mat.channel(-1)
    with pytest.raises(IndexError):
        mat.channel_range(1, 2)
    # row, depth and range are bounded by the Mat's data: 2 channels of 3 rows of 4 floats
    rows = mat.total() // mat.w
    depths = mat.total() // (mat.w * mat.h)
    assert len(mat.row(rows - 1)) == 4
    with pytest.raises(IndexError):
        mat.row(rows)
    with pytest.raises(IndexError):
        mat.row_range(rows - 1, 2)
    with pytest.raises(IndexError):
        mat.depth(depths)
    with pytest.raises(IndexError):
        mat.depth_range(depths - 1, 2)
    with pytest.raises(IndexError):
        mat.range(0, mat.total() + 1)
    with pytest.raises(IndexError):
        mat.batch(1)


def test_substract_mean_normalize_checks_sizes():
    mat = ncnn.Mat((2, 2, 3))
    mat.fill(1.0)
    with pytest.raises(RuntimeError, match="one value per channel"):
        mat.substract_mean_normalize([1.0], [])
    with pytest.raises(RuntimeError, match="one value per channel"):
        mat.substract_mean_normalize([], [2.0])
    mat.substract_mean_normalize([1.0, 1.0, 1.0], [])
    assert np.all(np.array(mat) == 0.0)
