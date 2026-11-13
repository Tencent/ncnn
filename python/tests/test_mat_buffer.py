# Copyright 2020 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import gc
import pytest
import numpy as np

import ncnn


def test_mat_from_temporary_array_keeps_buffer_alive():
    def make():
        return ncnn.Mat(np.full((3, 64, 64), 7.0, dtype=np.float32))

    mat = make()
    gc.collect()
    # churn the allocator so a dangling pointer would hold garbage
    for _ in range(50):
        np.random.rand(3, 64, 64).astype(np.float32)

    arr = np.array(mat)
    assert arr.min() == 7.0
    assert arr.max() == 7.0


def test_mat_from_strided_view_reads_expected_values():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)

    assert np.array_equal(np.array(ncnn.Mat(a.T)), a.T)
    assert np.array_equal(np.array(ncnn.Mat(a[:, ::2])), a[:, ::2])


def test_mat_from_image_style_views():
    img = np.random.rand(8, 9, 3).astype(np.float32)

    chw = img.transpose(2, 0, 1)
    assert np.array_equal(np.array(ncnn.Mat(chw)), chw)

    bgr = img[:, :, ::-1]
    assert np.array_equal(np.array(ncnn.Mat(bgr)), bgr)


def test_mat_shallow_copy_outlives_source():
    mat = ncnn.Mat(np.full((4, 4), 2.0, dtype=np.float32))
    shallow = ncnn.Mat(mat)
    del mat
    gc.collect()

    assert np.array(ncnn.Mat(shallow)).mean() == 2.0


def test_mat_from_contiguous_array_unchanged():
    a = np.random.rand(4, 5).astype(np.float32)
    assert np.array_equal(np.array(ncnn.Mat(a)), a)


def test_mat_rejects_zero_dim_array():
    with pytest.raises(RuntimeError, match="dims >= 1"):
        ncnn.Mat(np.array(3.0))

    with pytest.raises(RuntimeError, match="dims >= 1"):
        ncnn.Mat(np.float32(1.5))
