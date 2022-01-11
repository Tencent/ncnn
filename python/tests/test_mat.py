# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import sys
import numpy as np
import pytest

import ncnn


def test_mat_dims1():
    mat = ncnn.Mat(1)
    assert mat.dims == 1 and mat.w == 1
    mat = ncnn.Mat(2, elemsize=4)
    assert mat.dims == 1 and mat.w == 2 and mat.elemsize == 4
    mat = ncnn.Mat(3, elemsize=4, elempack=1)
    assert mat.dims == 1 and mat.w == 3 and mat.elemsize == 4 and mat.elempack == 1
    mat = ncnn.Mat(4, elemsize=4, elempack=1, allocator=None)
    assert (
        mat.dims == 1
        and mat.w == 4
        and mat.elemsize == 4
        and mat.elempack == 1
        and mat.allocator == None
    )

    mat = ncnn.Mat((1,))
    assert mat.dims == 1 and mat.w == 1
    mat = ncnn.Mat((2,), elemsize=4)
    assert mat.dims == 1 and mat.w == 2 and mat.elemsize == 4
    mat = ncnn.Mat((3,), elemsize=4, elempack=1)
    assert mat.dims == 1 and mat.w == 3 and mat.elemsize == 4 and mat.elempack == 1
    mat = ncnn.Mat((4,), elemsize=4, elempack=1, allocator=None)
    assert (
        mat.dims == 1
        and mat.w == 4
        and mat.elemsize == 4
        and mat.elempack == 1
        and mat.allocator == None
    )


def test_mat_dims2():
    mat = ncnn.Mat(1, 2)
    assert mat.dims == 2 and mat.w == 1 and mat.h == 2
    mat = ncnn.Mat(3, 4, elemsize=4)
    assert mat.dims == 2 and mat.w == 3 and mat.h == 4 and mat.elemsize == 4
    mat = ncnn.Mat(5, 6, elemsize=4, elempack=1)
    assert (
        mat.dims == 2
        and mat.w == 5
        and mat.h == 6
        and mat.elemsize == 4
        and mat.elempack == 1
    )
    mat = ncnn.Mat(7, 8, elemsize=4, elempack=1, allocator=None)
    assert (
        mat.dims == 2
        and mat.w == 7
        and mat.h == 8
        and mat.elemsize == 4
        and mat.elempack == 1
        and mat.allocator == None
    )

    mat = ncnn.Mat((1, 2))
    assert mat.dims == 2 and mat.w == 1 and mat.h == 2
    mat = ncnn.Mat((3, 4), elemsize=4)
    assert mat.dims == 2 and mat.w == 3 and mat.h == 4 and mat.elemsize == 4
    mat = ncnn.Mat((5, 6), elemsize=4, elempack=1)
    assert (
        mat.dims == 2
        and mat.w == 5
        and mat.h == 6
        and mat.elemsize == 4
        and mat.elempack == 1
    )
    mat = ncnn.Mat((7, 8), elemsize=4, elempack=1, allocator=None)
    assert (
        mat.dims == 2
        and mat.w == 7
        and mat.h == 8
        and mat.elemsize == 4
        and mat.elempack == 1
        and mat.allocator == None
    )


def test_mat_dims3():
    mat = ncnn.Mat(1, 2, 3)
    assert mat.dims == 3 and mat.w == 1 and mat.h == 2 and mat.c == 3
    mat = ncnn.Mat(4, 5, 6, elemsize=4)
    assert (
        mat.dims == 3 and mat.w == 4 and mat.h == 5 and mat.c == 6 and mat.elemsize == 4
    )
    mat = ncnn.Mat(7, 8, 9, elemsize=4, elempack=1)
    assert (
        mat.dims == 3
        and mat.w == 7
        and mat.h == 8
        and mat.c == 9
        and mat.elemsize == 4
        and mat.elempack == 1
    )
    mat = ncnn.Mat(10, 11, 12, elemsize=4, elempack=1, allocator=None)
    assert (
        mat.dims == 3
        and mat.w == 10
        and mat.h == 11
        and mat.c == 12
        and mat.elemsize == 4
        and mat.elempack == 1
        and mat.allocator == None
    )

    mat = ncnn.Mat((1, 2, 3))
    assert mat.dims == 3 and mat.w == 1 and mat.h == 2 and mat.c == 3
    mat = ncnn.Mat((4, 5, 6), elemsize=4)
    assert (
        mat.dims == 3 and mat.w == 4 and mat.h == 5 and mat.c == 6 and mat.elemsize == 4
    )
    mat = ncnn.Mat((7, 8, 9), elemsize=4, elempack=1)
    assert (
        mat.dims == 3
        and mat.w == 7
        and mat.h == 8
        and mat.c == 9
        and mat.elemsize == 4
        and mat.elempack == 1
    )
    mat = ncnn.Mat((10, 11, 12), elemsize=4, elempack=1, allocator=None)
    assert (
        mat.dims == 3
        and mat.w == 10
        and mat.h == 11
        and mat.c == 12
        and mat.elemsize == 4
        and mat.elempack == 1
        and mat.allocator == None
    )


def test_mat_dims4():
    mat = ncnn.Mat(1, 2, 3, 4)
    assert mat.dims == 4 and mat.w == 1 and mat.h == 2 and mat.d == 3 and mat.c == 4
    mat = ncnn.Mat(4, 5, 6, 7, elemsize=4)
    assert (
        mat.dims == 4 and mat.w == 4 and mat.h == 5 and mat.d == 6 and mat.c == 7 and mat.elemsize == 4
    )
    mat = ncnn.Mat(7, 8, 9, 10, elemsize=4, elempack=1)
    assert (
        mat.dims == 4
        and mat.w == 7
        and mat.h == 8
        and mat.d == 9
        and mat.c == 10
        and mat.elemsize == 4
        and mat.elempack == 1
    )
    mat = ncnn.Mat(10, 11, 12, 13, elemsize=4, elempack=1, allocator=None)
    assert (
        mat.dims == 4
        and mat.w == 10
        and mat.h == 11
        and mat.d == 12
        and mat.c == 13
        and mat.elemsize == 4
        and mat.elempack == 1
        and mat.allocator == None
    )

    mat = ncnn.Mat((1, 2, 3, 4))
    assert mat.dims == 4 and mat.w == 1 and mat.h == 2 and mat.d == 3 and mat.c == 4
    mat = ncnn.Mat((4, 5, 6, 7), elemsize=4)
    assert (
        mat.dims == 4 and mat.w == 4 and mat.h == 5 and mat.d == 6 and mat.c == 7 and mat.elemsize == 4
    )
    mat = ncnn.Mat((7, 8, 9, 10), elemsize=4, elempack=1)
    assert (
        mat.dims == 4
        and mat.w == 7
        and mat.h == 8
        and mat.d == 9
        and mat.c == 10
        and mat.elemsize == 4
        and mat.elempack == 1
    )
    mat = ncnn.Mat((10, 11, 12, 13), elemsize=4, elempack=1, allocator=None)
    assert (
        mat.dims == 4
        and mat.w == 10
        and mat.h == 11
        and mat.d == 12
        and mat.c == 13
        and mat.elemsize == 4
        and mat.elempack == 1
        and mat.allocator == None
    )


def test_numpy():
    mat = ncnn.Mat(1)
    array = np.array(mat)
    assert mat.dims == array.ndim and mat.w == array.shape[0]
    mat = ncnn.Mat(2, 3)
    array = np.array(mat)
    assert (
        mat.dims == array.ndim and mat.w == array.shape[1] and mat.h == array.shape[0]
    )
    mat = ncnn.Mat(4, 5, 6)
    array = np.array(mat)
    assert (
        mat.dims == array.ndim
        and mat.w == array.shape[2]
        and mat.h == array.shape[1]
        and mat.c == array.shape[0]
    )
    mat = ncnn.Mat(7, 8, 9, 10)
    array = np.array(mat)
    assert (
        mat.dims == array.ndim
        and mat.w == array.shape[3]
        and mat.h == array.shape[2]
        and mat.d == array.shape[1]
        and mat.c == array.shape[0]
    )

    mat = ncnn.Mat(1, elemsize=1)
    array = np.array(mat)
    assert array.dtype == np.int8
    mat = ncnn.Mat(1, elemsize=2)
    array = np.array(mat)
    assert array.dtype == np.float16
    # pybind11 def_buffer throw bug
    # with pytest.raises(RuntimeError) as execinfo:
    #     mat = ncnn.Mat(1, elemsize=3)
    #     array = np.array(mat)
    #     assert "convert ncnn.Mat to numpy.ndarray only elemsize 1, 2, 4 support now, but given 3" in str(
    #         execinfo.value
    #     )
    assert array.dtype == np.float16
    mat = ncnn.Mat(1, elemsize=4)
    array = np.array(mat)
    assert array.dtype == np.float32

    mat = np.random.randint(0, 128, size=(12,)).astype(np.uint8)
    array = np.array(mat)
    assert (mat == array).all()
    mat = np.random.rand(12).astype(np.float32)
    array = np.array(mat)
    assert (mat == array).all()
    mat = np.random.randint(0, 128, size=(12, 11)).astype(np.uint8)
    array = np.array(mat)
    assert (mat == array).all()
    mat = np.random.rand(12, 11).astype(np.float32)
    array = np.array(mat)
    assert (mat == array).all()
    mat = np.random.randint(0, 256, size=(12, 11, 3)).astype(np.uint8)
    array = np.array(mat)
    assert (mat == array).all()
    mat = np.random.rand(12, 11, 3).astype(np.float32)
    array = np.array(mat)
    assert (mat == array).all()
    mat = np.random.randint(0, 256, size=(12, 11, 7, 3)).astype(np.uint8)
    array = np.array(mat)
    assert (mat == array).all()
    mat = np.random.rand(12, 11, 7, 3).astype(np.float32)
    array = np.array(mat)
    assert (mat == array).all()


def test_fill():
    mat = ncnn.Mat(1)
    mat.fill(1.0)
    array = np.array(mat)
    assert np.abs(array[0] - 1.0) < sys.float_info.min


def test_clone():
    mat1 = ncnn.Mat(1)
    mat2 = mat1.clone()
    assert mat1.dims == mat2.dims and mat1.w == mat2.w

    mat1 = ncnn.Mat(2, 3)
    mat2 = mat1.clone()
    assert mat1.dims == mat2.dims and mat1.w == mat2.w and mat1.h == mat2.h

    mat1 = ncnn.Mat(4, 5, 6)
    mat2 = mat1.clone()
    assert (
        mat1.dims == mat2.dims
        and mat1.w == mat2.w
        and mat1.h == mat2.h
        and mat1.c == mat2.c
    )

    mat1 = ncnn.Mat(7, 8, 9, 10)
    mat2 = mat1.clone()
    assert (
        mat1.dims == mat2.dims
        and mat1.w == mat2.w
        and mat1.h == mat2.h
        and mat1.d == mat2.d
        and mat1.c == mat2.c
    )

    mat1 = ncnn.Mat((1,))
    mat2 = mat1.clone()
    assert mat1.dims == mat2.dims and mat1.w == mat2.w

    mat1 = ncnn.Mat((2, 3))
    mat2 = mat1.clone()
    assert mat1.dims == mat2.dims and mat1.w == mat2.w and mat1.h == mat2.h

    mat1 = ncnn.Mat((4, 5, 6))
    mat2 = mat1.clone()
    assert (
        mat1.dims == mat2.dims
        and mat1.w == mat2.w
        and mat1.h == mat2.h
        and mat1.c == mat2.c
    )

    mat1 = ncnn.Mat((7, 8, 9, 10))
    mat2 = mat1.clone()
    assert (
        mat1.dims == mat2.dims
        and mat1.w == mat2.w
        and mat1.h == mat2.h
        and mat1.d == mat2.d
        and mat1.c == mat2.c
    )


def test_clone_from():
    mat2 = ncnn.Mat()

    mat1 = ncnn.Mat(1)
    mat2.clone_from(mat1)
    assert mat1.dims == mat2.dims and mat1.w == mat2.w

    mat1 = ncnn.Mat(2, 3)
    mat2.clone_from(mat1)
    assert mat1.dims == mat2.dims and mat1.w == mat2.w and mat1.h == mat2.h

    mat1 = ncnn.Mat(4, 5, 6)
    mat2.clone_from(mat1)
    assert (
        mat1.dims == mat2.dims
        and mat1.w == mat2.w
        and mat1.h == mat2.h
        and mat1.c == mat2.c
    )

    mat1 = ncnn.Mat(7, 8, 9, 10)
    mat2.clone_from(mat1)
    assert (
        mat1.dims == mat2.dims
        and mat1.w == mat2.w
        and mat1.h == mat2.h
        and mat1.d == mat2.d
        and mat1.c == mat2.c
    )

    mat1 = ncnn.Mat((1,))
    mat2.clone_from(mat1)
    assert mat1.dims == mat2.dims and mat1.w == mat2.w

    mat1 = ncnn.Mat((2, 3))
    mat2.clone_from(mat1)
    assert mat1.dims == mat2.dims and mat1.w == mat2.w and mat1.h == mat2.h

    mat1 = ncnn.Mat((4, 5, 6))
    mat2.clone_from(mat1)
    assert (
        mat1.dims == mat2.dims
        and mat1.w == mat2.w
        and mat1.h == mat2.h
        and mat1.c == mat2.c
    )

    mat1 = ncnn.Mat((7, 8, 9, 10))
    mat2.clone_from(mat1)
    assert (
        mat1.dims == mat2.dims
        and mat1.w == mat2.w
        and mat1.h == mat2.h
        and mat1.d == mat2.d
        and mat1.c == mat2.c
    )


def test_reshape():
    mat1 = ncnn.Mat()
    mat2 = mat1.reshape(1)
    assert mat2.dims == 0
    mat2 = mat1.reshape(1, 1)
    assert mat2.dims == 0
    mat2 = mat1.reshape(1, 1, 1)
    assert mat2.dims == 0
    mat2 = mat1.reshape(1, 1, 1, 1)
    assert mat2.dims == 0

    mat1 = ncnn.Mat(1)
    mat2 = mat1.reshape(1, 1)
    assert mat2.dims == 2 and mat2.w == 1 and mat2.h == 1
    mat2 = mat1.reshape(1, 1, 1)
    assert mat2.dims == 3 and mat2.w == 1 and mat2.h == 1 and mat2.c == 1
    mat2 = mat1.reshape(1, 1, 1, 1)
    assert mat2.dims == 4 and mat2.w == 1 and mat2.h == 1 and mat2.d == 1 and mat2.c == 1

    mat1 = ncnn.Mat(1, 2)
    mat2 = mat1.reshape(2)
    assert mat2.dims == 1 and mat2.w == 2
    mat2 = mat1.reshape(2, 1)
    assert mat2.dims == 2 and mat2.w == 2 and mat2.h == 1
    mat2 = mat1.reshape(2, 1, 1)
    assert mat2.dims == 3 and mat2.w == 2 and mat2.h == 1 and mat2.c == 1
    mat2 = mat1.reshape(2, 1, 1, 1)
    assert mat2.dims == 4 and mat2.w == 2 and mat2.h == 1 and mat2.d == 1 and mat2.c == 1

    mat1 = ncnn.Mat(1, 2, 3)
    mat2 = mat1.reshape(6)
    assert mat2.dims == 1 and mat2.w == 6
    mat2 = mat1.reshape(2, 3)
    assert mat2.dims == 2 and mat2.w == 2 and mat2.h == 3
    mat2 = mat1.reshape(2, 3, 1)
    assert mat2.dims == 3 and mat2.w == 2 and mat2.h == 3 and mat2.c == 1
    mat2 = mat1.reshape(2, 1, 3, 1)
    assert mat2.dims == 4 and mat2.w == 2 and mat2.h == 1 and mat2.d == 3 and mat2.c == 1

    mat1 = ncnn.Mat((1,))
    mat2 = mat1.reshape((1, 1))
    assert mat2.dims == 2 and mat2.w == 1 and mat2.h == 1
    mat2 = mat1.reshape((1, 1, 1))
    assert mat2.dims == 3 and mat2.w == 1 and mat2.h == 1 and mat2.c == 1
    mat2 = mat1.reshape((1, 1, 1, 1))
    assert mat2.dims == 4 and mat2.w == 1 and mat2.h == 1 and mat2.d == 1 and mat2.c == 1

    mat1 = ncnn.Mat((1, 2))
    mat2 = mat1.reshape((2,))
    assert mat2.dims == 1 and mat2.w == 2
    mat2 = mat1.reshape((2, 1))
    assert mat2.dims == 2 and mat2.w == 2 and mat2.h == 1
    mat2 = mat1.reshape((2, 1, 1))
    assert mat2.dims == 3 and mat2.w == 2 and mat2.h == 1 and mat2.c == 1
    mat2 = mat1.reshape((2, 1, 1, 1))
    assert mat2.dims == 4 and mat2.w == 2 and mat2.h == 1 and mat2.d == 1 and mat2.c == 1

    mat1 = ncnn.Mat((1, 2, 3))
    mat2 = mat1.reshape((6,))
    assert mat2.dims == 1 and mat2.w == 6
    mat2 = mat1.reshape((2, 3))
    assert mat2.dims == 2 and mat2.w == 2 and mat2.h == 3 and mat2.c == 1
    mat2 = mat1.reshape((2, 3, 1))
    assert mat2.dims == 3 and mat2.w == 2 and mat2.h == 3 and mat2.c == 1
    mat2 = mat1.reshape((2, 1, 3, 1))
    assert mat2.dims == 4 and mat2.w == 2 and mat2.h == 1 and mat2.d == 3 and mat2.c == 1

    with pytest.raises(RuntimeError) as execinfo:
        mat1.reshape((1, 1, 1, 1, 1))
    assert "shape must be 1, 2, 3 or 4 dims, not 5" in str(execinfo.value)


def test_create():
    mat = ncnn.Mat()
    mat.create(1)
    assert mat.dims == 1 and mat.w == 1
    mat.create(2, 3)
    assert mat.dims == 2 and mat.w == 2 and mat.h == 3
    mat.create(4, 5, 6)
    assert mat.dims == 3 and mat.w == 4 and mat.h == 5 and mat.c == 6
    mat.create(7, 8, 9, 10)
    assert mat.dims == 4 and mat.w == 7 and mat.h == 8 and mat.d == 9 and mat.c == 10

    mat.create((1,))
    assert mat.dims == 1 and mat.w == 1
    mat.create((2, 3))
    assert mat.dims == 2 and mat.w == 2 and mat.h == 3
    mat.create((4, 5, 6))
    assert mat.dims == 3 and mat.w == 4 and mat.h == 5 and mat.c == 6
    mat.create((7, 8, 9, 10))
    assert mat.dims == 4 and mat.w == 7 and mat.h == 8 and mat.d == 9 and mat.c == 10


def test_create_like():
    mat2 = ncnn.Mat()

    mat1 = ncnn.Mat(1)
    mat2.create_like(mat1)
    assert mat1.dims == mat2.dims and mat1.w == mat2.w
    mat1 = ncnn.Mat(2, 3)
    mat2.create_like(mat1)
    assert mat1.dims == mat2.dims and mat1.w == mat2.w and mat1.h == mat2.h
    mat1 = ncnn.Mat(4, 5, 6)
    mat2.create_like(mat1)
    assert (
        mat1.dims == mat2.dims
        and mat1.w == mat2.w
        and mat1.h == mat2.h
        and mat1.c == mat2.c
    )
    mat1 = ncnn.Mat(7, 8, 9, 10)
    mat2.create_like(mat1)
    assert (
        mat1.dims == mat2.dims
        and mat1.w == mat2.w
        and mat1.h == mat2.h
        and mat1.d == mat2.d
        and mat1.c == mat2.c
    )


def test_addref_release():
    mat = ncnn.Mat(1)
    assert mat.refcount == 1

    mat.addref()
    assert mat.refcount == 2

    mat.release()
    assert mat.refcount == None


def test_empty():
    mat = ncnn.Mat()
    assert mat.empty() == True

    mat = ncnn.Mat(1)
    assert mat.empty() == False


def test_total():
    mat = ncnn.Mat(1)
    assert mat.total() == 1
    mat = ncnn.Mat(2, 3)
    assert mat.total() == 2 * 3
    mat = ncnn.Mat(4, 5, 6)
    assert mat.total() == 4 * 5 * 6
    mat = ncnn.Mat(7, 8, 9, 10)
    assert mat.total() == 7 * 8 * 9 * 10


def test_elembits():
    mat = ncnn.Mat(1, elemsize=1, elempack=1)
    assert mat.elembits() == 8
    mat = ncnn.Mat(2, elemsize=2, elempack=1)
    assert mat.elembits() == 16
    mat = ncnn.Mat(3, elemsize=4, elempack=1)
    assert mat.elembits() == 32


def test_shape():
    mat = ncnn.Mat(1)
    shape = mat.shape()
    assert shape.dims == 1 and shape.w == 1
    mat = ncnn.Mat(2, 3)
    shape = mat.shape()
    assert shape.dims == 2 and shape.w == 2 and shape.h == 3
    mat = ncnn.Mat(4, 5, 6)
    shape = mat.shape()
    assert shape.dims == 3 and shape.w == 4 and shape.h == 5 and shape.c == 6
    mat = ncnn.Mat(7, 8, 9, 10)
    shape = mat.shape()
    assert shape.dims == 4 and shape.w == 7 and shape.h == 8 and shape.d == 9 and shape.c == 10


def test_channel_depth_row():
    mat = ncnn.Mat(2, 3, 4, 5)
    mat.fill(6.0)
    channel = mat.channel(1)
    assert channel.dims == 3 and channel.w == 2 and channel.h == 3 and channel.c == 4

    depth = channel.depth(1)
    assert depth.dims == 2 and depth.w == 2 and depth.h == 3

    row = depth.row(1)
    assert len(row) == 2 and np.abs(row[0] - 6.0) < sys.float_info.min


def test_channel_row():
    mat = ncnn.Mat(2, 3, 4)
    mat.fill(4.0)
    channel = mat.channel(1)
    assert channel.dims == 2 and channel.w == 2 and channel.h == 3 and channel.c == 1

    row = channel.row(1)
    assert len(row) == 2 and np.abs(row[0] - 4.0) < sys.float_info.min


def test_channel_range():
    mat = ncnn.Mat(1, 2, 3)
    channel_range = mat.channel_range(0, 2)
    assert (
        channel_range.dims == 3
        and channel_range.w == 1
        and channel_range.h == 2
        and channel_range.c == 2
    )


def test_depth_range():
    mat = ncnn.Mat(1, 2, 3, 4)
    depth_range = mat.channel(1).depth_range(1, 2)
    assert (
        depth_range.dims == 3
        and depth_range.w == 1
        and depth_range.h == 2
        and depth_range.c == 2
    )


def test_row_range():
    mat = ncnn.Mat(1, 2)
    row_range = mat.row_range(0, 2)
    assert row_range.dims == 2 and row_range.w == 1 and row_range.h == 2


def test_range():
    mat = ncnn.Mat(2)
    range = mat.range(0, 2)
    assert range.dims == 1 and range.w == 2


def test_getitem_setitem():
    mat = ncnn.Mat(2)
    mat.fill(1)
    assert (
        np.abs(mat[0] - 1.0) < sys.float_info.min
        and np.abs(mat[1] - 1.0) < sys.float_info.min
    )

    mat[0] = 2.0
    assert (
        np.abs(mat[0] - 2.0) < sys.float_info.min
        and np.abs(mat[1] - 1.0) < sys.float_info.min
    )


def test_from_pixels():
    pixels = np.random.randint(0, 256, size=(300, 400, 3)).astype(np.uint8)  # hwc
    mat = ncnn.Mat.from_pixels(pixels, ncnn.Mat.PixelType.PIXEL_RGB, 400, 300)  # chw
    assert mat.dims == 3 and mat.w == 400 and mat.h == 300 and mat.c == 3
    assert pixels[0, 0, 0] == mat.channel(0).row(0)[0]
    assert pixels[200, 150, 1] == mat.channel(1).row(200)[150]
    assert pixels[299, 399, 2] == mat.channel(2).row(299)[399]

    pixels = np.random.randint(0, 256, size=(300, 500, 3)).astype(np.uint8)  # hwc
    mat = ncnn.Mat.from_pixels(
        pixels, ncnn.Mat.PixelType.PIXEL_RGB, 400, 300, stride=500 * 3
    )  # chw
    assert mat.dims == 3 and mat.w == 400 and mat.h == 300 and mat.c == 3
    assert pixels[0, 0, 0] == mat.channel(0).row(0)[0]
    assert pixels[200, 150, 1] == mat.channel(1).row(200)[150]
    assert pixels[299, 399, 2] == mat.channel(2).row(299)[399]


def test_from_pixels_resize():
    pixels = np.random.randint(0, 256, size=(300, 400, 3)).astype(np.uint8)  # hwc
    mat = ncnn.Mat.from_pixels_resize(
        pixels, ncnn.Mat.PixelType.PIXEL_BGR2RGB, 400, 300, 200, 150
    )  # chw
    assert mat.dims == 3 and mat.w == 200 and mat.h == 150 and mat.c == 3

    pixels = np.random.randint(0, 256, size=(300, 400, 3)).astype(np.uint8)  # hwc
    mat = ncnn.Mat.from_pixels_resize(
        pixels, ncnn.Mat.PixelType.PIXEL_BGR2RGB, 400, 300, 400, 300
    )  # chw
    assert mat.dims == 3 and mat.w == 400 and mat.h == 300 and mat.c == 3
    assert pixels[0, 0, 0] == mat.channel(2).row(0)[0]
    assert pixels[200, 150, 1] == mat.channel(1).row(200)[150]
    assert pixels[299, 399, 2] == mat.channel(0).row(299)[399]

    pixels = np.random.randint(0, 256, size=(300, 500, 3)).astype(np.uint8)  # hwc
    mat = ncnn.Mat.from_pixels_resize(
        pixels, ncnn.Mat.PixelType.PIXEL_BGR2RGB, 400, 300, 500 * 3, 200, 150
    )  # chw
    assert mat.dims == 3 and mat.w == 200 and mat.h == 150 and mat.c == 3

    pixels = np.random.randint(0, 256, size=(300, 500, 3)).astype(np.uint8)  # hwc
    mat = ncnn.Mat.from_pixels_resize(
        pixels, ncnn.Mat.PixelType.PIXEL_BGR2RGB, 400, 300, 500 * 3, 400, 300
    )  # chw
    assert mat.dims == 3 and mat.w == 400 and mat.h == 300 and mat.c == 3
    assert pixels[0, 0, 0] == mat.channel(2).row(0)[0]
    assert pixels[200, 150, 1] == mat.channel(1).row(200)[150]
    assert pixels[299, 399, 2] == mat.channel(0).row(299)[399]


def test_from_pixels_roi():
    pixels = np.random.randint(0, 256, size=(300, 400, 3)).astype(np.uint8)  # hwc
    mat = ncnn.Mat.from_pixels_roi(
        pixels, ncnn.Mat.PixelType.PIXEL_RGB, 400, 300, 100, 75, 200, 150
    )  # chw
    assert mat.dims == 3 and mat.w == 200 and mat.h == 150 and mat.c == 3
    assert pixels[75, 100, 0] == mat.channel(0).row(0)[0]
    assert pixels[150, 200, 1] == mat.channel(1).row(75)[100]
    assert pixels[224, 299, 2] == mat.channel(2).row(149)[199]

    pixels = np.random.randint(0, 256, size=(300, 500, 3)).astype(np.uint8)  # hwc
    mat = ncnn.Mat.from_pixels_roi(
        pixels, ncnn.Mat.PixelType.PIXEL_RGB, 400, 300, 500 * 3, 100, 75, 200, 150
    )  # chw
    assert mat.dims == 3 and mat.w == 200 and mat.h == 150 and mat.c == 3
    assert pixels[75, 100, 0] == mat.channel(0).row(0)[0]
    assert pixels[150, 200, 1] == mat.channel(1).row(75)[100]
    assert pixels[224, 299, 2] == mat.channel(2).row(149)[199]


def test_from_pixels_roi_resize():
    pixels = np.random.randint(0, 256, size=(300, 400, 3)).astype(np.uint8)  # hwc
    mat = ncnn.Mat.from_pixels_roi_resize(
        pixels, ncnn.Mat.PixelType.PIXEL_RGB, 400, 300, 100, 75, 200, 150, 100, 75
    )  # chw
    assert mat.dims == 3 and mat.w == 100 and mat.h == 75 and mat.c == 3

    pixels = np.random.randint(0, 256, size=(300, 500, 3)).astype(np.uint8)  # hwc
    mat = ncnn.Mat.from_pixels_roi_resize(
        pixels,
        ncnn.Mat.PixelType.PIXEL_RGB,
        400,
        300,
        500 * 3,
        100,
        75,
        200,
        150,
        100,
        75,
    )  # chw
    assert mat.dims == 3 and mat.w == 100 and mat.h == 75 and mat.c == 3


def test_substract_mean_normalize():
    pixels = np.random.randint(0, 256, size=(300, 400, 3)).astype(np.uint8)  # hwc
    mean_vals = [127.5, 127.5, 127.5]
    norm_vals = [0.007843, 0.007843, 0.007843]

    mat = ncnn.Mat.from_pixels(pixels, ncnn.Mat.PixelType.PIXEL_RGB, 400, 300)  # chw
    mat.substract_mean_normalize([], norm_vals)
    assert np.abs(pixels[0, 0, 0] * 0.007843 - mat.channel(0).row(0)[0]) < 1e-5

    mat = ncnn.Mat.from_pixels(pixels, ncnn.Mat.PixelType.PIXEL_RGB, 400, 300)  # chw
    mat.substract_mean_normalize(mean_vals, [])
    assert np.abs((pixels[0, 0, 0] - 127.5) - mat.channel(0).row(0)[0]) < 1e-5

    mat = ncnn.Mat.from_pixels(pixels, ncnn.Mat.PixelType.PIXEL_RGB, 400, 300)  # chw
    mat.substract_mean_normalize(mean_vals, norm_vals)
    assert (
        np.abs((pixels[0, 0, 0] - 127.5) * 0.007843 - mat.channel(0).row(0)[0]) < 1e-5
    )
