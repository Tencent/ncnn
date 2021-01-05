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


def test_mat2np():
    mat = ncnn.Mat(1)
    array = np.array(mat)
    assert mat.dims == array.ndim and mat.w == array.shape[0]
    mat = ncnn.Mat(1, 2)
    array = np.array(mat)
    assert (
        mat.dims == array.ndim and mat.w == array.shape[1] and mat.h == array.shape[0]
    )
    mat = ncnn.Mat(1, 2, 3)
    array = np.array(mat)
    assert (
        mat.dims == array.ndim
        and mat.w == array.shape[2]
        and mat.h == array.shape[1]
        and mat.c == array.shape[0]
    )

    mat = ncnn.Mat(1, elemsize=1)
    array = np.array(mat)
    assert array.dtype == np.int8
    mat = ncnn.Mat(1, elemsize=2)
    array = np.array(mat)
    assert array.dtype == np.float16
    mat = ncnn.Mat(1, elemsize=4)
    array = np.array(mat)
    assert array.dtype == np.float32


def test_create():
    mat = ncnn.Mat()
    mat.create(1)
    assert mat.dims == 1 and mat.w == 1
    mat.create(2, 3)
    assert mat.dims == 2 and mat.w == 2 and mat.h == 3
    mat.create(4, 5, 6)
    assert mat.dims == 3 and mat.w == 4 and mat.h == 5 and mat.c == 6
    mat.create(7, 8, 9, elemsize=4)
    assert (
        mat.dims == 3 and mat.w == 7 and mat.h == 8 and mat.c == 9 and mat.elemsize == 4
    )
    mat = ncnn.Mat((10, 11, 12), elemsize=4, elempack=1)
    assert (
        mat.dims == 3
        and mat.w == 10
        and mat.h == 11
        and mat.c == 12
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
