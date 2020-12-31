import numpy as np
import pytest

import ncnn

def test_mat_dims1():
    mat = ncnn.Mat(1)
    assert mat.dims == 1 and mat.w == 1
    mat = ncnn.Mat(2, elemsize=4)
    assert mat.dims == 1 and mat.w == 2
    mat = ncnn.Mat(3, elemsize=4, elempack=1)
    assert mat.dims == 1 and mat.w == 3
    mat = ncnn.Mat(4, elemsize=4, elempack=1, allocator=None)
    assert mat.dims == 1 and mat.w == 4

    mat = ncnn.Mat((1,))
    assert mat.dims == 1 and mat.w == 1
    mat = ncnn.Mat((2,), elemsize=4)
    assert mat.dims == 1 and mat.w == 2
    mat = ncnn.Mat((3,), elemsize=4, elempack=1)
    assert mat.dims == 1 and mat.w == 3
    mat = ncnn.Mat((4,), elemsize=4, elempack=1, allocator=None)
    assert mat.dims == 1 and mat.w == 4


def test_mat_dims2():
    mat = ncnn.Mat(1, 2)
    assert mat.dims == 2 and mat.w == 1 and mat.h == 2
    mat = ncnn.Mat(3, 4, elemsize=4)
    assert mat.dims == 2 and mat.w == 3 and mat.h == 4
    mat = ncnn.Mat(5, 6, elemsize=4, elempack=1)
    assert mat.dims == 2 and mat.w == 5 and mat.h == 6
    mat = ncnn.Mat(7, 8, elemsize=4, elempack=1, allocator=None)
    assert mat.dims == 2 and mat.w == 7 and mat.h == 8

    mat = ncnn.Mat((1, 2))
    assert mat.dims == 2
    mat = ncnn.Mat((3, 4), elemsize=4)
    assert mat.dims == 2
    mat = ncnn.Mat((5, 6), elemsize=4, elempack=1)
    assert mat.dims == 2
    mat = ncnn.Mat((7, 8), elemsize=4, elempack=1, allocator=None)
    assert mat.dims == 2


def test_mat_dims3():
    mat = ncnn.Mat(1, 2, 3)
    assert mat.dims == 3
    mat = ncnn.Mat(4, 5, 6, elemsize=4)
    assert mat.dims == 3
    mat = ncnn.Mat(7, 8, 9, elemsize=4, elempack=1)
    assert mat.dims == 3
    mat = ncnn.Mat(10, 11, 12, elemsize=4, elempack=1, allocator=None)
    assert mat.dims == 3
    
    mat = ncnn.Mat((1, 2, 3))
    assert mat.dims == 3
    mat = ncnn.Mat((4, 5, 6), elemsize=4)
    assert mat.dims == 3
    mat = ncnn.Mat((7, 8, 9), elemsize=4, elempack=1)
    assert mat.dims == 3
    mat = ncnn.Mat((10, 11, 12), elemsize=4, elempack=1, allocator=None)
    assert mat.dims == 3


def test_mat2np():
    mat = ncnn.Mat(1)
    array = np.array(mat)
    assert mat.dims == array.ndim and mat.w == array.shape[0]
    mat = ncnn.Mat(1, 2)
    array = np.array(mat)
    assert mat.dims == array.ndim and mat.w == array.shape[1] and mat.h == array.shape[0]
    mat = ncnn.Mat(1, 2, 3)
    array = np.array(mat)
    assert mat.dims == array.ndim and mat.w == array.shape[2] and mat.h == array.shape[1] and mat.c == array.shape[0]

    mat = ncnn.Mat(1, elemsize=1)
    array = np.array(mat)
    assert array.dtype == np.int8
    mat = ncnn.Mat(1, elemsize=2)
    array = np.array(mat)
    assert array.dtype == np.float16
    mat = ncnn.Mat(1, elemsize=4)
    array = np.array(mat)
    assert array.dtype == np.float32
