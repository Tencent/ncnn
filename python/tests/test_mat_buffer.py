# Copyright 2020 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import gc
import weakref
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


@pytest.mark.parametrize(
    "shape,make_view,expected",
    [
        ((3, 4, 5), lambda m: m.channel(1), lambda a: a[1]),
        ((3, 4, 5), lambda m: m.channel_range(1, 2), lambda a: a[1:3]),
        ((1, 3, 4, 5), lambda m: m.channel(0).depth(1), lambda a: a[0, 1]),
        ((1, 3, 4, 5), lambda m: m.channel(0).depth_range(1, 2), lambda a: a[0, 1:3]),
        ((4, 5), lambda m: m.row_range(1, 2), lambda a: a[1:3]),
        ((10,), lambda m: m.range(2, 5), lambda a: a[2:7]),
        ((3, 4, 5), lambda m: m.batch(0), lambda a: a),
    ],
    ids=["channel", "channel_range", "depth", "depth_range", "row_range", "range", "batch"],
)
def test_buffer_subview_retains_owner(shape, make_view, expected):
    array = np.arange(np.prod(shape), dtype=np.float32).reshape(shape).copy()
    owner = weakref.ref(array)
    values = expected(array).copy().ravel()
    parent = ncnn.Mat(array)
    view = make_view(parent)
    shallow = ncnn.Mat(view)
    del array, parent, view
    gc.collect()

    assert owner() is not None
    assert shallow.allocator is None
    np.testing.assert_array_equal(np.array(shallow).ravel(), values)
    del shallow
    gc.collect()
    assert owner() is None


def test_buffer_view_chain_survives_parent_release():
    array = np.arange(60, dtype=np.float32).reshape(3, 4, 5).copy()
    owner = weakref.ref(array)
    values = array[1, 1:3].copy()
    parent = ncnn.Mat(array)
    view = parent.channel(1).row_range(1, 2)
    parent.release()
    del array, parent
    gc.collect()
    assert owner() is not None
    np.testing.assert_array_equal(np.array(view), values)
    view.release()
    gc.collect()
    assert owner() is None


@pytest.mark.parametrize("subview", [False, True])
def test_buffer_allocator_cannot_be_replaced(subview):
    array = np.arange(20, dtype=np.float32).reshape(4, 5).copy()
    owner = weakref.ref(array)
    mat = ncnn.Mat(array)
    if subview:
        mat = mat.row_range(1, 2)
    for allocator in (mat.allocator, ncnn.PoolAllocator()):
        with pytest.raises(ValueError, match="cannot replace the allocator"):
            mat.allocator = allocator
    del array
    gc.collect()
    assert owner() is not None
    mat.release()
    gc.collect()
    assert owner() is None


def test_contiguous_buffer_is_zero_copy_and_released():
    array = np.arange(20, dtype=np.float32).reshape(4, 5).copy()
    owner = weakref.ref(array)
    mat = ncnn.Mat(array)
    assert np.shares_memory(np.asarray(mat), array)
    array[1, 2] = 42
    assert np.asarray(mat)[1, 2] == 42
    mat[0] = 99
    assert array[0, 0] == 99
    del array
    gc.collect()
    assert owner() is not None
    mat.release()
    gc.collect()
    assert owner() is None


def test_strided_buffer_is_independent_copy():
    array = np.arange(20, dtype=np.float32).reshape(4, 5)
    view = array[:, ::2]
    expected = view.copy()
    mat = ncnn.Mat(view)
    assert not np.shares_memory(np.asarray(mat), array)
    array.fill(-1)
    np.testing.assert_array_equal(np.array(mat), expected)


@pytest.mark.parametrize("subview", [False, True])
def test_extractor_retains_buffer_after_python_objects_die(subview):
    net = ncnn.Net()
    net.opt.use_packing_layout = False
    assert net.load_param_mem("""7767517
2 2
Input input 0 1 data
ReLU relu 1 1 data output
""") == 0
    assert net.load_model(ncnn.DataReaderFromEmpty()) == 0
    array = np.arange(20, dtype=np.float32).reshape(4, 5).copy()
    owner = weakref.ref(array)
    values = array[1:3].copy() if subview else array.copy()
    mat = ncnn.Mat(array)
    if subview:
        mat = mat.row_range(1, 2)
    ex = net.create_extractor()
    assert ex.input("data", mat) == 0
    del array, mat
    gc.collect()
    assert owner() is not None
    ret, output = ex.extract("output")
    assert ret == 0
    np.testing.assert_array_equal(np.array(output), values)
    ex.clear()
    del output
    gc.collect()
    assert owner() is None

@pytest.mark.parametrize("subview", [False, True])
def test_release_clears_allocator(subview):
    array = np.arange(20, dtype=np.float32).reshape(4, 5).copy()
    mat = ncnn.Mat(array)
    if subview:
        mat = mat.row_range(1, 2)
    assert mat.allocator is None
    mat.release()
    # the buffer owner deletes itself in fastFree(), so a released Mat must clear
    # the stored allocator instead of leaving a dangling pointer behind
    assert mat.allocator is None
    assert mat.empty()


def test_allocator_assignment_after_release_is_safe():
    mat = ncnn.Mat(np.arange(6, dtype=np.float32).reshape(2, 3).copy())
    mat.release()
    mat.allocator = None
    assert mat.allocator is None


def test_parent_release_keeps_subview_alive():
    array = np.arange(20, dtype=np.float32).reshape(4, 5).copy()
    owner = weakref.ref(array)
    parent = ncnn.Mat(array)
    view = parent.row_range(1, 2)
    del array
    gc.collect()
    assert owner() is not None

    parent.release()
    gc.collect()
    assert owner() is not None
    np.testing.assert_array_equal(np.array(view), np.arange(20, dtype=np.float32).reshape(4, 5)[1:3])

    view.release()
    gc.collect()
    assert owner() is None

@pytest.mark.parametrize("subview", [False, True])
def test_refcount_is_read_only(subview):
    mat = ncnn.Mat(np.arange(20, dtype=np.float32).reshape(4, 5).copy())
    if subview:
        mat = mat.row_range(1, 2)
    assert mat.refcount == 1

    # the refcount of a buffer-backed Mat lives inside the allocator that owns
    # the numpy buffer; writing it would rebind the pointer to a temporary
    with pytest.raises(AttributeError):
        mat.refcount = mat.refcount

    mat.addref()
    assert mat.refcount == 2
    mat.release()
    assert mat.refcount is None
    assert mat.allocator is None


def test_native_refcount_is_read_only():
    mat = ncnn.Mat(1)
    assert mat.refcount == 1
    with pytest.raises(AttributeError):
        mat.refcount = 1
    mat.addref()
    assert mat.refcount == 2
    mat.release()
    assert mat.refcount is None
