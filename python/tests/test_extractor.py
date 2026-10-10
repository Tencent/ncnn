# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import ncnn

alloctor = ncnn.PoolAllocator()


def test_extractor():
    with pytest.raises(TypeError, match="No constructor"):
        ex = ncnn.Extractor()

    dr = ncnn.DataReaderFromEmpty()

    net = ncnn.Net()
    net.load_param("tests/test.param")
    net.load_model(dr)

    in_mat = ncnn.Mat((227, 227, 3))
    with net.create_extractor() as ex:
        ex.set_light_mode(True)

        ex.set_blob_allocator(alloctor)
        ex.set_workspace_allocator(alloctor)

        ex.input("data", in_mat)
        ret, out_mat = ex.extract("conv0_fwd")
        assert (
            ret == 0
            and out_mat.dims == 3
            and out_mat.w == 225
            and out_mat.h == 225
            and out_mat.c == 3
        )
        assert out_mat.allocator is None

        ret, out_mat_raw = ex.extract("conv0_fwd", type=1)
        assert ret == 0 and out_mat_raw.allocator is alloctor

        ret, out_mat = ex.extract("output")
        assert ret == 0 and out_mat.dims == 1 and out_mat.w == 1

    out_mat_raw_copy = out_mat_raw.clone()
    assert out_mat_raw_copy.dims == 3 and out_mat_raw_copy.w == 225


def test_extractor_index():
    with pytest.raises(TypeError, match="No constructor"):
        ex = ncnn.Extractor()

    dr = ncnn.DataReaderFromEmpty()

    net = ncnn.Net()
    net.load_param("tests/test.param")
    net.load_model(dr)

    in_mat = ncnn.Mat((227, 227, 3))
    ex = net.create_extractor()
    ex.set_light_mode(True)

    ex.set_blob_allocator(alloctor)
    ex.set_workspace_allocator(alloctor)

    ex.input(0, in_mat)
    ret, out_mat = ex.extract(1)
    assert (
        ret == 0
        and out_mat.dims == 3
        and out_mat.w == 225
        and out_mat.h == 225
        and out_mat.c == 3
    )
    assert out_mat.allocator is None

    ret, out_mat_raw = ex.extract(1, type=1)
    assert ret == 0 and out_mat_raw.allocator is alloctor

    ret, out_mat = ex.extract(2)
    assert ret == 0 and out_mat.dims == 1 and out_mat.w == 1

    # not use with sentence, call clear manually to ensure ex destruct before net
    ex.clear()

    out_mat_raw_copy = out_mat_raw.clone()
    assert out_mat_raw_copy.dims == 3 and out_mat_raw_copy.w == 225
