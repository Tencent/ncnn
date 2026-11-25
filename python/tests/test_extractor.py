# Copyright 2021 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import threading
import time

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


def test_extractor_releases_gil():
    # Another Python thread must keep running while extract is busy.
    dr = ncnn.DataReaderFromEmpty()

    net = ncnn.Net()
    # One thread keeps the call long enough to measure.
    net.opt.num_threads = 1
    net.load_param("tests/test.param")
    net.load_model(dr)

    in_mat = ncnn.Mat((1024, 1024, 3))

    def measure():
        timing = {}
        go = threading.Event()

        def worker():
            with net.create_extractor() as ex:
                ex.input("data", in_mat)
                go.wait()
                start = time.perf_counter()
                timing["ret"], _ = ex.extract("conv0_fwd")
                timing["duration"] = time.perf_counter() - start

        thread = threading.Thread(target=worker)
        thread.start()
        max_gap = 0.0
        last = time.perf_counter()
        go.set()
        while thread.is_alive():
            now = time.perf_counter()
            max_gap = max(max_gap, now - last)
            last = now
        # Count a stall that ended the call before this loop could run.
        max_gap = max(max_gap, time.perf_counter() - last)
        thread.join()
        assert timing["ret"] == 0
        return max_gap, timing["duration"]

    # Holding the GIL stalls this loop for the whole call, so retries
    # only absorb scheduler hiccups on busy CI runners.
    for _ in range(3):
        max_gap, duration = measure()
        if max_gap < duration / 2:
            break
    assert max_gap < duration / 2
