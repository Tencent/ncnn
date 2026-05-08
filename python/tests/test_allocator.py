# Copyright 2020 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import ncnn


def test_pool_allocator():
    pa = ncnn.PoolAllocator()
    assert pa is not None
    pa.set_size_compare_ratio(0.5)
    buf = pa.fastMalloc(10 * 1024)
    assert buf is not None
    pa.fastFree(buf)
    pa.clear()


def test_unlocked_pool_allocator():
    upa = ncnn.UnlockedPoolAllocator()
    assert upa is not None
    upa.set_size_compare_ratio(0.5)
    buf = upa.fastMalloc(10 * 1024)
    assert buf is not None
    upa.fastFree(buf)
    upa.clear()
