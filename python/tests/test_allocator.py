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
