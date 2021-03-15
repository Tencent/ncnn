# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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


def test_vk_blob_allocator():
    if not hasattr(ncnn, "get_gpu_count"):
        return

    vkdev = ncnn.get_gpu_device(0)
    assert vkdev is not None
    allocator = ncnn.VkBlobAllocator(vkdev)
    assert allocator.buffer_memory_type_index >= 0
    assert allocator.image_memory_type_index >= 0

    mappable = allocator.mappable
    allocator.mappable = not mappable
    assert allocator.mappable == (not mappable)

    coherent = allocator.coherent
    allocator.coherent = not coherent
    assert allocator.coherent == (not coherent)

    bufmem = allocator.fastMalloc(10 * 1024)
    assert bufmem is not None
    allocator.fastFree(bufmem)

    imgmem = allocator.fastMalloc(4, 4, 3, 4, 1)
    assert imgmem is not None
    allocator.fastFree(imgmem)


def test_vk_weight_allocator():
    if not hasattr(ncnn, "get_gpu_count"):
        return

    vkdev = ncnn.get_gpu_device(0)
    assert vkdev is not None
    allocator = ncnn.VkWeightAllocator(vkdev)
    assert allocator.buffer_memory_type_index >= 0
    assert allocator.image_memory_type_index >= 0

    mappable = allocator.mappable
    allocator.mappable = not mappable
    assert allocator.mappable == (not mappable)

    coherent = allocator.coherent
    allocator.coherent = not coherent
    assert allocator.coherent == (not coherent)

    bufmem = allocator.fastMalloc(10 * 1024)
    assert bufmem is not None
    allocator.fastFree(bufmem)

    imgmem = allocator.fastMalloc(4, 4, 3, 4, 1)
    assert imgmem is not None
    allocator.fastFree(imgmem)


def test_vk_staging_allocator():
    if not hasattr(ncnn, "get_gpu_count"):
        return

    vkdev = ncnn.get_gpu_device(0)
    assert vkdev is not None
    allocator = ncnn.VkStagingAllocator(vkdev)
    assert allocator.buffer_memory_type_index >= 0
    assert allocator.image_memory_type_index >= 0

    mappable = allocator.mappable
    allocator.mappable = not mappable
    assert allocator.mappable == (not mappable)

    coherent = allocator.coherent
    allocator.coherent = not coherent
    assert allocator.coherent == (not coherent)

    bufmem = allocator.fastMalloc(10 * 1024)
    assert bufmem is not None
    allocator.fastFree(bufmem)

    imgmem = allocator.fastMalloc(4, 4, 3, 4, 1)
    assert imgmem is not None
    allocator.fastFree(imgmem)


def test_vk_weight_staging_allocator():
    if not hasattr(ncnn, "get_gpu_count"):
        return

    vkdev = ncnn.get_gpu_device(0)
    assert vkdev is not None
    allocator = ncnn.VkWeightStagingAllocator(vkdev)
    assert allocator.buffer_memory_type_index >= 0
    assert allocator.image_memory_type_index >= 0

    mappable = allocator.mappable
    allocator.mappable = not mappable
    assert allocator.mappable == (not mappable)

    coherent = allocator.coherent
    allocator.coherent = not coherent
    assert allocator.coherent == (not coherent)

    bufmem = allocator.fastMalloc(10 * 1024)
    assert bufmem is not None
    allocator.fastFree(bufmem)

    imgmem = allocator.fastMalloc(4, 4, 3, 4, 1)
    assert imgmem is None
