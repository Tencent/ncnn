# Copyright 2020 Tencent
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import ncnn


def test_option():
    allocator = ncnn.PoolAllocator()

    opt = ncnn.Option()

    opt.lightmode = True
    assert opt.lightmode == True
    opt.lightmode = False
    assert opt.lightmode == False

    assert opt.num_threads == ncnn.get_physical_big_cpu_count()
    opt.num_threads = 1
    assert opt.num_threads == 1

    assert opt.blob_allocator is None
    opt.blob_allocator = allocator
    assert opt.blob_allocator == allocator

    assert opt.workspace_allocator is None
    opt.workspace_allocator = allocator
    assert opt.workspace_allocator == allocator

    assert opt.openmp_blocktime == 20
    opt.openmp_blocktime = 40
    assert opt.openmp_blocktime == 40

    opt.use_winograd_convolution = True
    assert opt.use_winograd_convolution == True
    opt.use_winograd_convolution = False
    assert opt.use_winograd_convolution == False

    opt.use_sgemm_convolution = True
    assert opt.use_sgemm_convolution == True
    opt.use_sgemm_convolution = False
    assert opt.use_sgemm_convolution == False

    opt.use_int8_inference = True
    assert opt.use_int8_inference == True
    opt.use_int8_inference = False
    assert opt.use_int8_inference == False

    opt.use_vulkan_compute = True
    assert opt.use_vulkan_compute == True
    opt.use_vulkan_compute = False
    assert opt.use_vulkan_compute == False

    opt.use_bf16_storage = True
    assert opt.use_bf16_storage == True
    opt.use_bf16_storage = False
    assert opt.use_bf16_storage == False

    opt.use_fp16_packed = True
    assert opt.use_fp16_packed == True
    opt.use_fp16_packed = False
    assert opt.use_fp16_packed == False

    opt.use_fp16_storage = True
    assert opt.use_fp16_storage == True
    opt.use_fp16_storage = False
    assert opt.use_fp16_storage == False

    opt.use_fp16_arithmetic = True
    assert opt.use_fp16_arithmetic == True
    opt.use_fp16_arithmetic = False
    assert opt.use_fp16_arithmetic == False

    opt.use_int8_packed = True
    assert opt.use_int8_packed == True
    opt.use_int8_packed = False
    assert opt.use_int8_packed == False

    opt.use_int8_storage = True
    assert opt.use_int8_storage == True
    opt.use_int8_storage = False
    assert opt.use_int8_storage == False

    opt.use_int8_arithmetic = True
    assert opt.use_int8_arithmetic == True
    opt.use_int8_arithmetic = False
    assert opt.use_int8_arithmetic == False

    opt.use_packing_layout = True
    assert opt.use_packing_layout == True
    opt.use_packing_layout = False
    assert opt.use_packing_layout == False

    opt.use_subgroup_ops = True
    assert opt.use_subgroup_ops == True
    opt.use_subgroup_ops = False
    assert opt.use_subgroup_ops == False

    opt.use_tensor_storage = True
    assert opt.use_tensor_storage == True
    opt.use_tensor_storage = False
    assert opt.use_tensor_storage == False
