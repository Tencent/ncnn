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


def test_option():
    allocator = ncnn.PoolAllocator()

    opt = ncnn.Option()

    opt.lightmode = True
    assert opt.lightmode == True
    opt.lightmode = False
    assert opt.lightmode == False

    assert opt.num_threads == ncnn.get_cpu_count()
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

    opt.use_shader_pack8 = True
    assert opt.use_shader_pack8 == True
    opt.use_shader_pack8 = False
    assert opt.use_shader_pack8 == False

    opt.use_subgroup_basic = True
    assert opt.use_subgroup_basic == True
    opt.use_subgroup_basic = False
    assert opt.use_subgroup_basic == False

    opt.use_subgroup_vote = True
    assert opt.use_subgroup_vote == True
    opt.use_subgroup_vote = False
    assert opt.use_subgroup_vote == False

    opt.use_subgroup_ballot = True
    assert opt.use_subgroup_ballot == True
    opt.use_subgroup_ballot = False
    assert opt.use_subgroup_ballot == False

    opt.use_subgroup_shuffle = True
    assert opt.use_subgroup_shuffle == True
    opt.use_subgroup_shuffle = False
    assert opt.use_subgroup_shuffle == False

    opt.use_image_storage = True
    assert opt.use_image_storage == True
    opt.use_image_storage = False
    assert opt.use_image_storage == False

    opt.use_tensor_storage = True
    assert opt.use_tensor_storage == True
    opt.use_tensor_storage = False
    assert opt.use_tensor_storage == False
