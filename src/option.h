// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef NCNN_OPTION_H
#define NCNN_OPTION_H

#include "platform.h"

namespace ncnn {

#if NCNN_VULKAN
class VkAllocator;
class PipelineCache;
#endif // NCNN_VULKAN

class Allocator;
class NCNN_EXPORT Option
{
public:
    // default option
    Option();

public:
    // light mode
    // intermediate blob will be recycled when enabled
    // enabled by default
    bool lightmode;

    // thread count
    // default value is the one returned by get_cpu_count()
    int num_threads;

    // blob memory allocator
    Allocator* blob_allocator;

    // workspace memory allocator
    Allocator* workspace_allocator;

#if NCNN_VULKAN
    // blob memory allocator
    VkAllocator* blob_vkallocator;

    // workspace memory allocator
    VkAllocator* workspace_vkallocator;

    // staging memory allocator
    VkAllocator* staging_vkallocator;

    // pipeline cache
    PipelineCache* pipeline_cache;
#endif // NCNN_VULKAN

    // the time openmp threads busy-wait for more work before going to sleep
    // default value is 20ms to keep the cores enabled
    // without too much extra power consumption afterwards
    int openmp_blocktime;

    // enable winograd convolution optimization
    // improve convolution 3x3 stride1 performance, may consume more memory
    // changes should be applied before loading network structure and weight
    // enabled by default
    bool use_winograd_convolution;

    // enable sgemm convolution optimization
    // improve convolution 1x1 stride1 performance, may consume more memory
    // changes should be applied before loading network structure and weight
    // enabled by default
    bool use_sgemm_convolution;

    // enable quantized int8 inference
    // use low-precision int8 path for quantized model
    // changes should be applied before loading network structure and weight
    // enabled by default
    bool use_int8_inference;

    // enable vulkan compute
    bool use_vulkan_compute;

    // enable bf16 data type for storage
    // improve most operator performance on all arm devices, may consume more memory
    bool use_bf16_storage;

    // enable options for gpu inference
    bool use_fp16_packed;
    bool use_fp16_storage;
    bool use_fp16_arithmetic;
    bool use_int8_packed;
    bool use_int8_storage;
    bool use_int8_arithmetic;

    // enable simd-friendly packed memory layout
    // improve all operator performance on all arm devices, will consume more memory
    // changes should be applied before loading network structure and weight
    // enabled by default
    bool use_packing_layout;

    bool use_shader_pack8;

    // subgroup option
    bool use_subgroup_basic;
    bool use_subgroup_vote;
    bool use_subgroup_ballot;
    bool use_subgroup_shuffle;

    // turn on for adreno
    bool use_image_storage;
    bool use_tensor_storage;

    // used for fp16 weight storage in AVX
    // TODO drop this option
    bool use_weight_fp16_storage;

    // enable DAZ(Denormals-Are-Zero) and FTZ(Flush-To-Zero)
    // default value is 3
    // 0 = DAZ OFF, FTZ OFF
    // 1 = DAZ ON , FTZ OFF
    // 2 = DAZ OFF, FTZ ON
    // 3 = DAZ ON,  FTZ ON
    int flush_denormals;

    bool use_local_pool_allocator;

    bool use_reserved_1;
    bool use_reserved_2;
    bool use_reserved_3;
    bool use_reserved_4;
    bool use_reserved_5;
    bool use_reserved_6;
    bool use_reserved_7;
    bool use_reserved_8;
    bool use_reserved_9;
    bool use_reserved_10;
    bool use_reserved_11;
};

} // namespace ncnn

#endif // NCNN_OPTION_H
