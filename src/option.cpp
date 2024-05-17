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

#include "option.h"

#include "cpu.h"

namespace ncnn {

Option::Option()
{
    lightmode = true;
    num_threads = get_physical_big_cpu_count();
    blob_allocator = 0;
    workspace_allocator = 0;

#if NCNN_VULKAN
    blob_vkallocator = 0;
    workspace_vkallocator = 0;
    staging_vkallocator = 0;
    pipeline_cache = 0;
#endif // NCNN_VULKAN

    openmp_blocktime = 20;

    use_winograd_convolution = true;
    use_sgemm_convolution = true;
    use_int8_inference = true;
    use_vulkan_compute = false; // TODO enable me

    use_bf16_storage = false;

    use_fp16_packed = true;
    use_fp16_storage = true;
    use_fp16_arithmetic = true;
    use_int8_packed = true;
    use_int8_storage = true;
    use_int8_arithmetic = false;

    use_packing_layout = true;

    use_shader_pack8 = false;

    use_subgroup_basic = false;
    use_subgroup_vote = false;
    use_subgroup_ballot = false;
    use_subgroup_shuffle = false;

    use_image_storage = false;
    use_tensor_storage = false;

    use_reserved_0 = false;

    flush_denormals = 3;

    use_local_pool_allocator = true;

    use_shader_local_memory = true;
    use_cooperative_matrix = true;

    use_winograd23_convolution = true;
    use_winograd43_convolution = true;
    use_winograd63_convolution = true;

    use_a53_a55_optimized_kernel = is_current_thread_running_on_a53_a55();

    use_fp16_uniform = true;
    use_int8_uniform = true;
}

} // namespace ncnn
