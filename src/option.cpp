// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "option.h"

#include "cpu.h"

namespace ncnn {

Option::Option()
{
    lightmode = true;
    use_reserved_m0 = false;
    use_subgroup_ops = false;
    use_reserved_0 = false;

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

    vulkan_device_index = -1;
    use_reserved_1 = false;

    use_tensor_storage = false;
    use_reserved_1p = false;

    use_reserved_2 = false;

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

    use_reserved_9 = false;
    use_reserved_10 = false;
    use_reserved_11 = false;
}

} // namespace ncnn
