// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gpu.h"
#include "testutil.h"

#include <stdio.h>
#include <string.h>

static int test_compile_spirv_module(const char* source, int expected, int comp_data_size = -1)
{
    const int source_size = strlen(source);
    char* comp_data = new char[source_size];
    memcpy(comp_data, source, source_size);

    if (comp_data_size == -1)
        comp_data_size = source_size;

    ncnn::Option opt;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;

    std::vector<uint32_t> spirv;
    int ret = ncnn::compile_spirv_module(comp_data, comp_data_size, opt, spirv);
    delete[] comp_data;

    if (ret != expected)
    {
        fprintf(stderr, "test_compile_spirv_module failed ret=%d expected=%d comp_data_size=%d source=%s\n", ret, expected, comp_data_size, source);
        return -1;
    }

    return 0;
}

static int test_compile_spirv_module_0()
{
    return 0
           || test_compile_spirv_module("#version 450\nlayout(local_size_x = 1) in;\nvoid main() {}", 0)
           || test_compile_spirv_module("#version 450\nlayout(local_size_x = 1) in;\nvoid main() {}\n", 0)
           || test_compile_spirv_module("#version 450\r\nlayout(local_size_x = 1) in;\r\nvoid main() {}", 0)
           || test_compile_spirv_module("#version 450\rlayout(local_size_x = 1) in;\rvoid main() {}", 0)
           || test_compile_spirv_module("// leading comment\n#version\t450\n \t\r\nlayout(local_size_x = 1) in;\nvoid main() {}", 0)
           || test_compile_spirv_module("#version 450\n", 0)
           || test_compile_spirv_module("#version 450\n \t\r\n", 0)
           || test_compile_spirv_module("#version 450\n   #error outside source", 0, 13);
}

static int test_compile_spirv_module_1()
{
    return 0
           || test_compile_spirv_module("", -1)
           || test_compile_spirv_module("#version", -1)
           || test_compile_spirv_module("#version \t", -1)
           || test_compile_spirv_module("#version +", -1)
           || test_compile_spirv_module("#version nope\n", -1)
           || test_compile_spirv_module("void main() {}", -1)
           || test_compile_spirv_module("x#version 450\n", -1)
           || test_compile_spirv_module("#version 450\n#error test", -1)
           || test_compile_spirv_module("#version 450\n#error test\n", -1);
}

static int test_compile_spirv_module_subgroup_option()
{
    for (int device_index = 0; device_index < ncnn::get_gpu_count(); device_index++)
    {
        const uint32_t subgroup_ops = ncnn::get_gpu_info(device_index).support_subgroup_ops();
        for (int i = 0; i < 4; i++)
        {
            ncnn::Option opt;
            opt.vulkan_device_index = device_index;
            opt.use_fp16_packed = false;
            opt.use_fp16_storage = false;
            opt.use_fp16_arithmetic = false;
            opt.use_subgroup_ops = (i & 1) != 0;
            opt.use_cooperative_matrix = (i & 2) != 0;

            // The option must not change the independent hardware capability macros.
            char source[512];
            snprintf(source, sizeof(source),
                     "#version 450\n"
                     "#if NCNN_subgroup_ops != %d\n#error subgroup option mismatch\n#endif\n"
                     "#if ncnn_subgroup_basic != %d || ncnn_subgroup_arithmetic != %d\n#error subgroup capability mismatch\n#endif\n"
                     "layout(local_size_x = 1) in;\nvoid main() {}\n",
                     opt.use_subgroup_ops ? 1 : 0,
                     (subgroup_ops & VK_SUBGROUP_FEATURE_BASIC_BIT) ? 1 : 0,
                     (subgroup_ops & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) ? 1 : 0);

            std::vector<uint32_t> spirv;
            if (ncnn::compile_spirv_module(source, strlen(source), opt, spirv) != 0)
            {
                fprintf(stderr, "test_compile_spirv_module_subgroup_option failed device=%d subgroup=%d cooperative_matrix=%d\n", device_index, opt.use_subgroup_ops, opt.use_cooperative_matrix);
                return -1;
            }
        }
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    if (ncnn::get_gpu_count() == 0)
        return 0;

    return 0
           || test_compile_spirv_module_0()
           || test_compile_spirv_module_1()
           || test_compile_spirv_module_subgroup_option();
}
