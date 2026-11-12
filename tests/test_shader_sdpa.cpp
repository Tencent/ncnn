// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gpu.h"
#include "layer/vulkan/shader/sdpa_fa_cm.comp.hex.h"

#include <stdio.h>
#include <string.h>

static int test_sdpa_fa_cm(bool khr, bool bf16, bool maintenance1, bool conversions, unsigned int vendor)
{
    // Override device macros in the shader body, after compile_spirv_module injects
    // its device definitions. Compilation needs a Vulkan device, but does not
    // require that device to support cooperative matrices or maintenance1.
    char defines[1024];
    snprintf(defines, sizeof(defines),
             "#undef ncnn_VK_KHR_cooperative_matrix\n#define ncnn_VK_KHR_cooperative_matrix %d\n"
             "#undef ncnn_VK_NV_cooperative_matrix\n#define ncnn_VK_NV_cooperative_matrix %d\n"
             "#undef ncnn_VK_EXT_cooperative_matrix_maintenance1\n#define ncnn_VK_EXT_cooperative_matrix_maintenance1 %d\n"
             "#undef ncnn_cooperativeMatrixConversionsEXT\n#define ncnn_cooperativeMatrixConversionsEXT %d\n"
             "#undef ncnn_vendorID\n#define ncnn_vendorID %u\n",
             khr, !khr, maintenance1, conversions, vendor);
    const size_t source_size = sizeof(sdpa_fa_cm_comp_data);
    const size_t defines_size = strlen(defines);
    char* source = new char[source_size + defines_size + 1];
    memcpy(source, sdpa_fa_cm_comp_data, source_size);
    source[source_size] = 0;
    char* version = strstr(source, "#version");
    char* version_end = version ? strchr(version, '\n') : 0;
    if (!version_end)
    {
        delete[] source;
        return -1;
    }
    const size_t prefix_size = version_end + 1 - source;
    memmove(source + prefix_size + defines_size, source + prefix_size, source_size - prefix_size + 1);
    memcpy(source + prefix_size, defines, defines_size);

    ncnn::Option opt;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = !bf16;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = bf16;
    opt.use_cooperative_matrix = true;

    std::vector<uint32_t> spirv;
    int ret = ncnn::compile_spirv_module(source, (int)(source_size + defines_size), opt, spirv);
    delete[] source;
    if (ret != 0 || spirv.size() < 5)
    {
        fprintf(stderr, "sdpa_fa_cm compile failed khr=%d bf16=%d maintenance1=%d conversions=%d vendor=%x\n", khr, bf16, maintenance1, conversions, vendor);
        return -1;
    }

    bool has_conversions = false;
    bool has_maintenance1 = false;
    int matrix_a_conversions = 0;
    std::vector<bool> zero_constants(spirv[3], false);
    std::vector<bool> matrix_a_types(spirv[3], false);
    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t word_count = spirv[i] >> 16;
        const uint32_t opcode = spirv[i] & 0xffff;
        if (word_count == 0 || word_count > spirv.size() - i)
            return -1;
        // OpCapability CooperativeMatrixConversionsEXT. Use the numeric values
        // so this test does not depend on private glslang/SPIR-V headers.
        if (opcode == 17 && word_count == 2 && spirv[i + 1] == 5429)
            has_conversions = true;
        // OpExtension SPV_EXT_cooperative_matrix_maintenance1.
        const char extension[] = "SPV_EXT_cooperative_matrix_maintenance1";
        if (opcode == 10 && (word_count - 1) * 4 >= sizeof(extension)
            && memcmp(&spirv[i + 1], extension, sizeof(extension)) == 0)
            has_maintenance1 = true;
        // OpConstant 0 identifies gl_MatrixUseA in OpTypeCooperativeMatrixKHR.
        if (opcode == 43 && word_count == 4 && spirv[i + 3] == 0)
            zero_constants[spirv[i + 2]] = true;
        if (opcode == 4456 && word_count == 7 && zero_constants[spirv[i + 6]])
            matrix_a_types[spirv[i + 1]] = true;
        // Mixed component/use constructors emit OpFConvert, not the same-type
        // OpCooperativeMatrixConvertUseEXT. Both attention loops must convert.
        if (opcode == 115 && word_count == 4 && matrix_a_types[spirv[i + 1]])
            matrix_a_conversions++;
        i += word_count;
    }
    const bool expected = khr && maintenance1 && conversions;
    if (has_conversions != expected || has_maintenance1 != expected || matrix_a_conversions != (expected ? 2 : 0))
    {
        fprintf(stderr, "sdpa_fa_cm conversion capability mismatch khr=%d bf16=%d maintenance1=%d conversions=%d vendor=%x expected=%d capability=%d extension=%d matrix_a_conversions=%d\n", khr, bf16,
                maintenance1, conversions, vendor, expected, has_conversions, has_maintenance1, matrix_a_conversions);
        return -1;
    }
    return 0;
}

int main()
{
    if (ncnn::get_gpu_count() == 0)
        return 0;

    for (int bf16 = 0; bf16 < 2; bf16++)
    {
        for (int amd = 0; amd < 2; amd++)
        {
            for (int maintenance1 = 0; maintenance1 < 2; maintenance1++)
            {
                for (int conversions = 0; conversions < 2; conversions++)
                {
                    if (test_sdpa_fa_cm(true, bf16, maintenance1, conversions, amd ? 0x1002 : 0x10de))
                        return -1;
                }
            }
        }
    }
    return test_sdpa_fa_cm(false, false, false, false, 0x10de);
}
