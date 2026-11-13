// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gpu.h"
#include "layer_shader_type.h"
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

static int test_cooperative_matrix_conversion_features()
{
    for (int i = 0; i < ncnn::get_gpu_count(); i++)
    {
        const ncnn::GpuInfo& info = ncnn::get_gpu_info(i);
        const VkPhysicalDeviceCooperativeMatrixConversionFeaturesQCOM& features = info.queryCooperativeMatrixConversionFeaturesQCOM();
        bool forced = false;

#if defined __ANDROID__
        forced = info.vendor_id() == 0x5143 && info.support_VK_KHR_cooperative_matrix() && !info.support_VK_QCOM_cooperative_matrix_conversion();
#endif // __ANDROID__

        const bool expected = info.queryCooperativeMatrixFeatures().cooperativeMatrix
                              && ((info.support_VK_QCOM_cooperative_matrix_conversion() && features.cooperativeMatrixConversion) || forced);

        struct Feature
        {
            VkStructureType sType;
            const void* pNext;
        };
        bool chained = false;
        for (const Feature* f = (const Feature*)info.queryExtensionFeatures(); f; f = (const Feature*)f->pNext)
        {
            if (f->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_CONVERSION_FEATURES_QCOM)
                chained = true;
        }

        if (info.support_cooperative_matrix_conversion())
        {
            const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(i);
            if (!vkdev || !vkdev->is_valid())
            {
                fprintf(stderr, "test_cooperative_matrix_conversion_features failed device creation gpu=%d\n", i);
                return -1;
            }
        }

        if (features.sType != VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_CONVERSION_FEATURES_QCOM
                || (!info.support_VK_QCOM_cooperative_matrix_conversion() && features.cooperativeMatrixConversion)
                || chained != (info.support_VK_QCOM_cooperative_matrix_conversion() != 0)
                || (info.support_VK_QCOM_cooperative_matrix_conversion() && !info.support_VK_KHR_cooperative_matrix())
                || info.support_cooperative_matrix_conversion() != expected)
        {
            fprintf(stderr, "test_cooperative_matrix_conversion_features failed gpu=%d\n", i);
            return -1;
        }
    }

    return 0;
}

static int test_compile_spirv_module_cooperative_matrix_conversion()
{
#if NCNN_TEST_LAYER_convolution || NCNN_TEST_LAYER_convolution1d || NCNN_TEST_LAYER_deconvolution || NCNN_TEST_LAYER_gemm || NCNN_TEST_LAYER_sdpa
    struct Shader
    {
        int type;
        const char* name;
        bool int8;
    };
    const Shader shaders[] = {
#if NCNN_TEST_LAYER_convolution
        {ncnn::LayerShaderType::convolution_1x1s1d1_cm_qcom, "convolution_1x1s1d1_cm_qcom", false},
        {ncnn::LayerShaderType::convolution_gemm_cm_qcom, "convolution_gemm_cm_qcom", false},
        {ncnn::LayerShaderType::convolution_winograd_gemm_cm_qcom, "convolution_winograd_gemm_cm_qcom", false},
#if NCNN_INT8
        {ncnn::LayerShaderType::convolution_1x1s1d1_int8_cm_qcom, "convolution_1x1s1d1_int8_cm_qcom", true},
        {ncnn::LayerShaderType::convolution_gemm_int8_cm_qcom, "convolution_gemm_int8_cm_qcom", true},
        {ncnn::LayerShaderType::convolution_3x3s1d1_winograd_gemm_int8_cm_qcom, "convolution_3x3s1d1_winograd_gemm_int8_cm_qcom", true},
#endif
#endif
#if NCNN_TEST_LAYER_convolution1d
        {ncnn::LayerShaderType::convolution1d_1x1s1d1_cm_qcom, "convolution1d_1x1s1d1_cm_qcom", false},
        {ncnn::LayerShaderType::convolution1d_gemm_cm_qcom, "convolution1d_gemm_cm_qcom", false},
#endif
#if NCNN_TEST_LAYER_deconvolution
        {ncnn::LayerShaderType::deconvolution_gemm_cm_qcom, "deconvolution_gemm_cm_qcom", false},
#endif
#if NCNN_TEST_LAYER_gemm
        {ncnn::LayerShaderType::gemm_cm_qcom, "gemm_cm_qcom", false},
#if NCNN_INT8
        {ncnn::LayerShaderType::gemm_int8_cm_qcom, "gemm_int8_cm_qcom", true},
#endif
#endif
#if NCNN_TEST_LAYER_sdpa
        {ncnn::LayerShaderType::sdpa_cross_cm_qcom, "sdpa_cross_cm_qcom", false},
        {ncnn::LayerShaderType::sdpa_fa_cm_qcom, "sdpa_fa_cm_qcom", false},
#endif
    };

    for (int device_index = 0; device_index < ncnn::get_gpu_count(); device_index++)
    {
        // compile only, matrix dimensions are supplied by pipeline specialization
        for (size_t i = 0; i < sizeof(shaders) / sizeof(shaders[0]); i++)
        {
            for (int fp16_storage = 0; fp16_storage < 2; fp16_storage++)
            {
                for (int fp16_arithmetic = 0; fp16_arithmetic < 2; fp16_arithmetic++)
                {
                    for (int int8_storage = 0; int8_storage < (shaders[i].int8 ? 2 : 1); int8_storage++)
                    {
                        ncnn::Option opt;
                        opt.vulkan_device_index = device_index;
                        opt.use_cooperative_matrix = true;
                        opt.use_subgroup_ops = true;
                        opt.use_bf16_storage = false;
                        opt.use_fp16_packed = !fp16_storage;
                        opt.use_fp16_storage = fp16_storage;
                        opt.use_fp16_arithmetic = fp16_arithmetic;
                        opt.use_fp16_uniform = false;
                        opt.use_int8_packed = shaders[i].int8 && !int8_storage;
                        opt.use_int8_storage = int8_storage;
                        opt.use_int8_arithmetic = shaders[i].int8;
                        opt.use_int8_uniform = false;

                        std::vector<uint32_t> spirv;
                        int ret = ncnn::compile_spirv_module(shaders[i].type, opt, spirv);
                        if (ret != 0 || spirv.empty())
                        {
                            fprintf(stderr, "test_compile_spirv_module_cooperative_matrix_conversion failed gpu=%d shader=%s fp16_storage=%d fp16_arithmetic=%d int8_storage=%d ret=%d\n", device_index, shaders[i].name,
                                    fp16_storage, fp16_arithmetic, int8_storage, ret);
                            return -1;
                        }
                    }
                }
            }
        }
    }
#endif

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
           || test_cooperative_matrix_conversion_features()
           || test_compile_spirv_module_cooperative_matrix_conversion();
}
