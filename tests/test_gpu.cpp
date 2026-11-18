// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "datareader.h"
#include "gpu.h"
#include "layer_shader_type.h"
#include "net.h"
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

static int test_cooperative_matrix_conversion_bias()
{
#if NCNN_STDIO && NCNN_STRING && (NCNN_TEST_LAYER_convolution || NCNN_TEST_LAYER_convolution1d)
    struct
    {
        const char* type;
        int dims;
        int kernel;
        int channels;
    } cases[] = {
#if NCNN_TEST_LAYER_convolution
        {"Convolution", 3, 1, 256},
        {"Convolution", 3, 3, 64},
#endif
#if NCNN_TEST_LAYER_convolution1d
        {"Convolution1D", 2, 1, 256},
        {"Convolution1D", 2, 3, 64},
#endif
    };

    for (int device_index = 0; device_index < ncnn::get_gpu_count(); device_index++)
    {
        const ncnn::GpuInfo& info = ncnn::get_gpu_info(device_index);
        if (!info.support_cooperative_matrix_conversion() || !info.support_fp16_storage() || !info.support_fp16_arithmetic())
            continue;

        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
        {
            // 48 output channels select N=16 on Adreno; 196 rows also exercise the M tail.
            const int num_output = 48;
            const int maxk = cases[i].dims == 3 ? cases[i].kernel * cases[i].kernel : cases[i].kernel;
            const int weight_size = num_output * cases[i].channels * maxk;
            char param[256];
            sprintf(param, "7767517\n1 2\n%s conv 1 1 in out 0=%d 1=%d 4=%d 5=1 6=%d\n",
                    cases[i].type, num_output, cases[i].kernel, cases[i].kernel / 2, weight_size);

            std::vector<float> model(1 + weight_size + num_output);
            model[0] = 0.f;
            for (int j = 0; j < weight_size + num_output; j++)
                model[1 + j] = (j * 13 % 7 - 3) * 0.0625f;

            ncnn::Mat input = cases[i].dims == 3 ? ncnn::Mat(14, 14, cases[i].channels) : ncnn::Mat(196, cases[i].channels);
            for (int q = 0; q < input.c; q++)
            {
                float* p = input.channel(q);
                for (int j = 0; j < input.w * input.h; j++)
                    p[j] = ((j * 17 + q * 3) % 13 - 6) * 0.0625f;
            }

            ncnn::Mat expected;
            for (int gpu = 0; gpu < 2; gpu++)
            {
                ncnn::Net net;
                net.opt.num_threads = 1;
                net.opt.vulkan_device_index = device_index;
                net.opt.use_vulkan_compute = gpu;
                net.opt.use_packing_layout = gpu;
                net.opt.use_fp16_packed = gpu;
                net.opt.use_fp16_storage = gpu;
                net.opt.use_fp16_arithmetic = gpu;
                net.opt.use_bf16_packed = false;
                net.opt.use_bf16_storage = false;
                net.opt.use_cooperative_matrix = true;
                net.opt.use_sgemm_convolution = true;
                net.opt.use_winograd_convolution = false;

                const unsigned char* mem = (const unsigned char*)model.data();
                ncnn::DataReaderFromMemory dr(mem, model.size() * sizeof(float));
                if (net.load_param_mem(param) != 0 || net.load_model(dr) != 0)
                    return -1;

                ncnn::Extractor ex = net.create_extractor();
                ncnn::Mat output;
                if (ex.input("in", input) != 0 || ex.extract("out", output) != 0)
                    return -1;

                if (!gpu)
                {
                    expected = output;
                }
                else if (CompareMat(expected, output, 0.01f) != 0)
                {
                    fprintf(stderr, "test_cooperative_matrix_conversion_bias failed gpu=%d type=%s kernel=%d channels=%d\n", device_index, cases[i].type, cases[i].kernel, cases[i].channels);
                    return -1;
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
           || test_compile_spirv_module_cooperative_matrix_conversion()
           || test_cooperative_matrix_conversion_bias();
}
