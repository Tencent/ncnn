// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gpu.h"
#include "layer_shader_type.h"
#include "testutil.h"

static int test_fast_math(bool fp16_packed, bool fp16_storage, bool fp16_arithmetic)
{
    const ncnn::GpuInfo& info = ncnn::get_gpu_info();

    ncnn::Option opt;
    opt.use_subgroup_ops = false;
    opt.use_cooperative_matrix = false;
    opt.use_fp16_packed = fp16_packed && info.support_fp16_packed();
    opt.use_fp16_storage = fp16_storage && info.support_fp16_storage();
    opt.use_fp16_arithmetic = fp16_arithmetic && info.support_fp16_arithmetic();

    std::vector<uint32_t> spirv;
    int ret = ncnn::compile_spirv_module(ncnn::LayerShaderType::absval, opt, spirv);
    if (ret != 0)
        return ret;

    std::vector<uint32_t> float_types;
    std::vector<uint32_t> fast_math_types;
    uint32_t fast_math_id = 0;
    uint32_t fast_math_flags = 0;
    for (size_t i = 5; i < spirv.size();)
    {
        uint32_t wordcount = spirv[i] >> 16;
        uint32_t op = spirv[i] & 0xffff;

        if (op == 22) // floating-point type
        {
            float_types.push_back(spirv[i + 1]);
        }
        else if (op == 331 && spirv[i + 2] == 6028) // default fast math execution mode
        {
            fast_math_types.push_back(spirv[i + 3]);
            fast_math_id = spirv[i + 4];
        }
        else if (op == 43 && spirv[i + 2] == fast_math_id) // fast math constant
        {
            fast_math_flags = spirv[i + 3];
        }

        i += wordcount;
    }

    if (info.support_fp_fast_math())
    {
        if (spirv[1] < 0x10200 || float_types.empty() || fast_math_types != float_types || fast_math_flags != 0x7000f)
        {
            fprintf(stderr, "test_fast_math failed fp16_packed=%d fp16_storage=%d fp16_arithmetic=%d\n", fp16_packed, fp16_storage, fp16_arithmetic);
            return -1;
        }
    }
    else if (!fast_math_types.empty())
    {
        fprintf(stderr, "test_fast_math enabled on unsupported device\n");
        return -1;
    }

    ncnn::ParamDict pd;
    std::vector<ncnn::Mat> weights;
    return test_layer_opt("AbsVal", pd, weights, opt, RandomMat(13, 7, 5));
}

static int test_fast_math_no_uint()
{
    const char shader[] = "#version 450\n"
                          "layout(binding = 0) buffer data { float value; };\n"
                          "void main()\n"
                          "{\n"
                          "    value = value * 2.0 + 1.0;\n"
                          "}\n";

    ncnn::Option opt;
    opt.use_subgroup_ops = false;
    opt.use_cooperative_matrix = false;

    std::vector<uint32_t> spirv;
    int ret = ncnn::compile_spirv_module(shader, opt, spirv);
    if (ret != 0)
        return ret;

    uint32_t uint32_type_id = 0;
    uint32_t float_type_id = 0;
    uint32_t fast_math_type_id = 0;
    uint32_t fast_math_id = 0;
    uint32_t fast_math_constant_type_id = 0;
    uint32_t fast_math_flags = 0;
    for (size_t i = 5; i < spirv.size();)
    {
        uint32_t wordcount = spirv[i] >> 16;
        uint32_t op = spirv[i] & 0xffff;

        if (op == 21 && spirv[i + 2] == 32 && spirv[i + 3] == 0) // unsigned integer type
            uint32_type_id = spirv[i + 1];
        else if (op == 22) // floating-point type
            float_type_id = spirv[i + 1];
        else if (op == 331 && spirv[i + 2] == 6028) // default fast math execution mode
        {
            fast_math_type_id = spirv[i + 3];
            fast_math_id = spirv[i + 4];
        }
        else if (op == 43 && spirv[i + 2] == fast_math_id) // fast math constant
        {
            fast_math_constant_type_id = spirv[i + 1];
            fast_math_flags = spirv[i + 3];
        }

        i += wordcount;
    }

    if (ncnn::get_gpu_info().support_fp_fast_math())
    {
        if (float_type_id == 0 || fast_math_type_id != float_type_id || uint32_type_id == 0 || fast_math_constant_type_id != uint32_type_id || fast_math_flags != 0x7000f)
        {
            fprintf(stderr, "test_fast_math_no_uint failed\n");
            return -1;
        }
    }
    else if (fast_math_id != 0)
    {
        fprintf(stderr, "test_fast_math_no_uint enabled on unsupported device\n");
        return -1;
    }

    return 0;
}

static int test_fast_math_precise()
{
    const char shader[] = "#version 450\n"
                          "layout(binding = 0) buffer data { float values[]; };\n"
                          "void main()\n"
                          "{\n"
                          "    uint i = gl_GlobalInvocationID.x;\n"
                          "    precise float v = values[i] * 2.0 + values[i];\n"
                          "    values[i] = v;\n"
                          "}\n";

    ncnn::Option opt;
    std::vector<uint32_t> spirv;
    int ret = ncnn::compile_spirv_module(shader, opt, spirv);
    if (ret != 0)
        return ret;

    bool no_contraction = false;
    for (size_t i = 5; i < spirv.size();)
    {
        uint32_t wordcount = spirv[i] >> 16;
        uint32_t op = spirv[i] & 0xffff;
        if ((op == 16 || op == 331) && spirv[i + 2] == 6028)
        {
            fprintf(stderr, "test_fast_math_precise enabled fast math\n");
            return -1;
        }
        if (op == 71 && spirv[i + 2] == 42)
            no_contraction = true;

        i += wordcount;
    }

    if (!no_contraction)
    {
        fprintf(stderr, "test_fast_math_precise lost no contraction decoration\n");
        return -1;
    }

    return 0;
}

int main()
{
    if (ncnn::get_gpu_count() == 0)
        return 0;

    SRAND(7767517);

    return 0
           || test_fast_math(false, false, false)
           || test_fast_math(true, false, true)
           || test_fast_math(true, true, false)
           || test_fast_math(true, true, true)
           || test_fast_math_no_uint()
           || test_fast_math_precise();
}
