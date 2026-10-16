// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_clip(const ncnn::Mat& a, float min, float max, int flag = 0)
{
    ncnn::ParamDict pd;
    pd.set(0, min);
    pd.set(1, max);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Clip", pd, weights, a, 0.001, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_clip failed a.dims=%d a=(%d %d %d %d) min=%f max=%f\n", a.dims, a.w, a.h, a.d, a.c, min, max);
    }

    return ret;
}

// cpu pack8/pack16 cases reuse the Vulkan pack4 path covered by the pack4 cases
// keep the 1d sizes for dispatch boundary coverage
static int test_clip_0()
{
    return 0
           || test_clip(RandomMat(3, 3, 3, 48), -1.f, 1.f, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_clip(RandomMat(5, 6, 7, 24), -1.f, 1.f, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_clip(RandomMat(7, 8, 9, 12), -1.f, 1.f)
           || test_clip(RandomMat(3, 4, 5, 13), -1.f, 1.f);
}

static int test_clip_1()
{
    return 0
           || test_clip(RandomMat(3, 3, 48), -1.f, 1.f, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_clip(RandomMat(5, 7, 24), -1.f, 1.f, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_clip(RandomMat(7, 9, 12), -1.f, 1.f)
           || test_clip(RandomMat(3, 5, 13), -1.f, 1.f);
}

static int test_clip_2()
{
    return 0
           || test_clip(RandomMat(19, 48), -1.f, 1.f, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_clip(RandomMat(15, 24), -1.f, 1.f, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_clip(RandomMat(17, 12), -1.f, 1.f)
           || test_clip(RandomMat(19, 15), -1.f, 1.f);
}

static int test_clip_3()
{
    return 0
           || test_clip(RandomMat(128), -1.f, 1.f)
           || test_clip(RandomMat(124), -1.f, 1.f)
           || test_clip(RandomMat(127), -1.f, 1.f);
}

#if NCNN_BF16
static int test_clip_bf16_storage(int elempack)
{
    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = elempack != 1;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = true;

    ncnn::Layer* op = ncnn::create_layer_cpu("Clip");
    if (!op->support_bf16_storage || (elempack != 1 && !op->support_packing))
    {
        delete op;
        return 0;
    }

    ncnn::ParamDict pd;
    // native conversions may round midpoints to even instead of ncnn rounding
    pd.set(0, -1.00390625f);
    pd.set(1, 1.00390625f);

    const unsigned short input[] = {0xc000, 0xbf80, 0xbf00, 0x8000, 0x0000, 0x3f00, 0x3f80, 0x4000, 0xff80, 0x7f80, 0x7fc0, 0xffc0};
    const int input_count = sizeof(input) / sizeof(input[0]);
    const unsigned short expected[] = {0xbf81, 0xbf80, 0xbf00, 0x8000, 0x0000, 0x3f00, 0x3f80, 0x3f81, 0xbf81, 0x3f81, 0x7fc0, 0xffc0};
    ncnn::Mat a(elempack == 1 ? 127 : 3, 3, 3, 2, (size_t)2u * elempack, elempack);
    const int size = a.w * a.h * a.d * elempack;
    for (int q = 0; q < a.c; q++)
    {
        unsigned short* ptr = a.channel(q);
        for (int i = 0; i < size; i++)
            ptr[i] = input[(i + q) % input_count];
    }

    int ret = op->load_param(pd);
    if (ret == 0) ret = op->create_pipeline(opt);
    if (ret == 0) ret = op->forward_inplace(a, opt);
    op->destroy_pipeline(opt);
    delete op;
    if (ret != 0)
        return ret;

    for (int q = 0; q < a.c; q++)
    {
        const unsigned short* ptr = a.channel(q);
        for (int i = 0; i < size; i++)
        {
            const int index = (i + q) % input_count;
            bool matches = ptr[i] == expected[index];
            if (expected[index] == 0xbf81 || expected[index] == 0x3f81)
                matches = matches || ptr[i] == (expected[index] & 0xfffe);
            if ((input[index] & 0x7fff) > 0x7f80)
            {
                // native min/max and fast-math may propagate NaNs or select either bound
                matches = (ptr[i] & 0x7fff) > 0x7f80
                          || ptr[i] == 0xbf80 || ptr[i] == 0xbf81
                          || ptr[i] == 0x3f80 || ptr[i] == 0x3f81;
            }
            if (!matches)
            {
                fprintf(stderr, "test_clip_bf16_storage failed pack=%d q=%d i=%d expected=%04x actual=%04x\n", elempack, q, i, expected[(i + q) % input_count], ptr[i]);
                return -1;
            }
        }
    }

    // check that BF16 Clip leaves subsequent conversions using the FP16 format
    const unsigned int fp32_bits[] = {0x00000000u, 0x80000000u, 0x3f000000u, 0xbf000000u, 0x3f800000u, 0xc0000000u};
    const unsigned short fp16_bits[] = {0x0000, 0x8000, 0x3800, 0xb800, 0x3c00, 0xc000};
    ncnn::Mat fp32(6);
    memcpy(fp32.data, fp32_bits, sizeof(fp32_bits));
    ncnn::Mat fp16;
    ncnn::Mat roundtrip;
    ncnn::cast_float32_to_float16(fp32, fp16, opt);
    ncnn::cast_float16_to_float32(fp16, roundtrip, opt);
    if (fp16.empty() || roundtrip.empty())
        return -1;
    const unsigned short* p16 = fp16;
    const float* p32 = roundtrip;
    for (int i = 0; i < 6; i++)
    {
        unsigned int bits;
        memcpy(&bits, p32 + i, sizeof(bits));
        if (p16[i] != fp16_bits[i] || bits != fp32_bits[i])
        {
            fprintf(stderr, "test_clip_bf16_storage subsequent fp16 conversion failed pack=%d i=%d\n", elempack, i);
            return -1;
        }
    }

    return 0;
}
#endif // NCNN_BF16

int main()
{
    SRAND(7767517);

#if NCNN_BF16
    if (test_clip_bf16_storage(1) != 0 || test_clip_bf16_storage(4) != 0)
        return -1;
#endif

    return 0
           || test_clip_0()
           || test_clip_1()
           || test_clip_2()
           || test_clip_3();
}
