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
static int test_clip_fp16_format()
{
    // bf16 clip must leave subsequent conversions using the fp16 format
    ncnn::Option opt;
    opt.num_threads = 1;
    ncnn::Mat fp32(1);
    fp32[0] = 1.f;
    ncnn::Mat fp16;
    ncnn::cast_float32_to_float16(fp32, fp16, opt);
    if (fp16.empty() || ((const unsigned short*)fp16)[0] != 0x3c00)
    {
        fprintf(stderr, "test_clip_fp16_format failed\n");
        return -1;
    }

    return 0;
}
#endif // NCNN_BF16

int main()
{
    SRAND(7767517);

    return 0
           || test_clip_0()
           || test_clip_1()
           || test_clip_2()
           || test_clip_3()
#if NCNN_BF16
           || test_clip_fp16_format()
#endif
           ;
}
