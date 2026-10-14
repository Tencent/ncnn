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

#if NCNN_BF16 && NCNN_RVV
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
        fprintf(stderr, "test_clip_bf16_storage missing bf16 support pack=%d\n", elempack);
        delete op;
        return -1;
    }

    ncnn::ParamDict pd;
    // Midpoints must round away from zero, not to the even bf16 value.
    pd.set(0, -1.00390625f);
    pd.set(1, 1.00390625f);

    const unsigned short input[] = {0xc000, 0xbf80, 0xbf00, 0x8000, 0x0000, 0x3f00, 0x3f80, 0x4000};
    const unsigned short expected[] = {0xbf81, 0xbf80, 0xbf00, 0x8000, 0x0000, 0x3f00, 0x3f80, 0x3f81};
    ncnn::Mat a(elempack == 1 ? 127 : 3, 3, 3, 2, (size_t)2u * elempack, elempack);
    const int size = a.w * a.h * a.d * elempack;
    for (int q = 0; q < a.c; q++)
    {
        unsigned short* ptr = a.channel(q);
        for (int i = 0; i < size; i++)
            ptr[i] = input[(i + q) % 8];
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
            if (ptr[i] != expected[(i + q) % 8])
            {
                fprintf(stderr, "test_clip_bf16_storage failed pack=%d q=%d i=%d expected=%04x actual=%04x\n", elempack, q, i, expected[(i + q) % 8], ptr[i]);
                return -1;
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16 && NCNN_RVV

int main()
{
    SRAND(7767517);

#if NCNN_BF16 && NCNN_RVV
    if (test_clip_bf16_storage(1) != 0)
        return -1;
    if (ncnn::cpu_support_riscv_v() && test_clip_bf16_storage(ncnn::cpu_riscv_vlenb() / 2) != 0)
        return -1;
#endif

    return 0
           || test_clip_0()
           || test_clip_1()
           || test_clip_2()
           || test_clip_3();
}
