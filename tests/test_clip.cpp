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

#if NCNN_BF16 && (NCNN_RVV || NCNN_XTHEADVECTOR)
static int test_clip_bf16_storage(int elempack, unsigned int mode)
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
    // RVV rounds midpoints to even; the scalar fallback uses ncnn rounding.
    pd.set(0, -1.00390625f);
    pd.set(1, 1.00390625f);

    const unsigned short input[] = {0xc000, 0xbf80, 0xbf00, 0x8000, 0x0000, 0x3f00, 0x3f80, 0x4000, 0xff80, 0x7f80, 0x7fc0, 0xffc0};
#if __riscv_vector || !NCNN_RUNTIME_CPU
    bool use_vector_rounding = true;
#else
    bool use_vector_rounding = ncnn::cpu_support_riscv_v();
#endif
#if NCNN_XTHEADVECTOR
    use_vector_rounding = use_vector_rounding || ncnn::cpu_support_riscv_xtheadvector();
#endif
    const unsigned short min_expected = use_vector_rounding ? 0xbf80 : 0xbf81;
    const unsigned short max_expected = use_vector_rounding ? 0x3f80 : 0x3f81;
    // Vector min/max select the numeric bound for NaNs. Scalar NaN behavior depends on fast-math.
    const int input_count = use_vector_rounding ? sizeof(input) / sizeof(input[0]) : 10;
    const unsigned short expected[] = {min_expected, 0xbf80, 0xbf00, 0x8000, 0x0000, 0x3f00, 0x3f80, max_expected, min_expected, max_expected, min_expected, min_expected};
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
#if __riscv_flen
    unsigned int original_fcsr;
    // Keep a pre-existing divide-by-zero flag across the conversion.
    const unsigned int fcsr = (mode << 5) | 8;
    asm volatile("csrrw %0, fcsr, %1"
                 : "=r"(original_fcsr)
                 : "r"(fcsr)
                 : "memory");
#endif
    if (ret == 0) ret = op->forward_inplace(a, opt);
#if __riscv_flen
    unsigned int actual_fcsr;
    asm volatile("csrr %0, fcsr"
                 : "=r"(actual_fcsr)
                 :
                 : "memory");
    asm volatile("csrw fcsr, %0"
                 :
                 : "r"(original_fcsr)
                 : "memory");
    if ((actual_fcsr >> 5) != mode || (actual_fcsr & 8) == 0)
    {
        fprintf(stderr, "test_clip_bf16_storage changed floating-point state mode=%u fcsr=%u\n", mode, actual_fcsr);
        ret = -1;
    }
#endif
    op->destroy_pipeline(opt);
    delete op;
    if (ret != 0)
        return ret;

    for (int q = 0; q < a.c; q++)
    {
        const unsigned short* ptr = a.channel(q);
        for (int i = 0; i < size; i++)
        {
            if (ptr[i] != expected[(i + q) % input_count])
            {
                fprintf(stderr, "test_clip_bf16_storage failed pack=%d q=%d i=%d expected=%04x actual=%04x\n", elempack, q, i, expected[(i + q) % input_count], ptr[i]);
                return -1;
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16 && (NCNN_RVV || NCNN_XTHEADVECTOR)

int main()
{
    SRAND(7767517);

#if NCNN_BF16 && (NCNN_RVV || NCNN_XTHEADVECTOR)
#if __riscv_vector || !NCNN_RUNTIME_CPU
    bool use_vector = true;
#else
    bool use_vector = ncnn::cpu_support_riscv_v();
#endif
#if NCNN_XTHEADVECTOR
    use_vector = use_vector || ncnn::cpu_support_riscv_xtheadvector();
#endif
    for (unsigned int mode = 0; mode < 5; mode++)
    {
        if (test_clip_bf16_storage(1, mode) != 0)
            return -1;
        if (use_vector && test_clip_bf16_storage(4, mode) != 0)
            return -1;
    }
#endif

    return 0
           || test_clip_0()
           || test_clip_1()
           || test_clip_2()
           || test_clip_3();
}
