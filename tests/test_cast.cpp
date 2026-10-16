// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_float32_to_float16(float value, unsigned short expected)
{
    unsigned short actual = ncnn::float32_to_float16(value);
    if (actual != expected)
    {
        fprintf(stderr, "test_float32_to_float16 failed value=%f expected=%04x actual=%04x\n", value, expected, actual);
        return -1;
    }

    return 0;
}

static int test_float16_to_float32(unsigned short value, unsigned int expected)
{
    float actual = ncnn::float16_to_float32(value);
    unsigned int actual_bits;
    memcpy(&actual_bits, &actual, sizeof(float));
    if (actual_bits != expected)
    {
        fprintf(stderr, "test_float16_to_float32 failed value=%04x expected=%08x actual=%08x\n", value, expected, actual_bits);
        return -1;
    }

    return 0;
}

static int test_float16_to_float8(unsigned short value, unsigned char expected)
{
    unsigned char actual = ncnn::float16_to_float8(value);
    if (actual != expected)
    {
        fprintf(stderr, "test_float16_to_float8 failed value=%04x expected=%02x actual=%02x\n", value, expected, actual);
        return -1;
    }

    return 0;
}

static int test_float32_to_bfloat16(float value, unsigned short expected)
{
    unsigned short actual = ncnn::float32_to_bfloat16(value);
    if (actual != expected)
    {
        fprintf(stderr, "test_float32_to_bfloat16 failed value=%f expected=%04x actual=%04x\n", value, expected, actual);
        return -1;
    }

    return 0;
}

static int test_float16_to_bfloat8(unsigned short value, unsigned char expected)
{
    unsigned char actual = ncnn::float16_to_bfloat8(value);
    if (actual != expected)
    {
        fprintf(stderr, "test_float16_to_bfloat8 failed value=%04x expected=%02x actual=%02x\n", value, expected, actual);
        return -1;
    }

    return 0;
}

static int test_float8_to_float16(unsigned char value, unsigned short expected)
{
    unsigned short actual = ncnn::float8_to_float16(value);
    if (actual != expected)
    {
        fprintf(stderr, "test_float8_to_float16 failed value=%02x expected=%04x actual=%04x\n", value, expected, actual);
        return -1;
    }

    return 0;
}

static int test_cast_rounding()
{
    const unsigned int inf_bits = 0x7f800000u;
    const unsigned int nan_bits = 0x7fc00000u;
    float inf;
    float nan;
    memcpy(&inf, &inf_bits, sizeof(float));
    memcpy(&nan, &nan_bits, sizeof(float));

    return 0
           || test_float32_to_float16(0.7f, 0x399a)
           || test_float32_to_float16(-0.7f, 0xb99a)
           || test_float32_to_float16(1.00048828125f, 0x3c01)
           || test_float32_to_float16(1.99951171875f, 0x4000)
           || test_float32_to_float16(65504.f, 0x7bff)
           || test_float32_to_float16(65520.f, 0x7c00)
           || test_float32_to_float16(65536.f, 0x7c00)
           || test_float32_to_float16(-65536.f, 0xfc00)
           || test_float32_to_float16(inf, 0x7c00)
           || test_float32_to_float16(-inf, 0xfc00)
           || test_float32_to_float16(nan, 0x7e00)
           || test_float32_to_float16(0.00006103515625f, 0x0400)
           || test_float32_to_float16(0.000030517578125f, 0x0000)
           || test_float16_to_float32(0x0000, 0x00000000u)
           || test_float16_to_float32(0x8000, 0x80000000u)
           || test_float16_to_float32(0x7c00, 0x7f800000u)
           || test_float16_to_float32(0xfc00, 0xff800000u)
           || test_float16_to_float32(0x7e00, 0x7fc00000u)
           || test_float16_to_float8(0x3c40, 0x39)
           || test_float16_to_float8(0x3fc0, 0x40)
           || test_float16_to_float8(0x5b80, 0x77)
           || test_float16_to_float8(0x5bc0, 0x78)
           || test_float16_to_float8(0x2400, 0x08)
           || test_float16_to_float8(0x2000, 0x00)
           || test_float32_to_bfloat16(0.7f, 0x3f33)
           || test_float32_to_bfloat16(-0.7f, 0xbf33)
           || test_float32_to_bfloat16(1.00390625f, 0x3f81)
           || test_float32_to_bfloat16(1.99609375f, 0x4000)
           || test_float16_to_bfloat8(0x399a, 0x3a)
           || test_float16_to_bfloat8(0xb99a, 0xba)
           || test_float16_to_bfloat8(0x3c80, 0x3d)
           || test_float16_to_bfloat8(0x3f80, 0x40)
           // subnormal decoding and E4M3 zero, overflow and NaN boundaries
           || test_float16_to_float32(0x0001, 0x33800000u)
           || test_float16_to_float32(0x8001, 0xb3800000u)
           || test_float16_to_float32(0x0200, 0x38000000u)
           || test_float16_to_float8(0x0000, 0x00)
           || test_float16_to_float8(0x8001, 0x80)
           || test_float16_to_float8(0x7c00, 0x7f)
           || test_float16_to_float8(0xfe00, 0xff)
           || test_float16_to_float8(0x7bff, 0x7f)
           || test_float8_to_float16(0x00, 0x0000)
           || test_float8_to_float16(0x80, 0x8000)
           || test_float8_to_float16(0x01, 0x0000)
           || test_float8_to_float16(0x81, 0x8000)
           || test_float8_to_float16(0x7f, 0x7e00)
           || test_float8_to_float16(0xff, 0xfe00)
           || test_float8_to_float16(0x38, 0x3c00)
           || test_float8_to_float16(0xb8, 0xbc00);
}

#if NCNN_BF16 && __riscv
static int test_cast_bfloat16_bits(int elempack)
{
    // include both sides of ties, signed zero, subnormals, overflow, Inf and NaN payloads
    const unsigned int fp32_bits[] = {
        0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u,
        0x00007fffu, 0x00008000u, 0x00008001u, 0x80008000u,
        0x007fffffu, 0x00800000u, 0x3f807fffu, 0x3f808000u,
        0x3f808001u, 0x3f818000u, 0xbf807fffu, 0xbf808000u,
        0xbf808001u, 0xbf818000u, 0x7f7f7fffu, 0x7f7f8000u,
        0xff7f7fffu, 0xff7f8000u, 0x7f800000u, 0xff800000u,
        0x7f800001u, 0xff800001u, 0x7fc00000u, 0xffc00000u,
        0x7fff7fffu, 0x7fff8000u, 0xffff7fffu, 0xffff8000u,
        0x7fffffffu, 0xffffffffu
    };
    const unsigned short bf16_bits[] = {
        0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x0080,
        0x3f80, 0xbf80, 0x7f7f, 0xff7f, 0x7f80, 0xff80,
        0x7f81, 0xff81, 0x7fc0, 0xffc0, 0x7fff, 0xffff
    };
#if __riscv_vector || (!NCNN_RUNTIME_CPU && (NCNN_RVV || NCNN_XTHEADVECTOR))
    bool use_vector_rounding = true;
#else
    bool use_vector_rounding = false;
#endif
#if NCNN_RVV
    use_vector_rounding = use_vector_rounding || ncnn::cpu_support_riscv_v();
#endif
#if NCNN_XTHEADVECTOR
    use_vector_rounding = use_vector_rounding || ncnn::cpu_support_riscv_xtheadvector();
#endif
    bool use_native_conversion = false;
#if NCNN_ZVFBFMIN
#if NCNN_RUNTIME_CPU
    use_native_conversion = use_vector_rounding && ncnn::cpu_support_riscv_zvfbfmin();
#elif __riscv_zvfbfmin || (NCNN_RVV && !__riscv_xtheadvector)
    use_native_conversion = true;
#endif
#endif
    bool use_xthead_conversion = false;
#if NCNN_XTHEADVECTOR
#if __riscv_xtheadvector || (!NCNN_RUNTIME_CPU && !NCNN_RVV)
    use_xthead_conversion = !use_native_conversion;
#elif NCNN_RUNTIME_CPU
    use_xthead_conversion = !use_native_conversion && ncnn::cpu_support_riscv_xtheadvector();
#endif
    use_native_conversion = use_native_conversion || use_xthead_conversion;
#endif
    const int batch = NCNN_BATCH ? 2 : 1;
    ncnn::Mat a(37, 1, 3, (size_t)4u * elempack, elempack, batch);
    ncnn::Mat a16(37, 1, 3, (size_t)2u * elempack, elempack, batch);
    const int size = a.w * a.h * elempack;
    for (int b = 0; b < batch; b++)
    {
        for (int q = 0; q < a.c; q++)
        {
            float* p = a.batch(b).channel(q);
            unsigned short* p16 = a16.batch(b).channel(q);
            for (int i = 0; i < size; i++)
            {
                const unsigned int bits = fp32_bits[(i + q + b) % (sizeof(fp32_bits) / sizeof(fp32_bits[0]))];
                memcpy(p + i, &bits, sizeof(bits));
                p16[i] = bf16_bits[(i + q + b) % (sizeof(bf16_bits) / sizeof(bf16_bits[0]))];
            }
        }
    }

    ncnn::Option opt;
    opt.num_threads = 1;
    ncnn::Mat out16;
    ncnn::Mat out32;
    ncnn::cast_float32_to_bfloat16(a, out16, opt);
    ncnn::cast_bfloat16_to_float32(a16, out32, opt);
    if (out16.empty() || out32.empty() || out16.n != batch || out32.n != batch
            || out16.elempack != elempack || out32.elempack != elempack
            || out16.elemsize != 2u * elempack || out32.elemsize != 4u * elempack)
    {
        fprintf(stderr, "test_cast_bfloat16_bits layout failed elempack=%d\n", elempack);
        return -1;
    }
    for (int b = 0; b < batch; b++)
    {
        for (int q = 0; q < a.c; q++)
        {
            const float* p = a.batch(b).channel(q);
            const unsigned short* p16 = a16.batch(b).channel(q);
            const unsigned short* r16 = out16.batch(b).channel(q);
            const float* r32 = out32.batch(b).channel(q);
            for (int i = 0; i < size; i++)
            {
                unsigned int bits;
                memcpy(&bits, r32 + i, sizeof(bits));
                unsigned short expected = ncnn::float32_to_bfloat16(p[i]);
                if (use_vector_rounding || use_native_conversion)
                {
                    // RVV vnclipu rounds ties to even and saturates unsigned overflow
                    unsigned int fp32;
                    memcpy(&fp32, p + i, sizeof(fp32));
                    unsigned int rounded = (fp32 >> 16) + ((fp32 & 0xffff) > 0x8000 || ((fp32 & 0xffff) == 0x8000 && ((fp32 >> 16) & 1)));
                    expected = rounded > 0xffff ? 0xffff : (unsigned short)rounded;
                }
                unsigned int expected32 = (unsigned int)p16[i] << 16;
                if (use_native_conversion)
                {
                    // native BF16 conversion uses RNE; XThead DQNaN may propagate NaN payloads
                    unsigned int fp32;
                    memcpy(&fp32, p + i, sizeof(fp32));
                    if ((fp32 & 0x7fffffff) > 0x7f800000)
                        expected = 0x7fc0;
                    if ((p16[i] & 0x7fff) > 0x7f80)
                        expected32 = 0x7fc00000;
                }
                bool narrow_matches = r16[i] == expected;
                bool widen_matches = bits == expected32;
                if (use_xthead_conversion)
                {
                    unsigned int fp32;
                    memcpy(&fp32, p + i, sizeof(fp32));
                    if ((fp32 & 0x7fffffff) > 0x7f800000)
                        narrow_matches = (r16[i] & 0x7fc0) == 0x7fc0;
                    if ((p16[i] & 0x7fff) > 0x7f80)
                        widen_matches = (bits & 0x7fc00000) == 0x7fc00000;
                }
                if (!narrow_matches || !widen_matches)
                {
                    fprintf(stderr, "test_cast_bfloat16_bits failed elempack=%d batch=%d channel=%d i=%d native=%d expected=%04x/%08x actual=%04x/%08x\n", elempack, b, q, i, use_native_conversion, expected, expected32, r16[i], bits);
                    return -1;
                }
            }
        }
    }
    return 0;
}

static int test_cast_bfloat16_rounding()
{
#if __riscv_flen
    unsigned int original_fcsr;
    asm volatile("csrr %0, fcsr"
                 : "=r"(original_fcsr)
                 :
                 : "memory");
    // RNE, RTZ, RDN, RUP and RMM must all leave the BF16 conversion unchanged
    for (unsigned int mode = 0; mode < 5; mode++)
    {
        // keep a pre-existing divide-by-zero flag across both conversions
        const unsigned int fcsr = (mode << 5) | 8;
        asm volatile("csrw fcsr, %0"
                     :
                     : "r"(fcsr)
                     : "memory");
        int ret = test_cast_bfloat16_bits(1) || test_cast_bfloat16_bits(4);
        unsigned int actual_fcsr;
        asm volatile("csrr %0, fcsr"
                     : "=r"(actual_fcsr)
                     :
                     : "memory");
        asm volatile("csrw fcsr, %0"
                     :
                     : "r"(original_fcsr)
                     : "memory");
        if (ret != 0 || (actual_fcsr >> 5) != mode || (actual_fcsr & 8) == 0)
        {
            fprintf(stderr, "test_cast_bfloat16_rounding failed mode=%u fcsr=%u\n", mode, actual_fcsr);
            return -1;
        }
    }
    return 0;
#else
    return test_cast_bfloat16_bits(1) || test_cast_bfloat16_bits(4);
#endif
}
#endif // NCNN_BF16 && __riscv

static int cast_cpu_naive(const ncnn::Mat& a, ncnn::Mat& b, int type_from, int type_to)
{
    ncnn::ParamDict pd;
    pd.set(0, type_from);
    pd.set(1, type_to);

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;

    ncnn::Layer* op = ncnn::create_layer_naive("Cast");

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    op->create_pipeline(opt);

    op->forward(a, b, opt);

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

static int test_cast_cpu(const ncnn::Mat& a, int type_from, int type_to)
{
    ncnn::ParamDict pd;
    pd.set(0, type_from);
    pd.set(1, type_to);

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = false;
    opt.use_int8_inference = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer_cpu("Cast");

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    op->create_pipeline(opt);

    ncnn::Mat a_fp16;
    cast_cpu_naive(a, a_fp16, 1, type_from);

    ncnn::Mat b;
    cast_cpu_naive(a_fp16, b, type_from, type_to);

    ncnn::Mat c;
    op->forward(a_fp16, c, opt);

    op->destroy_pipeline(opt);

    delete op;

    if (CompareMat(b, c, 0.001) != 0)
    {
        fprintf(stderr, "test_cast_cpu failed a.dims=%d a=(%d %d %d %d) type_from=%d type_to=%d\n", a.dims, a.w, a.h, a.d, a.c, type_from, type_to);
        return -1;
    }

    return 0;
}

static int test_cast_cpu_packed(const ncnn::Mat& a, int type_from, int type_to)
{
    ncnn::ParamDict pd;
    pd.set(0, type_from);
    pd.set(1, type_to);

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer_cpu("Cast");

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    op->create_pipeline(opt);

    ncnn::Mat a_fp16;
    cast_cpu_naive(a, a_fp16, 1, type_from);

    ncnn::Mat b;
    cast_cpu_naive(a_fp16, b, type_from, type_to);

    ncnn::Mat a4;
    ncnn::convert_packing(a, a4, 4, opt);

    ncnn::Mat a4_fp16;
    cast_cpu_naive(a4, a4_fp16, 1, type_from);

    ncnn::Mat c;
    op->forward(a4_fp16, c, opt);

    op->destroy_pipeline(opt);

    delete op;

    if (CompareMat(b, c, 0.001) != 0)
    {
        fprintf(stderr, "test_cast_cpu_packed failed a.dims=%d a=(%d %d %d %d) type_from=%d type_to=%d\n", a.dims, a.w, a.h, a.d, a.c, type_from, type_to);
        return -1;
    }

    return 0;
}

#if NCNN_VULKAN
static int test_cast_gpu_fp16p(const ncnn::Mat& a, int type_from, int type_to)
{
    if (type_to == 4 || type_from == 4)
        return 0;
    ncnn::ParamDict pd;
    pd.set(0, type_from);
    pd.set(1, type_to);

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_int8_inference = false;
    opt.use_fp16_packed = true;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;
    opt.use_packing_layout = true;

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();

    if (!vkdev->info.support_fp16_packed())
        return 0;

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    if (!vkdev->info.support_fp16_storage()) opt.use_fp16_storage = false;

    ncnn::Layer* op = ncnn::create_layer_vulkan("Cast");

    op->vkdev = vkdev;

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    op->create_pipeline(opt);

    ncnn::Mat a_fp16;
    if (type_from == 2)
    {
        ncnn::cast_float32_to_float16(a, a_fp16, opt);
    }
    else
    {
        a_fp16 = a;
    }

    ncnn::Mat b;
    cast_cpu_naive(a_fp16, b, type_from, type_to);

    ncnn::Mat d;

    // pack
    ncnn::Mat a4;
    ncnn::convert_packing(a, a4, 4, opt);

    ncnn::Mat a4_fp16;
    if (type_from == 2)
    {
        ncnn::cast_float32_to_float16(a4, a4_fp16, opt);
    }
    else
    {
        a4_fp16 = a4;
    }

    // forward
    ncnn::VkCompute cmd(vkdev);

    // upload
    ncnn::VkMat a4_gpu;
    cmd.record_clone(a4_fp16, a4_gpu, opt);

    ncnn::VkMat d4_gpu;
    if (op->support_inplace)
    {
        op->forward_inplace(a4_gpu, cmd, opt);

        d4_gpu = a4_gpu;
    }
    else
    {
        op->forward(a4_gpu, d4_gpu, cmd, opt);
    }

    // download
    cmd.record_clone(d4_gpu, d, opt);

    cmd.submit_and_wait();

    op->destroy_pipeline(opt);

    delete op;

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    if (CompareMat(b, d, 0.001) != 0)
    {
        fprintf(stderr, "test_cast_gpu_fp16p failed a.dims=%d a=(%d %d %d %d) type_from=%d type_to=%d\n", a.dims, a.w, a.h, a.d, a.c, type_from, type_to);
        return -1;
    }

    return 0;
}
#endif // NCNN_VULKAN

static int test_cast(const ncnn::Mat& a, int type_from, int type_to)
{
    return 0
           || test_cast_cpu(a, type_from, type_to)
           || test_cast_cpu_packed(a, type_from, type_to)
#if NCNN_VULKAN
           || test_cast_gpu_fp16p(a, type_from, type_to)
#endif // NCNN_VULKAN
           ;
}

static int test_cast_0()
{
    return 0
           || test_cast(RandomMat(5, 6, 7, 16), 1, 2)
           || test_cast(RandomMat(3, 4, 5, 13), 1, 2)
           || test_cast(RandomMat(5, 6, 7, 16), 2, 1)
           || test_cast(RandomMat(3, 4, 5, 13), 2, 1)
           || test_cast(RandomMat(5, 6, 7, 16), 1, 4)
           || test_cast(RandomMat(3, 4, 5, 13), 1, 4)
           || test_cast(RandomMat(5, 6, 7, 16), 4, 1)
           || test_cast(RandomMat(3, 4, 5, 13), 4, 1);
}

static int test_cast_1()
{
    return 0
           || test_cast(RandomMat(5, 7, 16), 1, 2)
           || test_cast(RandomMat(3, 5, 13), 1, 2)
           || test_cast(RandomMat(5, 7, 16), 2, 1)
           || test_cast(RandomMat(3, 5, 13), 2, 1)
           || test_cast(RandomMat(5, 7, 16), 1, 4)
           || test_cast(RandomMat(3, 5, 13), 1, 4)
           || test_cast(RandomMat(5, 7, 16), 4, 1)
           || test_cast(RandomMat(3, 5, 13), 4, 1);
}

static int test_cast_2()
{
    return 0
           || test_cast(RandomMat(6, 16), 1, 2)
           || test_cast(RandomMat(7, 15), 1, 2)
           || test_cast(RandomMat(6, 16), 2, 1)
           || test_cast(RandomMat(7, 15), 2, 1)
           || test_cast(RandomMat(6, 16), 1, 4)
           || test_cast(RandomMat(7, 15), 1, 4)
           || test_cast(RandomMat(6, 16), 4, 1)
           || test_cast(RandomMat(7, 15), 4, 1);
}

static int test_cast_3()
{
    return 0
           || test_cast(RandomMat(128), 1, 2)
           || test_cast(RandomMat(127), 1, 2)
           || test_cast(RandomMat(128), 2, 1)
           || test_cast(RandomMat(127), 2, 1)
           || test_cast(RandomMat(128), 1, 4)
           || test_cast(RandomMat(127), 1, 4)
           || test_cast(RandomMat(128), 4, 1)
           || test_cast(RandomMat(127), 4, 1);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_cast_rounding()
#if NCNN_BF16 && __riscv
           || test_cast_bfloat16_rounding()
#endif
           || test_cast_0()
           || test_cast_1()
           || test_cast_2()
           || test_cast_3();
}
