// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gelu_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#endif // __ARM_NEON

#include "arm_usability.h"
#include "cpu.h"

namespace ncnn {

GELU_arm::GELU_arm()
{
#if __ARM_NEON
    support_packing = true;
#if NCNN_ARM82
    support_fp16_storage = cpu_support_arm_asimdhp();
#endif
#endif // __ARM_NEON

#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int GELU_arm::create_pipeline(const Option& /*opt*/)
{
    return 0;
}

static void gelu_fp32(float* ptr, int size, int fast_gelu)
{
    int i = 0;

#if __ARM_NEON
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _pLoad = vld1q_f32(ptr);
        float32x4_t _blob = fast_gelu ? fast_gelu_ps(_pLoad) : gelu_ps(_pLoad);
        vst1q_f32(ptr, _blob);
        ptr += 4;
    }
#endif // __ARM_NEON

    if (fast_gelu)
    {
        for (; i < size; i++)
        {
            *ptr = 0.5f * *ptr * (1.0f + tanhf(0.79788452f * (*ptr + 0.044715f * *ptr * *ptr * *ptr)));
            ptr++;
        }
    }
    else
    {
        for (; i < size; i++)
        {
            *ptr = 0.5f * *ptr * erfcf(-0.70710678f * *ptr);
            ptr++;
        }
    }
}

static int gelu_parallel_rows(Mat& blob, int rows, int row_size, int fast_gelu, const Option& opt)
{
    const int dims = blob.dims;
    const int h = blob.h;
    const int d = blob.d;

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < rows; y++)
        {
            float* ptr = blob.row<float>(y);
            gelu_fp32(ptr, row_size, fast_gelu);
        }
    }
    if (dims == 3)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < rows; y++)
        {
            const int q = y / h;
            float* ptr = blob.channel(q).row<float>(y - q * h);
            gelu_fp32(ptr, row_size, fast_gelu);
        }
    }
    if (dims == 4)
    {
        const int dh = d * h;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < rows; y++)
        {
            const int q = y / dh;
            const int zy = y - q * dh;
            const int z = zy / h;
            float* ptr = blob.channel(q).depth(z).row<float>(zy - z * h);
            gelu_fp32(ptr, row_size, fast_gelu);
        }
    }

    return 0;
}

int GELU_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_inplace_fp16sa(bottom_top_blob, opt);
        else
            return forward_inplace_fp16s(bottom_top_blob, opt);
    }
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int elempack = bottom_top_blob.elempack;
    int channels = bottom_top_blob.c;
    int size = w * h * d * elempack;

    const int row_size = w * elempack;
    const int rows = bottom_top_blob.dims == 2 ? h : bottom_top_blob.dims == 3 ? channels * h : channels * d * h;

    // avoid thread team startup for small tensors, then split independent rows
    if (opt.num_threads > 1 && channels < opt.num_threads && rows >= opt.num_threads && size >= 4096)
    {
        return gelu_parallel_rows(bottom_top_blob, rows, row_size, fast_gelu, opt);
    }

    if (opt.num_threads > 1 && channels > 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            gelu_fp32(ptr, size, fast_gelu);
        }
    }
    else
    {
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            gelu_fp32(ptr, size, fast_gelu);
        }
    }

    return 0;
}

#if NCNN_BF16
static void gelu_bf16s(unsigned short* ptr, int size, int fast_gelu)
{
    int i = 0;

#if __ARM_NEON
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _pLoad = bfloat2float(vld1_u16(ptr));
        float32x4_t _blob = fast_gelu ? fast_gelu_ps(_pLoad) : gelu_ps(_pLoad);
        vst1_u16(ptr, float2bfloat(_blob));
        ptr += 4;
    }
#endif // __ARM_NEON

    if (fast_gelu)
    {
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            v = 0.5f * v * (1.0f + tanhf(0.79788452f * (v + 0.044715f * v * v * v)));
            *ptr = float32_to_bfloat16(v);
            ptr++;
        }
    }
    else
    {
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            v = 0.5f * v * erfcf(-0.70710678f * v);
            *ptr = float32_to_bfloat16(v);
            ptr++;
        }
    }
}

static int gelu_bf16s_parallel_rows(Mat& blob, int rows, int row_size, int fast_gelu, const Option& opt)
{
    const int dims = blob.dims;
    const int h = blob.h;
    const int d = blob.d;

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < rows; y++)
        {
            unsigned short* ptr = blob.row<unsigned short>(y);
            gelu_bf16s(ptr, row_size, fast_gelu);
        }
    }
    if (dims == 3)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < rows; y++)
        {
            const int q = y / h;
            unsigned short* ptr = blob.channel(q).row<unsigned short>(y - q * h);
            gelu_bf16s(ptr, row_size, fast_gelu);
        }
    }
    if (dims == 4)
    {
        const int dh = d * h;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < rows; y++)
        {
            const int q = y / dh;
            const int zy = y - q * dh;
            const int z = zy / h;
            unsigned short* ptr = blob.channel(q).depth(z).row<unsigned short>(zy - z * h);
            gelu_bf16s(ptr, row_size, fast_gelu);
        }
    }

    return 0;
}

int GELU_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int elempack = bottom_top_blob.elempack;
    int channels = bottom_top_blob.c;
    int size = w * h * d * elempack;

    const int row_size = w * elempack;
    const int rows = bottom_top_blob.dims == 2 ? h : bottom_top_blob.dims == 3 ? channels * h : channels * d * h;

    if (opt.num_threads > 1 && channels < opt.num_threads && rows >= opt.num_threads && size >= 4096)
    {
        return gelu_bf16s_parallel_rows(bottom_top_blob, rows, row_size, fast_gelu, opt);
    }

    if (opt.num_threads > 1 && channels > 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);
            gelu_bf16s(ptr, size, fast_gelu);
        }
    }
    else
    {
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);
            gelu_bf16s(ptr, size, fast_gelu);
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
