// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gelu_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#include "neon_mathfun.h"
#if NCNN_ARM82
#include "neon_mathfun_fp16s.h"
#endif
#endif // __ARM_NEON

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static void gelu_fp16s(__fp16* ptr, int size, int fast_gelu)
{
    int i = 0;

    for (; i + 3 < size; i += 4)
    {
        float32x4_t _pLoad = vcvt_f32_f16(vld1_f16(ptr));
        float32x4_t _blob = fast_gelu ? fast_gelu_ps(_pLoad) : gelu_ps(_pLoad);
        vst1_f16(ptr, vcvt_f16_f32(_blob));
        ptr += 4;
    }

    if (fast_gelu)
    {
        for (; i < size; i++)
        {
            float v = (float)*ptr;
            v = 0.5f * v * (1.0f + tanhf(0.79788452f * (v + 0.044715f * v * v * v)));
            *ptr = (__fp16)v;
            ptr++;
        }
    }
    else
    {
        for (; i < size; i++)
        {
            float v = (float)*ptr;
            v = 0.5f * v * erfcf(-0.70710678f * v);
            *ptr = (__fp16)v;
            ptr++;
        }
    }
}

static void gelu_fp16sa(__fp16* ptr, int size, int fast_gelu)
{
    int i = 0;

    if (fast_gelu)
    {
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _pLoad = vld1q_f16(ptr);
            float16x8_t _blob = fast_gelu_ps_f16(_pLoad);
            vst1q_f16(ptr, _blob);
            ptr += 8;
        }

        for (; i < size; i++)
        {
            *ptr = (__fp16)0.5f * *ptr * (__fp16)(1.0f + tanhf((__fp16)0.79788452f * (*ptr + (__fp16)0.044715f * *ptr * *ptr * *ptr)));
            ptr++;
        }
    }
    else
    {
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _pLoad = vld1q_f16(ptr);
            float16x8_t _blob = gelu_ps_f16(_pLoad);
            vst1q_f16(ptr, _blob);
            ptr += 8;
        }

        for (; i < size; i++)
        {
            float v = (float)*ptr;
            v = 0.5f * v * erfcf(-0.70710678f * v);
            *ptr = (__fp16)v;
            ptr++;
        }
    }
}

static int gelu_fp16s_parallel_rows(Mat& blob, int rows, int row_size, int fast_gelu, int fp16sa, const Option& opt)
{
    const int dims = blob.dims;
    const int h = blob.h;
    const int d = blob.d;

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < rows; y++)
        {
            __fp16* ptr = (__fp16*)blob.row<__fp16>(y);
            if (fp16sa)
                gelu_fp16sa(ptr, row_size, fast_gelu);
            else
                gelu_fp16s(ptr, row_size, fast_gelu);
        }
    }
    if (dims == 3)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < rows; y++)
        {
            const int q = y / h;
            __fp16* ptr = (__fp16*)blob.channel(q).row<__fp16>(y - q * h);
            if (fp16sa)
                gelu_fp16sa(ptr, row_size, fast_gelu);
            else
                gelu_fp16s(ptr, row_size, fast_gelu);
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
            __fp16* ptr = (__fp16*)blob.channel(q).depth(z).row<__fp16>(zy - z * h);
            if (fp16sa)
                gelu_fp16sa(ptr, row_size, fast_gelu);
            else
                gelu_fp16s(ptr, row_size, fast_gelu);
        }
    }

    return 0;
}

static int gelu_fp16_forward(Mat& blob, int size, int fast_gelu, int fp16sa, const Option& opt)
{
    const int w = blob.w;
    const int h = blob.h;
    const int d = blob.d;
    const int elempack = blob.elempack;
    const int channels = blob.c;
    const int row_size = w * elempack;
    const int rows = blob.dims == 2 ? h : blob.dims == 3 ? channels * h : channels * d * h;

    // avoid thread team startup for small tensors, then split independent rows
    if (opt.num_threads > 1 && channels < opt.num_threads && rows >= opt.num_threads && size >= 4096)
    {
        return gelu_fp16s_parallel_rows(blob, rows, row_size, fast_gelu, fp16sa, opt);
    }

    if (opt.num_threads > 1 && channels > 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = (__fp16*)blob.channel(q);
            if (fp16sa)
                gelu_fp16sa(ptr, size, fast_gelu);
            else
                gelu_fp16s(ptr, size, fast_gelu);
        }
    }
    else
    {
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = (__fp16*)blob.channel(q);
            if (fp16sa)
                gelu_fp16sa(ptr, size, fast_gelu);
            else
                gelu_fp16s(ptr, size, fast_gelu);
        }
    }

    return 0;
}

int GELU_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    const int size = bottom_top_blob.w * bottom_top_blob.h * bottom_top_blob.d * bottom_top_blob.elempack;

    return gelu_fp16_forward(bottom_top_blob, size, fast_gelu, 0, opt);
}

int GELU_arm::forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const
{
    const int size = bottom_top_blob.w * bottom_top_blob.h * bottom_top_blob.d * bottom_top_blob.elempack;

    return gelu_fp16_forward(bottom_top_blob, size, fast_gelu, 1, opt);
}
#endif

} // namespace ncnn
