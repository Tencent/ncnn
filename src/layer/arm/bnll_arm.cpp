// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#include "bnll_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#endif // __ARM_NEON

#include "arm_usability.h"
#include "cpu.h"

namespace ncnn {

#if __ARM_NEON
static NCNN_FORCEINLINE float32x4_t bnll_ps(float32x4_t x)
{
    const float32x4_t _zero = vdupq_n_f32(0.f);
    const float32x4_t _one = vdupq_n_f32(1.f);

    float32x4_t _abs_x = vabsq_f32(x);
    float32x4_t y = log_ps(vaddq_f32(_one, exp_ps(vnegq_f32(_abs_x))));
    return vaddq_f32(y, vmaxq_f32(x, _zero));
}

static NCNN_FORCEINLINE void bnll_fp32_row(float* ptr, int size)
{
    int i = 0;
#if __aarch64__
    for (; i + 7 < size; i += 8)
    {
        float32x4_t _p0 = vld1q_f32(ptr);
        float32x4_t _p1 = vld1q_f32(ptr + 4);
        _p0 = bnll_ps(_p0);
        _p1 = bnll_ps(_p1);
        vst1q_f32(ptr, _p0);
        vst1q_f32(ptr + 4, _p1);
        ptr += 8;
    }
#endif // __aarch64__
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _p = vld1q_f32(ptr);
        _p = bnll_ps(_p);
        vst1q_f32(ptr, _p);
        ptr += 4;
    }
    for (; i < size; i++)
    {
        if (*ptr > 0)
            *ptr = *ptr + logf(1.f + expf(-*ptr));
        else
            *ptr = logf(1.f + expf(*ptr));
        ptr++;
    }
}
#endif // __ARM_NEON

BNLL_arm::BNLL_arm()
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

int BNLL_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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

    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;
    const int row_size = w * elempack;
    const int rows = h * d;
    const int total_rows = channels * rows;

    const bool use_parallel = opt.num_threads > 1 && total_rows > 1 && total_rows * row_size >= 4096;

    #pragma omp parallel for num_threads(opt.num_threads) if (use_parallel)
    for (int i = 0; i < total_rows; i++)
    {
        const int q = i / rows;
        const int y = i % rows;
        float* ptr = (float*)bottom_top_blob.channel(q) + y * row_size;

#if __ARM_NEON
        bnll_fp32_row(ptr, row_size);
#else
        for (int j = 0; j < row_size; j++)
        {
            if (ptr[j] > 0)
                ptr[j] = ptr[j] + logf(1.f + expf(-ptr[j]));
            else
                ptr[j] = logf(1.f + expf(ptr[j]));
        }
#endif // __ARM_NEON
    }

    return 0;
}

#if NCNN_BF16
int BNLL_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;
    const int row_size = w * elempack;
    const int rows = h * d;
    const int total_rows = channels * rows;

    const bool use_parallel = opt.num_threads > 1 && total_rows > 1 && total_rows * row_size >= 4096;

    #pragma omp parallel for num_threads(opt.num_threads) if (use_parallel)
    for (int i = 0; i < total_rows; i++)
    {
        const int q = i / rows;
        const int y = i % rows;
        unsigned short* ptr = (unsigned short*)bottom_top_blob.channel(q) + y * row_size;

        int j = 0;
#if __ARM_NEON
#if __aarch64__
        for (; j + 7 < row_size; j += 8)
        {
            uint16x8_t _p = vld1q_u16(ptr);
            float32x4_t _p0 = bnll_ps(bfloat2float(vget_low_u16(_p)));
            float32x4_t _p1 = bnll_ps(bfloat2float(vget_high_u16(_p)));
            vst1q_u16(ptr, vcombine_u16(float2bfloat(_p0), float2bfloat(_p1)));
            ptr += 8;
        }
#endif // __aarch64__
        for (; j + 3 < row_size; j += 4)
        {
            float32x4_t _p = bnll_ps(bfloat2float(vld1_u16(ptr)));
            vst1_u16(ptr, float2bfloat(_p));
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; j < row_size; j++)
        {
            float v = bfloat16_to_float32(*ptr);
            if (v > 0)
                v = v + logf(1.f + expf(-v));
            else
                v = logf(1.f + expf(v));
            *ptr = float32_to_bfloat16(v);
            ptr++;
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
