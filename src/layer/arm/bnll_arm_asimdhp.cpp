// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#include "bnll_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#include "arm_usability.h"
#endif // __ARM_NEON

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static void bnll_fp16s(Mat& bottom_top_blob, const Option& opt)
{
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;
    const int row_size = w * elempack;
    const int rows = h * d;
    const int total_rows = channels * rows;

    const float32x4_t _zero = vdupq_n_f32(0.f);
    const float32x4_t _one = vdupq_n_f32(1.f);

    const bool use_parallel = opt.num_threads > 1 && total_rows > 1 && total_rows * row_size >= 4096;

    #pragma omp parallel for num_threads(opt.num_threads) if (use_parallel)
    for (int i = 0; i < total_rows; i++)
    {
        const int q = i / rows;
        const int y = i % rows;
        __fp16* ptr = (__fp16*)bottom_top_blob.channel(q) + y * row_size;

        int j = 0;
#if __aarch64__
        for (; j + 7 < row_size; j += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            float32x4_t _p0 = vcvt_f32_f16(vget_low_f16(_p));
            float32x4_t _p1 = vcvt_f32_f16(vget_high_f16(_p));
            float32x4_t _y0 = log_ps(vaddq_f32(_one, exp_ps(vnegq_f32(vabsq_f32(_p0)))));
            float32x4_t _y1 = log_ps(vaddq_f32(_one, exp_ps(vnegq_f32(vabsq_f32(_p1)))));
            _p0 = vaddq_f32(_y0, vmaxq_f32(_p0, _zero));
            _p1 = vaddq_f32(_y1, vmaxq_f32(_p1, _zero));
            vst1q_f16(ptr, vcombine_f16(vcvt_f16_f32(_p0), vcvt_f16_f32(_p1)));
            ptr += 8;
        }
#endif // __aarch64__
        for (; j + 3 < row_size; j += 4)
        {
            float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
            float32x4_t _y = log_ps(vaddq_f32(_one, exp_ps(vnegq_f32(vabsq_f32(_p)))));
            _p = vaddq_f32(_y, vmaxq_f32(_p, _zero));
            vst1_f16(ptr, vcvt_f16_f32(_p));
            ptr += 4;
        }
        for (; j < row_size; j++)
        {
            float v = (float) * ptr;
            if (v > 0)
                v = v + logf(1.f + expf(-v));
            else
                v = logf(1.f + expf(v));
            *ptr = (__fp16)v;
            ptr++;
        }
    }
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

int BNLL_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    bnll_fp16s(bottom_top_blob, opt);

    return 0;
}

int BNLL_arm::forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const
{
    bnll_fp16s(bottom_top_blob, opt);

    return 0;
}

} // namespace ncnn
