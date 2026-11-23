// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#include "softplus_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#include "arm_usability.h"
#endif // __ARM_NEON

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static NCNN_FORCEINLINE float32x4_t softplus_ps_f16(float32x4_t x)
{
    const float32x4_t _one = vdupq_n_f32(1.f);
    const float32x4_t _exp_hi = vdupq_n_f32(88.3762626647949f);

    float32x4_t y = log_ps(vaddq_f32(exp_ps(x), _one));
    uint32x4_t _mask = vcleq_f32(x, _exp_hi);
    return vbslq_f32(_mask, y, x);
}

static int softplus_fp16s(Mat& bottom_top_blob, const Option& opt)
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
        __fp16* ptr = (__fp16*)bottom_top_blob.channel(q) + y * row_size;

        int j = 0;
#if __aarch64__
        for (; j + 7 < row_size; j += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            float32x4_t _p0 = softplus_ps_f16(vcvt_f32_f16(vget_low_f16(_p)));
            float32x4_t _p1 = softplus_ps_f16(vcvt_f32_f16(vget_high_f16(_p)));
            vst1q_f16(ptr, vcombine_f16(vcvt_f16_f32(_p0), vcvt_f16_f32(_p1)));
            ptr += 8;
        }
#endif // __aarch64__
        for (; j + 3 < row_size; j += 4)
        {
            float32x4_t _p = softplus_ps_f16(vcvt_f32_f16(vld1_f16(ptr)));
            vst1_f16(ptr, vcvt_f16_f32(_p));
            ptr += 4;
        }
        for (; j < row_size; j++)
        {
            float v = (float)*ptr;
            if (v <= 88.3762626647949f)
                *ptr = (__fp16)logf(expf(v) + 1.f);
            ptr++;
        }
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

int Softplus_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    return softplus_fp16s(bottom_top_blob, opt);
}

int Softplus_arm::forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const
{
    return softplus_fp16s(bottom_top_blob, opt);
}

} // namespace ncnn
