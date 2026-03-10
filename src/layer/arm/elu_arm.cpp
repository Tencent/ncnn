// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "elu_arm.h"

#if __ARM_NEON
#include <arm_neon.h>

#include "neon_mathfun.h"
#endif // __ARM_NEON

namespace ncnn {

ELU_arm::ELU_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON
}

int ELU_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;

#if __ARM_NEON
        float32x4_t _alpha = vdupq_n_f32(alpha);
        float32x4_t _one = vdupq_n_f32(1.f);
        float32x4_t _zero = vdupq_n_f32(0.f);
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vld1q_f32(ptr);
            uint32x4_t _lemask = vcleq_f32(_p, _zero);

            float32x4_t _nps = exp_ps(_p);
            _nps = vsubq_f32(_nps, _one);
            _nps = vmulq_f32(_nps, _alpha);

            _p = vbslq_f32(_lemask, _nps, _p);
            vst1q_f32(ptr, _p);

            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            if (*ptr < 0.f)
                *ptr = alpha * (expf(*ptr) - 1.f);
            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
