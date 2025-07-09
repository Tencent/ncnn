// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "selu_arm.h"

#if __ARM_NEON
#include "neon_mathfun.h"

#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

int SELU_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int size = w * h * d;
    float alphaxlambda = alpha * lambda;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
#if __ARM_NEON
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
        float32x4_t _one = vdupq_n_f32(1.f);
        float32x4_t _zero = vdupq_n_f32(0.f);
        float32x4_t _alphaxlambda = vdupq_n_f32(alphaxlambda);
        float32x4_t _lambda = vdupq_n_f32(lambda);
        for (; nn > 0; nn--)
        {
            float32x4_t _p = vld1q_f32(ptr);
            uint32x4_t _lemask = vcleq_f32(_p, _zero);

            float32x4_t _nps = exp_ps(_p);
            _nps = vsubq_f32(_nps, _one);
            _nps = vmulq_f32(_nps, _alphaxlambda);

            _p = vmulq_f32(_p, _lambda);

            _p = vbslq_f32(_lemask, _nps, _p);
            vst1q_f32(ptr, _p);

            ptr += 4;
        }
#endif // __ARM_NEON
        for (; remain > 0; remain--)
        {
            if (*ptr < 0.f)
                *ptr = (expf(*ptr) - 1.f) * alphaxlambda;
            else
                *ptr *= lambda;

            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
