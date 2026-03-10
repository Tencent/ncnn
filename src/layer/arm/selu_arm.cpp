// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "selu_arm.h"

#if __ARM_NEON
#include "neon_mathfun.h"
#include "arm_usability.h"

#include <arm_neon.h>
#endif // __ARM_NEON

#include "cpu.h"

namespace ncnn {

SELU_arm::SELU_arm()
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

int SELU_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
        return forward_inplace_fp16s(bottom_top_blob, opt);
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;
    float alphaxlambda = alpha * lambda;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;

#if __ARM_NEON
        float32x4_t _one = vdupq_n_f32(1.f);
        float32x4_t _zero = vdupq_n_f32(0.f);
        float32x4_t _alphaxlambda = vdupq_n_f32(alphaxlambda);
        float32x4_t _lambda = vdupq_n_f32(lambda);
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vld1q_f32(ptr);
            uint32x4_t _lemask = vcleq_f32(_p, _zero);

            float32x4_t _nps = exp_ps(_p);
            _nps = vsubq_f32(_nps, _one);
            _nps = vmulq_f32(_nps, _alphaxlambda);

            float32x4_t _pps = vmulq_f32(_p, _lambda);

            _p = vbslq_f32(_lemask, _nps, _pps);
            vst1q_f32(ptr, _p);

            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
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

#if NCNN_BF16
int SELU_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;
    float alphaxlambda = alpha * lambda;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = bottom_top_blob.channel(q);

        int i = 0;

#if __ARM_NEON
        float32x4_t _one = vdupq_n_f32(1.f);
        float32x4_t _zero = vdupq_n_f32(0.f);
        float32x4_t _alphaxlambda = vdupq_n_f32(alphaxlambda);
        float32x4_t _lambda = vdupq_n_f32(lambda);

        for (; i + 15 < size; i += 16)
        {
            uint16x8_t _p = vld1q_u16(ptr);
            uint16x8_t _q = vld1q_u16(ptr + 8);
            float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
            float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
            float32x4_t _p2 = bfloat2float(vget_low_u16(_q));
            float32x4_t _p3 = bfloat2float(vget_high_u16(_q));
            uint32x4_t _lemask0 = vcleq_f32(_p0, _zero);
            uint32x4_t _lemask1 = vcleq_f32(_p1, _zero);
            uint32x4_t _lemask2 = vcleq_f32(_p2, _zero);
            uint32x4_t _lemask3 = vcleq_f32(_p3, _zero);

            float32x4_t _nps0 = exp_ps(_p0);
            float32x4_t _nps1 = exp_ps(_p1);
            float32x4_t _nps2 = exp_ps(_p2);
            float32x4_t _nps3 = exp_ps(_p3);

            _nps0 = vsubq_f32(_nps0, _one);
            _nps1 = vsubq_f32(_nps1, _one);
            _nps2 = vsubq_f32(_nps2, _one);
            _nps3 = vsubq_f32(_nps3, _one);

            _nps0 = vmulq_f32(_nps0, _alphaxlambda);
            _nps1 = vmulq_f32(_nps1, _alphaxlambda);
            _nps2 = vmulq_f32(_nps2, _alphaxlambda);
            _nps3 = vmulq_f32(_nps3, _alphaxlambda);

            float32x4_t _pps0 = vmulq_f32(_p0, _lambda);
            float32x4_t _pps1 = vmulq_f32(_p1, _lambda);
            float32x4_t _pps2 = vmulq_f32(_p2, _lambda);
            float32x4_t _pps3 = vmulq_f32(_p3, _lambda);

            _p0 = vbslq_f32(_lemask0, _nps0, _pps0);
            _p1 = vbslq_f32(_lemask1, _nps1, _pps1);
            _p2 = vbslq_f32(_lemask2, _nps2, _pps2);
            _p3 = vbslq_f32(_lemask3, _nps3, _pps3);

            _p = vcombine_u16(float2bfloat(_p0), float2bfloat(_p1));
            _q = vcombine_u16(float2bfloat(_p2), float2bfloat(_p3));
            vst1q_u16(ptr, _p);
            vst1q_u16(ptr + 8, _q);
            ptr += 16;
        }
        for (; i + 7 < size; i += 8)
        {
            uint16x8_t _p = vld1q_u16(ptr);
            float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
            float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
            uint32x4_t _lemask0 = vcleq_f32(_p0, _zero);
            uint32x4_t _lemask1 = vcleq_f32(_p1, _zero);

            float32x4_t _nps0 = exp_ps(_p0);
            float32x4_t _nps1 = exp_ps(_p1);

            _nps0 = vsubq_f32(_nps0, _one);
            _nps1 = vsubq_f32(_nps1, _one);
            _nps0 = vmulq_f32(_nps0, _alphaxlambda);
            _nps1 = vmulq_f32(_nps1, _alphaxlambda);

            float32x4_t _pps0 = vmulq_f32(_p0, _lambda);
            float32x4_t _pps1 = vmulq_f32(_p1, _lambda);

            _p0 = vbslq_f32(_lemask0, _nps0, _pps0);
            _p1 = vbslq_f32(_lemask1, _nps1, _pps1);

            _p = vcombine_u16(float2bfloat(_p0), float2bfloat(_p1));
            vst1q_u16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr));
            uint32x4_t _lemask = vcleq_f32(_p, _zero);

            float32x4_t _nps = exp_ps(_p);
            _nps = vsubq_f32(_nps, _one);
            _nps = vmulq_f32(_nps, _alphaxlambda);

            float32x4_t _pps = vmulq_f32(_p, _lambda);
            _p = vbslq_f32(_lemask, _nps, _pps);

            vst1_u16(ptr, float2bfloat(_p));
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(ptr[0]);
            if (v < 0.f)
                v = (expf(v) - 1.f) * alphaxlambda;
            else
                v *= lambda;
            ptr[0] = float32_to_bfloat16(v);
            ptr++;
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
