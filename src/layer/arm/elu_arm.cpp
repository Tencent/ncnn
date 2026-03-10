// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "elu_arm.h"

#if __ARM_NEON
#include <arm_neon.h>

#include "arm_usability.h"
#include "neon_mathfun.h"
#endif // __ARM_NEON

#include "cpu.h"

namespace ncnn {

ELU_arm::ELU_arm()
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

int ELU_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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

#if NCNN_BF16
int ELU_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
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
        unsigned short* ptr = bottom_top_blob.channel(q);

        int i = 0;

#if __ARM_NEON
        float32x4_t _alpha = vdupq_n_f32(alpha);
        float32x4_t _one = vdupq_n_f32(1.f);
        float32x4_t _zero = vdupq_n_f32(0.f);

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
            _p0 = exp_ps(_p0);
            _p1 = exp_ps(_p1);
            _p2 = exp_ps(_p2);
            _p3 = exp_ps(_p3);
            _p0 = vsubq_f32(_p0, _one);
            _p1 = vsubq_f32(_p1, _one);
            _p2 = vsubq_f32(_p2, _one);
            _p3 = vsubq_f32(_p3, _one);
            _p0 = vmulq_f32(_p0, _alpha);
            _p1 = vmulq_f32(_p1, _alpha);
            _p2 = vmulq_f32(_p2, _alpha);
            _p3 = vmulq_f32(_p3, _alpha);
            float32x4_t _orig0 = bfloat2float(vget_low_u16(vld1q_u16(ptr)));
            float32x4_t _orig1 = bfloat2float(vget_high_u16(vld1q_u16(ptr)));
            float32x4_t _orig2 = bfloat2float(vget_low_u16(vld1q_u16(ptr + 8)));
            float32x4_t _orig3 = bfloat2float(vget_high_u16(vld1q_u16(ptr + 8)));
            _p0 = vbslq_f32(_lemask0, _p0, _orig0);
            _p1 = vbslq_f32(_lemask1, _p1, _orig1);
            _p2 = vbslq_f32(_lemask2, _p2, _orig2);
            _p3 = vbslq_f32(_lemask3, _p3, _orig3);
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
            _p0 = exp_ps(_p0);
            _p1 = exp_ps(_p1);
            _p0 = vsubq_f32(_p0, _one);
            _p1 = vsubq_f32(_p1, _one);
            _p0 = vmulq_f32(_p0, _alpha);
            _p1 = vmulq_f32(_p1, _alpha);
            float32x4_t _orig0 = bfloat2float(vget_low_u16(vld1q_u16(ptr)));
            float32x4_t _orig1 = bfloat2float(vget_high_u16(vld1q_u16(ptr)));
            _p0 = vbslq_f32(_lemask0, _p0, _orig0);
            _p1 = vbslq_f32(_lemask1, _p1, _orig1);
            _p = vcombine_u16(float2bfloat(_p0), float2bfloat(_p1));
            vst1q_u16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr));
            uint32x4_t _lemask = vcleq_f32(_p, _zero);
            _p = exp_ps(_p);
            _p = vsubq_f32(_p, _one);
            _p = vmulq_f32(_p, _alpha);
            float32x4_t _orig = bfloat2float(vld1_u16(ptr));
            _p = vbslq_f32(_lemask, _p, _orig);
            vst1_u16(ptr, float2bfloat(_p));
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(ptr[0]);
            if (v < 0.f)
                v = alpha * (expf(v) - 1.f);
            ptr[0] = float32_to_bfloat16(v);
            ptr++;
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
