// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "scale_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#endif // __ARM_NEON

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static void scale_fp16s(__fp16* ptr, const float* scale, const float* bias, int size, int elempack)
{
    int i = 0;
    if (elempack == 8)
    {
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr + i);
            float32x4_t _p0 = vcvt_f32_f16(vget_low_f16(_p));
            float32x4_t _p1 = vcvt_f32_f16(vget_high_f16(_p));
            float32x4_t _s0 = vld1q_f32(scale);
            float32x4_t _s1 = vld1q_f32(scale + 4);
            if (bias)
            {
                _p0 = vmlaq_f32(vld1q_f32(bias), _p0, _s0);
                _p1 = vmlaq_f32(vld1q_f32(bias + 4), _p1, _s1);
            }
            else
            {
                _p0 = vmulq_f32(_p0, _s0);
                _p1 = vmulq_f32(_p1, _s1);
            }
            vst1q_f16(ptr + i, vcombine_f16(vcvt_f16_f32(_p0), vcvt_f16_f32(_p1)));
        }
    }
    else if (elempack >= 4)
    {
        for (; i + elempack - 1 < size; i += elempack)
        {
            for (int k = 0; k < elempack; k += 4)
            {
                float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr + i + k));
                float32x4_t _s = vld1q_f32(scale + k);
                if (bias)
                    _p = vmlaq_f32(vld1q_f32(bias + k), _p, _s);
                else
                    _p = vmulq_f32(_p, _s);
                vst1_f16(ptr + i + k, vcvt_f16_f32(_p));
            }
        }
    }
    else if (elempack == 1)
    {
        float32x4_t _s = vdupq_n_f32(scale[0]);
        float32x4_t _b = bias ? vdupq_n_f32(bias[0]) : vdupq_n_f32(0.f);
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr + i));
            if (bias)
                _p = vmlaq_f32(_b, _p, _s);
            else
                _p = vmulq_f32(_p, _s);
            vst1_f16(ptr + i, vcvt_f16_f32(_p));
        }
    }
    for (; i < size; i++)
    {
        float v = (float)ptr[i] * scale[i % elempack];
        if (bias)
            v += bias[i % elempack];
        ptr[i] = (__fp16)v;
    }
}

int Scale_arm::forward_inplace_fp16s(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    Mat& bottom_top_blob = bottom_top_blobs[0];
    const Mat& scale_blob = bottom_top_blobs[1];
    const float* bias_ptr = bias_data;

    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;

    Mat scale_fp32;
    const float* scale = 0;
    if (scale_blob.elembits() == 16)
    {
        const int scale_data_size = scale_blob.w * scale_blob.elempack;
        scale_fp32.create(scale_data_size, 4u, 1, opt.workspace_allocator);
        if (scale_fp32.empty())
            return -100;

        const __fp16* src = scale_blob;
        float* dst = scale_fp32;
        for (int i = 0; i < scale_data_size; i++)
        {
            dst[i] = (float)src[i];
        }
        scale = scale_fp32;
    }
    else
    {
        scale = scale_blob;
    }

    if (dims == 1)
    {
        __fp16* ptr = bottom_top_blob;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < w; ii++)
        {
            const int i = ii * elempack;

            __fp16* p = ptr + i;
            const float* s = scale + i;
            const float* b = bias_term ? bias_ptr + i : 0;
            if (elempack == 4)
            {
                float32x4_t _p = vcvt_f32_f16(vld1_f16(p));
                float32x4_t _s = vld1q_f32(s);
                if (bias_term)
                    _p = vmlaq_f32(vld1q_f32(b), _p, _s);
                else
                    _p = vmulq_f32(_p, _s);
                vst1_f16(p, vcvt_f16_f32(_p));
            }
            else
            {
                for (int k = 0; k < elempack; k++)
                {
                    float v = (float)p[k] * s[k];
                    if (bias_term)
                        v += b[k];
                    p[k] = (__fp16)v;
                }
            }
        }
    }

    if (dims == 2)
    {
        const int size = w * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            __fp16* ptr = bottom_top_blob.row<__fp16>(i);
            scale_fp16s(ptr, scale + i * elempack, bias_term ? bias_ptr + i * elempack : 0, size, elempack);
        }
    }

    if (dims == 3 || dims == 4)
    {
        const int size = w * h * d * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);
            scale_fp16s(ptr, scale + q * elempack, bias_term ? bias_ptr + q * elempack : 0, size, elempack);
        }
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn
