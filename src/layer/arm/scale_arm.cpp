// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "scale_arm.h"

#include "cpu.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#endif // __ARM_NEON

namespace ncnn {

Scale_arm::Scale_arm()
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

int Scale_arm::forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    Mat& bottom_top_blob = bottom_top_blobs[0];
    const Mat& scale_blob = bottom_top_blobs[1];

    const int elembits = bottom_top_blob.elembits();

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
        return forward_inplace_fp16s(bottom_top_blobs, opt);
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_inplace_bf16s(bottom_top_blobs, opt);
#endif

    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;

#if __ARM_NEON
    if (elempack == 4)
    {
        if (dims == 1)
        {
            int w = bottom_top_blob.w;

            const float* scale = scale_blob;
            if (bias_term)
            {
                const float* bias = bias_data;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float* ptr = (float*)bottom_top_blob + i * 4;

                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _s = vld1q_f32(scale + i * 4);
                    float32x4_t _bias = vld1q_f32(bias + i * 4);
                    _p = vmlaq_f32(_bias, _p, _s);
                    vst1q_f32(ptr, _p);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float* ptr = (float*)bottom_top_blob + i * 4;

                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _s = vld1q_f32(scale + i * 4);
                    _p = vmulq_f32(_p, _s);
                    vst1q_f32(ptr, _p);
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;

            if (bias_term)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.row(i);
                    float32x4_t _s = vld1q_f32((const float*)scale_blob + i * 4);
                    float32x4_t _bias = vld1q_f32((const float*)bias_data + i * 4);

                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _p = vld1q_f32(ptr);
                        _p = vmlaq_f32(_bias, _p, _s);
                        vst1q_f32(ptr, _p);

                        ptr += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.row(i);
                    float32x4_t _s = vld1q_f32((const float*)scale_blob + i * 4);

                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _p = vld1q_f32(ptr);
                        _p = vmulq_f32(_p, _s);
                        vst1q_f32(ptr, _p);

                        ptr += 4;
                    }
                }
            }
        }

        if (dims == 3 || dims == 4)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int d = bottom_top_blob.d;
            int channels = bottom_top_blob.c;
            int size = w * h * d;

            if (bias_term)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    float* ptr = bottom_top_blob.channel(q);
                    float32x4_t _s = vld1q_f32((const float*)scale_blob + q * 4);
                    float32x4_t _bias = vld1q_f32((const float*)bias_data + q * 4);

                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _p = vld1q_f32(ptr);
                        _p = vmlaq_f32(_bias, _p, _s);
                        vst1q_f32(ptr, _p);

                        ptr += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    float* ptr = bottom_top_blob.channel(q);
                    float32x4_t _s = vld1q_f32((const float*)scale_blob + q * 4);

                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _p = vld1q_f32(ptr);
                        _p = vmulq_f32(_p, _s);
                        vst1q_f32(ptr, _p);

                        ptr += 4;
                    }
                }
            }
        }

        return 0;
    }
#endif // __ARM_NEON

    if (dims != 3 && dims != 4)
        return Scale::forward_inplace(bottom_top_blobs, opt);

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int size = w * h * d;

    if (bias_term)
    {
        const float* scale_ptr = scale_blob;
        const float* bias_ptr = bias_data;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            float s = scale_ptr[q];
            float bias = bias_ptr[q];

#if __ARM_NEON
            int nn = size >> 2;
            int remain = size - (nn << 2);
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
            float32x4_t _s = vdupq_n_f32(s);
            float32x4_t _bias = vdupq_n_f32(bias);
            for (; nn > 0; nn--)
            {
                float32x4_t _p = vld1q_f32(ptr);
                _p = vmlaq_f32(_bias, _p, _s);
                vst1q_f32(ptr, _p);

                ptr += 4;
            }
#endif // __ARM_NEON

            for (; remain > 0; remain--)
            {
                *ptr = *ptr * s + bias;

                ptr++;
            }
        }
    }
    else
    {
        const float* scale_ptr = scale_blob;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            float s = scale_ptr[q];

#if __ARM_NEON
            int nn = size >> 2;
            int remain = size - (nn << 2);
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
            float32x4_t _s = vdupq_n_f32(s);
            for (; nn > 0; nn--)
            {
                float32x4_t _p = vld1q_f32(ptr);
                _p = vmulq_f32(_p, _s);
                vst1q_f32(ptr, _p);

                ptr += 4;
            }
#endif // __ARM_NEON

            for (; remain > 0; remain--)
            {
                *ptr *= s;

                ptr++;
            }
        }
    }

    return 0;
}

#if NCNN_BF16
static void scale_bf16s(unsigned short* ptr, const float* scale, const float* bias, int size, int elempack)
{
    int i = 0;
#if __ARM_NEON
    if (elempack == 8)
    {
        for (; i + 7 < size; i += 8)
        {
            uint16x8_t _p = vld1q_u16(ptr + i);
            float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
            float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
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
            vst1q_u16(ptr + i, vcombine_u16(float2bfloat(_p0), float2bfloat(_p1)));
        }
    }
    else if (elempack >= 4)
    {
        for (; i + elempack - 1 < size; i += elempack)
        {
            for (int k = 0; k < elempack; k += 4)
            {
                float32x4_t _p = bfloat2float(vld1_u16(ptr + i + k));
                float32x4_t _s = vld1q_f32(scale + k);
                if (bias)
                    _p = vmlaq_f32(vld1q_f32(bias + k), _p, _s);
                else
                    _p = vmulq_f32(_p, _s);
                vst1_u16(ptr + i + k, float2bfloat(_p));
            }
        }
    }
    else if (elempack == 1)
    {
        float32x4_t _s = vdupq_n_f32(scale[0]);
        float32x4_t _b = bias ? vdupq_n_f32(bias[0]) : vdupq_n_f32(0.f);
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr + i));
            if (bias)
                _p = vmlaq_f32(_b, _p, _s);
            else
                _p = vmulq_f32(_p, _s);
            vst1_u16(ptr + i, float2bfloat(_p));
        }
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        float v = bfloat16_to_float32(ptr[i]) * scale[i % elempack];
        if (bias)
            v += bias[i % elempack];
        ptr[i] = float32_to_bfloat16(v);
    }
}

int Scale_arm::forward_inplace_bf16s(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
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

        const unsigned short* src = scale_blob;
        float* dst = scale_fp32;
        for (int i = 0; i < scale_data_size; i++)
        {
            dst[i] = bfloat16_to_float32(src[i]);
        }
        scale = scale_fp32;
    }
    else
    {
        scale = scale_blob;
    }

    if (dims == 1)
    {
        unsigned short* ptr = bottom_top_blob;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < w; ii++)
        {
            const int i = ii * elempack;

            unsigned short* p = ptr + i;
            const float* s = scale + i;
            const float* b = bias_term ? bias_ptr + i : 0;
            if (elempack == 4)
            {
                float32x4_t _p = bfloat2float(vld1_u16(p));
                float32x4_t _s = vld1q_f32(s);
                if (bias_term)
                    _p = vmlaq_f32(vld1q_f32(b), _p, _s);
                else
                    _p = vmulq_f32(_p, _s);
                vst1_u16(p, float2bfloat(_p));
            }
            else
            {
                for (int k = 0; k < elempack; k++)
                {
                    float v = bfloat16_to_float32(p[k]) * s[k];
                    if (bias_term)
                        v += b[k];
                    p[k] = float32_to_bfloat16(v);
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
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);
            scale_bf16s(ptr, scale + i * elempack, bias_term ? bias_ptr + i * elempack : 0, size, elempack);
        }
    }

    if (dims == 3 || dims == 4)
    {
        const int size = w * h * d * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);
            scale_bf16s(ptr, scale + q * elempack, bias_term ? bias_ptr + q * elempack : 0, size, elempack);
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
