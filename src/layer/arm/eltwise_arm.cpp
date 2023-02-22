// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "eltwise_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_usability.h"

#include "cpu.h"

namespace ncnn {

Eltwise_arm::Eltwise_arm()
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

int Eltwise_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    int elembits = bottom_blobs[0].elembits();

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(bottom_blobs, top_blobs, opt);
        else
            return forward_fp16s(bottom_blobs, top_blobs, opt);
    }
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s(bottom_blobs, top_blobs, opt);
#endif

    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int elempack = bottom_blob.elempack;
    int size = w * h * d * elempack;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (op_type == Operation_PROD)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

            int i = 0;
#if __ARM_NEON
            for (; i + 7 < size; i += 8)
            {
                float32x4_t _p0 = vld1q_f32(ptr);
                float32x4_t _p1 = vld1q_f32(ptr + 4);
                float32x4_t _q0 = vld1q_f32(ptr1);
                float32x4_t _q1 = vld1q_f32(ptr1 + 4);
                _p0 = vmulq_f32(_p0, _q0);
                _p1 = vmulq_f32(_p1, _q1);
                vst1q_f32(outptr, _p0);
                vst1q_f32(outptr + 4, _p1);

                ptr += 8;
                ptr1 += 8;
                outptr += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = vld1q_f32(ptr);
                float32x4_t _q = vld1q_f32(ptr1);
                _p = vmulq_f32(_p, _q);
                vst1q_f32(outptr, _p);

                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
#endif // __ARM_NEON
            for (; i < size; i++)
            {
                *outptr = *ptr * *ptr1;

                ptr++;
                ptr1++;
                outptr++;
            }
        }

        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                int i = 0;
#if __ARM_NEON
                for (; i + 7 < size; i += 8)
                {
                    float32x4_t _p0 = vld1q_f32(outptr);
                    float32x4_t _p1 = vld1q_f32(outptr + 4);
                    float32x4_t _q0 = vld1q_f32(ptr);
                    float32x4_t _q1 = vld1q_f32(ptr + 4);
                    _p0 = vmulq_f32(_p0, _q0);
                    _p1 = vmulq_f32(_p1, _q1);
                    vst1q_f32(outptr, _p0);
                    vst1q_f32(outptr + 4, _p1);

                    ptr += 8;
                    outptr += 8;
                }
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _p = vld1q_f32(outptr);
                    float32x4_t _q = vld1q_f32(ptr);
                    _p = vmulq_f32(_p, _q);
                    vst1q_f32(outptr, _p);

                    ptr += 4;
                    outptr += 4;
                }
#endif // __ARM_NEON
                for (; i < size; i++)
                {
                    *outptr *= *ptr;

                    ptr++;
                    outptr++;
                }
            }
        }
    }
    if (op_type == Operation_SUM)
    {
        if (coeffs.w == 0)
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                int i = 0;
#if __ARM_NEON
                for (; i + 7 < size; i += 8)
                {
                    float32x4_t _p0 = vld1q_f32(ptr);
                    float32x4_t _p1 = vld1q_f32(ptr + 4);
                    float32x4_t _q0 = vld1q_f32(ptr1);
                    float32x4_t _q1 = vld1q_f32(ptr1 + 4);
                    _p0 = vaddq_f32(_p0, _q0);
                    _p1 = vaddq_f32(_p1, _q1);
                    vst1q_f32(outptr, _p0);
                    vst1q_f32(outptr + 4, _p1);

                    ptr += 8;
                    ptr1 += 8;
                    outptr += 8;
                }
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _q = vld1q_f32(ptr1);
                    _p = vaddq_f32(_p, _q);
                    vst1q_f32(outptr, _p);

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
#endif // __ARM_NEON
                for (; i < size; i++)
                {
                    *outptr = *ptr + *ptr1;

                    ptr++;
                    ptr1++;
                    outptr++;
                }
            }

            for (size_t b = 2; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob.channel(q);

                    int i = 0;
#if __ARM_NEON
                    for (; i + 7 < size; i += 8)
                    {
                        float32x4_t _p0 = vld1q_f32(outptr);
                        float32x4_t _p1 = vld1q_f32(outptr + 4);
                        float32x4_t _q0 = vld1q_f32(ptr);
                        float32x4_t _q1 = vld1q_f32(ptr + 4);
                        _p0 = vaddq_f32(_p0, _q0);
                        _p1 = vaddq_f32(_p1, _q1);
                        vst1q_f32(outptr, _p0);
                        vst1q_f32(outptr + 4, _p1);

                        ptr += 8;
                        outptr += 8;
                    }
                    for (; i + 3 < size; i += 4)
                    {
                        float32x4_t _p = vld1q_f32(outptr);
                        float32x4_t _q = vld1q_f32(ptr);
                        _p = vaddq_f32(_p, _q);
                        vst1q_f32(outptr, _p);

                        ptr += 4;
                        outptr += 4;
                    }
#endif // __ARM_NEON
                    for (; i < size; i++)
                    {
                        *outptr += *ptr;

                        ptr++;
                        outptr++;
                    }
                }
            }
        }
        else
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                const float coeff0 = coeffs[0];
                const float coeff1 = coeffs[1];

                int i = 0;
#if __ARM_NEON
                float32x4_t _coeff0 = vdupq_n_f32(coeff0);
                float32x4_t _coeff1 = vdupq_n_f32(coeff1);
                for (; i + 7 < size; i += 8)
                {
                    float32x4_t _p0 = vld1q_f32(ptr);
                    float32x4_t _p1 = vld1q_f32(ptr + 4);
                    float32x4_t _q0 = vld1q_f32(ptr1);
                    float32x4_t _q1 = vld1q_f32(ptr1 + 4);
                    _p0 = vmulq_f32(_p0, _coeff0);
                    _p1 = vmulq_f32(_p1, _coeff0);
                    _p0 = vmlaq_f32(_p0, _q0, _coeff1);
                    _p1 = vmlaq_f32(_p1, _q1, _coeff1);
                    vst1q_f32(outptr, _p0);
                    vst1q_f32(outptr + 4, _p1);

                    ptr += 8;
                    ptr1 += 8;
                    outptr += 8;
                }
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _q = vld1q_f32(ptr1);
                    _p = vmulq_f32(_p, _coeff0);
                    _p = vmlaq_f32(_p, _q, _coeff1);
                    vst1q_f32(outptr, _p);

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
#endif // __ARM_NEON
                for (; i < size; i++)
                {
                    *outptr = *ptr * coeff0 + *ptr1 * coeff1;

                    ptr++;
                    ptr1++;
                    outptr++;
                }
            }

            for (size_t b = 2; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob.channel(q);

                    const float coeff = coeffs[b];

                    int i = 0;
#if __ARM_NEON
                    float32x4_t _coeff = vdupq_n_f32(coeff);
                    for (; i + 7 < size; i += 8)
                    {
                        float32x4_t _p0 = vld1q_f32(outptr);
                        float32x4_t _p1 = vld1q_f32(outptr + 4);
                        float32x4_t _q0 = vld1q_f32(ptr);
                        float32x4_t _q1 = vld1q_f32(ptr + 4);
                        _p0 = vmlaq_f32(_p0, _q0, _coeff);
                        _p1 = vmlaq_f32(_p1, _q1, _coeff);
                        vst1q_f32(outptr, _p0);
                        vst1q_f32(outptr + 4, _p1);

                        ptr += 8;
                        outptr += 8;
                    }
                    for (; i + 3 < size; i += 4)
                    {
                        float32x4_t _p = vld1q_f32(outptr);
                        float32x4_t _q = vld1q_f32(ptr);
                        _p = vmlaq_f32(_p, _q, _coeff);
                        vst1q_f32(outptr, _p);

                        ptr += 4;
                        outptr += 4;
                    }
#endif // __ARM_NEON
                    for (; i < size; i++)
                    {
                        *outptr += *ptr * coeff;

                        ptr++;
                        outptr++;
                    }
                }
            }
        }
    }
    if (op_type == Operation_MAX)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

            int i = 0;
#if __ARM_NEON
            for (; i + 7 < size; i += 8)
            {
                float32x4_t _p0 = vld1q_f32(ptr);
                float32x4_t _p1 = vld1q_f32(ptr + 4);
                float32x4_t _q0 = vld1q_f32(ptr1);
                float32x4_t _q1 = vld1q_f32(ptr1 + 4);
                _p0 = vmaxq_f32(_p0, _q0);
                _p1 = vmaxq_f32(_p1, _q1);
                vst1q_f32(outptr, _p0);
                vst1q_f32(outptr + 4, _p1);

                ptr += 8;
                ptr1 += 8;
                outptr += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = vld1q_f32(ptr);
                float32x4_t _q = vld1q_f32(ptr1);
                _p = vmaxq_f32(_p, _q);
                vst1q_f32(outptr, _p);

                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
#endif // __ARM_NEON
            for (; i < size; i++)
            {
                *outptr = std::max(*ptr, *ptr1);

                ptr++;
                ptr1++;
                outptr++;
            }
        }

        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                int i = 0;
#if __ARM_NEON
                for (; i + 7 < size; i += 8)
                {
                    float32x4_t _p0 = vld1q_f32(outptr);
                    float32x4_t _p1 = vld1q_f32(outptr + 4);
                    float32x4_t _q0 = vld1q_f32(ptr);
                    float32x4_t _q1 = vld1q_f32(ptr + 4);
                    _p0 = vmaxq_f32(_p0, _q0);
                    _p1 = vmaxq_f32(_p1, _q1);
                    vst1q_f32(outptr, _p0);
                    vst1q_f32(outptr + 4, _p1);

                    ptr += 8;
                    outptr += 8;
                }
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _p = vld1q_f32(outptr);
                    float32x4_t _q = vld1q_f32(ptr);
                    _p = vmaxq_f32(_p, _q);
                    vst1q_f32(outptr, _p);

                    ptr += 4;
                    outptr += 4;
                }
#endif // __ARM_NEON
                for (; i < size; i++)
                {
                    *outptr = std::max(*ptr, *outptr);

                    ptr++;
                    outptr++;
                }
            }
        }
    }

    return 0;
}

#if NCNN_BF16
int Eltwise_arm::forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int elempack = bottom_blob.elempack;
    int size = w * h * d * elempack;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (bottom_blobs.size() == 2)
    {
        // fast path without fp32 accumulator
        if (op_type == Operation_PROD)
        {
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                int i = 0;
#if __ARM_NEON
                for (; i + 7 < size; i += 8)
                {
                    uint16x8_t _p01 = vld1q_u16(ptr);
                    uint16x8_t _q01 = vld1q_u16(ptr1);
                    float32x4_t _p0 = bfloat2float(vget_low_u16(_p01));
                    float32x4_t _p1 = bfloat2float(vget_high_u16(_p01));
                    float32x4_t _q0 = bfloat2float(vget_low_u16(_q01));
                    float32x4_t _q1 = bfloat2float(vget_high_u16(_q01));
                    _p0 = vmulq_f32(_p0, _q0);
                    _p1 = vmulq_f32(_p1, _q1);
                    vst1q_u16(outptr, vcombine_u16(float2bfloat(_p0), float2bfloat(_p1)));

                    ptr += 8;
                    ptr1 += 8;
                    outptr += 8;
                }
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _p = bfloat2float(vld1_u16(ptr));
                    float32x4_t _q = bfloat2float(vld1_u16(ptr1));
                    _p = vmulq_f32(_p, _q);
                    vst1_u16(outptr, float2bfloat(_p));

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
#endif // __ARM_NEON
                for (; i < size; i++)
                {
                    *outptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * bfloat16_to_float32(*ptr1));

                    ptr++;
                    ptr1++;
                    outptr++;
                }
            }
        }
        if (op_type == Operation_SUM)
        {
            if (coeffs.w == 0)
            {
                const Mat& bottom_blob1 = bottom_blobs[1];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const unsigned short* ptr = bottom_blob.channel(q);
                    const unsigned short* ptr1 = bottom_blob1.channel(q);
                    unsigned short* outptr = top_blob.channel(q);

                    int i = 0;
#if __ARM_NEON
                    for (; i + 7 < size; i += 8)
                    {
                        uint16x8_t _p01 = vld1q_u16(ptr);
                        uint16x8_t _q01 = vld1q_u16(ptr1);
                        float32x4_t _p0 = bfloat2float(vget_low_u16(_p01));
                        float32x4_t _p1 = bfloat2float(vget_high_u16(_p01));
                        float32x4_t _q0 = bfloat2float(vget_low_u16(_q01));
                        float32x4_t _q1 = bfloat2float(vget_high_u16(_q01));
                        _p0 = vaddq_f32(_p0, _q0);
                        _p1 = vaddq_f32(_p1, _q1);
                        vst1q_u16(outptr, vcombine_u16(float2bfloat(_p0), float2bfloat(_p1)));

                        ptr += 8;
                        ptr1 += 8;
                        outptr += 8;
                    }
                    for (; i + 3 < size; i += 4)
                    {
                        float32x4_t _p = bfloat2float(vld1_u16(ptr));
                        float32x4_t _q = bfloat2float(vld1_u16(ptr1));
                        _p = vaddq_f32(_p, _q);
                        vst1_u16(outptr, float2bfloat(_p));

                        ptr += 4;
                        ptr1 += 4;
                        outptr += 4;
                    }
#endif // __ARM_NEON
                    for (; i < size; i++)
                    {
                        *outptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) + bfloat16_to_float32(*ptr1));

                        ptr++;
                        ptr1++;
                        outptr++;
                    }
                }
            }
            else
            {
                const Mat& bottom_blob1 = bottom_blobs[1];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const unsigned short* ptr = bottom_blob.channel(q);
                    const unsigned short* ptr1 = bottom_blob1.channel(q);
                    unsigned short* outptr = top_blob.channel(q);

                    const float coeff0 = coeffs[0];
                    const float coeff1 = coeffs[1];

                    int i = 0;
#if __ARM_NEON
                    float32x4_t _coeff0 = vdupq_n_f32(coeff0);
                    float32x4_t _coeff1 = vdupq_n_f32(coeff1);
                    for (; i + 7 < size; i += 8)
                    {
                        uint16x8_t _p01 = vld1q_u16(ptr);
                        uint16x8_t _q01 = vld1q_u16(ptr1);
                        float32x4_t _p0 = bfloat2float(vget_low_u16(_p01));
                        float32x4_t _p1 = bfloat2float(vget_high_u16(_p01));
                        float32x4_t _q0 = bfloat2float(vget_low_u16(_q01));
                        float32x4_t _q1 = bfloat2float(vget_high_u16(_q01));
                        _p0 = vmulq_f32(_p0, _coeff0);
                        _p1 = vmulq_f32(_p1, _coeff0);
                        _p0 = vmlaq_f32(_p0, _q0, _coeff1);
                        _p1 = vmlaq_f32(_p1, _q1, _coeff1);
                        vst1q_u16(outptr, vcombine_u16(float2bfloat(_p0), float2bfloat(_p1)));

                        ptr += 8;
                        ptr1 += 8;
                        outptr += 8;
                    }
                    for (; i + 3 < size; i += 4)
                    {
                        float32x4_t _p = bfloat2float(vld1_u16(ptr));
                        float32x4_t _q = bfloat2float(vld1_u16(ptr1));
                        _p = vmulq_f32(_p, _coeff0);
                        _p = vmlaq_f32(_p, _q, _coeff1);
                        vst1_u16(outptr, float2bfloat(_p));

                        ptr += 4;
                        ptr1 += 4;
                        outptr += 4;
                    }
#endif // __ARM_NEON
                    for (; i < size; i++)
                    {
                        *outptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * coeff0 + bfloat16_to_float32(*ptr1) * coeff1);

                        ptr++;
                        ptr1++;
                        outptr++;
                    }
                }
            }
        }
        if (op_type == Operation_MAX)
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                int i = 0;
#if __ARM_NEON
                for (; i + 7 < size; i += 8)
                {
                    uint16x8_t _p01 = vld1q_u16(ptr);
                    uint16x8_t _q01 = vld1q_u16(ptr1);
                    float32x4_t _p0 = bfloat2float(vget_low_u16(_p01));
                    float32x4_t _p1 = bfloat2float(vget_high_u16(_p01));
                    float32x4_t _q0 = bfloat2float(vget_low_u16(_q01));
                    float32x4_t _q1 = bfloat2float(vget_high_u16(_q01));
                    _p0 = vmaxq_f32(_p0, _q0);
                    _p1 = vmaxq_f32(_p1, _q1);
                    vst1q_u16(outptr, vcombine_u16(float2bfloat(_p0), float2bfloat(_p1)));

                    ptr += 8;
                    ptr1 += 8;
                    outptr += 8;
                }
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _p = bfloat2float(vld1_u16(ptr));
                    float32x4_t _q = bfloat2float(vld1_u16(ptr1));
                    _p = vmaxq_f32(_p, _q);
                    vst1_u16(outptr, float2bfloat(_p));

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
#endif // __ARM_NEON
                for (; i < size; i++)
                {
                    *outptr = float32_to_bfloat16(std::max(bfloat16_to_float32(*ptr), bfloat16_to_float32(*ptr1)));

                    ptr++;
                    ptr1++;
                    outptr++;
                }
            }
        }

        return 0;
    }

    Mat top_blob_fp32(w, h, d, channels, (size_t)4u * elempack, elempack, opt.workspace_allocator);
    if (top_blob_fp32.empty())
        return -100;

    if (op_type == Operation_PROD)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);
            const unsigned short* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob_fp32.channel(q);

            int i = 0;
#if __ARM_NEON
            for (; i + 7 < size; i += 8)
            {
                uint16x8_t _p01 = vld1q_u16(ptr);
                uint16x8_t _q01 = vld1q_u16(ptr1);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p01));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p01));
                float32x4_t _q0 = bfloat2float(vget_low_u16(_q01));
                float32x4_t _q1 = bfloat2float(vget_high_u16(_q01));
                _p0 = vmulq_f32(_p0, _q0);
                _p1 = vmulq_f32(_p1, _q1);
                vst1q_f32(outptr, _p0);
                vst1q_f32(outptr + 4, _p1);

                ptr += 8;
                ptr1 += 8;
                outptr += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = bfloat2float(vld1_u16(ptr));
                float32x4_t _q = bfloat2float(vld1_u16(ptr1));
                _p = vmulq_f32(_p, _q);
                vst1q_f32(outptr, _p);

                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
#endif // __ARM_NEON
            for (; i < size; i++)
            {
                *outptr = bfloat16_to_float32(*ptr) * bfloat16_to_float32(*ptr1);

                ptr++;
                ptr1++;
                outptr++;
            }
        }

        size_t b = 2;
        for (; b < bottom_blobs.size() - 1; b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob1.channel(q);
                float* outptr = top_blob_fp32.channel(q);

                int i = 0;
#if __ARM_NEON
                for (; i + 7 < size; i += 8)
                {
                    float32x4_t _p0 = vld1q_f32(outptr);
                    float32x4_t _p1 = vld1q_f32(outptr + 4);
                    uint16x8_t _q01 = vld1q_u16(ptr);
                    float32x4_t _q0 = bfloat2float(vget_low_u16(_q01));
                    float32x4_t _q1 = bfloat2float(vget_high_u16(_q01));
                    _p0 = vmulq_f32(_p0, _q0);
                    _p1 = vmulq_f32(_p1, _q1);
                    vst1q_f32(outptr, _p0);
                    vst1q_f32(outptr + 4, _p1);

                    ptr += 8;
                    outptr += 8;
                }
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _p = vld1q_f32(outptr);
                    float32x4_t _q = bfloat2float(vld1_u16(ptr));
                    _p = vmulq_f32(_p, _q);
                    vst1q_f32(outptr, _p);

                    ptr += 4;
                    outptr += 4;
                }
#endif // __ARM_NEON
                for (; i < size; i++)
                {
                    *outptr *= bfloat16_to_float32(*ptr);

                    ptr++;
                    outptr++;
                }
            }
        }
        for (; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob1.channel(q);
                const float* ptr0 = top_blob_fp32.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                int i = 0;
#if __ARM_NEON
                for (; i + 7 < size; i += 8)
                {
                    float32x4_t _p0 = vld1q_f32(ptr0);
                    float32x4_t _p1 = vld1q_f32(ptr0 + 4);
                    uint16x8_t _q01 = vld1q_u16(ptr);
                    float32x4_t _q0 = bfloat2float(vget_low_u16(_q01));
                    float32x4_t _q1 = bfloat2float(vget_high_u16(_q01));
                    _p0 = vmulq_f32(_p0, _q0);
                    _p1 = vmulq_f32(_p1, _q1);
                    vst1q_u16(outptr, vcombine_u16(float2bfloat(_p0), float2bfloat(_p1)));

                    ptr += 8;
                    ptr0 += 8;
                    outptr += 8;
                }
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _p = vld1q_f32(ptr0);
                    float32x4_t _q = bfloat2float(vld1_u16(ptr));
                    _p = vmulq_f32(_p, _q);
                    vst1_u16(outptr, float2bfloat(_p));

                    ptr += 4;
                    ptr0 += 4;
                    outptr += 4;
                }
#endif // __ARM_NEON
                for (; i < size; i++)
                {
                    *outptr = float32_to_bfloat16(*ptr0 * bfloat16_to_float32(*ptr));

                    ptr++;
                    ptr0++;
                    outptr++;
                }
            }
        }
    }
    if (op_type == Operation_SUM)
    {
        if (coeffs.w == 0)
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob_fp32.channel(q);

                int i = 0;
#if __ARM_NEON
                for (; i + 7 < size; i += 8)
                {
                    uint16x8_t _p01 = vld1q_u16(ptr);
                    uint16x8_t _q01 = vld1q_u16(ptr1);
                    float32x4_t _p0 = bfloat2float(vget_low_u16(_p01));
                    float32x4_t _p1 = bfloat2float(vget_high_u16(_p01));
                    float32x4_t _q0 = bfloat2float(vget_low_u16(_q01));
                    float32x4_t _q1 = bfloat2float(vget_high_u16(_q01));
                    _p0 = vaddq_f32(_p0, _q0);
                    _p1 = vaddq_f32(_p1, _q1);
                    vst1q_f32(outptr, _p0);
                    vst1q_f32(outptr + 4, _p1);

                    ptr += 8;
                    ptr1 += 8;
                    outptr += 8;
                }
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _p = bfloat2float(vld1_u16(ptr));
                    float32x4_t _q = bfloat2float(vld1_u16(ptr1));
                    _p = vaddq_f32(_p, _q);
                    vst1q_f32(outptr, _p);

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
#endif // __ARM_NEON
                for (; i < size; i++)
                {
                    *outptr = bfloat16_to_float32(*ptr) + bfloat16_to_float32(*ptr1);

                    ptr++;
                    ptr1++;
                    outptr++;
                }
            }

            size_t b = 2;
            for (; b < bottom_blobs.size() - 1; b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const unsigned short* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob_fp32.channel(q);

                    int i = 0;
#if __ARM_NEON
                    for (; i + 7 < size; i += 8)
                    {
                        float32x4_t _p0 = vld1q_f32(outptr);
                        float32x4_t _p1 = vld1q_f32(outptr + 4);
                        uint16x8_t _q01 = vld1q_u16(ptr);
                        float32x4_t _q0 = bfloat2float(vget_low_u16(_q01));
                        float32x4_t _q1 = bfloat2float(vget_high_u16(_q01));
                        _p0 = vaddq_f32(_p0, _q0);
                        _p1 = vaddq_f32(_p1, _q1);
                        vst1q_f32(outptr, _p0);
                        vst1q_f32(outptr + 4, _p1);

                        ptr += 8;
                        outptr += 8;
                    }
                    for (; i + 3 < size; i += 4)
                    {
                        float32x4_t _p = vld1q_f32(outptr);
                        float32x4_t _q = bfloat2float(vld1_u16(ptr));
                        _p = vaddq_f32(_p, _q);
                        vst1q_f32(outptr, _p);

                        ptr += 4;
                        outptr += 4;
                    }
#endif // __ARM_NEON
                    for (; i < size; i++)
                    {
                        *outptr += bfloat16_to_float32(*ptr);

                        ptr++;
                        outptr++;
                    }
                }
            }
            for (; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const unsigned short* ptr = bottom_blob1.channel(q);
                    const float* ptr0 = top_blob_fp32.channel(q);
                    unsigned short* outptr = top_blob.channel(q);

                    int i = 0;
#if __ARM_NEON
                    for (; i + 7 < size; i += 8)
                    {
                        float32x4_t _p0 = vld1q_f32(ptr0);
                        float32x4_t _p1 = vld1q_f32(ptr0 + 4);
                        uint16x8_t _q01 = vld1q_u16(ptr);
                        float32x4_t _q0 = bfloat2float(vget_low_u16(_q01));
                        float32x4_t _q1 = bfloat2float(vget_high_u16(_q01));
                        _p0 = vaddq_f32(_p0, _q0);
                        _p1 = vaddq_f32(_p1, _q1);
                        vst1q_u16(outptr, vcombine_u16(float2bfloat(_p0), float2bfloat(_p1)));

                        ptr += 8;
                        ptr0 += 8;
                        outptr += 8;
                    }
                    for (; i + 3 < size; i += 4)
                    {
                        float32x4_t _p = vld1q_f32(ptr0);
                        float32x4_t _q = bfloat2float(vld1_u16(ptr));
                        _p = vaddq_f32(_p, _q);
                        vst1_u16(outptr, float2bfloat(_p));

                        ptr += 4;
                        ptr0 += 4;
                        outptr += 4;
                    }
#endif // __ARM_NEON
                    for (; i < size; i++)
                    {
                        *outptr = float32_to_bfloat16(*ptr0 + bfloat16_to_float32(*ptr));

                        ptr++;
                        ptr0++;
                        outptr++;
                    }
                }
            }
        }
        else
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob_fp32.channel(q);

                const float coeff0 = coeffs[0];
                const float coeff1 = coeffs[1];

                int i = 0;
#if __ARM_NEON
                float32x4_t _coeff0 = vdupq_n_f32(coeff0);
                float32x4_t _coeff1 = vdupq_n_f32(coeff1);
                for (; i + 7 < size; i += 8)
                {
                    uint16x8_t _p01 = vld1q_u16(ptr);
                    uint16x8_t _q01 = vld1q_u16(ptr1);
                    float32x4_t _p0 = bfloat2float(vget_low_u16(_p01));
                    float32x4_t _p1 = bfloat2float(vget_high_u16(_p01));
                    float32x4_t _q0 = bfloat2float(vget_low_u16(_q01));
                    float32x4_t _q1 = bfloat2float(vget_high_u16(_q01));
                    _p0 = vmulq_f32(_p0, _coeff0);
                    _p1 = vmulq_f32(_p1, _coeff0);
                    _p0 = vmlaq_f32(_p0, _q0, _coeff1);
                    _p1 = vmlaq_f32(_p1, _q1, _coeff1);
                    vst1q_f32(outptr, _p0);
                    vst1q_f32(outptr + 4, _p1);

                    ptr += 8;
                    ptr1 += 8;
                    outptr += 8;
                }
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _p = bfloat2float(vld1_u16(ptr));
                    float32x4_t _q = bfloat2float(vld1_u16(ptr1));
                    _p = vmulq_f32(_p, _coeff0);
                    _p = vmlaq_f32(_p, _q, _coeff1);
                    vst1q_f32(outptr, _p);

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
#endif // __ARM_NEON
                for (; i < size; i++)
                {
                    *outptr = bfloat16_to_float32(*ptr) * coeff0 + bfloat16_to_float32(*ptr1) * coeff1;

                    ptr++;
                    ptr1++;
                    outptr++;
                }
            }

            size_t b = 2;
            for (; b < bottom_blobs.size() - 1; b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const unsigned short* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob_fp32.channel(q);

                    const float coeff = coeffs[b];

                    int i = 0;
#if __ARM_NEON
                    float32x4_t _coeff = vdupq_n_f32(coeff);
                    for (; i + 7 < size; i += 8)
                    {
                        float32x4_t _p0 = vld1q_f32(outptr);
                        float32x4_t _p1 = vld1q_f32(outptr + 4);
                        uint16x8_t _q01 = vld1q_u16(ptr);
                        float32x4_t _q0 = bfloat2float(vget_low_u16(_q01));
                        float32x4_t _q1 = bfloat2float(vget_high_u16(_q01));
                        _p0 = vmlaq_f32(_p0, _q0, _coeff);
                        _p1 = vmlaq_f32(_p1, _q1, _coeff);
                        vst1q_f32(outptr, _p0);
                        vst1q_f32(outptr + 4, _p1);

                        ptr += 8;
                        outptr += 8;
                    }
                    for (; i + 3 < size; i += 4)
                    {
                        float32x4_t _p = vld1q_f32(outptr);
                        float32x4_t _q = bfloat2float(vld1_u16(ptr));
                        _p = vmlaq_f32(_p, _q, _coeff);
                        vst1q_f32(outptr, _p);

                        ptr += 4;
                        outptr += 4;
                    }
#endif // __ARM_NEON
                    for (; i < size; i++)
                    {
                        *outptr += bfloat16_to_float32(*ptr) * coeff;

                        ptr++;
                        outptr++;
                    }
                }
            }
            for (; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const unsigned short* ptr = bottom_blob1.channel(q);
                    const float* ptr0 = top_blob_fp32.channel(q);
                    unsigned short* outptr = top_blob.channel(q);

                    const float coeff = coeffs[b];

                    int i = 0;
#if __ARM_NEON
                    float32x4_t _coeff = vdupq_n_f32(coeff);
                    for (; i + 7 < size; i += 8)
                    {
                        float32x4_t _p0 = vld1q_f32(ptr0);
                        float32x4_t _p1 = vld1q_f32(ptr0 + 4);
                        uint16x8_t _q01 = vld1q_u16(ptr);
                        float32x4_t _q0 = bfloat2float(vget_low_u16(_q01));
                        float32x4_t _q1 = bfloat2float(vget_high_u16(_q01));
                        _p0 = vmlaq_f32(_p0, _q0, _coeff);
                        _p1 = vmlaq_f32(_p1, _q1, _coeff);
                        vst1q_u16(outptr, vcombine_u16(float2bfloat(_p0), float2bfloat(_p1)));

                        ptr += 8;
                        ptr0 += 8;
                        outptr += 8;
                    }
                    for (; i + 3 < size; i += 4)
                    {
                        float32x4_t _p = vld1q_f32(ptr0);
                        float32x4_t _q = bfloat2float(vld1_u16(ptr));
                        _p = vmlaq_f32(_p, _q, _coeff);
                        vst1_u16(outptr, float2bfloat(_p));

                        ptr += 4;
                        ptr0 += 4;
                        outptr += 4;
                    }
#endif // __ARM_NEON
                    for (; i < size; i++)
                    {
                        *outptr = float32_to_bfloat16(*ptr0 + bfloat16_to_float32(*ptr) * coeff);

                        ptr++;
                        ptr0++;
                        outptr++;
                    }
                }
            }
        }
    }
    if (op_type == Operation_MAX)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);
            const unsigned short* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob_fp32.channel(q);

            int i = 0;
#if __ARM_NEON
            for (; i + 7 < size; i += 8)
            {
                uint16x8_t _p01 = vld1q_u16(ptr);
                uint16x8_t _q01 = vld1q_u16(ptr1);
                float32x4_t _p0 = bfloat2float(vget_low_u16(_p01));
                float32x4_t _p1 = bfloat2float(vget_high_u16(_p01));
                float32x4_t _q0 = bfloat2float(vget_low_u16(_q01));
                float32x4_t _q1 = bfloat2float(vget_high_u16(_q01));
                _p0 = vmaxq_f32(_p0, _q0);
                _p1 = vmaxq_f32(_p1, _q1);
                vst1q_f32(outptr, _p0);
                vst1q_f32(outptr + 4, _p1);

                ptr += 8;
                ptr1 += 8;
                outptr += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = bfloat2float(vld1_u16(ptr));
                float32x4_t _q = bfloat2float(vld1_u16(ptr1));
                _p = vmaxq_f32(_p, _q);
                vst1q_f32(outptr, _p);

                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
#endif // __ARM_NEON
            for (; i < size; i++)
            {
                *outptr = std::max(bfloat16_to_float32(*ptr), bfloat16_to_float32(*ptr1));

                ptr++;
                ptr1++;
                outptr++;
            }
        }

        size_t b = 2;
        for (; b < bottom_blobs.size() - 1; b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob1.channel(q);
                float* outptr = top_blob_fp32.channel(q);

                int i = 0;
#if __ARM_NEON
                for (; i + 7 < size; i += 8)
                {
                    float32x4_t _p0 = vld1q_f32(outptr);
                    float32x4_t _p1 = vld1q_f32(outptr + 4);
                    uint16x8_t _q01 = vld1q_u16(ptr);
                    float32x4_t _q0 = bfloat2float(vget_low_u16(_q01));
                    float32x4_t _q1 = bfloat2float(vget_high_u16(_q01));
                    _p0 = vmaxq_f32(_p0, _q0);
                    _p1 = vmaxq_f32(_p1, _q1);
                    vst1q_f32(outptr, _p0);
                    vst1q_f32(outptr + 4, _p1);

                    ptr += 8;
                    outptr += 8;
                }
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _p = vld1q_f32(outptr);
                    float32x4_t _q = bfloat2float(vld1_u16(ptr));
                    _p = vmaxq_f32(_p, _q);
                    vst1q_f32(outptr, _p);

                    ptr += 4;
                    outptr += 4;
                }
#endif // __ARM_NEON
                for (; i < size; i++)
                {
                    *outptr = std::max(bfloat16_to_float32(*ptr), *outptr);

                    ptr++;
                    outptr++;
                }
            }
        }
        for (; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob1.channel(q);
                const float* ptr0 = top_blob_fp32.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                int i = 0;
#if __ARM_NEON
                for (; i + 7 < size; i += 8)
                {
                    float32x4_t _p0 = vld1q_f32(ptr0);
                    float32x4_t _p1 = vld1q_f32(ptr0 + 4);
                    uint16x8_t _q01 = vld1q_u16(ptr);
                    float32x4_t _q0 = bfloat2float(vget_low_u16(_q01));
                    float32x4_t _q1 = bfloat2float(vget_high_u16(_q01));
                    _p0 = vmaxq_f32(_p0, _q0);
                    _p1 = vmaxq_f32(_p1, _q1);
                    vst1q_u16(outptr, vcombine_u16(float2bfloat(_p0), float2bfloat(_p1)));

                    ptr += 8;
                    ptr0 += 8;
                    outptr += 8;
                }
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _p = vld1q_f32(ptr0);
                    float32x4_t _q = bfloat2float(vld1_u16(ptr));
                    _p = vmaxq_f32(_p, _q);
                    vst1_u16(outptr, float2bfloat(_p));

                    ptr += 4;
                    ptr0 += 4;
                    outptr += 4;
                }
#endif // __ARM_NEON
                for (; i < size; i++)
                {
                    *outptr = float32_to_bfloat16(std::max(bfloat16_to_float32(*ptr), *ptr0));

                    ptr++;
                    ptr0++;
                    outptr++;
                }
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
