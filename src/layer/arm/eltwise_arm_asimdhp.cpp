// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int Eltwise_arm::forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int elempack = bottom_blob.elempack;
    int size = w * h;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (bottom_blobs.size() == 2)
    {
        // fast path without fp32 accumulator
        if (elempack == 8)
        {
            if (op_type == Operation_PROD)
            {
                const Mat& bottom_blob1 = bottom_blobs[1];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob.channel(q);
                    const __fp16* ptr1 = bottom_blob1.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        float16x8_t _p = vld1q_f16(ptr);
                        float16x8_t _p1 = vld1q_f16(ptr1);
                        _p = vmulq_f16(_p, _p1);
                        vst1q_f16(outptr, _p);

                        ptr += 8;
                        ptr1 += 8;
                        outptr += 8;
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
                        const __fp16* ptr = bottom_blob.channel(q);
                        const __fp16* ptr1 = bottom_blob1.channel(q);
                        __fp16* outptr = top_blob.channel(q);

                        for (int i = 0; i < size; i++)
                        {
                            float16x8_t _p = vld1q_f16(ptr);
                            float16x8_t _p1 = vld1q_f16(ptr1);
                            _p = vaddq_f16(_p, _p1);
                            vst1q_f16(outptr, _p);

                            ptr += 8;
                            ptr1 += 8;
                            outptr += 8;
                        }
                    }
                }
                else
                {
                    const Mat& bottom_blob1 = bottom_blobs[1];
                    float16x8_t _coeff0 = vdupq_n_f16((__fp16)coeffs[0]);
                    float16x8_t _coeff1 = vdupq_n_f16((__fp16)coeffs[1]);
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const __fp16* ptr = bottom_blob.channel(q);
                        const __fp16* ptr1 = bottom_blob1.channel(q);
                        __fp16* outptr = top_blob.channel(q);

                        for (int i = 0; i < size; i++)
                        {
                            float16x8_t _p = vld1q_f16(ptr);
                            float16x8_t _p1 = vld1q_f16(ptr1);
                            _p = vmulq_f16(_p, _coeff0);
                            _p = vfmaq_f16(_p, _p1, _coeff1);
                            vst1q_f16(outptr, _p);

                            ptr += 8;
                            ptr1 += 8;
                            outptr += 8;
                        }
                    }
                }
            }
            if (op_type == Operation_MAX)
            {
                const Mat& bottom_blob1 = bottom_blobs[1];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob.channel(q);
                    const __fp16* ptr1 = bottom_blob1.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        float16x8_t _p = vld1q_f16(ptr);
                        float16x8_t _p1 = vld1q_f16(ptr1);
                        _p = vmaxq_f16(_p, _p1);
                        vst1q_f16(outptr, _p);

                        ptr += 8;
                        ptr1 += 8;
                        outptr += 8;
                    }
                }
            }

            return 0;
        }

        if (elempack == 4)
        {
            if (op_type == Operation_PROD)
            {
                const Mat& bottom_blob1 = bottom_blobs[1];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob.channel(q);
                    const __fp16* ptr1 = bottom_blob1.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        float16x4_t _p = vld1_f16(ptr);
                        float16x4_t _p1 = vld1_f16(ptr1);
                        _p = vmul_f16(_p, _p1);
                        vst1_f16(outptr, _p);

                        ptr += 4;
                        ptr1 += 4;
                        outptr += 4;
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
                        const __fp16* ptr = bottom_blob.channel(q);
                        const __fp16* ptr1 = bottom_blob1.channel(q);
                        __fp16* outptr = top_blob.channel(q);

                        for (int i = 0; i < size; i++)
                        {
                            float16x4_t _p = vld1_f16(ptr);
                            float16x4_t _p1 = vld1_f16(ptr1);
                            _p = vadd_f16(_p, _p1);
                            vst1_f16(outptr, _p);

                            ptr += 4;
                            ptr1 += 4;
                            outptr += 4;
                        }
                    }
                }
                else
                {
                    const Mat& bottom_blob1 = bottom_blobs[1];
                    float16x4_t _coeff0 = vdup_n_f16((__fp16)coeffs[0]);
                    float16x4_t _coeff1 = vdup_n_f16((__fp16)coeffs[1]);
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const __fp16* ptr = bottom_blob.channel(q);
                        const __fp16* ptr1 = bottom_blob1.channel(q);
                        __fp16* outptr = top_blob.channel(q);

                        for (int i = 0; i < size; i++)
                        {
                            float16x4_t _p = vld1_f16(ptr);
                            float16x4_t _p1 = vld1_f16(ptr1);
                            _p = vmul_f16(_p, _coeff0);
                            _p = vfma_f16(_p, _p1, _coeff1);
                            vst1_f16(outptr, _p);

                            ptr += 4;
                            ptr1 += 4;
                            outptr += 4;
                        }
                    }
                }
            }
            if (op_type == Operation_MAX)
            {
                const Mat& bottom_blob1 = bottom_blobs[1];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob.channel(q);
                    const __fp16* ptr1 = bottom_blob1.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        float16x4_t _p = vld1_f16(ptr);
                        float16x4_t _p1 = vld1_f16(ptr1);
                        _p = vmax_f16(_p, _p1);
                        vst1_f16(outptr, _p);

                        ptr += 4;
                        ptr1 += 4;
                        outptr += 4;
                    }
                }
            }

            return 0;
        }

        if (op_type == Operation_PROD)
        {
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = bottom_blob.channel(q);
                const __fp16* ptr1 = bottom_blob1.channel(q);
                __fp16* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    *outptr = *ptr * *ptr1;

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
                    const __fp16* ptr = bottom_blob.channel(q);
                    const __fp16* ptr1 = bottom_blob1.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        *outptr = *ptr + *ptr1;

                        ptr++;
                        ptr1++;
                        outptr++;
                    }
                }
            }
            else
            {
                const Mat& bottom_blob1 = bottom_blobs[1];
                float coeff0 = coeffs[0];
                float coeff1 = coeffs[1];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob.channel(q);
                    const __fp16* ptr1 = bottom_blob1.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        *outptr = (__fp16)((float)(*ptr) * coeff0 + (float)(*ptr1) * coeff1);

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
                const __fp16* ptr = bottom_blob.channel(q);
                const __fp16* ptr1 = bottom_blob1.channel(q);
                __fp16* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    *outptr = std::max(*ptr, *ptr1);

                    ptr++;
                    ptr1++;
                    outptr++;
                }
            }
        }

        return 0;
    }

    if (op_type == Operation_MAX)
    {
        return forward_fp16sa(bottom_blobs, top_blobs, opt);
    }

    Mat top_blob_fp32(w, h, channels, (size_t)4u * elempack, elempack, opt.workspace_allocator);
    if (top_blob_fp32.empty())
        return -100;

    if (elempack == 4)
    {
        if (op_type == Operation_PROD)
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = bottom_blob.channel(q);
                const __fp16* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob_fp32.channel(q);

                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                    float32x4_t _p1 = vcvt_f32_f16(vld1_f16(ptr1));
                    _p = vmulq_f32(_p, _p1);
                    vst1q_f32(outptr, _p);

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
            }

            size_t b = 2;
            for (; b < bottom_blobs.size() - 1; b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob_fp32.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _p = vld1q_f32(outptr);
                        float32x4_t _p1 = vcvt_f32_f16(vld1_f16(ptr));
                        _p = vmulq_f32(_p, _p1);
                        vst1q_f32(outptr, _p);

                        ptr += 4;
                        outptr += 4;
                    }
                }
            }
            for (; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob1.channel(q);
                    const float* ptr0 = top_blob_fp32.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _p = vld1q_f32(ptr0);
                        float32x4_t _p1 = vcvt_f32_f16(vld1_f16(ptr));
                        _p = vmulq_f32(_p, _p1);
                        vst1_f16(outptr, vcvt_f16_f32(_p));

                        ptr += 4;
                        ptr0 += 4;
                        outptr += 4;
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
                    const __fp16* ptr = bottom_blob.channel(q);
                    const __fp16* ptr1 = bottom_blob1.channel(q);
                    float* outptr = top_blob_fp32.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                        float32x4_t _p1 = vcvt_f32_f16(vld1_f16(ptr1));
                        _p = vaddq_f32(_p, _p1);
                        vst1q_f32(outptr, _p);

                        ptr += 4;
                        ptr1 += 4;
                        outptr += 4;
                    }
                }

                size_t b = 2;
                for (; b < bottom_blobs.size() - 1; b++)
                {
                    const Mat& bottom_blob1 = bottom_blobs[b];
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const __fp16* ptr = bottom_blob1.channel(q);
                        float* outptr = top_blob_fp32.channel(q);

                        for (int i = 0; i < size; i++)
                        {
                            float32x4_t _p = vld1q_f32(outptr);
                            float32x4_t _p1 = vcvt_f32_f16(vld1_f16(ptr));
                            _p = vaddq_f32(_p, _p1);
                            vst1q_f32(outptr, _p);

                            ptr += 4;
                            outptr += 4;
                        }
                    }
                }
                for (; b < bottom_blobs.size(); b++)
                {
                    const Mat& bottom_blob1 = bottom_blobs[b];
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const __fp16* ptr = bottom_blob1.channel(q);
                        const float* ptr0 = top_blob_fp32.channel(q);
                        __fp16* outptr = top_blob.channel(q);

                        for (int i = 0; i < size; i++)
                        {
                            float32x4_t _p = vld1q_f32(ptr0);
                            float32x4_t _p1 = vcvt_f32_f16(vld1_f16(ptr));
                            _p = vaddq_f32(_p, _p1);
                            vst1_f16(outptr, vcvt_f16_f32(_p));

                            ptr += 4;
                            ptr0 += 4;
                            outptr += 4;
                        }
                    }
                }
            }
            else
            {
                // first blob
                const Mat& bottom_blob1 = bottom_blobs[1];
                float32x4_t _coeff0 = vdupq_n_f32(coeffs[0]);
                float32x4_t _coeff1 = vdupq_n_f32(coeffs[1]);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob.channel(q);
                    const __fp16* ptr1 = bottom_blob1.channel(q);
                    float* outptr = top_blob_fp32.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                        float32x4_t _p1 = vcvt_f32_f16(vld1_f16(ptr1));
                        _p = vmulq_f32(_p, _coeff0);
                        _p = vfmaq_f32(_p, _p1, _coeff1);
                        vst1q_f32(outptr, _p);

                        ptr += 4;
                        ptr1 += 4;
                        outptr += 4;
                    }
                }

                size_t b = 2;
                for (; b < bottom_blobs.size() - 1; b++)
                {
                    const Mat& bottom_blob1 = bottom_blobs[b];
                    float32x4_t _coeff = vdupq_n_f32(coeffs[b]);
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const __fp16* ptr = bottom_blob1.channel(q);
                        float* outptr = top_blob_fp32.channel(q);

                        for (int i = 0; i < size; i++)
                        {
                            float32x4_t _p = vld1q_f32(outptr);
                            float32x4_t _p1 = vcvt_f32_f16(vld1_f16(ptr));
                            _p = vfmaq_f32(_p, _p1, _coeff);
                            vst1q_f32(outptr, _p);

                            ptr += 4;
                            outptr += 4;
                        }
                    }
                }
                for (; b < bottom_blobs.size(); b++)
                {
                    const Mat& bottom_blob1 = bottom_blobs[b];
                    float32x4_t _coeff = vdupq_n_f32(coeffs[b]);
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const __fp16* ptr = bottom_blob1.channel(q);
                        const float* ptr0 = top_blob_fp32.channel(q);
                        __fp16* outptr = top_blob.channel(q);

                        for (int i = 0; i < size; i++)
                        {
                            float32x4_t _p = vld1q_f32(ptr0);
                            float32x4_t _p1 = vcvt_f32_f16(vld1_f16(ptr));
                            _p = vfmaq_f32(_p, _p1, _coeff);
                            vst1_f16(outptr, vcvt_f16_f32(_p));

                            ptr += 4;
                            ptr0 += 4;
                            outptr += 4;
                        }
                    }
                }
            }
        }

        return 0;
    }

    if (op_type == Operation_PROD)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const __fp16* ptr = bottom_blob.channel(q);
            const __fp16* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob_fp32.channel(q);

            for (int i = 0; i < size; i++)
            {
                *outptr = (float)(*ptr) * (float)(*ptr1);

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
                const __fp16* ptr = bottom_blob1.channel(q);
                float* outptr = top_blob_fp32.channel(q);

                for (int i = 0; i < size; i++)
                {
                    *outptr *= (float)(*ptr);

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
                const __fp16* ptr = bottom_blob1.channel(q);
                const float* ptr0 = top_blob_fp32.channel(q);
                __fp16* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    *outptr = (__fp16)(*ptr0 * (float)(*ptr));

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
                const __fp16* ptr = bottom_blob.channel(q);
                const __fp16* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob_fp32.channel(q);

                for (int i = 0; i < size; i++)
                {
                    *outptr = (float)(*ptr) + (float)(*ptr1);

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
                    const __fp16* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob_fp32.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        *outptr += (float)(*ptr);

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
                    const __fp16* ptr = bottom_blob1.channel(q);
                    const float* ptr0 = top_blob_fp32.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        *outptr = (__fp16)(*ptr0 + (float)(*ptr));

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
            float coeff0 = coeffs[0];
            float coeff1 = coeffs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = bottom_blob.channel(q);
                const __fp16* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob_fp32.channel(q);

                for (int i = 0; i < size; i++)
                {
                    *outptr = (float)(*ptr) * coeff0 + (float)(*ptr1) * coeff1;

                    ptr++;
                    ptr1++;
                    outptr++;
                }
            }

            size_t b = 2;
            for (; b < bottom_blobs.size() - 1; b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                float coeff = coeffs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob_fp32.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        *outptr += (float)(*ptr) * coeff;

                        ptr++;
                        outptr++;
                    }
                }
            }
            for (; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                float coeff = coeffs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob1.channel(q);
                    const float* ptr0 = top_blob_fp32.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        *outptr = (__fp16)(*ptr0 + (float)(*ptr) * coeff);

                        ptr++;
                        ptr0++;
                        outptr++;
                    }
                }
            }
        }
    }

    return 0;
}

int Eltwise_arm::forward_fp16sa(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int elempack = bottom_blob.elempack;
    int size = w * h;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (bottom_blobs.size() == 2)
    {
        // fast path without fp32 accumulator
        return forward_fp16s(bottom_blobs, top_blobs, opt);
    }

    if (elempack == 8)
    {
        if (op_type == Operation_PROD)
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = bottom_blob.channel(q);
                const __fp16* ptr1 = bottom_blob1.channel(q);
                __fp16* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _p1 = vld1q_f16(ptr1);
                    _p = vmulq_f16(_p, _p1);
                    vst1q_f16(outptr, _p);

                    ptr += 8;
                    ptr1 += 8;
                    outptr += 8;
                }
            }

            size_t b = 2;
            for (; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob1.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        float16x8_t _p = vld1q_f16(outptr);
                        float16x8_t _p1 = vld1q_f16(ptr);
                        _p = vmulq_f16(_p, _p1);
                        vst1q_f16(outptr, _p);

                        ptr += 8;
                        outptr += 8;
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
                    const __fp16* ptr = bottom_blob.channel(q);
                    const __fp16* ptr1 = bottom_blob1.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        float16x8_t _p = vld1q_f16(ptr);
                        float16x8_t _p1 = vld1q_f16(ptr1);
                        _p = vaddq_f16(_p, _p1);
                        vst1q_f16(outptr, _p);

                        ptr += 8;
                        ptr1 += 8;
                        outptr += 8;
                    }
                }

                size_t b = 2;
                for (; b < bottom_blobs.size(); b++)
                {
                    const Mat& bottom_blob1 = bottom_blobs[b];
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const __fp16* ptr = bottom_blob1.channel(q);
                        __fp16* outptr = top_blob.channel(q);

                        for (int i = 0; i < size; i++)
                        {
                            float16x8_t _p = vld1q_f16(outptr);
                            float16x8_t _p1 = vld1q_f16(ptr);
                            _p = vaddq_f16(_p, _p1);
                            vst1q_f16(outptr, _p);

                            ptr += 8;
                            outptr += 8;
                        }
                    }
                }
            }
            else
            {
                // first blob
                const Mat& bottom_blob1 = bottom_blobs[1];
                float16x8_t _coeff0 = vdupq_n_f16((__fp16)coeffs[0]);
                float16x8_t _coeff1 = vdupq_n_f16((__fp16)coeffs[1]);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob.channel(q);
                    const __fp16* ptr1 = bottom_blob1.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        float16x8_t _p = vld1q_f16(ptr);
                        float16x8_t _p1 = vld1q_f16(ptr1);
                        _p = vmulq_f16(_p, _coeff0);
                        _p = vfmaq_f16(_p, _p1, _coeff1);
                        vst1q_f16(outptr, _p);

                        ptr += 8;
                        ptr1 += 8;
                        outptr += 8;
                    }
                }

                size_t b = 2;
                for (; b < bottom_blobs.size(); b++)
                {
                    const Mat& bottom_blob1 = bottom_blobs[b];
                    float16x8_t _coeff = vdupq_n_f16((__fp16)coeffs[b]);
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const __fp16* ptr = bottom_blob1.channel(q);
                        __fp16* outptr = top_blob.channel(q);

                        for (int i = 0; i < size; i++)
                        {
                            float16x8_t _p = vld1q_f16(outptr);
                            float16x8_t _p1 = vld1q_f16(ptr);
                            _p = vfmaq_f16(_p, _p1, _coeff);
                            vst1q_f16(outptr, _p);

                            ptr += 8;
                            outptr += 8;
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
                const __fp16* ptr = bottom_blob.channel(q);
                const __fp16* ptr1 = bottom_blob1.channel(q);
                __fp16* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _p1 = vld1q_f16(ptr1);
                    _p = vmaxq_f16(_p, _p1);
                    vst1q_f16(outptr, _p);

                    ptr += 8;
                    ptr1 += 8;
                    outptr += 8;
                }
            }

            size_t b = 2;
            for (; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob1.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        float16x8_t _p = vld1q_f16(outptr);
                        float16x8_t _p1 = vld1q_f16(ptr);
                        _p = vmaxq_f16(_p, _p1);
                        vst1q_f16(outptr, _p);

                        ptr += 8;
                        outptr += 8;
                    }
                }
            }
        }

        return 0;
    }

    if (elempack == 4)
    {
        if (op_type == Operation_PROD)
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = bottom_blob.channel(q);
                const __fp16* ptr1 = bottom_blob1.channel(q);
                __fp16* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _p1 = vld1_f16(ptr1);
                    _p = vmul_f16(_p, _p1);
                    vst1_f16(outptr, _p);

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
            }

            size_t b = 2;
            for (; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob1.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        float16x4_t _p = vld1_f16(outptr);
                        float16x4_t _p1 = vld1_f16(ptr);
                        _p = vmul_f16(_p, _p1);
                        vst1_f16(outptr, _p);

                        ptr += 4;
                        outptr += 4;
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
                    const __fp16* ptr = bottom_blob.channel(q);
                    const __fp16* ptr1 = bottom_blob1.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        float16x4_t _p = vld1_f16(ptr);
                        float16x4_t _p1 = vld1_f16(ptr1);
                        _p = vadd_f16(_p, _p1);
                        vst1_f16(outptr, _p);

                        ptr += 4;
                        ptr1 += 4;
                        outptr += 4;
                    }
                }

                size_t b = 2;
                for (; b < bottom_blobs.size(); b++)
                {
                    const Mat& bottom_blob1 = bottom_blobs[b];
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const __fp16* ptr = bottom_blob1.channel(q);
                        __fp16* outptr = top_blob.channel(q);

                        for (int i = 0; i < size; i++)
                        {
                            float16x4_t _p = vld1_f16(outptr);
                            float16x4_t _p1 = vld1_f16(ptr);
                            _p = vadd_f16(_p, _p1);
                            vst1_f16(outptr, _p);

                            ptr += 4;
                            outptr += 4;
                        }
                    }
                }
            }
            else
            {
                // first blob
                const Mat& bottom_blob1 = bottom_blobs[1];
                float16x4_t _coeff0 = vdup_n_f16((__fp16)coeffs[0]);
                float16x4_t _coeff1 = vdup_n_f16((__fp16)coeffs[1]);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob.channel(q);
                    const __fp16* ptr1 = bottom_blob1.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        float16x4_t _p = vld1_f16(ptr);
                        float16x4_t _p1 = vld1_f16(ptr1);
                        _p = vmul_f16(_p, _coeff0);
                        _p = vfma_f16(_p, _p1, _coeff1);
                        vst1_f16(outptr, _p);

                        ptr += 4;
                        ptr1 += 4;
                        outptr += 4;
                    }
                }

                size_t b = 2;
                for (; b < bottom_blobs.size(); b++)
                {
                    const Mat& bottom_blob1 = bottom_blobs[b];
                    float16x4_t _coeff = vdup_n_f16((__fp16)coeffs[b]);
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const __fp16* ptr = bottom_blob1.channel(q);
                        __fp16* outptr = top_blob.channel(q);

                        for (int i = 0; i < size; i++)
                        {
                            float16x4_t _p = vld1_f16(outptr);
                            float16x4_t _p1 = vld1_f16(ptr);
                            _p = vfma_f16(_p, _p1, _coeff);
                            vst1_f16(outptr, _p);

                            ptr += 4;
                            outptr += 4;
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
                const __fp16* ptr = bottom_blob.channel(q);
                const __fp16* ptr1 = bottom_blob1.channel(q);
                __fp16* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _p1 = vld1_f16(ptr1);
                    _p = vmax_f16(_p, _p1);
                    vst1_f16(outptr, _p);

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
            }

            size_t b = 2;
            for (; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob1.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        float16x4_t _p = vld1_f16(outptr);
                        float16x4_t _p1 = vld1_f16(ptr);
                        _p = vmax_f16(_p, _p1);
                        vst1_f16(outptr, _p);

                        ptr += 4;
                        outptr += 4;
                    }
                }
            }
        }

        return 0;
    }

    if (op_type == Operation_PROD)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const __fp16* ptr = bottom_blob.channel(q);
            const __fp16* ptr1 = bottom_blob1.channel(q);
            __fp16* outptr = top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                *outptr = *ptr * *ptr1;

                ptr++;
                ptr1++;
                outptr++;
            }
        }

        size_t b = 2;
        for (; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = bottom_blob1.channel(q);
                __fp16* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
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
                const __fp16* ptr = bottom_blob.channel(q);
                const __fp16* ptr1 = bottom_blob1.channel(q);
                __fp16* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    *outptr = *ptr + *ptr1;

                    ptr++;
                    ptr1++;
                    outptr++;
                }
            }

            size_t b = 2;
            for (; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob1.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
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
            __fp16 coeff0 = (__fp16)coeffs[0];
            __fp16 coeff1 = (__fp16)coeffs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = bottom_blob.channel(q);
                const __fp16* ptr1 = bottom_blob1.channel(q);
                __fp16* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    *outptr = *ptr * coeff0 + *ptr1 * coeff1;

                    ptr++;
                    ptr1++;
                    outptr++;
                }
            }

            size_t b = 2;
            for (; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                __fp16 coeff = (__fp16)coeffs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob1.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
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
            const __fp16* ptr = bottom_blob.channel(q);
            const __fp16* ptr1 = bottom_blob1.channel(q);
            __fp16* outptr = top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                *outptr = std::max(*ptr, *ptr1);

                ptr++;
                ptr1++;
                outptr++;
            }
        }

        size_t b = 2;
        for (; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = bottom_blob1.channel(q);
                __fp16* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
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
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn
