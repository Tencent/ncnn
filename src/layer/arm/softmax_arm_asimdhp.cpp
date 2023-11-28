// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "softmax_arm.h"

#include <float.h>

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#include "neon_mathfun_fp16s.h"
#endif // __ARM_NEON

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int Softmax_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    const int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        const int w = bottom_top_blob.w;
        const int size = w * elempack;

        __fp16 max = -65504.f;
        float16x8_t _max = vdupq_n_f16(-65504.f);
        {
            const __fp16* ptr = bottom_top_blob;

            int i = 0;
            for (; i + 7 < size; i += 8)
            {
                float16x8_t _p = vld1q_f16(ptr);
                _max = vmaxq_f16(_max, _p);
                ptr += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                float16x4_t _p = vld1_f16(ptr);
                _max = vcombine_f16(vmax_f16(vget_low_f16(_max), _p), vget_high_f16(_max));
                ptr += 4;
            }
            for (; i < size; i++)
            {
                max = std::max(max, *ptr);
                ptr++;
            }

            max = std::max(max, vmaxvq_f16(_max));
            _max = vdupq_n_f16(max);
        }

        __fp16 sum = 0.f;
        float16x8_t _sum = vdupq_n_f16(0.f);
        {
            __fp16* ptr = bottom_top_blob;

            int i = 0;
            for (; i + 7 < size; i += 8)
            {
                float16x8_t _p = vld1q_f16(ptr);
                _p = exp_ps_f16(vsubq_f16(_p, _max));
                vst1q_f16(ptr, _p);
                _sum = vaddq_f16(_sum, _p);
                ptr += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                float16x4_t _p = vld1_f16(ptr);
                _p = exp_ps_f16(vsub_f16(_p, vget_low_f16(_max)));
                vst1_f16(ptr, _p);
                _sum = vcombine_f16(vadd_f16(vget_low_f16(_sum), _p), vget_high_f16(_sum));
                ptr += 4;
            }
            for (; i < size; i++)
            {
                *ptr = (__fp16)expf(*ptr - max);
                sum += *ptr;
                ptr++;
            }

            float16x4_t _sum2 = vadd_f16(vget_low_f16(_sum), vget_high_f16(_sum));
            float16x4_t _ss2 = vpadd_f16(_sum2, _sum2);
            sum += vget_lane_f16(_ss2, 0) + vget_lane_f16(_ss2, 1);
            _sum = vdupq_n_f16(sum);
        }

        {
            __fp16* ptr = bottom_top_blob;

            int i = 0;
            float16x8_t _reciprocal_sum = vrecpeq_f16(_sum);
            _reciprocal_sum = vmulq_f16(vrecpsq_f16(_sum, _reciprocal_sum), _reciprocal_sum);
            for (; i + 7 < size; i += 8)
            {
                float16x8_t _p = vld1q_f16(ptr);
                _p = vmulq_f16(_p, _reciprocal_sum);
                vst1q_f16(ptr, _p);
                ptr += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                float16x4_t _p = vld1_f16(ptr);
                _p = vmul_f16(_p, vget_low_f16(_reciprocal_sum));
                vst1_f16(ptr, _p);
                ptr += 4;
            }
            for (; i < size; i++)
            {
                *ptr = *ptr / sum;
                ptr++;
            }
        }
    }

    if (dims == 2 && positive_axis == 0)
    {
        const int w = bottom_top_blob.w;
        const int h = bottom_top_blob.h;

        Mat maxsum(w, 1, 2, 2u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        Mat max = maxsum.channel(0);
        max.fill<__fp16>((__fp16)-65504.f);

        for (int i = 0; i < h; i++)
        {
            const __fp16* ptr = bottom_top_blob.row<const __fp16>(i);
            __fp16* maxptr = max;

            if (elempack == 8)
            {
                int j = 0;
                for (; j + 3 < w; j += 4)
                {
                    float16x8_t _p0 = vld1q_f16(ptr);
                    float16x8_t _p1 = vld1q_f16(ptr + 8);
                    float16x8_t _p2 = vld1q_f16(ptr + 16);
                    float16x8_t _p3 = vld1q_f16(ptr + 24);
                    float16x4_t _max = vld1_f16(maxptr);
                    float16x8_t _max01 = vpmaxq_f16(_p0, _p1);
                    float16x8_t _max23 = vpmaxq_f16(_p2, _p3);
                    float16x8_t _max2 = vpmaxq_f16(_max01, _max23);
                    _max = vmax_f16(_max, vpmax_f16(vget_low_f16(_max2), vget_high_f16(_max2)));
                    vst1_f16(maxptr, _max);
                    ptr += 32;
                    maxptr += 4;
                }
                for (; j < w; j++)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    __fp16 max0 = vmaxvq_f16(_p);
                    *maxptr = std::max(*maxptr, max0);
                    ptr += 8;
                    maxptr++;
                }
            }
            if (elempack == 4)
            {
                int j = 0;
                for (; j + 3 < w; j += 4)
                {
                    float16x8_t _p0 = vld1q_f16(ptr);
                    float16x8_t _p1 = vld1q_f16(ptr + 8);
                    float16x4_t _max = vld1_f16(maxptr);
                    float16x8_t _max2 = vpmaxq_f16(_p0, _p1);
                    _max = vmax_f16(_max, vpmax_f16(vget_low_f16(_max2), vget_high_f16(_max2)));
                    vst1_f16(maxptr, _max);
                    ptr += 16;
                    maxptr += 4;
                }
                for (; j < w; j++)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    __fp16 max0 = vmaxv_f16(_p);
                    *maxptr = std::max(*maxptr, max0);
                    ptr += 4;
                    maxptr++;
                }
            }
            if (elempack == 1)
            {
                int j = 0;
                for (; j + 7 < w; j += 8)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _max = vld1q_f16(maxptr);
                    _max = vmaxq_f16(_max, _p);
                    vst1q_f16(maxptr, _max);
                    ptr += 8;
                    maxptr += 8;
                }
                for (; j + 3 < w; j += 4)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _max = vld1_f16(maxptr);
                    _max = vmax_f16(_max, _p);
                    vst1_f16(maxptr, _max);
                    ptr += 4;
                    maxptr += 4;
                }
                for (; j < w; j++)
                {
                    *maxptr = std::max(*maxptr, *ptr);
                    ptr++;
                    maxptr++;
                }
            }
        }

        Mat sum = maxsum.channel(1);
        sum.fill<__fp16>((__fp16)0.f);

        for (int i = 0; i < h; i++)
        {
            __fp16* ptr = bottom_top_blob.row<__fp16>(i);
            const __fp16* maxptr = max;
            __fp16* sumptr = sum;

            if (elempack == 8)
            {
                int j = 0;
                for (; j + 3 < w; j += 4)
                {
                    float16x8_t _p0 = vld1q_f16(ptr);
                    float16x8_t _p1 = vld1q_f16(ptr + 8);
                    float16x8_t _p2 = vld1q_f16(ptr + 16);
                    float16x8_t _p3 = vld1q_f16(ptr + 24);
                    float16x4_t _max = vld1_f16(maxptr);
                    float16x4_t _sum = vld1_f16(sumptr);
                    _p0 = exp_ps_f16(vsubq_f16(_p0, vdupq_lane_f16(_max, 0)));
                    _p1 = exp_ps_f16(vsubq_f16(_p1, vdupq_lane_f16(_max, 1)));
                    _p2 = exp_ps_f16(vsubq_f16(_p2, vdupq_lane_f16(_max, 2)));
                    _p3 = exp_ps_f16(vsubq_f16(_p3, vdupq_lane_f16(_max, 3)));
                    vst1q_f16(ptr, _p0);
                    vst1q_f16(ptr + 8, _p1);
                    vst1q_f16(ptr + 16, _p2);
                    vst1q_f16(ptr + 24, _p3);
                    float16x8_t _ss01 = vpaddq_f16(_p0, _p1);
                    float16x8_t _ss23 = vpaddq_f16(_p2, _p3);
                    float16x8_t _ss2 = vpaddq_f16(_ss01, _ss23);
                    _sum = vadd_f16(_sum, vpmax_f16(vget_low_f16(_ss2), vget_high_f16(_ss2)));
                    vst1_f16(sumptr, _sum);
                    ptr += 32;
                    maxptr += 4;
                    sumptr += 4;
                }
                for (; j < w; j++)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _max = vdupq_n_f16(*maxptr);
                    _p = exp_ps_f16(vsubq_f16(_p, _max));
                    vst1q_f16(ptr, _p);
                    float16x4_t _sum2 = vadd_f16(vget_low_f16(_p), vget_high_f16(_p));
                    float16x4_t _ss2 = vpadd_f16(_sum2, _sum2);
                    __fp16 sum0 = vget_lane_f16(_ss2, 0) + vget_lane_f16(_ss2, 1);
                    *sumptr += sum0;
                    ptr += 8;
                    maxptr++;
                    sumptr++;
                }
            }
            if (elempack == 4)
            {
                int j = 0;
                for (; j + 3 < w; j += 4)
                {
                    float16x8_t _p0 = vld1q_f16(ptr);
                    float16x8_t _p1 = vld1q_f16(ptr + 8);
                    float16x4_t _max = vld1_f16(maxptr);
                    float16x4_t _sum = vld1_f16(sumptr);
                    float16x8_t _max0 = vcombine_f16(vdup_lane_f16(_max, 0), vdup_lane_f16(_max, 1));
                    float16x8_t _max1 = vcombine_f16(vdup_lane_f16(_max, 2), vdup_lane_f16(_max, 3));
                    _p0 = exp_ps_f16(vsubq_f16(_p0, _max0));
                    _p1 = exp_ps_f16(vsubq_f16(_p1, _max1));
                    vst1q_f16(ptr, _p0);
                    vst1q_f16(ptr + 8, _p1);
                    float16x8_t _ss2 = vpaddq_f16(_p0, _p1);
                    _sum = vadd_f16(_sum, vpmax_f16(vget_low_f16(_ss2), vget_high_f16(_ss2)));
                    vst1_f16(sumptr, _sum);
                    ptr += 16;
                    maxptr += 4;
                    sumptr += 4;
                }
                for (; j < w; j++)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _max = vdup_n_f16(*maxptr);
                    _p = exp_ps_f16(vsub_f16(_p, _max));
                    vst1_f16(ptr, _p);
                    float16x4_t _ss2 = vpadd_f16(_p, _p);
                    __fp16 sum0 = vget_lane_f16(_ss2, 0) + vget_lane_f16(_ss2, 1);
                    *sumptr += sum0;
                    ptr += 4;
                    maxptr++;
                    sumptr++;
                }
            }
            if (elempack == 1)
            {
                int j = 0;
                for (; j + 7 < w; j += 8)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _max = vld1q_f16(maxptr);
                    float16x8_t _sum = vld1q_f16(sumptr);
                    _p = exp_ps_f16(vsubq_f16(_p, _max));
                    _sum = vaddq_f16(_sum, _p);
                    vst1q_f16(ptr, _p);
                    vst1q_f16(sumptr, _sum);
                    ptr += 8;
                    maxptr += 8;
                    sumptr += 8;
                }
                for (; j + 3 < w; j += 4)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _max = vld1_f16(maxptr);
                    float16x4_t _sum = vld1_f16(sumptr);
                    _p = exp_ps_f16(vsub_f16(_p, _max));
                    _sum = vadd_f16(_sum, _p);
                    vst1_f16(ptr, _p);
                    vst1_f16(sumptr, _sum);
                    ptr += 4;
                    maxptr += 4;
                    sumptr += 4;
                }
                for (; j < w; j++)
                {
                    *ptr = (__fp16)expf(*ptr - *maxptr);
                    *sumptr += *ptr;
                    ptr++;
                    maxptr++;
                    sumptr++;
                }
            }
        }

        for (int i = 0; i < h; i++)
        {
            __fp16* ptr = bottom_top_blob.row<__fp16>(i);
            const __fp16* sumptr = sum;

            if (elempack == 8)
            {
                for (int j = 0; j < w; j++)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _sum = vld1q_dup_f16(sumptr);
                    _p = vdivq_f16(_p, _sum);
                    vst1q_f16(ptr, _p);
                    ptr += 8;
                    sumptr++;
                }
            }
            if (elempack == 4)
            {
                for (int j = 0; j < w; j++)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _sum = vld1_dup_f16(sumptr);
                    _p = vdiv_f16(_p, _sum);
                    vst1_f16(ptr, _p);
                    ptr += 4;
                    sumptr++;
                }
            }
            if (elempack == 1)
            {
                int j = 0;
                for (; j + 7 < w; j += 8)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _sum = vld1q_f16(sumptr);
                    _p = vdivq_f16(_p, _sum);
                    vst1q_f16(ptr, _p);
                    ptr += 8;
                    sumptr += 8;
                }
                for (; j + 3 < w; j += 4)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _sum = vld1_f16(sumptr);
                    _p = vdiv_f16(_p, _sum);
                    vst1_f16(ptr, _p);
                    ptr += 4;
                    sumptr += 4;
                }
                for (; j < w; j++)
                {
                    *ptr /= *sumptr;
                    ptr++;
                    sumptr++;
                }
            }
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        const int w = bottom_top_blob.w;
        const int h = bottom_top_blob.h;
        const int size = w * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            __fp16 max = -65504.f;
            float16x8_t _max = vdupq_n_f16(-65504.f);
            {
                const __fp16* ptr = bottom_top_blob.row<const __fp16>(i);

                int j = 0;
                for (; j + 7 < size; j += 8)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    _max = vmaxq_f16(_max, _p);
                    ptr += 8;
                }
                for (; j + 3 < size; j += 4)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    _max = vcombine_f16(vmax_f16(vget_low_f16(_max), _p), vget_high_f16(_max));
                    ptr += 4;
                }
                for (; j < size; j++)
                {
                    max = std::max(max, *ptr);
                    ptr++;
                }

                if (elempack == 4)
                {
                    float16x4_t _max2 = vmax_f16(vget_low_f16(_max), vget_high_f16(_max));
                    _max = vcombine_f16(_max2, _max2);
                }
                if (elempack == 1)
                {
                    max = std::max(max, vmaxvq_f16(_max));
                    _max = vdupq_n_f16(max);
                }
            }

            __fp16 sum = 0.f;
            float16x8_t _sum = vdupq_n_f16(0.f);
            {
                __fp16* ptr = bottom_top_blob.row<__fp16>(i);

                int j = 0;
                for (; j + 7 < size; j += 8)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    _p = exp_ps_f16(vsubq_f16(_p, _max));
                    vst1q_f16(ptr, _p);
                    _sum = vaddq_f16(_sum, _p);
                    ptr += 8;
                }
                for (; j + 3 < size; j += 4)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    _p = exp_ps_f16(vsub_f16(_p, vget_low_f16(_max)));
                    vst1_f16(ptr, _p);
                    _sum = vcombine_f16(vadd_f16(vget_low_f16(_sum), _p), vget_high_f16(_sum));
                    ptr += 4;
                }
                for (; j < size; j++)
                {
                    *ptr = (__fp16)expf(*ptr - max);
                    sum += *ptr;
                    ptr++;
                }

                if (elempack == 4)
                {
                    float16x4_t _sum2 = vadd_f16(vget_low_f16(_sum), vget_high_f16(_sum));
                    _sum = vcombine_f16(_sum2, _sum2);
                }
                if (elempack == 1)
                {
                    float16x4_t _sum2 = vadd_f16(vget_low_f16(_sum), vget_high_f16(_sum));
                    float16x4_t _ss2 = vpadd_f16(_sum2, _sum2);
                    sum += vget_lane_f16(_ss2, 0) + vget_lane_f16(_ss2, 1);
                    _sum = vdupq_n_f16(sum);
                }
            }

            {
                __fp16* ptr = bottom_top_blob.row<__fp16>(i);

                int j = 0;
                float16x8_t _reciprocal_sum = vrecpeq_f16(_sum);
                _reciprocal_sum = vmulq_f16(vrecpsq_f16(_sum, _reciprocal_sum), _reciprocal_sum);
                for (; j + 7 < size; j += 8)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    _p = vmulq_f16(_p, _reciprocal_sum);
                    vst1q_f16(ptr, _p);
                    ptr += 8;
                }
                for (; j + 3 < size; j += 4)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    _p = vmul_f16(_p, vget_low_f16(_reciprocal_sum));
                    vst1_f16(ptr, _p);
                    ptr += 4;
                }
                for (; j < size; j++)
                {
                    *ptr /= sum;
                    ptr++;
                }
            }
        }
    }

    if (dims == 3 && positive_axis == 0)
    {
        const int w = bottom_top_blob.w;
        const int h = bottom_top_blob.h;
        const int channels = bottom_top_blob.c;
        const int size = w * h;

        Mat maxsum(w, h, 2, 2u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        Mat max = maxsum.channel(0);
        max.fill<__fp16>((__fp16)-65504.f);

        for (int q = 0; q < channels; q++)
        {
            const __fp16* ptr = bottom_top_blob.channel(q);
            __fp16* maxptr = max;

            if (elempack == 8)
            {
                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    float16x8_t _p0 = vld1q_f16(ptr);
                    float16x8_t _p1 = vld1q_f16(ptr + 8);
                    float16x8_t _p2 = vld1q_f16(ptr + 16);
                    float16x8_t _p3 = vld1q_f16(ptr + 24);
                    float16x4_t _max = vld1_f16(maxptr);
                    float16x8_t _max01 = vpmaxq_f16(_p0, _p1);
                    float16x8_t _max23 = vpmaxq_f16(_p2, _p3);
                    float16x8_t _max2 = vpmaxq_f16(_max01, _max23);
                    _max = vmax_f16(_max, vpmax_f16(vget_low_f16(_max2), vget_high_f16(_max2)));
                    vst1_f16(maxptr, _max);
                    ptr += 32;
                    maxptr += 4;
                }
                for (; i < size; i++)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    __fp16 max0 = vmaxvq_f16(_p);
                    *maxptr = std::max(*maxptr, max0);
                    ptr += 8;
                    maxptr++;
                }
            }
            if (elempack == 4)
            {
                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    float16x8_t _p0 = vld1q_f16(ptr);
                    float16x8_t _p1 = vld1q_f16(ptr + 8);
                    float16x4_t _max = vld1_f16(maxptr);
                    float16x8_t _max2 = vpmaxq_f16(_p0, _p1);
                    _max = vmax_f16(_max, vpmax_f16(vget_low_f16(_max2), vget_high_f16(_max2)));
                    vst1_f16(maxptr, _max);
                    ptr += 16;
                    maxptr += 4;
                }
                for (; i < size; i++)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    __fp16 max0 = vmaxv_f16(_p);
                    *maxptr = std::max(*maxptr, max0);
                    ptr += 4;
                    maxptr++;
                }
            }
            if (elempack == 1)
            {
                int i = 0;
                for (; i + 7 < size; i += 8)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _max = vld1q_f16(maxptr);
                    _max = vmaxq_f16(_max, _p);
                    vst1q_f16(maxptr, _max);
                    ptr += 8;
                    maxptr += 8;
                }
                for (; i + 3 < size; i += 4)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _max = vld1_f16(maxptr);
                    _max = vmax_f16(_max, _p);
                    vst1_f16(maxptr, _max);
                    ptr += 4;
                    maxptr += 4;
                }
                for (; i < size; i++)
                {
                    *maxptr = std::max(*maxptr, *ptr);
                    ptr++;
                    maxptr++;
                }
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);
            const __fp16* maxptr = max;

            if (elempack == 8)
            {
                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    float16x8_t _p0 = vld1q_f16(ptr);
                    float16x8_t _p1 = vld1q_f16(ptr + 8);
                    float16x8_t _p2 = vld1q_f16(ptr + 16);
                    float16x8_t _p3 = vld1q_f16(ptr + 24);
                    float16x4_t _max = vld1_f16(maxptr);
                    _p0 = exp_ps_f16(vsubq_f16(_p0, vdupq_lane_f16(_max, 0)));
                    _p1 = exp_ps_f16(vsubq_f16(_p1, vdupq_lane_f16(_max, 1)));
                    _p2 = exp_ps_f16(vsubq_f16(_p2, vdupq_lane_f16(_max, 2)));
                    _p3 = exp_ps_f16(vsubq_f16(_p3, vdupq_lane_f16(_max, 3)));
                    vst1q_f16(ptr, _p0);
                    vst1q_f16(ptr + 8, _p1);
                    vst1q_f16(ptr + 16, _p2);
                    vst1q_f16(ptr + 24, _p3);
                    ptr += 32;
                    maxptr += 4;
                }
                for (; i < size; i++)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _max = vdupq_n_f16(*maxptr);
                    _p = exp_ps_f16(vsubq_f16(_p, _max));
                    vst1q_f16(ptr, _p);
                    ptr += 8;
                    maxptr++;
                }
            }
            if (elempack == 4)
            {
                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    float16x8_t _p0 = vld1q_f16(ptr);
                    float16x8_t _p1 = vld1q_f16(ptr + 8);
                    float16x4_t _max = vld1_f16(maxptr);
                    float16x8_t _max0 = vcombine_f16(vdup_lane_f16(_max, 0), vdup_lane_f16(_max, 1));
                    float16x8_t _max1 = vcombine_f16(vdup_lane_f16(_max, 2), vdup_lane_f16(_max, 3));
                    _p0 = exp_ps_f16(vsubq_f16(_p0, _max0));
                    _p1 = exp_ps_f16(vsubq_f16(_p1, _max1));
                    vst1q_f16(ptr, _p0);
                    vst1q_f16(ptr + 8, _p1);
                    ptr += 16;
                    maxptr += 4;
                }
                for (; i < size; i++)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _max = vdup_n_f16(*maxptr);
                    _p = exp_ps_f16(vsub_f16(_p, _max));
                    vst1_f16(ptr, _p);
                    ptr += 4;
                    maxptr++;
                }
            }
            if (elempack == 1)
            {
                int i = 0;
                for (; i + 7 < size; i += 8)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _max = vld1q_f16(maxptr);
                    _p = exp_ps_f16(vsubq_f16(_p, _max));
                    vst1q_f16(ptr, _p);
                    ptr += 8;
                    maxptr += 8;
                }
                for (; i + 3 < size; i += 4)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _max = vld1_f16(maxptr);
                    _p = exp_ps_f16(vsub_f16(_p, _max));
                    vst1_f16(ptr, _p);
                    ptr += 4;
                    maxptr += 4;
                }
                for (; i < size; i++)
                {
                    *ptr = (__fp16)expf(*ptr - *maxptr);
                    ptr++;
                    maxptr++;
                }
            }
        }

        Mat sum = maxsum.channel(1);
        sum.fill<__fp16>((__fp16)0.f);

        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);
            __fp16* sumptr = sum;

            if (elempack == 8)
            {
                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    float16x8_t _p0 = vld1q_f16(ptr);
                    float16x8_t _p1 = vld1q_f16(ptr + 8);
                    float16x8_t _p2 = vld1q_f16(ptr + 16);
                    float16x8_t _p3 = vld1q_f16(ptr + 24);
                    float16x4_t _sum = vld1_f16(sumptr);
                    float16x8_t _ss01 = vpaddq_f16(_p0, _p1);
                    float16x8_t _ss23 = vpaddq_f16(_p2, _p3);
                    float16x8_t _ss2 = vpaddq_f16(_ss01, _ss23);
                    _sum = vadd_f16(_sum, vpmax_f16(vget_low_f16(_ss2), vget_high_f16(_ss2)));
                    vst1_f16(sumptr, _sum);
                    ptr += 32;
                    sumptr += 4;
                }
                for (; i < size; i++)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x4_t _sum2 = vadd_f16(vget_low_f16(_p), vget_high_f16(_p));
                    float16x4_t _ss2 = vpadd_f16(_sum2, _sum2);
                    __fp16 sum0 = vget_lane_f16(_ss2, 0) + vget_lane_f16(_ss2, 1);
                    *sumptr += sum0;
                    ptr += 8;
                    sumptr++;
                }
            }
            if (elempack == 4)
            {
                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    float16x8_t _p0 = vld1q_f16(ptr);
                    float16x8_t _p1 = vld1q_f16(ptr + 8);
                    float16x4_t _sum = vld1_f16(sumptr);
                    float16x8_t _ss2 = vpaddq_f16(_p0, _p1);
                    _sum = vadd_f16(_sum, vpmax_f16(vget_low_f16(_ss2), vget_high_f16(_ss2)));
                    vst1_f16(sumptr, _sum);
                    ptr += 16;
                    sumptr += 4;
                }
                for (; i < size; i++)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _ss2 = vpadd_f16(_p, _p);
                    __fp16 sum0 = vget_lane_f16(_ss2, 0) + vget_lane_f16(_ss2, 1);
                    *sumptr += sum0;
                    ptr += 4;
                    sumptr++;
                }
            }
            if (elempack == 1)
            {
                int i = 0;
                for (; i + 7 < size; i += 8)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _sum = vld1q_f16(sumptr);
                    _sum = vaddq_f16(_sum, _p);
                    vst1q_f16(sumptr, _sum);
                    ptr += 8;
                    sumptr += 8;
                }
                for (; i + 3 < size; i += 4)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _sum = vld1_f16(sumptr);
                    _sum = vadd_f16(_sum, _p);
                    vst1_f16(sumptr, _sum);
                    ptr += 4;
                    sumptr += 4;
                }
                for (; i < size; i++)
                {
                    *sumptr += *ptr;
                    ptr++;
                    sumptr++;
                }
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);
            const __fp16* sumptr = sum;

            if (elempack == 8)
            {
                for (int i = 0; i < size; i++)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _sum = vld1q_dup_f16(sumptr);
                    _p = vdivq_f16(_p, _sum);
                    vst1q_f16(ptr, _p);
                    ptr += 8;
                    sumptr++;
                }
            }
            if (elempack == 4)
            {
                for (int i = 0; i < size; i++)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _sum = vld1_dup_f16(sumptr);
                    _p = vdiv_f16(_p, _sum);
                    vst1_f16(ptr, _p);
                    ptr += 4;
                    sumptr++;
                }
            }
            if (elempack == 1)
            {
                int i = 0;
                for (; i + 7 < size; i += 8)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _sum = vld1q_f16(sumptr);
                    _p = vdivq_f16(_p, _sum);
                    vst1q_f16(ptr, _p);
                    ptr += 8;
                    sumptr += 8;
                }
                for (; i + 3 < size; i += 4)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _sum = vld1_f16(sumptr);
                    _p = vdiv_f16(_p, _sum);
                    vst1_f16(ptr, _p);
                    ptr += 4;
                    sumptr += 4;
                }
                for (; i < size; i++)
                {
                    *ptr /= *sumptr;
                    ptr++;
                    sumptr++;
                }
            }
        }
    }

    if (dims == 3 && positive_axis == 1)
    {
        const int w = bottom_top_blob.w;
        const int h = bottom_top_blob.h;
        const int channels = bottom_top_blob.c;
        const int size = w * elempack;

        Mat maxsum(w * elempack, channels, 2, 2u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        Mat max = maxsum.channel(0);
        max.fill<__fp16>((__fp16)-65504.f);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const __fp16* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                __fp16* maxptr = max.row<__fp16>(q);

                int j = 0;
                for (; j + 7 < size; j += 8)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _max = vld1q_f16(maxptr);
                    _max = vmaxq_f16(_max, _p);
                    vst1q_f16(maxptr, _max);
                    ptr += 8;
                    maxptr += 8;
                }
                for (; j + 3 < size; j += 4)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _max = vld1_f16(maxptr);
                    _max = vmax_f16(_max, _p);
                    vst1_f16(maxptr, _max);
                    ptr += 4;
                    maxptr += 4;
                }
                for (; j < size; j++)
                {
                    *maxptr = std::max(*maxptr, *ptr);
                    ptr++;
                    maxptr++;
                }
            }
        }

        Mat sum = maxsum.channel(1);
        sum.fill<__fp16>((__fp16)0.f);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                const __fp16* maxptr = max.row<const __fp16>(q);
                __fp16* sumptr = sum.row<__fp16>(q);

                int j = 0;
                for (; j + 7 < size; j += 8)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _max = vld1q_f16(maxptr);
                    float16x8_t _sum = vld1q_f16(sumptr);
                    _p = exp_ps_f16(vsubq_f16(_p, _max));
                    _sum = vaddq_f16(_sum, _p);
                    vst1q_f16(ptr, _p);
                    vst1q_f16(sumptr, _sum);
                    ptr += 8;
                    maxptr += 8;
                    sumptr += 8;
                }
                for (; j + 3 < size; j += 4)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _max = vld1_f16(maxptr);
                    float16x4_t _sum = vld1_f16(sumptr);
                    _p = exp_ps_f16(vsub_f16(_p, _max));
                    _sum = vadd_f16(_sum, _p);
                    vst1_f16(ptr, _p);
                    vst1_f16(sumptr, _sum);
                    ptr += 4;
                    maxptr += 4;
                    sumptr += 4;
                }
                for (; j < size; j++)
                {
                    *ptr = (__fp16)expf(*ptr - *maxptr);
                    *sumptr += *ptr;
                    ptr++;
                    maxptr++;
                    sumptr++;
                }
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                const __fp16* sumptr = sum.row<const __fp16>(q);

                int j = 0;
                for (; j + 7 < size; j += 8)
                {
                    float16x8_t _p = vld1q_f16(ptr);
                    float16x8_t _sum = vld1q_f16(sumptr);
                    _p = vdivq_f16(_p, _sum);
                    vst1q_f16(ptr, _p);
                    ptr += 8;
                    sumptr += 8;
                }
                for (; j + 3 < size; j += 4)
                {
                    float16x4_t _p = vld1_f16(ptr);
                    float16x4_t _sum = vld1_f16(sumptr);
                    _p = vdiv_f16(_p, _sum);
                    vst1_f16(ptr, _p);
                    ptr += 4;
                    sumptr += 4;
                }
                for (; j < size; j++)
                {
                    *ptr /= *sumptr;
                    ptr++;
                    sumptr++;
                }
            }
        }
    }

    if (dims == 3 && positive_axis == 2)
    {
        const int w = bottom_top_blob.w;
        const int h = bottom_top_blob.h;
        const int channels = bottom_top_blob.c;
        const int size = w * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            for (int i = 0; i < h; i++)
            {
                __fp16 max = -65504.f;
                float16x8_t _max = vdupq_n_f16(-65504.f);
                {
                    const __fp16* ptr = bottom_top_blob.channel(q).row<const __fp16>(i);

                    int j = 0;
                    for (; j + 7 < size; j += 8)
                    {
                        float16x8_t _p = vld1q_f16(ptr);
                        _max = vmaxq_f16(_max, _p);
                        ptr += 8;
                    }
                    for (; j + 3 < size; j += 4)
                    {
                        float16x4_t _p = vld1_f16(ptr);
                        _max = vcombine_f16(vmax_f16(vget_low_f16(_max), _p), vget_high_f16(_max));
                        ptr += 4;
                    }
                    for (; j < size; j++)
                    {
                        max = std::max(max, *ptr);
                        ptr++;
                    }

                    if (elempack == 4)
                    {
                        float16x4_t _max2 = vmax_f16(vget_low_f16(_max), vget_high_f16(_max));
                        _max = vcombine_f16(_max2, _max2);
                    }
                    if (elempack == 1)
                    {
                        max = std::max(max, vmaxvq_f16(_max));
                        _max = vdupq_n_f16(max);
                    }
                }

                __fp16 sum = 0.f;
                float16x8_t _sum = vdupq_n_f16(0.f);
                {
                    __fp16* ptr = bottom_top_blob.channel(q).row<__fp16>(i);

                    int j = 0;
                    for (; j + 7 < size; j += 8)
                    {
                        float16x8_t _p = vld1q_f16(ptr);
                        _p = exp_ps_f16(vsubq_f16(_p, _max));
                        vst1q_f16(ptr, _p);
                        _sum = vaddq_f16(_sum, _p);
                        ptr += 8;
                    }
                    for (; j + 3 < size; j += 4)
                    {
                        float16x4_t _p = vld1_f16(ptr);
                        _p = exp_ps_f16(vsub_f16(_p, vget_low_f16(_max)));
                        vst1_f16(ptr, _p);
                        _sum = vcombine_f16(vadd_f16(vget_low_f16(_sum), _p), vget_high_f16(_sum));
                        ptr += 4;
                    }
                    for (; j < size; j++)
                    {
                        *ptr = (__fp16)expf(*ptr - max);
                        sum += *ptr;
                        ptr++;
                    }

                    if (elempack == 4)
                    {
                        float16x4_t _sum2 = vadd_f16(vget_low_f16(_sum), vget_high_f16(_sum));
                        _sum = vcombine_f16(_sum2, _sum2);
                    }
                    if (elempack == 1)
                    {
                        float16x4_t _sum2 = vadd_f16(vget_low_f16(_sum), vget_high_f16(_sum));
                        float16x4_t _ss2 = vpadd_f16(_sum2, _sum2);
                        sum += vget_lane_f16(_ss2, 0) + vget_lane_f16(_ss2, 1);
                        _sum = vdupq_n_f16(sum);
                    }
                }

                {
                    __fp16* ptr = bottom_top_blob.channel(q).row<__fp16>(i);

                    int j = 0;
                    float16x8_t _reciprocal_sum = vrecpeq_f16(_sum);
                    _reciprocal_sum = vmulq_f16(vrecpsq_f16(_sum, _reciprocal_sum), _reciprocal_sum);
                    for (; j + 7 < size; j += 8)
                    {
                        float16x8_t _p = vld1q_f16(ptr);
                        _p = vmulq_f16(_p, _reciprocal_sum);
                        vst1q_f16(ptr, _p);
                        ptr += 8;
                    }
                    for (; j + 3 < size; j += 4)
                    {
                        float16x4_t _p = vld1_f16(ptr);
                        _p = vmul_f16(_p, vget_low_f16(_reciprocal_sum));
                        vst1_f16(ptr, _p);
                        ptr += 4;
                    }
                    for (; j < size; j++)
                    {
                        *ptr /= sum;
                        ptr++;
                    }
                }
            }
        }
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn
