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

#include "softmax_arm.h"

#include <float.h>

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#endif // __ARM_NEON

#include "arm_usability.h"
#include "cpu.h"

namespace ncnn {

Softmax_arm::Softmax_arm()
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

int Softmax_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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

    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    const int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        const int w = bottom_top_blob.w;
        const int size = w * elempack;

        float max = -FLT_MAX;
#if __ARM_NEON
        float32x4_t _max = vdupq_n_f32(-FLT_MAX);
#endif // __ARM_NEON
        {
            const float* ptr = bottom_top_blob;

            int i = 0;
#if __ARM_NEON
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = vld1q_f32(ptr);
                _max = vmaxq_f32(_max, _p);
                ptr += 4;
            }
#endif // __ARM_NEON
            for (; i < size; i++)
            {
                max = std::max(max, *ptr);
                ptr++;
            }

#if __ARM_NEON
#if __aarch64__
            max = std::max(max, vmaxvq_f32(_max));
#else
            float32x2_t _max2 = vmax_f32(vget_low_f32(_max), vget_high_f32(_max));
            float32x2_t _mm2 = vpmax_f32(_max2, _max2);
            max = std::max(max, vget_lane_f32(_mm2, 0));
#endif
            _max = vdupq_n_f32(max);
#endif // __ARM_NEON
        }

        float sum = 0.f;
#if __ARM_NEON
        float32x4_t _sum = vdupq_n_f32(0.f);
#endif // __ARM_NEON
        {
            float* ptr = bottom_top_blob;

            int i = 0;
#if __ARM_NEON
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = vld1q_f32(ptr);
                _p = exp_ps(vsubq_f32(_p, _max));
                vst1q_f32(ptr, _p);
                _sum = vaddq_f32(_sum, _p);
                ptr += 4;
            }
#endif // __ARM_NEON
            for (; i < size; i++)
            {
                *ptr = (float)expf(*ptr - max);
                sum += *ptr;
                ptr++;
            }

#if __ARM_NEON
#if __aarch64__
            sum += vaddvq_f32(_sum);
#else
            float32x2_t _sum2 = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
            float32x2_t _ss2 = vpadd_f32(_sum2, _sum2);
            sum += vget_lane_f32(_ss2, 0);
#endif
            _sum = vdupq_n_f32(sum);
#endif // __ARM_NEON
        }

        {
            float* ptr = bottom_top_blob;

            int i = 0;
#if __ARM_NEON
            float32x4_t _reciprocal_sum = vrecpeq_f32(_sum);
            _reciprocal_sum = vmulq_f32(vrecpsq_f32(_sum, _reciprocal_sum), _reciprocal_sum);
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = vld1q_f32(ptr);
                _p = vmulq_f32(_p, _reciprocal_sum);
                vst1q_f32(ptr, _p);
                ptr += 4;
            }
#endif // __ARM_NEON
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

        Mat maxsum(w, 1, 2, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        Mat max = maxsum.channel(0);
        max.fill(-FLT_MAX);

        for (int i = 0; i < h; i++)
        {
            const float* ptr = bottom_top_blob.row(i);
            float* maxptr = max;

#if __ARM_NEON
            if (elempack == 4)
            {
                for (int j = 0; j < w; j++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
#if __aarch64__
                    float max0 = vmaxvq_f32(_p);
#else
                    float32x2_t _max2 = vmax_f32(vget_low_f32(_p), vget_high_f32(_p));
                    float32x2_t _mm2 = vpmax_f32(_max2, _max2);
                    float max0 = vget_lane_f32(_mm2, 0);
#endif
                    *maxptr = std::max(*maxptr, max0);
                    ptr += 4;
                    maxptr++;
                }
            }
#endif // __ARM_NEON
            if (elempack == 1)
            {
                int j = 0;
#if __ARM_NEON
                for (; j + 3 < w; j += 4)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _max = vld1q_f32(maxptr);
                    _max = vmaxq_f32(_max, _p);
                    vst1q_f32(maxptr, _max);
                    ptr += 4;
                    maxptr += 4;
                }
#endif // __ARM_NEON
                for (; j < w; j++)
                {
                    *maxptr = std::max(*maxptr, *ptr);
                    ptr++;
                    maxptr++;
                }
            }
        }

        Mat sum = maxsum.channel(1);
        sum.fill(0.f);

        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            const float* maxptr = max;
            float* sumptr = sum;

#if __ARM_NEON
            if (elempack == 4)
            {
                for (int j = 0; j < w; j++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _max = vdupq_n_f32(*maxptr);
                    _p = exp_ps(vsubq_f32(_p, _max));
                    vst1q_f32(ptr, _p);
#if __aarch64__
                    float sum0 = vaddvq_f32(_p);
#else
                    float32x2_t _sum2 = vadd_f32(vget_low_f32(_p), vget_high_f32(_p));
                    float32x2_t _ss2 = vpadd_f32(_sum2, _sum2);
                    float sum0 = vget_lane_f32(_ss2, 0);
#endif
                    *sumptr += sum0;
                    ptr += 4;
                    maxptr++;
                    sumptr++;
                }
            }
#endif // __ARM_NEON
            if (elempack == 1)
            {
                int j = 0;
#if __ARM_NEON
                for (; j + 3 < w; j += 4)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _max = vld1q_f32(maxptr);
                    float32x4_t _sum = vld1q_f32(sumptr);
                    _p = exp_ps(vsubq_f32(_p, _max));
                    _sum = vaddq_f32(_sum, _p);
                    vst1q_f32(ptr, _p);
                    vst1q_f32(sumptr, _sum);
                    ptr += 4;
                    maxptr += 4;
                    sumptr += 4;
                }
#endif // __ARM_NEON
                for (; j < w; j++)
                {
                    *ptr = (float)expf(*ptr - *maxptr);
                    *sumptr += *ptr;
                    ptr++;
                    maxptr++;
                    sumptr++;
                }
            }
        }

        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            const float* sumptr = sum;

#if __ARM_NEON
            if (elempack == 4)
            {
                for (int j = 0; j < w; j++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _sum = vld1q_dup_f32(sumptr);
                    _p = div_ps(_p, _sum);
                    vst1q_f32(ptr, _p);
                    ptr += 4;
                    sumptr++;
                }
            }
#endif // __ARM_NEON
            if (elempack == 1)
            {
                int j = 0;
#if __ARM_NEON
                for (; j + 3 < w; j += 4)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _sum = vld1q_f32(sumptr);
                    _p = div_ps(_p, _sum);
                    vst1q_f32(ptr, _p);
                    ptr += 4;
                    sumptr += 4;
                }
#endif // __ARM_NEON
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
            float max = -FLT_MAX;
#if __ARM_NEON
            float32x4_t _max = vdupq_n_f32(-FLT_MAX);
#endif // __ARM_NEON
            {
                const float* ptr = bottom_top_blob.row(i);

                int j = 0;
#if __ARM_NEON
                for (; j + 3 < size; j += 4)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    _max = vmaxq_f32(_max, _p);
                    ptr += 4;
                }
#endif // __ARM_NEON
                for (; j < size; j++)
                {
                    max = std::max(max, *ptr);
                    ptr++;
                }

#if __ARM_NEON
                if (elempack == 1)
                {
#if __aarch64__
                    max = std::max(max, vmaxvq_f32(_max));
#else
                    float32x2_t _max2 = vmax_f32(vget_low_f32(_max), vget_high_f32(_max));
                    float32x2_t _mm2 = vpmax_f32(_max2, _max2);
                    max = std::max(max, vget_lane_f32(_mm2, 0));
#endif
                    _max = vdupq_n_f32(max);
                }
#endif // __ARM_NEON
            }

            float sum = 0.f;
#if __ARM_NEON
            float32x4_t _sum = vdupq_n_f32(0.f);
#endif // __ARM_NEON
            {
                float* ptr = bottom_top_blob.row(i);

                int j = 0;
#if __ARM_NEON
                for (; j + 3 < size; j += 4)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    _p = exp_ps(vsubq_f32(_p, _max));
                    vst1q_f32(ptr, _p);
                    _sum = vaddq_f32(_sum, _p);
                    ptr += 4;
                }
#endif // __ARM_NEON
                for (; j < size; j++)
                {
                    *ptr = (float)expf(*ptr - max);
                    sum += *ptr;
                    ptr++;
                }

#if __ARM_NEON
                if (elempack == 1)
                {
#if __aarch64__
                    sum += vaddvq_f32(_sum);
#else
                    float32x2_t _sum2 = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                    float32x2_t _ss2 = vpadd_f32(_sum2, _sum2);
                    sum += vget_lane_f32(_ss2, 0);
#endif
                    _sum = vdupq_n_f32(sum);
                }
#endif // __ARM_NEON
            }

            {
                float* ptr = bottom_top_blob.row(i);

                int j = 0;
#if __ARM_NEON
                float32x4_t _reciprocal_sum = vrecpeq_f32(_sum);
                _reciprocal_sum = vmulq_f32(vrecpsq_f32(_sum, _reciprocal_sum), _reciprocal_sum);
                for (; j + 3 < size; j += 4)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    _p = vmulq_f32(_p, _reciprocal_sum);
                    vst1q_f32(ptr, _p);
                    ptr += 4;
                }
#endif // __ARM_NEON
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

        Mat maxsum(w, h, 2, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        Mat max = maxsum.channel(0);
        max.fill(-FLT_MAX);

        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_top_blob.channel(q);
            float* maxptr = max;

#if __ARM_NEON
            if (elempack == 4)
            {
                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
#if __aarch64__
                    float max0 = vmaxvq_f32(_p);
#else
                    float32x2_t _max2 = vmax_f32(vget_low_f32(_p), vget_high_f32(_p));
                    float32x2_t _mm2 = vpmax_f32(_max2, _max2);
                    float max0 = vget_lane_f32(_mm2, 0);
#endif
                    *maxptr = std::max(*maxptr, max0);
                    ptr += 4;
                    maxptr++;
                }
            }
#endif // __ARM_NEON
            if (elempack == 1)
            {
                int i = 0;
#if __ARM_NEON
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _max = vld1q_f32(maxptr);
                    _max = vmaxq_f32(_max, _p);
                    vst1q_f32(maxptr, _max);
                    ptr += 4;
                    maxptr += 4;
                }
#endif // __ARM_NEON
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
            float* ptr = bottom_top_blob.channel(q);
            const float* maxptr = max;

#if __ARM_NEON
            if (elempack == 4)
            {
                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _max = vdupq_n_f32(*maxptr);
                    _p = exp_ps(vsubq_f32(_p, _max));
                    vst1q_f32(ptr, _p);
                    ptr += 4;
                    maxptr++;
                }
            }
#endif // __ARM_NEON
            if (elempack == 1)
            {
                int i = 0;
#if __ARM_NEON
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _max = vld1q_f32(maxptr);
                    _p = exp_ps(vsubq_f32(_p, _max));
                    vst1q_f32(ptr, _p);
                    ptr += 4;
                    maxptr += 4;
                }
#endif // __ARM_NEON
                for (; i < size; i++)
                {
                    *ptr = (float)expf(*ptr - *maxptr);
                    ptr++;
                    maxptr++;
                }
            }
        }

        Mat sum = maxsum.channel(1);
        sum.fill(0.f);

        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float* sumptr = sum;

#if __ARM_NEON
            if (elempack == 4)
            {
                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
#if __aarch64__
                    float sum0 = vaddvq_f32(_p);
#else
                    float32x2_t _sum2 = vadd_f32(vget_low_f32(_p), vget_high_f32(_p));
                    float32x2_t _ss2 = vpadd_f32(_sum2, _sum2);
                    float sum0 = vget_lane_f32(_ss2, 0);
#endif
                    *sumptr += sum0;
                    ptr += 4;
                    sumptr++;
                }
            }
#endif // __ARM_NEON
            if (elempack == 1)
            {
                int i = 0;
#if __ARM_NEON
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _sum = vld1q_f32(sumptr);
                    _sum = vaddq_f32(_sum, _p);
                    vst1q_f32(sumptr, _sum);
                    ptr += 4;
                    sumptr += 4;
                }
#endif // __ARM_NEON
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
            float* ptr = bottom_top_blob.channel(q);
            const float* sumptr = sum;

#if __ARM_NEON
            if (elempack == 4)
            {
                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _sum = vdupq_n_f32(*sumptr);
                    _p = div_ps(_p, _sum);
                    vst1q_f32(ptr, _p);
                    ptr += 4;
                    sumptr++;
                }
            }
#endif // __ARM_NEON
            if (elempack == 1)
            {
                int i = 0;
#if __ARM_NEON
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _sum = vld1q_f32(sumptr);
                    _p = div_ps(_p, _sum);
                    vst1q_f32(ptr, _p);
                    ptr += 4;
                    sumptr += 4;
                }
#endif // __ARM_NEON
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

        Mat maxsum(w * elempack, channels, 2, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        Mat max = maxsum.channel(0);
        max.fill(-FLT_MAX);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                float* maxptr = max.row(q);

                int j = 0;
#if __ARM_NEON
                for (; j + 3 < size; j += 4)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _max = vld1q_f32(maxptr);
                    _max = vmaxq_f32(_max, _p);
                    vst1q_f32(maxptr, _max);
                    ptr += 4;
                    maxptr += 4;
                }
#endif // __ARM_NEON
                for (; j < size; j++)
                {
                    *maxptr = std::max(*maxptr, *ptr);
                    ptr++;
                    maxptr++;
                }
            }
        }

        Mat sum = maxsum.channel(1);
        sum.fill(0.f);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                const float* maxptr = max.row(q);
                float* sumptr = sum.row(q);

                int j = 0;
#if __ARM_NEON
                for (; j + 3 < size; j += 4)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _max = vld1q_f32(maxptr);
                    float32x4_t _sum = vld1q_f32(sumptr);
                    _p = exp_ps(vsubq_f32(_p, _max));
                    _sum = vaddq_f32(_sum, _p);
                    vst1q_f32(ptr, _p);
                    vst1q_f32(sumptr, _sum);
                    ptr += 4;
                    maxptr += 4;
                    sumptr += 4;
                }
#endif // __ARM_NEON
                for (; j < size; j++)
                {
                    *ptr = (float)expf(*ptr - *maxptr);
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
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                const float* sumptr = sum.row(q);

                int j = 0;
#if __ARM_NEON
                for (; j + 3 < size; j += 4)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _sum = vld1q_f32(sumptr);
                    _p = div_ps(_p, _sum);
                    vst1q_f32(ptr, _p);
                    ptr += 4;
                    sumptr += 4;
                }
#endif // __ARM_NEON
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
                float max = -FLT_MAX;
#if __ARM_NEON
                float32x4_t _max = vdupq_n_f32(-FLT_MAX);
#endif // __ARM_NEON
                {
                    const float* ptr = bottom_top_blob.channel(q).row(i);

                    int j = 0;
#if __ARM_NEON
                    for (; j + 3 < size; j += 4)
                    {
                        float32x4_t _p = vld1q_f32(ptr);
                        _max = vmaxq_f32(_max, _p);
                        ptr += 4;
                    }
#endif // __ARM_NEON
                    for (; j < size; j++)
                    {
                        max = std::max(max, *ptr);
                        ptr++;
                    }

#if __ARM_NEON
                    if (elempack == 1)
                    {
#if __aarch64__
                        max = std::max(max, vmaxvq_f32(_max));
#else
                        float32x2_t _max2 = vmax_f32(vget_low_f32(_max), vget_high_f32(_max));
                        float32x2_t _mm2 = vpmax_f32(_max2, _max2);
                        max = std::max(max, vget_lane_f32(_mm2, 0));
#endif
                        _max = vdupq_n_f32(max);
                    }
#endif // __ARM_NEON
                }

                float sum = 0.f;
#if __ARM_NEON
                float32x4_t _sum = vdupq_n_f32(0.f);
#endif // __ARM_NEON
                {
                    float* ptr = bottom_top_blob.channel(q).row(i);

                    int j = 0;
#if __ARM_NEON
                    for (; j + 3 < size; j += 4)
                    {
                        float32x4_t _p = vld1q_f32(ptr);
                        _p = exp_ps(vsubq_f32(_p, _max));
                        vst1q_f32(ptr, _p);
                        _sum = vaddq_f32(_sum, _p);
                        ptr += 4;
                    }
#endif // __ARM_NEON
                    for (; j < size; j++)
                    {
                        *ptr = (float)expf(*ptr - max);
                        sum += *ptr;
                        ptr++;
                    }

#if __ARM_NEON
                    if (elempack == 1)
                    {
#if __aarch64__
                        sum += vaddvq_f32(_sum);
#else
                        float32x2_t _sum2 = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                        float32x2_t _ss2 = vpadd_f32(_sum2, _sum2);
                        sum += vget_lane_f32(_ss2, 0);
#endif
                        _sum = vdupq_n_f32(sum);
                    }
#endif // __ARM_NEON
                }

                {
                    float* ptr = bottom_top_blob.channel(q).row(i);

                    int j = 0;
#if __ARM_NEON
                    float32x4_t _reciprocal_sum = vrecpeq_f32(_sum);
                    _reciprocal_sum = vmulq_f32(vrecpsq_f32(_sum, _reciprocal_sum), _reciprocal_sum);
                    for (; j + 3 < size; j += 4)
                    {
                        float32x4_t _p = vld1q_f32(ptr);
                        _p = vmulq_f32(_p, _reciprocal_sum);
                        vst1q_f32(ptr, _p);
                        ptr += 4;
                    }
#endif // __ARM_NEON
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

#if NCNN_BF16
int Softmax_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    const int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        const int w = bottom_top_blob.w;
        const int size = w * elempack;

        float max = -FLT_MAX;
#if __ARM_NEON
        float32x4_t _max = vdupq_n_f32(-FLT_MAX);
#endif // __ARM_NEON
        {
            const unsigned short* ptr = bottom_top_blob;

            int i = 0;
#if __ARM_NEON
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = bfloat2float(vld1_u16(ptr));
                _max = vmaxq_f32(_max, _p);
                ptr += 4;
            }
#endif // __ARM_NEON
            for (; i < size; i++)
            {
                max = std::max(max, bfloat16_to_float32(*ptr));
                ptr++;
            }

#if __ARM_NEON
#if __aarch64__
            max = std::max(max, vmaxvq_f32(_max));
#else
            float32x2_t _max2 = vmax_f32(vget_low_f32(_max), vget_high_f32(_max));
            float32x2_t _mm2 = vpmax_f32(_max2, _max2);
            max = std::max(max, vget_lane_f32(_mm2, 0));
#endif
            _max = vdupq_n_f32(max);
#endif // __ARM_NEON
        }

        float sum = 0.f;
#if __ARM_NEON
        float32x4_t _sum = vdupq_n_f32(0.f);
#endif // __ARM_NEON
        {
            unsigned short* ptr = bottom_top_blob;

            int i = 0;
#if __ARM_NEON
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = bfloat2float(vld1_u16(ptr));
                _p = exp_ps(vsubq_f32(_p, _max));
                vst1_u16(ptr, float2bfloat(_p));
                _sum = vaddq_f32(_sum, _p);
                ptr += 4;
            }
#endif // __ARM_NEON
            for (; i < size; i++)
            {
                float v = (float)expf(bfloat16_to_float32(*ptr) - max);
                *ptr = float32_to_bfloat16(v);
                sum += v;
                ptr++;
            }

#if __ARM_NEON
#if __aarch64__
            sum += vaddvq_f32(_sum);
#else
            float32x2_t _sum2 = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
            float32x2_t _ss2 = vpadd_f32(_sum2, _sum2);
            sum += vget_lane_f32(_ss2, 0);
#endif
            _sum = vdupq_n_f32(sum);
#endif // __ARM_NEON
        }

        {
            unsigned short* ptr = bottom_top_blob;

            int i = 0;
#if __ARM_NEON
            float32x4_t _reciprocal_sum = vrecpeq_f32(_sum);
            _reciprocal_sum = vmulq_f32(vrecpsq_f32(_sum, _reciprocal_sum), _reciprocal_sum);
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = bfloat2float(vld1_u16(ptr));
                _p = vmulq_f32(_p, _reciprocal_sum);
                vst1_u16(ptr, float2bfloat(_p));
                ptr += 4;
            }
#endif // __ARM_NEON
            for (; i < size; i++)
            {
                *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) / sum);
                ptr++;
            }
        }
    }

    if (dims == 2 && positive_axis == 0)
    {
        const int w = bottom_top_blob.w;
        const int h = bottom_top_blob.h;

        Mat maxsum(w, 1, 2, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        Mat max = maxsum.channel(0);
        max.fill(-FLT_MAX);

        for (int i = 0; i < h; i++)
        {
            const unsigned short* ptr = bottom_top_blob.row<const unsigned short>(i);
            float* maxptr = max;

#if __ARM_NEON
            if (elempack == 4)
            {
                for (int j = 0; j < w; j++)
                {
                    float32x4_t _p = bfloat2float(vld1_u16(ptr));
#if __aarch64__
                    float max0 = vmaxvq_f32(_p);
#else
                    float32x2_t _max2 = vmax_f32(vget_low_f32(_p), vget_high_f32(_p));
                    float32x2_t _mm2 = vpmax_f32(_max2, _max2);
                    float max0 = vget_lane_f32(_mm2, 0);
#endif
                    *maxptr = std::max(*maxptr, max0);
                    ptr += 4;
                    maxptr++;
                }
            }
#endif // __ARM_NEON
            if (elempack == 1)
            {
                int j = 0;
#if __ARM_NEON
                for (; j + 3 < w; j += 4)
                {
                    float32x4_t _p = bfloat2float(vld1_u16(ptr));
                    float32x4_t _max = vld1q_f32(maxptr);
                    _max = vmaxq_f32(_max, _p);
                    vst1q_f32(maxptr, _max);
                    ptr += 4;
                    maxptr += 4;
                }
#endif // __ARM_NEON
                for (; j < w; j++)
                {
                    *maxptr = std::max(*maxptr, bfloat16_to_float32(*ptr));
                    ptr++;
                    maxptr++;
                }
            }
        }

        Mat sum = maxsum.channel(1);
        sum.fill(0.f);

        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);
            const float* maxptr = max;
            float* sumptr = sum;

#if __ARM_NEON
            if (elempack == 4)
            {
                for (int j = 0; j < w; j++)
                {
                    float32x4_t _p = bfloat2float(vld1_u16(ptr));
                    float32x4_t _max = vdupq_n_f32(*maxptr);
                    _p = exp_ps(vsubq_f32(_p, _max));
                    vst1_u16(ptr, float2bfloat(_p));
#if __aarch64__
                    float sum0 = vaddvq_f32(_p);
#else
                    float32x2_t _sum2 = vadd_f32(vget_low_f32(_p), vget_high_f32(_p));
                    float32x2_t _ss2 = vpadd_f32(_sum2, _sum2);
                    float sum0 = vget_lane_f32(_ss2, 0);
#endif
                    *sumptr += sum0;
                    ptr += 4;
                    maxptr++;
                    sumptr++;
                }
            }
#endif // __ARM_NEON
            if (elempack == 1)
            {
                int j = 0;
#if __ARM_NEON
                for (; j + 3 < w; j += 4)
                {
                    float32x4_t _p = bfloat2float(vld1_u16(ptr));
                    float32x4_t _max = vld1q_f32(maxptr);
                    float32x4_t _sum = vld1q_f32(sumptr);
                    _p = exp_ps(vsubq_f32(_p, _max));
                    vst1_u16(ptr, float2bfloat(_p));
                    _sum = vaddq_f32(_sum, _p);
                    vst1q_f32(sumptr, _sum);
                    ptr += 4;
                    maxptr += 4;
                    sumptr += 4;
                }
#endif // __ARM_NEON
                for (; j < w; j++)
                {
                    float v = (float)expf(bfloat16_to_float32(*ptr) - *maxptr);
                    *ptr = float32_to_bfloat16(v);
                    *sumptr += v;
                    ptr++;
                    maxptr++;
                    sumptr++;
                }
            }
        }

        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);
            const float* sumptr = sum;

#if __ARM_NEON
            if (elempack == 4)
            {
                for (int j = 0; j < w; j++)
                {
                    float32x4_t _p = bfloat2float(vld1_u16(ptr));
                    float32x4_t _sum = vld1q_dup_f32(sumptr);
                    _p = div_ps(_p, _sum);
                    vst1_u16(ptr, float2bfloat(_p));
                    ptr += 4;
                    sumptr++;
                }
            }
#endif // __ARM_NEON
            if (elempack == 1)
            {
                int j = 0;
#if __ARM_NEON
                for (; j + 3 < w; j += 4)
                {
                    float32x4_t _p = bfloat2float(vld1_u16(ptr));
                    float32x4_t _sum = vld1q_f32(sumptr);
                    _p = div_ps(_p, _sum);
                    vst1_u16(ptr, float2bfloat(_p));
                    ptr += 4;
                    sumptr += 4;
                }
#endif // __ARM_NEON
                for (; j < w; j++)
                {
                    *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) / *sumptr);
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
            float max = -FLT_MAX;
#if __ARM_NEON
            float32x4_t _max = vdupq_n_f32(-FLT_MAX);
#endif // __ARM_NEON
            {
                const unsigned short* ptr = bottom_top_blob.row<const unsigned short>(i);

                int j = 0;
#if __ARM_NEON
                for (; j + 3 < size; j += 4)
                {
                    float32x4_t _p = bfloat2float(vld1_u16(ptr));
                    _max = vmaxq_f32(_max, _p);
                    ptr += 4;
                }
#endif // __ARM_NEON
                for (; j < size; j++)
                {
                    max = std::max(max, bfloat16_to_float32(*ptr));
                    ptr++;
                }

#if __ARM_NEON
                if (elempack == 1)
                {
#if __aarch64__
                    max = std::max(max, vmaxvq_f32(_max));
#else
                    float32x2_t _max2 = vmax_f32(vget_low_f32(_max), vget_high_f32(_max));
                    float32x2_t _mm2 = vpmax_f32(_max2, _max2);
                    max = std::max(max, vget_lane_f32(_mm2, 0));
#endif
                    _max = vdupq_n_f32(max);
                }
#endif // __ARM_NEON
            }

            float sum = 0.f;
#if __ARM_NEON
            float32x4_t _sum = vdupq_n_f32(0.f);
#endif // __ARM_NEON
            {
                unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);

                int j = 0;
#if __ARM_NEON
                for (; j + 3 < size; j += 4)
                {
                    float32x4_t _p = bfloat2float(vld1_u16(ptr));
                    _p = exp_ps(vsubq_f32(_p, _max));
                    vst1_u16(ptr, float2bfloat(_p));
                    _sum = vaddq_f32(_sum, _p);
                    ptr += 4;
                }
#endif // __ARM_NEON
                for (; j < size; j++)
                {
                    float v = (float)expf(bfloat16_to_float32(*ptr) - max);
                    *ptr = float32_to_bfloat16(v);
                    sum += v;
                    ptr++;
                }

#if __ARM_NEON
                if (elempack == 1)
                {
#if __aarch64__
                    sum += vaddvq_f32(_sum);
#else
                    float32x2_t _sum2 = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                    float32x2_t _ss2 = vpadd_f32(_sum2, _sum2);
                    sum += vget_lane_f32(_ss2, 0);
#endif
                    _sum = vdupq_n_f32(sum);
                }
#endif // __ARM_NEON
            }

            {
                unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);

                int j = 0;
#if __ARM_NEON
                float32x4_t _reciprocal_sum = vrecpeq_f32(_sum);
                _reciprocal_sum = vmulq_f32(vrecpsq_f32(_sum, _reciprocal_sum), _reciprocal_sum);
                for (; j + 3 < size; j += 4)
                {
                    float32x4_t _p = bfloat2float(vld1_u16(ptr));
                    _p = vmulq_f32(_p, _reciprocal_sum);
                    vst1_u16(ptr, float2bfloat(_p));
                    ptr += 4;
                }
#endif // __ARM_NEON
                for (; j < size; j++)
                {
                    *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) / sum);
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

        Mat maxsum(w, h, 2, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        Mat max = maxsum.channel(0);
        max.fill(-FLT_MAX);

        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = bottom_top_blob.channel(q);
            float* maxptr = max;

#if __ARM_NEON
            if (elempack == 4)
            {
                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p = bfloat2float(vld1_u16(ptr));
#if __aarch64__
                    float max0 = vmaxvq_f32(_p);
#else
                    float32x2_t _max2 = vmax_f32(vget_low_f32(_p), vget_high_f32(_p));
                    float32x2_t _mm2 = vpmax_f32(_max2, _max2);
                    float max0 = vget_lane_f32(_mm2, 0);
#endif
                    *maxptr = std::max(*maxptr, max0);
                    ptr += 4;
                    maxptr++;
                }
            }
#endif // __ARM_NEON
            if (elempack == 1)
            {
                int i = 0;
#if __ARM_NEON
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _p = bfloat2float(vld1_u16(ptr));
                    float32x4_t _max = vld1q_f32(maxptr);
                    _max = vmaxq_f32(_max, _p);
                    vst1q_f32(maxptr, _max);
                    ptr += 4;
                    maxptr += 4;
                }
#endif // __ARM_NEON
                for (; i < size; i++)
                {
                    *maxptr = std::max(*maxptr, bfloat16_to_float32(*ptr));
                    ptr++;
                    maxptr++;
                }
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);
            const float* maxptr = max;

#if __ARM_NEON
            if (elempack == 4)
            {
                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p = bfloat2float(vld1_u16(ptr));
                    float32x4_t _max = vdupq_n_f32(*maxptr);
                    _p = exp_ps(vsubq_f32(_p, _max));
                    vst1_u16(ptr, float2bfloat(_p));
                    ptr += 4;
                    maxptr++;
                }
            }
#endif // __ARM_NEON
            if (elempack == 1)
            {
                int i = 0;
#if __ARM_NEON
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _p = bfloat2float(vld1_u16(ptr));
                    float32x4_t _max = vld1q_f32(maxptr);
                    _p = exp_ps(vsubq_f32(_p, _max));
                    vst1_u16(ptr, float2bfloat(_p));
                    ptr += 4;
                    maxptr += 4;
                }
#endif // __ARM_NEON
                for (; i < size; i++)
                {
                    *ptr = float32_to_bfloat16(expf(bfloat16_to_float32(*ptr) - *maxptr));
                    ptr++;
                    maxptr++;
                }
            }
        }

        Mat sum = maxsum.channel(1);
        sum.fill(0.f);

        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);
            float* sumptr = sum;

#if __ARM_NEON
            if (elempack == 4)
            {
                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p = bfloat2float(vld1_u16(ptr));
#if __aarch64__
                    float sum0 = vaddvq_f32(_p);
#else
                    float32x2_t _sum2 = vadd_f32(vget_low_f32(_p), vget_high_f32(_p));
                    float32x2_t _ss2 = vpadd_f32(_sum2, _sum2);
                    float sum0 = vget_lane_f32(_ss2, 0);
#endif
                    *sumptr += sum0;
                    ptr += 4;
                    sumptr++;
                }
            }
#endif // __ARM_NEON
            if (elempack == 1)
            {
                int i = 0;
#if __ARM_NEON
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _p = bfloat2float(vld1_u16(ptr));
                    float32x4_t _sum = vld1q_f32(sumptr);
                    _sum = vaddq_f32(_sum, _p);
                    vst1q_f32(sumptr, _sum);
                    ptr += 4;
                    sumptr += 4;
                }
#endif // __ARM_NEON
                for (; i < size; i++)
                {
                    *sumptr += bfloat16_to_float32(*ptr);
                    ptr++;
                    sumptr++;
                }
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);
            const float* sumptr = sum;

#if __ARM_NEON
            if (elempack == 4)
            {
                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p = bfloat2float(vld1_u16(ptr));
                    float32x4_t _sum = vdupq_n_f32(*sumptr);
                    _p = div_ps(_p, _sum);
                    vst1_u16(ptr, float2bfloat(_p));
                    ptr += 4;
                    sumptr++;
                }
            }
#endif // __ARM_NEON
            if (elempack == 1)
            {
                int i = 0;
#if __ARM_NEON
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _p = bfloat2float(vld1_u16(ptr));
                    float32x4_t _sum = vld1q_f32(sumptr);
                    _p = div_ps(_p, _sum);
                    vst1_u16(ptr, float2bfloat(_p));
                    ptr += 4;
                    sumptr += 4;
                }
#endif // __ARM_NEON
                for (; i < size; i++)
                {
                    *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) / *sumptr);
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

        Mat maxsum(w * elempack, channels, 2, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        Mat max = maxsum.channel(0);
        max.fill(-FLT_MAX);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                float* maxptr = max.row(q);

                int j = 0;
#if __ARM_NEON
                for (; j + 3 < size; j += 4)
                {
                    float32x4_t _p = bfloat2float(vld1_u16(ptr));
                    float32x4_t _max = vld1q_f32(maxptr);
                    _max = vmaxq_f32(_max, _p);
                    vst1q_f32(maxptr, _max);
                    ptr += 4;
                    maxptr += 4;
                }
#endif // __ARM_NEON
                for (; j < size; j++)
                {
                    *maxptr = std::max(*maxptr, bfloat16_to_float32(*ptr));
                    ptr++;
                    maxptr++;
                }
            }
        }

        Mat sum = maxsum.channel(1);
        sum.fill(0.f);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                const float* maxptr = max.row(q);
                float* sumptr = sum.row(q);

                int j = 0;
#if __ARM_NEON
                for (; j + 3 < size; j += 4)
                {
                    float32x4_t _p = bfloat2float(vld1_u16(ptr));
                    float32x4_t _max = vld1q_f32(maxptr);
                    float32x4_t _sum = vld1q_f32(sumptr);
                    _p = exp_ps(vsubq_f32(_p, _max));
                    vst1_u16(ptr, float2bfloat(_p));
                    _sum = vaddq_f32(_sum, _p);
                    vst1q_f32(sumptr, _sum);
                    ptr += 4;
                    maxptr += 4;
                    sumptr += 4;
                }
#endif // __ARM_NEON
                for (; j < size; j++)
                {
                    float v = (float)expf(bfloat16_to_float32(*ptr) - *maxptr);
                    *ptr = float32_to_bfloat16(v);
                    *sumptr += v;
                    ptr++;
                    maxptr++;
                    sumptr++;
                }
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                const float* sumptr = sum.row(q);

                int j = 0;
#if __ARM_NEON
                for (; j + 3 < size; j += 4)
                {
                    float32x4_t _p = bfloat2float(vld1_u16(ptr));
                    float32x4_t _sum = vld1q_f32(sumptr);
                    _p = div_ps(_p, _sum);
                    vst1_u16(ptr, float2bfloat(_p));
                    ptr += 4;
                    sumptr += 4;
                }
#endif // __ARM_NEON
                for (; j < size; j++)
                {
                    *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) / *sumptr);
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
                float max = -FLT_MAX;
#if __ARM_NEON
                float32x4_t _max = vdupq_n_f32(-FLT_MAX);
#endif // __ARM_NEON
                {
                    const unsigned short* ptr = bottom_top_blob.channel(q).row<const unsigned short>(i);

                    int j = 0;
#if __ARM_NEON
                    for (; j + 3 < size; j += 4)
                    {
                        float32x4_t _p = bfloat2float(vld1_u16(ptr));
                        _max = vmaxq_f32(_max, _p);
                        ptr += 4;
                    }
#endif // __ARM_NEON
                    for (; j < size; j++)
                    {
                        max = std::max(max, bfloat16_to_float32(*ptr));
                        ptr++;
                    }

#if __ARM_NEON
                    if (elempack == 1)
                    {
#if __aarch64__
                        max = std::max(max, vmaxvq_f32(_max));
#else
                        float32x2_t _max2 = vmax_f32(vget_low_f32(_max), vget_high_f32(_max));
                        float32x2_t _mm2 = vpmax_f32(_max2, _max2);
                        max = std::max(max, vget_lane_f32(_mm2, 0));
#endif
                        _max = vdupq_n_f32(max);
                    }
#endif // __ARM_NEON
                }

                float sum = 0.f;
#if __ARM_NEON
                float32x4_t _sum = vdupq_n_f32(0.f);
#endif // __ARM_NEON
                {
                    unsigned short* ptr = bottom_top_blob.channel(q).row<unsigned short>(i);

                    int j = 0;
#if __ARM_NEON
                    for (; j + 3 < size; j += 4)
                    {
                        float32x4_t _p = bfloat2float(vld1_u16(ptr));
                        _p = exp_ps(vsubq_f32(_p, _max));
                        vst1_u16(ptr, float2bfloat(_p));
                        _sum = vaddq_f32(_sum, _p);
                        ptr += 4;
                    }
#endif // __ARM_NEON
                    for (; j < size; j++)
                    {
                        float v = (float)expf(bfloat16_to_float32(*ptr) - max);
                        *ptr = float32_to_bfloat16(v);
                        sum += v;
                        ptr++;
                    }

#if __ARM_NEON
                    if (elempack == 1)
                    {
#if __aarch64__
                        sum += vaddvq_f32(_sum);
#else
                        float32x2_t _sum2 = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                        float32x2_t _ss2 = vpadd_f32(_sum2, _sum2);
                        sum += vget_lane_f32(_ss2, 0);
#endif
                        _sum = vdupq_n_f32(sum);
                    }
#endif // __ARM_NEON
                }

                {
                    unsigned short* ptr = bottom_top_blob.channel(q).row<unsigned short>(i);

                    int j = 0;
#if __ARM_NEON
                    float32x4_t _reciprocal_sum = vrecpeq_f32(_sum);
                    _reciprocal_sum = vmulq_f32(vrecpsq_f32(_sum, _reciprocal_sum), _reciprocal_sum);
                    for (; j + 3 < size; j += 4)
                    {
                        float32x4_t _p = bfloat2float(vld1_u16(ptr));
                        _p = vmulq_f32(_p, _reciprocal_sum);
                        vst1_u16(ptr, float2bfloat(_p));
                        ptr += 4;
                    }
#endif // __ARM_NEON
                    for (; j < size; j++)
                    {
                        *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) / sum);
                        ptr++;
                    }
                }
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
