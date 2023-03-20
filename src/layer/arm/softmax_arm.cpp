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
#include <math.h>

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#endif // __ARM_NEON

namespace ncnn {

Softmax_arm::Softmax_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON
}

int Softmax_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
                *ptr = (float)(exp(*ptr - max));
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

        Mat max;
        max.create(w, 4u, opt.workspace_allocator);
        if (max.empty())
            return -100;
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

        Mat sum;
        sum.create(w, 4u, opt.workspace_allocator);
        if (sum.empty())
            return -100;
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
                    *ptr = (float)(exp(*ptr - *maxptr));
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
                    float32x4_t _sum = vdupq_n_f32(sum[j]);
                    _p = div_ps(_p, _sum);
                    vst1q_f32(ptr, _p);
                    ptr += 4;
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
                    *ptr = (float)(exp(*ptr - max));
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

        Mat max;
        max.create(w, h, 4u, opt.workspace_allocator);
        if (max.empty())
            return -100;
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
                    *ptr = exp(*ptr - *maxptr);
                    ptr++;
                    maxptr++;
                }
            }
        }

        Mat sum;
        sum.create(w, h, 4u, opt.workspace_allocator);
        if (sum.empty())
            return -100;
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

        Mat max;
        max.create(w * elempack, channels, 4u, opt.workspace_allocator);
        if (max.empty())
            return -100;
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

        Mat sum;
        sum.create(w * elempack, channels, 4u, opt.workspace_allocator);
        if (sum.empty())
            return -100;
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
                    *ptr = (float)(exp(*ptr - *maxptr));
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
                        *ptr = (float)(exp(*ptr - max));
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

} // namespace ncnn
