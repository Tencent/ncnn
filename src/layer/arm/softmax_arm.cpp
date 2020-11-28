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
#include "neon_mathfun.h"

#include <arm_neon.h>
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
    int dims = bottom_top_blob.dims;
    size_t elemsize = bottom_top_blob.elemsize;
    int elempack = bottom_top_blob.elempack;
    int positive_axis = axis < 0 ? dims + axis : axis;

#if __ARM_NEON
    if (elempack == 4)
    {
        if (dims == 1) // positive_axis == 0
        {
            int w = bottom_top_blob.w;

            float* ptr = bottom_top_blob;

            float32x4_t _max = vdupq_n_f32(-FLT_MAX);
            for (int i = 0; i < w; i++)
            {
                float32x4_t _p = vld1q_f32(ptr + i * 4);
                _max = vmaxq_f32(_max, _p);
            }
#if __aarch64__
            _max = vpmaxq_f32(_max, _max);
            _max = vpmaxq_f32(_max, _max);
#else
            _max = vmaxq_f32(_max, vrev64q_f32(_max));
            _max = vmaxq_f32(_max, vextq_f32(_max, _max, 2));
#endif

            float32x4_t _sum = vdupq_n_f32(0.f);
            for (int i = 0; i < w; i++)
            {
                float32x4_t _p = vld1q_f32(ptr + i * 4);
                _p = exp_ps(vsubq_f32(_p, _max));
                vst1q_f32(ptr + i * 4, _p);
                _sum = vaddq_f32(_sum, _p);
            }
#if __aarch64__
            _sum = vpaddq_f32(_sum, _sum);
            _sum = vpaddq_f32(_sum, _sum);
#else
            _sum = vaddq_f32(_sum, vrev64q_f32(_sum));
            _sum = vaddq_f32(_sum, vextq_f32(_sum, _sum, 2));
#endif

            for (int i = 0; i < w; i++)
            {
                float32x4_t _p = vld1q_f32(ptr + i * 4);
#if __aarch64__
                _p = vdivq_f32(_p, _sum);
#else
                _p = div_ps(_p, _sum);
#endif
                vst1q_f32(ptr + i * 4, _p);
            }

            return 0;
        }

        if (dims == 2 && positive_axis == 0)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;

            Mat max;
            max.create(w, 4u, 1, opt.workspace_allocator);
            if (max.empty())
                return -100;
            max.fill(-FLT_MAX);

            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_top_blob.row(i);
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
                    max[j] = std::max(max[j], max0);
                    ptr += 4;
                }
            }

            Mat sum;
            sum.create(w, 4u, 1, opt.workspace_allocator);
            if (sum.empty())
                return -100;
            sum.fill(0.f);

            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);
                for (int j = 0; j < w; j++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _max = vdupq_n_f32(max[j]);
                    _p = exp_ps(vsubq_f32(_p, _max));
                    vst1q_f32(ptr, _p);
#if __aarch64__
                    float sum0 = vaddvq_f32(_p);
#else
                    float32x2_t _sum2 = vadd_f32(vget_low_f32(_p), vget_high_f32(_p));
                    float32x2_t _ss2 = vpadd_f32(_sum2, _sum2);
                    float sum0 = vget_lane_f32(_ss2, 0);
#endif
                    sum[j] += sum0;
                    ptr += 4;
                }
            }

            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);
                for (int j = 0; j < w; j++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _sum = vdupq_n_f32(sum[j]);
#if __aarch64__
                    _p = vdivq_f32(_p, _sum);
#else
                    _p = div_ps(_p, _sum);
#endif
                    vst1q_f32(ptr, _p);
                    ptr += 4;
                }
            }

            return 0;
        }

        if (dims == 2 && positive_axis == 1)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);

                float32x4_t _max = vdupq_n_f32(-FLT_MAX);
                for (int j = 0; j < w; j++)
                {
                    float32x4_t _p = vld1q_f32(ptr + j * 4);
                    _max = vmaxq_f32(_max, _p);
                }

                float32x4_t _sum = vdupq_n_f32(0.f);
                for (int j = 0; j < w; j++)
                {
                    float32x4_t _p = vld1q_f32(ptr + j * 4);
                    _p = exp_ps(vsubq_f32(_p, _max));
                    vst1q_f32(ptr + j * 4, _p);
                    _sum = vaddq_f32(_sum, _p);
                }

                for (int j = 0; j < w; j++)
                {
                    float32x4_t _p = vld1q_f32(ptr + j * 4);
#if __aarch64__
                    _p = vdivq_f32(_p, _sum);
#else
                    _p = div_ps(_p, _sum);
#endif
                    vst1q_f32(ptr + j * 4, _p);
                }
            }

            return 0;
        }

        if (dims == 3 && positive_axis == 0)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;
            int size = w * h;

            Mat max;
            max.create(w, h, 4u, 1, opt.workspace_allocator);
            if (max.empty())
                return -100;
            max.fill(-FLT_MAX);
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_top_blob.channel(q);

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
                    max[i] = std::max(max[i], max0);
                    ptr += 4;
                }
            }

            Mat sum;
            sum.create(w, h, 4u, 1, opt.workspace_allocator);
            if (sum.empty())
                return -100;
            sum.fill(0.f);
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _max = vdupq_n_f32(max[i]);
                    _p = exp_ps(vsubq_f32(_p, _max));
                    vst1q_f32(ptr, _p);
#if __aarch64__
                    float sum0 = vaddvq_f32(_p);
#else
                    float32x2_t _sum2 = vadd_f32(vget_low_f32(_p), vget_high_f32(_p));
                    float32x2_t _ss2 = vpadd_f32(_sum2, _sum2);
                    float sum0 = vget_lane_f32(_ss2, 0);
#endif
                    sum[i] += sum0;
                    ptr += 4;
                }
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _sum = vdupq_n_f32(sum[i]);
#if __aarch64__
                    _p = vdivq_f32(_p, _sum);
#else
                    _p = div_ps(_p, _sum);
#endif
                    vst1q_f32(ptr, _p);
                    ptr += 4;
                }
            }

            return 0;
        }

        if (dims == 3 && positive_axis == 1)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;

            Mat max;
            max.create(w, channels, elemsize, elempack, opt.workspace_allocator);
            if (max.empty())
                return -100;
            max.fill(vdupq_n_f32(-FLT_MAX));
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < h; i++)
                {
                    float* maxptr = max.row(q);

                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _p = vld1q_f32(ptr);
                        float32x4_t _max = vld1q_f32(maxptr);
                        _max = vmaxq_f32(_max, _p);
                        vst1q_f32(maxptr, _max);
                        ptr += 4;
                        maxptr += 4;
                    }
                }
            }

            Mat sum;
            sum.create(w, channels, elemsize, elempack, opt.workspace_allocator);
            if (sum.empty())
                return -100;
            sum.fill(vdupq_n_f32(0.f));
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < h; i++)
                {
                    float* maxptr = max.row(q);
                    float* sumptr = sum.row(q);

                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _p = vld1q_f32(ptr);
                        float32x4_t _max = vld1q_f32(maxptr);
                        _p = exp_ps(vsubq_f32(_p, _max));
                        vst1q_f32(ptr, _p);
                        float32x4_t _sum = vld1q_f32(sumptr);
                        _sum = vaddq_f32(_sum, _p);
                        vst1q_f32(sumptr, _sum);
                        ptr += 4;
                        maxptr += 4;
                        sumptr += 4;
                    }
                }
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < h; i++)
                {
                    float* sumptr = sum.row(q);

                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _p = vld1q_f32(ptr);
                        float32x4_t _sum = vld1q_f32(sumptr);
#if __aarch64__
                        _p = vdivq_f32(_p, _sum);
#else
                        _p = div_ps(_p, _sum);
#endif
                        vst1q_f32(ptr, _p);
                        ptr += 4;
                        sumptr += 4;
                    }
                }
            }

            return 0;
        }

        if (dims == 3 && positive_axis == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < h; i++)
                {
                    float32x4_t _max = vdupq_n_f32(-FLT_MAX);
                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _p = vld1q_f32(ptr + j * 4);
                        _max = vmaxq_f32(_max, _p);
                    }

                    float32x4_t _sum = vdupq_n_f32(0.f);
                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _p = vld1q_f32(ptr + j * 4);
                        _p = exp_ps(vsubq_f32(_p, _max));
                        vst1q_f32(ptr + j * 4, _p);
                        _sum = vaddq_f32(_sum, _p);
                    }

                    for (int j = 0; j < w; j++)
                    {
                        float32x4_t _p = vld1q_f32(ptr + j * 4);
#if __aarch64__
                        _p = vdivq_f32(_p, _sum);
#else
                        _p = div_ps(_p, _sum);
#endif
                        vst1q_f32(ptr + j * 4, _p);
                    }

                    ptr += w * 4;
                }
            }

            return 0;
        }

        return 0;
    }
#endif // __ARM_NEON

    if (dims != 3 || positive_axis != 0)
        return Softmax::forward_inplace(bottom_top_blob, opt);

    // value = exp( value - global max value )
    // sum all value
    // value = value / sum

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    Mat max;
    max.create(w, h, elemsize, opt.workspace_allocator);
    if (max.empty())
        return -100;
    max.fill(-FLT_MAX);
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        float* maxptr = max;

        for (int i = 0; i < size; i++)
        {
            maxptr[i] = std::max(maxptr[i], ptr[i]);
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        float* maxptr = max;

#if __ARM_NEON
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
        for (; nn > 0; nn--)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _max = vld1q_f32(maxptr);

            _p = exp_ps(vsubq_f32(_p, _max));

            vst1q_f32(ptr, _p);

            ptr += 4;
            maxptr += 4;
        }
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *ptr = exp(*ptr - *maxptr);

            ptr++;
            maxptr++;
        }
    }

    Mat sum;
    sum.create(w, h, elemsize, opt.workspace_allocator);
    if (sum.empty())
        return -100;
    sum.fill(0.f);
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        float* sumptr = sum;

#if __ARM_NEON
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
        for (; nn > 0; nn--)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _sum = vld1q_f32(sumptr);
            _sum = vaddq_f32(_sum, _p);
            vst1q_f32(sumptr, _sum);

            ptr += 4;
            sumptr += 4;
        }
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *sumptr += *ptr;

            ptr++;
            sumptr++;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        float* sumptr = sum;

#if __ARM_NEON
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
        for (; nn > 0; nn--)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _sum = vld1q_f32(sumptr);
#if __aarch64__
            _p = vdivq_f32(_p, _sum);
#else
            _p = div_ps(_p, _sum);
#endif // __aarch64__
            vst1q_f32(ptr, _p);

            ptr += 4;
            sumptr += 4;
        }
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *ptr /= *sumptr;

            ptr++;
            sumptr++;
        }
    }

    return 0;
}

} // namespace ncnn
