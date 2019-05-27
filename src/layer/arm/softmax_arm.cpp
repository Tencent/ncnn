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

DEFINE_LAYER_CREATOR(Softmax_arm)

int Softmax_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;

    if (dims != 3 || axis != 0)
        return Softmax::forward_inplace(bottom_top_blob, opt);

    // value = exp( value - global max value )
    // sum all value
    // value = value / sum

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    size_t elemsize = bottom_top_blob.elemsize;
    int size = w * h;

    Mat max;
    max.create(w, h, elemsize, opt.workspace_allocator);
    if (max.empty())
        return -100;
    max.fill(-FLT_MAX);
    for (int q=0; q<channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        float* maxptr = max;

        for (int i=0; i<size; i++)
        {
            maxptr[i] = std::max(maxptr[i], ptr[i]);
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q=0; q<channels; q++)
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
        for (; nn>0; nn--)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _max = vld1q_f32(maxptr);

            _p = exp_ps(vsubq_f32(_p, _max));

            vst1q_f32(ptr, _p);

            ptr += 4;
            maxptr += 4;
        }
#endif // __ARM_NEON

        for (; remain>0; remain--)
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
    for (int q=0; q<channels; q++)
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
        for (; nn>0; nn--)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _sum = vld1q_f32(sumptr);
            _sum = vaddq_f32(_sum, _p);
            vst1q_f32(sumptr, _sum);

            ptr += 4;
            sumptr += 4;
        }
#endif // __ARM_NEON

        for (; remain>0; remain--)
        {
            *sumptr += *ptr;

            ptr++;
            sumptr++;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q=0; q<channels; q++)
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
        for (; nn>0; nn--)
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

        for (; remain>0; remain--)
        {
            *ptr /= *sumptr;

            ptr++;
            sumptr++;
        }
    }

    return 0;
}

} // namespace ncnn
