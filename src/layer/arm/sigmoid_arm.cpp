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

#include "sigmoid_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#endif // __ARM_NEON

#include <math.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(Sigmoid_arm)

int Sigmoid_arm::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;

    top_blob.create(w, h, channels);
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);

#if __ARM_NEON
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
        float32x4_t _one = vdupq_n_f32(1.f);
        for (; nn>0; nn--)
        {
            float32x4_t _p = vld1q_f32(ptr);
            _p = vnegq_f32(_p);
            _p = exp_ps(_p);
            _p = vaddq_f32(_p, _one);
            float32x4_t _outp = vrecpeq_f32(_p);
            _outp = vmulq_f32(vrecpsq_f32(_p, _outp), _outp);
//             _outp = vmulq_f32(vrecpsq_f32(_p, _outp), _outp);
            vst1q_f32(outptr, _outp);

            ptr += 4;
            outptr += 4;
        }
#endif // __ARM_NEON
        for (; remain>0; remain--)
        {
            *outptr = 1.f / (1.f + exp(-*ptr));

            ptr++;
            outptr++;
        }
    }

    return 0;
}

int Sigmoid_arm::forward_inplace(Mat& bottom_top_blob) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    #pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

#if __ARM_NEON
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
        float32x4_t _one = vdupq_n_f32(1.f);
        for (; nn>0; nn--)
        {
            float32x4_t _p = vld1q_f32(ptr);
            _p = vnegq_f32(_p);
            _p = exp_ps(_p);
            _p = vaddq_f32(_p, _one);
            _p = vrecpeq_f32(_p);
            _p = vmulq_f32(vrecpsq_f32(_p, _p), _p);
//             _p = vmulq_f32(vrecpsq_f32(_p, _p), _p);
            vst1q_f32(ptr, _p);

            ptr += 4;
        }
#endif // __ARM_NEON
        for (; remain>0; remain--)
        {
            *ptr = 1.f / (1.f + exp(-*ptr));

            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
