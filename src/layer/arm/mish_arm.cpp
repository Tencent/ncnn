// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "mish_arm.h"

#if __ARM_NEON
#include "neon_mathfun.h"

#include <arm_neon.h>
#endif // __ARM_NEON

#include <math.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(Mish_arm)

Mish_arm::Mish_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON

    support_bf16_storage = true;
}

int Mish_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    if (opt.use_bf16_storage)
        return forward_inplace_bf16s(bottom_top_blob, opt);

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;
    int elempack = bottom_top_blob.elempack;

#if __ARM_NEON
    if (elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                float32x4_t _p = vld1q_f32(ptr);
                _p = vmulq_f32(_p, tanh_ps(log_ps(vaddq_f32(exp_ps(_p), vdupq_n_f32(1.f)))));
                vst1q_f32(ptr, _p);
                ptr += 4;
            }
        }

        return 0;
    }
#endif // __ARM_NEON

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

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
            _p = vmulq_f32(_p, tanh_ps(log_ps(vaddq_f32(exp_ps(_p), vdupq_n_f32(1.f)))));
            vst1q_f32(ptr, _p);
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; remain > 0; remain--)
        {
            *ptr = *ptr * tanh(log(exp(*ptr) + 1.f));
            ptr++;
        }
    }

    return 0;
}

int Mish_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;
    int elempack = bottom_top_blob.elempack;

#if __ARM_NEON
    if (elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                float32x4_t _p = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr), 16));
                _p = vmulq_f32(_p, tanh_ps(log_ps(vaddq_f32(exp_ps(_p), vdupq_n_f32(1.f)))));
                vst1_u16(ptr, vshrn_n_u32(vreinterpretq_u32_f32(_p), 16));
                ptr += 4;
            }
        }

        return 0;
    }
#endif // __ARM_NEON

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = bottom_top_blob.channel(q);

#if __ARM_NEON
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
        for (; nn > 0; nn--)
        {
            float32x4_t _p = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr), 16));
            _p = vmulq_f32(_p, tanh_ps(log_ps(vaddq_f32(exp_ps(_p), vdupq_n_f32(1.f)))));
            vst1_u16(ptr, vshrn_n_u32(vreinterpretq_u32_f32(_p), 16));
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; remain > 0; remain--)
        {
            float v = bfloat16_to_float32(*ptr);
            v = v * tanh(log(exp(v) + 1.f));
            *ptr = float32_to_bfloat16(v);
            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
