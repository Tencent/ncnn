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

#include "swish_arm.h"

#include <math.h>

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#endif // __ARM_NEON

#include "arm_usability.h"
#include "cpu.h"

namespace ncnn {

Swish_arm::Swish_arm()
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

int Swish_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_inplace_fp16sa(bottom_top_blob, opt);
        else
            return forward_inplace_fp16s(bottom_top_blob, opt);
    }
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __ARM_NEON
        float32x4_t _one = vdupq_n_f32(1.f);
#if __aarch64__
        for (; i + 15 < size; i += 16)
        {
            float32x4_t _p0 = vld1q_f32(ptr);
            float32x4_t _p1 = vld1q_f32(ptr + 4);
            float32x4_t _p2 = vld1q_f32(ptr + 8);
            float32x4_t _p3 = vld1q_f32(ptr + 12);
            _p0 = div_ps(_p0, vaddq_f32(_one, exp_ps(vnegq_f32(_p0))));
            _p1 = div_ps(_p1, vaddq_f32(_one, exp_ps(vnegq_f32(_p1))));
            _p2 = div_ps(_p2, vaddq_f32(_one, exp_ps(vnegq_f32(_p2))));
            _p3 = div_ps(_p3, vaddq_f32(_one, exp_ps(vnegq_f32(_p3))));
            vst1q_f32(ptr, _p0);
            vst1q_f32(ptr + 4, _p1);
            vst1q_f32(ptr + 8, _p2);
            vst1q_f32(ptr + 12, _p3);
            ptr += 16;
        }
#endif // __aarch64__
        for (; i + 7 < size; i += 8)
        {
            float32x4_t _p0 = vld1q_f32(ptr);
            float32x4_t _p1 = vld1q_f32(ptr + 4);
            _p0 = div_ps(_p0, vaddq_f32(_one, exp_ps(vnegq_f32(_p0))));
            _p1 = div_ps(_p1, vaddq_f32(_one, exp_ps(vnegq_f32(_p1))));
            vst1q_f32(ptr, _p0);
            vst1q_f32(ptr + 4, _p1);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vld1q_f32(ptr);
            _p = div_ps(_p, vaddq_f32(_one, exp_ps(vnegq_f32(_p))));
            vst1q_f32(ptr, _p);
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            *ptr = *ptr / (1.f + exp(-*ptr));

            ptr++;
        }
    }

    return 0;
}

#if NCNN_BF16
int Swish_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __ARM_NEON
        float32x4_t _one = vdupq_n_f32(1.f);
#if __aarch64__
        for (; i + 15 < size; i += 16)
        {
            uint16x8_t _p01 = vld1q_u16(ptr);
            uint16x8_t _p23 = vld1q_u16(ptr + 8);
            float32x4_t _p0 = float2bfloat(vget_low_u16(_p01));
            float32x4_t _p1 = float2bfloat(vget_high_u16(_p01));
            float32x4_t _p2 = float2bfloat(vget_low_u16(_p23));
            float32x4_t _p3 = float2bfloat(vget_high_u16(_p23));
            _p0 = div_ps(_p0, vaddq_f32(_one, exp_ps(vnegq_f32(_p0))));
            _p1 = div_ps(_p1, vaddq_f32(_one, exp_ps(vnegq_f32(_p1))));
            _p2 = div_ps(_p2, vaddq_f32(_one, exp_ps(vnegq_f32(_p2))));
            _p3 = div_ps(_p3, vaddq_f32(_one, exp_ps(vnegq_f32(_p3))));
            _p01 = vcombine_u16(bfloat2float(_p0), bfloat2float(_p1));
            _p23 = vcombine_u16(bfloat2float(_p2), bfloat2float(_p3));
            vst1q_u16(ptr, _p01);
            vst1q_u16(ptr + 8, _p23);
            ptr += 16;
        }
#endif // __aarch64__
        for (; i + 7 < size; i += 8)
        {
            uint16x8_t _p = vld1q_u16(ptr);
            float32x4_t _p0 = float2bfloat(vget_low_u16(_p));
            float32x4_t _p1 = float2bfloat(vget_high_u16(_p));
            _p0 = div_ps(_p0, vaddq_f32(_one, exp_ps(vnegq_f32(_p0))));
            _p1 = div_ps(_p1, vaddq_f32(_one, exp_ps(vnegq_f32(_p1))));
            _p = vcombine_u16(bfloat2float(_p0), bfloat2float(_p1));
            vst1q_u16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = float2bfloat(vld1_u16(ptr));
            _p = div_ps(_p, vaddq_f32(_one, exp_ps(vnegq_f32(_p))));
            vst1_u16(ptr, bfloat2float(_p));
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            v = v / (1.f + exp(-v));
            *ptr = float32_to_bfloat16(v);

            ptr++;
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
