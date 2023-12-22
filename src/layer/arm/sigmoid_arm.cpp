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

#include "arm_usability.h"

#include "cpu.h"

namespace ncnn {

Sigmoid_arm::Sigmoid_arm()
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

int Sigmoid_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
#if __aarch64__
        for (; i + 15 < size; i += 16)
        {
            float32x4_t _p0 = vld1q_f32(ptr);
            float32x4_t _p1 = vld1q_f32(ptr + 4);
            float32x4_t _p2 = vld1q_f32(ptr + 8);
            float32x4_t _p3 = vld1q_f32(ptr + 12);
            _p0 = sigmoid_ps(_p0);
            _p1 = sigmoid_ps(_p1);
            _p2 = sigmoid_ps(_p2);
            _p3 = sigmoid_ps(_p3);
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
            _p0 = sigmoid_ps(_p0);
            _p1 = sigmoid_ps(_p1);
            vst1q_f32(ptr, _p0);
            vst1q_f32(ptr + 4, _p1);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vld1q_f32(ptr);
            _p = sigmoid_ps(_p);
            vst1q_f32(ptr, _p);
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            *ptr = 1.f / (1.f + expf(-*ptr));

            ptr++;
        }
    }

    return 0;
}

#if NCNN_BF16
int Sigmoid_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
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
#if __aarch64__
        for (; i + 15 < size; i += 16)
        {
            uint16x8_t _p01 = vld1q_u16(ptr);
            uint16x8_t _p23 = vld1q_u16(ptr + 8);
            float32x4_t _p0 = bfloat2float(vget_low_u16(_p01));
            float32x4_t _p1 = bfloat2float(vget_high_u16(_p01));
            float32x4_t _p2 = bfloat2float(vget_low_u16(_p23));
            float32x4_t _p3 = bfloat2float(vget_high_u16(_p23));
            _p0 = sigmoid_ps(_p0);
            _p1 = sigmoid_ps(_p1);
            _p2 = sigmoid_ps(_p2);
            _p3 = sigmoid_ps(_p3);
            _p01 = vcombine_u16(float2bfloat(_p0), float2bfloat(_p1));
            _p23 = vcombine_u16(float2bfloat(_p2), float2bfloat(_p3));
            vst1q_u16(ptr, _p01);
            vst1q_u16(ptr + 8, _p23);
            ptr += 16;
        }
#endif // __aarch64__
        for (; i + 7 < size; i += 8)
        {
            uint16x8_t _p = vld1q_u16(ptr);
            float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
            float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
            _p0 = sigmoid_ps(_p0);
            _p1 = sigmoid_ps(_p1);
            _p = vcombine_u16(float2bfloat(_p0), float2bfloat(_p1));
            vst1q_u16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr));
            _p = sigmoid_ps(_p);
            vst1_u16(ptr, float2bfloat(_p));
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            v = 1.f / (1.f + expf(-v));
            *ptr = float32_to_bfloat16(v);

            ptr++;
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
