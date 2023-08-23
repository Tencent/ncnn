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

#include "gelu_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#endif // __ARM_NEON

#include "arm_usability.h"
#include "cpu.h"

namespace ncnn {

GELU_arm::GELU_arm()
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

int GELU_arm::create_pipeline(const Option& /*opt*/)
{
    if (!fast_gelu)
    {
        support_packing = false;
        support_fp16_storage = false;
        support_bf16_storage = false;
    }
    return 0;
}

int GELU_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    if (!fast_gelu)
    {
        return GELU::forward_inplace(bottom_top_blob, opt);
    }

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
    int elempack = bottom_top_blob.elempack;
    int channels = bottom_top_blob.c;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;

#if __ARM_NEON
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _pLoad = vld1q_f32(ptr);

            float32x4_t _blob = vmulq_f32(_pLoad, _pLoad);
            _blob = vmulq_f32(_pLoad, _blob);
            _blob = vmulq_f32(vdupq_n_f32(0.044715f * 0.79788452f), _blob);
            _blob = vmlaq_f32(_blob, vdupq_n_f32(0.79788452f), _pLoad);
            _blob = tanh_ps(_blob);
            _blob = vaddq_f32(vdupq_n_f32(1.f), _blob);
            _blob = vmulq_f32(vdupq_n_f32(0.5f), vmulq_f32(_blob, _pLoad));
            vst1q_f32(ptr, _blob);
            ptr += 4;
        }
#endif
        for (; i < size; i++)
        {
            // y = 0.5x * (1 + tanh(sqrt(2/Pi) * (x + 0.044715x^3)))
            *ptr = 0.5f * *ptr * (1.0f + tanhf(0.79788452f * (*ptr + 0.044715f * *ptr * *ptr * *ptr)));

            ptr++;
        }
    }

    return 0;
}

#if NCNN_BF16
int GELU_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int elempack = bottom_top_blob.elempack;
    int channels = bottom_top_blob.c;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = bottom_top_blob.channel(q);

        int i = 0;

#if __ARM_NEON
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _pLoad = bfloat2float(vld1_u16(ptr));

            float32x4_t _blob = vmulq_f32(_pLoad, _pLoad);
            _blob = vmulq_f32(_pLoad, _blob);
            _blob = vmulq_f32(vdupq_n_f32(0.044715f * 0.79788452f), _blob);
            _blob = vmlaq_f32(_blob, vdupq_n_f32(0.79788452f), _pLoad);
            _blob = tanh_ps(_blob);
            _blob = vaddq_f32(vdupq_n_f32(1.f), _blob);
            _blob = vmulq_f32(vdupq_n_f32(0.5f), vmulq_f32(_blob, _pLoad));
            vst1_u16(ptr, float2bfloat(_blob));
            ptr += 4;
        }
#endif // __ARM_NEON

        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            v = 0.5f * v * (1.0f + tanhf(0.79788452f * (v + 0.044715f * v * v * v)));
            *ptr = float32_to_bfloat16(v);
            ptr++;
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
