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
#include <arm_neon.h>
#include "arm_usability.h"
#include "neon_mathfun.h"
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "neon_mathfun_fp16s.h"
#endif
#endif // __ARM_NEON

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int Mish_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int size = w * h * d;
    int elempack = bottom_top_blob.elempack;

    if (elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                _p = vmulq_f32(_p, tanh_ps(log_ps(vaddq_f32(exp_ps(_p), vdupq_n_f32(1.f)))));
                vst1_f16(ptr, vcvt_f16_f32(_p));

                ptr += 4;
            }
        }

        return 0;
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = bottom_top_blob.channel(q);

        int i = 0;
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
            _p = vmulq_f32(_p, tanh_ps(log_ps(vaddq_f32(exp_ps(_p), vdupq_n_f32(1.f)))));
            vst1_f16(ptr, vcvt_f16_f32(_p));

            ptr += 4;
        }
        for (; i < size; i++)
        {
            float v = (float)*ptr;
            v = v * tanhf(logf(expf(v) + 1.f));
            *ptr = (__fp16)v;
            ptr++;
        }
    }

    return 0;
}

int Mish_arm::forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int size = w * h * d;
    int elempack = bottom_top_blob.elempack;

    if (elempack == 8)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                float16x8_t _p = vld1q_f16(ptr);
                _p = vmulq_f16(_p, tanh_ps_f16(log_ps_f16(vaddq_f16(exp_ps_f16(_p), vdupq_n_f16(1.f)))));
                vst1q_f16(ptr, _p);

                ptr += 8;
            }
        }

        return 0;
    }

    if (elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            __fp16* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                float16x4_t _p = vld1_f16(ptr);
                _p = vmul_f16(_p, tanh_ps_f16(log_ps_f16(vadd_f16(exp_ps_f16(_p), vdup_n_f16(1.f)))));
                vst1_f16(ptr, _p);

                ptr += 4;
            }
        }

        return 0;
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = bottom_top_blob.channel(q);

        int i = 0;
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            _p = vmul_f16(_p, tanh_ps_f16(log_ps_f16(vadd_f16(exp_ps_f16(_p), vdup_n_f16(1.f)))));
            vst1_f16(ptr, _p);

            ptr += 4;
        }
        for (; i < size; i++)
        {
            __fp16 v = *ptr;
            v = v * (__fp16)tanhf(logf(expf(v) + 1.f));
            *ptr = v;
            ptr++;
        }
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn
